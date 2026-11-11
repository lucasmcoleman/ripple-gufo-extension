#include "engine.hpp"

#include <algorithm>
#include <cstring>
#include <gufo/ext/sha256.hpp>
#include <stdexcept>
#include <utility>

#include "text/chat_template.hpp"

#if defined(RICURSIVE_HAVE_GPU_EXECUTOR)
#include "hip/gpu_executor_factory.hpp"
#endif

namespace gufo::models::ricursive {
namespace {

namespace rt = ::ricursive::text;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
}

std::vector<std::int32_t> TokenTypes(const core::GgufReader& reader) {
  std::vector<std::int32_t> types;
  const auto* entry = reader.FindMetadata("tokenizer.ggml.token_type");
  if (entry == nullptr) return types;
  if (const auto* v = std::get_if<std::vector<std::int64_t>>(&entry->value)) {
    types.assign(v->begin(), v->end());
  } else if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&entry->value)) {
    types.assign(u->begin(), u->end());
  }
  return types;
}

std::unique_ptr<rt::Tokenizer> MakeTokenizer(const core::GgufReader& reader,
                                             std::uint32_t vocab,
                                             std::vector<bool>* stop,
                                             std::string* error) {
  const auto token_views = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
  const auto merge_views = reader.GetMetadataStringArray("tokenizer.ggml.merges");
  if (token_views.empty()) {
    SetError(error, "GGUF has no tokenizer.ggml.tokens");
    return nullptr;
  }
  if (token_views.size() != vocab) {
    SetError(error, "tokenizer.ggml.tokens has " + std::to_string(token_views.size()) +
                        " entries, ricursive.vocab_size is " + std::to_string(vocab));
    return nullptr;
  }
  const std::vector<std::string> tokens(token_views.begin(), token_views.end());
  const std::vector<std::string> merges(merge_views.begin(), merge_views.end());
  const auto types = TokenTypes(reader);
  if (!types.empty() && types.size() != tokens.size()) {
    SetError(error, "tokenizer.ggml.token_type length differs from the vocabulary");
    return nullptr;
  }
  rt::SpecialTokenIds ids;
  const auto id = [&](const char* key) {
    const auto v = reader.GetMetadataUint32(key);
    return v ? static_cast<rt::TokenId>(*v) : rt::kInvalidTokenId;
  };
  ids.bos = id("tokenizer.ggml.bos_token_id");
  ids.eos = id("tokenizer.ggml.eos_token_id");
  ids.eot = id("tokenizer.ggml.eot_token_id");
  ids.pad = id("tokenizer.ggml.padding_token_id");
  ids.unk = id("tokenizer.ggml.unknown_token_id");
  // The model card's EOS set, plus whatever the GGUF names (the same ids in
  // the real file); only ids inside the vocabulary count.
  stop->assign(vocab, false);
  const auto add_stop = [&](std::int64_t token) {
    if (token >= 0 && token < static_cast<std::int64_t>(vocab) && !(*stop)[static_cast<std::size_t>(token)]) {
      (*stop)[static_cast<std::size_t>(token)] = true;
      ids.stop.push_back(static_cast<rt::TokenId>(token));
    }
  };
  for (const std::int32_t t : kStopTokens) add_stop(t);
  if (ids.eos != rt::kInvalidTokenId) add_stop(ids.eos);
  if (ids.eot != rt::kInvalidTokenId) add_stop(ids.eot);
  std::string tokenizer_error;
  auto tokenizer = rt::Tokenizer::Create(tokens, merges, types, ids, {}, &tokenizer_error);
  if (!tokenizer) SetError(error, "tokenizer: " + tokenizer_error);
  return tokenizer;
}

std::uint32_t RoundUp(std::uint32_t value, std::uint32_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

}  // namespace

std::shared_ptr<Model> Model::Load(const std::string& path, const ModelOptions& options,
                                   std::string* error) {
  try {
    std::shared_ptr<const core::GgufReader> reader = options.reader;
    if (!reader) {
      std::string open_error;
      std::shared_ptr<const core::GgufReader> opened =
          core::GgufReader::OpenFile(path, &open_error);
      if (!opened) {
        SetError(error, "cannot open " + path + ": " + open_error);
        return nullptr;
      }
      reader = std::move(opened);
    }
    std::string detail;
    const auto config = ParseConfig(*reader, &detail);
    if (!config) {
      SetError(error, detail);
      return nullptr;
    }
    auto weights = ResolveWeights(*reader, *config, &detail);
    if (!weights) {
      SetError(error, detail);
      return nullptr;
    }
    auto model = std::shared_ptr<Model>(new Model(*weights, reader));
#if defined(RICURSIVE_HAVE_GPU_EXECUTOR)
    // CreateGpuExecutor receives only the weights table; the device upload needs
    // the reader it was resolved from (docs/INTEGRATION.md, mismatch 1).
    RegisterGpuWeightsReader(model->weights_, reader);
#endif
    const Config& c = model->config();

    // The renderer implements one template; another one would silently give
    // wrong prompts.
    const auto tmpl = reader->GetMetadataString("tokenizer.chat_template");
    model->template_sha256_ =
        tmpl ? gufo::crypto::Sha256Hex(std::span(reinterpret_cast<const std::uint8_t*>(tmpl->data()), tmpl->size()))
             : std::string("none");
    if (model->template_sha256_ != rt::kRippleChatTemplateSha256 &&
        !options.allow_template_mismatch) {
      SetError(error, "chat template hash " + model->template_sha256_ +
                          " differs from the pinned " + std::string(rt::kRippleChatTemplateSha256) +
                          "; the renderer implements only the pinned template (override: "
                          "--allow-chat-template-mismatch)");
      return nullptr;
    }
    model->tokenizer_ = MakeTokenizer(*reader, c.vocab_size, &model->stop_, error);
    if (!model->tokenizer_) return nullptr;
    model->name_ = std::string(reader->GetMetadataString("general.name").value_or("ricursive"));

    if (options.backend) {
      model->backend_ = *options.backend;
    } else {
      const auto env = BackendFromEnvironment(error);
      if (!env) return nullptr;
      model->backend_ = *env;
    }
    const std::uint32_t cap = options.max_context != 0
                                  ? options.max_context
                                  : std::min(kDefaultContextCap, c.context_length);
    if (cap > c.context_length) {
      SetError(error, "context cap " + std::to_string(cap) + " exceeds the model's " +
                          std::to_string(c.context_length) + " positions");
      return nullptr;
    }
    if (options.prefill_rows == 0 || options.window_chunk_rows == 0) {
      SetError(error, "prefill rows and window chunk rows must be positive");
      return nullptr;
    }
    ExecutorConfig ec;
    ec.max_context = cap;
    ec.max_rows = std::min(options.prefill_rows, cap);
    if (options.verify_rows > kMaxVerifyRows) {
      SetError(error, "verify rows " + std::to_string(options.verify_rows) + " exceed the maximum of " +
                          std::to_string(kMaxVerifyRows));
      return nullptr;
    }
    // Drafting off (0): one logits row, the engine as it was before.
    ec.max_logit_rows = std::max<std::uint32_t>(1, std::min(options.verify_rows, ec.max_rows));
    ec.capture_layers = options.capture_layers;
    ec.kv_type = model->backend_.kv_type;
    ec.exact = model->backend_.exact;
    model->window_chunk_rows_ = options.window_chunk_rows;
    // A ring holds what a pass writes before it reads (MinRingRows) and what
    // a checkpoint's window components keep (window_rows + chunk - 1).
    const std::uint32_t window_rows = c.sliding_window - 1;
    ec.ring_rows = RoundUp(std::max(MinRingRows(c.sliding_window, ec.max_rows),
                                    window_rows + options.window_chunk_rows - 1),
                           options.window_chunk_rows);
    model->executor_config_ = ec;
    model->executor_override_ = options.executor_override;
    return model;
  } catch (const std::exception& e) {
    SetError(error, std::string("loading the model failed: ") + e.what());
    return nullptr;
  }
}

std::uint64_t Model::FullLayerBytesPerToken() const {
  const std::uint64_t full =
      static_cast<std::uint64_t>(std::ranges::count(config().sliding, false));
  const std::uint64_t element = backend_.kv_type == KvType::kF16 ? 2 : 4;
  return full * 2 * config().KvWidth() * element;
}

std::uint64_t Model::RingBytes() const {
  const std::uint64_t sliding =
      static_cast<std::uint64_t>(std::ranges::count(config().sliding, true));
  const std::uint64_t element = backend_.kv_type == KvType::kF16 ? 2 : 4;
  return sliding * 2 * executor_config_.ring_rows * config().KvWidth() * element;
}

std::uint64_t Model::SessionBytes(std::uint32_t context) const {
  return RingBytes() + FullLayerBytesPerToken() * context +
         static_cast<std::uint64_t>(config().vocab_size) * sizeof(float);
}

bool Model::IsStopToken(std::int32_t id) const noexcept {
  return id >= 0 && static_cast<std::size_t>(id) < stop_.size() && stop_[static_cast<std::size_t>(id)];
}

std::string Model::TokenText(std::int32_t id) const {
  if (id < 0) return {};
  return tokenizer_->TokenBytes(static_cast<rt::TokenId>(id));
}

std::string Model::Decode(std::span<const std::int32_t> ids) const {
  std::vector<rt::TokenId> converted(ids.begin(), ids.end());
  return tokenizer_->Decode(converted);
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  const auto ids = tokenizer_->Encode(text, true);
  return {ids.begin(), ids.end()};
}

std::optional<ChatPrompt> Model::PrepareChat(const rt::JsonValue& messages,
                                             const rt::JsonValue* tools,
                                             const rt::ChatRenderOptions& options,
                                             std::string* error) const {
  const auto rendered = RenderChatProtected(messages, tools, options, error);
  if (!rendered) return std::nullopt;
  const auto ids = TokenizeChat(*tokenizer_, *rendered);
  ChatPrompt prompt;
  prompt.tokens.assign(ids.begin(), ids.end());
  if (options.add_generation_prompt) {
    // The generation prompt starts with a control token, so it tokenizes
    // apart from the text before it.
    const std::string_view generation = "<|im_start|>assistant\n";
    const auto start = tokenizer_->FindToken("<|im_start|>");
    if (rendered->text.ends_with(generation) && start != rt::kInvalidTokenId) {
      std::vector<rt::TokenId> suffix{start};
      const auto rest = tokenizer_->Encode("assistant\n", false);
      suffix.insert(suffix.end(), rest.begin(), rest.end());
      if (suffix.size() < ids.size() &&
          std::equal(suffix.begin(), suffix.end(), ids.end() - static_cast<std::ptrdiff_t>(suffix.size()))) {
        prompt.stable_prefix_tokens = ids.size() - suffix.size();
      }
    }
  }
  return prompt;
}

std::unique_ptr<Session> Model::CreateSession(std::string* error) {
  auto executor = executor_override_ ? executor_override_(weights_, executor_config_, error)
                                     : CreateExecutor(backend_, weights_, executor_config_, error);
  if (!executor) return nullptr;
  return std::make_unique<Session>(shared_from_this(), std::move(executor));
}

Session::Session(std::shared_ptr<Model> model, std::unique_ptr<Executor> executor)
    : model_(std::move(model)), executor_(std::move(executor)) {
  if (!model_ || !executor_) throw std::invalid_argument("session needs a model and an executor");
}

void Session::Fail() noexcept {
  tokens_.clear();
  logits_valid_ = false;
  verify_open_ = false;
  saved_logits_.clear();
  verify_logits_.clear();
  ClearTaps();
}

void Session::SettleVerification() noexcept {
  verify_open_ = false;
  saved_logits_.clear();
}

void Session::ClearTaps() noexcept {
  taps_first_ = 0;
  taps_rows_ = 0;
  taps_.clear();
}

void Session::Reset() noexcept {
  const std::lock_guard lock(mutex_);
  executor_->Reset();
  Fail();
}

// The executor wrote [tap][row][hidden]; keep [row][tap][hidden].
void Session::AppendTaps(std::uint32_t first_position, std::uint32_t rows, std::span<const float> capture) {
  if (capture.empty()) return;
  const std::size_t hidden = executor_->Info().hidden;
  const std::size_t taps = executor_->Info().capture_layers.size();
  if (taps_rows_ == 0 || taps_first_ + taps_rows_ != first_position) {
    taps_.clear();
    taps_rows_ = 0;
    taps_first_ = first_position;
  }
  const std::size_t base = taps_.size();
  taps_.resize(base + std::size_t{rows} * taps * hidden);
  for (std::size_t k = 0; k < taps; ++k) {
    for (std::size_t t = 0; t < rows; ++t) {
      std::copy_n(capture.data() + (k * rows + t) * hidden, hidden,
                  taps_.data() + base + (t * taps + k) * hidden);
    }
  }
  taps_rows_ += rows;
}

void Session::TrimTaps(std::uint32_t position) noexcept {
  if (taps_rows_ == 0) return;
  if (position <= taps_first_) {
    ClearTaps();
    return;
  }
  if (position >= taps_first_ + taps_rows_) return;
  taps_rows_ = position - taps_first_;
  taps_.resize(std::size_t{taps_rows_} * executor_->Info().capture_layers.size() * executor_->Info().hidden);
}

TapRows Session::TakeTaps() {
  const std::lock_guard lock(mutex_);
  if (verify_open_) throw std::logic_error("settle the open verification before taking taps");
  TapRows out;
  out.first_position = taps_first_;
  out.rows = taps_rows_;
  out.data = std::move(taps_);
  taps_.clear();
  taps_first_ += taps_rows_;
  taps_rows_ = 0;
  return out;
}

void Session::AdoptRestored(std::vector<std::int32_t> tokens, std::vector<float> logits) {
  const std::lock_guard lock(mutex_);
  SettleVerification();
  verify_logits_.clear();
  ClearTaps();
  if (tokens.size() != executor_->Position()) {
    throw std::logic_error("restored tokens do not match the executor position");
  }
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  logits_valid_ = logits_.size() == model_->VocabSize();
}

// Every pass returns the logits of its last row: a checkpoint can be taken
// after any pass and a prompt that ends at a pass end needs nothing more.
void Session::ForwardChunks(std::span<const std::int32_t> tokens) {
  const std::uint32_t max_rows = executor_->Info().max_rows;
  const std::uint32_t vocab = executor_->Info().vocab;
  const std::size_t tap_floats =
      executor_->Info().capture_layers.size() * std::size_t{executor_->Info().hidden};
  std::size_t done = 0;
  while (done < tokens.size()) {
    const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(max_rows, tokens.size() - done));
    const auto chunk = tokens.subspan(done, n);
    // Everything the pass will overwrite is handed to the cache first (not
    // under the lock: the guard waits for readers that take it); a refusal
    // here or in the executor's argument checks changes nothing.
    const WritePlan plan = executor_->PlanForward(n);
    if (guard_) guard_(plan);
    const std::lock_guard lock(mutex_);
    std::vector<float> logits(vocab);
    std::vector<float> capture(tap_floats * n);
    const std::uint32_t first = executor_->Position();
    try {
      executor_->Forward(chunk, first, LogitRows::kLast, logits, tap_, capture);
    } catch (...) {
      // kFailed: rows are undefined, so nothing about the session is kept.
      // kReady: the call was refused and the session is as it was.
      if (executor_->State() != ExecState::kReady) Fail();
      throw;
    }
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
    logits_ = std::move(logits);
    logits_valid_ = true;
    AppendTaps(first, n, capture);
    done += n;
  }
}

void Session::SyncImpl(std::span<const std::int32_t> target) {
  if (target.empty()) throw std::invalid_argument("empty prompt");
  const std::uint32_t cap = executor_->Info().max_context;
  if (target.size() > cap) {
    throw std::invalid_argument("prompt of " + std::to_string(target.size()) +
                                " tokens exceeds the context cap of " + std::to_string(cap) +
                                " tokens");
  }
  {
    const std::lock_guard lock(mutex_);
    SettleVerification();
    verify_logits_.clear();
  }
  if (executor_->State() != ExecState::kReady) {
    const std::lock_guard lock(mutex_);
    executor_->Reset();
    Fail();
  }
  std::size_t common = 0;
  const std::size_t most = std::min(tokens_.size(), target.size() - 1);
  while (common < most && tokens_[common] == target[common]) ++common;
  if (common < tokens_.size()) {
    // The cache keeps a sliding window; Rewind says whether the rows the
    // continuation reads are still there. If not, only a full recompute is
    // correct (a shallower position needs even older rows).
    bool rewound = false;
    {
      const std::lock_guard lock(mutex_);
      rewound = executor_->Rewind(static_cast<std::uint32_t>(common));
      if (rewound) {
        tokens_.resize(common);
        logits_valid_ = false;
        TrimTaps(static_cast<std::uint32_t>(common));
      }
    }
    if (!rewound) {
      // Rows below may be read by captures in flight, and the recompute
      // reuses their ring slots: tell the cache before anything changes.
      if (guard_) guard_(WritePlan{{0, cap}, {0, 0}});
      const std::lock_guard lock(mutex_);
      executor_->Reset();
      Fail();
      common = 0;
    }
  }
  ForwardChunks(target.subspan(common));
}

bool Session::Sync(std::span<const std::int32_t> target, std::string* error) {
  try {
    SyncImpl(target);
    return true;
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return false;
  }
}

bool Session::Evaluate(std::int32_t token, std::string* error) {
  try {
    const std::uint32_t cap = executor_->Info().max_context;
    if (tokens_.size() + 1 > cap) {
      throw std::invalid_argument("context cap of " + std::to_string(cap) + " tokens reached");
    }
    if (executor_->State() != ExecState::kReady || executor_->Position() != tokens_.size()) {
      throw std::logic_error("session holds no valid state to continue");
    }
    {
      const std::lock_guard lock(mutex_);
      SettleVerification();
      verify_logits_.clear();
    }
    const std::int32_t one[] = {token};
    ForwardChunks(one);
    return true;
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return false;
  }
}

bool Session::Verify(std::span<const std::int32_t> tokens, std::string* error) {
  try {
    const ExecutorInfo& info = executor_->Info();
    const auto rows = static_cast<std::uint32_t>(tokens.size());
    if (rows == 0 || rows > info.max_logit_rows || rows > info.max_rows) {
      throw std::invalid_argument("a verification covers 1 to " +
                                  std::to_string(std::min(info.max_logit_rows, info.max_rows)) +
                                  " tokens, not " + std::to_string(rows));
    }
    // Exactness: only passes the executor computes row for row like one-row
    // passes may stand in for one-row decoding.
    if (rows > info.batch_invariant_rows) {
      throw std::invalid_argument("a verification of " + std::to_string(rows) +
                                  " rows is wider than the executor's batch-invariant width of " +
                                  std::to_string(info.batch_invariant_rows));
    }
    if (tokens_.size() + rows > info.max_context) {
      throw std::invalid_argument("context cap of " + std::to_string(info.max_context) + " tokens reached");
    }
    if (executor_->State() != ExecState::kReady || executor_->Position() != tokens_.size() || !logits_valid_) {
      throw std::logic_error("session holds no valid state to verify from");
    }
    const WritePlan plan = executor_->PlanForward(rows);
    if (guard_) guard_(plan);
    const std::lock_guard lock(mutex_);
    const std::size_t vocab = info.vocab;
    std::vector<float> logits(std::size_t{rows} * vocab);
    std::vector<float> capture(info.capture_layers.size() * std::size_t{info.hidden} * rows);
    const std::uint32_t first = executor_->Position();
    try {
      executor_->Forward(tokens, first, LogitRows::kAll, logits, tap_, capture);
    } catch (...) {
      if (executor_->State() != ExecState::kReady) Fail();
      throw;
    }
    saved_logits_ = std::move(logits_);
    tokens_.insert(tokens_.end(), tokens.begin(), tokens.end());
    logits_.assign(logits.end() - static_cast<std::ptrdiff_t>(vocab), logits.end());
    logits_valid_ = true;
    verify_logits_ = std::move(logits);
    verify_base_ = first;
    verify_count_ = rows;
    verify_open_ = true;
    AppendTaps(first, rows, capture);
    return true;
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return false;
  }
}

bool Session::AcceptVerified(std::uint32_t accepted, std::string* error) {
  try {
    const std::lock_guard lock(mutex_);
    if (!verify_open_) throw std::logic_error("no open verification");
    if (accepted > verify_count_) {
      throw std::invalid_argument("accepted " + std::to_string(accepted) + " of " +
                                  std::to_string(verify_count_) + " verified tokens");
    }
    // Where plain decoding would be after `accepted` of the verified tokens.
    const std::uint32_t target = faults_.rewind_from_end
                                     ? static_cast<std::uint32_t>(tokens_.size()) - accepted
                                     : verify_base_ + accepted;
    if (!executor_->Rewind(target)) {
      // Cannot happen with a ring of window + max_rows - 1 rows; if it does,
      // the rows below are gone and only a recompute is correct.
      executor_->Reset();
      Fail();
      throw std::logic_error("the executor could not rewind to position " + std::to_string(target));
    }
    const std::size_t vocab = executor_->Info().vocab;
    tokens_.resize(target);
    if (faults_.stale_frontier) {
      // logits_ stays the last verified row's
    } else if (accepted > 0) {
      logits_.assign(verify_logits_.begin() + static_cast<std::ptrdiff_t>((accepted - 1) * vocab),
                     verify_logits_.begin() + static_cast<std::ptrdiff_t>(accepted * vocab));
    } else {
      logits_ = std::move(saved_logits_);
    }
    logits_valid_ = true;
    if (!faults_.keep_rejected_taps) TrimTaps(target);
    SettleVerification();
    return true;
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return false;
  }
}

}  // namespace gufo::models::ricursive

// The serving runner. Base class: ext::TextModelRunner, not HipTextModelRunner:
// that one owns a DeviceProbe, which allocates device memory on construction,
// and the CPU backend must never touch the GPU. The runner creates a probe
// itself only when the backend uses the device (PollDevice / DeviceUsable).

#include "runner.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <gufo/ext/gguf_identity.hpp>
#include <gufo/ext/host.hpp>
#include <gufo/ext/host_budget.hpp>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "chat_request.hpp"
#include "continuation_adapter.hpp"
#include "drafter_config.hpp"
#include "output_parser.hpp"
#if defined(RICURSIVE_WITH_CPU_BACKEND)
#include "drafter_engine_cpu.hpp"
#endif
#if defined(RICURSIVE_HAVE_GPU_EXECUTOR)
#include "hip/drafter_gpu.hpp"
#endif

#if defined(RICURSIVE_HIP_CACHE)
#include <gufo/ext/component_cache_resources.hpp>
#endif

namespace gufo::models::ricursive {
namespace {

using ext::TextRunnerToken;

constexpr std::string_view kStateAbi = "ricursive-session-v1";

std::int32_t EngineToken(TextRunnerToken token) {
  if (token > static_cast<TextRunnerToken>(std::numeric_limits<std::int32_t>::max())) {
    throw std::invalid_argument("ricursive token ID exceeds engine range");
  }
  return static_cast<std::int32_t>(token);
}

std::vector<std::int32_t> EngineTokens(std::span<const TextRunnerToken> tokens) {
  std::vector<std::int32_t> converted;
  converted.reserve(tokens.size());
  for (const auto token : tokens) converted.push_back(EngineToken(token));
  return converted;
}

bool IsSha256Hex(std::string_view value) {
  return value.size() == 64 && value.find_first_not_of("0123456789abcdef") == std::string_view::npos;
}

class State final : public ext::TextRunnerState {
 public:
  State(Session& session, cache::Adapter& adapter, cache::Slot& slot)
      : session_(&session), adapter_(&adapter), slot_(&slot) {}
  void Invalidate() noexcept override { (void)adapter_->Invalidate(*slot_); }
  [[nodiscard]] ext::TextRunnerMeasuredResources MeasuredResources() const noexcept override {
    std::uint64_t drafter_bytes = 0;
#if defined(ENGINE_ENABLE_HIP)
    if (drafting_) drafter_bytes = drafting_->engine().DeviceBytes();  // its rings and scratch (0: not tracked)
#endif
    return {.per_request_state_bytes = session_->model().SessionBytes(session_->model().MaxContext()) + drafter_bytes,
            .temporary_scratch_bytes = 0};
  }
  [[nodiscard]] Session& session() const { return *session_; }

#if defined(ENGINE_ENABLE_HIP)
  /// Drafting for this state's session (see DraftedSession).
  void EnableDrafting(const SpeculativeSetup& setup) { drafting_ = DraftedSession::For(*session_, setup); }
  [[nodiscard]] DraftedSession* drafting() const noexcept { return drafting_.get(); }
  [[nodiscard]] SpeculativeDecoder* decoder() const noexcept {
    return drafting_ ? &drafting_->decoder() : nullptr;
  }
#endif

 private:
  Session* session_;
  cache::Adapter* adapter_;
  cache::Slot* slot_;
#if defined(ENGINE_ENABLE_HIP)
  std::shared_ptr<DraftedSession> drafting_;  ///< the session's own (it outlives this state)
#endif
};

State& Get(ext::TextRunnerState& state) {
  auto* ours = dynamic_cast<State*>(&state);
  if (ours == nullptr) throw std::logic_error("text runner state is not ricursive");
  return *ours;
}
const State& Get(const ext::TextRunnerState& state) {
  const auto* ours = dynamic_cast<const State*>(&state);
  if (ours == nullptr) throw std::logic_error("text runner state is not ricursive");
  return *ours;
}

class Runner final : public ext::TextModelRunner {
 public:
  Runner(std::shared_ptr<Model> model, const std::string& artifact_fingerprint,
         std::shared_ptr<const SpeculativeSetup> speculative)
      : model_(std::move(model)), speculative_(std::move(speculative)) {
    std::ostringstream identity;
    if (!artifact_fingerprint.empty()) {
      identity << "schema=gufo-text-continuation-v2\n"
               << "model_kind=ricursive\n"
               << "artifact_id=" << ext::kGgufIdentityScheme << ':' << artifact_fingerprint << '\n'
               << "tokenizer=embedded-in-artifact\n"
               << "chat_template_sha256=" << model_->TemplateSha256() << '\n'
               << "state_abi=" << kStateAbi << '\n'
               << "kv_layout=token-major-global+sliding-ring-logical\n"
               << "position_policy=absolute-v1\n"
               << "adapters=none\n";
    } else {
      // Without an artifact fingerprint checkpoints stay in this process.
      identity << "schema=gufo-process-text-continuation-v1\n"
               << "model_instance=" << model_.get() << '\n';
    }
    const std::string canonical = identity.str();
    identity_.assign(canonical.begin(), canonical.end());
    if (model_->backend().uses_device) probe_ = std::make_unique<ext::DeviceProbe>();
  }

  [[nodiscard]] DeviceProbeStatus PollDevice() const override {
    return probe_ ? probe_->Poll() : DeviceProbeStatus::kUsable;
  }
  [[nodiscard]] bool DeviceUsable() const override {
    return probe_ ? probe_->WaitUsable(std::chrono::seconds(5), "scheduler") : true;
  }

  [[nodiscard]] std::unique_ptr<ext::ComponentCacheResources> CreateComponentCacheResources(
      cache::ResourceLedger& ledger, std::size_t) const override {
    if (!model_->backend().uses_device) {
      throw std::runtime_error(
          "the ricursive CPU backend serves no component cache (prompt, chat and bench only); " +
          BackendHelp());
    }
#if defined(RICURSIVE_HIP_CACHE)
    auto adapter = std::make_unique<ContinuationAdapter>(model_, identity_);
    hip::ComponentCacheHooks hooks{
        [] { return ext::HostSnapshotCeilingBytes(); },
        [](std::string_view message) { server::Logger::Warn("cache", message); },
        [](std::string_view message) { server::Logger::Info("cache", message); }};
    auto resources = hip::CreateHipComponentCacheResources(std::move(adapter), ledger, hooks);
    // Durable captures need this hook, but the adapter lends no rows
    // (BorrowRowsView is empty), so persistence never reads a slot through it.
    resources->copy_observed_rows = [](auto, auto, cache::Stream&) -> cache::Completion {
      throw std::logic_error("ricursive lends no rows");
    };
    return resources;
#else
    (void)ledger;
    throw std::runtime_error("this build has no HIP component cache");
#endif
  }
  [[nodiscard]] std::unique_ptr<ext::TextRunnerState> BindComponentState(
      cache::Adapter& adapter, cache::Slot& slot) const override {
    auto& ours = dynamic_cast<ContinuationAdapter&>(adapter);
    auto state = std::make_unique<State>(ours.GetSession(slot), adapter, slot);
#if defined(ENGINE_ENABLE_HIP)
    if (speculative_) state->EnableDrafting(*speculative_);
#endif
    return state;
  }
  void ReconcileComponentState(ext::TextRunnerState& state, std::size_t position) const override {
    if (Get(state).session().Position() != position) {
      throw std::logic_error("ricursive component state position differs");
    }
    // The session told the drafter's context to forget everything below the restored position
    // (tap sink); nothing is pending any more.
    Restart(state);
  }

  [[nodiscard]] ext::TextRunnerDescriptor Descriptor() const override {
    return {
        .model_id = model_->ModelName(),
        .state_abi = std::string(kStateAbi),
        .max_context = model_->MaxContext(),
        .capabilities =
            ext::TextRunnerCapabilities{
                .supports_vision = false,
                .incremental_prefill = true,
                // Every token is evaluated when it is advanced.
                // With drafting the last emitted token stays pending (not in the target) when a
                // request ends; the next request's prompt carries it.
                .final_token_advance_required = speculative_ == nullptr,
                // Chunked prefill and single-token decode agree bit for bit
                // only on a backend that never rounds activations by width.
                .incremental_text_is_exact = model_->backend().exact,
                .multi_token_decode = speculative_ != nullptr,
                .batched_multi_token_decode = false,
                .prefix_reuse = true,
                .in_pass_checkpoint = false,
                .partial_prefill_retention = false,
                .prefill_rollback = false,
                .prefill_alignment = 1,
            },
        // The model card gives no sampling recommendation: gufo's defaults.
        .sampling_defaults = std::nullopt,
    };
  }

  [[nodiscard]] ext::TextRunnerResourceClaim ResourceClaim() const override {
    return {
        // The drafter's weights and, per session, its context rings and scratch (hip/drafter_executor.hpp).
        .resident_weights_bytes = model_->reader().GetSize() + (speculative_ ? speculative_->reader->GetSize() : 0),
        .state_capacity_bytes = std::nullopt,
        .per_request_state_bytes =
            model_->SessionBytes(model_->MaxContext()) +
            (speculative_ ? DrafterSessionBytes(speculative_->weights.config, model_->VocabSize(),
                                                EffectiveDrafterOptions(*speculative_, model_->backend()))
                          : 0),
        .temporary_scratch_bytes = 0,
        .retained_snapshot_capacity_bytes = ext::HostSnapshotBudgetBytes(),
        .retained_snapshot_ceiling_bytes = ext::HostSnapshotCeilingBytes(),
        .requires_device_runtime_lock = model_->backend().uses_device,
    };
  }

  [[nodiscard]] std::vector<ext::TextExecutionPlan> SupportedPlans() const override {
    return {{.kind = ext::TextExecutionPlanKind::kSerial, .physical_width = 1}};
  }

  [[nodiscard]] std::string ToolDialectId() const override { return std::string(kToolDialectId); }
  [[nodiscard]] ReasoningMarkup ReasoningDelimiters() const override {
    return MakeReasoningDelimiters();
  }
  // `<think>` is plain text of several tokens, not one control token, and
  // there is no switch to turn reasoning off or limit it: no budget tags.
  [[nodiscard]] ReasoningTags ReasoningBudgetTags() const override { return {}; }
  [[nodiscard]] std::uint32_t VocabularySize() const override { return model_->VocabSize(); }

  [[nodiscard]] std::vector<TextRunnerToken> Tokenize(std::string_view text) const override {
    const auto tokens = model_->Tokenize(text);
    return {tokens.begin(), tokens.end()};
  }
  [[nodiscard]] std::optional<std::vector<TextRunnerToken>> RenderAndTokenize(
      const ext::ChatRequest& request) const override {
    auto prepared = PreparePrompt(request);
    if (!prepared) return std::nullopt;
    return std::move(prepared->tokens);
  }
  [[nodiscard]] std::optional<ext::TextPreparedPrompt> PreparePrompt(
      const ext::ChatRequest& request) const override {
    const ChatPrompt prompt = PrepareRequest(*model_, request);
    ext::TextPreparedPrompt prepared;
    prepared.tokens.assign(prompt.tokens.begin(), prompt.tokens.end());
    prepared.cache_prefix_tokens = prompt.stable_prefix_tokens;
    return prepared;
  }
  [[nodiscard]] std::optional<std::string> RenderPrompt(const ext::ChatRequest& request) const override {
    return RenderRequest(request);
  }
  [[nodiscard]] ext::InitialOutputState InitialOutputState(const ext::ChatRequest&) const override {
    // The generation prompt never opens <think>: the model decides.
    return ext::InitialOutputState::kAuto;
  }

  [[nodiscard]] std::shared_ptr<const sampling::ConstraintVocabulary> BuildConstraintVocabulary()
      const override {
    return std::make_shared<const sampling::ConstraintVocabulary>(
        model_->VocabSize(), [this](std::uint32_t id) {
          const auto token = static_cast<std::int32_t>(id);
          return sampling::ConstraintVocabulary::Piece{model_->TokenText(token),
                                                       model_->IsStopToken(token)};
        });
  }

  [[nodiscard]] std::string Decode(std::span<const TextRunnerToken> tokens) const override {
    return model_->Decode(EngineTokens(tokens));
  }

  void PreparePrefixReuse(ext::TextRunnerState& state,
                          std::span<const TextRunnerToken> prefix) const override {
    if (Get(state).session().Position() != prefix.size()) {
      throw std::logic_error("ricursive reused prefix does not match checkpoint");
    }
    Restart(state);
  }

  // One pass per call: at most Model::PrefillRows() tokens, the same
  // boundaries ContinuationAdapter::PlanPrefill reports.
  [[nodiscard]] ext::TextPrefillStep Prefill(ext::TextRunnerState& state,
                                             std::span<const TextRunnerToken> prompt,
                                             std::size_t offset,
                                             std::size_t max_input_tokens) const override {
    auto& ours = Get(state);
    auto& session = ours.session();
    Restart(state);
    if (offset != session.Position() || offset >= prompt.size()) {
      throw std::logic_error("ricursive prefill offset does not match state");
    }
    const std::size_t step = std::min({std::max<std::size_t>(max_input_tokens, 1),
                                       std::size_t{model_->PrefillRows()}, prompt.size() - offset});
    const auto prefix = EngineTokens(prompt.first(offset + step));
    std::string error;
    if (!session.Sync(prefix, &error)) {
      ours.Invalidate();
      throw std::runtime_error("ricursive prefill failed: " + error);
    }
    return {.consumed_tokens = step, .decode_ready = offset + step == prompt.size()};
  }

  [[nodiscard]] ext::TextDecodeSelection SelectNext(ext::TextRunnerState& state,
                                                    sampling::SamplerState& sampler) const override {
    auto& session = Get(state).session();
    if (session.Position() >= model_->MaxContext()) return {.stop = true, .piece = {}};
    const auto logits = session.Logits();
    if (logits.empty()) throw std::runtime_error("ricursive token selection has no logits");
    const auto token = static_cast<std::int32_t>(sampler.Sample(logits));
    auto logprobs = sampling::CollectLogprobs(logits, static_cast<sampling::TokenId>(token),
                                              sampler.config().logprobs);
    if (state.stop_at_eos() && model_->IsStopToken(token)) {
      return {.stop = true, .token = 0, .piece = {}, .logprobs = std::move(logprobs)};
    }
    return {.stop = false,
            .token = static_cast<TextRunnerToken>(token),
            .piece = model_->TokenText(token),
            .logprobs = std::move(logprobs)};
  }

  [[nodiscard]] ext::TextDecodeStep DecodeStep(ext::TextRunnerState& state, std::size_t max_tokens,
                                               sampling::SamplerState& sampler) const override {
#if defined(ENGINE_ENABLE_HIP)
    SpeculativeDecoder* decoder = Get(state).decoder();
    // Sampled requests, penalties, grammars, reasoning budgets and log-probabilities draft
    // nothing: select and advance, as without speculation.
    if (decoder != nullptr && max_tokens > 0 && sampler.config().can_use_unmodified_argmax() &&
        !sampler.config().logprobs) {
      ext::TextDecodeStep step;
      DecodeResult result;
      try {
        result = decoder->Step(max_tokens, sampler, state.stop_at_eos());
      } catch (...) {
        Get(state).Invalidate();
        throw;
      }
      for (const auto token : result.tokens) {
        step.selections.push_back({.stop = false,
                                   .token = static_cast<TextRunnerToken>(token),
                                   .piece = model_->TokenText(token)});
      }
      step.stop = result.stop;
      step.draft_rounds = result.draft_tokens > 0 ? 1 : 0;
      step.draft_tokens = result.draft_tokens;
      step.draft_accepted_tokens = result.accepted_tokens;
      return step;
    }
#endif
    return ext::TextModelRunner::DecodeStep(state, max_tokens, sampler);
  }

  void Advance(ext::TextRunnerState& state, TextRunnerToken token) const override {
    auto& ours = Get(state);
    std::string error;
    if (!ours.session().Evaluate(EngineToken(token), &error)) {
      ours.Invalidate();
      throw std::runtime_error("ricursive decode failed: " + error);
    }
  }

  [[nodiscard]] std::optional<ext::TextDecodeSelection> PreviewFirstToken(
      ext::TextRunnerState& state, sampling::SamplerState& sampler) const override {
    return SelectNext(state, sampler);
  }

  [[nodiscard]] std::size_t CheckpointPosition(const ext::TextRunnerState& state) const override {
    return Get(state).session().Position();
  }

 private:
  /// A new request (or a changed state) on this session: no drafted token is pending.
  static void Restart(ext::TextRunnerState& state) {
#if defined(ENGINE_ENABLE_HIP)
    if (auto* decoder = Get(state).decoder()) decoder->Restart();
#else
    (void)state;
#endif
  }

  std::shared_ptr<Model> model_;
  std::shared_ptr<const SpeculativeSetup> speculative_;
  cache::Identity identity_;
  std::unique_ptr<ext::DeviceProbe> probe_;
};

void SetError(std::string* error, std::string message) {
  if (error != nullptr) *error = std::move(message);
}

}  // namespace

std::shared_ptr<const SpeculativeSetup> LoadSpeculativeSetup(const std::string& draft_model_path,
                                                             const core::GgufReader& target_reader,
                                                             std::uint32_t max_draft_tokens, std::string* error) {
  try {
    std::string detail;
    const auto target = ParseConfig(target_reader, &detail);
    if (!target) {
      SetError(error, detail);
      return nullptr;
    }
    std::string open_error;
    std::shared_ptr<const core::GgufReader> reader = core::GgufReader::OpenFile(draft_model_path, &open_error);
    if (!reader) {
      SetError(error, "cannot open the drafter " + draft_model_path + ": " + open_error);
      return nullptr;
    }
    const auto config = ParseDrafterConfig(*reader, *target, &detail);
    if (!config) {
      SetError(error, "drafter: " + detail);
      return nullptr;
    }
    auto weights = ResolveDrafterWeights(*reader, *config, *target, &detail);
    if (!weights) {
      SetError(error, "drafter: " + detail);
      return nullptr;
    }
    auto setup = std::make_shared<SpeculativeSetup>();
    setup->reader = std::move(reader);
    setup->weights = std::move(*weights);
    setup->max_draft_tokens = std::clamp<std::uint32_t>(max_draft_tokens, 1, kMaxVerifyRows - 1);
    return setup;
  } catch (const std::exception& e) {
    SetError(error, std::string("loading the drafter failed: ") + e.what());
    return nullptr;
  }
}

void ArmForSpeculation(const SpeculativeSetup& setup, ModelOptions& options) {
  options.verify_rows = kMaxVerifyRows;
  options.capture_layers = setup.weights.config.tap_layers;
}

bool HaveDrafterEngine() noexcept {
#if defined(ENGINE_ENABLE_HIP) && (defined(RICURSIVE_WITH_CPU_BACKEND) || defined(RICURSIVE_HAVE_GPU_EXECUTOR))
  return true;
#else
  return false;
#endif
}

void PinSharedDrafter(const SpeculativeSetup& setup, const Model& model) {
  static std::mutex mutex;
  const std::lock_guard lock(mutex);
  if (setup.shared) return;
  if (setup.shared_factory) {
    setup.shared = setup.shared_factory(setup, model);
    return;
  }
#if defined(RICURSIVE_HAVE_GPU_EXECUTOR)
  if (model.backend().uses_device && !setup.engine_factory) {
    std::string error;
    setup.shared = PinGpuDrafter(setup.weights, setup.reader, &error);
    if (!setup.shared) throw std::runtime_error("the GPU drafter could not be uploaded: " + error);
  }
#endif
}

DrafterOptions EffectiveDrafterOptions(const SpeculativeSetup& setup, const BackendChoice& backend) {
  DrafterOptions options = setup.drafter_options;
  options.kv_type = backend.kv_type;
  options.exact = backend.exact;
  if (!options.q8.Any()) options.q8 = DrafterRoutesFromEnvironment();
  return options;
}

std::unique_ptr<DrafterEngine> MakeDrafterEngine(const SpeculativeSetup& setup, const Model& model) {
  if (setup.engine_factory) return setup.engine_factory(setup, model);
  return MakeDefaultDrafterEngine(setup, model);
}

std::unique_ptr<DrafterEngine> MakeDefaultDrafterEngine(const SpeculativeSetup& setup, const Model& model) {
#if defined(RICURSIVE_HAVE_GPU_EXECUTOR)
  if (model.backend().uses_device) {
    const DrafterOptions options = EffectiveDrafterOptions(setup, model.backend());
    std::string error;
    auto engine = MakeGpuDrafterEngine(setup.weights, setup.reader, model.weights(), options, &error);
    if (!engine) throw std::runtime_error("the GPU drafter could not be built: " + error);
    return engine;
  }
#endif
#if defined(RICURSIVE_WITH_CPU_BACKEND)
  return std::make_unique<CpuDrafterEngine>(setup.weights, model.weights(), setup.engine_faults);
#else
  return nullptr;
#endif
}

std::shared_ptr<ext::TextModelRunner> CreateRunnerForModel(std::shared_ptr<Model> model,
                                                           const std::string& artifact_fingerprint,
                                                           std::shared_ptr<const SpeculativeSetup> speculative) {
  if (speculative) {
    if (!HaveDrafterEngine()) throw std::runtime_error("this build has no drafter implementation");
    if (model->MaxVerifyRows() < 2 ||
        model->executor_config().capture_layers != speculative->weights.config.tap_layers) {
      throw std::invalid_argument("the model is not armed for speculation (ArmForSpeculation)");
    }
  }
  return std::make_shared<Runner>(std::move(model), artifact_fingerprint, std::move(speculative));
}

#if defined(ENGINE_ENABLE_HIP)
DraftedSession* DraftingOf(ext::TextRunnerState& state) { return Get(state).drafting(); }
const DrafterHealth* DraftingHealthOf(ext::TextRunnerState& state) { return DraftingHealthOf(Get(state).session()); }

DraftedSession::DraftedSession(Session& session, const SpeculativeSetup& setup, std::shared_ptr<DrafterHealth> health)
    : session_(&session), setup_(&setup) {
  PinSharedDrafter(setup, session.model());
  auto engine = MakeDrafterEngine(setup, session.model());
  if (!engine) throw std::runtime_error("this build has no drafter implementation");
  if (engine->TapLayers() != session.model().executor_config().capture_layers) {
    throw std::logic_error("the model does not capture the drafter's tap layers");
  }
  engine_ = engine.get();
  engine->ResetTo(session.Position());  // rows the session already holds were never delivered
  health_ = health ? std::move(health) : std::make_shared<DrafterHealth>();
  ingest_ = std::make_unique<DrafterIngest>(*engine, health_.get());
  auto backend = std::make_unique<DFlash2DraftBackend>(std::move(engine), setup.faults, health_.get());
  backend_ = backend.get();
  decoder_ = std::make_unique<SpeculativeDecoder>(session, std::move(backend), setup.max_draft_tokens);
  session.SetTapSink(ingest_.get());
}

DraftedSession::~DraftedSession() { session_->SetTapSink(nullptr); }

namespace {
/// What a session carries for drafting (its attachment): the health record, which outlives a drafter that could not be
/// built, and the drafting itself once one was.
struct SessionDrafting final : TapSink {
  explicit SessionDrafting(Session& session, const SpeculativeSetup& setup) : session_(&session), setup(&setup) { session.SetTapSink(this); }
  ~SessionDrafting() override {
    if (!drafted) session_->SetTapSink(nullptr);  // (a drafted session's destructor detaches its own sink)
  }
  // Until a drafter is built this is the session's sink: it hears only of a cold start, which clears the health record (a new
  // conversation tries to build the drafter again). Once built, the drafter's ingest is the sink and does the same.
  void Rows(std::uint32_t, std::uint32_t, std::span<const float>) noexcept override {}
  void Rewound(std::uint32_t) noexcept override {}
  void Replaced(std::uint32_t position) noexcept override {
    if (position == 0) health->Clear();
    else health->Restored();
  }
  Session* session_;
  const SpeculativeSetup* setup;
  std::shared_ptr<DrafterHealth> health = std::make_shared<DrafterHealth>();
  std::shared_ptr<DraftedSession> drafted;
};

void WarnOff(std::string_view what, std::string_view why) noexcept {
  try {
    std::cerr << "ricursive: drafting is off for this request: " << what << " (" << why << "); it generates without drafts\n";
  } catch (...) {
  }
}

std::string DescribeCurrent() noexcept {
  try {
    throw;
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "unknown exception";
  }
}
}  // namespace

const DrafterHealth* DraftingHealthOf(Session& session) {
  const auto held = std::static_pointer_cast<SessionDrafting>(session.attachment());
  return held ? held->health.get() : nullptr;
}

std::shared_ptr<DraftedSession> DraftedSession::For(Session& session, const SpeculativeSetup& setup) noexcept {
  try {
    auto held = std::static_pointer_cast<SessionDrafting>(session.attachment());
    if (held && held->setup != &setup) {
      session.SetAttachment(nullptr);  // another setup: detach the old sink before the new one attaches
      held.reset();
    }
    if (!held) {
      held = std::make_shared<SessionDrafting>(session, setup);
      session.SetAttachment(held);
    }
    if (held->drafted) {
      // A drafter that is switched off is neither reset nor warned about again: its ingest ignores rows, so its context is stale on purpose.
      if (held->health->disabled()) {
        held->drafted->decoder_->Restart();
        return held->drafted;
      }
      // The sink told the engine of every change since the last request; a context that does not end at the
      // session's position (it cannot, short of a missed event) restarts there.
      try {
        if (held->drafted->engine_->End() != session.Position()) held->drafted->engine_->ResetTo(session.Position());
      } catch (...) {
        const std::string why = DescribeCurrent();
        held->health->ResetFailed();
        held->health->Failed("reset at the bind", why, false);
        WarnOff("the drafter's context could not be brought to the session's position", why);
        return nullptr;
      }
      held->drafted->decoder_->Restart();
      return held->drafted;
    }
    if (held->health->disabled() || held->health->build_given_up()) return nullptr;
    try {
      held->drafted = std::make_shared<DraftedSession>(session, setup, held->health);
    } catch (...) {
      const std::string why = DescribeCurrent();
      held->health->BuildFailed(why);
      WarnOff("the drafter could not be built", why);
      return nullptr;
    }
    held->health->BuildSucceeded();
    return held->drafted;
  } catch (...) {
    return nullptr;  // not even the bookkeeping could be made (memory): plain generation
  }
}
#endif

std::shared_ptr<ext::TextModelRunner> CreateRicursiveTextRunner(const ext::TextRunnerOptions& options,
                                                                std::string* error) {
  const bool drafting = options.speculative.backend == ext::TextSpeculativeBackend::kDFlash;
  if (options.speculative.backend != ext::TextSpeculativeBackend::kDisabled && !drafting) {
    SetError(error, "ricursive supports --speculative dflash2 only");
    return nullptr;
  }
  ModelOptions model_options;
  model_options.reader = options.reader;
  model_options.max_context = options.max_context;
  try {
    ApplyProviderOptions(options.provider_options, model_options);
  } catch (const std::invalid_argument& e) {
    SetError(error, e.what());
    return nullptr;
  }
  std::shared_ptr<const SpeculativeSetup> speculative;
  if (drafting) {
    if (!HaveDrafterEngine()) {
      SetError(error,
               "this build has no drafter implementation (the GPU drafter needs the HIP build with the GPU executor; "
               "the CPU reference is built with tests only)");
      return nullptr;
    }
    if (options.speculative.draft_model_path.empty()) {
      SetError(error, "--speculative dflash2 needs --dflash-model");
      return nullptr;
    }
    if (!model_options.reader) {
      std::string open_error;
      std::shared_ptr<const core::GgufReader> opened = core::GgufReader::OpenFile(options.model_path, &open_error);
      if (!opened) {
        SetError(error, "cannot open " + options.model_path + ": " + open_error);
        return nullptr;
      }
      model_options.reader = std::move(opened);
    }
    speculative = LoadSpeculativeSetup(options.speculative.draft_model_path, *model_options.reader,
                                       options.speculative.max_draft_tokens, error);
    if (!speculative) return nullptr;
    ArmForSpeculation(*speculative, model_options);
    const auto backend = BackendFromEnvironment(nullptr);
    if (!backend || !backend->uses_device) {
      std::cerr << "ricursive: --speculative dflash2 on the CPU backend runs the scalar CPU reference drafter (about a second per "
                   "draft step, tests and tiny models): the output is exact, decoding is slower than without speculation\n";
    }
  }
  std::string load_error;
  auto model = Model::Load(options.model_path, model_options, &load_error);
  if (!model) {
    SetError(error, "Failed to create ricursive model: " + load_error);
    return nullptr;
  }
  std::string artifact = options.disk_cache.model_artifact_fingerprint;
  try {
    if (!options.disk_cache.directory.empty()) {
      if (artifact.empty()) artifact = core::GgufIdentityHex(model->reader());
      if (!IsSha256Hex(artifact) || options.disk_cache.capacity_bytes == 0) {
        SetError(error, "ricursive persistent disk cache configuration is invalid");
        return nullptr;
      }
    }
    return CreateRunnerForModel(std::move(model), artifact, std::move(speculative));
  } catch (const std::exception& e) {
    SetError(error, e.what());
    return nullptr;
  }
}

}  // namespace gufo::models::ricursive

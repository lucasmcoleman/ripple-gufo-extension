#include "hip/drafter_executor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "hip/drafter_math.hpp"

namespace gufo::models::ricursive {
namespace {

namespace km = rocm;

constexpr float kMaskScale = 1.0F;  // the drafter reads the target's raw table rows

WeightView View(const TensorRef& t) {
  return WeightView{t.data, t.type, static_cast<std::uint32_t>(t.rows), static_cast<std::uint32_t>(t.cols),
                    t.RowBytes() * t.rows};
}
const float* F32(const TensorRef& t) {
  if (t.type != core::GgmlType::kF32) {
    throw std::runtime_error("ricursive drafter: " + std::string(t.name) + " must be F32");
  }
  return static_cast<const float*>(t.data);
}

/// Element counts of every buffer (one list for the constructor and for DrafterSessionBytes).
struct Sizes {
  std::uint64_t context_pass{0};  ///< floats of the context pass buffers
  std::uint64_t block{0};         ///< floats of the block buffers
  std::uint64_t select{0};        ///< 4-byte elements of the selector's buffers
  std::uint64_t rings{0};         ///< bytes of the K/V rings
  std::uint64_t scratch{0};       ///< bytes of the projection scratch (Q8_1 activations)
  /// The (rows, cols) pairs ReserveProjection is called with; scratch is the largest of their sizes.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> reserve;
  std::uint32_t window{0};
  std::uint32_t append_rows{0};
  std::vector<std::uint32_t> capacity;
};

Sizes Plan(const DrafterConfig& c, std::uint32_t vocab, const DrafterOptions& o) {
  Sizes s;
  s.window = c.context_window != 0 ? c.context_window : o.max_context;
  if (s.window == 0) throw std::invalid_argument("ricursive drafter: max_context is 0");
  const std::uint64_t E = c.embedding_length, Q = c.QueryWidth(), KV = c.KvWidth(), FF = c.feed_forward_length,
                      B = c.block_size, F = c.FeatureWidth(), DYN = 2ULL * c.conv_kernel_size * c.ConvGroups(),
                      R = B - 1, K = c.selector_top_k, P = km::DraftSelectPartials(vocab);
  std::uint32_t min_capacity = UINT32_MAX;
  const std::uint64_t esize = o.kv_type == KvType::kF16 ? 2 : 4;
  for (std::uint32_t l = 0; l < c.block_count; ++l) {
    const std::uint32_t keep = c.sliding[l] ? std::min(s.window, c.sliding_window) : s.window;
    const std::uint32_t cap = keep + o.ring_slack;
    s.capacity.push_back(cap);
    min_capacity = std::min(min_capacity, cap);
    s.rings += 2 * std::uint64_t{cap} * KV * esize;
  }
  s.append_rows = std::max<std::uint32_t>(1, std::min(o.max_append_rows, min_capacity));
  const std::uint64_t M = s.append_rows;
  s.context_pass = M * F + 2 * M * E + 2 * M * KV;
  // h a x0 attn fin (E), q o (Q), kb vb bk bv (KV), dyn, g u (FF), logits, hproj, the 15 MASK rows (E)
  s.block = 5 * B * E + 2 * B * Q + 4 * B * KV + B * DYN + 2 * B * FF + R * vocab + R * c.selector_rank + R * E /*MASK rows*/;
  s.select = B /*tokens*/ + 2 * R * K /*candidates*/ + 2 * R * P * K /*partials*/;
  // The projection scratch holds the Q8_1 activations of the 8-bit routes only: the context pass at more than 16 rows
  // (Project's own W8A8 route) or on the chosen route, the block pass and the head when chosen. `exact` uses none.
  // The widest input of the context pass is fc's (F); of the block pass the FFN down projection's (FF); of the head E.
  if (!o.exact) {
    if (o.q8.context || M > 16) s.reserve.emplace_back(M, F);
    if (o.q8.block) s.reserve.emplace_back(B, std::max({E, Q, FF}));
    if (o.q8.head) s.reserve.emplace_back(R, E);
  }
  for (const auto& [rows, cols] : s.reserve) s.scratch = std::max(s.scratch, km::Q8ActivationBytes(rows, cols));
  return s;
}

}  // namespace

DrafterRoutes ParseDrafterRoutes(std::string_view list) {
  DrafterRoutes r;
  std::size_t at = 0;
  while (at <= list.size()) {
    std::size_t end = list.find(',', at);
    if (end == std::string_view::npos) end = list.size();
    const std::string_view item = list.substr(at, end - at);
    if (item == "block") r.block = true;
    else if (item == "head") r.head = true;
    else if (item == "context") r.context = true;
    else if (item == "all") r = DrafterRoutes{true, true, true};
    else if (!item.empty() && item != "none") {
      throw std::invalid_argument("ricursive drafter: unknown projection route group '" + std::string(item) +
                                  "' (block, head, context, all or none)");
    }
    at = end + 1;
  }
  return r;
}

DrafterRoutes DrafterRoutesFromEnvironment() {
  const char* env = std::getenv("RICURSIVE_DRAFTER_Q8");
  return env != nullptr ? ParseDrafterRoutes(env) : DrafterRoutes{};
}

std::string ToString(const DrafterRoutes& routes) {
  std::string out;
  for (const auto& [on, name] : {std::pair{routes.block, "block"}, std::pair{routes.head, "head"}, std::pair{routes.context, "context"}}) {
    if (on) out += (out.empty() ? "" : ",") + std::string(name);
  }
  return out.empty() ? "none" : out;
}

std::shared_ptr<DrafterHandles> BuildHostDrafterHandles(const DrafterWeights& w) {
  auto h = std::make_shared<DrafterHandles>();
  h->config = w.config;
  h->fc = View(w.fc);
  h->selector_hidden = View(w.selector_hidden);
  h->selector_predecessor = View(w.selector_predecessor);
  h->selector_successor = View(w.selector_successor);
  h->enc_output_norm = F32(w.enc_output_norm);
  h->output_norm = F32(w.output_norm);
  for (const DrafterLayerWeights& l : w.layers) {
    DrafterLayerHandles d;
    d.attn_q = View(l.attn_q);
    d.attn_k = View(l.attn_k);
    d.attn_v = View(l.attn_v);
    d.attn_output = View(l.attn_output);
    d.attn_conv_proj = View(l.attn_conv_proj);
    d.ffn_gate = View(l.ffn_gate);
    d.ffn_up = View(l.ffn_up);
    d.ffn_down = View(l.ffn_down);
    d.ffn_conv_proj = View(l.ffn_conv_proj);
    d.attn_norm = F32(l.attn_norm);
    d.attn_q_norm = F32(l.attn_q_norm);
    d.attn_k_norm = F32(l.attn_k_norm);
    d.attn_conv_base = F32(l.attn_conv_base);
    d.ffn_norm = F32(l.ffn_norm);
    d.ffn_conv_base = F32(l.ffn_conv_base);
    h->layers.push_back(d);
  }
  return h;
}

DrafterTarget BuildHostDrafterTarget(const ModelWeights& target) {
  DrafterTarget t;
  t.table = EmbeddingTable{target.token_embd.data, target.token_embd.type,
                           static_cast<std::size_t>(target.token_embd.rows),
                           static_cast<std::size_t>(target.token_embd.cols)};
  t.output = View(target.output);
  return t;
}

std::uint64_t DrafterSessionBytes(const DrafterConfig& config, std::uint32_t vocab_size,
                                  const DrafterOptions& options) {
  const Sizes s = Plan(config, vocab_size, options);
  return s.rings + s.scratch + 4 * (s.context_pass + s.block + s.select);
}

// ---- buffers -------------------------------------------------------------------------------

struct DrafterExecutor::Buffers {
  explicit Buffers(Ops& o) : ops(o) {}
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
  ~Buffers() {
    try {
      ops.Synchronize();  // nothing may still be reading what is freed here
    } catch (...) {
    }
    for (void* p : owned) ops.Free(p);
  }
  template<class T = float>
  T* Take(std::size_t count) {
    const std::size_t bytes = std::max<std::size_t>(count, 1) * sizeof(T);
    void* p = ops.Allocate(bytes);
    owned.push_back(p);
    total += bytes;
    return static_cast<T*>(p);
  }

  Ops& ops;
  std::vector<void*> owned;
  std::uint64_t total{0};
  // context pass
  float *feats{}, *proj{}, *ctx{}, *kt{}, *vt{};
  // block pass
  float *h{}, *a{}, *x0{}, *attn{}, *fin{}, *q{}, *o{}, *kb{}, *vb{}, *bk{}, *bv{}, *dyn{}, *g{}, *u{},
      *logits{}, *hproj{};
  float* mask{};  ///< the 15 MASK rows of the block input, embedded once per executor
  /// One buffer for what a draft step reads back, so that it is one copy-out: tokens [B] (tokens[0] the anchor, written
  /// before the step), candidate ids [R][K], candidate scores [R][K] (each region begins right after the previous one).
  std::int32_t* tokens{};
  float *cand_scores{}, *part_scores{};
  std::int32_t *cand_ids{}, *part_ids{};
  std::vector<void*> kr, vr;  ///< per layer K and V rings
};

struct DrafterExecutor::Profile {
  using Clock = std::chrono::steady_clock;
  struct Entry {
    std::string name;
    double seconds{0};
  };
  std::vector<Entry> draft, append;  ///< stages in the order first seen
  Clock::time_point last;
  std::uint64_t draft_steps{0}, append_passes{0};
  bool in_append{false};
  void Add(const char* name, double seconds) {
    auto& list = in_append ? append : draft;
    for (auto& e : list) {
      if (e.name == name) {
        e.seconds += seconds;
        return;
      }
    }
    list.push_back({name, seconds});
  }
};

void DrafterExecutor::LapStart(bool context_pass) {
  ops_->Synchronize();
  prof_->in_append = context_pass;
  (context_pass ? prof_->append_passes : prof_->draft_steps) += 1;
  prof_->last = Profile::Clock::now();
}

void DrafterExecutor::LapSlow(const char* stage) {
  ops_->Synchronize();  // the same wait the executor's own profile uses
  const auto now = Profile::Clock::now();
  prof_->Add(stage, std::chrono::duration<double>(now - prof_->last).count());
  prof_->last = now;
}

std::string DrafterExecutor::ProfileReport() {
  if (!prof_) return {};
  std::string out;
  char line[160];
  const auto section = [&](const char* title, const std::vector<Profile::Entry>& list, std::uint64_t n) {
    if (n == 0) return;
    double total = 0;
    for (const auto& e : list) total += e.seconds;
    std::snprintf(line, sizeof line, "%s (%llu, mean ms per call, device waited for at every stage boundary)\n", title,
                  static_cast<unsigned long long>(n));
    out += line;
    for (const auto& e : list) {
      std::snprintf(line, sizeof line, "  %-22s %9.3f ms %5.1f %%\n", e.name.c_str(), 1e3 * e.seconds / static_cast<double>(n),
                    total > 0 ? 100.0 * e.seconds / total : 0.0);
      out += line;
    }
    std::snprintf(line, sizeof line, "  %-22s %9.3f ms\n", "total", 1e3 * total / static_cast<double>(n));
    out += line;
  };
  section("drafter draft steps", prof_->draft, prof_->draft_steps);
  section("drafter context passes", prof_->append, prof_->append_passes);
  prof_->draft.clear();
  prof_->append.clear();
  prof_->draft_steps = prof_->append_passes = 0;
  return out;
}

DrafterExecutor::DrafterExecutor(std::unique_ptr<Ops> ops, std::shared_ptr<const DrafterHandles> drafter,
                                 const DrafterTarget& target, std::shared_ptr<const void> keep_alive,
                                 const DrafterOptions& options, const DrafterExecFaults& faults)
    : ops_(std::move(ops)),
      drafter_(std::move(drafter)),
      target_(target),
      keep_alive_(std::move(keep_alive)),
      options_(options),
      faults_(faults),
      buf_(std::make_unique<Buffers>(*ops_)) {
  if (const char* env = std::getenv("RICURSIVE_PROFILE"); env != nullptr && std::strcmp(env, "1") == 0) {
    prof_ = std::make_unique<Profile>();
  }
  const DrafterConfig& c = drafter_->config;
  if (c.head_dim == 0 || c.head_dim % 2 != 0 || c.head_dim > km::kDraftMaxHeadDim) {
    throw std::invalid_argument("ricursive drafter: head_dim " + std::to_string(c.head_dim) +
                                " (the operations take even values up to 256)");
  }
  if (c.block_size < 2 || c.selector_top_k == 0 || c.selector_top_k > km::kDraftMaxTopK) {
    throw std::invalid_argument("ricursive drafter: block_size or selector_top_k out of range (top-k up to 32)");
  }
  const std::uint32_t vocab = target_.output.rows;
  if (target_.output.cols != c.embedding_length || target_.table.hidden != c.embedding_length ||
      drafter_->selector_predecessor.rows != vocab || drafter_->selector_successor.rows != vocab ||
      target_.table.vocab != vocab || c.selector_top_k > vocab) {
    throw std::invalid_argument("ricursive drafter: drafter and target disagree on hidden size or vocabulary");
  }
  for (const WeightView* w : {&drafter_->selector_predecessor, &drafter_->selector_successor}) {
    if (EncodingOf(w->type) == km::Enc::kNone ||
        km::DequantRowBytes(EncodingOf(w->type), c.selector_rank) == 0) {
      throw std::invalid_argument("ricursive drafter: the selector codebooks must be F32, BF16, Q8_0 or Q4_K");
    }
  }
  const Sizes plan = Plan(c, vocab, options_);
  window_ = plan.window;
  capacity_ = plan.capacity;
  append_rows_ = plan.append_rows;

  const std::size_t E = c.embedding_length, Q = c.QueryWidth(), KV = c.KvWidth(), FF = c.feed_forward_length,
                    B = c.block_size, F = c.FeatureWidth(), DYN = 2 * std::size_t{c.conv_kernel_size} * c.ConvGroups(),
                    R = B - 1, K = c.selector_top_k, P = km::DraftSelectPartials(vocab), M = append_rows_;
  Buffers& b = *buf_;
  b.feats = b.Take(M * F);
  b.proj = b.Take(M * E);
  b.ctx = b.Take(M * E);
  b.kt = b.Take(M * KV);
  b.vt = b.Take(M * KV);
  b.h = b.Take(B * E);
  b.a = b.Take(B * E);
  b.x0 = b.Take(B * E);
  b.attn = b.Take(B * E);
  b.fin = b.Take(B * E);
  b.q = b.Take(B * Q);
  b.o = b.Take(B * Q);
  b.kb = b.Take(B * KV);
  b.vb = b.Take(B * KV);
  b.bk = b.Take(B * KV);
  b.bv = b.Take(B * KV);
  b.dyn = b.Take(B * DYN);
  b.g = b.Take(B * FF);
  b.u = b.Take(B * FF);
  b.logits = b.Take(R * vocab);
  b.hproj = b.Take(R * c.selector_rank);
  b.mask = b.Take(R * E);
  b.tokens = b.Take<std::int32_t>(B + 2 * R * K);
  b.cand_ids = b.tokens + B;
  b.cand_scores = reinterpret_cast<float*>(b.tokens + B + R * K);
  b.part_scores = b.Take(R * P * K);
  b.part_ids = b.Take<std::int32_t>(R * P * K);
  const std::size_t esize = options_.kv_type == KvType::kF16 ? 2 : 4;
  for (std::uint32_t l = 0; l < c.block_count; ++l) {
    b.kr.push_back(b.Take<std::byte>(std::size_t{capacity_[l]} * KV * esize));
    b.vr.push_back(b.Take<std::byte>(std::size_t{capacity_[l]} * KV * esize));
  }
  for (const auto& [rows, cols] : plan.reserve) ops_->ReserveProjection(rows, cols);
  {  // The MASK rows never change: embed them once (this call waits for the device, once per executor).
    const std::vector<std::int32_t> mask_ids(R, static_cast<std::int32_t>(c.mask_token_id));
    ops_->EmbedRows(target_.table, mask_ids, kMaskScale, b.mask);
  }
}

DrafterExecutor::~DrafterExecutor() = default;

std::uint64_t DrafterExecutor::DeviceBytes() const noexcept { return buf_->total + ops_->ProjectionScratchBytes(); }

void DrafterExecutor::Proj(bool q8, const WeightView& w, const float* x, int rows, float* y) {
  if (q8 && !options_.exact) {
    ops_->ProjectQ8(w, x, rows, y);
  } else {
    ops_->Project(w, x, rows, y, options_.exact);
  }
}

bool DrafterExecutor::LayerHasRope(std::size_t layer) const noexcept {
  return drafter_->config.rope[layer] || faults_.rope_on_nope_layer;
}

void DrafterExecutor::ResetTo(std::uint32_t base) {
  base_ = base;
  end_ = base;
  high_ = base;
}

void DrafterExecutor::TruncateTo(std::uint32_t position) {
  if (position >= end_) return;
  if (position <= base_) {
    ResetTo(position);
    return;
  }
  if (std::uint64_t{position} + options_.ring_slack <= high_) {
    // A full window below `position` is gone from the rings (docs/DRAFTER-S2.md, "Stage 2b", item 4). Keep what they still hold:
    // every layer has the rows [high_ - its capacity, high_), so all layers have the rows from high_ - (the smallest capacity)
    // on; the base rises to that row. Nothing left below the new end: the context is empty there. (Until stage 2b this
    // emptied the context whenever the truncation was this deep.)
    const std::uint32_t smallest = *std::min_element(capacity_.begin(), capacity_.end());
    const std::uint32_t oldest = high_ > smallest ? high_ - smallest : 0;
    if (oldest >= position) {
      ResetTo(position);
      return;
    }
    base_ = std::max(base_, oldest);
  }
  end_ = position;
}

void DrafterExecutor::Emit(const DrafterStageFn& stage, std::string_view name, int layer, const float* data,
                           std::size_t rows, std::size_t width) {
  if (!stage) return;
  std::vector<float> host(rows * width);
  ops_->CopyOut(host.data(), data, host.size() * sizeof(float));
  stage(name, layer, host.data(), rows, width);
}

void DrafterExecutor::EmitRing(const DrafterStageFn& stage, std::string_view name, int layer, const void* ring,
                               std::uint32_t capacity, std::uint32_t first_position, std::size_t rows) {
  if (!stage) return;
  const std::size_t KV = drafter_->config.KvWidth();
  const std::size_t esize = options_.kv_type == KvType::kF16 ? 2 : 4;
  std::vector<float> host(rows * KV);
  std::vector<std::uint8_t> raw(KV * esize);
  for (std::size_t r = 0; r < rows; ++r) {
    const std::size_t slot = (std::size_t{first_position} + r) % capacity;
    ops_->CopyOut(raw.data(), static_cast<const std::uint8_t*>(ring) + slot * KV * esize, raw.size());
    for (std::size_t i = 0; i < KV; ++i) {
      if (esize == 2) {
        std::uint16_t bits;
        std::memcpy(&bits, raw.data() + 2 * i, 2);
        host[r * KV + i] = HalfToFloat(bits);
      } else {
        std::memcpy(&host[r * KV + i], raw.data() + 4 * i, 4);
      }
    }
  }
  stage(name, layer, host.data(), rows, KV);
}

void DrafterExecutor::AppendContext(std::span<const float> features, std::size_t rows,
                                    const DrafterStageFn& stage) {
  const DrafterConfig& c = drafter_->config;
  const std::size_t E = c.embedding_length, F = c.FeatureWidth();
  if (features.size() != rows * F) {
    throw std::invalid_argument("ricursive drafter: AppendContext expects rows x taps x hidden floats");
  }
  Buffers& b = *buf_;
  for (std::size_t done = 0; done < rows;) {
    const std::uint32_t n = static_cast<std::uint32_t>(std::min<std::size_t>(append_rows_, rows - done));
    const std::uint32_t first = end_;
    if (prof_) LapStart(true);
    ops_->CopyIn(b.feats, features.data() + done * F, std::size_t{n} * F * sizeof(float));
    Lap("ctx_upload");
    Proj(options_.q8.context, drafter_->fc, b.feats, static_cast<int>(n), b.proj);
    Lap("ctx_fc");
    ops_->RmsNorm(b.proj, drafter_->enc_output_norm, b.ctx, n, static_cast<std::uint32_t>(E), c.rms_eps);
    Lap("ctx_norm");
    Emit(stage, "ctx_h", -1, b.ctx, n, E);
    for (std::uint32_t l = 0; l < c.block_count; ++l) {
      const DrafterLayerHandles& L = drafter_->layers[l];
      Proj(options_.q8.context, L.attn_k, b.ctx, static_cast<int>(n), b.kt);
      Proj(options_.q8.context, L.attn_v, b.ctx, static_cast<int>(n), b.vt);
      Lap("ctx_kv_proj");
      // Norm and rotary of the K rows, then the rows into the ring at slot position % capacity.
      ops_->DraftQkNormRope(nullptr, b.kt, b.vt, L.attn_q_norm, L.attn_k_norm, b.kr[l], b.vr[l],
                            options_.kv_type, n, 0, c.head_count_kv, c.head_dim, c.rms_eps, first, first,
                            capacity_[l], LayerHasRope(l), c.rope_freq_base);
      Lap("ctx_norm_rope_store");
      if (stage) {
        EmitRing(stage, "ctx_k", static_cast<int>(l), b.kr[l], capacity_[l], first, n);
        EmitRing(stage, "ctx_v", static_cast<int>(l), b.vr[l], capacity_[l], first, n);
      }
    }
    end_ += n;
    high_ = std::max(high_, end_);
    done += n;
  }
}

DrafterExecutor::Draft DrafterExecutor::DraftStep(std::int32_t anchor_token, std::uint32_t anchor_position,
                                                  std::uint32_t draft_tokens, const DrafterStageFn& stage) {
  const DrafterConfig& c = drafter_->config;
  Draft result;
  if (anchor_position > end_) {
    throw std::invalid_argument("ricursive drafter: anchor position " + std::to_string(anchor_position) +
                                " is beyond the " + std::to_string(end_) + " context rows held");
  }
  const std::uint32_t d = std::min(draft_tokens, c.block_size - 1);
  if (d == 0) return result;
  const std::uint32_t E = c.embedding_length, FF = c.feed_forward_length, B = c.block_size, NH = c.head_count,
                      NKV = c.head_count_kv, D = c.head_dim, K = c.conv_kernel_size;
  const std::uint32_t vocab = target_.output.rows;
  if (anchor_token < 0 || static_cast<std::uint32_t>(anchor_token) >= vocab) {  // the selector reads the codebook row of it
    throw std::invalid_argument("ricursive drafter: anchor token " + std::to_string(anchor_token) + " outside [0, " +
                                std::to_string(vocab) + ")");
  }
  Buffers& b = *buf_;
  Ops& ops = *ops_;
  const std::uint32_t s = anchor_position;
  // Rows below base_ were never given; rows the rings no longer hold (below high_ - capacity) are not read.
  const auto first_row = [&](std::uint32_t layer) {
    const std::uint32_t cap = capacity_[layer];
    const std::uint32_t window_first = s > window_ ? s - window_ : 0;
    const std::uint32_t ring_first = high_ > cap ? high_ - cap : 0;
    return std::min(std::max({window_first, base_, ring_first}), s);
  };
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));

  // Block input: the anchor token and MASK rows of the target's embedding table, unscaled. Only the anchor's row is
  // embedded (on the host) and uploaded per step; the 15 MASK rows are copied on the device from the copy made at
  // construction. The anchor token goes to the selector's buffer now, while the stream is idle, so that no host
  // transfer waits for work in the middle of the step.
  if (prof_) LapStart(false);
  ops.CopyIn(b.tokens, &anchor_token, sizeof(std::int32_t));
  ops.EmbedRows(target_.table, std::span<const std::int32_t>(&anchor_token, 1), kMaskScale, b.h);
  ops.CopyInternal(b.h + E, b.mask, std::size_t{B - 1} * E * sizeof(float));
  Lap("embed");
  Emit(stage, "emb", -1, b.h, B, E);

  for (std::uint32_t l = 0; l < c.block_count; ++l) {
    const DrafterLayerHandles& L = drafter_->layers[l];
    const int il = static_cast<int>(l);
    // ---- attention ----
    ops.RmsNorm(b.h, L.attn_norm, b.a, B, E, c.rms_eps);
    Lap("norm");
    Emit(stage, "attn_norm", il, b.a, B, E);
    Proj(options_.q8.block, L.attn_conv_proj, b.a, static_cast<int>(B), b.dyn);
    Lap("conv_proj");
    ops.DynamicConv(b.a, b.dyn, L.attn_conv_base, b.x0, B, E, K, c.conv_group_size, 0, false);
    Lap("conv");
    Emit(stage, "attn_conv_in", il, b.x0, B, E);
    Proj(options_.q8.block, L.attn_q, b.x0, static_cast<int>(B), b.q);
    Proj(options_.q8.block, L.attn_k, b.x0, static_cast<int>(B), b.kb);
    Proj(options_.q8.block, L.attn_v, b.x0, static_cast<int>(B), b.vb);
    Lap("qkv_proj");
    // q in place; the block's K and V rows go to the block buffers (fp32, row t in slot t).
    ops.DraftQkNormRope(b.q, b.kb, b.vb, L.attn_q_norm, L.attn_k_norm, b.bk, b.bv, KvType::kF32, B, NH, NKV, D,
                        c.rms_eps, s, 0, B, LayerHasRope(l), c.rope_freq_base);
    Lap("qk_norm_rope");
    const bool sliding = c.sliding[l];
    const std::uint32_t window = sliding ? c.sliding_window + (faults_.window_off_by_one ? 1U : 0U) : 0U;
    ops.DraftAttention(b.q, b.kr[l], b.vr[l], options_.kv_type, capacity_[l], first_row(l), s, b.bk, b.bv, b.o, B,
                       NH, NKV, D, window, scale);
    Lap("attention");
    Proj(options_.q8.block, L.attn_output, b.o, static_cast<int>(B), b.attn);
    Lap("o_proj");
    Emit(stage, "attn_o", il, b.attn, B, E);
    ops.DynamicConv(b.attn, b.dyn, L.attn_conv_base, b.h, B, E, K, c.conv_group_size, 1, true);
    Lap("conv");
    Emit(stage, "h_attn", il, b.h, B, E);

    // ---- feed-forward ----
    ops.RmsNorm(b.h, L.ffn_norm, b.a, B, E, c.rms_eps);
    Lap("norm");
    Emit(stage, "ffn_norm", il, b.a, B, E);
    Proj(options_.q8.block, L.ffn_conv_proj, b.a, static_cast<int>(B), b.dyn);
    Lap("conv_proj");
    ops.DynamicConv(b.a, b.dyn, L.ffn_conv_base, b.x0, B, E, K, c.conv_group_size, 0, false);
    Lap("conv");
    Emit(stage, "mlp_conv_in", il, b.x0, B, E);
    Proj(options_.q8.block, L.ffn_gate, b.x0, static_cast<int>(B), b.g);
    Proj(options_.q8.block, L.ffn_up, b.x0, static_cast<int>(B), b.u);
    Lap("ffn_gate_up");
    ops.SiluMul(b.g, b.u, B * FF);  // no clamp
    Lap("silu_mul");
    Proj(options_.q8.block, L.ffn_down, b.g, static_cast<int>(B), b.attn);
    Lap("ffn_down");
    Emit(stage, "mlp_o", il, b.attn, B, E);
    ops.DynamicConv(b.attn, b.dyn, L.ffn_conv_base, b.h, B, E, K, c.conv_group_size, 1, true);
    Lap("conv");
    Emit(stage, "h", il, b.h, B, E);
  }

  ops.RmsNorm(b.h, drafter_->output_norm, b.fin, B, E, c.rms_eps);
  Lap("norm");
  Emit(stage, "final", -1, b.fin, B, E);

  // ---- logits of rows 1..d with the target's output head (multiplier 1, no softcap), then the selector ----
  Proj(options_.q8.head, target_.output, b.fin + E, static_cast<int>(d), b.logits);
  Lap("head_logits");
  Emit(stage, "logits", -1, b.logits, d, vocab);
  Proj(options_.q8.head, drafter_->selector_hidden, b.fin + E, static_cast<int>(d), b.hproj);
  Lap("selector_proj");
  Emit(stage, "hproj", -1, b.hproj, d, c.selector_rank);
  const std::uint32_t top_k =
      faults_.selector_top_k_limit != 0 ? std::min(c.selector_top_k, faults_.selector_top_k_limit) : c.selector_top_k;
  ops.DraftSelect(b.logits, b.hproj, drafter_->selector_predecessor, drafter_->selector_successor, b.tokens,
                  b.cand_scores, b.cand_ids, b.part_scores, b.part_ids, d, vocab, c.selector_rank, top_k);
  Lap("select");
  // One copy-out (3.9 KB for the real configuration, whatever d is) returns the tokens and the candidates.
  const std::size_t sel_rows = B - 1, sel_k = c.selector_top_k;
  std::vector<std::int32_t> readback(B + 2 * sel_rows * sel_k);
  ops.CopyOut(readback.data(), b.tokens, readback.size() * sizeof(std::int32_t));
  result.tokens.assign(readback.begin() + 1, readback.begin() + 1 + d);
  result.top_k = top_k;
  const auto* ids_at = readback.data() + B;
  result.candidates.resize(std::size_t{d} * top_k);
  result.scores.resize(std::size_t{d} * top_k);
  std::memcpy(result.candidates.data(), ids_at, result.candidates.size() * sizeof(std::int32_t));
  std::memcpy(result.scores.data(), ids_at + sel_rows * sel_k, result.scores.size() * sizeof(float));
  std::uint32_t full_layer = 0;  // the context the draft read is the full layer's (the widest)
  for (std::uint32_t l = 0; l < c.block_count; ++l) {
    if (!c.sliding[l]) {
      full_layer = l;
      break;
    }
  }
  result.context_first = first_row(full_layer);
  result.context_end = s;
  Lap("readback");
  return result;
}

}  // namespace gufo::models::ricursive

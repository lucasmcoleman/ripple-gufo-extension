#include "hip/executor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace gufo::models::ricursive {
namespace {

constexpr std::uint32_t kGroupsSupported[] = {1, 2, 10};

[[noreturn]] void Unsupported(const std::string& what) {
  throw std::invalid_argument("ricursive executor: not supported: " + what);
}

ExecutorInfo MakeInfo(const ModelHandles& model, const ExecutorConfig& config,
                      std::uint32_t batch_invariant_rows) {
  return ExecutorInfo::Make(model.config, NormalizeConfig(model.config, config),
                            batch_invariant_rows);
}

}  // namespace

ExecutorConfig NormalizeConfig(const Config& model, ExecutorConfig config) {
  if (model.head_dim != rocm::kHeadDim) {
    Unsupported("head_dim " + std::to_string(model.head_dim) +
                " (the attention kernels are written for 128)");
  }
  const std::uint32_t group =
      model.head_count_kv == 0 ? 0 : model.head_count / model.head_count_kv;
  if (std::find(std::begin(kGroupsSupported), std::end(kGroupsSupported),
                group) == std::end(kGroupsSupported)) {
    Unsupported("GQA group " + std::to_string(group) + " (kernels: 1, 2 or 10)");
  }
  for (std::size_t l = 0; l < model.swiglu_clamp.size(); ++l) {
    const float c = model.swiglu_clamp[l];
    if (!(std::isfinite(c) && c > 0.0F)) {
      Unsupported("layer " + std::to_string(l) + " has swiglu_clamp " +
                  std::to_string(c) +
                  " (0 = no clamp): ClampedSwiGlu requires a finite clamp > 0");
    }
  }
  if (model.swiglu_clamp.size() != model.block_count) {
    Unsupported("swiglu_clamp has " + std::to_string(model.swiglu_clamp.size()) +
                " entries for " + std::to_string(model.block_count) + " layers");
  }
  if (config.max_context == 0) {
    config.max_context = std::min(kDefaultContextCap, model.context_length);
  }
  if (config.max_rows != 0 && config.max_logit_rows > config.max_rows) {
    config.max_logit_rows = config.max_rows;
  }
  return config;
}

// ---- scratch ---------------------------------------------------------------

struct OpsExecutor::Buffers {
  explicit Buffers(Ops& o) : ops(o) {}
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
  ~Buffers() {
    // Nothing may still be reading what is freed here.
    try {
      ops.Synchronize();
    } catch (...) {  // a failed device: freeing is all that is left to do
    }
    for (void* p : owned) ops.Free(p);
  }
  template<class T = float>
  T* Take(std::size_t count) {
    void* p = ops.Allocate(std::max<std::size_t>(count, 1) * sizeof(T));
    owned.push_back(p);
    return static_cast<T*>(p);
  }

  Ops& ops;
  std::vector<void*> owned;
  // Activations, one set per executor (never shared between streams).
  float *x0{}, *h{}, *a{}, *vin{}, *q{}, *k{}, *v{}, *o{}, *gate{}, *branch{},
      *f{}, *g{}, *u{}, *hg{}, *hn{}, *logits{};
  std::int32_t* ids{};
  float* cap{};  ///< capture slots [tap][max_rows][hidden]; null when capture is off
  std::vector<void*> kc, vc;  ///< per layer K and V caches
  // Tap-only scratch, allocated on the first tapped Forward.
  float *tq{}, *tkv{}, *te{}, *ones{};
};

OpsExecutor::OpsExecutor(std::unique_ptr<Ops> ops, const ModelHandles& model,
                         std::shared_ptr<const void> keep_alive,
                         const ExecutorConfig& config,
                         std::uint32_t batch_invariant_rows,
                         const ExecFaults& faults)
    : Executor(MakeInfo(model, config, batch_invariant_rows), faults.ledger),
      ops_(std::move(ops)),
      model_(model),
      keep_alive_(std::move(keep_alive)),
      faults_(faults),
      exact_(config.exact),
      buf_(std::make_unique<Buffers>(*ops_)) {
  if (const char* env = std::getenv("RICURSIVE_PROFILE"); env != nullptr && std::strcmp(env, "1") == 0) {
    prof_ = std::make_unique<profile::StageProfile>();
  }
  const ExecutorInfo& info = Info();
  const Config& c = model_.config;
  if (faults_.ring_shortfall >= info.ring_rows) {
    throw std::invalid_argument("ricursive executor: ring_shortfall too large");
  }
  const std::size_t M = info.max_rows, E = c.embedding_length,
                    Q = c.QueryWidth(), KV = c.KvWidth(),
                    FF = c.feed_forward_length, L = info.max_logit_rows;
  Buffers& b = *buf_;
  b.x0 = b.Take(M * E);
  b.h = b.Take(M * E);
  b.a = b.Take(M * E);
  b.vin = b.Take(M * E);
  b.q = b.Take(M * Q);
  b.k = b.Take(M * KV);
  b.v = b.Take(M * KV);
  b.o = b.Take(M * Q);
  b.gate = b.Take(M * Q);
  b.branch = b.Take(M * E);
  b.f = b.Take(M * E);
  b.g = b.Take(M * FF);
  b.u = b.Take(M * FF);
  b.hg = b.Take(L * E);
  b.hn = b.Take(L * E);
  b.logits = b.Take(L * info.vocab);
  b.ids = b.Take<std::int32_t>(L);
  if (!info.capture_layers.empty()) {
    b.cap = b.Take(info.capture_layers.size() * M * E);
    capture_slot_.assign(info.layers, -1);
    for (std::size_t k = 0; k < info.capture_layers.size(); ++k) {
      capture_slot_[info.capture_layers[k]] = static_cast<std::int8_t>(k);
    }
  }
  const std::size_t row_bytes = info.RowBytes();
  b.kc.resize(info.layers);
  b.vc.resize(info.layers);
  for (std::uint32_t l = 0; l < info.layers; ++l) {
    const std::size_t bytes = CacheRows(l) * row_bytes;
    b.kc[l] = b.Take<std::byte>(bytes);
    b.vc[l] = b.Take<std::byte>(bytes);
  }
  // The widest activation row any projection reads (Q8_1 staging above 16 rows).
  ops_->ReserveProjection(
      M, std::max<std::size_t>({E, Q, FF}));
}

OpsExecutor::~OpsExecutor() {
  if (prof_ && prof_->TotalPasses() > profile_reported_) {
    try {
      const std::string report = prof_->Report();
      std::fputs(report.c_str(), stderr);
    } catch (...) {  // a report that cannot be printed must not throw from a destructor
    }
  }
}

void OpsExecutor::LapSlow(profile::StageId id, std::size_t bytes) {
  ops_->Synchronize();  // the same wait the executor uses before reading results back
  prof_->Mark(id, bytes);
}

std::string OpsExecutor::ProfileReport() {
  if (!prof_) return {};
  profile_reported_ = prof_->TotalPasses();
  return prof_->Report();
}

std::size_t OpsExecutor::CacheRows(std::uint32_t layer) const noexcept {
  return Info().sliding[layer] ? Info().ring_rows : Info().max_context;
}

std::uint32_t OpsExecutor::Slot(std::uint32_t layer,
                                std::uint32_t position) const noexcept {
  return Info().sliding[layer] ? position % Info().ring_rows : position;
}

void* OpsExecutor::CacheBase(std::uint32_t layer, KvPart part) const noexcept {
  return part == KvPart::kKey ? buf_->kc[layer] : buf_->vc[layer];
}

void OpsExecutor::EnsureTapScratch() {
  Buffers& b = *buf_;
  if (b.tq != nullptr) return;
  const Config& c = model_.config;
  const std::size_t M = Info().max_rows, E = c.embedding_length,
                    Q = c.QueryWidth(), KV = c.KvWidth();
  // Allocate everything before publishing any of it.
  float* tq = b.Take(M * Q);
  float* tkv = b.Take(M * KV);
  float* te = b.Take(M * E);
  float* ones = b.Take(M * Q);
  const std::vector<float> one(M * Q, 1.0F);
  ops_->CopyIn(ones, one.data(), one.size() * sizeof(float));
  b.tkv = tkv;
  b.te = te;
  b.ones = ones;
  b.tq = tq;
}

void OpsExecutor::Emit(const Tap& tap, std::string_view name, int layer,
                       const float* data, std::size_t rows, std::size_t width) {
  if (!tap) return;
  std::vector<float> host(rows * width);
  ops_->CopyOut(host.data(), data, host.size() * sizeof(float));
  tap(name, layer, host.data(), rows, width);
}

// ---- one layer -------------------------------------------------------------

void OpsExecutor::Layer(std::uint32_t il, std::uint32_t n, std::uint32_t p0,
                        const Tap& tap) {
  const ExecutorInfo& info = Info();
  const Config& c = model_.config;
  const LayerHandles& L = model_.layers[il];
  Buffers& b = *buf_;
  Ops& ops = *ops_;
  const std::uint32_t E = c.embedding_length, Q = c.QueryWidth(),
                      KV = c.KvWidth(), FF = c.feed_forward_length,
                      H = c.head_count, KVH = c.head_count_kv,
                      D = c.head_dim;
  const float eps = c.rms_eps;
  const int il_tap = static_cast<int>(il);
  const bool sliding = info.sliding[il];
  const bool tapped = static_cast<bool>(tap);
  using profile::LayerStage;

  // attn_norm; Q, K and the gate project from the normed input.
  ops.RmsNorm(b.h, L.attn_norm, b.a, n, E, eps);
  Lap(il, LayerStage::kAttnNorm);
  Emit(tap, "attn_norm", il_tap, b.a, n, E);
  // Value residual: V projects from x0 + sigmoid(lambda) * (a - x0). The mix
  // goes to its own buffer (x0 is kept for the whole pass).
  if (!faults_.stale_scratch || il == 0) {
    ops.ValueMix(b.x0, b.a, L.attn_vres_lambda, b.vin, n, E);
  }
  Lap(il, LayerStage::kValueMix);
  Emit(tap, "vres_in", il_tap, b.vin, n, E);
  const float* qk_in = faults_.value_mix_into_qk ? b.vin : b.a;
  const float* v_in = faults_.value_mix_into_qk ? b.a : b.vin;
  ops.Project(L.attn_q, qk_in, static_cast<int>(n), b.q, exact_);
  Lap(il, LayerStage::kProjQ, L.attn_q.bytes);
  ops.Project(L.attn_k, qk_in, static_cast<int>(n), b.k, exact_);
  Lap(il, LayerStage::kProjK, L.attn_k.bytes);
  ops.Project(L.attn_gate, faults_.gate_from_residual ? b.h : b.a,
              static_cast<int>(n), b.gate, exact_);
  Lap(il, LayerStage::kProjGate, L.attn_gate.bytes);
  ops.Project(L.attn_v, v_in, static_cast<int>(n), b.v, exact_);
  Lap(il, LayerStage::kProjV, L.attn_v.bytes);
  Emit(tap, "Vcur", il_tap, b.v, n, KV);

  if (tapped) {
    // The fused kernel gives no access to the norm-only values; recompute them
    // on copies with the plain RmsNorm (tap mode only).
    EnsureTapScratch();
    ops.CopyInternal(b.tq, b.q, std::size_t{n} * Q * sizeof(float));
    ops.RmsNorm(b.tq, L.attn_q_norm, b.tq, n * H, D, eps);
    Emit(tap, "Qcur_normed", il_tap, b.tq, n, Q);
    ops.CopyInternal(b.tkv, b.k, std::size_t{n} * KV * sizeof(float));
    ops.RmsNorm(b.tkv, L.attn_k_norm, b.tkv, n * KVH, D, eps);
    Emit(tap, "Kcur_normed", il_tap, b.tkv, n, KV);
  }

  // Norm, RoPE (sliding layers only) and append; then attention over the ring
  // (sliding) or all positions (full). All rows are appended before any row
  // attends, hence capacity >= window + n - 1.
  const std::uint32_t capacity =
      sliding ? info.ring_rows - faults_.ring_shortfall : info.max_context;
  const std::uint32_t rotary = (c.HasRope(il) || faults_.rope_on_full)
                                   ? static_cast<std::uint32_t>(rocm::kHeadDim)
                                   : 0U;
  const std::uint32_t append_at = faults_.chunk_local_rope ? 0U : p0;
  ops.QkNormRopeAppend(b.q, b.k, b.v, L.attn_q_norm, L.attn_k_norm, b.kc[il],
                       b.vc[il], info.kv_type, n, H, KVH, eps, append_at,
                       capacity, rotary, c.rope_freq_base);
  Lap(il, LayerStage::kQkNormRopeKv);
  Emit(tap, "Qcur", il_tap, b.q, n, Q);
  if (tapped) {
    // K as stored (after norm and RoPE), read back from the cache.
    const std::size_t rb = info.RowBytes();
    std::vector<std::byte> bytes(std::size_t{n} * rb);
    for (std::uint32_t t = 0; t < n; ++t) {
      ops.CopyOut(bytes.data() + std::size_t{t} * rb,
                  static_cast<const std::byte*>(b.kc[il]) +
                      std::size_t{append_at + t} % capacity * rb,
                  rb);
    }
    std::vector<float> k(std::size_t{n} * KV);
    for (std::size_t i = 0; i < k.size(); ++i) {
      if (info.kv_type == KvType::kF32) {
        std::memcpy(&k[i], bytes.data() + i * 4, 4);
      } else {
        std::uint16_t half;
        std::memcpy(&half, bytes.data() + i * 2, 2);
        k[i] = HalfToFloat(half);
      }
    }
    tap("Kcur", il_tap, k.data(), n, KV);
  }
  const std::uint32_t window =
      sliding ? c.sliding_window - faults_.window_shortfall : 0U;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
  ops.Attention(b.q, b.kc[il], b.vc[il], info.kv_type, b.o, n, H, KVH, window,
                p0, capacity, scale);
  Lap(il, LayerStage::kAttention);
  Emit(tap, "attn_out", il_tap, b.o, n, Q);

  // out = attn_output(o * sigmoid(gate)).
  if (tapped) {
    ops.SigmoidGate(b.ones, b.gate, b.tq, n, Q);  // sigmoid(gate) alone
    Emit(tap, "attn_gate", il_tap, b.tq, n, Q);
  }
  ops.SigmoidGate(b.o, b.gate, b.o, n, Q);
  Lap(il, LayerStage::kSigmoidGate);
  Emit(tap, "attn_gated", il_tap, b.o, n, Q);
  ops.Project(L.attn_output, b.o, static_cast<int>(n), b.branch, exact_);
  Lap(il, LayerStage::kProjAttnOut, L.attn_output.bytes);
  const float* post_attn =
      faults_.swap_post_norms ? L.post_ffw_norm : L.post_attention_norm;
  const float* post_ffn =
      faults_.swap_post_norms ? L.post_attention_norm : L.post_ffw_norm;
  if (tapped) {
    ops.Memset(b.te, 0, std::size_t{n} * E * sizeof(float));
    ops.ResidualPostNorm(b.te, b.branch, post_attn, n, E, eps);  // 0 + normed
    Emit(tap, "attn_post_norm", il_tap, b.te, n, E);
  }
  ops.ResidualPostNorm(b.h, b.branch, post_attn, n, E, eps);
  Lap(il, LayerStage::kPostAttnResidual);
  Emit(tap, "ffn_inp", il_tap, b.h, n, E);

  ops.RmsNorm(b.h, L.ffn_norm, b.f, n, E, eps);
  Lap(il, LayerStage::kFfnNorm);
  Emit(tap, "ffn_norm", il_tap, b.f, n, E);
  ops.Project(L.ffn_gate, b.f, static_cast<int>(n), b.g, exact_);
  Lap(il, LayerStage::kProjFfnGate, L.ffn_gate.bytes);
  ops.Project(L.ffn_up, b.f, static_cast<int>(n), b.u, exact_);
  Lap(il, LayerStage::kProjFfnUp, L.ffn_up.bytes);
  ops.ClampedSwiGlu(b.g, b.u, b.g, n, FF, c.swiglu_clamp[il]);
  Lap(il, LayerStage::kSwiGlu);
  Emit(tap, "ffn_swiglu", il_tap, b.g, n, FF);
  ops.Project(L.ffn_down, b.g, static_cast<int>(n), b.branch, exact_);
  Lap(il, LayerStage::kProjFfnDown, L.ffn_down.bytes);
  Emit(tap, "ffn_out", il_tap, b.branch, n, E);
  if (tapped) {
    ops.Memset(b.te, 0, std::size_t{n} * E * sizeof(float));
    ops.ResidualPostNorm(b.te, b.branch, post_ffn, n, E, eps);
    Emit(tap, "ffn_post_norm", il_tap, b.te, n, E);
  }
  ops.ResidualPostNorm(b.h, b.branch, post_ffn, n, E, eps);
  Lap(il, LayerStage::kPostFfnResidual);
  Emit(tap, "l_out", il_tap, b.h, n, E);
  if (capture_active_ && capture_slot_[il] >= 0) {
    // Capture: one device-side copy of the n rows into this layer's slot.
    ops.CopyInternal(b.cap + std::size_t{static_cast<std::uint8_t>(capture_slot_[il])} *
                                 info.max_rows * E,
                     b.h, std::size_t{n} * E * sizeof(float));
  }
}

// ---- one forward -----------------------------------------------------------

void OpsExecutor::Run(std::span<const std::int32_t> tokens, std::uint32_t p0,
                      LogitRows which, std::span<float> logits,
                      const Tap& tap) {
  const ExecutorInfo& info = Info();
  const Config& c = model_.config;
  Buffers& b = *buf_;
  Ops& ops = *ops_;
  const auto n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t E = c.embedding_length;
  const std::size_t row_bytes = std::size_t{n} * E * sizeof(float);
  const std::span<float> capture = Capture();
  capture_active_ = !capture.empty();

  // x0 = embed * embedding_scale, kept for every layer's value mix; h starts
  // as the same row and becomes the residual stream.
  if (prof_) prof_->BeginPass(n, p0 + n);
  ops.EmbedRows(model_.table, tokens, 1.0F, b.h);
  Lap(profile::PassStage::kEmbed);
  if (faults_.x0_before_scale) ops.CopyInternal(b.x0, b.h, row_bytes);
  ops.ScaleRows(b.h, b.h, n, E, c.embedding_scale);
  if (!faults_.x0_before_scale) ops.CopyInternal(b.x0, b.h, row_bytes);
  Lap(profile::PassStage::kEmbedScale);
  Emit(tap, "embd_scaled", -1, b.h, n, E);

  for (std::uint32_t il = 0; il < info.layers; ++il) Layer(il, n, p0, tap);

  if (which != LogitRows::kNone) {
    const std::uint32_t rows = which == LogitRows::kLast ? 1U : n;
    const float* src = b.h;
    if (which == LogitRows::kLast) {
      const std::int32_t row = faults_.wrong_logit_row && n > 1
                                   ? static_cast<std::int32_t>(n) - 2
                                   : static_cast<std::int32_t>(n) - 1;
      ops.CopyIn(b.ids, &row, sizeof(row));
      ops.GatherRows(b.h, b.ids, b.hg, 1, n, E);
      src = b.hg;
      Lap(profile::PassStage::kRowGather);
    }
    ops.RmsNorm(src, model_.output_norm, b.hn, rows, E, c.rms_eps);
    Lap(profile::PassStage::kFinalNorm);
    Emit(tap, "result_norm", -1, b.hn, rows, E);
    ops.Project(model_.output, b.hn, static_cast<int>(rows), b.logits, exact_);
    Lap(profile::PassStage::kOutputProj, model_.output.bytes);
    Emit(tap, "result_output", -1, b.logits, rows, info.vocab);
    if (logits.size() != std::size_t{rows} * info.vocab) {
      throw std::logic_error("ricursive executor: logit buffer size mismatch");
    }
    ops.CopyOut(logits.data(), b.logits, logits.size() * sizeof(float));
    Lap(profile::PassStage::kLogitsReadback);
  }
  if (capture_active_) {
    // Read the slots back (CopyOut waits for everything enqueued before it).
    for (std::size_t k = 0; k < info.capture_layers.size(); ++k) {
      ops.CopyOut(capture.data() + k * n * E, b.cap + k * info.max_rows * E, row_bytes);
    }
    capture_active_ = false;
  }
  ops.Synchronize();
  if (prof_) prof_->EndPass();
}

void OpsExecutor::DoForward(std::span<const std::int32_t> tokens,
                            std::uint32_t first_position, LogitRows which,
                            std::span<float> logits, const Tap& tap) {
  try {
    Run(tokens, first_position, which, logits, tap);
  } catch (...) {
    // Finish what was enqueued so nothing runs against buffers the caller
    // may reset or free next; the Executor base latches kFailed.
    try {
      ops_->Synchronize();
    } catch (...) {
    }
    throw;
  }
}

// ---- row export / import ---------------------------------------------------

void OpsExecutor::DoExport(std::uint32_t layer, KvPart part, RowRange rows,
                           std::span<std::byte> out) const {
  const std::size_t rb = Info().RowBytes();
  auto* base = static_cast<const std::byte*>(CacheBase(layer, part));
  for (std::uint32_t p = rows.first; p < rows.end;) {
    const std::uint32_t slot = Slot(layer, p);
    std::uint32_t run = 1;
    while (p + run < rows.end && Slot(layer, p + run) == slot + run) ++run;
    ops_->CopyOut(out.data() + std::size_t{p - rows.first} * rb,
                  base + std::size_t{slot} * rb, std::size_t{run} * rb);
    p += run;
  }
}

void OpsExecutor::DoImport(std::uint32_t layer, KvPart part, RowRange rows,
                           std::span<const std::byte> bytes) {
  const std::size_t rb = Info().RowBytes();
  auto* base = static_cast<std::byte*>(CacheBase(layer, part));
  for (std::uint32_t p = rows.first; p < rows.end;) {
    const std::uint32_t slot = Slot(layer, p);
    std::uint32_t run = 1;
    while (p + run < rows.end && Slot(layer, p + run) == slot + run) ++run;
    ops_->CopyIn(base + std::size_t{slot} * rb,
                 bytes.data() + std::size_t{p - rows.first} * rb,
                 std::size_t{run} * rb);
    p += run;
  }
  ops_->Synchronize();
}

}  // namespace gufo::models::ricursive

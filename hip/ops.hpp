#ifndef RICURSIVE_HIP_OPS_HPP_
#define RICURSIVE_HIP_OPS_HPP_

// The operations the executor's orchestration (hip/executor.cpp) is written
// against, with two implementations: ops_gpu.cpp (device memory, the real
// launch functions of hip/kernels.hpp, hip/project.hpp, hip/embed.hpp) and
// ops_host.cpp (host memory, scalar loops with the same argument conventions).
// This header needs no HIP header, so the orchestration compiles and runs on
// the CPU. Docs: docs/S2-EXECUTOR.md.
//
// Conventions (the kernels' own, see hip/kernels.hpp):
//  - every pointer is a pointer into the backend's memory (Allocate) unless a
//    parameter says "host"; the orchestration does the offset arithmetic, so
//    buffer sizes and offsets are exercised by whichever backend runs;
//  - float buffers are row-major fp32; "in place" means out == in is allowed;
//  - operations are enqueued in order and may complete later; CopyOut,
//    EmbedRows and Synchronize return after everything before them finished;
//  - a failure (bad argument, device error) is a std::runtime_error carrying
//    the operation's name; nothing is returned as a status.
//
// Memory is never freed while enqueued work may still use it: Free is only
// called after Synchronize on the failure and destruction paths.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <span>
#include <vector>

#include "hip/dequant_math.hpp"
#include "hip/embed.hpp"
#include "hip/kernels_math.hpp"
#include "src/config.hpp"
#include "src/executor_iface.hpp"

namespace gufo::models::ricursive {

/// A weight matrix in backend memory, GGUF layout: `rows` outputs of `cols`
/// contiguous inputs. Does not own `data`. On the GPU the allocation is
/// followed by kWeightTailBytes readable bytes (hip/project.hpp).
struct WeightView {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t rows{0};
  std::uint32_t cols{0};
  std::size_t bytes{0};
};

struct LayerHandles {
  WeightView attn_q, attn_k, attn_v, attn_output, attn_gate;
  WeightView ffn_gate, ffn_up, ffn_down;
  /// F32 vectors in backend memory: [E] norms, [128] q/k norms, [1] lambda.
  const float *attn_norm{nullptr}, *attn_q_norm{nullptr}, *attn_k_norm{nullptr},
      *post_attention_norm{nullptr}, *ffn_norm{nullptr}, *post_ffw_norm{nullptr},
      *attn_vres_lambda{nullptr};
};

/// Everything the orchestration reads of the model; owned by a DeviceModel
/// (GPU) or a HostModel (host ops), which outlive every executor using it.
struct ModelHandles {
  Config config;
  EmbeddingTable table;  ///< host mapping (no GPU path for IQ4_XS)
  WeightView output;
  const float* output_norm{nullptr};
  std::vector<LayerHandles> layers;
};

// ---- binary16 helpers shared by the orchestration and the host ops ---------

/// IEEE binary16 -> float (exact).
[[nodiscard]] inline float HalfToFloat(std::uint16_t h) noexcept {
  const std::uint32_t sign = (h & 0x8000U) << 16, exp = (h >> 10) & 0x1FU,
                      man = h & 0x3FFU;
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {  // subnormal half: renormalize
      int e = -1;
      std::uint32_t m = man;
      do {
        ++e;
        m <<= 1;
      } while ((m & 0x400U) == 0);
      bits = sign | ((127 - 15 - static_cast<std::uint32_t>(e)) << 23) |
             ((m & 0x3FFU) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000U | (man << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

/// float -> IEEE binary16, round to nearest even (as __float2half).
[[nodiscard]] inline std::uint16_t FloatToHalf(float value) noexcept {
  std::uint32_t x;
  std::memcpy(&x, &value, 4);
  const std::uint32_t sign = (x >> 16) & 0x8000U;
  const std::uint32_t abs = x & 0x7FFFFFFFU;
  if (abs > 0x7F800000U) {  // NaN: keep it a NaN
    return static_cast<std::uint16_t>(sign | 0x7E00U | ((abs >> 13) & 0x1FFU));
  }
  if (abs >= 0x477FF000U) {  // rounds to >= 65520: infinity
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (abs < 0x38800000U) {  // below the smallest normal half (2^-14)
    if (abs < 0x33000000U) return static_cast<std::uint16_t>(sign);  // < 2^-25
    const int shift = 126 - static_cast<int>(abs >> 23);  // 14..24
    const std::uint32_t mant = (abs & 0x7FFFFFU) | 0x800000U;
    std::uint32_t half = mant >> shift;
    const std::uint32_t rest = mant & ((1U << shift) - 1U);
    const std::uint32_t mid = 1U << (shift - 1);
    if (rest > mid || (rest == mid && (half & 1U) != 0)) ++half;
    return static_cast<std::uint16_t>(sign | half);
  }
  std::uint32_t half = ((abs >> 13) - (112U << 10));  // rebias, drop 13 bits
  const std::uint32_t rest = abs & 0x1FFFU;
  if (rest > 0x1000U || (rest == 0x1000U && (half & 1U) != 0)) ++half;
  return static_cast<std::uint16_t>(sign | half);
}

class Ops {
 public:
  virtual ~Ops() = default;
  Ops() = default;
  Ops(const Ops&) = delete;
  Ops& operator=(const Ops&) = delete;

  [[nodiscard]] virtual const char* Name() const noexcept = 0;

  // ---- memory -----------------------------------------------------------
  /// Uninitialized memory (the host backend poisons it with NaN patterns and
  /// guards both ends). Throws on failure. Never returns nullptr.
  [[nodiscard]] virtual void* Allocate(std::size_t bytes) = 0;
  /// Frees memory from Allocate; nullptr is ignored. Call only after
  /// Synchronize.
  virtual void Free(void* ptr) noexcept = 0;
  virtual void Memset(void* dst, int byte, std::size_t bytes) = 0;
  /// Host -> backend. The host source may be reused as soon as this returns.
  virtual void CopyIn(void* dst, const void* host_src, std::size_t bytes) = 0;
  /// Backend -> host; returns after all earlier work finished.
  virtual void CopyOut(void* host_dst, const void* src, std::size_t bytes) = 0;
  /// Backend -> backend, no overlap.
  virtual void CopyInternal(void* dst, const void* src, std::size_t bytes) = 0;
  /// Waits for everything enqueued; throws on a deferred error.
  virtual void Synchronize() = 0;
  /// False when the device context reports a sticky error (never throws).
  [[nodiscard]] virtual bool Usable() noexcept = 0;
  /// Sizes the activation-quantization scratch of Project for up to `rows`
  /// rows of `cols` inputs, so Project never allocates inside a pass.
  virtual void ReserveProjection(std::size_t rows, std::size_t cols) = 0;

  // ---- operators --------------------------------------------------------
  /// out[i][:] = dequantized table row of tokens[i], times `scale` (one fp32
  /// multiply). Reads the mapped table on the host, so it finishes earlier
  /// work first.
  virtual void EmbedRows(const EmbeddingTable& table,
                         std::span<const std::int32_t> tokens, float scale,
                         float* out) = 0;
  virtual void ScaleRows(const float* in, float* out, std::uint32_t rows,
                         std::uint32_t dim, float scale) = 0;
  virtual void RmsNorm(const float* x, const float* weight, float* out,
                       std::uint32_t rows, std::uint32_t dim, float eps) = 0;
  virtual void ResidualPostNorm(float* x, const float* branch,
                                const float* weight, std::uint32_t rows,
                                std::uint32_t dim, float eps) = 0;
  virtual void ValueMix(const float* x0, const float* a, const float* lambda,
                        float* out, std::uint32_t rows, std::uint32_t dim) = 0;
  /// y[rows][w.rows] = x[rows][w.cols] * W^T. `exact`: FP32 activations at
  /// every row count (no activation rounding).
  virtual void Project(const WeightView& w, const float* x, int rows, float* y,
                       bool exact) = 0;
  /// Bytes of device memory the operations hold for the projection scratch (the Q8_1 activations of the 8-bit routes,
  /// sized by ReserveProjection): what is allocated on the GPU, the same formula on the host (rocm::Q8ActivationBytes).
  [[nodiscard]] virtual std::uint64_t ProjectionScratchBytes() const noexcept = 0;
  virtual void QkNormRopeAppend(float* q, const float* k, const float* v,
                                const float* q_gamma, const float* k_gamma,
                                void* k_cache, void* v_cache, KvType type,
                                std::uint32_t tokens, std::uint32_t heads,
                                std::uint32_t kv_heads, float eps,
                                std::uint32_t start_pos,
                                std::uint32_t capacity,
                                std::uint32_t rotary_dim, float theta) = 0;
  virtual void Attention(const float* q, const void* k_cache,
                         const void* v_cache, KvType type, float* out,
                         std::uint32_t tokens, std::uint32_t heads,
                         std::uint32_t kv_heads, std::uint32_t window,
                         std::uint32_t start_pos, std::uint32_t capacity,
                         float scale) = 0;
  virtual void SigmoidGate(const float* attn, const float* gate, float* out,
                           std::uint32_t rows, std::uint32_t dim) = 0;
  virtual void ClampedSwiGlu(const float* gate, const float* up, float* out,
                             std::uint32_t rows, std::uint32_t dim,
                             float clamp) = 0;
  /// dst[i][:] = src[ids[i]][:]; `ids` is backend memory.
  virtual void GatherRows(const float* src, const std::int32_t* ids,
                          float* dst, std::uint32_t n_ids,
                          std::uint32_t src_rows, std::uint32_t dim) = 0;

  // ---- the drafter's operations (docs/DRAFTER-S2.md) --------------------
  /// A projection on the 8-bit activation route at ANY row count (docs/DRAFTER-S2.md, "Stage 2b"): the activations are
  /// quantized to Q8_1 blocks and multiplied by the weights on the W8A8 GEMM, which `Project` takes only above 16 rows
  /// (at or below it the FP32 small-batch kernels, about 3 times a one-row pass at 16 rows). Rows are NOT bit-equal to
  /// `Project`'s: they differ by the activation rounding, which the drafter tolerates (it only changes how many drafts the
  /// verifier accepts). F32 and BF16 weights have no 8-bit route and take `Project`'s. Needs the scratch of
  /// ReserveProjection(rocm::Q8ReserveRows(rows), w.cols) reserved beforehand (the Q8_0 kernel reads whole token macro tiles) (it never allocates inside a pass). The host operations have no
  /// model of the activation rounding: they compute `Project`'s result and count the call (HostOpsOptions::counters).
  /// Only the drafter calls it; `Project` and its callers are unchanged.
  virtual void ProjectQ8(const WeightView& w, const float* x, int rows, float* y) = 0;
  // Additions for the DFlash2 drafter (hip/drafter_executor.cpp); the target's
  // executor never calls them. Their GPU kernels are hip/drafter_kernels.hip.cpp,
  // adapted from gufo's Qwen DFlash kernels; the host versions below use the same
  // loop bounds, indexing and parameter names. Any head_dim up to 256 (the target's
  // operators are written for 128 only).

  /// Per-head RMSNorm of q [tokens][heads][head_dim] (in place) and k
  /// [tokens][kv_heads][head_dim] (k is not modified), optional NeoX RoPE on both at
  /// positions start_pos + t (`rope`: float32 frequencies theta^(-2i/head_dim), as the
  /// drafter reference computes them), then the rows of k (normed, rotated) and v (raw)
  /// are stored into k_store / v_store: [capacity][kv_heads * head_dim] elements of
  /// `type`, row t into slot (slot_start + t) % capacity. heads == 0: no query rows
  /// (q may be null). tokens <= capacity.
  virtual void DraftQkNormRope(float* q, const float* k, const float* v,
                               const float* q_gamma, const float* k_gamma,
                               void* k_store, void* v_store, KvType type,
                               std::uint32_t tokens, std::uint32_t heads,
                               std::uint32_t kv_heads, std::uint32_t head_dim,
                               float eps, std::uint32_t start_pos,
                               std::uint32_t slot_start,
                               std::uint32_t capacity, bool rope,
                               float theta) = 0;
  /// Non-causal attention of `tokens` block rows (positions anchor + t) over the context
  /// rows [ctx_first, anchor) held in the ring k_ctx / v_ctx (position j in slot
  /// j % ctx_capacity, elements of `type`) followed by the block's own rows k_blk / v_blk
  /// ([tokens][kv_heads * head_dim] fp32, row i at position anchor + i). `window` == 0:
  /// every key; else key j is visible to the query at p iff |p - j| < window. Query head
  /// h reads KV head h / (heads / kv_heads). out [tokens][heads * head_dim], no overlap
  /// with q. Softmax and the output sum in fp32 (GPU) / double (host).
  virtual void DraftAttention(const float* q, const void* k_ctx,
                              const void* v_ctx, KvType type,
                              std::uint32_t ctx_capacity,
                              std::uint32_t ctx_first, std::uint32_t anchor,
                              const float* k_blk, const float* v_blk,
                              float* out, std::uint32_t tokens,
                              std::uint32_t heads, std::uint32_t kv_heads,
                              std::uint32_t head_dim, std::uint32_t window,
                              float scale) = 0;
  /// The grouped dynamic causal convolution over the rows of one block:
  /// out[t][c] (+)= sum over taps o < kernel_size, t >= o, of
  /// (base[side*kernel_size + o][c] + dyn[t][side*kernel_size*G + o*G + c / group_size]) *
  /// in[t - o][c], G = hidden_size / group_size, dyn rows 2 * kernel_size * G wide. Rows
  /// before the block start are zero, never read. `accumulate`: output += sum, not = sum.
  /// output may alias neither input nor dynamic_coefficients unless accumulate.
  virtual void DynamicConv(const float* input, const float* dynamic_coefficients,
                           const float* base_kernel, float* output,
                           std::uint32_t num_tokens, std::uint32_t hidden_size,
                           std::uint32_t kernel_size, std::uint32_t group_size,
                           std::uint32_t side, bool accumulate) = 0;
  /// gate[i] = silu(gate[i]) * up[i], no clamp.
  virtual void SiluMul(float* gate, const float* up,
                       std::uint32_t total_elements) = 0;
  /// The candidate selector (top-k up to 32, the walk left to right). logits
  /// [rows][vocab_size], projected_hidden [rows][selector_rank]. tokens: [rows + 1] int32;
  /// tokens[0] is the anchor token (the first predecessor), written by the caller;
  /// tokens[1 + r] receives row r's choice. For row r the candidates are the selector_top_k
  /// largest logits (candidate_ids [rows][top_k], by descending logit, equal logits by
  /// ascending id), score_k = logit_k + sum_i (pred[prev][i] * hidden[r][i]) * succ[id_k][i]
  /// (candidate_scores [rows][top_k]), the choice the largest score (equal scores: the lowest token id, rocm::DraftRanksAbove),
  /// prev = the previous row's choice. The codebooks are [vocab][selector_rank] matrices of
  /// F32, BF16, Q8_0 or Q4_K. partial_scores / partial_ids: scratch of
  /// rows * DraftSelectPartials(vocab_size) * selector_top_k elements.
  virtual void DraftSelect(const float* logits, const float* projected_hidden,
                           const WeightView& predecessor_codebook,
                           const WeightView& successor_codebook,
                           std::int32_t* tokens, float* candidate_scores,
                           std::int32_t* candidate_ids, float* partial_scores,
                           std::int32_t* partial_ids, std::uint32_t rows,
                           std::uint32_t vocab_size,
                           std::uint32_t selector_rank,
                           std::uint32_t selector_top_k) = 0;
};

/// Calls the host operations received (docs/DRAFTER-S2.md, "Stage 2b": the route selection is tested by counting).
struct HostOpsCounters {
  std::uint64_t project{0};      ///< Project calls
  std::uint64_t project_q8{0};   ///< ProjectQ8 calls
  std::uint64_t q8_rows{0};      ///< rows of the ProjectQ8 calls
  std::vector<const void*> q8_weights;  ///< the weight data pointer of every ProjectQ8 call, in order
  // What crosses between host and device and where the host waits (docs/DRAFTER-S2.md, "Stage 2b", item 3). On the GPU
  // every one of these returns after the device finished all earlier work (EmbedRows, CopyIn and CopyOut synchronize
  // the stream; Synchronize is the explicit wait): a call here is one synchronization point there.
  std::uint64_t embed_calls{0}, embed_rows{0};
  std::uint64_t copy_in_calls{0}, copy_in_bytes{0};
  std::uint64_t copy_out_calls{0}, copy_out_bytes{0};
  std::uint64_t copy_internal_calls{0}, copy_internal_bytes{0};
  std::uint64_t synchronize_calls{0};
  /// Calls that wait for the device.
  [[nodiscard]] std::uint64_t SyncPoints() const noexcept { return embed_calls + copy_in_calls + copy_out_calls + synchronize_calls; }
};

/// Test-only deliberate defects of the host operations (docs/DRAFTER-S2.md, the twins).
struct HostOpsOptions {
  bool lenient_capacity{false};   ///< see HostOpsExecutor::Options
  bool poison_memory{true};       ///< see HostOpsExecutor::Options
  /// DynamicConv reads the rows of the block's end for taps that reach before its start
  /// (a cyclic buffer instead of zero padding).
  bool conv_across_block_start{false};
  /// Counts the projection calls here (not owned; must outlive the operations).
  HostOpsCounters* counters{nullptr};
};
/// A host operations object (ops_host.cpp) / a GPU one on its own stream (ops_gpu.cpp, HIP builds).
[[nodiscard]] std::unique_ptr<Ops> CreateHostOps(const HostOpsOptions& options = {});
[[nodiscard]] std::unique_ptr<Ops> CreateGpuOps();

/// The DequantAt encoding of a matrix type (Enc::kNone: not usable as a selector codebook).
[[nodiscard]] inline rocm::Enc EncodingOf(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kF32: return rocm::Enc::kF32;
    case core::GgmlType::kBF16: return rocm::Enc::kBF16;
    case core::GgmlType::kQ8_0: return rocm::Enc::kQ8_0;
    case core::GgmlType::kQ4_K: return rocm::Enc::kQ4_K;
    default: return rocm::Enc::kNone;
  }
}

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_HIP_OPS_HPP_

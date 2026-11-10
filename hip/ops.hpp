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
#include <span>
#include <vector>

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
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_HIP_OPS_HPP_

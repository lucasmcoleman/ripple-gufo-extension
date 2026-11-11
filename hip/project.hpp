// Quantized and dense matrix projections for ricursive: y[rows][M] = x[rows][K] * W[M][K]^T.
// Work package A of S2 (see docs/S2-PROJECT.md). Dispatch mirrors the Gemma 4 executor's Project
// without its WMMA/grouped/half branches; the kernels are the vendored hip/ggml/* ones.
#ifndef GUFO_MODELS_RICURSIVE_HIP_PROJECT_HPP_
#define GUFO_MODELS_RICURSIVE_HIP_PROJECT_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>

namespace gufo::models::ricursive::rocm {

/// Rows beyond this take the W8A8 prefill GEMM (activations rounded to int8 blocks); at or below it
/// the activations stay FP32. Same value as the Gemma 4 extension's kSplitRows.
inline constexpr int kSplitRows = 16;

/// Every weight allocation must be followed by at least this many readable bytes (zeroed by the
/// uploader): the quantized kernels read whole 256-element iterations and may over-read the last row.
inline constexpr std::size_t kWeightTailBytes = 4096;

/// One weight matrix resident on the device, GGUF layout: `rows` output rows (GGUF ne[1]) of `cols`
/// contiguous inputs (GGUF ne[0]); row r starts at byte r * RowBytes(type, cols). The struct does not
/// own `data`.
struct DeviceTensor {
  const void* data{nullptr};  ///< device pointer; kWeightTailBytes readable bytes must follow `bytes`
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t rows{0};  ///< M: output features
  std::uint32_t cols{0};  ///< K: input features
  std::size_t bytes{0};   ///< encoded size, must equal rows * RowBytes(type, cols)
};

/// Encoded bytes of one row of `cols` elements for a type Project accepts; 0 if the type is not one of
/// F32, BF16, Q8_0, Q4_K, Q5_K, Q6_K or `cols` is not block aligned.
[[nodiscard]] std::size_t RowBytes(core::GgmlType type, std::size_t cols) noexcept;

/// True for the types Project can run. IQ4_XS (the token table) and everything else is not one of them.
[[nodiscard]] bool IsProjectable(core::GgmlType type) noexcept;

/// Device buffer for the Q8_1-quantized activations of the rows > kSplitRows route. Reserve() is the
/// only place that allocates; Project calls it when the buffer is too small, which synchronizes the
/// device, so reserve the maximum (rows, cols) up front when the call is inside a graph capture.
class Scratch {
 public:
  Scratch() = default;
  ~Scratch();
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  void Reserve(std::size_t rows, std::size_t cols);
  [[nodiscard]] void* data() const noexcept { return q8_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

 private:
  void* q8_{nullptr};
  std::size_t capacity_{0};
};

/// y[r][o] = sum_c x[r][c] * W[o][c] for r < rows, o < w.rows, c < w.cols. x is [rows][w.cols] FP32,
/// y is [rows][w.rows] FP32, both device memory; asynchronous on `stream`.
///  - Quantized weights (Q4_K, Q5_K, Q6_K, Q8_0): rows == 1 uses the GEMV, rows 2..16 the FP32 small-batch
///    kernel (bit-equal to the GEMV per row), rows > 16 quantize x to Q8_1 into `scratch` and run the
///    W8A8 WMMA GEMM (within the activation-rounding envelope, not bit-equal).
///  - exact = true forces the FP32-activation path for any row count (16-row groups of the small-batch
///    kernel; slow, for tests).
///  - F32 and BF16 weights: per-row GEMV at any row count (test/tiny-model path).
/// Throws std::runtime_error naming the type for anything else (IQ4_XS, F16, Q4_0, ...), for a size that
/// does not match rows*cols*type, a null pointer, rows < 1, or cols that are not block aligned.
void Project(const DeviceTensor& w, const float* x, int rows, float* y, Scratch& scratch,
             hipStream_t stream, bool exact = false);

/// The drafter's route (docs/DRAFTER-S2.md, "Stage 2b"): y = x * W^T with the activations quantized to Q8_1 blocks and the
/// W8A8 WMMA GEMM at ANY row count (Project takes it only above kSplitRows). Quantized weights only take the 8-bit route
/// (rows within the activation-rounding envelope of Project's W8A8 path, not bit-equal to the small-batch kernels); F32 and
/// BF16 weights run Project's per-row GEMV. `scratch` must already hold QuantizedActivationBytes(rows, w.cols)
/// (Scratch::Reserve): a reservation inside the pass would synchronize the device. Same validation and errors as Project.
/// `Project` itself is unchanged; only the drafter calls this.
void ProjectW8A8(const DeviceTensor& w, const float* x, int rows, float* y, Scratch& scratch, hipStream_t stream);

}  // namespace gufo::models::ricursive::rocm

#endif  // GUFO_MODELS_RICURSIVE_HIP_PROJECT_HPP_

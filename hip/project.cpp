#include "hip/project.hpp"

#include <algorithm>
#include <cstdint>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <stdexcept>
#include <string>

#include "hip/ggml/gemm.hpp"

namespace gufo::models::ricursive::rocm {

namespace {

using core::GgmlType;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("ricursive Project: " + message);
}

void Check(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    Fail(std::string(what) + ": " + hipGetErrorString(err));
  }
}

bool IsDense(GgmlType type) noexcept { return type == GgmlType::kF32 || type == GgmlType::kBF16; }

}  // namespace

bool IsProjectable(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
    case GgmlType::kBF16:
    case GgmlType::kQ8_0:
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:
      return true;
    default:
      return false;
  }
}

std::size_t RowBytes(GgmlType type, std::size_t cols) noexcept {
  if (!IsProjectable(type) || cols == 0) {
    return 0;
  }
  if (type == GgmlType::kF32) {
    return cols * 4;
  }
  if (type == GgmlType::kBF16) {
    return cols * 2;
  }
  return quant::QuantizedRowBytes(type, cols);  // 0 when cols is not block aligned
}

Scratch::~Scratch() {
  if (q8_ != nullptr) {
    (void)hipFree(q8_);
  }
}

void Scratch::Reserve(std::size_t rows, std::size_t cols) {
  const std::size_t need = ggml::QuantizedActivationBytes(rows, cols);
  if (need <= capacity_) {
    return;
  }
  if (q8_ != nullptr) {
    Check(hipFree(q8_), "free scratch");
    q8_ = nullptr;
    capacity_ = 0;
  }
  Check(hipMalloc(&q8_, need), "allocate Q8_1 scratch");
  capacity_ = need;
}

namespace {

void Dispatch(const DeviceTensor& w, const float* x, int rows, float* y, Scratch& scratch,
              hipStream_t stream, bool exact) {
  const std::string type_name(core::ToString(w.type));
  if (!IsProjectable(w.type)) {
    Fail("no GPU projection path for type " + type_name +
         " (supported: F32, BF16, Q8_0, Q4_K, Q5_K, Q6_K)");
  }
  if (w.data == nullptr || x == nullptr || y == nullptr) {
    Fail("null pointer (weight " + type_name + ")");
  }
  if (rows < 1 || w.rows == 0 || w.cols == 0) {
    Fail("empty projection: rows=" + std::to_string(rows) + " weight " + std::to_string(w.rows) +
         "x" + std::to_string(w.cols) + " " + type_name);
  }
  const std::size_t row_bytes = RowBytes(w.type, w.cols);
  if (row_bytes == 0) {
    Fail("cols=" + std::to_string(w.cols) + " is not block aligned for type " + type_name);
  }
  if (w.bytes != row_bytes * w.rows) {
    Fail("byte size " + std::to_string(w.bytes) + " does not match " + std::to_string(w.rows) + "x" +
         std::to_string(w.cols) + " " + type_name + " (" + std::to_string(row_bytes * w.rows) + ")");
  }

  const std::size_t n = static_cast<std::size_t>(rows);
  const std::size_t m = w.rows;
  const std::size_t k = w.cols;

  if (IsDense(w.type)) {
    // No batched dense kernel: one GEMV per row.
    for (std::size_t r = 0; r < n; ++r) {
      ggml::LaunchGEMV(w.data, w.type, x + r * k, y + r * m, m, k, stream);
    }
    return;
  }
  if (exact) {
    // FP32 activations at every row count: 16-row groups of the small-batch kernel, whose rows equal
    // the GEMV bit for bit. LaunchBatchedQuantGEMMFp32 itself accepts any batch (eight-row passes).
    for (std::size_t r = 0; r < n; r += kSplitRows) {
      const std::size_t group = std::min<std::size_t>(kSplitRows, n - r);
      ggml::LaunchBatchedQuantGEMMFp32(w.type, w.data, x + r * k, y + r * m, group, m, k, stream);
    }
    return;
  }
  if (rows > kSplitRows) {
    scratch.Reserve(n, k);
    ggml::LaunchQuantizeActivationQ8_1FromFp32(x, scratch.data(), n, k, stream);
    ggml::LaunchBatchedQuantGEMMPreQuantized(w.type, w.data, scratch.data(), y, n, m, k, stream);
    return;
  }
  if (rows > 1) {
    ggml::LaunchBatchedQuantGEMMFp32(w.type, w.data, x, y, n, m, k, stream);
    return;
  }
  // The staged kernel loads the input vector 16 bytes at a time, so it assumes a 16-byte-aligned x; anything
  // else (one pointer test) falls back to the GEMV, which has no alignment requirement. Defensive: with the
  // test removed, a misaligned x still gave identical bits on this gfx1151 (tests route_*), so the test case
  // does not prove the guard is needed, only that the fallback path is bit-equal.
  // Long-K Q4_K/Q5_K rows (ffn_down): the single-row GEMV re-reads the whole activation vector from L2
  // for every row; the small-batch kernel at batch 1 stages it in LDS and is bit-equal to the GEMV
  // (same per-lane order). Measured +22..25 % on 5120x40960; Q6_K was slower there and stays on the GEMV.
  if ((w.type == GgmlType::kQ4_K || w.type == GgmlType::kQ5_K) && k >= 16384 &&
      reinterpret_cast<std::uintptr_t>(x) % 16 == 0 &&
      ggml::LaunchKQuantSmallBatchDoubleStage(w.type, w.data, x, y, 1, m, k, stream)) {
    return;
  }
  ggml::LaunchGEMV(w.data, w.type, x, y, m, k, stream);
}

}  // namespace

void Project(const DeviceTensor& w, const float* x, int rows, float* y, Scratch& scratch,
             hipStream_t stream, bool exact) {
  (void)hipGetLastError();  // drop an unrelated earlier error so the check below is ours
  Dispatch(w, x, rows, y, scratch, stream, exact);
  Check(hipGetLastError(), "kernel launch");
}

}  // namespace gufo::models::ricursive::rocm

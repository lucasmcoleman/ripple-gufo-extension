// Projection kernels at the Ripple-65B (ricursive) shapes (work package A of S2).
//
//   projection_ops_test --list           case names (host only)
//   projection_ops_test <case>           one case on the GPU (run through the lock wrapper)
//   projection_ops_test <case> --cpu     the same harness against a CPU implementation of Project
//                                        (scaled-down shapes): proves the harness and its twins, says
//                                        nothing about the kernels
//
// Cases: <type>_<M>x<K> (the real target shapes, one weight type and shape per invocation, rows
// 1..16, 17 and 512 inside it), f32-tiny (F32/BF16 matrices of the tiny fixtures), twins (broken twins),
// reject (Project refuses IQ4_XS and other types without running).
//
// What a shape case checks (M = outputs, K = inputs, weights = random blocks with bounded finite scales,
// x = N(0,1) rows):
//  1. rows == 1 (GEMV) for 512 distinct x rows; rows 0 and 15 against an FP64 dot product of the
//     CPU-dequantized weights (gufo::quant::DequantizeRow, itself checked bit for bit against an
//     independent decoder in embed_test) on ~100 sampled outputs including the first and last 16.
//     Error is relative to sum|w x|. Bound 2e-6: FP32 accumulation of K <= 40960 products in lanes of
//     <= ~700 sequential adds has worst-case relative error depth*u (u = 6e-8, ~4e-5) and a random-walk
//     error of ~sqrt(depth)*u/sqrt(3) ~ 1e-6, so 2e-6 passes a correct kernel with margin while a
//     wrong-type or wrong-layout kernel is off by O(1), 5 orders above the bound.
//  2. one-hot probes at columns 0, K-1, K-32 and K-256 return the dequantized weight (last columns).
//  3. rows 2..16 (FP32 small batch) bit-equal to the single-row results, row by row, every output.
//  4. rows 17 and 512 with exact = true bit-equal to the single-row results.
//  5. rows 17 and 512 in normal mode (Q8_1 activations, W8A8 WMMA) against the single-row results:
//     all elements within 0.06 x RMS of the row (activation rounding noise is ~0.0055 x ||w_o|| and the
//     bound is 6 sigma with ||w_o|| up to 1.2 x the row RMS), whole-matrix RMS relative error <= 1.5e-2
//     (expected ~5.5e-3: int8 rounding of N(0,1) activations in blocks of 32), and on the sampled outputs
//     against FP64 within 6 sigma_o + 2e-6 sum|w x|, sigma_o^2 = sum_blocks d_b^2 sum_c w_c^2 / 12,
//     d_b = amax_b / 127 (the rounding-noise standard deviation, per output).
//  Every y buffer is poisoned with a NaN pattern with 4096 guard words each side; unwritten, non-finite
//  or out-of-range writes fail with the first bad row/column. Weights carry the 4 KiB zero tail the
//  uploader adds followed by 64 KiB of 0xFF (NaN scales) to catch reads beyond the margin.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "hip/project.hpp"
#include "tests/project/check.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "hip/ggml/gemm.hpp"
#endif

using gufo::core::GgmlType;
using gufo::models::ricursive::rocm::DeviceTensor;
using gufo::models::ricursive::rocm::kWeightTailBytes;
using gufo::models::ricursive::rocm::RowBytes;
using ricursive_test::Require;

namespace {

constexpr std::size_t kGuard = 4096;
constexpr std::uint32_t kPattern = 0x7FC0DEADU;
constexpr double kDotBound = 2e-6;
constexpr std::size_t kPadAfterTail = 65536;

std::string TypeName(GgmlType t) { return std::string(gufo::core::ToString(t)); }

// ---- backends ----------------------------------------------------------------------------------------

class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* Name() const = 0;
  virtual void* Alloc(std::size_t bytes) = 0;
  virtual void Free(void* p) = 0;
  virtual void Upload(void* dst, const void* src, std::size_t bytes) = 0;
  virtual void Download(void* dst, const void* src, std::size_t bytes) = 0;
  virtual void Fill(void* dst, int byte, std::size_t bytes) = 0;
  virtual void Project(const DeviceTensor& w, const float* x, int rows, float* y, bool exact) = 0;
  /// Launch the single-row kernel for `type` directly, bypassing Project's validation (twins only).
  virtual void RawGemv(GgmlType type, const void* w, const float* x, float* y, std::size_t m,
                       std::size_t k) = 0;
  virtual void Sync() = 0;
};

#if defined(ENGINE_ENABLE_HIP)
void HipOk(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("HIP ") + what + ": " + hipGetErrorString(err));
  }
}

class GpuBackend : public Backend {
 public:
  const char* Name() const override { return "gpu"; }
  void* Alloc(std::size_t bytes) override {
    void* p = nullptr;
    HipOk(hipMalloc(&p, bytes), "hipMalloc");
    return p;
  }
  void Free(void* p) override { (void)hipFree(p); }
  void Upload(void* dst, const void* src, std::size_t bytes) override {
    HipOk(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice), "upload");
  }
  void Download(void* dst, const void* src, std::size_t bytes) override {
    HipOk(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost), "download");
  }
  void Fill(void* dst, int byte, std::size_t bytes) override {
    HipOk(hipMemset(dst, byte, bytes), "memset");
  }
  void Project(const DeviceTensor& w, const float* x, int rows, float* y, bool exact) override {
    gufo::models::ricursive::rocm::Project(w, x, rows, y, scratch_, nullptr, exact);
  }
  void RawGemv(GgmlType type, const void* w, const float* x, float* y, std::size_t m,
               std::size_t k) override {
    gufo::models::ricursive::ggml::LaunchGEMV(w, type, x, y, m, k, nullptr);
  }
  void Sync() override { HipOk(hipDeviceSynchronize(), "synchronize"); }

 private:
  gufo::models::ricursive::rocm::Scratch scratch_;
};
#endif

/// Host implementation of Project's contract for the harness self-check: dequantize with the public
/// decoder, accumulate in FP64. rows > 16 in normal mode round the activations to int8 blocks of 32
/// (the W8A8 path's noise model); everything else is exact FP32 activations.
class CpuBackend : public Backend {
 public:
  const char* Name() const override { return "cpu"; }
  void* Alloc(std::size_t bytes) override { return std::calloc(1, bytes); }
  void Free(void* p) override { std::free(p); }
  void Upload(void* dst, const void* src, std::size_t bytes) override { std::memcpy(dst, src, bytes); }
  void Download(void* dst, const void* src, std::size_t bytes) override { std::memcpy(dst, src, bytes); }
  void Fill(void* dst, int byte, std::size_t bytes) override { std::memset(dst, byte, bytes); }
  void Sync() override {}
  void Project(const DeviceTensor& w, const float* x, int rows, float* y, bool exact) override {
    Require(gufo::models::ricursive::rocm::IsProjectable(w.type), "cpu backend: type " + TypeName(w.type));
    const std::size_t m = w.rows, k = w.cols;
    Require(w.bytes == RowBytes(w.type, k) * m, "cpu backend: size mismatch");
    const auto& dec = Decoded(w.data, w.type, m, k);
    const bool round = rows > 16 && !exact && w.type != GgmlType::kF32 && w.type != GgmlType::kBF16;
    std::vector<float> xr(k);
    for (int r = 0; r < rows; ++r) {
      std::memcpy(xr.data(), x + static_cast<std::size_t>(r) * k, k * 4);
      if (round) {
        for (std::size_t b = 0; b < k / 32; ++b) {
          float amax = 0;
          for (std::size_t c = 0; c < 32; ++c) amax = std::max(amax, std::fabs(xr[b * 32 + c]));
          const float d = amax / 127.0F;
          for (std::size_t c = 0; c < 32; ++c) {
            xr[b * 32 + c] = d == 0 ? 0.0F : std::lround(xr[b * 32 + c] / d) * d;
          }
        }
      }
      for (std::size_t o = 0; o < m; ++o) {
        double acc = 0;
        const float* wr = &dec[o * k];
        for (std::size_t c = 0; c < k; ++c) acc += double{wr[c]} * xr[c];
        y[static_cast<std::size_t>(r) * m + o] = static_cast<float>(acc);
      }
    }
  }
  void RawGemv(GgmlType type, const void* w, const float* x, float* y, std::size_t m,
               std::size_t k) override {
    const auto& dec = Decoded(w, type, m, k);
    for (std::size_t o = 0; o < m; ++o) {
      double acc = 0;
      for (std::size_t c = 0; c < k; ++c) acc += double{dec[o * k + c]} * x[c];
      y[o] = static_cast<float>(acc);
    }
  }

 private:
  const std::vector<float>& Decoded(const void* p, GgmlType type, std::size_t m, std::size_t k) {
    const auto key = std::make_tuple(p, type, m, k);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    cache_.clear();  // one weight at a time is enough
    std::vector<float> out(m * k);
    const std::size_t row_bytes = RowBytes(type, k);
    Require(row_bytes != 0, "cpu backend: row bytes");
    for (std::size_t o = 0; o < m; ++o) {
      Require(gufo::quant::DequantizeRow(type, static_cast<const std::uint8_t*>(p) + o * row_bytes,
                                         &out[o * k], k),
              "cpu backend: decode");
    }
    return cache_.emplace(key, std::move(out)).first->second;
  }
  std::map<std::tuple<const void*, GgmlType, std::size_t, std::size_t>, std::vector<float>> cache_;
};

// ---- formats and data --------------------------------------------------------------------------------

struct Fmt {
  GgmlType type;
  std::size_t block;  // elements per block
  std::size_t bytes;  // bytes per block
  std::vector<std::size_t> half_offsets;
  const char* name;
};

const Fmt& FmtOf(GgmlType type) {
  static const Fmt formats[] = {
      {GgmlType::kQ4_K, 256, 144, {0, 2}, "Q4_K"}, {GgmlType::kQ5_K, 256, 176, {0, 2}, "Q5_K"},
      {GgmlType::kQ6_K, 256, 210, {208}, "Q6_K"},  {GgmlType::kQ8_0, 32, 34, {0}, "Q8_0"},
      {GgmlType::kF32, 1, 4, {}, "F32"},           {GgmlType::kBF16, 1, 2, {}, "BF16"},
  };
  for (const auto& f : formats) {
    if (f.type == type) return f;
  }
  throw std::runtime_error("no format for " + TypeName(type));
}

struct SplitMix {
  std::uint64_t s;
  std::uint64_t Next() {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double Uniform() { return (Next() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
  float Normal() {
    const double u1 = 1.0 - Uniform(), u2 = Uniform();
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2));
  }
};

std::uint16_t Half(float x) {
  const _Float16 h = static_cast<_Float16>(x);
  std::uint16_t bits;
  std::memcpy(&bits, &h, 2);
  return bits;
}

/// Random matrix of `bytes` encoded bytes (rows x cols); dense types get N(0, 0.05) values.
std::vector<std::uint8_t> RandomMatrix(const Fmt& f, std::size_t bytes, SplitMix& rng) {
  std::vector<std::uint8_t> data(bytes);
  if (f.type == GgmlType::kF32 || f.type == GgmlType::kBF16) {
    for (std::size_t i = 0; i < bytes / f.bytes; ++i) {
      const float v = 0.05F * rng.Normal();
      std::uint32_t bits;
      std::memcpy(&bits, &v, 4);
      if (f.type == GgmlType::kF32) {
        std::memcpy(&data[i * 4], &bits, 4);
      } else {
        const std::uint16_t hi = static_cast<std::uint16_t>(bits >> 16);
        std::memcpy(&data[i * 2], &hi, 2);
      }
    }
    return data;
  }
  std::size_t i = 0;
  for (; i + 8 <= bytes; i += 8) {
    const std::uint64_t v = rng.Next();
    std::memcpy(&data[i], &v, 8);
  }
  for (; i < bytes; ++i) data[i] = static_cast<std::uint8_t>(rng.Next());
  const std::size_t blocks = bytes / f.bytes;
  for (std::size_t b = 0; b < blocks; ++b) {
    for (std::size_t off : f.half_offsets) {
      const std::uint16_t h = Half(static_cast<float>(0.0005 + 0.0035 * rng.Uniform()));
      std::memcpy(&data[b * f.bytes + off], &h, 2);
    }
  }
  return data;
}

// ---- device buffers with sentinels ---------------------------------------------------------------------

std::size_t g_device_bytes = 0;
constexpr std::size_t kDeviceLimit = 3500ULL << 20;  // brief: under 4 GB of GPU memory

class Weights {
 public:
  Weights(Backend& b, const std::vector<std::uint8_t>& host, std::size_t bytes) : b_(b) {
    const std::size_t total = bytes + kWeightTailBytes + kPadAfterTail;
    g_device_bytes += total;
    Require(g_device_bytes <= kDeviceLimit, "test plan exceeds the 3.5 GB device budget");
    base_ = b.Alloc(total);
    b.Upload(base_, host.data(), bytes);
    b.Fill(static_cast<std::uint8_t*>(base_) + bytes, 0, kWeightTailBytes);
    b.Fill(static_cast<std::uint8_t*>(base_) + bytes + kWeightTailBytes, 0xFF, kPadAfterTail);
    total_ = total;
  }
  ~Weights() {
    b_.Free(base_);
    g_device_bytes -= total_;
  }
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;
  const void* data() const { return base_; }

 private:
  Backend& b_;
  void* base_{nullptr};
  std::size_t total_{0};
};

class Buf {
 public:
  Buf(Backend& b, std::size_t n) : b_(b), n_(n) {
    g_device_bytes += (n + 2 * kGuard) * 4;
    Require(g_device_bytes <= kDeviceLimit, "test plan exceeds the 3.5 GB device budget");
    base_ = static_cast<float*>(b.Alloc((n + 2 * kGuard) * 4));
    Poison(true);
  }
  ~Buf() {
    b_.Free(base_);
    g_device_bytes -= (n_ + 2 * kGuard) * 4;
  }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  float* ptr() { return base_ + kGuard; }
  std::size_t size() const { return n_; }
  /// Fills the guards (if asked) and the first `count` interior floats with the NaN pattern.
  void Poison(bool with_guards, std::size_t count = static_cast<std::size_t>(-1)) {
    static const std::vector<std::uint32_t> pat(1U << 20, kPattern);
    float* start = with_guards ? base_ : ptr();
    std::size_t total = with_guards ? n_ + 2 * kGuard : std::min(count, n_);
    for (std::size_t i = 0; i < total; i += pat.size()) {
      b_.Upload(start + i, pat.data(), std::min(pat.size(), total - i) * 4);
    }
  }
  void Set(const std::vector<float>& v) {
    Require(v.size() <= n_, "Buf::Set larger than buffer");
    b_.Upload(ptr(), v.data(), v.size() * 4);
  }
  /// Copy of the first `count` interior floats. `guards_ok` reports whether the guard before the buffer
  /// and the kGuard words after `count` still hold the pattern (nothing wrote past the expected end).
  /// Afterwards the region [0, count) is poisoned again, so the buffer is clean for the next call.
  std::vector<float> Fetch(std::size_t count, bool* guards_ok) {
    Require(count <= n_, "Buf::Fetch beyond the buffer");
    std::vector<float> out(count), tail(kGuard), head(kGuard);
    b_.Sync();
    b_.Download(head.data(), base_, kGuard * 4);
    b_.Download(out.data(), ptr(), count * 4);
    b_.Download(tail.data(), ptr() + count, kGuard * 4);
    auto ok = [](const std::vector<float>& g) {
      return std::all_of(g.begin(), g.end(), [](float v) {
        std::uint32_t u;
        std::memcpy(&u, &v, 4);
        return u == kPattern;
      });
    };
    *guards_ok = ok(head) && ok(tail);
    Poison(false, count);
    return out;
  }

 private:
  Backend& b_;
  std::size_t n_;
  float* base_;
};

bool IsPattern(float v) {
  std::uint32_t u;
  std::memcpy(&u, &v, 4);
  return u == kPattern;
}

/// First index of an unwritten or non-finite value, or -1.
long FirstBad(const std::vector<float>& v) {
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (IsPattern(v[i]) || !std::isfinite(v[i])) return static_cast<long>(i);
  }
  return -1;
}

// ---- the matrix under test -----------------------------------------------------------------------------

struct Matrix {
  Backend& b;
  GgmlType type;
  std::size_t m, k;
  std::vector<std::uint8_t> host;  // bytes() valid, possibly more (twins)
  std::unique_ptr<Weights> dev;
  std::size_t bytes() const { return m * RowBytes(type, k); }
  DeviceTensor Tensor() const {
    DeviceTensor t;
    t.data = dev->data();
    t.type = type;
    t.rows = static_cast<std::uint32_t>(m);
    t.cols = static_cast<std::uint32_t>(k);
    t.bytes = bytes();
    return t;
  }
};

std::unique_ptr<Matrix> MakeMatrix(Backend& b, GgmlType type, std::size_t m, std::size_t k, std::uint64_t seed,
                                   double extra = 1.0) {
  auto mat = std::make_unique<Matrix>(Matrix{b, type, m, k, {}, nullptr});
  const std::size_t rb = RowBytes(type, k);
  Require(rb != 0, "row bytes for " + TypeName(type) + " k=" + std::to_string(k));
  SplitMix rng{seed};
  const std::size_t alloc = static_cast<std::size_t>(mat->bytes() * extra);
  mat->host = RandomMatrix(FmtOf(type), alloc / FmtOf(type).bytes * FmtOf(type).bytes, rng);
  mat->dev = std::make_unique<Weights>(b, mat->host, mat->host.size());
  return mat;
}

std::vector<float> RandomX(std::size_t rows, std::size_t k, std::uint64_t seed) {
  SplitMix rng{seed};
  std::vector<float> x(rows * k);
  for (float& v : x) v = rng.Normal();
  return x;
}

std::vector<std::size_t> SampleOutputs(std::size_t m) {
  std::set<std::size_t> s;
  for (std::size_t i = 0; i < 16 && i < m; ++i) {
    s.insert(i);
    s.insert(m - 1 - i);
  }
  for (std::size_t t = 128; t < m && s.size() < 48; t *= 2) {
    s.insert(t - 1);
    s.insert(t);
  }
  s.insert(m / 2);
  SplitMix rng{m * 7919 + 13};
  while (s.size() < 100 && s.size() < m) s.insert(rng.Next() % m);
  return {s.begin(), s.end()};
}

// ---- references ------------------------------------------------------------------------------------------

struct Fp64Stats {
  double worst_rel{0};  // |got - want| / sum|w x|
  double max_z{0};      // |got - want| / sigma_o (activation-rounding noise), W8A8 only
  double sum_z2{0};
  std::size_t n{0};
  double excess{0};  // max |got-want| / (6 sigma_o + 2e-6 mag) -- must stay <= 1
};

/// Decodes each sampled weight row once and compares got[r][o] with the FP64 dot of x[r].
/// got_rows[i] points at M outputs for x_rows[i]. A non-finite or poisoned output gives infinity.
Fp64Stats Fp64Compare(const Matrix& mat, const std::vector<std::size_t>& outputs,
                      const std::vector<const float*>& x_rows, const std::vector<const float*>& got_rows) {
  Fp64Stats st;
  const std::size_t k = mat.k, rb = RowBytes(mat.type, k);
  std::vector<float> row(k);
  for (std::size_t o : outputs) {
    Require(gufo::quant::DequantizeRow(mat.type, mat.host.data() + o * rb, row.data(), k), "reference decode");
    for (std::size_t i = 0; i < x_rows.size(); ++i) {
      const float* x = x_rows[i];
      double want = 0, mag = 0, noise = 0;
      for (std::size_t b = 0; b * 32 < k; ++b) {
        float amax = 0;
        const std::size_t lim = std::min<std::size_t>(32, k - b * 32);
        for (std::size_t c = 0; c < lim; ++c) amax = std::max(amax, std::fabs(x[b * 32 + c]));
        const double d = amax / 127.0;
        for (std::size_t c = 0; c < lim; ++c) {
          const double w = row[b * 32 + c];
          const double p = w * x[b * 32 + c];
          want += p;
          mag += std::fabs(p);
          noise += d * d * w * w / 12.0;
        }
      }
      const float g = got_rows[i][o];
      if (IsPattern(g) || !std::isfinite(g)) {
        st.worst_rel = std::numeric_limits<double>::infinity();
        st.max_z = std::numeric_limits<double>::infinity();
        st.excess = std::numeric_limits<double>::infinity();
        continue;
      }
      const double err = std::fabs(double{g} - want);
      st.worst_rel = std::max(st.worst_rel, err / mag);
      const double sigma = std::sqrt(noise);
      st.max_z = std::max(st.max_z, err / sigma);
      st.sum_z2 += err * err / noise;
      ++st.n;
      st.excess = std::max(st.excess, err / (6 * sigma + kDotBound * mag));
    }
  }
  return st;
}

// ---- running Project on the harness buffers --------------------------------------------------------------

struct Harness {
  Backend& b;
  Matrix& mat;
  Buf x;
  Buf y;
  std::vector<float> xh;
  std::string ctx;

  Harness(Backend& be, Matrix& m, const std::vector<float>& xhost, std::size_t max_rows, std::size_t max_out)
      : b(be), mat(m), x(be, xhost.size()), y(be, max_rows * max_out), xh(xhost) {
    x.Set(xhost);
    ctx = TypeName(m.type) + " " + std::to_string(m.m) + "x" + std::to_string(m.k);
  }

  /// Project(rows from x row `first`), poison first, fetch, verify the sentinels. No exception on bad
  /// values: the caller decides (twins want to see them).
  std::vector<float> Run(DeviceTensor t, int rows, std::size_t first, bool exact, bool* guards_ok) {
    b.Project(t, x.ptr() + first * mat.k, rows, y.ptr(), exact);
    return y.Fetch(static_cast<std::size_t>(rows) * t.rows, guards_ok);
  }

  /// Like Run for a call that must succeed: throws with the first bad row/column.
  std::vector<float> RunChecked(int rows, std::size_t first, bool exact) {
    bool guards = true;
    auto out = Run(mat.Tensor(), rows, first, exact, &guards);
    const std::string where = ctx + " rows=" + std::to_string(rows) + (exact ? " exact" : "") + ": ";
    Require(guards, where + "wrote outside [y, y + rows*M) (guard words changed)");
    const long bad = FirstBad(out);
    Require(bad < 0, where + "output not written or not finite at row " + std::to_string(bad / mat.m) +
                         " column " + std::to_string(bad % mat.m));
    return out;
  }
};

void RequireBitEqual(const float* got, const float* want, std::size_t rows, std::size_t m,
                     const std::string& where) {
  if (std::memcmp(got, want, rows * m * 4) == 0) return;
  for (std::size_t i = 0; i < rows * m; ++i) {
    if (std::memcmp(&got[i], &want[i], 4) != 0) {
      throw std::runtime_error(where + ": first difference at row " + std::to_string(i / m) + " column " +
                               std::to_string(i % m) + " got " + std::to_string(got[i]) + " want " +
                               std::to_string(want[i]));
    }
  }
}

double Seconds(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---- cases -----------------------------------------------------------------------------------------------

struct ShapeCase {
  std::string name;
  GgmlType type;
  std::size_t m, k;
};

const std::vector<ShapeCase>& ShapeCases() {
  static const std::vector<ShapeCase> cases = {
      {"q4k_10240x5120", GgmlType::kQ4_K, 10240, 5120},   // attn_q, attn_gate
      {"q8_0_1024x5120", GgmlType::kQ8_0, 1024, 5120},    // attn_k, attn_v
      {"q6k_5120x10240", GgmlType::kQ6_K, 5120, 10240},   // attn_output
      {"q4k_40960x5120", GgmlType::kQ4_K, 40960, 5120},   // ffn_gate, ffn_up (50 layers)
      {"q5k_40960x5120", GgmlType::kQ5_K, 40960, 5120},   // ffn_gate, ffn_up (30 layers)
      {"q4k_5120x40960", GgmlType::kQ4_K, 5120, 40960},   // ffn_down
      {"q5k_5120x40960", GgmlType::kQ5_K, 5120, 40960},   // ffn_down
      {"q6k_100352x5120", GgmlType::kQ6_K, 100352, 5120}, // output head
  };
  return cases;
}

struct Options {
  bool cpu{false};
  std::size_t max_rows{512};
};

void RunShape(Backend& b, const ShapeCase& c, const Options& opt) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::size_t m = opt.cpu ? std::min<std::size_t>(c.m, 392) : c.m;
  const std::size_t k = c.k;
  const std::size_t rmax = opt.max_rows;
  g_device_bytes = 0;
  auto mat = MakeMatrix(b, c.type, m, k, 0x5EED0000 + m * 31 + k);
  const auto xh = RandomX(rmax, k, 0xC0FFEE + k);
  // Room for the largest call; one-hot probes are appended as extra x rows.
  const std::size_t probes = 4;
  std::vector<float> xall = xh;
  const std::size_t probe_cols[probes] = {0, k - 1, k - 32, k - 256};
  xall.resize((rmax + probes) * k, 0.0F);
  for (std::size_t i = 0; i < probes; ++i) xall[(rmax + i) * k + probe_cols[i]] = 1.0F;
  Harness h(b, *mat, xall, std::max<std::size_t>(rmax, 20), m);
  const std::string name = h.ctx;
  const auto S = SampleOutputs(m);

  // 1. single rows
  std::vector<float> single(rmax * m);
  for (std::size_t r = 0; r < rmax; ++r) {
    const auto out = h.RunChecked(1, r, false);
    std::memcpy(&single[r * m], out.data(), m * 4);
  }
  {
    std::vector<const float*> xr = {&xh[0], &xh[std::min<std::size_t>(15, rmax - 1) * k]};
    std::vector<const float*> gr = {&single[0], &single[std::min<std::size_t>(15, rmax - 1) * m]};
    const auto st = Fp64Compare(*mat, S, xr, gr);
    std::cout << name << ": GEMV vs FP64 on " << S.size() << " outputs x 2 rows: worst relative error "
              << st.worst_rel << " (bound " << kDotBound << ")\n";
    Require(st.worst_rel < kDotBound, name + ": GEMV error " + std::to_string(st.worst_rel) + " above the bound");
  }
  // 2. one-hot probes: the dequantized weight of the first/last columns, via GEMV, small batch, exact
  {
    const std::size_t rb = RowBytes(c.type, k);
    std::vector<float> wrow(k);
    const auto batch = h.RunChecked(static_cast<int>(probes), rmax, false);
    const auto exact = h.RunChecked(static_cast<int>(probes), rmax, true);
    double worst = 0;
    std::vector<std::vector<float>> ones;
    for (std::size_t i = 0; i < probes; ++i) ones.push_back(h.RunChecked(1, rmax + i, false));
    for (std::size_t o : S) {
      Require(gufo::quant::DequantizeRow(c.type, mat->host.data() + o * rb, wrow.data(), k), "probe decode");
      double rowmax = 0;
      for (float v : wrow) rowmax = std::max(rowmax, std::fabs(double{v}));
      for (std::size_t i = 0; i < probes; ++i) {
        const double want = wrow[probe_cols[i]];
        for (const float* got : {static_cast<const float*>(&ones[i][o]), static_cast<const float*>(&batch[i * m + o]), static_cast<const float*>(&exact[i * m + o])}) {
          const double e = std::fabs(double{*got} - want) / rowmax;
          worst = std::max(worst, e);
          Require(e < 2e-6, name + ": one-hot probe column " + std::to_string(probe_cols[i]) + " output " +
                                std::to_string(o) + ": got " + std::to_string(*got) + " want " +
                                std::to_string(want));
        }
      }
    }
    std::cout << name << ": one-hot probes at columns 0, K-1, K-32, K-256: worst |err|/max|w| " << worst << "\n";
  }
  // 3. widths 2..16 bit-equal to the single rows
  for (std::size_t w = 2; w <= 16 && w <= rmax; ++w) {
    const auto out = h.RunChecked(static_cast<int>(w), 0, false);
    RequireBitEqual(out.data(), single.data(), w, m, name + ": rows=" + std::to_string(w) + " vs single rows");
  }
  std::cout << name << ": rows 2..16 bit-equal to the single-row results (all outputs)\n";
  // 4/5. rows > 16: exact and normal mode
  for (std::size_t rows : {std::size_t{17}, std::size_t{512}}) {
    if (rows > rmax) continue;
    const auto exact = h.RunChecked(static_cast<int>(rows), 0, true);
    RequireBitEqual(exact.data(), single.data(), rows, m, name + ": rows=" + std::to_string(rows) + " exact vs single rows");
    const auto fast = h.RunChecked(static_cast<int>(rows), 0, false);
    double err2 = 0, ref2 = 0, worst_ratio = 0;
    for (std::size_t r = 0; r < rows; ++r) {
      double row2 = 0;
      for (std::size_t o = 0; o < m; ++o) row2 += double{single[r * m + o]} * single[r * m + o];
      const double rms = std::sqrt(row2 / m);
      for (std::size_t o = 0; o < m; ++o) {
        const double d = double{fast[r * m + o]} - single[r * m + o];
        err2 += d * d;
        ref2 += double{single[r * m + o]} * single[r * m + o];
        worst_ratio = std::max(worst_ratio, std::fabs(d) / rms);
        if (std::fabs(d) > 0.06 * rms) {
          throw std::runtime_error(name + ": rows=" + std::to_string(rows) + " normal mode: row " +
                                   std::to_string(r) + " column " + std::to_string(o) + " differs from the exact result by " +
                                   std::to_string(d) + " (> 0.06 x row RMS " + std::to_string(rms) + ")");
        }
      }
    }
    const double rel = std::sqrt(err2 / ref2);
    Require(rel < 1.5e-2, name + ": rows=" + std::to_string(rows) + " normal mode RMS relative error " + std::to_string(rel));
    std::vector<std::size_t> pick = {0, 1, 15, 16};
    if (rows > 17) pick.insert(pick.end(), {255, rows - 1});
    std::vector<const float*> xr, gr;
    for (std::size_t r : pick) {
      xr.push_back(&xh[r * k]);
      gr.push_back(&fast[r * m]);
    }
    const auto st = Fp64Compare(*mat, S, xr, gr);
    Require(st.excess <= 1.0, name + ": rows=" + std::to_string(rows) + " normal mode outside the FP64 envelope (excess " +
                                  std::to_string(st.excess) + ")");
    std::cout << name << ": rows=" << rows << " exact bit-equal to single rows; normal mode: RMS rel " << rel
              << ", max |diff|/rowRMS " << worst_ratio << " (limit 0.06), FP64 sampled: max z "
              << st.max_z << ", rms z " << std::sqrt(st.sum_z2 / std::max<std::size_t>(1, st.n))
              << " (sigma units; envelope 6), excess " << st.excess << "\n";
  }
  const double secs = Seconds(t0);
  std::cout << name << ": done in " << secs << " s" << (secs > 20 ? "  WARNING: over the 20 s budget" : "") << "\n";
}

void RunDense(Backend& b, GgmlType type, std::size_t m, std::size_t k) {
  g_device_bytes = 0;
  auto mat = MakeMatrix(b, type, m, k, 0xD35E + m * 3 + k);
  const std::size_t rmax = 33;
  const auto xh = RandomX(rmax, k, 0xF00D + k);
  Harness h(b, *mat, xh, rmax, m);
  std::vector<float> single(rmax * m);
  for (std::size_t r = 0; r < rmax; ++r) {
    const auto out = h.RunChecked(1, r, false);
    std::memcpy(&single[r * m], out.data(), m * 4);
  }
  const auto S = SampleOutputs(m);
  const auto st = Fp64Compare(*mat, S, {&xh[0], &xh[(rmax - 1) * k]}, {&single[0], &single[(rmax - 1) * m]});
  Require(st.worst_rel < kDotBound, h.ctx + ": GEMV error " + std::to_string(st.worst_rel));
  for (int rows : {2, 8, 9, 16, 17, 33}) {
    const auto out = h.RunChecked(rows, 0, false);
    RequireBitEqual(out.data(), single.data(), rows, m, h.ctx + ": rows=" + std::to_string(rows));
    const auto ex = h.RunChecked(rows, 0, true);
    RequireBitEqual(ex.data(), single.data(), rows, m, h.ctx + ": exact rows=" + std::to_string(rows));
  }
  std::cout << h.ctx << ": rows 1,2,8,9,16,17,33 equal per-row GEMV; FP64 worst relative error " << st.worst_rel << "\n";
}

void RunF32Tiny(Backend& b, const Options&) {
  // hidden 256/512, query width 1280/2560, FFN 2048/4096, vocabulary 512 (tiny fixtures), both directions.
  const std::pair<std::size_t, std::size_t> shapes[] = {
      {1280, 256}, {256, 1280}, {2048, 256}, {256, 2048}, {512, 256}, {256, 512},
      {2560, 512}, {512, 2560}, {4096, 512}, {512, 4096}, {512, 512}};
  for (const auto& [m, k] : shapes) RunDense(b, GgmlType::kF32, m, k);
  RunDense(b, GgmlType::kBF16, 2048, 256);
  RunDense(b, GgmlType::kBF16, 512, 4096);
}

/// Margin of a broken twin over the correct kernel: error relative to sum|w x| on the sampled outputs.
double TwinError(const Matrix& mat, const std::vector<float>& got, const std::vector<float>& x, std::size_t out_count) {
  const auto S = SampleOutputs(std::min(mat.m, out_count));
  const auto st = Fp64Compare(mat, S, {x.data()}, {got.data()});
  return st.worst_rel;
}

void RunTwins(Backend& b, const Options& opt) {
  const std::size_t m = 1024, k = 5120;
  // Reading buffers wrongly can over-read: every twin matrix carries 1.7x the bytes it declares.
  for (GgmlType type : {GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K, GgmlType::kQ8_0}) {
    g_device_bytes = 0;
    auto mat = MakeMatrix(b, type, m, k, 0x7A11 + static_cast<std::uint64_t>(type), 1.7);
    const std::size_t xrows = 16;
    // x wide enough for the transposed twin (K' = M < K here, but rows' = K > M outputs).
    const auto xh = RandomX(xrows, k, 99);
    Harness h(b, *mat, xh, xrows, k);  // y room for the transposed twin's K outputs per row
    const std::string name = TypeName(type) + " " + std::to_string(m) + "x" + std::to_string(k);
    const auto good = h.RunChecked(9, 0, false);
    const double e_good = TwinError(*mat, good, xh, m);
    Require(e_good < kDotBound, name + ": baseline error " + std::to_string(e_good));
    auto report = [&](const char* what, double e) {
      std::cout << "twin " << name << " " << what << ": error " << e << " vs correct " << e_good << " (bound "
                << kDotBound << ")\n";
      Require(!(e < 1e-3 && std::isfinite(e)), name + ": twin '" + what + "' is not clearly wrong: " + std::to_string(e));
    };
    // (a) weights of one type read as another.
    const GgmlType wrong = type == GgmlType::kQ4_K ? GgmlType::kQ5_K
                           : type == GgmlType::kQ5_K ? GgmlType::kQ4_K
                           : type == GgmlType::kQ6_K ? GgmlType::kQ8_0 : GgmlType::kQ6_K;
    {
      b.RawGemv(wrong, mat->dev->data(), h.x.ptr(), h.y.ptr(), m, k);
      bool g;
      const auto out = h.y.Fetch(m, &g);
      report(("read as " + TypeName(wrong)).c_str(), TwinError(*mat, out, xh, m));
    }
    // (b) Project refuses the same confusion when the declared size disagrees (the CPU stand-in only
    // mimics the contract, so this is a real-Project check).
    if (!opt.cpu) {
      DeviceTensor t = mat->Tensor();
      t.type = wrong;
      bool refused = false;
      try {
        bool g;
        (void)h.Run(t, 1, 0, false, &g);
      } catch (const std::runtime_error& e) {
        refused = std::string(e.what()).find("byte size") != std::string::npos;
      }
      Require(refused, name + ": Project accepted a tensor whose size does not match its declared type");
    }
    // (c) row count off by one: the last row is never computed.
    {
      b.Project(mat->Tensor(), h.x.ptr(), 8, h.y.ptr(), false);
      // the 9th row stays poisoned in the device buffer: read it back as the harness would
      bool g2;
      auto full = h.y.Fetch(9 * m, &g2);
      const std::size_t bad = static_cast<std::size_t>(FirstBad(full));
      Require(FirstBad(full) >= 0 && bad / m == 8, name + ": off-by-one row count not detected as an unwritten row");
      std::cout << "twin " << name << " rows=8 instead of 9: row 8 unwritten (sentinel at row " << bad / m
                << " column " << bad % m << "); the harness check rejects it\n";
    }
    // (d) output count off by one: the last output of every row is never computed.
    {
      DeviceTensor t = mat->Tensor();
      t.rows = static_cast<std::uint32_t>(m - 1);
      t.bytes = (m - 1) * RowBytes(type, k);
      b.Project(t, h.x.ptr(), 1, h.y.ptr(), false);
      bool g;
      auto full = h.y.Fetch(m, &g);
      const long bad = FirstBad(full);
      Require(bad == static_cast<long>(m - 1), name + ": M-1 twin did not leave exactly the last output unwritten");
      std::cout << "twin " << name << " M-1: output " << bad << " unwritten; the harness check rejects it\n";
    }
    // (e) transposed dims (same byte count): rows = K, cols = M.
    {
      DeviceTensor t = mat->Tensor();
      t.rows = static_cast<std::uint32_t>(k);
      t.cols = static_cast<std::uint32_t>(m);
      Require(t.bytes == k * RowBytes(type, m), "transposed twin must keep the byte count");
      bool g;
      const auto out = h.Run(t, 1, 0, false, &g);
      std::vector<float> first(out.begin(), out.begin() + m);
      report("transposed dims", TwinError(*mat, first, xh, m));
    }
  }
}

void RunReject(Backend&, const Options&) {
#if defined(ENGINE_ENABLE_HIP)
  using gufo::models::ricursive::rocm::Project;
  gufo::models::ricursive::rocm::Scratch scratch;
  float dummy = 0;
  DeviceTensor t;
  t.data = &dummy;  // never dereferenced: validation throws before any launch
  t.rows = 256;
  t.cols = 256;
  struct Probe {
    GgmlType type;
    const char* name;
  };
  for (const Probe& p : {Probe{GgmlType::kIQ4_XS, "IQ4_XS"}, Probe{GgmlType::kF16, "F16"}, Probe{GgmlType::kQ4_0, "Q4_0"},
                         Probe{GgmlType::kQ3_K, "Q3_K"}, Probe{GgmlType::kIQ3_S, "IQ3_S"}}) {
    t.type = p.type;
    t.bytes = 136 * 256;
    for (int rows : {1, 4, 17}) {
      bool threw = false;
      try {
        Project(t, &dummy, rows, &dummy, scratch, nullptr, rows == 4);
      } catch (const std::runtime_error& e) {
        threw = std::string(e.what()).find(p.name) != std::string::npos;
      }
      Require(threw, std::string("Project did not reject ") + p.name + " (rows " + std::to_string(rows) + ") with its name in the message");
    }
  }
  std::cout << "Project rejects IQ4_XS, F16, Q4_0, Q3_K, IQ3_S naming the type, at rows 1, 4 (exact) and 17\n";
  // Wrong size, bad rows, null pointers.
  t.type = GgmlType::kQ4_K;
  t.bytes = 12345;
  bool threw = false;
  try {
    Project(t, &dummy, 1, &dummy, scratch, nullptr, false);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "wrong byte size accepted");
  t.bytes = 256 * 144;
  t.cols = 250;  // not block aligned
  threw = false;
  try {
    Project(t, &dummy, 1, &dummy, scratch, nullptr, false);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "unaligned cols accepted");
  t.cols = 256;
  threw = false;
  try {
    Project(t, &dummy, 0, &dummy, scratch, nullptr, false);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "rows 0 accepted");
  t.data = nullptr;
  threw = false;
  try {
    Project(t, &dummy, 1, &dummy, scratch, nullptr, false);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  Require(threw, "null weight accepted");
  std::cout << "Project rejects wrong byte size, unaligned cols, rows 0 and a null weight\n";
#endif
}

std::vector<std::string> AllCases() {
  std::vector<std::string> names;
  for (const auto& c : ShapeCases()) names.push_back(c.name);
  names.push_back("f32-tiny");
  names.push_back("twins");
  names.push_back("reject");
  return names;
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  std::string which;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--list") {
      for (const auto& n : AllCases()) std::cout << n << "\n";
      return 0;
    }
    if (a == "--cpu") {
      opt.cpu = true;
      opt.max_rows = 40;
    } else if (a == "--max-rows" && i + 1 < argc) {
      opt.max_rows = static_cast<std::size_t>(std::atol(argv[++i]));
    } else {
      which = a;
    }
  }
  if (which.empty()) {
    std::cerr << "usage: projection_ops_test --list | <case> [--cpu] [--max-rows N]\n";
    return 2;
  }
  std::unique_ptr<Backend> backend;
  if (opt.cpu) {
    backend = std::make_unique<CpuBackend>();
  } else {
#if defined(ENGINE_ENABLE_HIP)
    int devices = 0;
    if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
      std::cout << "SKIP: no HIP device\n";
      return 77;
    }
    backend = std::make_unique<GpuBackend>();
#else
    std::cout << "SKIP: built without HIP\n";
    return 77;
#endif
  }
  Backend& b = *backend;
  return ricursive_test::Run([&] {
    std::cout << "backend: " << b.Name() << ", case " << which << "\n";
    if (which == "f32-tiny") {
      RunF32Tiny(b, opt);
    } else if (which == "twins") {
      RunTwins(b, opt);
    } else if (which == "reject") {
      Require(!opt.cpu, "reject needs the real Project");
      RunReject(b, opt);
    } else {
      const auto& cases = ShapeCases();
      const auto it = std::find_if(cases.begin(), cases.end(), [&](const ShapeCase& c) { return c.name == which; });
      Require(it != cases.end(), "unknown case " + which);
      RunShape(b, *it, opt);
    }
  });
}

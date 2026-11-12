// ricursive_kernel_bench: times one ricursive GPU kernel on synthetic tensors of the real Ripple-65B
// shapes (docs/PERF-KERNELS.md). Run only through the GPU wrapper.
//
//   kernel_bench attn  KEYS [sliding|full] [REPS]      Attention128, 1 row, 80 heads / 8 KV heads, F16 caches
//   kernel_bench proj  TYPE M K [REPS]                 Project, 1 row; TYPE = q4k|q5k|q6k|q8_0
//   kernel_bench suite                                 every case listed in the task brief
//   kernel_bench gemm16 [REPS]                         9..16 rows: split 8-row passes vs one DoubleStage pass,
//                                                      memcmp bit-equality and alternating timing (exit 4 on a diff)
//
// Each case rotates over several distinct weight / cache copies (so the data is not warm in cache,
// like a real token that streams 40 GB), times batches of 8 launches with hip events, reports the
// median per-call time, and prints an FNV-1a hash of the output bits (identical hash = identical bits).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/gguf_reader.hpp>
#include <string>
#include <vector>

#include "hip/ggml/gemm.hpp"
#include "hip/kernels.hpp"
#include "hip/project.hpp"

namespace k = gufo::models::ricursive::rocm;
using gufo::core::GgmlType;

#define HCHECK(x)                                                              \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x,           \
                   hipGetErrorString(e_));                                     \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

namespace {

struct Rng {
  std::uint64_t s;
  std::uint32_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<std::uint32_t>(s >> 16);
  }
  float Unit() { return (static_cast<float>(Next() & 0xFFFF) / 32768.0F) - 1.0F; }
};

std::uint64_t Fnv(const std::vector<float>& v) {
  std::uint64_t h = 1469598103934665603ULL;
  for (float f : v) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    for (int i = 0; i < 4; ++i) {
      h ^= (u >> (8 * i)) & 0xFF;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

std::uint64_t FnvBytes(const std::vector<std::uint8_t>& v, std::uint64_t h = 1469598103934665603ULL) {
  for (std::uint8_t b : v) {
    h ^= b;
    h *= 1099511628211ULL;
  }
  return h;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

/// Times fn(i) for `reps` samples of 8 launches each (i = launch index); returns the median
/// microseconds per launch.
template<class F>
double TimeIt(int reps, F&& fn) {
  hipEvent_t a, b;
  HCHECK(hipEventCreate(&a));
  HCHECK(hipEventCreate(&b));
  for (int i = 0; i < 16; ++i) fn(i);  // warm up
  HCHECK(hipDeviceSynchronize());
  std::vector<double> us;
  int n = 0;
  for (int r = 0; r < reps; ++r) {
    HCHECK(hipEventRecord(a, nullptr));
    for (int i = 0; i < 8; ++i) fn(n++);
    HCHECK(hipEventRecord(b, nullptr));
    HCHECK(hipEventSynchronize(b));
    float ms = 0;
    HCHECK(hipEventElapsedTime(&ms, a, b));
    us.push_back(ms * 1000.0 / 8.0);
  }
  return Median(us);
}

// ---- attention ----------------------------------------------------------------

void RunAttention(int keys, bool sliding, int reps) {
  constexpr int kHeads = 80, kKv = 8, kDim = 128;
  constexpr int kCopies = 16;  // distinct layers' caches
  const std::uint32_t window = sliding ? 1024 : 0;
  const std::uint32_t capacity = sliding ? 1024 : static_cast<std::uint32_t>(keys) + 1;
  const std::uint32_t pos = static_cast<std::uint32_t>(keys) - 1;  // sees `keys` keys (window clamps)
  const std::size_t cache_elems = std::size_t{capacity} * kKv * kDim;
  Rng rng{0x9E3779B97F4A7C15ULL};
  std::vector<__half> hc(cache_elems);
  std::vector<float> hq(std::size_t{kHeads} * kDim);
  for (auto& q : hq) q = rng.Unit();
  std::vector<void*> kc(kCopies), vc(kCopies);
  for (int c = 0; c < kCopies; ++c) {
    for (auto& x : hc) x = __float2half(rng.Unit());
    HCHECK(hipMalloc(&kc[c], cache_elems * 2));
    HCHECK(hipMemcpy(kc[c], hc.data(), cache_elems * 2, hipMemcpyHostToDevice));
    for (auto& x : hc) x = __float2half(rng.Unit());
    HCHECK(hipMalloc(&vc[c], cache_elems * 2));
    HCHECK(hipMemcpy(vc[c], hc.data(), cache_elems * 2, hipMemcpyHostToDevice));
  }
  float *q, *out;
  HCHECK(hipMalloc(&q, hq.size() * 4));
  HCHECK(hipMemcpy(q, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
  HCHECK(hipMalloc(&out, hq.size() * 4));
  std::string err;
  auto call = [&](int i) {
    const int c = i % kCopies;
    if (!k::Attention128(q, kc[c], vc[c], k::KvType::kF16, out, 1, kHeads, kKv, window, pos, capacity,
                         0.08838834764831845F, nullptr, &err)) {
      std::fprintf(stderr, "Attention128: %s\n", err.c_str());
      std::exit(2);
    }
  };
  // output hash from layer 0 (deterministic: same inputs)
  call(0);
  HCHECK(hipDeviceSynchronize());
  std::vector<float> ho(hq.size());
  HCHECK(hipMemcpy(ho.data(), out, ho.size() * 4, hipMemcpyDeviceToHost));
  const double us = TimeIt(reps, call);
  std::printf("attn %-7s keys=%-5d  %8.1f us/layer   hash=%016llx\n", sliding ? "sliding" : "full", keys, us,
              static_cast<unsigned long long>(Fnv(ho)));
  for (int c = 0; c < kCopies; ++c) {
    HCHECK(hipFree(kc[c]));
    HCHECK(hipFree(vc[c]));
  }
  HCHECK(hipFree(q));
  HCHECK(hipFree(out));
}

// ---- multi-row passes (prompt processing) -------------------------------------------

/// QkNormRopeAppend for `rows` rows (position 0..rows-1) into F16 caches. The hash covers q (rotated in
/// place), the K cache and the V cache. Each timed call recomputes from the same inputs: q is restored first
/// (the kernel works in place), which is part of the timed region only as a device copy of 84 MB at 2048 rows,
/// so it is timed separately and subtracted.
void RunQkRope(int rows, bool sliding, int reps) {
  constexpr int kHeads = 80, kKv = 8, kDim = 128;
  const std::uint32_t capacity = sliding ? 1024 + rows - 1 : rows;
  Rng rng{0xABCDEF12345ULL};
  std::vector<float> hq(std::size_t{rows} * kHeads * kDim), hk(std::size_t{rows} * kKv * kDim),
      hv(hk.size()), gq(kDim), gk(kDim);
  for (auto& x : hq) x = rng.Unit();
  for (auto& x : hk) x = rng.Unit();
  for (auto& x : hv) x = rng.Unit();
  for (auto& x : gq) x = 0.5F + rng.Unit() * 0.25F;
  for (auto& x : gk) x = 0.5F + rng.Unit() * 0.25F;
  float *q, *q0, *kk, *vv, *dgq, *dgk;
  void *kc, *vc;
  const std::size_t cbytes = std::size_t{capacity} * kKv * kDim * 2;
  HCHECK(hipMalloc(&q, hq.size() * 4));
  HCHECK(hipMalloc(&q0, hq.size() * 4));
  HCHECK(hipMalloc(&kk, hk.size() * 4));
  HCHECK(hipMalloc(&vv, hv.size() * 4));
  HCHECK(hipMalloc(&dgq, kDim * 4));
  HCHECK(hipMalloc(&dgk, kDim * 4));
  HCHECK(hipMalloc(&kc, cbytes));
  HCHECK(hipMalloc(&vc, cbytes));
  HCHECK(hipMemcpy(q0, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
  HCHECK(hipMemcpy(kk, hk.data(), hk.size() * 4, hipMemcpyHostToDevice));
  HCHECK(hipMemcpy(vv, hv.data(), hv.size() * 4, hipMemcpyHostToDevice));
  HCHECK(hipMemcpy(dgq, gq.data(), kDim * 4, hipMemcpyHostToDevice));
  HCHECK(hipMemcpy(dgk, gk.data(), kDim * 4, hipMemcpyHostToDevice));
  std::string err;
  auto call = [&](int) {
    HCHECK(hipMemcpyAsync(q, q0, hq.size() * 4, hipMemcpyDeviceToDevice, nullptr));
    if (!k::QkNormRopeAppend(q, kk, vv, dgq, dgk, kc, vc, k::KvType::kF16, rows, kHeads, kKv, 1e-6F, 0, capacity,
                             kDim, 10000.0F, nullptr, &err)) {
      std::fprintf(stderr, "QkNormRopeAppend: %s\n", err.c_str());
      std::exit(2);
    }
  };
  call(0);
  HCHECK(hipDeviceSynchronize());
  std::vector<std::uint8_t> bq(hq.size() * 4), bk(cbytes), bv(cbytes);
  HCHECK(hipMemcpy(bq.data(), q, bq.size(), hipMemcpyDeviceToHost));
  HCHECK(hipMemcpy(bk.data(), kc, cbytes, hipMemcpyDeviceToHost));
  HCHECK(hipMemcpy(bv.data(), vc, cbytes, hipMemcpyDeviceToHost));
  const std::uint64_t h = FnvBytes(bv, FnvBytes(bk, FnvBytes(bq)));
  const double both = TimeIt(reps, call);
  const double copy = TimeIt(reps, [&](int) {
    HCHECK(hipMemcpyAsync(q, q0, hq.size() * 4, hipMemcpyDeviceToDevice, nullptr));
  });
  std::printf("qkrope %-7s rows=%-5d  %9.1f us (q restore %.1f us subtracted)   hash=%016llx\n",
              sliding ? "sliding" : "full", rows, both - copy, copy, static_cast<unsigned long long>(h));
  for (void* p : {static_cast<void*>(q), static_cast<void*>(q0), static_cast<void*>(kk), static_cast<void*>(vv),
                  static_cast<void*>(dgq), static_cast<void*>(dgk), kc, vc})
    HCHECK(hipFree(p));
}

/// Attention128 for a pass of `rows` rows starting at position `start` (window 0 = full).
void RunAttentionRows(int rows, int start, int window, int reps) {
  constexpr int kHeads = 80, kKv = 8, kDim = 128;
  const std::uint32_t capacity = window != 0 ? static_cast<std::uint32_t>(window + rows - 1)
                                             : static_cast<std::uint32_t>(start + rows);
  Rng rng{0x5151515151ULL};
  const std::size_t cache_elems = std::size_t{capacity} * kKv * kDim;
  std::vector<__half> hc(cache_elems);
  std::vector<float> hq(std::size_t{rows} * kHeads * kDim);
  for (auto& x : hq) x = rng.Unit();
  void *kc, *vc;
  HCHECK(hipMalloc(&kc, cache_elems * 2));
  HCHECK(hipMalloc(&vc, cache_elems * 2));
  for (auto x = hc.begin(); x != hc.end(); ++x) *x = __float2half(rng.Unit());
  HCHECK(hipMemcpy(kc, hc.data(), cache_elems * 2, hipMemcpyHostToDevice));
  for (auto x = hc.begin(); x != hc.end(); ++x) *x = __float2half(rng.Unit());
  HCHECK(hipMemcpy(vc, hc.data(), cache_elems * 2, hipMemcpyHostToDevice));
  float *q, *out;
  HCHECK(hipMalloc(&q, hq.size() * 4));
  HCHECK(hipMalloc(&out, hq.size() * 4));
  HCHECK(hipMemcpy(q, hq.data(), hq.size() * 4, hipMemcpyHostToDevice));
  std::string err;
  auto call = [&](int) {
    if (!k::Attention128(q, kc, vc, k::KvType::kF16, out, rows, kHeads, kKv, window, start, capacity,
                         0.08838834764831845F, nullptr, &err)) {
      std::fprintf(stderr, "Attention128: %s\n", err.c_str());
      std::exit(2);
    }
  };
  call(0);
  HCHECK(hipDeviceSynchronize());
  std::vector<float> ho(hq.size());
  HCHECK(hipMemcpy(ho.data(), out, ho.size() * 4, hipMemcpyDeviceToHost));
  const double us = TimeIt(reps, call);
  std::printf("attn-rows %-7s rows=%-5d start=%-5d  %10.1f us   hash=%016llx\n", window != 0 ? "sliding" : "full",
              rows, start, us, static_cast<unsigned long long>(Fnv(ho)));
  for (void* p : {static_cast<void*>(kc), static_cast<void*>(vc), static_cast<void*>(q), static_cast<void*>(out)})
    HCHECK(hipFree(p));
}

// ---- projection ----------------------------------------------------------------

void RunProjection(const std::string& tname, int M, int K, int reps) {
  GgmlType type;
  std::size_t block_bytes, qk;
  std::size_t scale_off[2] = {0, 2};
  int nscale = 2;
  if (tname == "q4k") { type = GgmlType::kQ4_K; block_bytes = 144; qk = 256; }
  else if (tname == "q5k") { type = GgmlType::kQ5_K; block_bytes = 176; qk = 256; }
  else if (tname == "q6k") { type = GgmlType::kQ6_K; block_bytes = 210; qk = 256; scale_off[0] = 208; nscale = 1; }
  else if (tname == "q8_0") { type = GgmlType::kQ8_0; block_bytes = 34; qk = 32; nscale = 1; }
  else { std::fprintf(stderr, "unknown type %s\n", tname.c_str()); std::exit(1); }
  const std::size_t row_bytes = (static_cast<std::size_t>(K) / qk) * block_bytes;
  const std::size_t bytes = row_bytes * static_cast<std::size_t>(M);
  // Enough distinct copies (>= ~192 MB) that the weights are not warm in the 32 MB infinity cache.
  const int copies = static_cast<int>(std::max<std::size_t>(3, std::min<std::size_t>(40, (192u << 20) / bytes)));
  Rng rng{0x1234567ULL + static_cast<std::uint64_t>(M) * 31 + K};
  std::vector<std::uint8_t> host(bytes + k::kWeightTailBytes, 0);
  std::vector<const void*> w(copies);
  for (int c = 0; c < copies; ++c) {
    for (std::size_t i = 0; i < bytes; i += 4) {
      const std::uint32_t r = rng.Next() | (rng.Next() << 16);
      std::memcpy(&host[i], &r, std::min<std::size_t>(4, bytes - i));
    }
    for (std::size_t b = 0; b < bytes / block_bytes; ++b) {
      for (int s = 0; s < nscale; ++s) {
        const std::uint16_t h = static_cast<std::uint16_t>(0x2000 + (rng.Next() & 0x0FFF));  // 2^-9 .. 2^-6
        std::memcpy(&host[b * block_bytes + scale_off[s]], &h, 2);
      }
    }
    void* d;
    HCHECK(hipMalloc(&d, host.size()));
    HCHECK(hipMemcpy(d, host.data(), host.size(), hipMemcpyHostToDevice));
    w[c] = d;
  }
  std::vector<float> hx(K);
  for (auto& v : hx) v = rng.Unit();
  float *x, *y;
  HCHECK(hipMalloc(&x, std::size_t{4} * K));
  HCHECK(hipMemcpy(x, hx.data(), std::size_t{4} * K, hipMemcpyHostToDevice));
  HCHECK(hipMalloc(&y, std::size_t{4} * M));
  k::Scratch scratch;
  auto call = [&](int i) {
    k::DeviceTensor t;
    t.data = w[i % copies];
    t.type = type;
    t.rows = static_cast<std::uint32_t>(M);
    t.cols = static_cast<std::uint32_t>(K);
    t.bytes = bytes;
    k::Project(t, x, 1, y, scratch, nullptr, false);
  };
  call(0);
  HCHECK(hipDeviceSynchronize());
  std::vector<float> ho(M);
  HCHECK(hipMemcpy(ho.data(), y, ho.size() * 4, hipMemcpyDeviceToHost));
  const double us = TimeIt(reps, call);
  std::printf("proj %-5s %6dx%-6d %8.1f us  %6.1f GB/s   hash=%016llx\n", tname.c_str(), M, K, us,
              static_cast<double>(bytes) / (us * 1e-6) / 1e9, static_cast<unsigned long long>(Fnv(ho)));
  for (auto p : w) HCHECK(hipFree(const_cast<void*>(p)));
  HCHECK(hipFree(x));
  HCHECK(hipFree(y));
}

// ---- 9..16 rows: split 8-row passes against one LaunchKQuantSmallBatchDoubleStage pass (gemm16-20261112) ----

/// Synthetic K-quant weights: random bytes, every fp16 block scale set to 2^-9..2^-6 (as RunProjection).
std::vector<std::uint8_t> RandomKQuant(GgmlType type, std::size_t bytes, Rng& rng) {
  const std::size_t block_bytes = type == GgmlType::kQ4_K ? 144 : type == GgmlType::kQ5_K ? 176 : 210;
  std::vector<std::size_t> scale_off = type == GgmlType::kQ6_K ? std::vector<std::size_t>{208}
                                                               : std::vector<std::size_t>{0, 2};
  std::vector<std::uint8_t> host(bytes + k::kWeightTailBytes, 0);
  for (std::size_t i = 0; i < bytes; i += 4) {
    const std::uint32_t r = rng.Next() | (rng.Next() << 16);
    std::memcpy(&host[i], &r, std::min<std::size_t>(4, bytes - i));
  }
  for (std::size_t b = 0; b < bytes / block_bytes; ++b) {
    for (std::size_t off : scale_off) {
      const std::uint16_t h = static_cast<std::uint16_t>(0x2000 + (rng.Next() & 0x0FFF));
      std::memcpy(&host[b * block_bytes + off], &h, 2);
    }
  }
  return host;
}

/// 16 activation rows of K: uniform [-1, 1) with zeros, denormals and large values mixed in; row 3 all zeros,
/// row 5 all denormals, row 7 mostly large values (1e3 .. 1e30, both signs).
std::vector<float> SpecialX(std::size_t k, Rng& rng) {
  std::vector<float> x(16 * k);
  const float denormals[4] = {1.0e-39F, -1.4e-45F, 5.9e-39F, -2.3e-41F};
  const float large[5] = {1.0e3F, -3.7e6F, 2.5e12F, -8.1e20F, 1.0e30F};
  for (std::size_t r = 0; r < 16; ++r) {
    for (std::size_t c = 0; c < k; ++c) {
      float v = rng.Unit();
      const std::uint32_t pick = rng.Next() % 64;
      if (pick == 0) v = 0.0F;
      else if (pick == 1) v = denormals[rng.Next() % 4];
      else if (pick == 2) v = large[rng.Next() % 5] * (rng.Next() % 2 ? 1.0F : -1.0F);
      if (r == 3) v = 0.0F;
      if (r == 5) v = denormals[rng.Next() % 4];
      if (r == 7 && rng.Next() % 4 != 0) v = large[rng.Next() % 5] * (rng.Next() % 2 ? 1.0F : -1.0F);
      x[r * k + c] = v;
    }
  }
  return x;
}

struct Gemm16Case {
  GgmlType type;
  int m, k;
  std::vector<int> time_rows;  ///< row counts timed (16 for the block shapes, 15 and 16 for the head)
  const char* use;
};

/// Returns false if any bit-equality case failed. `reps` timed samples per method and row count.
bool RunGemm16(const Gemm16Case& c, int reps) {
  namespace gg = gufo::models::ricursive::ggml;
  const char* tname = c.type == GgmlType::kQ4_K ? "Q4_K" : c.type == GgmlType::kQ5_K ? "Q5_K" : "Q6_K";
  const std::size_t M = static_cast<std::size_t>(c.m), K = static_cast<std::size_t>(c.k);
  const std::size_t bytes = M * k::RowBytes(c.type, K);
  const int copies = static_cast<int>(std::max<std::size_t>(3, std::min<std::size_t>(40, (192u << 20) / bytes)));
  Rng rng{0x6E3316ULL + M * 131 + K * 7 + static_cast<std::uint64_t>(c.type)};
  std::vector<void*> w(copies);
  for (int i = 0; i < copies; ++i) {
    const auto host = RandomKQuant(c.type, bytes, rng);
    HCHECK(hipMalloc(&w[i], host.size()));
    HCHECK(hipMemcpy(w[i], host.data(), host.size(), hipMemcpyHostToDevice));
  }
  float *x, *ya, *yb;
  HCHECK(hipMalloc(&x, 16 * K * 4));
  HCHECK(hipMalloc(&ya, 16 * M * 4));
  HCHECK(hipMalloc(&yb, 16 * M * 4));
  const auto split = [&](int copy, std::size_t rows, float* y) {
    gg::LaunchBatchedQuantGEMMFp32(c.type, w[copy], x, y, rows, M, K, nullptr);
  };
  const auto one = [&](int copy, std::size_t rows, float* y) {
    if (!gg::LaunchKQuantSmallBatchDoubleStage(c.type, w[copy], x, y, rows, M, K, nullptr)) {
      std::fprintf(stderr, "LaunchKQuantSmallBatchDoubleStage refused %s %dx%d rows %zu\n", tname, c.m, c.k, rows);
      std::exit(3);
    }
  };
  // Bit-equality: 3 seeds (weight copy + activations), rows 9, 12, 15, 16; outputs poisoned with different
  // patterns so an element neither pass writes still differs.
  bool all_ok = true;
  int cases = 0, passed = 0;
  std::vector<float> ha(16 * M), hb(16 * M);
  for (int seed = 0; seed < 3; ++seed) {
    Rng xr{0xA11CE5ULL + static_cast<std::uint64_t>(seed) * 7919 + K};
    const auto hx = SpecialX(K, xr);
    HCHECK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
    for (std::size_t rows : {std::size_t{9}, std::size_t{12}, std::size_t{15}, std::size_t{16}}) {
      HCHECK(hipMemset(ya, 0xFF, 16 * M * 4));
      HCHECK(hipMemset(yb, 0xFE, 16 * M * 4));
      split(seed % copies, rows, ya);
      one(seed % copies, rows, yb);
      HCHECK(hipDeviceSynchronize());
      HCHECK(hipMemcpy(ha.data(), ya, rows * M * 4, hipMemcpyDeviceToHost));
      HCHECK(hipMemcpy(hb.data(), yb, rows * M * 4, hipMemcpyDeviceToHost));
      ++cases;
      if (std::memcmp(ha.data(), hb.data(), rows * M * 4) == 0) {
        ++passed;
        continue;
      }
      all_ok = false;
      std::size_t i = 0;
      while (std::memcmp(&ha[i], &hb[i], 4) == 0) ++i;
      std::size_t ndiff = 0;
      for (std::size_t j = 0; j < rows * M; ++j) ndiff += std::memcmp(&ha[j], &hb[j], 4) != 0;
      std::printf("gemm16 DIFF %s %dx%d seed %d rows %zu: first at row %zu col %zu split %.9g one-pass %.9g; %zu of %zu differ\n",
                  tname, c.m, c.k, seed, rows, i / M, i % M, ha[i], hb[i], ndiff, rows * M);
    }
  }
  std::printf("gemm16 biteq %s %6dx%-6d rows 9,12,15,16 x 3 seeds: %d of %d cases memcmp-equal  %s  (%s)\n", tname, c.m,
              c.k, passed, cases, passed == cases ? "PASS" : "FAIL", c.use);
  // Timing: split and one pass alternating (order swapped every repetition), each call synchronised, rotating weights.
  {
    Rng xr{0x7173ULL + K};
    std::vector<float> hx(16 * K);
    for (auto& v : hx) v = xr.Unit();
    HCHECK(hipMemcpy(x, hx.data(), hx.size() * 4, hipMemcpyHostToDevice));
  }
  hipEvent_t ea, eb;
  HCHECK(hipEventCreate(&ea));
  HCHECK(hipEventCreate(&eb));
  for (int rows : c.time_rows) {
    int n = 0;
    const auto sample = [&](bool use_one) {
      const int copy = n++ % copies;
      HCHECK(hipDeviceSynchronize());
      HCHECK(hipEventRecord(ea, nullptr));
      if (use_one) one(copy, static_cast<std::size_t>(rows), ya);
      else split(copy, static_cast<std::size_t>(rows), ya);
      HCHECK(hipEventRecord(eb, nullptr));
      HCHECK(hipEventSynchronize(eb));
      float ms = 0;
      HCHECK(hipEventElapsedTime(&ms, ea, eb));
      return ms * 1000.0;
    };
    for (int i = 0; i < 2 * copies + 4; ++i) (void)sample(i % 2 == 1);  // warm-up, both methods, every copy
    std::vector<double> ts, to;
    for (int r = 0; r < reps; ++r) {
      if (r % 2 == 0) {
        ts.push_back(sample(false));
        to.push_back(sample(true));
      } else {
        to.push_back(sample(true));
        ts.push_back(sample(false));
      }
    }
    const double ms_ = Median(ts), mo = Median(to);
    std::printf("gemm16 time %s %6dx%-6d rows %2d: split median %8.1f us min %8.1f | one-pass median %8.1f us min %8.1f"
                " | one/split %.3f (%+.1f %%) | one-pass %.1f GB/s  (%d reps each, %d weight copies; %s)\n",
                tname, c.m, c.k, rows, ms_, *std::min_element(ts.begin(), ts.end()), mo,
                *std::min_element(to.begin(), to.end()), mo / ms_, (ms_ - mo) / ms_ * 100.0,
                static_cast<double>(bytes) / (mo * 1e-6) / 1e9, reps, copies, c.use);
  }
  HCHECK(hipEventDestroy(ea));
  HCHECK(hipEventDestroy(eb));
  for (void* p : w) HCHECK(hipFree(p));
  HCHECK(hipFree(x));
  HCHECK(hipFree(ya));
  HCHECK(hipFree(yb));
  return all_ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: see tools/kernel_bench.cpp\n");
    return 1;
  }
  const std::string mode = argv[1];
  if (mode == "attn" && argc >= 3) {
    RunAttention(std::atoi(argv[2]), argc < 4 || std::string(argv[3]) != "full", argc >= 5 ? std::atoi(argv[4]) : 60);
  } else if (mode == "proj" && argc >= 5) {
    RunProjection(argv[2], std::atoi(argv[3]), std::atoi(argv[4]), argc >= 6 ? std::atoi(argv[5]) : 40);
  } else if (mode == "rows-suite") {
    for (int rows : {510, 2048}) {
      RunQkRope(rows, true, 3);
      RunQkRope(rows, false, 3);
    }
    RunAttentionRows(510, 0, 1024, 3);
    RunAttentionRows(2048, 0, 1024, 3);
    RunAttentionRows(2048, 2048, 1024, 3);
    RunAttentionRows(510, 0, 0, 3);
    RunAttentionRows(2048, 0, 0, 3);
    RunAttentionRows(2048, 2048, 0, 3);
  } else if (mode == "gemm16") {
    // Every distinct K-quant (type, shape) the drafter (DFlash2 Q4_K_M header) and the target (verify, same
    // Project) multiply with 9..16 rows; the head at 15 (drafter, d = 15) and 16 rows (target verify).
    const int reps = argc >= 3 ? std::atoi(argv[2]) : 40;
    const std::vector<Gemm16Case> cases = {
        {GgmlType::kQ4_K, 1024, 5120, {16}, "drafter attn_k/v"},
        {GgmlType::kQ4_K, 10240, 5120, {16}, "drafter+target attn_q"},
        {GgmlType::kQ4_K, 256, 5120, {16}, "drafter selector_hidden"},
        {GgmlType::kQ4_K, 2560, 5120, {16}, "drafter conv_proj"},
        {GgmlType::kQ4_K, 40960, 5120, {16}, "drafter+target ffn_gate/up"},
        {GgmlType::kQ4_K, 5120, 10240, {16}, "drafter attn_output"},
        {GgmlType::kQ4_K, 5120, 25600, {16}, "drafter fc"},
        {GgmlType::kQ4_K, 5120, 40960, {16}, "drafter+target ffn_down"},
        {GgmlType::kQ6_K, 1024, 5120, {16}, "drafter attn_v L2,L4"},
        {GgmlType::kQ6_K, 5120, 40960, {16}, "drafter ffn_down L2,L4"},
        {GgmlType::kQ6_K, 5120, 10240, {16}, "target attn_output"},
        {GgmlType::kQ6_K, 100352, 5120, {15, 16}, "head"},
        {GgmlType::kQ5_K, 40960, 5120, {16}, "target ffn_gate/up"},
        {GgmlType::kQ5_K, 5120, 40960, {16}, "target ffn_down"},
    };
    bool ok = true;
    for (const auto& c : cases) ok = RunGemm16(c, reps) && ok;
    std::printf("gemm16 bit-equality overall: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 4;
  } else if (mode == "attn-suite") {
    RunAttention(540, true, 60);
    RunAttention(1024, true, 60);
    RunAttention(540, false, 60);
    RunAttention(4100, false, 60);
  } else if (mode == "proj-suite") {
    for (const char* t : {"q4k", "q5k", "q6k"}) {
      RunProjection(t, 5120, 40960, 40);
      RunProjection(t, 40960, 5120, 40);
    }
    RunProjection("q4k", 10240, 5120, 40);
    RunProjection("q8_0", 1024, 5120, 40);
    RunProjection("q6k", 10240, 5120, 40);
  } else {
    std::fprintf(stderr, "bad arguments\n");
    return 1;
  }
  return 0;
}

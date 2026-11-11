// ricursive_kernel_bench: times one ricursive GPU kernel on synthetic tensors of the real Ripple-65B
// shapes (docs/PERF-KERNELS.md). Run only through the GPU wrapper.
//
//   kernel_bench attn  KEYS [sliding|full] [REPS]      Attention128, 1 row, 80 heads / 8 KV heads, F16 caches
//   kernel_bench proj  TYPE M K [REPS]                 Project, 1 row; TYPE = q4k|q5k|q6k|q8_0
//   kernel_bench suite                                 every case listed in the task brief
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

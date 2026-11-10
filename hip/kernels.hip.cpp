// ricursive GPU elementwise and attention operators. See kernels.hpp.
//
// QkNormRopeAppendKernel and AttentionKernel are adapted from our Laguna
// port (src/models/laguna/kernels/attention.hip.cpp): ring/window addressing,
// 128-key tiles, online softmax in fp32. Written with plain IEEE arithmetic
// and no FP contraction (build with -fno-fast-math -ffp-contract=off): under
// fast math a row's result can depend on the rest of its launch.
#pragma clang fp contract(off)

#include "hip/kernels.hpp"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>

namespace gufo::models::ricursive::rocm {
namespace {

constexpr int kNormThreads = 256;
constexpr int kMaxWaves = 32;  // up to 1024 threads at wave32
constexpr std::uint32_t kEltThreads = 256;
constexpr std::uint32_t kEltMaxBlocks = 1U << 20;

bool Fail(std::string* error, const std::string& what) {
  if (error != nullptr) {
    *error = what;
  }
  return false;
}

bool Launched(std::string* error, const char* name) {
  const hipError_t status = hipGetLastError();
  if (status == hipSuccess) {
    return true;
  }
  return Fail(error, std::string(name) + ": " + hipGetErrorString(status));
}

// ---- device helpers --------------------------------------------------------

__device__ __forceinline__ float WaveSum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}
__device__ __forceinline__ float WaveMax(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset));
  }
  return v;
}
__device__ __forceinline__ float Sigmoid(float x) {
  return 1.0F / (1.0F + expf(-x));
}

/// Sum of x[i]^2 over i < dim, in a fixed order, returned to every thread.
/// Must be reached by the whole block.
__device__ float BlockSumSquares(const float* __restrict__ x,
                                 std::uint32_t dim) {
  __shared__ float part[kMaxWaves];
  float ss = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = x[i];
    ss += v * v;
  }
  ss = WaveSum(ss);
  if (threadIdx.x % warpSize == 0) {
    part[threadIdx.x / warpSize] = ss;
  }
  __syncthreads();
  float total = part[0];
  const int waves = (static_cast<int>(blockDim.x) + warpSize - 1) / warpSize;
  for (int i = 1; i < waves; ++i) {
    total += part[i];
  }
  __syncthreads();
  return total;
}

__device__ __forceinline__ void StoreCache(__half* p, float v) {
  *p = __float2half(v);
}
__device__ __forceinline__ void StoreCache(float* p, float v) {
  *p = v;
}
__device__ __forceinline__ float LoadCache(const __half* p) {
  return __half2float(*p);
}
__device__ __forceinline__ float LoadCache(const float* p) {
  return *p;
}
/// Elements 8c .. 8c+7 of a cache row as fp32.
__device__ __forceinline__ void LoadKeys8(const __half* row, int c,
                                          float (&kf)[8]) {
  const uint4 raw = reinterpret_cast<const uint4*>(row)[c];
  const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 f = __half22float2(h2[e]);
    kf[2 * e] = f.x;
    kf[2 * e + 1] = f.y;
  }
}
__device__ __forceinline__ void LoadKeys8(const float* row, int c,
                                          float (&kf)[8]) {
  const float4 a = reinterpret_cast<const float4*>(row)[2 * c];
  const float4 b = reinterpret_cast<const float4*>(row)[2 * c + 1];
  kf[0] = a.x; kf[1] = a.y; kf[2] = a.z; kf[3] = a.w;
  kf[4] = b.x; kf[5] = b.y; kf[6] = b.z; kf[7] = b.w;
}

// ---- norms and elementwise kernels ------------------------------------------

/// grid (rows), block kNormThreads.
__global__ void RmsNormKernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* __restrict__ out, std::uint32_t dim,
                              float eps) {
  const std::size_t base = std::size_t{blockIdx.x} * dim;
  const float total = BlockSumSquares(x + base, dim);
  const float scale = 1.0F / sqrtf(total / static_cast<float>(dim) + eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    out[base + i] = (x[base + i] * scale) * w[i];
  }
}

/// grid (rows), block kNormThreads. x is updated in place.
__global__ void ResidualPostNormKernel(float* __restrict__ x,
                                       const float* __restrict__ branch,
                                       const float* __restrict__ w,
                                       std::uint32_t dim, float eps) {
  const std::size_t base = std::size_t{blockIdx.x} * dim;
  const float total = BlockSumSquares(branch + base, dim);
  const float scale = 1.0F / sqrtf(total / static_cast<float>(dim) + eps);
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    x[base + i] += (branch[base + i] * scale) * w[i];
  }
}

__global__ void ScaleRowsKernel(const float* in, float* out, std::size_t n,
                                float scale) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    out[i] = in[i] * scale;
  }
}

__global__ void ValueMixKernel(const float* x0, const float* a,
                               const float* lambda, float* out,
                               std::size_t n) {
  const float s = Sigmoid(*lambda);
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    const float base = x0[i];
    out[i] = base + s * (a[i] - base);
  }
}

__global__ void SigmoidGateKernel(const float* attn, const float* gate,
                                  float* out, std::size_t n) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    out[i] = attn[i] * Sigmoid(gate[i]);
  }
}

__global__ void ClampedSwiGluKernel(const float* gate, const float* up,
                                    float* out, std::size_t n, float c) {
  for (std::size_t i = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
       i < n; i += std::size_t{gridDim.x} * blockDim.x) {
    float g = gate[i];
    float u = up[i];
    // Comparisons rather than fminf/fmaxf: those return the non-NaN operand
    // and would turn a NaN into the clamp value. Here NaN stays NaN.
    g = g > c ? c : g;
    u = u > c ? c : (u < -c ? -c : u);
    out[i] = (g * Sigmoid(g)) * u;
  }
}

/// grid (n_ids), block 256.
__global__ void GatherRowsKernel(const float* src, const std::int32_t* ids,
                                 float* dst, std::uint32_t src_rows,
                                 std::uint32_t dim) {
  const std::int32_t id = ids[blockIdx.x];
  const bool ok = id >= 0 && static_cast<std::uint32_t>(id) < src_rows;
  for (std::uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
    dst[std::size_t{blockIdx.x} * dim + i] =
        ok ? src[std::size_t{static_cast<std::uint32_t>(id)} * dim + i] : NAN;
  }
}

// ---- q/k norm, RoPE, KV append ----------------------------------------------

/// grid (tokens, heads + 2 * kv_heads), block 128 (one thread per dimension).
/// Block y < heads: query head (norm, RoPE, in place). Next kv_heads blocks:
/// key heads (norm, RoPE, into the cache). Last kv_heads blocks: value heads
/// (no norm, into the cache). Norm: y = (x * rsqrt(mean(x^2) + eps)) * gamma.
template<class CacheT>
__global__ void QkNormRopeAppendKernel(
    float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ q_gamma,
    const float* __restrict__ k_gamma, CacheT* __restrict__ k_cache,
    CacheT* __restrict__ v_cache, std::uint32_t heads, std::uint32_t kv_heads,
    float eps, std::uint32_t start_pos, std::uint32_t capacity,
    std::uint32_t rotary_dim, double theta) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t b = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::uint32_t pos = start_pos + t;
  const std::size_t kv_width = std::size_t{kv_heads} * kHeadDim;
  const std::size_t slot = RingSlot(pos, capacity);
  if (b >= heads + kv_heads) {
    const std::uint32_t h = b - heads - kv_heads;
    StoreCache(&v_cache[slot * kv_width + h * kHeadDim + d],
               v[std::size_t{t} * kv_width + h * kHeadDim + d]);
    return;
  }
  const bool is_q = b < heads;
  const std::uint32_t h = is_q ? b : b - heads;
  float* qrow = q + (std::size_t{t} * heads + h) * kHeadDim;
  const float* krow = k + std::size_t{t} * kv_width + h * kHeadDim;
  const float x = is_q ? qrow[d] : krow[d];
  __shared__ float part[kMaxWaves];
  __shared__ float normed[kHeadDim];
  float ss = WaveSum(x * x);
  if (d % warpSize == 0) {
    part[d / warpSize] = ss;
  }
  __syncthreads();
  float total = part[0];
  for (int i = 1; i < static_cast<int>(kHeadDim) / warpSize; ++i) {
    total += part[i];
  }
  const float mean = total / static_cast<float>(kHeadDim);
  const float scale = 1.0F / sqrtf(mean + eps);
  normed[d] = (x * scale) * (is_q ? q_gamma[d] : k_gamma[d]);
  __syncthreads();
  float out = normed[d];
  if (rotary_dim != 0) {
    const std::uint32_t pair = RopePair(d, rotary_dim);
    const double angle = RopeAngle(pos, pair, rotary_dim, theta);
    const float c = static_cast<float>(cos(angle));
    const float s = static_cast<float>(sin(angle));
    const float xa = normed[pair];
    const float xb = normed[pair + rotary_dim / 2];
    out = d < rotary_dim / 2 ? xa * c - xb * s : xa * s + xb * c;
  }
  if (is_q) {
    qrow[d] = out;
  } else {
    StoreCache(&k_cache[slot * kv_width + h * kHeadDim + d], out);
  }
}

// ---- attention ---------------------------------------------------------------

/// grid (tokens, kv_heads), block 128. A block serves the kGroup query heads
/// of one KV head for one row at position p. Keys stream in tiles of 128
/// from the first visible key: thread t scores key t of the tile for every
/// head (K row from the cache, q from LDS), the tile's max and sum are
/// reduced over the block, then thread d adds the tile's V column d with
/// online-softmax rescaling. Tile boundaries depend only on p and the window
/// (kernels_math.hpp), so a row's sum order is independent of the chunking.
template<int kGroup, class CacheT>
__global__ void AttentionKernel(const float* __restrict__ q,
                                const CacheT* __restrict__ k_cache,
                                const CacheT* __restrict__ v_cache,
                                float* __restrict__ out, std::uint32_t heads,
                                std::uint32_t kv_heads, std::uint32_t window,
                                std::uint32_t start_pos, std::uint32_t capacity,
                                float scale) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t kvh = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::uint32_t p = start_pos + t;
  const std::uint32_t first = FirstVisibleKey(p, window);
  const std::size_t kv_width = std::size_t{kv_heads} * kHeadDim;
  __shared__ float qs[kGroup][kHeadDim];
  __shared__ float probs[kGroup][kHeadDim];
  __shared__ float red[kMaxWaves][kGroup];
  const int wave = d / warpSize;
  const int lane = d % warpSize;
  const int waves = kHeadDim / warpSize;
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    qs[g][d] = q[(std::size_t{t} * heads + kvh * kGroup + g) * kHeadDim + d];
  }
  __syncthreads();
  float m[kGroup];
  float l[kGroup];
  float acc[kGroup];
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.0F;
    acc[g] = 0.0F;
  }
  const std::uint32_t tiles = TileCount(first, p);
  for (std::uint32_t tile = 0; tile < tiles; ++tile) {
    const std::uint32_t ts = TileStart(first, tile);
    const std::uint32_t j = ts + d;
    const bool valid = j <= p;
    float s[kGroup];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      s[g] = 0.0F;
    }
    if (valid) {
      const CacheT* krow =
          k_cache + std::size_t{RingSlot(j, capacity)} * kv_width +
          kvh * kHeadDim;
      for (int c = 0; c < static_cast<int>(kHeadDim) / 8; ++c) {
        float kf[8];
        LoadKeys8(krow, c, kf);
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            s[g] = fmaf(qs[g][c * 8 + e], kf[e], s[g]);
          }
        }
      }
    }
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      s[g] = valid ? s[g] * scale : -INFINITY;
    }
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      const float w = WaveMax(s[g]);
      if (lane == 0) {
        red[wave][g] = w;
      }
    }
    __syncthreads();
    float alpha[kGroup];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      float tile_max = red[0][g];
      for (int i = 1; i < waves; ++i) {
        tile_max = fmaxf(tile_max, red[i][g]);
      }
      const float m_new = fmaxf(m[g], tile_max);
      alpha[g] = expf(m[g] - m_new);
      m[g] = m_new;
      s[g] = valid ? expf(s[g] - m_new) : 0.0F;
      probs[g][d] = s[g];
    }
    __syncthreads();  // red is rewritten below
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      const float w = WaveSum(s[g]);
      if (lane == 0) {
        red[wave][g] = w;
      }
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      float tile_sum = red[0][g];
      for (int i = 1; i < waves; ++i) {
        tile_sum += red[i][g];
      }
      l[g] = l[g] * alpha[g] + tile_sum;
      acc[g] *= alpha[g];
    }
    const std::uint32_t count = TileKeys(ts, p);
    for (std::uint32_t jj = 0; jj < count; ++jj) {
      const float vv = LoadCache(&v_cache[std::size_t{RingSlot(ts + jj, capacity)} *
                                              kv_width +
                                          kvh * kHeadDim + d]);
#pragma unroll
      for (int g = 0; g < kGroup; ++g) {
        acc[g] = fmaf(probs[g][jj], vv, acc[g]);
      }
    }
    __syncthreads();  // probs and red are rewritten by the next tile
  }
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    out[(std::size_t{t} * heads + kvh * kGroup + g) * kHeadDim + d] =
        acc[g] / l[g];
  }
}

template<int kGroup, class CacheT>
void LaunchAttention(const float* q, const void* k, const void* v, float* out,
                     std::uint32_t heads, std::uint32_t kv_heads,
                     std::uint32_t window, std::uint32_t start_pos,
                     std::uint32_t tokens, std::uint32_t capacity, float scale,
                     hipStream_t stream) {
  AttentionKernel<kGroup, CacheT><<<dim3(tokens, kv_heads), kHeadDim, 0, stream>>>(
      q, static_cast<const CacheT*>(k), static_cast<const CacheT*>(v), out,
      heads, kv_heads, window, start_pos, capacity, scale);
}

template<class CacheT>
bool DispatchAttention(std::uint32_t group, const float* q, const void* k,
                       const void* v, float* out, std::uint32_t heads,
                       std::uint32_t kv_heads, std::uint32_t window,
                       std::uint32_t start_pos, std::uint32_t tokens,
                       std::uint32_t capacity, float scale,
                       hipStream_t stream) {
#define RICURSIVE_ATTN_CASE(G)                                              \
  case G:                                                                   \
    LaunchAttention<G, CacheT>(q, k, v, out, heads, kv_heads, window,       \
                               start_pos, tokens, capacity, scale, stream); \
    return true;
  switch (group) {
    RICURSIVE_ATTN_CASE(1)
    RICURSIVE_ATTN_CASE(2)
    RICURSIVE_ATTN_CASE(10)
    default:
      return false;
  }
#undef RICURSIVE_ATTN_CASE
}

// ---- launch helpers -----------------------------------------------------------

std::uint32_t EltBlocks(std::size_t n) {
  const std::size_t blocks = (n + kEltThreads - 1) / kEltThreads;
  return static_cast<std::uint32_t>(std::min<std::size_t>(blocks, kEltMaxBlocks));
}

bool CheckRowsDim(std::uint32_t dim, std::string* error, const char* name) {
  if (dim == 0) {
    return Fail(error, std::string(name) + ": dim must be > 0");
  }
  return true;
}

}  // namespace

bool RmsNorm(const float* x, const float* weight, float* out,
             std::uint32_t rows, std::uint32_t dim, float eps,
             hipStream_t stream, std::string* error) {
  if (!CheckRowsDim(dim, error, "RmsNorm")) return false;
  if (rows == 0) return true;
  if (x == nullptr || weight == nullptr || out == nullptr) {
    return Fail(error, "RmsNorm: null pointer");
  }
  RmsNormKernel<<<rows, kNormThreads, 0, stream>>>(x, weight, out, dim, eps);
  return Launched(error, "RmsNorm");
}

bool ResidualPostNorm(float* x, const float* branch, const float* weight,
                      std::uint32_t rows, std::uint32_t dim, float eps,
                      hipStream_t stream, std::string* error) {
  if (!CheckRowsDim(dim, error, "ResidualPostNorm")) return false;
  if (rows == 0) return true;
  if (x == nullptr || branch == nullptr || weight == nullptr) {
    return Fail(error, "ResidualPostNorm: null pointer");
  }
  if (x == branch) {
    return Fail(error, "ResidualPostNorm: x and branch must not alias");
  }
  ResidualPostNormKernel<<<rows, kNormThreads, 0, stream>>>(x, branch, weight,
                                                            dim, eps);
  return Launched(error, "ResidualPostNorm");
}

bool ScaleRows(const float* in, float* out, std::uint32_t rows,
               std::uint32_t dim, float scale, hipStream_t stream,
               std::string* error) {
  const std::size_t n = std::size_t{rows} * dim;
  if (n == 0) return true;
  if (in == nullptr || out == nullptr) {
    return Fail(error, "ScaleRows: null pointer");
  }
  ScaleRowsKernel<<<EltBlocks(n), kEltThreads, 0, stream>>>(in, out, n, scale);
  return Launched(error, "ScaleRows");
}

bool ValueMix(const float* x0, const float* a, const float* lambda, float* out,
              std::uint32_t rows, std::uint32_t dim, hipStream_t stream,
              std::string* error) {
  const std::size_t n = std::size_t{rows} * dim;
  if (n == 0) return true;
  if (x0 == nullptr || a == nullptr || lambda == nullptr || out == nullptr) {
    return Fail(error, "ValueMix: null pointer");
  }
  ValueMixKernel<<<EltBlocks(n), kEltThreads, 0, stream>>>(x0, a, lambda, out, n);
  return Launched(error, "ValueMix");
}

bool QkNormRopeAppend(float* q, const float* k, const float* v,
                      const float* q_gamma, const float* k_gamma,
                      void* k_cache, void* v_cache, KvType type,
                      std::uint32_t tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, float eps,
                      std::uint32_t start_pos, std::uint32_t capacity,
                      std::uint32_t rotary_dim, float theta,
                      hipStream_t stream, std::string* error) {
  if (heads == 0 || kv_heads == 0 || heads % kv_heads != 0) {
    return Fail(error, "QkNormRopeAppend: heads must be a positive multiple of kv_heads");
  }
  if (rotary_dim != 0 && rotary_dim != kHeadDim) {
    return Fail(error, "QkNormRopeAppend: rotary_dim must be 0 or 128");
  }
  if (rotary_dim != 0 && !(std::isfinite(theta) && theta > 1.0F)) {
    return Fail(error, "QkNormRopeAppend: theta must be finite and > 1");
  }
  if (capacity == 0 || tokens > capacity) {
    return Fail(error, "QkNormRopeAppend: capacity must be >= tokens");
  }
  if (std::uint64_t{start_pos} + tokens > (1ULL << 31)) {
    return Fail(error, "QkNormRopeAppend: position out of range");
  }
  if (tokens == 0) return true;
  if (q == nullptr || k == nullptr || v == nullptr || q_gamma == nullptr ||
      k_gamma == nullptr || k_cache == nullptr || v_cache == nullptr) {
    return Fail(error, "QkNormRopeAppend: null pointer");
  }
  const dim3 grid(tokens, heads + 2 * kv_heads);
  if (type == KvType::kF16) {
    QkNormRopeAppendKernel<__half><<<grid, kHeadDim, 0, stream>>>(
        q, k, v, q_gamma, k_gamma, static_cast<__half*>(k_cache),
        static_cast<__half*>(v_cache), heads, kv_heads, eps, start_pos,
        capacity, rotary_dim, static_cast<double>(theta));
  } else {
    QkNormRopeAppendKernel<float><<<grid, kHeadDim, 0, stream>>>(
        q, k, v, q_gamma, k_gamma, static_cast<float*>(k_cache),
        static_cast<float*>(v_cache), heads, kv_heads, eps, start_pos,
        capacity, rotary_dim, static_cast<double>(theta));
  }
  return Launched(error, "QkNormRopeAppend");
}

bool Attention128(const float* q, const void* k_cache, const void* v_cache,
                  KvType type, float* out, std::uint32_t tokens,
                  std::uint32_t heads, std::uint32_t kv_heads,
                  std::uint32_t window, std::uint32_t start_pos,
                  std::uint32_t capacity, float scale, hipStream_t stream,
                  std::string* error) {
  if (heads == 0 || kv_heads == 0 || heads % kv_heads != 0) {
    return Fail(error, "Attention128: heads must be a positive multiple of kv_heads");
  }
  const std::uint32_t group = heads / kv_heads;
  if (group != 1 && group != 2 && group != 10) {
    return Fail(error, "Attention128: group (heads / kv_heads) must be 1, 2 or 10");
  }
  if (!std::isfinite(scale)) {
    return Fail(error, "Attention128: scale must be finite");
  }
  if (std::uint64_t{start_pos} + tokens > (1ULL << 31)) {
    return Fail(error, "Attention128: position out of range");
  }
  if (capacity < MinCapacity(start_pos, tokens, window)) {
    return Fail(error, "Attention128: capacity < MinCapacity(start_pos, tokens, window)");
  }
  if (tokens == 0) return true;
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr ||
      out == nullptr) {
    return Fail(error, "Attention128: null pointer");
  }
  const bool ok =
      type == KvType::kF16
          ? DispatchAttention<__half>(group, q, k_cache, v_cache, out, heads,
                                      kv_heads, window, start_pos, tokens,
                                      capacity, scale, stream)
          : DispatchAttention<float>(group, q, k_cache, v_cache, out, heads,
                                     kv_heads, window, start_pos, tokens,
                                     capacity, scale, stream);
  if (!ok) return Fail(error, "Attention128: unsupported group");
  return Launched(error, "Attention128");
}

bool SigmoidGate(const float* attn, const float* gate, float* out,
                 std::uint32_t rows, std::uint32_t dim, hipStream_t stream,
                 std::string* error) {
  const std::size_t n = std::size_t{rows} * dim;
  if (n == 0) return true;
  if (attn == nullptr || gate == nullptr || out == nullptr) {
    return Fail(error, "SigmoidGate: null pointer");
  }
  SigmoidGateKernel<<<EltBlocks(n), kEltThreads, 0, stream>>>(attn, gate, out, n);
  return Launched(error, "SigmoidGate");
}

bool ClampedSwiGlu(const float* gate, const float* up, float* out,
                   std::uint32_t rows, std::uint32_t dim, float clamp,
                   hipStream_t stream, std::string* error) {
  if (!(std::isfinite(clamp) && clamp > 0.0F)) {
    return Fail(error, "ClampedSwiGlu: clamp must be finite and > 0");
  }
  const std::size_t n = std::size_t{rows} * dim;
  if (n == 0) return true;
  if (gate == nullptr || up == nullptr || out == nullptr) {
    return Fail(error, "ClampedSwiGlu: null pointer");
  }
  ClampedSwiGluKernel<<<EltBlocks(n), kEltThreads, 0, stream>>>(gate, up, out, n,
                                                                clamp);
  return Launched(error, "ClampedSwiGlu");
}

bool GatherRows(const float* src, const std::int32_t* ids, float* dst,
                std::uint32_t n_ids, std::uint32_t src_rows, std::uint32_t dim,
                hipStream_t stream, std::string* error) {
  if (n_ids == 0 || dim == 0) return true;
  if (src == nullptr || ids == nullptr || dst == nullptr) {
    return Fail(error, "GatherRows: null pointer");
  }
  GatherRowsKernel<<<n_ids, kEltThreads, 0, stream>>>(src, ids, dst, src_rows, dim);
  return Launched(error, "GatherRows");
}

}  // namespace gufo::models::ricursive::rocm

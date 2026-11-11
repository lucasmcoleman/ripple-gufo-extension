// GPU operators of the DFlash2 drafter. See hip/drafter_kernels.hpp.
//
// Adapted from gufo's Qwen DFlash kernels, src/families/qwen/kernels/dflash_kernels.hip at gufo
// commit 0e2de43 (MIT, Copyright (c) 2026 gufo contributors; THIRD_PARTY.md lists every change):
//   DraftDynamicConv   dflash_grouped_dynamic_conv_kernel   + `accumulate`
//   DraftSiluMul       dflash_silu_mul_kernel               unchanged
//   DraftSelect        dflash_selector_partial_topk_kernel and FinalizeSelector: top-k 32, greedy
//                      only, codebooks decoded element-wise (F32, BF16, Q8_0, Q4_K) instead of BF16
//   DraftAttention     dflash_noncausal_attention_*         replaced by the online-softmax tile
//                      loop of this extension's AttentionKernel (hip/kernels.hip.cpp), because the
//                      vendored kernels keep one fp32 score per key in LDS (65.6 KB for the 16,400
//                      keys of the full layer: more than a workgroup may have)
//   DraftQkNormRope   written for this extension from QkNormRopeAppendKernel (any head_dim, float32
//                      rotary frequencies as the drafter reference, ring slot start)
// Written with plain IEEE arithmetic and no FP contraction, like hip/kernels.hip.cpp.
// Not run on a GPU yet (docs/DRAFTER-S2.md lists where it differs from the host operations).
#pragma clang fp contract(off)

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>

#include "hip/drafter_kernels.hpp"
#include "hip/drafter_math.hpp"

namespace gufo::models::ricursive::rocm {
namespace {

constexpr int kMaxWaves = 32;  // up to 1024 threads at wave32

bool Fail(std::string* error, const std::string& what) {
  if (error != nullptr) *error = what;
  return false;
}

bool Launched(std::string* error, const char* name) {
  const hipError_t status = hipGetLastError();
  if (status == hipSuccess) return true;
  return Fail(error, std::string(name) + ": " + hipGetErrorString(status));
}

__device__ __forceinline__ float WaveSum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) v += __shfl_xor(v, offset);
  return v;
}
__device__ __forceinline__ float WaveMax(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor(v, offset));
  return v;
}

__device__ __forceinline__ void StoreCache(__half* p, float v) { *p = __float2half(v); }
__device__ __forceinline__ void StoreCache(float* p, float v) { *p = v; }
__device__ __forceinline__ float LoadCache(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ float LoadCache(const float* p) { return *p; }
/// Elements 8c .. 8c+7 of a cache row as fp32.
__device__ __forceinline__ void LoadKeys8(const __half* row, int c, float (&kf)[8]) {
  const uint4 raw = reinterpret_cast<const uint4*>(row)[c];
  const __half2* h2 = reinterpret_cast<const __half2*>(&raw);
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    const float2 f = __half22float2(h2[e]);
    kf[2 * e] = f.x;
    kf[2 * e + 1] = f.y;
  }
}
__device__ __forceinline__ void LoadKeys8(const float* row, int c, float (&kf)[8]) {
  const float4 a = reinterpret_cast<const float4*>(row)[2 * c];
  const float4 b = reinterpret_cast<const float4*>(row)[2 * c + 1];
  kf[0] = a.x; kf[1] = a.y; kf[2] = a.z; kf[3] = a.w;
  kf[4] = b.x; kf[5] = b.y; kf[6] = b.z; kf[7] = b.w;
}

// ---- q/k norm, rotary, store -----------------------------------------------------------------

/// grid (tokens, heads + 2 * kv_heads), block head_dim (one thread per dimension). Block index
/// b < heads: a query head (norm, rotary, in place); the next kv_heads: a key head (norm, rotary,
/// into k_store); the last kv_heads: a value head (into v_store). Norm: y = (x * (1 / sqrt(mean(x^2)
/// + eps))) * gamma; rotary: NeoX half split, inv = 1 / theta^(2 pair / head_dim),
/// angle = float(position) * inv, cosf / sinf (the drafter reference's float32 arithmetic).
template<class CacheT>
__global__ void DraftQkNormRopeKernel(float* __restrict__ q, const float* __restrict__ k,
                                      const float* __restrict__ v,
                                      const float* __restrict__ q_gamma,
                                      const float* __restrict__ k_gamma,
                                      CacheT* __restrict__ k_store, CacheT* __restrict__ v_store,
                                      std::uint32_t heads, std::uint32_t kv_heads,
                                      std::uint32_t head_dim, float eps, std::uint32_t start_pos,
                                      std::uint32_t slot_start, std::uint32_t capacity, bool rope,
                                      float theta) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t b = blockIdx.y;
  const std::uint32_t d = threadIdx.x;
  const std::size_t kv_width = std::size_t{kv_heads} * head_dim;
  const std::size_t slot = (std::size_t{slot_start} + t) % capacity;
  if (b >= heads + kv_heads) {
    const std::uint32_t h = b - heads - kv_heads;
    StoreCache(&v_store[slot * kv_width + h * head_dim + d],
               v[std::size_t{t} * kv_width + h * head_dim + d]);
    return;
  }
  const bool is_q = b < heads;
  const std::uint32_t h = is_q ? b : b - heads;
  float* qrow = q + (std::size_t{t} * heads + h) * head_dim;
  const float* krow = k + std::size_t{t} * kv_width + h * head_dim;
  const float x = is_q ? qrow[d] : krow[d];
  __shared__ float part[kMaxWaves];
  __shared__ float normed[kDraftMaxHeadDim];
  const float ss = WaveSum(x * x);
  if (d % warpSize == 0) part[d / warpSize] = ss;
  __syncthreads();
  float total = part[0];
  for (int i = 1; i < static_cast<int>(head_dim) / warpSize; ++i) total += part[i];
  const float mean = total / static_cast<float>(head_dim);
  const float scale = 1.0F / sqrtf(mean + eps);
  normed[d] = (x * scale) * (is_q ? q_gamma[d] : k_gamma[d]);
  __syncthreads();
  float out = normed[d];
  if (rope) {
    const std::uint32_t half = head_dim / 2;
    const std::uint32_t pair = d < half ? d : d - half;
    const float inv = 1.0F / powf(theta, static_cast<float>(2 * pair) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(start_pos + t) * inv;
    const float c = cosf(angle), s = sinf(angle);
    const float xa = normed[pair], xb = normed[pair + half];
    out = d < half ? xa * c - xb * s : xb * c + xa * s;
  }
  if (is_q) {
    qrow[d] = out;
  } else {
    StoreCache(&k_store[slot * kv_width + h * head_dim + d], out);
  }
}

// ---- attention ---------------------------------------------------------------------------------

/// One segment of keys of a row: the keys with indices [begin, end), key j in slot j % capacity of
/// the cache (positions for the context ring, block row numbers for the block's own rows). Keys
/// stream in tiles of blockDim.x from `begin`: thread t scores key t of the tile for every head of
/// the block (K row from the cache, q from LDS), the tile's max and sum are reduced over the block,
/// then thread d adds the tile's V column d with online-softmax rescaling (as AttentionKernel).
template<int kGroup, class CacheT>
__device__ void DraftSegment(const CacheT* __restrict__ k_cache, const CacheT* __restrict__ v_cache,
                             std::uint32_t capacity, std::uint32_t begin, std::uint32_t end,
                             std::uint32_t kvh, std::uint32_t kv_width, std::uint32_t head_dim,
                             float scale, float (&qs)[kGroup][kDraftMaxHeadDim],
                             float (&probs)[kGroup][kDraftMaxHeadDim],
                             float (&red)[kMaxWaves][kGroup], float (&m)[kGroup],
                             float (&l)[kGroup], float (&acc)[kGroup]) {
  const std::uint32_t d = threadIdx.x;
  const int wave = d / warpSize;
  const int lane = d % warpSize;
  const int waves = head_dim / warpSize;
  const std::uint32_t tiles = end > begin ? (end - begin + head_dim - 1) / head_dim : 0;
  for (std::uint32_t tile = 0; tile < tiles; ++tile) {
    const std::uint32_t ts = begin + tile * head_dim;
    const std::uint32_t j = ts + d;
    const bool valid = j < end;
    float s[kGroup];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) s[g] = 0.0F;
    if (valid) {
      const CacheT* krow = k_cache + std::size_t{j % capacity} * kv_width + kvh * head_dim;
      for (int c = 0; c < static_cast<int>(head_dim) / 8; ++c) {
        float kf[8];
        LoadKeys8(krow, c, kf);
#pragma unroll
        for (int g = 0; g < kGroup; ++g) {
#pragma unroll
          for (int e = 0; e < 8; ++e) s[g] = fmaf(qs[g][c * 8 + e], kf[e], s[g]);
        }
      }
    }
#pragma unroll
    for (int g = 0; g < kGroup; ++g) s[g] = valid ? s[g] * scale : -INFINITY;
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      const float w = WaveMax(s[g]);
      if (lane == 0) red[wave][g] = w;
    }
    __syncthreads();
    float alpha[kGroup];
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      float tile_max = red[0][g];
      for (int i = 1; i < waves; ++i) tile_max = fmaxf(tile_max, red[i][g]);
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
      if (lane == 0) red[wave][g] = w;
    }
    __syncthreads();
#pragma unroll
    for (int g = 0; g < kGroup; ++g) {
      float tile_sum = red[0][g];
      for (int i = 1; i < waves; ++i) tile_sum += red[i][g];
      l[g] = l[g] * alpha[g] + tile_sum;
      acc[g] *= alpha[g];
    }
    const std::uint32_t count = end - ts < head_dim ? end - ts : head_dim;
#pragma unroll 16
    for (std::uint32_t jj = 0; jj < count; ++jj) {
      const float vv = LoadCache(&v_cache[std::size_t{(ts + jj) % capacity} * kv_width + kvh * head_dim + d]);
#pragma unroll
      for (int g = 0; g < kGroup; ++g) acc[g] = fmaf(probs[g][jj], vv, acc[g]);
    }
    __syncthreads();  // probs and red are rewritten by the next tile
  }
}

/// grid (tokens, heads / kGroup), block head_dim. A block serves kGroup consecutive query heads
/// (all in one KV head) of one block row; the row's keys are the context rows of the ring, then the
/// block's own rows (DraftVisibleKeys, hip/drafter_math.hpp).
template<int kGroup, class CacheT>
__global__ void DraftAttentionKernel(const float* __restrict__ q,
                                     const CacheT* __restrict__ k_ctx,
                                     const CacheT* __restrict__ v_ctx, std::uint32_t ctx_capacity,
                                     std::uint32_t ctx_first, std::uint32_t anchor,
                                     const float* __restrict__ k_blk,
                                     const float* __restrict__ v_blk, float* __restrict__ out,
                                     std::uint32_t tokens, std::uint32_t heads,
                                     std::uint32_t kv_heads, std::uint32_t head_dim,
                                     std::uint32_t window, float scale) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t group = heads / kv_heads;
  const std::uint32_t qh0 = blockIdx.y * kGroup;
  const std::uint32_t kvh = qh0 / group;
  const std::uint32_t d = threadIdx.x;
  const std::uint32_t kv_width = kv_heads * head_dim;
  const DraftKeys keys = DraftVisibleKeys(t, anchor, ctx_first, tokens, window);
  __shared__ float qs[kGroup][kDraftMaxHeadDim];
  __shared__ float probs[kGroup][kDraftMaxHeadDim];
  __shared__ float red[kMaxWaves][kGroup];
#pragma unroll
  for (int g = 0; g < kGroup; ++g) qs[g][d] = q[(std::size_t{t} * heads + qh0 + g) * head_dim + d];
  __syncthreads();
  float m[kGroup], l[kGroup], acc[kGroup];
#pragma unroll
  for (int g = 0; g < kGroup; ++g) {
    m[g] = -INFINITY;
    l[g] = 0.0F;
    acc[g] = 0.0F;
  }
  DraftSegment<kGroup, CacheT>(k_ctx, v_ctx, ctx_capacity, keys.ctx_begin, keys.ctx_end, kvh,
                               kv_width, head_dim, scale, qs, probs, red, m, l, acc);
  DraftSegment<kGroup, float>(k_blk, v_blk, tokens, keys.blk_begin, keys.blk_end, kvh, kv_width,
                              head_dim, scale, qs, probs, red, m, l, acc);
#pragma unroll
  for (int g = 0; g < kGroup; ++g)
    out[(std::size_t{t} * heads + qh0 + g) * head_dim + d] = acc[g] / l[g];
}

// ---- dynamic convolution and SiLU * up (vendored) -----------------------------------------------

__global__ void DraftDynamicConvKernel(const float* __restrict__ input,
                                       const float* __restrict__ dynamic_coefficients,
                                       const float* __restrict__ base_kernel,
                                       float* __restrict__ output, std::uint32_t num_tokens,
                                       std::uint32_t hidden_size, std::uint32_t kernel_size,
                                       std::uint32_t group_size, std::uint32_t side,
                                       bool accumulate) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::uint64_t total = static_cast<std::uint64_t>(num_tokens) * hidden_size;
  if (index >= total) return;

  const std::uint32_t token = index / hidden_size;
  const std::uint32_t channel = index % hidden_size;
  const std::uint32_t groups = hidden_size / group_size;
  const std::uint32_t group = channel / group_size;
  const std::uint32_t coefficients_per_token = 2U * kernel_size * groups;

  float sum = 0.0F;
  for (std::uint32_t tap = 0; tap < kernel_size; ++tap) {
    if (token < tap) continue;
    const std::uint64_t dynamic_index = static_cast<std::uint64_t>(token) * coefficients_per_token +
                                        (side * kernel_size + tap) * groups + group;
    const std::uint64_t base_index = static_cast<std::uint64_t>(side * kernel_size + tap) * hidden_size + channel;
    const std::uint64_t input_index = static_cast<std::uint64_t>(token - tap) * hidden_size + channel;
    sum += (base_kernel[base_index] + dynamic_coefficients[dynamic_index]) * input[input_index];
  }
  output[index] = accumulate ? output[index] + sum : sum;
}

__global__ void DraftSiluMulKernel(float* __restrict__ gate, const float* __restrict__ up,
                                   std::uint32_t total_elements) {
  const std::uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;
  const float g = gate[idx];
  const float silu_g = g / (1.0F + expf(-g));
  gate[idx] = silu_g * up[idx];
}

// ---- candidate selector (vendored, top-k up to 32, greedy) ---------------------------------------

__device__ __forceinline__ void InsertCandidate(float score, std::int32_t id, float* scores,
                                                std::int32_t* ids, std::uint32_t top_k) {
  if (score <= scores[top_k - 1U]) return;
  std::uint32_t insertion = top_k - 1U;
  while (insertion > 0U && score > scores[insertion - 1U]) {
    scores[insertion] = scores[insertion - 1U];
    ids[insertion] = ids[insertion - 1U];
    --insertion;
  }
  scores[insertion] = score;
  ids[insertion] = id;
}

/// Tree-merges the per-thread descending top-k lists held in shared memory into slot zero.
__device__ void MergeLists(float* scores, std::int32_t* ids, std::uint32_t top_k) {
  for (std::uint32_t stride = kDraftSelectThreads / 2U; stride > 0U; stride >>= 1U) {
    __syncthreads();
    if (threadIdx.x < stride) {
      float* mine_scores = scores + (threadIdx.x * kDraftMaxTopK);
      std::int32_t* mine_ids = ids + (threadIdx.x * kDraftMaxTopK);
      const float* other_scores = scores + ((threadIdx.x + stride) * kDraftMaxTopK);
      const std::int32_t* other_ids = ids + ((threadIdx.x + stride) * kDraftMaxTopK);
      float merged_scores[kDraftMaxTopK];
      std::int32_t merged_ids[kDraftMaxTopK];
      std::uint32_t mine = 0;
      std::uint32_t other = 0;
      for (std::uint32_t index = 0; index < top_k; ++index) {
        const bool take_mine =
            other >= top_k || (mine < top_k && mine_scores[mine] >= other_scores[other]);
        if (take_mine) {
          merged_scores[index] = mine_scores[mine];
          merged_ids[index] = mine_ids[mine];
          ++mine;
        } else {
          merged_scores[index] = other_scores[other];
          merged_ids[index] = other_ids[other];
          ++other;
        }
      }
      for (std::uint32_t index = 0; index < top_k; ++index) {
        mine_scores[index] = merged_scores[index];
        mine_ids[index] = merged_ids[index];
      }
    }
  }
  __syncthreads();
}

/// grid (partials, rows), block kDraftSelectThreads: the top-k of one block of
/// kDraftSelectTokensPerBlock vocabulary entries of one logits row.
__global__ void DraftPartialTopKKernel(const float* __restrict__ logits,
                                       float* __restrict__ partial_scores,
                                       std::int32_t* __restrict__ partial_ids,
                                       std::uint32_t vocab_size, std::uint32_t top_k) {
  logits += std::uint64_t(blockIdx.y) * vocab_size;
  partial_scores += std::uint64_t(blockIdx.y) * gridDim.x * top_k;
  partial_ids += std::uint64_t(blockIdx.y) * gridDim.x * top_k;
  __shared__ float shared_scores[kDraftSelectThreads * kDraftMaxTopK];
  __shared__ std::int32_t shared_ids[kDraftSelectThreads * kDraftMaxTopK];
  float local_scores[kDraftMaxTopK];
  std::int32_t local_ids[kDraftMaxTopK];
#pragma unroll
  for (std::uint32_t index = 0; index < kDraftMaxTopK; ++index) {
    local_scores[index] = -INFINITY;
    local_ids[index] = 0;
  }
  const std::uint32_t begin = blockIdx.x * kDraftSelectTokensPerBlock;
  const std::uint32_t requested_end = begin + kDraftSelectTokensPerBlock;
  const std::uint32_t end = requested_end < vocab_size ? requested_end : vocab_size;
  for (std::uint32_t token = begin + threadIdx.x; token < end; token += blockDim.x)
    InsertCandidate(logits[token], static_cast<std::int32_t>(token), local_scores, local_ids, top_k);
  const std::uint32_t shared_offset = threadIdx.x * kDraftMaxTopK;
#pragma unroll
  for (std::uint32_t index = 0; index < kDraftMaxTopK; ++index) {
    shared_scores[shared_offset + index] = local_scores[index];
    shared_ids[shared_offset + index] = local_ids[index];
  }
  MergeLists(shared_scores, shared_ids, top_k);
  const std::uint32_t output_offset = blockIdx.x * top_k;
  for (std::uint32_t index = threadIdx.x; index < top_k; index += blockDim.x) {
    partial_scores[output_offset + index] = shared_scores[index];
    partial_ids[output_offset + index] = shared_ids[index];
  }
}

/// grid 1, block kDraftSelectThreads: the rows one after the other; each merges its partial lists,
/// scores the transitions of the candidates (kDraftTransitionLanes threads per candidate), and
/// thread zero takes the best as the next row's predecessor.
__global__ void DraftFinalizeKernel(const float* __restrict__ partial_scores,
                                    const std::int32_t* __restrict__ partial_ids,
                                    const float* __restrict__ projected_hidden,
                                    const void* __restrict__ predecessor_codebook, Enc predecessor_enc,
                                    std::uint64_t predecessor_row_bytes,
                                    const void* __restrict__ successor_codebook, Enc successor_enc,
                                    std::uint64_t successor_row_bytes, std::int32_t* tokens,
                                    float* __restrict__ candidate_scores,
                                    std::int32_t* __restrict__ candidate_ids, std::uint32_t rows,
                                    std::uint32_t partial_count, std::uint32_t selector_rank,
                                    std::uint32_t top_k) {
  __shared__ float shared_scores[kDraftSelectThreads * kDraftMaxTopK];
  __shared__ std::int32_t shared_ids[kDraftSelectThreads * kDraftMaxTopK];
  __shared__ float shared_transition[kDraftMaxTopK * kDraftTransitionLanes];
  __shared__ std::int32_t predecessor;
  if (threadIdx.x == 0U) predecessor = tokens[0];
  __syncthreads();
  for (std::uint32_t row = 0; row < rows; ++row) {
    float local_scores[kDraftMaxTopK];
    std::int32_t local_ids[kDraftMaxTopK];
#pragma unroll
    for (std::uint32_t index = 0; index < kDraftMaxTopK; ++index) {
      local_scores[index] = -INFINITY;
      local_ids[index] = 0;
    }
    const std::uint64_t base = std::uint64_t(row) * partial_count * top_k;
    const std::uint32_t candidate_count = partial_count * top_k;
    for (std::uint32_t index = threadIdx.x; index < candidate_count; index += blockDim.x)
      InsertCandidate(partial_scores[base + index], partial_ids[base + index], local_scores,
                      local_ids, top_k);
    const std::uint32_t shared_offset = threadIdx.x * kDraftMaxTopK;
#pragma unroll
    for (std::uint32_t index = 0; index < kDraftMaxTopK; ++index) {
      shared_scores[shared_offset + index] = local_scores[index];
      shared_ids[shared_offset + index] = local_ids[index];
    }
    MergeLists(shared_scores, shared_ids, top_k);

    const auto* predecessor_row = static_cast<const std::uint8_t*>(predecessor_codebook) +
                                  std::uint64_t(static_cast<std::uint32_t>(predecessor)) * predecessor_row_bytes;
    {
      const std::uint32_t candidate_index = threadIdx.x / kDraftTransitionLanes;
      const std::uint32_t part = threadIdx.x % kDraftTransitionLanes;
      if (candidate_index < top_k) {
        const std::int32_t candidate = shared_ids[candidate_index];
        const auto* successor_row = static_cast<const std::uint8_t*>(successor_codebook) +
                                    std::uint64_t(static_cast<std::uint32_t>(candidate)) * successor_row_bytes;
        float partial = 0.0F;
        for (std::uint32_t rank = part; rank < selector_rank; rank += kDraftTransitionLanes)
          partial += DequantAt(predecessor_enc, predecessor_row, rank) *
                     projected_hidden[std::uint64_t(row) * selector_rank + rank] *
                     DequantAt(successor_enc, successor_row, rank);
        shared_transition[candidate_index * kDraftTransitionLanes + part] = partial;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
      float best_score = -INFINITY;
      std::uint32_t best_index = 0;
      for (std::uint32_t candidate_index = 0; candidate_index < top_k; ++candidate_index) {
        float transition = 0.0F;
        for (std::uint32_t part = 0; part < kDraftTransitionLanes; ++part)
          transition += shared_transition[candidate_index * kDraftTransitionLanes + part];
        const float score = shared_scores[candidate_index] + transition;
        candidate_scores[std::uint64_t(row) * top_k + candidate_index] = score;
        candidate_ids[std::uint64_t(row) * top_k + candidate_index] = shared_ids[candidate_index];
        if (score > best_score) {
          best_score = score;
          best_index = candidate_index;
        }
      }
      tokens[1U + row] = shared_ids[best_index];
      predecessor = shared_ids[best_index];
    }
    __syncthreads();  // the predecessor, and the shared lists, are rewritten by the next row
  }
}

std::uint32_t EltBlocks(std::uint64_t n) { return static_cast<std::uint32_t>((n + 255U) / 256U); }

template<int kGroup, class CacheT>
void LaunchAttention(const float* q, const void* k_ctx, const void* v_ctx, std::uint32_t ctx_capacity,
                     std::uint32_t ctx_first, std::uint32_t anchor, const float* k_blk,
                     const float* v_blk, float* out, std::uint32_t tokens, std::uint32_t heads,
                     std::uint32_t kv_heads, std::uint32_t head_dim, std::uint32_t window,
                     float scale, hipStream_t stream) {
  DraftAttentionKernel<kGroup, CacheT><<<dim3(tokens, heads / kGroup), head_dim, 0, stream>>>(
      q, static_cast<const CacheT*>(k_ctx), static_cast<const CacheT*>(v_ctx), ctx_capacity,
      ctx_first, anchor, k_blk, v_blk, out, tokens, heads, kv_heads, head_dim, window, scale);
}

}  // namespace

bool DraftQkNormRope(float* q, const float* k, const float* v, const float* q_gamma,
                     const float* k_gamma, void* k_store, void* v_store, KvType type,
                     std::uint32_t tokens, std::uint32_t heads, std::uint32_t kv_heads,
                     std::uint32_t head_dim, float eps, std::uint32_t start_pos,
                     std::uint32_t slot_start, std::uint32_t capacity, bool rope, float theta,
                     hipStream_t stream, std::string* error) {
  if (head_dim == 0 || head_dim % 32 != 0 || head_dim > kDraftMaxHeadDim)
    return Fail(error, "DraftQkNormRope: head_dim must be a multiple of 32 up to 256");
  if (kv_heads == 0 || (heads != 0 && heads % kv_heads != 0))
    return Fail(error, "DraftQkNormRope: heads must be 0 or a multiple of kv_heads");
  if (rope && !(std::isfinite(theta) && theta > 1.0F))
    return Fail(error, "DraftQkNormRope: theta must be finite and > 1");
  if (capacity == 0 || tokens > capacity)
    return Fail(error, "DraftQkNormRope: capacity must be >= tokens");
  if (std::uint64_t{start_pos} + tokens > (1ULL << 31))
    return Fail(error, "DraftQkNormRope: position out of range");
  if (tokens == 0) return true;
  if ((heads != 0 && q == nullptr) || k == nullptr || v == nullptr || q_gamma == nullptr ||
      k_gamma == nullptr || k_store == nullptr || v_store == nullptr)
    return Fail(error, "DraftQkNormRope: null pointer");
  const dim3 grid(tokens, heads + 2 * kv_heads);
  if (type == KvType::kF16) {
    DraftQkNormRopeKernel<__half><<<grid, head_dim, 0, stream>>>(
        q, k, v, q_gamma, k_gamma, static_cast<__half*>(k_store), static_cast<__half*>(v_store),
        heads, kv_heads, head_dim, eps, start_pos, slot_start, capacity, rope, theta);
  } else {
    DraftQkNormRopeKernel<float><<<grid, head_dim, 0, stream>>>(
        q, k, v, q_gamma, k_gamma, static_cast<float*>(k_store), static_cast<float*>(v_store),
        heads, kv_heads, head_dim, eps, start_pos, slot_start, capacity, rope, theta);
  }
  return Launched(error, "DraftQkNormRope");
}

bool DraftAttention(const float* q, const void* k_ctx, const void* v_ctx, KvType type,
                    std::uint32_t ctx_capacity, std::uint32_t ctx_first, std::uint32_t anchor,
                    const float* k_blk, const float* v_blk, float* out, std::uint32_t tokens,
                    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t head_dim,
                    std::uint32_t window, float scale, hipStream_t stream, std::string* error) {
  if (head_dim == 0 || head_dim % 32 != 0 || head_dim > kDraftMaxHeadDim)
    return Fail(error, "DraftAttention: head_dim must be a multiple of 32 up to 256");
  if (heads == 0 || kv_heads == 0 || heads % kv_heads != 0)
    return Fail(error, "DraftAttention: heads must be a positive multiple of kv_heads");
  if (!std::isfinite(scale)) return Fail(error, "DraftAttention: scale must be finite");
  if (ctx_capacity == 0) return Fail(error, "DraftAttention: ctx_capacity is 0");
  if (ctx_first > anchor) return Fail(error, "DraftAttention: ctx_first is above the anchor");
  if (std::uint64_t{anchor} + tokens > (1ULL << 31))
    return Fail(error, "DraftAttention: position out of range");
  for (std::uint32_t t = 0; t < tokens; ++t) {
    const DraftKeys keys = DraftVisibleKeys(t, anchor, ctx_first, tokens, window);
    if (keys.ctx_end - keys.ctx_begin > ctx_capacity)
      return Fail(error, "DraftAttention: the visible context rows do not fit the ring");
  }
  if (tokens == 0) return true;
  if (q == nullptr || k_ctx == nullptr || v_ctx == nullptr || k_blk == nullptr || v_blk == nullptr ||
      out == nullptr)
    return Fail(error, "DraftAttention: null pointer");
  const std::uint32_t group = heads / kv_heads;
  const bool two = group % 2 == 0;  // two query heads per block when the group allows it, else one
  if (type == KvType::kF16) {
    if (two) LaunchAttention<2, __half>(q, k_ctx, v_ctx, ctx_capacity, ctx_first, anchor, k_blk, v_blk, out, tokens, heads, kv_heads, head_dim, window, scale, stream);
    else LaunchAttention<1, __half>(q, k_ctx, v_ctx, ctx_capacity, ctx_first, anchor, k_blk, v_blk, out, tokens, heads, kv_heads, head_dim, window, scale, stream);
  } else {
    if (two) LaunchAttention<2, float>(q, k_ctx, v_ctx, ctx_capacity, ctx_first, anchor, k_blk, v_blk, out, tokens, heads, kv_heads, head_dim, window, scale, stream);
    else LaunchAttention<1, float>(q, k_ctx, v_ctx, ctx_capacity, ctx_first, anchor, k_blk, v_blk, out, tokens, heads, kv_heads, head_dim, window, scale, stream);
  }
  return Launched(error, "DraftAttention");
}

bool DraftDynamicConv(const float* input, const float* dynamic_coefficients,
                      const float* base_kernel, float* output, std::uint32_t num_tokens,
                      std::uint32_t hidden_size, std::uint32_t kernel_size,
                      std::uint32_t group_size, std::uint32_t side, bool accumulate,
                      hipStream_t stream, std::string* error) {
  if (hidden_size == 0 || kernel_size == 0 || group_size == 0 || hidden_size % group_size != 0)
    return Fail(error, "DraftDynamicConv: hidden_size must be a positive multiple of group_size and kernel_size > 0");
  if (side > 1) return Fail(error, "DraftDynamicConv: side must be 0 or 1");
  if (output == input) return Fail(error, "DraftDynamicConv: output must not alias input");
  const std::uint64_t total = std::uint64_t{num_tokens} * hidden_size;
  if (total == 0) return true;
  if (input == nullptr || dynamic_coefficients == nullptr || base_kernel == nullptr || output == nullptr)
    return Fail(error, "DraftDynamicConv: null pointer");
  DraftDynamicConvKernel<<<EltBlocks(total), 256, 0, stream>>>(
      input, dynamic_coefficients, base_kernel, output, num_tokens, hidden_size, kernel_size,
      group_size, side, accumulate);
  return Launched(error, "DraftDynamicConv");
}

bool DraftSiluMul(float* gate, const float* up, std::uint32_t total_elements, hipStream_t stream,
                  std::string* error) {
  if (total_elements == 0) return true;
  if (gate == nullptr || up == nullptr) return Fail(error, "DraftSiluMul: null pointer");
  DraftSiluMulKernel<<<EltBlocks(total_elements), 256, 0, stream>>>(gate, up, total_elements);
  return Launched(error, "DraftSiluMul");
}

bool DraftSelect(const float* logits, const float* projected_hidden, const void* predecessor,
                 Enc predecessor_enc, const void* successor, Enc successor_enc,
                 std::int32_t* tokens, float* candidate_scores, std::int32_t* candidate_ids,
                 float* partial_scores, std::int32_t* partial_ids, std::uint32_t rows,
                 std::uint32_t vocab_size, std::uint32_t selector_rank,
                 std::uint32_t selector_top_k, hipStream_t stream, std::string* error) {
  if (selector_top_k == 0 || selector_top_k > kDraftMaxTopK || selector_top_k > vocab_size)
    return Fail(error, "DraftSelect: selector_top_k must be in [1, 32] and <= vocab_size");
  if (selector_rank == 0) return Fail(error, "DraftSelect: selector_rank is 0");
  const std::uint64_t prow = DequantRowBytes(predecessor_enc, selector_rank);
  const std::uint64_t srow = DequantRowBytes(successor_enc, selector_rank);
  if (prow == 0 || srow == 0)
    return Fail(error, "DraftSelect: codebook type must be F32, BF16, Q8_0 or Q4_K with a block aligned rank");
  if (rows == 0) return true;
  if (logits == nullptr || projected_hidden == nullptr || predecessor == nullptr || successor == nullptr ||
      tokens == nullptr || candidate_scores == nullptr || candidate_ids == nullptr ||
      partial_scores == nullptr || partial_ids == nullptr)
    return Fail(error, "DraftSelect: null pointer");
  const std::uint32_t partials = DraftSelectPartials(vocab_size);
  DraftPartialTopKKernel<<<dim3(partials, rows), kDraftSelectThreads, 0, stream>>>(
      logits, partial_scores, partial_ids, vocab_size, selector_top_k);
  DraftFinalizeKernel<<<1, kDraftSelectThreads, 0, stream>>>(
      partial_scores, partial_ids, projected_hidden, predecessor, predecessor_enc, prow, successor,
      successor_enc, srow, tokens, candidate_scores, candidate_ids, rows, partials, selector_rank,
      selector_top_k);
  return Launched(error, "DraftSelect");
}

}  // namespace gufo::models::ricursive::rocm

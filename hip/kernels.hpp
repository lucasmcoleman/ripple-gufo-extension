// ricursive (Ripple-65B) GPU elementwise and attention operators.
//
// Plain launch functions over raw device pointers: no gufo header, no state.
// Every function enqueues work on `stream` and returns true when the
// arguments were valid and the launch was accepted; it does not synchronize.
// On false nothing was launched and `*error` (when not null) says why.
// All float buffers are row-major fp32. "In place" means output and input may
// be the same pointer (each thread reads an element before writing it); any
// other overlap is undefined. Each launch is independent of every other
// launch: a row's result never depends on how many rows are in the launch.
#ifndef RICURSIVE_HIP_KERNELS_HPP_
#define RICURSIVE_HIP_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

#include "hip/kernels_math.hpp"

namespace gufo::models::ricursive::rocm {

/// Element type of a layer's K and V cache.
enum class KvType {
  kF16,  // production: IEEE binary16, round to nearest even
  kF32,  // test mode: exact comparison against an fp64 reference
};

/// out[r][i] = weight[i] * x[r][i] * rsqrt(mean_i(x[r][i]^2) + eps), r < rows.
/// Statistics in fp32 with a fixed summation order. x, out: [rows][dim];
/// weight: [dim]; out may alias x. Also the per-head norm: pass
/// rows = tokens * heads, dim = 128 and the shared [128] weight.
bool RmsNorm(const float* x, const float* weight, float* out,
             std::uint32_t rows, std::uint32_t dim, float eps,
             hipStream_t stream, std::string* error = nullptr);

/// Residual update with a post-norm on the branch output:
/// x[r][i] += weight[i] * branch[r][i] * rsqrt(mean_i(branch[r][i]^2) + eps).
/// x (updated in place) and branch: [rows][dim]; branch is not modified;
/// weight: [dim]. x and branch must not overlap.
bool ResidualPostNorm(float* x, const float* branch, const float* weight,
                      std::uint32_t rows, std::uint32_t dim, float eps,
                      hipStream_t stream, std::string* error = nullptr);

/// out[r][i] = in[r][i] * scale (the embedding scale). [rows][dim]; may alias.
bool ScaleRows(const float* in, float* out, std::uint32_t rows,
               std::uint32_t dim, float scale, hipStream_t stream,
               std::string* error = nullptr);

/// Value-residual mix: out[r][i] = x0[r][i] + s * (a[r][i] - x0[r][i]) with
/// s = sigmoid(*lambda). x0 (the scaled embedding row), a (the attn-normed
/// input), out: [rows][dim]; lambda: one fp32 scalar in DEVICE memory (the
/// layer's attn_vres_lambda); out may alias x0 or a.
bool ValueMix(const float* x0, const float* a, const float* lambda, float* out,
              std::uint32_t rows, std::uint32_t dim, hipStream_t stream,
              std::string* error = nullptr);

/// Per-head RMSNorm of q and k (weights shared by all heads), optional NeoX
/// RoPE, and the append of K (after norm and RoPE) and V (raw) to the layer's
/// KV cache. Row t of the launch sits at absolute position start_pos + t.
///  q:        [tokens][heads][128] fp32, normed and rotated IN PLACE.
///  k, v:     [tokens][kv_heads][128] fp32 projections (not modified).
///  q_gamma, k_gamma: [128] fp32 norm weights.
///  k_cache, v_cache: [capacity][kv_heads][128] elements of `type`; row t is
///            written to slot (start_pos + t) % capacity.
///  eps:      norm epsilon (1e-6).
///  rotary_dim: 128 = NeoX RoPE over all 128 dims (sliding layers); 0 = no
///            rotation at all (full-attention layers). Nothing else is valid.
///  theta:    RoPE base (10000).
/// The capacity must be >= MinCapacity(start_pos, tokens, window) of the
/// matching Attention128 call (see its comment); a smaller ring is rejected
/// only by Attention128, because this call does not know the window, but it
/// does reject capacity < tokens (two rows of the chunk would share a slot).
/// Nothing but norm, optional RoPE and the store is done here: q is not
/// scaled (Attention128 takes the scale) and V is not normed.
bool QkNormRopeAppend(float* q, const float* k, const float* v,
                      const float* q_gamma, const float* k_gamma,
                      void* k_cache, void* v_cache, KvType type,
                      std::uint32_t tokens, std::uint32_t heads,
                      std::uint32_t kv_heads, float eps,
                      std::uint32_t start_pos, std::uint32_t capacity,
                      std::uint32_t rotary_dim, float theta,
                      hipStream_t stream, std::string* error = nullptr);

/// Causal attention over the KV cache for `tokens` rows at positions
/// start_pos .. start_pos + tokens - 1, head_dim 128. Query head h reads KV
/// head h / group, group = heads / kv_heads, which must be 1, 2 or 10.
///  q:        [tokens][heads][128] fp32 (after QkNormRopeAppend).
///  k_cache, v_cache: [capacity][kv_heads][128] of `type`; key j is read from
///            slot j % capacity. The keys of every row of this launch must
///            already be in the cache.
///  out:      [tokens][heads][128] fp32 (= [tokens][heads * 128]), must not
///            overlap q.
///  window:   0 = full attention; otherwise key j is visible to position p iff
///            p - (window - 1) <= j <= p (the window counts the query).
///  capacity: ring rows; must be >= MinCapacity(start_pos, tokens, window)
///            (window + tokens - 1 for a window, start_pos + tokens for full).
///  scale:    applied to q.k (1/sqrt(128)).
/// Online softmax and the output accumulation are fp32; the sum order of a
/// row depends only on its position and the window, so chunking rows into
/// several launches, or one row per launch, gives identical bits.
bool Attention128(const float* q, const void* k_cache, const void* v_cache,
                  KvType type, float* out, std::uint32_t tokens,
                  std::uint32_t heads, std::uint32_t kv_heads,
                  std::uint32_t window, std::uint32_t start_pos,
                  std::uint32_t capacity, float scale, hipStream_t stream,
                  std::string* error = nullptr);

/// out[i] = attn[i] * sigmoid(gate[i]) over [rows][dim] (dim = 10240), fp32.
/// out may alias attn or gate.
bool SigmoidGate(const float* attn, const float* gate, float* out,
                 std::uint32_t rows, std::uint32_t dim, hipStream_t stream,
                 std::string* error = nullptr);

/// out = silu(min(g, c)) * clamp(u, -c, c), elementwise over [rows][dim]:
/// the gate is clamped from above only, the up value on both sides. `clamp`
/// must be finite and > 0, otherwise the call returns false and launches
/// nothing: a clamp of 0 ("no clamp" in some configs) or a non-finite value
/// would otherwise silently change the model's numbers, so the caller has to
/// decide explicitly what a layer without a clamp means. NaN inputs give NaN
/// outputs. out may alias gate or up.
bool ClampedSwiGlu(const float* gate, const float* up, float* out,
                   std::uint32_t rows, std::uint32_t dim, float clamp,
                   hipStream_t stream, std::string* error = nullptr);

/// dst[i][:] = src[ids[i]][:] for i < n_ids (picking logit rows). src:
/// [src_rows][dim]; ids: int32 in DEVICE memory; dst: [n_ids][dim], must not
/// overlap src. An id outside [0, src_rows) fills that dst row with NaN (the
/// ids are on the device, so they cannot be rejected at launch).
bool GatherRows(const float* src, const std::int32_t* ids, float* dst,
                std::uint32_t n_ids, std::uint32_t src_rows, std::uint32_t dim,
                hipStream_t stream, std::string* error = nullptr);

}  // namespace gufo::models::ricursive::rocm

#endif  // RICURSIVE_HIP_KERNELS_HPP_

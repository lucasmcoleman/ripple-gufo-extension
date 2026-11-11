// GPU operators of the DFlash2 drafter (docs/DRAFTER-S2.md): launch functions over raw device
// pointers, like hip/kernels.hpp (same conventions: enqueue on `stream`, return true when the
// arguments were valid and the launch was accepted, `*error` says why not). Adapted from gufo's
// Qwen DFlash kernels (THIRD_PARTY.md). The host versions of the same operations are in
// hip/ops_host.cpp; hip/ops.hpp documents each operation.
#ifndef RICURSIVE_HIP_DRAFTER_KERNELS_HPP_
#define RICURSIVE_HIP_DRAFTER_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

#include "hip/dequant_math.hpp"
#include "hip/kernels.hpp"

namespace gufo::models::ricursive::rocm {

bool DraftQkNormRope(float* q, const float* k, const float* v, const float* q_gamma,
                     const float* k_gamma, void* k_store, void* v_store, KvType type,
                     std::uint32_t tokens, std::uint32_t heads, std::uint32_t kv_heads,
                     std::uint32_t head_dim, float eps, std::uint32_t start_pos,
                     std::uint32_t slot_start, std::uint32_t capacity, bool rope, float theta,
                     hipStream_t stream, std::string* error = nullptr);

bool DraftAttention(const float* q, const void* k_ctx, const void* v_ctx, KvType type,
                    std::uint32_t ctx_capacity, std::uint32_t ctx_first, std::uint32_t anchor,
                    const float* k_blk, const float* v_blk, float* out, std::uint32_t tokens,
                    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t head_dim,
                    std::uint32_t window, float scale, hipStream_t stream,
                    std::string* error = nullptr);

/// The second attention kernel ("tiled", docs/DRAFTER-S2.md "Stage 2c"): the same operation as DraftAttention, each K/V
/// tile read once for all the queries of its K/V head. head_dim 32, 64 or 128; 2 x tokens x (heads / kv_heads) <= 1024.
/// `scratch`: device floats, rocm::DraftTiledScratchFloats(tokens, heads, head_dim) of them (hip/drafter_math.hpp).
bool DraftAttentionTiled(const float* q, const void* k_ctx, const void* v_ctx, KvType type,
                         std::uint32_t ctx_capacity, std::uint32_t ctx_first, std::uint32_t anchor,
                         const float* k_blk, const float* v_blk, float* out, std::uint32_t tokens,
                         std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t head_dim,
                         std::uint32_t window, float scale, float* scratch, hipStream_t stream,
                         std::string* error = nullptr);

bool DraftDynamicConv(const float* input, const float* dynamic_coefficients,
                      const float* base_kernel, float* output, std::uint32_t num_tokens,
                      std::uint32_t hidden_size, std::uint32_t kernel_size,
                      std::uint32_t group_size, std::uint32_t side, bool accumulate,
                      hipStream_t stream, std::string* error = nullptr);

bool DraftSiluMul(float* gate, const float* up, std::uint32_t total_elements, hipStream_t stream,
                  std::string* error = nullptr);

/// `predecessor` and `successor`: device pointers of [vocab_size][selector_rank] codebooks of the
/// encodings `predecessor_enc` / `successor_enc`.
bool DraftSelect(const float* logits, const float* projected_hidden, const void* predecessor,
                 Enc predecessor_enc, const void* successor, Enc successor_enc,
                 std::int32_t* tokens, float* candidate_scores, std::int32_t* candidate_ids,
                 float* partial_scores, std::int32_t* partial_ids, std::uint32_t rows,
                 std::uint32_t vocab_size, std::uint32_t selector_rank,
                 std::uint32_t selector_top_k, hipStream_t stream, std::string* error = nullptr);

}  // namespace gufo::models::ricursive::rocm

#endif  // RICURSIVE_HIP_DRAFTER_KERNELS_HPP_

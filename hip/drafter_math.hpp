// Index math and constants of the drafter's operations, shared by the host operations
// (hip/ops_host.cpp) and the GPU kernels (hip/drafter_kernels.hip.cpp), like
// hip/kernels_math.hpp is for the target's: the kernels call these functions, so the host
// test of this file tests the code the GPU runs.
#ifndef RICURSIVE_HIP_DRAFTER_MATH_HPP_
#define RICURSIVE_HIP_DRAFTER_MATH_HPP_

#include <cstdint>

#include "hip/kernels_math.hpp"

namespace gufo::models::ricursive::rocm {

/// Largest head dimension of the drafter's attention and norm kernels (one thread per dimension).
inline constexpr std::uint32_t kDraftMaxHeadDim = 256;
/// Candidate selector (gufo's kernel caps it at 16; Ripple's drafter needs 32).
inline constexpr std::uint32_t kDraftMaxTopK = 32;
inline constexpr std::uint32_t kDraftSelectThreads = 128;
inline constexpr std::uint32_t kDraftSelectTokensPerBlock = 2048;
/// Threads that cooperate on one candidate's low-rank transition score.
inline constexpr std::uint32_t kDraftTransitionLanes = kDraftSelectThreads / kDraftMaxTopK;

/// The selector's total order, shared by the host operations and the kernels (docs/DRAFTER-S2.md, "Stage 2b"): a
/// candidate (score, id) ranks above (score2, id2) iff its score is larger, or the scores are equal and its token id is
/// lower. Used for the top-k lists (by logit) and for the final choice (by score): equal values always resolve to the
/// lowest token id, whatever order the lists were merged in, as DrafterReference does. A NaN ranks above nothing.
RICURSIVE_HD bool DraftRanksAbove(float score, std::int32_t id, float score2, std::int32_t id2) {
  return score > score2 || (score == score2 && id < id2);
}

/// Bytes of the tiled Q8_1 activation scratch of the 8-bit projection route for `rows` rows of `cols` inputs (the layout of
/// hip/ggml/prefill_quant_gemm.hpp: 16-token tiles x 32-element blocks x (576 payload + 64 sum sidecar) bytes). The executor's
/// memory plan and the host operations' model of the scratch use this; the GPU's real allocation is
/// ggml::QuantizedActivationBytes (a gate check compares the two: DrafterSessionBytes == DeviceBytes on the GPU).
RICURSIVE_HD std::uint64_t Q8ActivationBytes(std::uint64_t rows, std::uint64_t cols) {
  return ((rows + 15) / 16) * (cols / 32) * (576 + 64);
}

/// Partial top-k lists per logits row: the vocabulary in blocks of kDraftSelectTokensPerBlock.
RICURSIVE_HD std::uint32_t DraftSelectPartials(std::uint32_t vocab_size) {
  return (vocab_size + kDraftSelectTokensPerBlock - 1) / kDraftSelectTokensPerBlock;
}

/// The keys the block row `t` (position anchor + t) of a draft attends to: context rows
/// [ctx_begin, ctx_end) (positions), then block rows [blk_begin, blk_end) (indices; row i is
/// at position anchor + i). `window` == 0: all of them; else a key at position j is visible
/// iff |p - j| < window.
struct DraftKeys {
  std::uint32_t ctx_begin, ctx_end, blk_begin, blk_end;
};
RICURSIVE_HD DraftKeys DraftVisibleKeys(std::uint32_t t, std::uint32_t anchor,
                                        std::uint32_t ctx_first, std::uint32_t tokens,
                                        std::uint32_t window) {
  const std::uint32_t p = anchor + t;
  DraftKeys k;
  k.ctx_end = anchor;
  k.ctx_begin = ctx_first;
  if (window != 0 && p + 1 > window && p + 1 - window > k.ctx_begin) k.ctx_begin = p + 1 - window;
  if (k.ctx_begin > k.ctx_end) k.ctx_begin = k.ctx_end;
  k.blk_begin = (window == 0 || t + 1 <= window) ? 0U : t + 1 - window;
  k.blk_end = window == 0 || t + window > tokens ? tokens : t + window;
  return k;
}

}  // namespace gufo::models::ricursive::rocm

#endif  // RICURSIVE_HIP_DRAFTER_MATH_HPP_

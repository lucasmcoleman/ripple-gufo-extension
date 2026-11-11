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

// Pure index and angle math shared by the ricursive GPU kernels and their
// host-only tests. Everything here compiles for the host and the device (no
// HIP headers needed on the host), and the kernels call these same functions,
// so the host test of this file tests the code the GPU runs.
#ifndef RICURSIVE_HIP_KERNELS_MATH_HPP_
#define RICURSIVE_HIP_KERNELS_MATH_HPP_

#include <math.h>

#include <cstdint>

#if defined(__HIPCC__)
#define RICURSIVE_HD __host__ __device__ inline
#else
#define RICURSIVE_HD inline
#endif

namespace gufo::models::ricursive::rocm {

/// Head dimension of every attention head (q, k and v alike).
inline constexpr std::uint32_t kHeadDim = 128;
/// Keys per attention tile; also the thread count of the attention block.
inline constexpr std::uint32_t kKeyTile = 128;

/// Cache slot of absolute position `pos` in a ring of `capacity` rows.
RICURSIVE_HD std::uint32_t RingSlot(std::uint32_t pos, std::uint32_t capacity) {
  return pos % capacity;
}

/// First key position visible to the query at position `p`. `window` counts
/// keys including the query itself; `window == 0` means full attention.
/// Visible keys are exactly FirstVisibleKey(p, window) <= j <= p.
RICURSIVE_HD std::uint32_t FirstVisibleKey(std::uint32_t p,
                                           std::uint32_t window) {
  return (window == 0 || p + 1 <= window) ? 0U : p + 1 - window;
}

/// Whether key `j` is visible to the query at position `p`.
RICURSIVE_HD bool KeyVisible(std::uint32_t j, std::uint32_t p,
                             std::uint32_t window) {
  return j <= p && j >= FirstVisibleKey(p, window);
}

/// The attention walks the visible keys [first, p] in tiles of kKeyTile keys
/// that start at `first`, so the tiles of a row depend on p and the window
/// only, never on how rows were chunked into launches.
RICURSIVE_HD std::uint32_t TileCount(std::uint32_t first, std::uint32_t p) {
  return (p - first) / kKeyTile + 1;
}
RICURSIVE_HD std::uint32_t TileStart(std::uint32_t first, std::uint32_t tile) {
  return first + tile * kKeyTile;
}
/// Keys of the tile that starts at `tile_start` (the last tile is short).
RICURSIVE_HD std::uint32_t TileKeys(std::uint32_t tile_start, std::uint32_t p) {
  const std::uint32_t rest = p - tile_start + 1;
  return rest < kKeyTile ? rest : kKeyTile;
}

/// KV head read by query head `h` when `group` query heads share one KV head.
RICURSIVE_HD std::uint32_t KvHeadOf(std::uint32_t h, std::uint32_t group) {
  return h / group;
}

/// Smallest ring capacity a launch of `tokens` rows starting at `start_pos`
/// needs: the chunk's keys are all written before any row attends, so the
/// oldest key the first row can see must survive the chunk's last write.
/// Full attention (window 0) never overwrites: it needs every position.
RICURSIVE_HD std::uint64_t MinCapacity(std::uint32_t start_pos,
                                       std::uint32_t tokens,
                                       std::uint32_t window) {
  if (tokens == 0) {
    return 1;
  }
  return window == 0 ? std::uint64_t{start_pos} + tokens
                     : std::uint64_t{window} + tokens - 1;
}

// ---- NeoX RoPE: dimension d pairs with d +- rotary_dim / 2 ------------------

/// Pair (frequency) index of dimension d: d and d + rotary_dim/2 share it.
RICURSIVE_HD std::uint32_t RopePair(std::uint32_t d, std::uint32_t rotary_dim) {
  const std::uint32_t half = rotary_dim / 2;
  return d < half ? d : d - half;
}
/// The other member of d's rotation pair.
RICURSIVE_HD std::uint32_t RopePartner(std::uint32_t d,
                                       std::uint32_t rotary_dim) {
  const std::uint32_t half = rotary_dim / 2;
  return d < half ? d + half : d - half;
}
/// Rotation angle (radians) of pair `pair` at absolute position `pos`:
/// pos * theta^(-2 pair / rotary_dim). Evaluated in double so that angles at
/// positions in the thousands are not limited by float rounding (a float
/// angle at position 3000 is off by up to 1.2e-4 rad); the result is only
/// narrowed after cos/sin.
RICURSIVE_HD double RopeAngle(std::uint32_t pos, std::uint32_t pair,
                              std::uint32_t rotary_dim, double theta) {
  return static_cast<double>(pos) *
         pow(theta, -2.0 * static_cast<double>(pair) /
                        static_cast<double>(rotary_dim));
}

}  // namespace gufo::models::ricursive::rocm

#endif  // RICURSIVE_HIP_KERNELS_MATH_HPP_

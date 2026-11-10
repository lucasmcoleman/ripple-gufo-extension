// Host-only test (no GPU, no HIP): the index and angle math of
// hip/kernels_math.hpp against straightforward definitions, exhaustively over
// windows 1, 2, 8, 16, 1024, ring capacities around the window, and positions
// up to several capacities. The kernels call these same functions.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "hip/kernels_math.hpp"
#include "tests/check.hpp"

namespace k = gufo::models::ricursive::rocm;
using ricursive_test::Require;

namespace {

std::uint64_t g_checks = 0;

void Eq(std::uint64_t got, std::uint64_t want, const std::string& what) {
  ++g_checks;
  Require(got == want, what + ": got " + std::to_string(got) + ", want " +
                           std::to_string(want));
}

const std::uint32_t kWindows[] = {1, 2, 8, 16, 1024};

// Visibility, first visible key, tiles. Definition: j visible to p iff
// j <= p and (window == 0 or p - j < window).
void TestVisibility() {
  for (std::uint32_t window : {0U, 1U, 2U, 8U, 16U, 1024U}) {
    const std::uint32_t pmax = window == 0 ? 3000 : 5 * window + 300;
    for (std::uint32_t p = 0; p <= pmax; ++p) {
      // Brute force over every key up to p + 3 (keys after p are never visible).
      std::uint32_t lo = UINT32_MAX;
      std::uint32_t visible = 0;
      const std::uint32_t jmax = p + 3;
      for (std::uint32_t j = 0; j <= jmax; ++j) {
        const bool want = j <= p && (window == 0 || p - j < window);
        // Every key for small p; the neighbourhoods of both ends otherwise.
        const std::uint32_t f = p + 1 > window && window != 0 ? p + 1 - window : 0;
        const bool near_end = j + 4 >= p || (j + 4 >= f && j <= f + 4);
        if (p < 600 || near_end) {
          ++g_checks;
          Require(k::KeyVisible(j, p, window) == want,
                  "KeyVisible j=" + std::to_string(j) + " p=" +
                      std::to_string(p) + " w=" + std::to_string(window));
        }
        if (want) {
          lo = std::min(lo, j);
          ++visible;
        }
      }
      Eq(k::FirstVisibleKey(p, window), lo,
         "FirstVisibleKey p=" + std::to_string(p) + " w=" + std::to_string(window));
      Eq(p - k::FirstVisibleKey(p, window) + 1, visible, "visible key count");
      if (window != 0) {
        Require(visible == std::min(p + 1, window), "window size");
      }
      // Tiles: consecutive, <= 128 keys, starting at the first visible key,
      // covering [first, p] exactly once.
      const std::uint32_t first = k::FirstVisibleKey(p, window);
      std::uint32_t ts = first;
      std::uint32_t tile = 0;
      while (ts <= p) {
        Eq(k::TileStart(first, tile), ts, "TileStart");
        const std::uint32_t keys = std::min<std::uint32_t>(128, p - ts + 1);
        Eq(k::TileKeys(ts, p), keys, "TileKeys");
        ts += keys;
        ++tile;
      }
      Eq(k::TileCount(first, p), tile, "TileCount");
      Eq(ts, std::uint64_t{p} + 1, "tiles end after p");
    }
  }
}

// Ring: slot is pos mod capacity, and with capacity >= window the keys
// visible to one position have distinct slots. MinCapacity: simulate a chunk
// writing all its rows before any row attends; with MinCapacity every row
// still finds all its visible keys, with one slot fewer (window > 0, or full
// attention, and a long enough history) some row loses a key.
void TestRing() {
  for (std::uint32_t window : kWindows) {
    for (std::uint32_t capacity :
         {window, window + 1, window + 2, window + 127, 2 * window,
          2 * window + 1, window + 1024}) {
      for (std::uint32_t pos = 0; pos <= 5 * capacity + 3; ++pos) {
        Eq(k::RingSlot(pos, capacity), pos - capacity * (pos / capacity), "RingSlot");
      }
      for (std::uint32_t p = window; p < 3 * capacity + window; p += 1 + p / 50) {
        std::vector<bool> used(capacity, false);
        for (std::uint32_t j = k::FirstVisibleKey(p, window); j <= p; ++j) {
          const auto s = k::RingSlot(j, capacity);
          Require(!used[s], "visible keys share a ring slot");
          used[s] = true;
        }
        ++g_checks;
      }
    }
  }
  auto survives = [](std::uint32_t start, std::uint32_t tokens,
                     std::uint32_t window, std::uint64_t capacity) {
    // slot -> position held after the chunk wrote positions start..start+tokens-1
    // on top of a ring that held every earlier position.
    std::vector<std::int64_t> held(capacity, -1);
    for (std::uint32_t pos = 0; pos < start + tokens; ++pos) {
      held[pos % capacity] = pos;
    }
    for (std::uint32_t t = 0; t < tokens; ++t) {
      const std::uint32_t p = start + t;
      for (std::uint32_t j = k::FirstVisibleKey(p, window); j <= p; ++j) {
        if (held[j % capacity] != static_cast<std::int64_t>(j)) return false;
      }
    }
    return true;
  };
  for (std::uint32_t window : {0U, 1U, 2U, 8U, 16U, 1024U}) {
    for (std::uint32_t tokens : {1U, 2U, 3U, 7U, 16U, 33U}) {
      for (std::uint32_t start : {0U, 1U, 5U, 15U, 16U, 17U, 100U, 1023U, 1024U, 1025U, 2500U}) {
        const std::uint64_t need = k::MinCapacity(start, tokens, window);
        Require(survives(start, tokens, window, need),
                "MinCapacity too small w=" + std::to_string(window) + " start=" +
                    std::to_string(start) + " n=" + std::to_string(tokens));
        ++g_checks;
        if (need > 1 && (window == 0 || start >= window) &&
            (window != 0 || start + tokens >= 2)) {
          Require(!survives(start, tokens, window, need - 1),
                  "MinCapacity not tight w=" + std::to_string(window) + " start=" +
                      std::to_string(start) + " n=" + std::to_string(tokens));
          ++g_checks;
        }
      }
    }
  }
}

void TestGqa() {
  for (auto [heads, kv] : std::vector<std::pair<std::uint32_t, std::uint32_t>>{
           {80, 8}, {10, 1}, {20, 2}, {4, 4}, {6, 3}}) {
    const std::uint32_t group = heads / kv;
    std::vector<std::uint32_t> want(heads);
    for (std::uint32_t kvh = 0; kvh < kv; ++kvh) {
      for (std::uint32_t g = 0; g < group; ++g) want[kvh * group + g] = kvh;
    }
    for (std::uint32_t h = 0; h < heads; ++h) {
      Eq(k::KvHeadOf(h, group), want[h], "KvHeadOf");
    }
  }
  // The modulo mapping that the broken twin uses really differs.
  Require(k::KvHeadOf(9, 10) != 9 % 8 && k::KvHeadOf(10, 10) != 10 % 8, "gqa twin differs");
}

// RoPE: the pair/partner arithmetic equals np_ref's rotate_half form
// x*cos + rot*sin with rot = [-x2, x1] and the angle table pos * inv_freq
// duplicated over both halves.
void TestRope() {
  const std::uint32_t D = 128;
  const double theta = 10000.0;
  for (std::uint32_t d = 0; d < D; ++d) {
    const auto pair = k::RopePair(d, D);
    const auto partner = k::RopePartner(d, D);
    Eq(pair, d % 64, "RopePair");
    Eq(partner, (d + 64) % D, "RopePartner");
    Eq(k::RopePartner(partner, D), d, "partner involution");
    Eq(k::RopePair(partner, D), pair, "partner shares the frequency");
  }
  std::vector<int> count(64, 0);
  for (std::uint32_t d = 0; d < D; ++d) ++count[k::RopePair(d, D)];
  for (int c : count) Require(c == 2, "each pair has two members");
  double worst = 0;
  std::uint32_t positions[] = {0, 1, 2, 63, 64, 127, 128, 1023, 1024, 1025,
                               3000, 5000, 32767, 32768, 100000};
  for (std::uint32_t pos : positions) {
    // Reference vector x (deterministic, unequal values).
    std::vector<double> x(D), want(D), got(D);
    for (std::uint32_t d = 0; d < D; ++d) x[d] = std::sin(0.37 * d + 0.11 * pos) + 0.01 * d;
    for (std::uint32_t d = 0; d < D; ++d) {
      const std::uint32_t i = d % 64;
      const long double inv = 1.0L / powl(10000.0L, 2.0L * i / D);
      const long double ang = static_cast<long double>(pos) * inv;
      const double c = static_cast<double>(cosl(ang)), s = static_cast<double>(sinl(ang));
      const double rot = d < 64 ? -x[d + 64] : x[d - 64];  // rotate_half
      want[d] = x[d] * c + rot * s;
    }
    for (std::uint32_t d = 0; d < D; ++d) {
      const std::uint32_t pair = k::RopePair(d, D);
      const double ang = k::RopeAngle(pos, pair, D, theta);
      const double c = std::cos(ang), s = std::sin(ang);
      const double xa = x[pair], xb = x[pair + 64];
      got[d] = d < 64 ? xa * c - xb * s : xa * s + xb * c;
      worst = std::max(worst, std::fabs(got[d] - want[d]) / (1.0 + pos));
      ++g_checks;
    }
  }
  std::cout << "rope vs rotate_half form (long double reference): max abs error per (1+pos) " << worst << "\n";
  Require(worst < 1e-15, "rope formula differs from rotate_half form");
  // Angle table: pair 0 is the position itself; pair 63 is pos * theta^(-126/128).
  Require(k::RopeAngle(777, 0, D, theta) == 777.0, "pair 0 angle");
  Require(std::fabs(k::RopeAngle(777, 63, D, theta) / 777.0 - std::pow(theta, -126.0 / 128.0)) < 1e-15,
          "pair 63 frequency");
  // A float angle would be off by up to ~1e-4 rad at position 3000; ours is not.
  const double exact = static_cast<double>(3000.0L * powl(10000.0L, -2.0L * 5 / 128));
  Require(std::fabs(k::RopeAngle(3000, 5, D, theta) - exact) < 1e-9 * exact, "double angle");
}

}  // namespace

int main() {
  return ricursive_test::Run([] {
    TestVisibility();
    TestRing();
    TestGqa();
    TestRope();
    std::cout << g_checks << " exhaustive checks\n";
  });
}

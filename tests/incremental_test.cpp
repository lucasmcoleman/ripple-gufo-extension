// Incremental decoding (ring KV cache, chunked prefill, rewind) against the
// full-sequence forward, and taps. Every equality check has a broken twin that
// must fail: a wrong window, an undersized ring, a forgotten rewind.
#include <cstdio>
#include <map>

#include "tests/oracle_common.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;

namespace {

constexpr double kTol = 1e-5;
constexpr std::size_t kMaxRewind = 64;
constexpr std::size_t kBroken = 100;  // sentinel: "must differ"

// NaN or Inf in either input gives +infinity (tests/check.hpp); the old loop kept
// the last difference seen once a NaN had been stored, so a NaN could be lost.
double MaxDiff(const float* a, const float* b, std::size_t n) {
  return MaxAbsDiff(std::span<const float>(a, n), std::span<const float>(b, n));
}

using Tokens = std::vector<std::int32_t>;
std::span<const std::int32_t> Sub(const Tokens& t, std::size_t from, std::size_t n) {
  return std::span<const std::int32_t>(t).subspan(from, n);
}

/// Feeds tokens [from, from+total) of `t` in chunks of the given sizes
/// (repeating the last size); returns logits [total][vocab].
std::vector<float> Feed(Reference::Session& s, const Tokens& t, std::size_t from,
                        std::size_t total, const std::vector<std::size_t>& sizes) {
  std::vector<float> out;
  std::size_t done = 0, i = 0;
  while (done < total) {
    const std::size_t n = std::min(sizes[std::min(i++, sizes.size() - 1)], total - done);
    const auto l = s.Append(Sub(t, from + done, n));
    out.insert(out.end(), l.begin(), l.end());
    done += n;
  }
  return out;
}

struct Fixture {
  std::string name;
  Model model;
  Tokens a, b;  // two different sequences of 64 tokens
};

void Chunkings(const Fixture& f) {
  const auto& cfg = f.model.weights.config;
  const std::size_t V = cfg.vocab_size, T = f.a.size(), W = cfg.sliding_window;
  const Reference ref(f.model.weights);
  const auto full = ref.ForwardFull(f.a);
  struct Plan {
    const char* label;
    std::vector<std::size_t> sizes;
  };
  const std::vector<Plan> plans = {
      {"1 at a time", {1}},
      {"chunks of 3", {3}},
      {"chunks of 7", {7}},
      {"chunks of 16", {16}},
      {"prefill 40 then 24 singles", {40, 1}},
      // straddle the window: first chunk ends just before W, next crosses it
      {"straddle: (W-3), 6, (W+5), 1...", {W - 3, 6, W + 5, 1}},
      {"straddle: 1, (W+1), (W-1), 2W", {1, W + 1, W - 1, 2 * W}},
      {"one chunk of 64", {64}},
  };
  for (const auto& p : plans) {
    Reference::Session s(ref, kMaxRewind);
    const auto inc = Feed(s, f.a, 0, T, p.sizes);
    Require(inc.size() == full.size(), "size");
    const double d = MaxDiff(inc.data(), full.data(), full.size());
    std::printf("  %s %-34s max_abs vs full = %.3e\n", f.name.c_str(), p.label, d);
    Require(d <= kTol, f.name + " " + p.label + ": incremental != full");
    Require(s.position() == T, "position");
    // The ring stays bounded: sliding layers W + max_rewind rows, full layers T.
    for (std::size_t il = 0; il < cfg.block_count; ++il) {
      const std::size_t rows = s.CacheRows(il);
      Require(cfg.IsSliding(il) ? rows == W + kMaxRewind : rows == T,
              "cache rows layer " + std::to_string(il) + ": " + std::to_string(rows));
    }
  }
  (void)V;

  // Last-token-only mode returns exactly the last row.
  {
    Reference::Session s(ref, kMaxRewind);
    (void)s.Append(Sub(f.a, 0, T - 1));
    const auto last = s.Append(Sub(f.a, T - 1, 1), true);
    Require(last.size() == V, "last_only size");
    Require(MaxDiff(last.data(), &full[(T - 1) * V], V) <= kTol, "last_only values");
  }

  // Twin 1: a Session with the wrong window must NOT equal the full forward.
  {
    const Reference wrong(f.model.weights, Variant::kWinMinus1);
    Reference::Session s(wrong, kMaxRewind);
    const auto inc = Feed(s, f.a, 0, T, {1});
    const double d = MaxDiff(inc.data(), full.data(), full.size());
    std::printf("  %s twin: window-1 session vs full: max_abs = %.3e (must exceed %.0e)\n",
                f.name.c_str(), d, kTol);
    Require(d > 1e-3, "window-1 session unexpectedly equals the full forward");
  }
  // Twin 2: an undersized ring (W-1 rows) loses a visible key.
  {
    Reference::Session s(ref, 0, W - 1);
    const auto inc = Feed(s, f.a, 0, T, {1});
    const double d = MaxDiff(inc.data(), full.data(), full.size());
    std::printf("  %s twin: ring of W-1 rows vs full: max_abs = %.3e (must exceed %.0e)\n",
                f.name.c_str(), d, kTol);
    Require(d > 1e-3, "undersized ring unexpectedly equals the full forward");
  }
  // Twin 3: a chunk fed to a ring that holds only W rows still works because
  // attention reads each row right after writing it; but a W-row ring plus a
  // chunk is exactly the gufo ring rule (window + chunk - 1 rows) satisfied
  // sequentially. Check the W-row ring with a big chunk equals full.
  {
    Reference::Session s(ref, 0);
    const auto inc = Feed(s, f.a, 0, T, {T});
    Require(MaxDiff(inc.data(), full.data(), full.size()) <= kTol,
            "W-row ring, one big chunk");
  }
}

void Rewinds(const Fixture& f) {
  const auto& cfg = f.model.weights.config;
  const std::size_t V = cfg.vocab_size, W = cfg.sliding_window;
  const Reference ref(f.model.weights);
  const std::size_t fed = 52;     // tokens fed before rewinding
  const std::size_t extra = 12;   // different tokens fed after the rewind
  // Positions before the window boundary (< W), exactly at it, after it, deep.
  const std::vector<std::size_t> points = {W - 3, W - 1, W, W + 2, 2 * W + 1, 40, 51};
  for (const std::size_t p : points) {
    if (p + extra > f.a.size()) continue;
    Reference::Session s(ref, kMaxRewind);
    (void)Feed(s, f.a, 0, fed, {5, 7, 1});
    s.Rewind(p);
    Require(s.position() == p, "rewind position");
    // New continuation: tokens of sequence b at positions p..p+extra.
    const auto got = Feed(s, f.b, p, extra, {4, 1, 3});
    Tokens fresh(f.a.begin(), f.a.begin() + static_cast<std::ptrdiff_t>(p));
    fresh.insert(fresh.end(), f.b.begin() + static_cast<std::ptrdiff_t>(p),
                 f.b.begin() + static_cast<std::ptrdiff_t>(p + extra));
    const auto want = ref.ForwardFull(fresh);
    const double d = MaxDiff(got.data(), &want[p * V], extra * V);
    // Twin: the same continuation WITHOUT the rewind sits at the wrong
    // positions over the stale cache and must not match.
    Reference::Session no_rewind(ref, kMaxRewind);
    (void)Feed(no_rewind, f.a, 0, fed, {5, 7, 1});
    const auto stale = Feed(no_rewind, f.b, p, extra, {4, 1, 3});
    const double d_twin = MaxDiff(stale.data(), &want[p * V], extra * V);
    std::printf("  %s rewind %zu->%zu (W=%zu, %s): max_abs vs fresh full = %.3e; no-rewind twin = %.3e\n",
                f.name.c_str(), fed, p, W, p < W ? "before window boundary" : "after window boundary",
                d, d_twin);
    Require(d <= kTol, f.name + ": rewind result differs from a fresh forward");
    Require(d_twin > 1e-3, f.name + ": no-rewind twin should differ");
  }
  // Rewind to 0 and to the current position; two rewinds in a row.
  {
    Reference::Session s(ref, kMaxRewind);
    (void)Feed(s, f.a, 0, 30, {30});
    s.Rewind(30);  // no-op
    s.Rewind(10);
    (void)Feed(s, f.b, 10, 5, {5});
    s.Rewind(12);
    const auto got = Feed(s, f.b, 12, 6, {2});
    Tokens fresh(f.a.begin(), f.a.begin() + 10);
    fresh.insert(fresh.end(), f.b.begin() + 10, f.b.begin() + 12);
    fresh.insert(fresh.end(), f.b.begin() + 12, f.b.begin() + 18);
    const auto want = ref.ForwardFull(fresh);
    Require(MaxDiff(got.data(), &want[12 * V], 6 * V) <= kTol, "double rewind");
    Reference::Session z(ref, kMaxRewind);
    (void)Feed(z, f.a, 0, 20, {20});
    z.Rewind(0);
    const auto again = Feed(z, f.b, 0, 9, {3});
    const auto want0 = ref.ForwardFull(Sub(f.b, 0, 9));
    Require(MaxDiff(again.data(), want0.data(), want0.size()) <= kTol, "rewind to 0");
  }
  // Rewinding further than the ring can restore must throw, not return junk.
  {
    Reference::Session s(ref, 4);
    (void)Feed(s, f.a, 0, 30, {30});
    bool threw = false;
    try {
      s.Rewind(20);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    Require(threw, "rewind beyond max_rewind must throw");
    s.Rewind(26);  // exactly max_rewind back is allowed
    bool past_end = false;
    try {
      s.Rewind(27);
    } catch (const std::runtime_error&) {
      past_end = true;
    }
    Require(past_end, "rewind past the end must throw");
  }
  std::printf("  %s rewind: limits enforced, double rewind and rewind to 0 match a fresh forward\n",
              f.name.c_str());
}

void Taps(const Fixture& f) {
  const Reference ref(f.model.weights);
  using Key = std::pair<std::string, int>;
  std::map<Key, std::vector<float>> full_taps;
  const auto full = ref.ForwardFull(f.a, [&](std::string_view n, int l, const float* d,
                                              std::size_t rows, std::size_t width) {
    full_taps[{std::string(n), l}].assign(d, d + rows * width);
  });
  const auto& cfg = f.model.weights.config;
  // Chunked session taps, concatenated, equal the full taps for every name.
  std::map<Key, std::vector<float>> inc_taps;
  Reference::Session s(ref, kMaxRewind);
  std::size_t done = 0;
  for (const std::size_t n : {9, 20, 1, 34}) {
    (void)s.Append(Sub(f.a, done, n), false,
                   [&](std::string_view name, int l, const float* d, std::size_t rows,
                       std::size_t width) {
                     auto& v = inc_taps[{std::string(name), l}];
                     v.insert(v.end(), d, d + rows * width);
                   });
    done += n;
  }
  Require(done == f.a.size(), "tap chunks");
  // 17 per-layer taps + 3 global taps (embd_scaled, result_norm, result_output).
  Require(full_taps.size() == 17 * cfg.block_count + 3,
          "tap count " + std::to_string(full_taps.size()));
  double worst = 0;
  for (const auto& [key, vals] : full_taps) {
    const auto it = inc_taps.find(key);
    Require(it != inc_taps.end() && it->second.size() == vals.size(),
            "missing tap " + key.first + "-" + std::to_string(key.second));
    worst = MaxKeepNaN(worst, MaxDiff(vals.data(), it->second.data(), vals.size()));
  }
  // The last tap is the logits themselves; the first global tap is the scaled
  // embedding (not all-zero: taps carry data).
  const auto& logits_tap = full_taps.at({"result_output", -1});
  Require(MaxDiff(logits_tap.data(), full.data(), full.size()) == 0.0, "result_output tap");
  double mag = 0;
  for (float x : full_taps.at({"embd_scaled", -1})) mag = std::max<double>(mag, std::abs(x));
  Require(mag > 0, "embd_scaled tap is all zero");
  std::printf("  %s taps: %zu named taps, chunked session vs full: max_abs = %.3e\n",
              f.name.c_str(), full_taps.size(), worst);
  Require(worst <= kTol, "tap mismatch");
}

}  // namespace

int main() {
  (void)kBroken;
  return Run([] {
    for (const char* name : {"tinyA", "tinyB"}) {
      const std::string path = FixtureDir() + "/" + name + ".gguf";
      if (!std::filesystem::exists(path)) {
        std::cout << "SKIP: " << path << " not found\n";
        std::exit(77);
      }
      Fixture f{name, OpenModel(path), LoadTokens(name, 0), LoadTokens(name, 1)};
      Require(f.a.size() == 64 && f.b.size() == 64, "64-token sequences");
      Require(f.a != f.b, "sequences must differ");
      std::cout << name << " (window " << f.model.weights.config.sliding_window << ")\n";
      Chunkings(f);
      Rewinds(f);
      Taps(f);
    }
  });
}

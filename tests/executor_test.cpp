// Executor test (docs/S2-EXECUTOR.md): the orchestration of hip/executor.cpp on
// the backend linked into this binary (host ops: tests/executor_backends_host.cpp;
// GPU: tests/executor_backends_gpu.cpp), on the tiny fixtures.
//
//   executor_test                 every invocation
//   executor_test GROUP[:FIXTURE] one invocation (or one group over its fixtures)
//   executor_test --list          invocation names, one per line
//
//   taps:F        every tap of every layer against Reference::ForwardFull (F32 KV, exact)
//   llama-f32:F   logits, F32 KV, against llama.cpp F32-KV logits (2e-3, top-1 100%)
//   llama-f16:F   logits, F16 KV (production), against llama.cpp --kv-f16 (2e-2, top-1 100%)
//   chunk:F       chunked == one-shot, decode after prefill, last-row / no-row logits
//   rewind:F      rewind, then different tokens == a fresh run
//   quant:Q       quantized tinyB files against the Reference on dequantized weights
//   twins:F       orchestration-level broken twins (each must fail the llama comparison)
//   limits:F      refusals: clamp 0, context cap default, bad rows
// Exit 0 pass, 1 fail, 77 backend unavailable (no GPU).
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>

#include "tests/executor_test_backend.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;

namespace {

using Tokens = std::vector<std::int32_t>;
constexpr std::uint32_t kContext = 96;
constexpr std::uint32_t kSeq = 64;
constexpr double kLlamaF32 = 2e-3;
constexpr double kLlamaF16 = 2e-2;
constexpr double kTwinFactor = 10.0;  // a twin must exceed the pass threshold by this

ExecBackend g_backend;

struct Fixture {
  std::string name;  ///< file stem, e.g. tinyB-Q8_0
  std::string base;  ///< token / sequence stem: tinyA or tinyB
  Model model;
  std::uint32_t W{0};
  [[nodiscard]] std::uint32_t Rows() const { return W + 4; }
  [[nodiscard]] std::size_t V() const { return model.weights.config.vocab_size; }
};

Fixture Load(const std::string& name) {
  Fixture f;
  f.name = name;
  f.base = name.substr(0, 5);
  f.model = OpenModel(FixtureDir() + "/" + name + ".gguf");
  f.W = f.model.weights.config.sliding_window;
  return f;
}

ExecutorConfig Cfg(std::uint32_t max_rows, bool f16, bool exact) {
  ExecutorConfig c;
  c.max_context = kContext;
  c.max_rows = max_rows;
  c.max_logit_rows = max_rows;
  c.kv_type = f16 ? KvType::kF16 : KvType::kF32;
  c.exact = exact;
  return c;
}

std::unique_ptr<Executor> Make(const Fixture& f, const ExecutorConfig& c,
                               const ExecFaults& faults = {},
                               bool lenient = false) {
  return g_backend.make(f.model, c, faults, lenient);
}

/// Max abs difference; NaN or a size mismatch is infinite.
double MaxAbs(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size()) return INFINITY;
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::abs(static_cast<double>(a[i]) - b[i]);
    if (!(d <= m)) m = std::isnan(d) ? INFINITY : d;
  }
  return m;
}

/// Feeds t[from, to) in chunks of `sizes` (the last repeats), all-row logits.
std::vector<float> Feed(Executor& e, const Tokens& t, std::size_t from,
                        std::size_t to, const std::vector<std::uint32_t>& sizes,
                        const Tap& tap = {}) {
  const std::size_t V = e.Info().vocab;
  std::vector<float> out((to - from) * V);
  std::size_t done = 0, i = 0;
  while (from + done < to) {
    const std::size_t n = std::min<std::size_t>(
        {sizes[std::min(i++, sizes.size() - 1)], to - from - done, e.Info().max_rows});
    e.Forward(std::span(t).subspan(from + done, n),
              static_cast<std::uint32_t>(from + done), LogitRows::kAll,
              std::span(out).subspan(done * V, n * V), tap);
    done += n;
  }
  return out;
}

std::vector<float> Rows(const std::vector<float>& v, std::size_t V,
                        std::size_t first, std::size_t end) {
  return {v.begin() + static_cast<std::ptrdiff_t>(first * V),
          v.begin() + static_cast<std::ptrdiff_t>(end * V)};
}

std::vector<float> LlamaLogits(const Fixture& f, int seq, bool f16) {
  return ReadBinary<float>(OracleOutDir() + "/" + f.name + ".seq" +
                           std::to_string(seq) +
                           (f16 ? ".llama-kvf16.f32" : ".llama.f32"));
}

void Line(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void Line(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::printf("\n");
}

// ---- taps -------------------------------------------------------------------
void Taps(const Fixture& f) {
  struct Entry {
    std::string name;
    int layer;
    std::vector<float> v;
  };
  using Key = std::pair<std::string, int>;
  const Tokens t = LoadTokens(f.base, 0);
  std::map<Key, std::vector<float>> want;
  const Reference ref(f.model.weights);
  (void)ref.ForwardFull(t, [&](std::string_view n, int l, const float* d,
                               std::size_t r, std::size_t w) {
    want[{std::string(n), l}].assign(d, d + r * w);
  });
  std::vector<Key> order;
  std::map<Key, std::vector<float>> got;
  auto e = Make(f, Cfg(kSeq, false, true));
  Feed(*e, t, 0, kSeq, {kSeq},
       [&](std::string_view n, int l, const float* d, std::size_t r, std::size_t w) {
         const Key k{std::string(n), l};
         if (!got.contains(k)) order.push_back(k);
         got[k].assign(d, d + r * w);
       });
  for (const auto& [k, v] : want)
    Require(got.contains(k), "tap never emitted: " + k.first + " layer " + std::to_string(k.second));
  struct Stage {
    std::string name;
    int layer{-2};
    double rel{-1}, abs{0};
  };
  Stage first_diff, first_bad, worst;
  std::map<std::string, double> by_name;
  for (const Key& k : order) {
    const auto it = want.find(k);
    if (it == want.end()) continue;
    const auto& a = got[k];
    const auto& b = it->second;
    Require(a.size() == b.size(), "tap " + k.first + " has " + std::to_string(a.size()) +
                                      " values, the Reference " + std::to_string(b.size()));
    double mx = 0;
    for (const float x : b) mx = std::max(mx, std::abs(static_cast<double>(x)));
    const double abs = MaxAbs(a, b), rel = abs / std::max(mx, 1e-6);
    Stage s{k.first, k.second, rel, abs};
    if (rel > 0 && first_diff.layer == -2) first_diff = s;
    if (!(rel <= g_backend.tap_rel_tol) && first_bad.layer == -2) first_bad = s;
    if (!(rel <= worst.rel)) worst = s;
    by_name[k.first] = std::max(by_name[k.first], rel);
  }
  Line("  %zu taps (%zu names) compared with the Reference", order.size(), by_name.size());
  Line("  first stage with any difference: %s layer %d (rel %.3g, abs %.3g)",
       first_diff.name.c_str(), first_diff.layer, first_diff.rel, first_diff.abs);
  Line("  worst stage: %s layer %d (rel %.3g, abs %.3g); tolerance rel %.3g",
       worst.name.c_str(), worst.layer, worst.rel, worst.abs, g_backend.tap_rel_tol);
  for (const auto& [n, r] : by_name) Line("    %-16s worst rel %.3g", n.c_str(), r);
  Require(first_bad.layer == -2,
          "tap " + first_bad.name + " layer " + std::to_string(first_bad.layer) +
              " differs from the Reference by rel " + Sci(first_bad.rel));
}

// ---- logits against llama.cpp -----------------------------------------------
void Llama(const Fixture& f, bool f16) {
  const double tol = f16 ? kLlamaF16 : kLlamaF32;
  double worst = 0, worst_top1 = 1, kl = 0;
  for (int seq = 0; seq < 4; ++seq) {
    const Tokens t = LoadTokens(f.base, seq);
    auto e = Make(f, Cfg(f.Rows(), f16, !f16));
    const auto logits = Feed(*e, t, 0, t.size(), {f.Rows()});
    const Stats s = Compare(logits, LlamaLogits(f, seq, f16), f.V());
    Line("  seq%d: max abs %.3g, mean KL %.3g, top-1 %.4f, finite %d", seq, s.max_abs,
         s.mean_kl, s.top1, s.finite ? 1 : 0);
    Require(s.finite && s.max_abs <= tol && s.top1 >= 1.0,
            "seq" + std::to_string(seq) + " vs llama.cpp: max abs " + Sci(s.max_abs) +
                " (limit " + Sci(tol) + "), top-1 " + std::to_string(s.top1));
    worst = std::max(worst, s.max_abs);
    worst_top1 = std::min(worst_top1, s.top1);
    kl = std::max(kl, s.mean_kl);
  }
  Line("  worst max abs %.3g (limit %.3g), min top-1 %.4f, worst mean KL %.3g", worst,
       tol, worst_top1, kl);
}

// ---- chunked == one-shot, decode after prefill -------------------------------
void Chunk(const Fixture& f) {
  const std::uint32_t W = f.W, M = f.Rows();
  const std::size_t V = f.V();
  const Tokens t = LoadTokens(f.base, 0);
  for (const bool prod : {false, true}) {
    const double tol = prod ? g_backend.production_equal_tol : g_backend.exact_equal_tol;
    const char* mode = prod ? "F16 KV, production" : "F32 KV, exact";
    auto one = Make(f, Cfg(kSeq, prod, !prod));
    const auto whole = Feed(*one, t, 0, kSeq, {kSeq});
    for (const float x : whole) Require(std::isfinite(x), "non-finite logit in the one-shot run");
    double worst = 0;
    const std::vector<std::vector<std::uint32_t>> plans = {
        {1}, {3}, {M}, {W - 3, 6, M}, {1, W + 1, W - 1}};
    for (const auto& plan : plans) {
      auto e = Make(f, Cfg(M, prod, !prod));
      const double d = MaxAbs(Feed(*e, t, 0, kSeq, plan), whole);
      worst = std::max(worst, d);
      Require(d <= tol, std::string("chunked (first chunk ") + std::to_string(plan[0]) +
                            ") differs from one-shot by " + Sci(d) + " in " + mode);
    }
    Line("  %s: 5 chunkings vs one-shot: worst %.3g (tolerance %.3g)", mode, worst, tol);
    // decode after prefill, and kLast / kNone passes.
    auto e = Make(f, Cfg(M, prod, !prod));
    const std::uint32_t P = W + 3;
    double dd = MaxAbs(Feed(*e, t, 0, P, {P}), Rows(whole, V, 0, P));
    std::vector<float> row(V);
    for (std::uint32_t p = P; p < kSeq; ++p) {
      e->Forward(std::span(t).subspan(p, 1), p, LogitRows::kLast, row);
      dd = std::max(dd, MaxAbs(row, Rows(whole, V, p, p + 1)));
    }
    Line("  %s: prefill %u + %u single-token decodes vs one-shot: %.3g", mode, P, kSeq - P, dd);
    Require(dd <= tol, std::string("decode after prefill differs by ") + Sci(dd) + " in " + mode);
    auto g = Make(f, Cfg(M, prod, !prod));
    double dl = 0;
    std::vector<float> last(V), none;
    std::uint32_t at = 0;
    while (at < kSeq) {
      const std::uint32_t n = std::min<std::uint32_t>(at == 0 ? M : (at % 2 ? 5 : M), kSeq - at);
      const bool want_last = at + n == kSeq || at % 3 == 0;
      g->Forward(std::span(t).subspan(at, n), at,
                 want_last ? LogitRows::kLast : LogitRows::kNone,
                 want_last ? std::span<float>(last) : std::span<float>(none));
      if (want_last) dl = std::max(dl, MaxAbs(last, Rows(whole, V, at + n - 1, at + n)));
      at += n;
    }
    Line("  %s: kLast / kNone mix over the sequence: %.3g", mode, dl);
    Require(dl <= tol, std::string("last-row logits differ by ") + Sci(dl) + " in " + mode);
  }
}

// ---- rewind then different tokens ---------------------------------------------
void Rewind(const Fixture& f) {
  const std::uint32_t M = f.Rows(), W = f.W;
  const std::size_t V = f.V();
  const Tokens t = LoadTokens(f.base, 0), u = LoadTokens(f.base, 1);
  auto fresh = [&](std::uint32_t p) {
    Tokens mix(t.begin(), t.begin() + p);
    mix.insert(mix.end(), u.begin() + p, u.end());
    auto e = Make(f, Cfg(M, false, true));
    return Rows(Feed(*e, mix, 0, kSeq, {M}), V, p, kSeq);
  };
  double worst = 0;
  for (const std::uint32_t high : {W + 5, 2 * W + 9, 50U}) {
    auto e = Make(f, Cfg(M, false, true));
    Feed(*e, t, 0, high, {M});
    for (const std::uint32_t back : {1U, 3U, W - 1}) {
      if (back >= high) continue;
      const std::uint32_t p = high - back;
      if (!e->Rewind(p)) continue;  // outside the ring: covered by the contract suite
      const double d = MaxAbs(Feed(*e, u, p, kSeq, {5}), fresh(p));
      worst = std::max(worst, d);
      Require(d <= g_backend.exact_equal_tol, "rewind to " + std::to_string(p) + " (high " +
                                                   std::to_string(high) + ") then new tokens differs by " + Sci(d));
      e->Reset();
      Feed(*e, t, 0, high, {M});
    }
  }
  // Refused rewinds change nothing; Reset recovers.
  auto e = Make(f, Cfg(M, false, true));
  Feed(*e, t, 0, 50, {M});
  const std::uint32_t pos = e->Position();
  const bool refused = !e->Rewind(5);
  Require(refused && e->Position() == pos, "rewind outside the ring was accepted or changed state");
  Line("  rewind then different tokens == fresh run: worst %.3g (tolerance %.3g)", worst,
       g_backend.exact_equal_tol);
}

// ---- quantized files -----------------------------------------------------------
void Quant(const Fixture& f) {
  const std::size_t V = f.V();
  const Reference ref(f.model.weights);
  double worst_exact = 0, worst_prod = 0, kl_exact = 0, kl_prod = 0, kl_llama = 0,
         worst_llama = 0;
  double top1_exact = 1, top1_prod = 1, top1_llama = 1;
  for (int seq = 0; seq < 4; ++seq) {
    const Tokens t = LoadTokens(f.base, seq);
    const auto want = ref.ForwardFull(t);
    auto e = Make(f, Cfg(f.Rows(), false, true));
    const auto got = Feed(*e, t, 0, t.size(), {f.Rows()});
    const Stats s = Compare(got, want, V);
    auto p = Make(f, Cfg(f.Rows(), true, false));
    const auto got_p = Feed(*p, t, 0, t.size(), {f.Rows()});
    const Stats sp = Compare(got_p, want, V);
    const Stats sl = Compare(got, LlamaLogits(f, seq, false), V);
    Line("  seq%d vs Reference (exact, F32 KV): max abs %.3g mean KL %.3g top-1 %.4f | "
         "production (F16 KV): max abs %.3g mean KL %.3g top-1 %.4f | exact vs llama.cpp same file: "
         "max abs %.3g mean KL %.3g top-1 %.4f",
         seq, s.max_abs, s.mean_kl, s.top1, sp.max_abs, sp.mean_kl, sp.top1, sl.max_abs,
         sl.mean_kl, sl.top1);
    Require(s.finite && sp.finite, "non-finite logits");
    worst_exact = std::max(worst_exact, s.max_abs);
    worst_prod = std::max(worst_prod, sp.max_abs);
    kl_exact = std::max(kl_exact, s.mean_kl);
    kl_prod = std::max(kl_prod, sp.mean_kl);
    kl_llama = std::max(kl_llama, sl.mean_kl);
    worst_llama = std::max(worst_llama, sl.max_abs);
    top1_exact = std::min(top1_exact, s.top1);
    top1_prod = std::min(top1_prod, sp.top1);
    top1_llama = std::min(top1_llama, sl.top1);
  }
  Line("  %s: exact vs Reference: worst max abs %.3g, worst mean KL %.3g, min top-1 %.4f (limit max abs %.3g)",
       f.name.c_str(), worst_exact, kl_exact, top1_exact, g_backend.quant_exact_max_abs);
  Line("  %s: production vs Reference: worst max abs %.3g, worst mean KL %.3g, min top-1 %.4f (reported; limit mean KL 2e-2, top-1 0.95)",
       f.name.c_str(), worst_prod, kl_prod, top1_prod);
  Line("  %s: exact vs llama.cpp on the same file: worst max abs %.3g, worst mean KL %.3g, min top-1 %.4f (reported; S1 Reference had 0.2-0.29 / 7.8e-4..1.5e-3 / 95-98%%)",
       f.name.c_str(), worst_llama, kl_llama, top1_llama);
  Require(worst_exact <= g_backend.quant_exact_max_abs && top1_exact >= 1.0,
          "exact mode differs from the Reference on the same file");
  Require(kl_prod <= 2e-2 && top1_prod >= 0.95, "production mode too far from the Reference");
}

// ---- orchestration twins ---------------------------------------------------------
struct TwinResult {
  double err{0};
  bool refused{false};
  std::string note;
};

/// Max abs error against llama.cpp F32-KV logits of one backend run. With
/// `last_only` each chunk asks for its last row only.
TwinResult RunAgainstLlama(const Fixture& f, const ExecFaults& faults, bool lenient,
                           bool last_only) {
  TwinResult r;
  try {
    const std::size_t V = f.V();
    for (int seq = 0; seq < 4; ++seq) {
      const Tokens t = LoadTokens(f.base, seq);
      const auto llama = LlamaLogits(f, seq, false);
      auto e = Make(f, Cfg(f.Rows(), false, true), faults, lenient);
      if (!e) {
        r.note = "not available on this backend";
        r.err = -1;
        return r;
      }
      std::uint32_t at = 0;
      const std::uint32_t M = f.Rows();
      while (at < t.size()) {
        const std::uint32_t n = std::min<std::uint32_t>(M, static_cast<std::uint32_t>(t.size()) - at);
        std::vector<float> out((last_only ? 1 : n) * V);
        e->Forward(std::span(t).subspan(at, n), at,
                   last_only ? LogitRows::kLast : LogitRows::kAll, out);
        const auto want = last_only ? Rows(llama, V, at + n - 1, at + n) : Rows(llama, V, at, at + n);
        r.err = std::max(r.err, MaxAbs(out, want));
        at += n;
      }
    }
  } catch (const std::exception& e) {
    r.refused = true;
    r.err = INFINITY;
    r.note = std::string("refused: ") + e.what();
  }
  return r;
}

void OrchestrationTwins(const Fixture& f) {
  for (const bool last_only : {false, true}) {
    const TwinResult c = RunAgainstLlama(f, {}, false, last_only);
    Line("  control (%s): max abs %.3g", last_only ? "last row per chunk" : "all rows", c.err);
    Require(!c.refused && c.err <= kLlamaF32, "the correct run does not match llama.cpp: " + c.note);
  }
  struct Twin {
    const char* name;
    ExecFaults faults;
    bool lenient;
    bool last_only;
    bool needs_refusal;  // strict ops: the kernel guard must refuse
  };
  std::vector<Twin> twins;
  auto add = [&](const char* name, auto&& set, bool lenient = false, bool last = false,
                 bool refusal = false) {
    ExecFaults x;
    set(x);
    twins.push_back({name, x, lenient, last, refusal});
  };
  add("x0 taken before the embedding scale", [](ExecFaults& x) { x.x0_before_scale = true; });
  add("value mix fed to Q/K instead of V", [](ExecFaults& x) { x.value_mix_into_qk = true; });
  add("gate projected from the residual", [](ExecFaults& x) { x.gate_from_residual = true; });
  add("RoPE requested on full layers", [](ExecFaults& x) { x.rope_on_full = true; });
  add("ring capacity one row short (kernel guard)", [](ExecFaults& x) { x.ring_shortfall = 1; },
      false, false, true);
  add("ring capacity one row short (guard off)", [](ExecFaults& x) { x.ring_shortfall = 1; },
      true);
  add("post-norm weights swapped", [](ExecFaults& x) { x.swap_post_norms = true; });
  add("logits from the wrong row", [](ExecFaults& x) { x.wrong_logit_row = true; }, false, true);
  add("stale value-mix buffer across layers", [](ExecFaults& x) { x.stale_scratch = true; });
  double weakest = INFINITY;
  std::string weakest_name;
  for (const Twin& tw : twins) {
    const TwinResult r = RunAgainstLlama(f, tw.faults, tw.lenient, tw.last_only);
    if (r.err < 0) {
      Line("  twin %-44s skipped (%s)", tw.name, r.note.c_str());
      continue;
    }
    if (tw.needs_refusal) {
      Line("  twin %-44s %s", tw.name, r.refused ? r.note.c_str() : "NOT REFUSED");
      Require(r.refused, std::string("twin '") + tw.name + "' was not refused by the capacity guard");
      continue;
    }
    const double margin = std::isinf(r.err) ? INFINITY : r.err / kLlamaF32;
    Line("  twin %-44s max abs %.3g  margin %.3gx over the 2e-3 threshold%s", tw.name, r.err,
         margin, r.refused ? ("  (" + r.note + ")").c_str() : "");
    Require(margin >= kTwinFactor, std::string("twin '") + tw.name + "' margin only " + Sci(margin));
    if (margin < weakest) {
      weakest = margin;
      weakest_name = tw.name;
    }
  }
  Line("  weakest twin: %s, margin %.3gx", weakest_name.c_str(), weakest);
}

// ---- refusals and limits ---------------------------------------------------------
template<class Fn>
std::string Throws(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

void Limits(const Fixture& f) {
  // A layer with clamp 0 is refused up front, naming the layer.
  Model m = Model{nullptr, f.model.weights};
  m.weights.config.swiglu_clamp[2] = 0.0F;
  const std::string msg = Throws([&] { (void)g_backend.make(m, Cfg(f.Rows(), false, true), {}, false); });
  Line("  clamp 0 on layer 2: \"%s\"", msg.c_str());
  RequireContains(msg, "not supported", "clamp 0");
  RequireContains(msg, "layer 2", "clamp 0");
  // Context cap: 0 -> default cap, never the GGUF's nominal context (the real
  // file says 2,097,152; the tiny fixtures say less, so also try that value).
  ExecutorConfig c = Cfg(f.Rows(), false, true);
  c.max_context = 0;
  auto e = Make(f, c);
  const auto ctx = f.model.weights.config.context_length;
  Line("  max_context 0 -> %u (fixture's GGUF context %u, default cap %u)", e->Info().max_context,
       ctx, kDefaultContextCap);
  Require(e->Info().max_context == std::min(kDefaultContextCap, ctx), "context cap default wrong");
  Config nominal = f.model.weights.config;
  nominal.context_length = 2097152;
  const std::uint32_t cap = NormalizeConfig(nominal, c).max_context;
  Line("  GGUF context 2097152, max_context 0 -> %u", cap);
  Require(cap == kDefaultContextCap && kDefaultContextCap == 32768, "default cap is not 32768");
  // Bad rows are refused before anything runs.
  auto g = Make(f, Cfg(f.Rows(), false, true));
  std::vector<float> out(f.V());
  const Tokens t = LoadTokens(f.base, 0);
  const std::string r1 = Throws([&] {
    g->Forward(std::span(t).subspan(0, f.Rows() + 1), 0, LogitRows::kNone, {});
  });
  RequireContains(r1, "rows", "too many rows");
  const Tokens bad = {static_cast<std::int32_t>(f.V())};
  const std::string r2 = Throws([&] { g->Forward(bad, 0, LogitRows::kLast, out); });
  Require(!r2.empty() && g->State() == ExecState::kReady && g->Position() == 0,
          "an out-of-range token was not refused cleanly");
  Line("  refusals leave the executor ready at position 0: ok");
}

// ---- driver ------------------------------------------------------------------------
}  // namespace

int main(int argc, char** argv) {
  g_backend = TestBackend();
  using GroupFn = std::function<void(const Fixture&)>;
  const std::vector<std::pair<std::string, GroupFn>> groups = {
      {"taps", GroupFn(Taps)},
      {"llama-f32", GroupFn([](const Fixture& f) { Llama(f, false); })},
      {"llama-f16", GroupFn([](const Fixture& f) { Llama(f, true); })},
      {"chunk", GroupFn(Chunk)},
      {"rewind", GroupFn(Rewind)},
      {"twins", GroupFn(OrchestrationTwins)},
      {"limits", GroupFn(Limits)}};
  std::vector<std::pair<std::string, std::string>> all;  // group, fixture
  for (const auto& [g, fn] : groups) {
    if (g == "limits") {
      all.emplace_back(g, "tinyA");
      continue;
    }
    for (const char* f : {"tinyA", "tinyB"}) all.emplace_back(g, f);
  }
  for (const char* q : {"tinyB-Q4_K_M", "tinyB-Q5_K_M", "tinyB-Q6_K", "tinyB-Q8_0"})
    all.emplace_back("quant", q);
  const std::string arg = argc > 1 ? argv[1] : "";
  if (arg == "--list") {
    for (const auto& [g, f] : all) std::printf("%s:%s\n", g.c_str(), f.c_str());
    return 0;
  }
  if (!g_backend.available()) {
    std::printf("SKIP: backend %s unavailable (no GPU device)\n", g_backend.name.c_str());
    return 77;
  }
  int failed = 0, ran = 0;
  for (const auto& [g, f] : all) {
    const std::string name = g + ":" + f;
    if (!arg.empty() && arg != name && arg != g) continue;
    std::function<void(const Fixture&)> fn;
    if (g == "quant") {
      fn = Quant;
    } else {
      for (const auto& [gn, gf] : groups)
        if (gn == g) fn = gf;
    }
    const auto start = std::chrono::steady_clock::now();
    std::printf("== %s (%s)\n", name.c_str(), g_backend.name.c_str());
    try {
      fn(Load(f));
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      std::printf("PASS %s in %.1f s\n", name.c_str(), s);
    } catch (const std::exception& e) {
      std::printf("FAIL: %s: %s\n", name.c_str(), e.what());
      ++failed;
    }
    std::fflush(stdout);
    ++ran;
  }
  if (ran == 0) {
    std::printf("FAIL: no invocation matches \"%s\" (try --list)\n", arg.c_str());
    return 1;
  }
  std::printf("%s: %d invocation(s), %d failed\n", failed == 0 ? "PASS" : "FAIL", ran, failed);
  return failed == 0 ? 0 : 1;
}

// Executor contract tests (docs/EXECUTOR-CONTRACT.md), for every backend in
// ContractBackends(), on the tiny F32 fixtures. Each check returns an Outcome
// and runs twice: on the backend as built (must pass) and on a broken twin
// (must fail; numeric twins must also move logits or rows by >= kMustDiffer).
//
//   check                         twin(s)
//   1 chunked == one-shot         RoPE positions per chunk; ring one slot short
//   2 decode == prefill           RoPE positions per chunk
//   3 logits vs llama.cpp         attention window one key short
//   4 export/import/continue      import skips a layer; import one window row
//     (restore at W-1, W, W+1,    short; no coverage check + skipped layer
//      2W+3, 40)                  (silent corruption, measured)
//   5 rewinds                     retention from the logical end
//   6 failure mid-forward         failed forward leaves the executor ready
//   7 write plans / retained rows ring one slot short
//   8 taps and K/V rows vs the    compared one layer off
//     Reference
//   9 rewind to position 0        (none: the unfixed executor fails it, docs/DRAFTER-S4.md)
//     after a wrap and an import
#include <cstdio>
#include <cstring>
#include <gufo/ext/cache.hpp>
#include <map>
#include <optional>
#include <utility>

#include "tests/executor_contract.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;

namespace {

using Tokens = std::vector<std::int32_t>;
using Bytes = std::vector<std::byte>;
using Make =
    std::function<std::unique_ptr<Executor>(const ExecutorConfig& config)>;

constexpr double kMustDiffer = 1e-3;
constexpr std::uint32_t kContext = 96;
constexpr std::uint32_t kSeq = 64;
constexpr std::uint32_t kChunkRows = 4;  // adapter rows_per_chunk in restores

struct Fixture {
  std::string name;
  Model model;
  Tokens t, u;  // two different 64-token sequences
  std::uint32_t W{0};
  [[nodiscard]] std::uint32_t Rows() const { return W + 4; }  // max_rows
};

struct Ctx {
  const ContractBackend& backend;
  const Fixture& f;
  Make make;
};

struct Outcome {
  bool pass{true};
  double measure{0};  ///< max |difference| (or mismatching rows), see check
  std::string detail;
  void Fail(const std::string& why) {
    if (pass) detail = why;
    pass = false;
  }
};

// NaN or Inf in either input gives +infinity (tests/check.hpp), so two runs that are
// NaN in the same place no longer "agree".
double MaxDiff(std::span<const float> a, std::span<const float> b) {
  return MaxAbsDiff(a, b);
}

ExecutorConfig Config(const Ctx& c, std::uint32_t max_rows) {
  ExecutorConfig cfg;
  cfg.max_context = kContext;
  cfg.max_rows = max_rows;
  cfg.max_logit_rows = max_rows;
  cfg.kv_type = c.backend.kv_type;
  return cfg;
}

/// Tolerance of a run whose widest Forward had `widest` rows.
double Tol(const Ctx& c, const Executor& e, std::size_t widest) {
  return widest <= e.Info().batch_invariant_rows ? 0.0
                                                 : c.backend.rounding_tol;
}

/// Feeds t[from, to) in chunks of `sizes` (the last repeats, capped by
/// max_rows); returns all-row logits [to - from][vocab].
std::vector<float> Feed(Executor& e, const Tokens& t, std::size_t from,
                        std::size_t to, const std::vector<std::uint32_t>& sizes,
                        std::size_t* widest = nullptr, const Tap& tap = {}) {
  const std::size_t V = e.Info().vocab;
  std::vector<float> out((to - from) * V);
  std::size_t done = 0, i = 0;
  while (from + done < to) {
    const std::size_t n = std::min<std::size_t>(
        {sizes[std::min(i++, sizes.size() - 1)], to - from - done,
         e.Info().max_rows});
    e.Forward(std::span(t).subspan(from + done, n),
              static_cast<std::uint32_t>(from + done), LogitRows::kAll,
              std::span(out).subspan(done * V, n * V), tap);
    if (widest != nullptr) *widest = std::max(*widest, n);
    done += n;
  }
  return out;
}

Bytes Export(const Executor& e, std::uint32_t l, KvPart part, RowRange r) {
  Bytes out(std::size_t{r.size()} * e.Info().RowBytes());
  e.ExportRows(l, part, r, out);
  return out;
}

/// Every layer's K and V rows: sliding from `from`, full from 0.
Bytes ExportAll(const Executor& e, std::uint32_t from) {
  Bytes all;
  for (std::uint32_t l = 0; l < e.Info().layers; ++l) {
    for (const KvPart part : {KvPart::kKey, KvPart::kValue}) {
      const RowRange r{e.Info().sliding[l] ? from : 0, e.Position()};
      const Bytes b = Export(e, l, part, r);
      all.insert(all.end(), b.begin(), b.end());
    }
  }
  return all;
}

std::vector<float> Rows(const std::vector<float>& logits, std::size_t V,
                        std::size_t first, std::size_t end) {
  return {logits.begin() + static_cast<std::ptrdiff_t>(first * V),
          logits.begin() + static_cast<std::ptrdiff_t>(end * V)};
}

float HalfToFloat(std::uint16_t h) {
  const std::uint32_t sign = (h & 0x8000U) << 16, exp = (h >> 10) & 0x1FU,
                      man = h & 0x3FFU;
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {  // subnormal
      int e = -1;
      std::uint32_t m = man;
      do {
        ++e;
        m <<= 1;
      } while ((m & 0x400U) == 0);
      bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3FFU) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000U | (man << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (man << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

std::vector<float> Decode(const Bytes& b, KvType type) {
  std::vector<float> out(b.size() / (type == KvType::kF16 ? 2 : 4));
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (type == KvType::kF32) {
      std::memcpy(&out[i], b.data() + i * 4, 4);
    } else {
      std::uint16_t h;
      std::memcpy(&h, b.data() + i * 2, 2);
      out[i] = HalfToFloat(h);
    }
  }
  return out;
}

// ---- 1 chunked forward equals one-shot --------------------------------------
Outcome ChunkedVsOneShot(const Ctx& c) {
  Outcome o;
  const std::uint32_t W = c.f.W, M = c.f.Rows();
  auto one = c.make(Config(c, kSeq));
  const auto whole = Feed(*one, c.f.t, 0, kSeq, {kSeq});
  const std::vector<std::vector<std::uint32_t>> plans = {
      {1}, {3}, {M}, {W - 3, 6, M}, {1, W + 1, W - 1}};
  for (const auto& plan : plans) {
    auto e = c.make(Config(c, M));
    std::size_t widest = 0;
    const auto logits = Feed(*e, c.f.t, 0, kSeq, plan, &widest);
    const double d = MaxDiff(logits, whole);
    o.measure = MaxKeepNaN(o.measure, d);
    if (Exceeds(d, std::max(Tol(c, *one, kSeq), Tol(c, *e, widest))))
      o.Fail("plan starting " + std::to_string(plan[0]) + " differs by " +
             std::to_string(d));
  }
  return o;
}

// ---- 2 decode after prefill equals prefill ------------------------------------
Outcome DecodeVsPrefill(const Ctx& c) {
  Outcome o;
  const std::size_t V = c.f.model.weights.config.vocab_size;
  auto one = c.make(Config(c, kSeq));
  const auto whole = Feed(*one, c.f.t, 0, kSeq, {kSeq});
  auto e = c.make(Config(c, c.f.Rows()));
  const std::uint32_t P = c.f.W + 3;
  std::size_t widest = 0;
  const auto prefill = Feed(*e, c.f.t, 0, P, {P}, &widest);
  o.measure = MaxDiff(prefill, Rows(whole, V, 0, P));
  std::vector<float> row(V);
  for (std::uint32_t p = P; p < kSeq; ++p) {
    e->Forward(std::span(c.f.t).subspan(p, 1), p, LogitRows::kLast, row);
    o.measure = MaxKeepNaN(o.measure, MaxDiff(row, Rows(whole, V, p, p + 1)));
  }
  if (Exceeds(o.measure, std::max(Tol(c, *one, kSeq), Tol(c, *e, widest))))
    o.Fail("decode differs from prefill by " + std::to_string(o.measure));
  return o;
}

// ---- 3 logits against llama.cpp ------------------------------------------------
Outcome AgainstLlama(const Ctx& c) {
  Outcome o;
  const std::size_t V = c.f.model.weights.config.vocab_size;
  double top1 = 1;
  for (int seq = 0; seq < 4; ++seq) {
    const Tokens t = LoadTokens(c.f.name, seq);
    const auto llama = ReadBinary<float>(OracleOutDir() + "/" + c.f.name +
                                         ".seq" + std::to_string(seq) +
                                         c.backend.llama_suffix);
    auto e = c.make(Config(c, c.f.Rows()));
    const auto logits = Feed(*e, t, 0, t.size(), {c.f.Rows()});
    const Stats s = Compare(logits, llama, V);
    o.measure = MaxKeepNaN(o.measure, s.max_abs);
    top1 = std::min(top1, s.top1);
    if (!s.finite || Exceeds(s.max_abs, c.backend.llama_max_abs) || !(s.top1 >= 1))
      o.Fail("seq" + std::to_string(seq) + " max_abs " +
             std::to_string(s.max_abs) + " top1 " + std::to_string(s.top1));
  }
  o.detail += " (min top-1 " + std::to_string(top1) + ")";
  return o;
}

// ---- 4 export, import into a fresh executor, continue ---------------------------
enum class ImportDefect { kNone, kSkipLayer, kShortWindow };

Outcome Restore(const Ctx& c, std::uint32_t n, ImportDefect defect) {
  Outcome o;
  const std::uint32_t M = c.f.Rows();
  auto a = c.make(Config(c, M));
  std::size_t widest = 0;
  Feed(*a, c.f.t, 0, n, {M}, &widest);
  auto b = c.make(Config(c, M));
  const ExecutorInfo& info = a->Info();
  std::uint32_t skip = info.layers;  // first sliding layer
  for (std::uint32_t l = 0; l < info.layers && skip == info.layers; ++l)
    if (info.sliding[l]) skip = l;
  // What a cache::Adapter with kWindowRows components would load.
  gufo::cache::ComponentDescriptor window;
  window.kind = gufo::cache::ComponentKind::kWindowRows;
  window.rows_per_chunk = kChunkRows;
  window.window_rows = c.f.W - 1;
  b->BeginImport(n);
  for (std::uint32_t l = 0; l < info.layers; ++l) {
    if (defect == ImportDefect::kSkipLayer && l == skip) continue;
    for (const KvPart part : {KvPart::kKey, KvPart::kValue}) {
      std::uint32_t first = 0;
      if (info.sliding[l]) {
        first = static_cast<std::uint32_t>(gufo::cache::FirstChunkRow(window, n));
        if (defect == ImportDefect::kShortWindow) first = info.WindowFirst(n) + 1;
      }
      const Bytes rows = Export(*a, l, part, {first, n});
      b->ImportRows(l, part, {first, n}, rows);
    }
  }
  if (!b->CommitImport()) {
    o.Fail("CommitImport refused the import");
    return o;
  }
  auto from = [&] { return std::max(a->RetainedBegin(), b->RetainedBegin()); };
  if (ExportAll(*a, from()) != ExportAll(*b, from()))
    o.Fail("imported rows differ from the source");
  const auto la = Feed(*a, c.f.t, n, kSeq, {3}, &widest);
  const auto lb = Feed(*b, c.f.t, n, kSeq, {3}, &widest);
  o.measure = MaxDiff(la, lb);
  if (Exceeds(o.measure, Tol(c, *a, widest)))
    o.Fail("continuation differs by " + std::to_string(o.measure));
  if (ExportAll(*a, from()) != ExportAll(*b, from()))
    o.Fail("rows after the continuation differ");
  return o;
}

// ---- 5 rewinds ----------------------------------------------------------------
Outcome Rewinds(const Ctx& c) {
  Outcome o;
  const std::uint32_t M = c.f.Rows(), W = c.f.W, V = c.f.model.weights.config.vocab_size;
  constexpr std::uint32_t kHigh = 40;
  // Fresh run of t[0, p) then u[p, 64): what a correct rewind must equal.
  auto fresh = [&](std::uint32_t p) {
    Tokens mix(c.f.t.begin(), c.f.t.begin() + p);
    mix.insert(mix.end(), c.f.u.begin() + p, c.f.u.end());
    auto e = c.make(Config(c, M));
    return Rows(Feed(*e, mix, 0, kSeq, {M}), V, p, kSeq);
  };
  // Two rewinds, both inside what the ring keeps; the second lands exactly
  // on the lowest legal position.
  auto a = c.make(Config(c, M));
  Feed(*a, c.f.t, 0, kHigh, {M});
  const std::uint32_t R = a->Info().ring_rows;
  Require(kHigh > R, "fixture ring too large for the rewind test");
  const std::uint32_t retained = a->RetainedBegin();
  if (retained != kHigh - R) o.Fail("RetainedBegin is not high water - ring");
  const std::uint32_t lowest = retained + W - 1;
  if (!a->Rewind(kHigh - 2) || !a->Rewind(lowest)) o.Fail("legal rewind refused");
  std::size_t widest = 0;
  o.measure = MaxDiff(Feed(*a, c.f.u, a->Position(), kSeq, {5}, &widest),
                      fresh(a->Position()));
  // Second rewind one position below the legal range: needs a row the ring
  // overwrote, although Position() - ring would still allow it.
  auto b = c.make(Config(c, M));
  Feed(*b, c.f.t, 0, kHigh, {M});
  if (!b->Rewind(kHigh - 2)) o.Fail("legal rewind refused");
  const Bytes before = ExportAll(*b, b->RetainedBegin());
  if (b->Rewind(lowest - 1)) {
    o.Fail("rewind to " + std::to_string(lowest - 1) + " accepted");
  } else if (b->Position() != kHigh - 2 ||
             ExportAll(*b, b->RetainedBegin()) != before) {
    o.Fail("refused rewind changed the executor");
  }
  bool threw = false;
  try {
    (void)b->Rewind(b->Position() + 1);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) o.Fail("rewind beyond the position not refused");
  const double d = MaxDiff(Feed(*b, c.f.u, b->Position(), kSeq, {5}, &widest),
                           fresh(b->Position()));
  o.measure = MaxKeepNaN(o.measure, d);
  if (Exceeds(o.measure, Tol(c, *a, widest)))
    o.Fail("continuation after rewind differs by " + std::to_string(o.measure));
  return o;
}

// ---- 9 rewind to position 0 ------------------------------------------------------
// Position 0 reads no earlier row, but the high water mark and the import floor
// stay where they were: a rewind to 0 after the ring wrapped (or after an import)
// would leave RetainedBegin() above 0, and the next pass could not start. It must
// be refused, changing nothing, like every other rewind that needs recomputation;
// a rewind to 0 where nothing is lost (high water within the ring) still succeeds.
Outcome RewindToZero(const Ctx& c) {
  Outcome o;
  const std::uint32_t M = c.f.Rows(), V = c.f.model.weights.config.vocab_size;
  constexpr std::uint32_t kHigh = 40;
  const auto want = [&] {
    auto e = c.make(Config(c, M));
    return Feed(*e, c.f.u, 0, kSeq, {5});
  }();
  std::size_t widest = 0;
  const auto from_scratch = [&](Executor& e, const char* what) {
    // What every caller does when Rewind says false: Reset and recompute.
    const auto got = Feed(e, c.f.u, 0, kSeq, {5}, &widest);
    o.measure = MaxKeepNaN(o.measure, MaxDiff(got, want));
    if (Exceeds(o.measure, Tol(c, e, widest)))
      o.Fail(std::string(what) + ": a new text from position 0 differs from a fresh executor by " +
             std::to_string(o.measure));
  };
  (void)V;
  // (a) the ring wrapped
  {
    auto a = c.make(Config(c, M));
    Feed(*a, c.f.t, 0, kHigh, {M});
    Require(kHigh > a->Info().ring_rows, "fixture ring too large for the rewind test");
    const std::uint32_t retained = a->RetainedBegin();
    if (retained == 0) o.Fail("RetainedBegin is 0 after the ring wrapped");
    const Bytes before = ExportAll(*a, retained);
    if (a->Rewind(0)) {
      o.Fail("rewind to 0 accepted after the ring wrapped (RetainedBegin " + std::to_string(retained) + ")");
    } else if (a->Position() != kHigh || a->RetainedBegin() != retained ||
               ExportAll(*a, retained) != before) {
      o.Fail("refused rewind to 0 changed the executor");
    }
    if (a->State() == ExecState::kReady) {
      a->Reset();
      from_scratch(*a, "after the refused rewind");
    }
  }
  // (b) after an import: the floor is above 0
  {
    auto src = c.make(Config(c, M));
    Feed(*src, c.f.t, 0, kHigh, {M});
    auto b = c.make(Config(c, M));
    const ExecutorInfo& info = src->Info();
    gufo::cache::ComponentDescriptor window;
    window.kind = gufo::cache::ComponentKind::kWindowRows;
    window.rows_per_chunk = kChunkRows;
    window.window_rows = c.f.W - 1;
    b->BeginImport(kHigh);
    for (std::uint32_t l = 0; l < info.layers; ++l) {
      for (const KvPart part : {KvPart::kKey, KvPart::kValue}) {
        const std::uint32_t first =
            info.sliding[l] ? static_cast<std::uint32_t>(gufo::cache::FirstChunkRow(window, kHigh)) : 0;
        b->ImportRows(l, part, {first, kHigh}, Export(*src, l, part, {first, kHigh}));
      }
    }
    if (!b->CommitImport()) {
      o.Fail("CommitImport refused the import");
    } else {
      const std::uint32_t retained = b->RetainedBegin();
      if (retained == 0) o.Fail("RetainedBegin is 0 after an import at " + std::to_string(kHigh));
      if (b->Rewind(0)) {
        o.Fail("rewind to 0 accepted after an import (RetainedBegin " + std::to_string(retained) + ")");
      } else if (b->Position() != kHigh || b->RetainedBegin() != retained) {
        o.Fail("refused rewind to 0 changed the executor after an import");
      }
      if (b->State() == ExecState::kReady) {
        b->Reset();
        from_scratch(*b, "after the refused rewind (import)");
      }
    }
  }
  // (c) nothing lost: a rewind to 0 succeeds and the text from 0 is the fresh one
  {
    auto d = c.make(Config(c, M));
    Feed(*d, c.f.t, 0, d->Info().ring_rows, {M});
    if (d->RetainedBegin() != 0) o.Fail("RetainedBegin above 0 within the first ring");
    if (!d->Rewind(0) || d->Position() != 0) {
      o.Fail("rewind to 0 refused where no row was lost");
    } else {
      from_scratch(*d, "after the accepted rewind");
    }
  }
  return o;
}

// ---- 6 failure in the middle of a forward --------------------------------------
template<class Fn>
bool RefusedAsLogicError(Fn&& fn) {
  try {
    fn();
  } catch (const std::logic_error& e) {
    return dynamic_cast<const std::invalid_argument*>(&e) == nullptr;
  }
  return false;
}

Outcome Failure(const Ctx& c) {
  Outcome o;
  const std::uint32_t M = c.f.Rows(), V = c.f.model.weights.config.vocab_size;
  constexpr std::uint32_t kAt = 30;
  auto one = c.make(Config(c, kSeq));
  const auto whole = Feed(*one, c.f.t, 0, kSeq, {kSeq});
  auto a = c.make(Config(c, M));
  Feed(*a, c.f.t, 0, kAt, {M});
  const int layer = static_cast<int>(a->Info().layers / 2);
  const Tap boom = [&](std::string_view name, int l, const float*, std::size_t,
                       std::size_t) {
    if (name == "l_out" && l == layer) throw std::runtime_error("injected");
  };
  bool threw = false;
  try {
    a->Forward(std::span(c.f.t).subspan(kAt, M), kAt, LogitRows::kNone, {},
               boom);
  } catch (const std::runtime_error& e) {
    threw = std::string(e.what()) == "injected";
  }
  if (!threw) o.Fail("injected failure did not propagate");
  std::size_t widest = 0;
  if (a->State() != ExecState::kFailed || a->Position() != 0 ||
      a->RetainedBegin() != 0) {
    o.Fail("failed forward left the executor usable");
    // What the caller would get by trusting it: rewind inside the old
    // retention and continue.
    if (a->State() == ExecState::kReady && a->Rewind(20))
      o.measure = MaxDiff(Feed(*a, c.f.t, 20, kSeq, {M}), Rows(whole, V, 20, kSeq));
    return o;
  }
  std::vector<float> row(V);
  Bytes rows(a->Info().RowBytes());
  const bool refused =
      RefusedAsLogicError([&] {
        a->Forward(std::span(c.f.t).subspan(0, 1), 0, LogitRows::kLast, row);
      }) &&
      RefusedAsLogicError([&] { (void)a->Rewind(0); }) &&
      RefusedAsLogicError([&] { a->ExportRows(0, KvPart::kKey, {0, 1}, rows); }) &&
      RefusedAsLogicError([&] { (void)a->PlanForward(1); }) &&
      RefusedAsLogicError([&] { a->ImportRows(0, KvPart::kKey, {0, 1}, rows); }) &&
      RefusedAsLogicError([&] { (void)a->CommitImport(); });
  if (!refused) o.Fail("a call on the failed executor was not refused");
  // Recovery by restore: import a healthy executor's rows.
  auto healthy = c.make(Config(c, M));
  Feed(*healthy, c.f.t, 0, kAt, {M}, &widest);
  a->BeginImport(kAt);
  for (std::uint32_t l = 0; l < a->Info().layers; ++l)
    for (const KvPart part : {KvPart::kKey, KvPart::kValue}) {
      const RowRange r{a->Info().sliding[l] ? healthy->RetainedBegin() : 0, kAt};
      a->ImportRows(l, part, r, Export(*healthy, l, part, r));
    }
  if (!a->CommitImport()) o.Fail("restore after failure refused");
  o.measure = MaxDiff(Feed(*a, c.f.t, kAt, kSeq, {M}, &widest),
                      Rows(whole, V, kAt, kSeq));
  // Recovery by Reset and a full prefill.
  a->Reset();
  if (a->State() != ExecState::kReady || a->Position() != 0)
    o.Fail("Reset did not make the executor ready and empty");
  o.measure = MaxKeepNaN(o.measure, MaxDiff(Feed(*a, c.f.t, 0, kSeq, {M}), whole));
  if (Exceeds(o.measure, std::max(Tol(c, *one, kSeq), Tol(c, *a, widest))))
    o.Fail("recovered run differs by " + std::to_string(o.measure));
  return o;
}

// ---- 7 write plans: written/evicted ranges and untouched retained rows -----------
Outcome Plans(const Ctx& c) {
  Outcome o;
  const std::uint32_t M = c.f.Rows(), W = c.f.W;
  auto e = c.make(Config(c, M));
  const ExecutorInfo& info = e->Info();
  std::vector<float> none;
  const std::vector<std::uint32_t> sizes = {W - 3, M, 1, M, 7, M, 2, M, M, M};
  std::size_t step = 0;
  while (e->Position() + 1 < kSeq) {
    if (step == 4 && !e->Rewind(e->Position() - 2)) o.Fail("rewind refused");
    const std::uint32_t at = e->Position();
    const auto n = std::min<std::uint32_t>(sizes[step++ % sizes.size()], kSeq - at);
    const WritePlan plan = e->PlanForward(n);
    const std::uint32_t was = e->RetainedBegin();
    std::vector<Bytes> snap;
    for (std::uint32_t l = 0; l < info.layers; ++l)
      for (const KvPart part : {KvPart::kKey, KvPart::kValue})
        snap.push_back(Export(*e, l, part, {info.sliding[l] ? was : 0, at}));
    e->Forward(std::span(c.f.t).subspan(at, n), at, LogitRows::kNone, none);
    const std::uint32_t now = e->RetainedBegin();
    if (!(plan.written == RowRange{at, at + n})) o.Fail("written range wrong");
    if (plan.evicted.first != was && !plan.evicted.empty())
      o.Fail("evicted range does not start at RetainedBegin");
    if (now != (plan.evicted.empty() ? was : plan.evicted.end))
      o.Fail("RetainedBegin after the forward disagrees with the plan");
    std::size_t i = 0;
    for (std::uint32_t l = 0; l < info.layers; ++l) {
      for (const KvPart part : {KvPart::kKey, KvPart::kValue}) {
        const std::uint32_t first = info.sliding[l] ? was : 0;
        const std::uint32_t keep = info.sliding[l] ? now : 0;
        for (std::uint32_t p = keep; p < at; ++p) {
          const Bytes row = Export(*e, l, part, {p, p + 1});
          if (std::memcmp(row.data(),
                          snap[i].data() + std::size_t{p - first} * row.size(),
                          row.size()) != 0) {
            o.measure += 1;  // retained row changed by the forward
          }
        }
        ++i;
      }
    }
  }
  if (o.measure > 0)
    o.Fail(std::to_string(static_cast<long>(o.measure)) +
           " retained rows changed by forwards");
  return o;
}

// ---- 8 taps and exported K/V rows against the Reference -------------------------
Outcome TapsAndRows(const Ctx& c, int layer_shift) {
  Outcome o;
  using Key = std::pair<std::string, int>;
  std::map<Key, std::vector<float>> want, got;
  auto collect = [](std::map<Key, std::vector<float>>& into) {
    return [&into](std::string_view name, int layer, const float* data,
                   std::size_t rows, std::size_t width) {
      auto& v = into[{std::string(name), layer}];
      v.insert(v.end(), data, data + rows * width);
    };
  };
  const Reference ref(c.f.model.weights);
  (void)ref.ForwardFull(c.f.t, collect(want));
  auto e = c.make(Config(c, c.f.Rows()));
  Feed(*e, c.f.t, 0, kSeq, {c.f.Rows()}, nullptr, collect(got));
  const ExecutorInfo& info = e->Info();
  std::size_t compared = 0;
  for (const auto& [key, values] : got) {
    if (key.second < 0 && layer_shift != 0) continue;
    const int at = key.second < 0 ? key.second : key.second + layer_shift;
    const auto it = want.find({key.first, at});
    if (it == want.end() || it->second.size() != values.size()) continue;
    o.measure = MaxKeepNaN(o.measure, MaxDiff(values, it->second));
    ++compared;
  }
  for (std::uint32_t l = 0; l < info.layers; ++l)
    if (!got.contains({"l_out", static_cast<int>(l)}))
      o.Fail("no l_out tap for layer " + std::to_string(l));
  // Exported rows: K after norm and RoPE ("Kcur"), V as projected ("Vcur").
  for (std::uint32_t l = 0; l < info.layers; ++l) {
    const int ref_layer = static_cast<int>(l) + layer_shift;
    if (ref_layer >= static_cast<int>(info.layers)) continue;
    const std::uint32_t first = info.sliding[l] ? e->RetainedBegin() : 0;
    for (const auto& [part, name] : {std::pair{KvPart::kKey, "Kcur"},
                                     std::pair{KvPart::kValue, "Vcur"}}) {
      const auto rows = Decode(Export(*e, l, part, {first, kSeq}), info.kv_type);
      const auto& full = want.at({name, ref_layer});
      o.measure = MaxKeepNaN(
          o.measure,
          MaxDiff(rows, std::span(full).subspan(std::size_t{first} * info.kv_width,
                                                rows.size())));
      ++compared;
    }
  }
  o.detail = std::to_string(compared) + " taps/row sets compared";
  if (Exceeds(o.measure, c.backend.tap_tol))
    o.Fail("taps or rows differ from the Reference by " +
           std::to_string(o.measure) + "; " + o.detail);
  return o;
}

// ---- driver --------------------------------------------------------------------
void Report(const std::string& label, const Outcome& o, bool twin,
            bool numeric) {
  std::printf("  %-58s %10.3e  %s%s%s\n", label.c_str(), o.measure,
              o.pass ? "pass" : "FAIL", twin ? " (twin)" : "",
              o.detail.empty() ? "" : ("  " + o.detail).c_str());
  if (!twin) {
    Require(o.pass, label + ": " + o.detail);
  } else {
    Require(!o.pass, label + ": broken twin PASSED, the check cannot fail");
    if (numeric)
      Require(o.measure >= kMustDiffer,
              label + ": twin moved results by only " + std::to_string(o.measure));
  }
}

/// Runs `check` on a twin; an exception counts as a detected failure.
Outcome OnTwin(const std::function<Outcome()>& check) {
  try {
    return check();
  } catch (const std::exception& e) {
    Outcome o;
    o.Fail(std::string("threw: ") + e.what());
    return o;
  }
}

void RunBackend(const ContractBackend& b, const Fixture& f) {
  const Ctx ok{b, f, [&](const ExecutorConfig& cfg) { return b.make(f.model, cfg); }};
  auto twin_ctx = [&](Twin t) {
    return Ctx{b, f, [&b, &f, t](const ExecutorConfig& cfg) {
                 return b.make_twin(f.model, cfg, t);
               }};
  };
  const std::string p = f.name + " " + b.name + ": ";
  const bool twins = static_cast<bool>(b.make_twin);
  auto twin = [&](const std::string& label, Twin t, bool numeric,
                  const std::function<Outcome(const Ctx&)>& check) {
    if (!twins) return;
    const Ctx tc = twin_ctx(t);
    Report(p + label, OnTwin([&] { return check(tc); }), true, numeric);
  };

  Report(p + "1 chunked == one-shot", ChunkedVsOneShot(ok), false, true);
  twin("1 twin: RoPE positions per chunk", Twin::kChunkLocalRope, true, ChunkedVsOneShot);
  twin("1 twin: ring one slot short", Twin::kRingOneShort, true, ChunkedVsOneShot);

  Report(p + "2 decode after prefill == prefill", DecodeVsPrefill(ok), false, true);
  twin("2 twin: RoPE positions per chunk", Twin::kChunkLocalRope, true, DecodeVsPrefill);

  Report(p + "3 logits vs llama.cpp (4 seqs)", AgainstLlama(ok), false, true);
  twin("3 twin: window one key short", Twin::kWindowMinus1, true, AgainstLlama);

  const std::uint32_t W = f.W;
  for (const std::uint32_t n : {W - 1, W, W + 1, 2 * W + 3, 40U}) {
    Report(p + "4 restore at " + std::to_string(n) + " then continue",
           Restore(ok, n, ImportDefect::kNone), false, true);
  }
  if (twins) {
    Report(p + "4 twin: import skips a layer (n=W+1)",
           OnTwin([&] { return Restore(ok, W + 1, ImportDefect::kSkipLayer); }),
           true, false);
    Report(p + "4 twin: import one window row short (n=2W+3)",
           OnTwin([&] { return Restore(ok, 2 * W + 3, ImportDefect::kShortWindow); }),
           true, false);
    twin("4 twin: no coverage check + skipped layer", Twin::kNoImportCoverage, true,
         [&](const Ctx& c) { return Restore(c, W + 1, ImportDefect::kSkipLayer); });
  }

  Report(p + "5 rewinds (two consecutive, in and out)", Rewinds(ok), false, true);
  twin("5 twin: retention from the logical end", Twin::kRetentionFromEnd, true, Rewinds);

  Report(p + "6 failure mid-forward, refusal, recovery", Failure(ok), false, true);
  twin("6 twin: failed forward stays ready", Twin::kFailureKeepsReady, true, Failure);

  Report(p + "7 write plans and untouched retained rows", Plans(ok), false, false);
  twin("7 twin: ring one slot short", Twin::kRingOneShort, false, Plans);

  Report(p + "8 taps and K/V rows vs Reference",
         TapsAndRows(ok, 0), false, true);
  if (twins)
    Report(p + "8 twin: compared one layer off",
           OnTwin([&] { return TapsAndRows(ok, 1); }), true, true);

  Report(p + "9 rewind to position 0 (wrapped, imported, nothing lost)", RewindToZero(ok), false, true);
}

}  // namespace

int main() {
  return Run([] {
    const auto backends = ContractBackends();
    Require(!backends.empty(), "no backend registered");
    for (const char* name : {"tinyA", "tinyB"}) {
      const std::string path = FixtureDir() + "/" + name + ".gguf";
      Fixture f{name, OpenModel(path), LoadTokens(name, 0), LoadTokens(name, 1), 0};
      f.W = f.model.weights.config.sliding_window;
      Require(f.t.size() == kSeq && f.u.size() == kSeq, "token files must hold 64 ids");
      std::printf("%s: window %u, %u layers, max_rows %u\n", name, f.W,
                  f.model.weights.config.block_count, f.Rows());
      for (const auto& b : backends) RunBackend(b, f);
    }
  });
}

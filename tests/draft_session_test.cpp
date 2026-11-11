// Target side of speculative decoding (docs/DRAFTER-S3.md): Session::Verify,
// AcceptVerified (the exact rewind) and the captured hidden states, on the scalar
// CPU executor and the host-ops executor, tinyA (window 8) and tinyB (window 16).
// Every property is "drafted equals plain, bit for bit" and has a twin (a
// Session::Faults defect) that must fail it.
//
//   1. every k (1..16), every a (0..k), at anchors that put the pass across W,
//      2W+3 and the ring wrap: verify k rows, keep a, continue with 1-row decoding
//      for 8 tokens; logits and tokens equal plain 1-row decoding from the start
//   2. 64 random verify/rewind cycles (fixed seed) in sweeps, each ended by
//      comparing every observed position with plain decoding
//   3. the captured rows equal those of plain decoding, also after rewinds
//   4. twins: rewind from the logical end, stale frontier logits, rejected taps kept
//   5. a checkpoint (component cache adapter) taken after drafted steps, and a
//      prefix-reuse sync, give the same bytes and continuation as after plain decoding
//   6. refusals and the write guard
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <utility>

#include "src/continuation_adapter.hpp"
#include "tests/draft_util.hpp"

using namespace ricursive_test;

namespace {

constexpr std::uint32_t kContext = 96;
constexpr std::uint32_t kPrefillRows = 20;
constexpr std::uint32_t kChunk = 4;
constexpr std::size_t kSeq = 94;       // tokens of the plain stream
constexpr std::size_t kContinue = 8;   // plain steps after a rewind

bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::shared_ptr<EngineModel> Armed(const TinyFixture& f, const DraftBackend& be, std::uint32_t rows = kPrefillRows,
                                   std::vector<std::uint32_t> capture = {}) {
  return LoadDraftModel(f.path, be, kContext, rows, kChunk, [&](ModelOptions& o) {
    o.verify_rows = kMaxVerifyRows;
    o.capture_layers = capture;
  });
}

/// The plain stream: S, the logits after every prefix, one row at a time from
/// the first token (and, with capture on, the captured rows of every position).
struct Plain {
  std::vector<std::int32_t> S;
  std::vector<float> logits;  ///< [kSeq][vocab]
  std::vector<float> taps;    ///< [kSeq][tap][hidden]
  std::size_t vocab{0};
  [[nodiscard]] std::span<const float> Row(std::size_t pos) const {
    return std::span<const float>(logits).subspan(pos * vocab, vocab);
  }
};

Plain RunPlain(const std::shared_ptr<EngineModel>& model, std::vector<std::int32_t> S) {
  Plain p;
  p.S = std::move(S);
  p.vocab = model->VocabSize();
  std::string error;
  auto s = NewSession(model);
  for (std::size_t i = 0; i < p.S.size(); ++i) {
    if (i == 0) {
      Require(s->Sync(Slice(p.S, 0, 1), &error), error);
    } else {
      Require(s->Evaluate(p.S[i], &error), error);
    }
    p.logits.insert(p.logits.end(), s->Logits().begin(), s->Logits().end());
    auto t = s->TakeTaps();
    if (!model->executor_config().capture_layers.empty()) {
      Require(t.first_position == i && t.rows == 1, "plain taps: one row per step");
      p.taps.insert(p.taps.end(), t.data.begin(), t.data.end());
    }
  }
  return p;
}

std::vector<std::int32_t> Candidates(const Plain& plain, std::size_t p, std::uint32_t k, std::uint32_t a) {
  std::vector<std::int32_t> c;
  for (std::uint32_t i = 0; i < k; ++i) {
    const std::int32_t good = plain.S[p + i];
    c.push_back(i < a ? good : static_cast<std::int32_t>((good + 1 + i) % 256));  // never the plain token
  }
  return c;
}

/// One scenario of test 1. "" = every comparison held, else the first that did not.
std::string Scenario(const std::shared_ptr<EngineModel>& model, const Plain& plain, std::size_t p, std::uint32_t k,
                     std::uint32_t a, Session::Faults faults, std::size_t* rows_compared) {
  try {
    std::string error;
    auto s = NewSession(model);
    s->SetFaults(faults);
    if (!s->Sync(Slice(plain.S, 0, p), &error)) return "sync: " + error;
    if (!SameBits(s->Logits(), plain.Row(p - 1))) return "prompt logits differ from plain";
    const auto cand = Candidates(plain, p, k, a);
    if (!s->Verify(cand, &error)) return "verify: " + error;
    if (s->Position() != p + k || !s->VerificationOpen()) return "position after verify";
    const std::size_t V = plain.vocab;
    if (s->VerifyLogits().size() != k * V) return "verify logits size";
    for (std::uint32_t i = 0; i < a; ++i) {
      if (!SameBits(s->VerifyLogits().subspan(i * V, V), plain.Row(p + i))) {
        return "verify row " + std::to_string(i) + " differs from plain";
      }
      ++*rows_compared;
    }
    if (!SameBits(s->Logits(), s->VerifyLogits().subspan((k - 1) * V, V))) return "open frontier is not the last row";
    if (!s->AcceptVerified(a, &error)) return "accept: " + error;
    if (s->VerificationOpen()) return "verification still open";
    if (s->Position() != p + a || s->executor().Position() != p + a) {
      return "position after accepting " + std::to_string(a) + " of " + std::to_string(k) + " is " +
             std::to_string(s->Position()) + ", not " + std::to_string(p + a);
    }
    if (s->tokens() != Slice(plain.S, 0, p + a)) return "tokens after accept";
    if (!SameBits(s->Logits(), plain.Row(p + a - 1))) return "frontier logits after accept differ from plain";
    ++*rows_compared;
    for (std::size_t j = 0; j < kContinue; ++j) {
      if (!s->Evaluate(plain.S[p + a + j], &error)) return "evaluate: " + error;
      if (!SameBits(s->Logits(), plain.Row(p + a + j))) return "continuation logits differ at step " + std::to_string(j);
      if (Argmax(s->Logits()) != Argmax(plain.Row(p + a + j))) return "continuation token differs";
      ++*rows_compared;
    }
    return "";
  } catch (const std::exception& e) {
    return std::string("exception: ") + e.what();
  }
}

/// Anchors that put a k-row pass across `boundary`: rows [p, p+k) contain it
/// for k >= 2 (k = 1: the single row is the boundary position).
std::size_t Anchor(std::size_t boundary, std::uint32_t k) {
  return std::max<std::size_t>(1, boundary - k / 2);
}

std::vector<std::size_t> Boundaries(const TinyFixture& f, const EngineModel& m) {
  return {f.window, 2UL * f.window + 3, m.RingRows(), 2UL * m.RingRows()};
}

// ---- 1 ------------------------------------------------------------------------------

struct Matrix {
  std::size_t scenarios{0}, failed{0}, rows{0};
  std::string first_failure;
  std::set<std::pair<std::uint32_t, std::uint32_t>> failing;  ///< (k, a)
};

Matrix RunMatrix(const std::shared_ptr<EngineModel>& model, const Plain& plain, const TinyFixture& f,
                 const std::vector<std::size_t>& boundaries, Session::Faults faults) {
  Matrix m;
  (void)f;
  for (const std::size_t b : boundaries) {
    for (std::uint32_t k = 1; k <= kMaxVerifyRows; ++k) {
      const std::size_t p = Anchor(b, k);
      for (std::uint32_t a = 0; a <= k; ++a) {
        ++m.scenarios;
        const std::string r = Scenario(model, plain, p, k, a, faults, &m.rows);
        if (!r.empty()) {
          ++m.failed;
          m.failing.insert({k, a});
          if (m.first_failure.empty()) {
            m.first_failure = "anchor " + std::to_string(p) + " k " + std::to_string(k) + " a " + std::to_string(a) + ": " + r;
          }
        }
      }
    }
  }
  return m;
}

std::size_t VerifyRewindContinue(const TinyFixture& f, const DraftBackend& be,
                                 const std::shared_ptr<EngineModel>& model, const Plain& plain, int only_boundary) {
  auto boundaries = Boundaries(f, *model);
  if (only_boundary >= 0) boundaries = {boundaries.at(static_cast<std::size_t>(only_boundary))};
  const Matrix m = RunMatrix(model, plain, f, boundaries, {});
  Require(m.failed == 0, be.name + "/" + f.name + ": " + std::to_string(m.failed) + " of " +
                             std::to_string(m.scenarios) + " scenarios differ from plain; first: " + m.first_failure);
  std::printf("  %s/%s: 1. %zu scenarios (k 1..16, every a, anchors across %zu boundar%s of W=%u, 2W+3=%u, ring %u, 2 x ring), "
              "%zu logits rows compared: all bit-identical to plain 1-row decoding\n",
              be.name.c_str(), f.name.c_str(), m.scenarios, boundaries.size(), boundaries.size() == 1 ? "y" : "ies", f.window,
              2 * f.window + 3, model->RingRows(), m.rows);
  return m.scenarios;
}

// ---- 2, 3 ---------------------------------------------------------------------------

struct CycleStats {
  std::size_t cycles{0}, sweeps{0}, observed{0}, tap_rewinds{0}, positions_with_taps{0};
};

CycleStats Cycles(const TinyFixture& f, const DraftBackend& be, const std::shared_ptr<EngineModel>& model,
                  const Plain& plain, bool taps, std::size_t want_cycles) {
  Rng rng(0xD2AF7ULL + f.window * 131 + (be.hostops ? 7 : 0) + (taps ? 1000 : 0));
  CycleStats stats;
  const std::size_t V = plain.vocab;
  const std::size_t TW = taps ? model->executor_config().capture_layers.size() * model->weights().config.embedding_length : 0;
  std::string error;
  auto s = NewSession(model);
  std::vector<float> held;  // taps the "caller" holds: [position][TW]
  const auto take = [&] {
    if (!taps) return;
    auto t = s->TakeTaps();
    if (t.rows == 0) return;
    Require(t.first_position <= held.size() / TW, "taps: a gap before first_position " + std::to_string(t.first_position));
    if (t.first_position < held.size() / TW) ++stats.tap_rewinds;
    held.resize(std::size_t{t.first_position} * TW);
    held.insert(held.end(), t.data.begin(), t.data.end());
  };
  while (stats.cycles < want_cycles) {
    ++stats.sweeps;
    std::size_t pos = 1 + rng.Below(40);
    Require(s->Sync(Slice(plain.S, 0, pos), &error), error);
    take();
    std::vector<std::vector<float>> seen(kSeq);
    seen[pos - 1].assign(s->Logits().begin(), s->Logits().end());
    while (pos + 2 <= 80 && stats.cycles < want_cycles) {
      const std::uint32_t k = 1 + rng.Below(std::min<std::uint32_t>(16, static_cast<std::uint32_t>(kSeq - pos)));
      const std::uint32_t a = rng.Below(k + 1);
      Require(s->Verify(Candidates(plain, pos, k, a), &error), error);
      for (std::uint32_t i = 0; i < a; ++i) {
        seen[pos + i].assign(s->VerifyLogits().begin() + static_cast<std::ptrdiff_t>(i * V),
                             s->VerifyLogits().begin() + static_cast<std::ptrdiff_t>((i + 1) * V));
      }
      if (taps) {
        bool refused = false;
        try {
          (void)s->TakeTaps();
        } catch (const std::logic_error&) {
          refused = true;
        }
        Require(refused, "TakeTaps during an open verification must be refused");
      }
      Require(s->AcceptVerified(a, &error), error);
      pos += a;
      Require(s->Position() == pos && s->tokens() == Slice(plain.S, 0, pos), "tokens after a cycle");
      Require(SameBits(s->Logits(), plain.Row(pos - 1)), "frontier logits after a cycle differ from plain");
      take();
      if (a == 0 || rng.Below(3) == 0) {  // the bonus / correction token, one plain row
        Require(s->Evaluate(plain.S[pos], &error), error);
        seen[pos].assign(s->Logits().begin(), s->Logits().end());
        ++pos;
        take();
      }
      ++stats.cycles;
    }
    // The whole sequence against plain decoding, then more plain steps.
    for (std::size_t i = 0; i < pos; ++i) {
      if (seen[i].empty()) continue;
      Require(SameBits(seen[i], plain.Row(i)), "observed logits at position " + std::to_string(i) + " differ from plain");
      ++stats.observed;
    }
    for (std::size_t j = 0; j < kContinue && pos < kSeq; ++j, ++pos) {
      Require(s->Evaluate(plain.S[pos], &error), error);
      Require(SameBits(s->Logits(), plain.Row(pos)), "plain step after cycles differs");
      take();
    }
    if (taps) {
      Require(held.size() == pos * TW, "held taps cover " + std::to_string(held.size() / TW) + " positions, not " + std::to_string(pos));
      Require(std::memcmp(held.data(), plain.taps.data(), held.size() * sizeof(float)) == 0,
              "captured rows differ from the taps of plain decoding");
      stats.positions_with_taps += pos;
    }
  }
  return stats;
}

// ---- 4 ------------------------------------------------------------------------------

void Twins(const TinyFixture& f, const DraftBackend& be, const std::shared_ptr<EngineModel>& model, const Plain& plain) {
  const std::vector<std::size_t> one{2UL * f.window + 3, model->RingRows()};
  struct Case {
    const char* name;
    Session::Faults faults;
    bool (*should_fail)(std::uint32_t k, std::uint32_t a);
  };
  const Case cases[] = {
      // Rewinds to the end - a, so the position is right only when k == 2a.
      {"rewind from the logical end", {.rewind_from_end = true}, [](std::uint32_t k, std::uint32_t a) { return k != 2 * a; }},
      // The frontier stays the last verified row's: right only when everything was kept.
      {"stale frontier logits", {.stale_frontier = true}, [](std::uint32_t k, std::uint32_t a) { return a < k; }},
  };
  for (const Case& c : cases) {
    const Matrix m = RunMatrix(model, plain, f, one, c.faults);
    std::size_t expected = 0;
    bool exact = true;
    for (std::uint32_t k = 1; k <= kMaxVerifyRows; ++k) {
      for (std::uint32_t a = 0; a <= k; ++a) {
        const bool want = c.should_fail(k, a);
        expected += want ? 1 : 0;
        exact = exact && (m.failing.count({k, a}) != 0) == want;
      }
    }
    Require(m.failed > 0, be.name + "/" + f.name + ": the twin '" + c.name + "' passed all " + std::to_string(m.scenarios) + " scenarios");
    Require(m.failing.size() == expected, be.name + "/" + f.name + ": the twin '" + c.name + "' fails " + std::to_string(m.failing.size()) +
                                              " (k, a) pairs, expected " + std::to_string(expected));
    Require(exact, be.name + "/" + f.name + ": the twin '" + c.name + "' fails a different set of (k, a) than expected");
    std::printf("  %s/%s: 4. twin '%s': %zu of %zu scenarios fail, exactly the %zu of 152 (k, a) pairs where it must, at both anchors (first: %s)\n",
                be.name.c_str(), f.name.c_str(), c.name, m.failed, m.scenarios, expected, m.first_failure.c_str());
  }
}

void TapTwin(const TinyFixture& f, const DraftBackend& be, const std::shared_ptr<EngineModel>& model, const Plain& plain) {
  // keep_rejected_taps: the rejected rows stay pending, so the next rows are not contiguous with them.
  std::string error;
  std::size_t wrong = 0, total = 0;
  const std::size_t V = plain.vocab;
  (void)V;
  const std::size_t TW = model->executor_config().capture_layers.size() * model->weights().config.embedding_length;
  const std::size_t p = f.window + 1;
  for (std::uint32_t k = 1; k <= kMaxVerifyRows; ++k) {
    for (std::uint32_t a = 0; a <= k; ++a) {
      auto s = NewSession(model);
      s->SetFaults({.keep_rejected_taps = true});
      Require(s->Sync(Slice(plain.S, 0, p), &error), error);
      (void)s->TakeTaps();
      Require(s->Verify(Candidates(plain, p, k, a), &error), error);
      Require(s->AcceptVerified(a, &error), error);
      const auto t = s->TakeTaps();
      ++total;
      const bool good = t.rows == a && t.first_position == p && t.data.size() == a * TW &&
                        std::memcmp(t.data.data(), plain.taps.data() + p * TW, t.data.size() * sizeof(float)) == 0;
      wrong += good ? 0 : 1;
      Require(good == (a == k), be.name + "/" + f.name + ": taps twin outcome at k " + std::to_string(k) + " a " + std::to_string(a));
    }
  }
  Require(wrong > 0, "the taps twin passed everything");
  std::printf("  %s/%s: 4. twin 'rejected rows' taps kept': %zu of %zu scenarios hand out wrong taps (all with a < k)\n",
              be.name.c_str(), f.name.c_str(), wrong, total);
}

// ---- 5 ------------------------------------------------------------------------------

using Bytes = std::vector<std::byte>;

class HostStream final : public cache::Stream {
 public:
  [[nodiscard]] cache::TransferResult Synchronize() noexcept override { return cache::TransferResult::kSucceeded; }
};

class NullGuard final : public cache::MutationGuard {
 public:
  void BeforeOverwrite(cache::ComponentId, cache::Rows, cache::Rows) override {}
  void BeforeRelease(cache::ComponentId, cache::Rows, cache::Rows) noexcept override {}
  void AfterReset() noexcept override {}
};

struct Rig {
  std::shared_ptr<EngineModel> model;
  NullGuard guard;
  std::unique_ptr<ContinuationAdapter> adapter;
  std::unique_ptr<cache::Slot> slot;
  HostStream stream;
  Rig(const TinyFixture& f, const DraftBackend& be) {
    model = Armed(f, be, 32);  // a ring with slack: a checkpoint right after a deep rewind is capturable
    adapter = std::make_unique<ContinuationAdapter>(model, cache::Identity{1, 2, 3});
    slot = adapter->CreateSlot(guard);
  }
  Session& session() { return adapter->GetSession(*slot); }
};

struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<Bytes> data;
  std::vector<cache::Rows> first;
  cache::Rows n{0};
};

Checkpoint Capture(Rig& r) {
  Checkpoint ck;
  ck.positions = r.adapter->Positions(*r.slot);
  ck.n = ck.positions.front().valid_rows;
  for (const auto& d : r.adapter->Components()) {
    Bytes bytes;
    cache::Rows first = 0;
    if (cache::HasRows(d.kind)) {
      first = cache::FirstChunkRow(d, ck.n);
      bytes.resize((ck.n - first) * d.row_bytes);
      Require(r.adapter->CopyRowsOut(*r.slot, d.id, first, ck.n, bytes, r.stream).Wait() == cache::TransferResult::kSucceeded,
              "CopyRowsOut");
    } else {
      bytes.resize(d.state_bytes);
      Require(r.adapter->CapturePrivate(*r.slot, d.id, bytes, r.stream).Wait() == cache::TransferResult::kSucceeded,
              "CapturePrivate");
    }
    ck.data.push_back(std::move(bytes));
    ck.first.push_back(first);
  }
  return ck;
}

bool Restore(Rig& r, const Checkpoint& ck) {
  const auto comps = r.adapter->Components();
  r.adapter->BeginRestore(*r.slot, ck.positions);
  for (std::size_t k = 0; k < comps.size(); ++k) {
    const auto& d = comps[k];
    if (cache::HasRows(d.kind)) {
      Require(r.adapter->CopyRowsIn(*r.slot, d.id, ck.first[k], ck.n, ck.data[k], r.stream).Wait() == cache::TransferResult::kSucceeded,
              "CopyRowsIn");
    } else {
      Require(r.adapter->LoadPrivate(*r.slot, d.id, ck.data[k], r.stream).Wait() == cache::TransferResult::kSucceeded,
              "LoadPrivate");
    }
  }
  return r.adapter->Validate(*r.slot, ck.positions);
}

/// Drafted route to S[0, n): prompt, then verify/rewind cycles of the given (k, a) shapes, plain rows between.
void DraftTo(Session& s, const Plain& plain, std::size_t start, std::size_t n, const std::vector<std::pair<std::uint32_t, std::uint32_t>>& shapes) {
  std::string error;
  Require(s.Sync(Slice(plain.S, 0, start), &error), error);
  std::size_t pos = start, i = 0;
  while (pos < n) {
    auto [k, a] = shapes[i++ % shapes.size()];
    k = static_cast<std::uint32_t>(std::min<std::size_t>(k, n - pos));
    a = std::min(a, k);
    Require(s.Verify(Candidates(plain, pos, k, a), &error), error);
    Require(s.AcceptVerified(a, &error), error);
    pos += a;
    if (a == 0 && pos < n) {
      Require(s.Evaluate(plain.S[pos], &error), error);
      ++pos;
    }
  }
}

void Snapshots(const TinyFixture& f, const DraftBackend& be, const Plain& plain) {
  struct Route {
    std::size_t start;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> shapes;
  };
  const std::vector<Route> routes{
      {3, {{16, 0}}},                       // ends on a deepest rewind: everything rejected, then one plain row
      {5, {{16, 7}, {9, 9}, {4, 0}}},       // mixed
      {2, {{12, 12}, {16, 16}, {1, 1}}},    // everything accepted
  };
  std::size_t restores = 0;
  for (const std::size_t n : {std::size_t{40}, std::size_t{61}, std::size_t{73}}) {
    Rig plain_rig(f, be);
    std::string error;
    Require(plain_rig.session().Sync(Slice(plain.S, 0, 1), &error), error);
    for (std::size_t i = 1; i < n; ++i) Require(plain_rig.session().Evaluate(plain.S[i], &error), error);
    const Checkpoint want = Capture(plain_rig);
    for (const Route& route : routes) {
      Rig drafted(f, be);
      DraftTo(drafted.session(), plain, route.start, n, route.shapes);
      Require(drafted.session().Position() == n, "drafted route ended at the wrong position");
      const Checkpoint got = Capture(drafted);
      Require(got.n == want.n && got.first == want.first && got.data.size() == want.data.size(), "checkpoint layout");
      for (std::size_t c = 0; c < want.data.size(); ++c) {
        Require(got.data[c] == want.data[c], be.name + "/" + f.name + ": component " + std::to_string(c) +
                                                   " of a checkpoint at " + std::to_string(n) + " after drafted steps differs from plain");
      }
      // Restore the drafted checkpoint into a fresh rig and continue: same logits as plain.
      Rig fresh(f, be);
      Require(Restore(fresh, got), "restore of a checkpoint taken after drafted steps was refused");
      Require(SameBits(fresh.session().Logits(), plain.Row(n - 1)), "restored frontier logits");
      for (std::size_t j = 0; j < kContinue && n + j < kSeq; ++j) {
        Require(fresh.session().Evaluate(plain.S[n + j], &error), error);
        Require(SameBits(fresh.session().Logits(), plain.Row(n + j)), "continuation after restore differs from plain");
      }
      ++restores;
      // Prefix reuse: a new text sharing the first half, synced onto the drafted session and onto a plain one.
      auto target = Slice(plain.S, 0, n / 2);
      for (std::size_t i = 0; i < 9; ++i) target.push_back((plain.S[n / 2 + i] + 5) % 256);
      Rig reuse_plain(f, be);
      Require(reuse_plain.session().Sync(Slice(plain.S, 0, n), &error), error);
      Require(drafted.session().Sync(target, &error), error);
      Require(reuse_plain.session().Sync(target, &error), error);
      Require(SameBits(drafted.session().Logits(), reuse_plain.session().Logits()), "prefix reuse after drafted steps differs from plain");
      for (std::size_t j = 0; j < kContinue; ++j) {
        const std::int32_t t = static_cast<std::int32_t>((j * 37 + 11) % 256);
        Require(drafted.session().Evaluate(t, &error), error);
        Require(reuse_plain.session().Evaluate(t, &error), error);
        Require(SameBits(drafted.session().Logits(), reuse_plain.session().Logits()), "continuation after prefix reuse differs");
      }
    }
  }
  std::printf("  %s/%s: 5. checkpoints at 40, 61, 73 after 3 drafted routes each: every component equals the plain one; "
              "%zu restores continue 8 steps and prefix-reuse syncs equal plain, bit for bit\n",
              be.name.c_str(), f.name.c_str(), restores);
}

// ---- 6 ------------------------------------------------------------------------------

void Refusals(const TinyFixture& f, const DraftBackend& be, const Plain& plain) {
  std::string error;
  const auto model = Armed(f, be, kPrefillRows, {0, 2});
  {
    auto s = NewSession(model);
    Require(!s->Verify(Slice(plain.S, 0, 2), &error), "Verify before any prompt must be refused");
    Require(!s->AcceptVerified(0, &error) && error.find("no open verification") != std::string::npos, "AcceptVerified without a verification");
    Require(s->Sync(Slice(plain.S, 0, 10), &error), error);
    Require(!s->Verify({}, &error), "an empty verification must be refused");
    Require(!s->Verify(Slice(plain.S, 10, 10 + 17), &error), "17 rows must be refused");
    Require(s->Position() == 10 && !s->VerificationOpen() && s->IsValid(), "a refused Verify changed the session");
    // The write guard gets the pass's plan, once, and may refuse.
    std::vector<WritePlan> plans;
    s->SetWriteGuard([&](const WritePlan& pl) { plans.push_back(pl); });
    Require(s->Verify(Slice(plain.S, 10, 15), &error), error);
    Require(plans.size() == 1 && plans[0].written == RowRange{10, 15}, "the write guard did not see the verification's rows");
    Require(!s->AcceptVerified(6, &error) && s->VerificationOpen() && s->Position() == 15, "accepting more than verified");
    Require(s->AcceptVerified(5, &error) && s->Position() == 15, "accepting everything");
    Require(!s->AcceptVerified(0, &error), "a second AcceptVerified");
    s->SetWriteGuard([](const WritePlan&) { throw std::runtime_error("guard says no"); });
    Require(!s->Verify(Slice(plain.S, 15, 18), &error) && error.find("guard says no") != std::string::npos, "guard refusal");
    s->SetWriteGuard({});
    Require(s->Position() == 15 && s->IsValid() && !s->VerificationOpen(), "a refused guard changed the session");
    // Any other call settles an open verification (keeping all of it).
    Require(s->Verify(Slice(plain.S, 15, 18), &error) && s->Evaluate(plain.S[18], &error) && !s->VerificationOpen() && s->Position() == 19,
            "Evaluate settles");
    Require(s->Verify(Slice(plain.S, 19, 21), &error) && s->Sync(Slice(plain.S, 0, 20), &error) && !s->VerificationOpen() &&
                s->VerifyLogits().empty() && s->Position() == 20,
            "Sync settles");
    Require(SameBits(s->Logits(), plain.Row(19)), "Sync after a verification gives the plain logits");
    // The context cap.
    Require(s->Verify(Slice(plain.S, 20, 22), &error) && s->AcceptVerified(0, &error), "verify and reject all");
    Require(SameBits(s->Logits(), plain.Row(19)), "rejecting all restores the frontier from before the pass");
  }
  {  // drafting off (the default): one logits row, Verify refuses more than one token
    auto off = LoadDraftModel(f.path, be, kContext, kPrefillRows, kChunk);
    auto s = NewSession(off);
    Require(off->MaxVerifyRows() == 1 && off->executor_config().capture_layers.empty(), "drafting must be off by default");
    Require(s->Sync(Slice(plain.S, 0, 10), &error), error);
    Require(!s->Verify(Slice(plain.S, 10, 12), &error), "2 rows with drafting off must be refused");
    Require(s->Verify(Slice(plain.S, 10, 11), &error) && s->AcceptVerified(1, &error) && SameBits(s->Logits(), plain.Row(10)),
            "1 row with drafting off");
    const auto t = s->TakeTaps();
    Require(t.rows == 0 && t.data.empty(), "taps with capture off");
  }
  {  // limits of the options
    ModelOptions o;
    o.verify_rows = kMaxVerifyRows + 1;
    o.allow_template_mismatch = true;
    o.backend = CpuBackend();
    std::string e;
    Require(EngineModel::Load(f.path, o, &e) == nullptr && e.find("verify rows") != std::string::npos, "verify_rows above the maximum");
  }
  std::printf("  %s/%s: 6. refusals, write guard, settling calls, drafting off by default: ok\n", be.name.c_str(), f.name.c_str());
}

}  // namespace

// Argument: "<group>:<fixture>" with group verify (check 1) or rest (checks 2 to 6) and
// fixture tinyA or tinyB; none = everything. The GPU gates run one group of one fixture
// per GPU-lock acquisition. "verify:<fixture>:<n>" (n = 0..3) runs one of the four anchor
// boundaries (W, 2W+3, ring, 2 x ring) only: the way to split a verify run that is too long.
int main(int argc, char** argv) {
  std::string filter = argc > 1 ? argv[1] : "";
  int only_boundary = -1;
  if (std::count(filter.begin(), filter.end(), ':') == 2) {
    only_boundary = filter.back() - '0';
    Require(only_boundary >= 0 && only_boundary <= 3 && filter.size() > 2 && filter[filter.size() - 2] == ':',
            "the boundary index must be 0 to 3");
    filter.resize(filter.size() - 2);
  }
  const auto wants = [&](const char* group, const TinyFixture& f) {
    return filter.empty() || filter == std::string(group) + ":" + f.name;
  };
  return Run([&] {
    std::size_t scenario_total = 0, groups = 0;
    for (const auto& be : DraftBackends()) {
      for (const auto& f : TinyFixtures()) {
        if (!wants("verify", f) && !wants("rest", f)) continue;
        const auto S = SyntheticTokens(f.window * 100 + 3 + (f.window == 8 ? 0 : 5), kSeq);
        const auto model = Armed(f, be);
        const Plain plain = RunPlain(model, S);
        // Armed but unused changes nothing: the unarmed model gives the same bits.
        const Plain unarmed = RunPlain(LoadDraftModel(f.path, be, kContext, kPrefillRows, kChunk), S);
        Require(SameBits(plain.logits, unarmed.logits), "verify_rows/capture armed but unused changed plain decoding");
        if (wants("verify", f)) {
          scenario_total += VerifyRewindContinue(f, be, model, plain, only_boundary);
          ++groups;
        }
        if (!wants("rest", f)) continue;
        ++groups;
        const CycleStats c = Cycles(f, be, model, plain, false, 64);
        std::printf("  %s/%s: 2. %zu verify/rewind cycles in %zu sweeps, %zu observed positions: whole sequence bit-identical to plain\n",
                    be.name.c_str(), f.name.c_str(), c.cycles, c.sweeps, c.observed);

        const std::vector<std::uint32_t> layers{0, 2, 4};
        const auto capmodel = Armed(f, be, kPrefillRows, layers);
        const Plain capplain = RunPlain(capmodel, S);
        Require(SameBits(capplain.logits, plain.logits), "capture changed plain decoding");
        const CycleStats t = Cycles(f, be, capmodel, capplain, true, 64);
        Require(t.tap_rewinds > 0, "no sweep exercised a rewind of the captured rows");
        std::printf("  %s/%s: 3. captured rows (layers 0, 2, 4) after %zu cycles in %zu sweeps (%zu positions, %zu hand-overs below what the caller held): "
                    "bit-identical to plain decoding's\n",
                    be.name.c_str(), f.name.c_str(), t.cycles, t.sweeps, t.positions_with_taps, t.tap_rewinds);

        Twins(f, be, model, plain);
        TapTwin(f, be, capmodel, capplain);
        Snapshots(f, be, plain);
        Refusals(f, be, plain);
      }
    }
    Require(groups > 0, "no group matches the filter '" + filter + "'");
    if (scenario_total > 0) std::printf("  total verify scenarios of check 1: %zu\n", scenario_total);
  });
}

// Component-cache adapter (src/continuation_adapter.*), driven the way the
// cache drives it, on the CPU backend and the tiny fixtures: checkpoint and
// restore, restores that straddle the sliding window, rewinds, a failed
// forward, and the save-before-overwrite calls. Every check has a broken twin.
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>

#include "src/continuation_adapter.hpp"
#include "tests/fixtures_text/engine_util.hpp"

using namespace ricursive_test;

namespace {

using Bytes = std::vector<std::byte>;
constexpr std::uint32_t kContext = 96;

class HostStream final : public cache::Stream {
 public:
  [[nodiscard]] cache::TransferResult Synchronize() noexcept override { return cache::TransferResult::kSucceeded; }
};

struct GuardCall {
  std::uint32_t id;
  cache::Rows first, end;
};

class RecordingGuard final : public cache::MutationGuard {
 public:
  void BeforeOverwrite(cache::ComponentId id, cache::Rows first, cache::Rows end) override {
    overwrites.push_back({id.value, first, end});
    if (on_overwrite) on_overwrite(id, first, end);
  }
  void BeforeRelease(cache::ComponentId id, cache::Rows first, cache::Rows end) noexcept override {
    releases.push_back({id.value, first, end});
  }
  void AfterReset() noexcept override { ++resets; }
  std::vector<GuardCall> overwrites, releases;
  int resets{0};
  std::function<void(cache::ComponentId, cache::Rows, cache::Rows)> on_overwrite;
};

struct Rig {
  std::shared_ptr<EngineModel> model;
  RecordingGuard guard;
  std::unique_ptr<ContinuationAdapter> adapter;
  std::unique_ptr<cache::Slot> slot;
  HostStream stream;

  Rig(const TinyFixture& f, const CpuExecutor::Faults* faults = nullptr, std::uint32_t context = kContext,
      ContinuationAdapter::Faults adapter_faults = {}) {
    model = LoadCpu(f.path, context, f.window + 4, 4, faults);
    adapter = std::make_unique<ContinuationAdapter>(model, cache::Identity{1, 2, 3});
    adapter->SetFaults(adapter_faults);
    slot = adapter->CreateSlot(guard);
  }
  Session& session() { return adapter->GetSession(*slot); }
  void Sync(const std::vector<std::int32_t>& tokens) {
    std::string error;
    Require(session().Sync(tokens, &error), "sync: " + error);
  }
};

struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<Bytes> data;          // per component
  std::vector<cache::Rows> first;   // per component (rows)
  cache::Rows n{0};
};

Checkpoint Capture(Rig& r) {
  Checkpoint ck;
  ck.positions = r.adapter->Positions(*r.slot);
  ck.n = ck.positions.front().valid_rows;
  const auto comps = r.adapter->Components();
  for (const auto& d : comps) {
    Bytes bytes;
    cache::Rows first = 0;
    if (cache::HasRows(d.kind)) {
      first = cache::FirstChunkRow(d, ck.n);
      bytes.resize((ck.n - first) * d.row_bytes);
      Require(r.adapter->CopyRowsOut(*r.slot, d.id, first, ck.n, bytes, r.stream).Wait() ==
                  cache::TransferResult::kSucceeded, "CopyRowsOut");
    } else {
      bytes.resize(d.state_bytes);
      Require(r.adapter->CapturePrivate(*r.slot, d.id, bytes, r.stream).Wait() ==
                  cache::TransferResult::kSucceeded, "CapturePrivate");
    }
    ck.data.push_back(std::move(bytes));
    ck.first.push_back(first);
  }
  return ck;
}

/// The cache's restore: every component, rows in two pieces, in reverse
/// order for odd components, private state in two pieces. Returns Validate.
bool Restore(Rig& r, const Checkpoint& ck) {
  const auto comps = r.adapter->Components();
  r.adapter->BeginRestore(*r.slot, ck.positions);
  for (std::size_t k = 0; k < comps.size(); ++k) {
    const auto& d = comps[k];
    const Bytes& bytes = ck.data[k];
    if (cache::HasRows(d.kind)) {
      const cache::Rows a = ck.first[k], b = ck.n, mid = a + (b - a) / 2;
      const auto piece = [&](cache::Rows from, cache::Rows to) {
        const auto span = std::span<const std::byte>(bytes).subspan((from - a) * d.row_bytes, (to - from) * d.row_bytes);
        Require(r.adapter->CopyRowsIn(*r.slot, d.id, from, to, span, r.stream).Wait() ==
                    cache::TransferResult::kSucceeded, "CopyRowsIn");
      };
      if (k % 2 == 0) {
        piece(a, mid);
        piece(mid, b);
      } else {
        piece(mid, b);  // any order
        piece(a, mid);
      }
    } else {
      const std::size_t half = bytes.size() / 2;
      Require(r.adapter->LoadPrivatePiece(*r.slot, d.id, half, std::span<const std::byte>(bytes).subspan(half), r.stream)
                  .Wait() == cache::TransferResult::kSucceeded, "LoadPrivatePiece");
      Require(r.adapter->LoadPrivatePiece(*r.slot, d.id, 0, std::span<const std::byte>(bytes).first(half), r.stream)
                  .Wait() == cache::TransferResult::kSucceeded, "LoadPrivatePiece");
    }
  }
  return r.adapter->Validate(*r.slot, ck.positions);
}

/// Every K/V row a continuation reads, by layer and part.
std::map<std::string, Bytes> Rows(Rig& r) {
  std::map<std::string, Bytes> out;
  auto& exec = r.session().executor();
  const ExecutorInfo& info = exec.Info();
  const std::uint32_t n = exec.Position();
  for (std::uint32_t l = 0; l < info.layers; ++l) {
    for (const auto part : {KvPart::kKey, KvPart::kValue}) {
      const std::uint32_t from = info.sliding[l] ? info.WindowFirst(n) : 0;
      Bytes bytes(std::size_t{n - from} * info.RowBytes());
      exec.ExportRows(l, part, {from, n}, bytes);
      out[std::to_string(l) + (part == KvPart::kKey ? "K" : "V")] = std::move(bytes);
    }
  }
  return out;
}

std::size_t DifferingRowSets(const std::map<std::string, Bytes>& a, const std::map<std::string, Bytes>& b) {
  std::size_t differing = a.size() == b.size() ? 0 : 1;
  for (const auto& [key, bytes] : a) {
    const auto it = b.find(key);
    if (it == b.end() || it->second != bytes) ++differing;
  }
  return differing;
}

double Greedy(Rig& a, Rig& b, std::size_t count, std::vector<std::int32_t>* tokens) {
  double worst = 0;
  std::string error;
  for (std::size_t i = 0; i < count; ++i) {
    const auto ta = Argmax(a.session().Logits());
    const auto tb = Argmax(b.session().Logits());
    tokens->push_back(ta);
    worst = std::max(worst, MaxAbsDiff(a.session().Logits(), b.session().Logits()));
    if (ta != tb) return std::numeric_limits<double>::infinity();  // the chains part
    Require(a.session().Evaluate(ta, &error), error);
    Require(b.session().Evaluate(tb, &error), error);
  }
  return std::max(worst, MaxAbsDiff(a.session().Logits(), b.session().Logits()));
}

struct RestoreOutcome {
  bool validated{false};
  std::size_t row_sets_differ{0};
  double logit_diff{0};
  double next16{0};
};

RestoreOutcome RestoreAt(const TinyFixture& f, std::uint32_t n, const CpuExecutor::Faults* target_faults,
                         ContinuationAdapter::Faults adapter_faults) {
  const auto t = LoadTokens(f.name, 0);
  Rig source(f);
  source.Sync(Slice(t, 0, n));
  const Checkpoint ck = Capture(source);
  Rig target(f, target_faults, kContext, adapter_faults);
  RestoreOutcome out;
  out.validated = Restore(target, ck);
  if (!out.validated) return out;
  Require(target.session().Position() == n && target.session().tokens() == source.session().tokens(),
          "restored tokens");
  out.row_sets_differ = DifferingRowSets(Rows(source), Rows(target));
  out.logit_diff = MaxAbsDiff(source.session().Logits(), target.session().Logits());
  std::vector<std::int32_t> chain;
  out.next16 = Greedy(source, target, 16, &chain);
  out.row_sets_differ += DifferingRowSets(Rows(source), Rows(target));
  return out;
}

void CheckpointAndRestore(const TinyFixture& f) {
  const std::uint32_t W = f.window;
  for (const std::uint32_t n : {W - 1, W, W + 1, 2 * W + 3, 40U}) {
    const auto out = RestoreAt(f, n, nullptr, {});
    Require(out.validated, f.name + ": restore at " + std::to_string(n) + " was refused");
    Require(out.row_sets_differ == 0 && out.logit_diff == 0 && out.next16 == 0,
            f.name + ": restore at " + std::to_string(n) + " differs (" + std::to_string(out.row_sets_differ) +
                " row sets, logits " + Sci(out.logit_diff) + ", next 16 " + Sci(out.next16) + ")");
  }
  std::cout << "  " << f.name << ": checkpoint/restore at " << W - 1 << ", " << W << ", " << W + 1 << ", " << 2 * W + 3
            << ", 40: rows byte-identical, frontier logits and the next 16 tokens equal (max abs 0); "
            << "pieces loaded in two parts, odd components in reverse order\n";

  // Twin 1: a restore that skips one layer's rows is refused (coverage).
  const auto skipped = RestoreAt(f, 2 * W + 3, nullptr, {.skip_layer_on_load = true});
  std::cout << "    twin skip-one-layer: Validate " << (skipped.validated ? "accepted" : "refused") << "\n";
  Require(!skipped.validated, f.name + ": a restore missing a layer must be refused");
  // Twin 2: the same, with the executor's coverage check off: accepted, and
  // wrong. Measured.
  CpuExecutor::Faults no_coverage;
  no_coverage.ledger = Executor::LedgerFault::kNoImportCoverage;
  const auto silent = RestoreAt(f, 2 * W + 3, &no_coverage, {.skip_layer_on_load = true});
  std::cout << "    twin skip-one-layer + no coverage check: accepted=" << silent.validated << ", differing row sets "
            << silent.row_sets_differ << ", next-16 logits max abs " << Sci(silent.next16) << "\n";
  Require(silent.validated && (silent.row_sets_differ > 0 || silent.next16 > 1e-3),
          f.name + ": the unchecked skipped-layer restore must visibly corrupt the state");
}

// ---- rewinds ------------------------------------------------------------------------------------------

struct RewindOutcome {
  double logit_diff{0};
  std::size_t row_sets_differ{0};
  bool stale_row_refused{false};
};

/// `prev` up to `c`, then `extra` tokens, the first of which differs from prev[c].
std::vector<std::int32_t> Diverge(const std::vector<std::int32_t>& prev, std::uint32_t c, std::size_t extra,
                                  std::int32_t salt) {
  auto out = Slice(prev, 0, c);
  for (std::size_t i = 0; i < extra; ++i) {
    out.push_back((prev[(c + i) % prev.size()] + salt + static_cast<std::int32_t>(i)) % 256);
  }
  return out;
}

RewindOutcome Rewinds(const TinyFixture& f, const CpuExecutor::Faults* faults) {
  const std::uint32_t W = f.window;
  const auto t = LoadTokens(f.name, 2);
  Rig r(f, faults);
  const std::uint32_t ring = r.model->RingRows();
  const std::uint32_t high = 60;
  const std::uint32_t rb = high - ring;  // what the ring retains after 60 tokens
  r.Sync(Slice(t, 0, high));
  // 1. Inside what the ring retains (the first window just fits).
  const auto s2 = Diverge(Slice(t, 0, high), rb + W - 1 + 3, 8, 1);
  r.Sync(s2);
  // 2. Just outside: the window before c starts one row below the ring's
  // oldest row, which the pass up to 60 overwrote. Positions are now below
  // the high water, which is what a rule from the logical end gets wrong.
  const auto s3 = Diverge(s2, rb + W - 2, 9, 2);
  r.Sync(s3);
  // The engine recomputed from 0 here, so the ring now ends at s3 (high water
  // |s3|); that is the oldest row that is still true afterwards.
  const std::uint32_t true_rb = static_cast<std::uint32_t>(s3.size()) - ring;
  // 3. Inside again, at a third position.
  const auto s4 = Diverge(s3, rb + W + 1, 4, 3);
  r.Sync(s4);
  Rig fresh(f);
  fresh.Sync(s4);
  RewindOutcome out;
  if (faults == nullptr) {
    Require(r.session().executor().RetainedBegin() == true_rb, "retained begin after the recompute");
  }
  out.logit_diff = MaxAbsDiff(r.session().Logits(), fresh.session().Logits());
  out.row_sets_differ = DifferingRowSets(Rows(r), Rows(fresh));
  // A sliding row below the true ring is refused, never served stale. (After
  // the recompute the good run's RetainedBegin is exactly true_rb.)
  for (const auto& c : r.adapter->Layout()) {
    if (c.descriptor.kind != cache::ComponentKind::kWindowRows) continue;
    Bytes bytes(c.descriptor.row_bytes);
    try {
      (void)r.adapter->CopyRowsOut(*r.slot, c.descriptor.id, true_rb - 1, true_rb, bytes, r.stream);
    } catch (const std::invalid_argument&) {
      out.stale_row_refused = true;
    }
    break;
  }
  return out;
}

void RewindChecks(const TinyFixture& f) {
  const auto good = Rewinds(f, nullptr);
  Require(good.logit_diff == 0 && good.row_sets_differ == 0, f.name + ": rewinds differ from a fresh run");
  Require(good.stale_row_refused, f.name + ": rows that left the ring must be refused");
  CpuExecutor::Faults stale;
  stale.ledger = Executor::LedgerFault::kRetentionFromEnd;
  const auto bad = Rewinds(f, &stale);
  std::cout << "  " << f.name << ": rewinds inside and outside the ring, two consecutive at different positions: "
            << "logits and rows equal a fresh run; rows below the ring refused\n    twin retention-from-end: logits "
            << Sci(bad.logit_diff) << ", " << bad.row_sets_differ << " row sets differ, stale row refused: "
            << bad.stale_row_refused << '\n';
  Require(bad.logit_diff > 1e-3 || bad.row_sets_differ > 0 || !bad.stale_row_refused,
          f.name + ": the retention twin must be caught");
}

// ---- failed forward -----------------------------------------------------------------------------------------

struct FailureOutcome {
  bool invalid_after_failure{false};
  bool queries_refused{false};
  bool recovered_equal{false};
  double rewind_continue_diff{0};
};

FailureOutcome Failure(const TinyFixture& f, const CpuExecutor::Faults* faults) {
  const auto t = LoadTokens(f.name, 3);
  Rig r(f, faults);
  std::string error;
  const std::uint32_t layers = r.model->config().block_count;
  r.Sync(Slice(t, 0, 30));
  Session& live = r.session();  // GetSession refuses an invalid slot
  live.SetTap([&](std::string_view name, int layer, const float*, std::size_t, std::size_t) {
    if (name == "l_out" && layer == static_cast<int>(layers / 2)) throw std::runtime_error("injected device error");
  });
  Require(!live.Sync(Slice(t, 0, 50), &error), "an injected failure must fail the sync");
  FailureOutcome out;
  out.invalid_after_failure = !r.slot->IsValid();
  try {
    (void)r.adapter->Positions(*r.slot);
  } catch (const std::logic_error&) {
    out.queries_refused = true;
  }
  live.SetTap({});
  if (!out.invalid_after_failure) {
    // The twin keeps going: rewind to 20 and continue, as a cache would.
    auto& s = live;
    Require(s.Sync(Slice(t, 0, 25), &error), error);
    std::vector<std::int32_t> mix = Slice(t, 0, 20);
    for (std::size_t i = 0; i < 12; ++i) mix.push_back((t[20 + i] + 9) % 256);
    Require(s.Sync(mix, &error), error);
    Rig fresh(f);
    fresh.Sync(mix);
    out.rewind_continue_diff = MaxAbsDiff(s.Logits(), fresh.session().Logits());
    return out;
  }
  Require(r.adapter->Invalidate(*r.slot), "Invalidate");
  Require(r.slot->IsValid() && r.guard.resets == 1, "Invalidate leaves a valid empty slot and reports AfterReset");
  r.Sync(Slice(t, 0, 40));
  Rig fresh(f);
  fresh.Sync(Slice(t, 0, 40));
  out.recovered_equal = MaxAbsDiff(r.session().Logits(), fresh.session().Logits()) == 0;
  return out;
}

void FailureChecks(const TinyFixture& f) {
  const auto good = Failure(f, nullptr);
  Require(good.invalid_after_failure && good.queries_refused && good.recovered_equal,
          f.name + ": a failed forward must invalidate the slot and recovery must equal a fresh run");
  CpuExecutor::Faults keeps;
  keeps.ledger = Executor::LedgerFault::kFailureKeepsReady;
  const auto bad = Failure(f, &keeps);
  std::cout << "  " << f.name << ": failed forward -> slot invalid, Positions/CopyRowsOut refused, Invalidate recovers "
            << "(equal to fresh)\n    twin failed-forward-stays-ready: slot valid=" << !bad.invalid_after_failure
            << ", rewind-and-continue logits off by " << Sci(bad.rewind_continue_diff) << '\n';
  Require(!bad.invalid_after_failure && bad.rewind_continue_diff > 1e-3, f.name + ": the failure twin must be caught");
}

// ---- save before overwrite ----------------------------------------------------------------------------------------

struct GuardOutcome {
  std::size_t window_row_reads_in_guard{0};
  std::size_t calls{0};
  bool boundaries_match{false};
  bool evicted_gone_afterwards{true};
  bool written_covered{true};
};

GuardOutcome Guarding(const TinyFixture& f, ContinuationAdapter::Faults faults) {
  const std::uint32_t W = f.window;
  const auto t = LoadTokens(f.name, 1);
  Rig r(f, nullptr, kContext, faults);
  GuardOutcome out;
  std::vector<std::pair<std::uint32_t, std::pair<cache::Rows, cache::Rows>>> evicted_ranges;
  const std::uint32_t ring = r.model->RingRows();
  std::map<std::uint32_t, cache::ComponentDescriptor> by_id;
  for (const auto& d : r.adapter->Components()) by_id[d.id.value] = d;
  r.guard.on_overwrite = [&](cache::ComponentId id, cache::Rows first, cache::Rows end) {
    const auto& d = by_id.at(id.value);
    if (d.kind != cache::ComponentKind::kWindowRows || end > r.session().Position() || first >= end) return;
    // Rows that exist now and are about to be overwritten: still readable here.
    Bytes bytes((end - first) * d.row_bytes);
    (void)r.adapter->CopyRowsOut(*r.slot, id, first, end, bytes, r.stream);
    ++out.window_row_reads_in_guard;
    evicted_ranges.push_back({id.value, {first, end}});
  };
  // Three full passes and a 4-row remainder: 64 tokens on tinyB (window 16), all the token file holds.
  const auto tokens = Slice(t, 0, 3 * (W + 4) + 4);
  r.Sync(tokens);
  out.calls = r.guard.overwrites.size();
  // The written ranges of the token component are the prefill passes.
  std::vector<cache::Rows> ends;
  for (const auto& c : r.guard.overwrites) {
    if (c.id == 1 && (ends.empty() || ends.back() != c.end)) ends.push_back(c.end);
  }
  out.boundaries_match = ends == r.adapter->PlanPrefill(*r.slot, 0, tokens.size());
  // Every K/V component saw its written ranges.
  for (const auto& d : r.adapter->Components()) {
    if (!cache::HasRows(d.kind)) continue;
    cache::Rows covered = 0;
    for (const auto& c : r.guard.overwrites) {
      if (c.id == d.id.value && c.first == covered) covered = c.end;
    }
    if (covered < tokens.size()) out.written_covered = false;
  }
  // The evicted rows are gone after the pass: reading them is refused.
  const std::uint32_t rb = r.session().executor().RetainedBegin();
  (void)ring;
  for (const auto& [id, range] : evicted_ranges) {
    if (range.first >= rb) continue;
    Bytes bytes((range.second - range.first) * by_id.at(id).row_bytes);
    try {
      (void)r.adapter->CopyRowsOut(*r.slot, cache::ComponentId{id}, range.first, std::min<cache::Rows>(range.second, rb), bytes, r.stream);
      out.evicted_gone_afterwards = false;
    } catch (const std::invalid_argument&) {
    }
  }
  return out;
}

void GuardChecks(const TinyFixture& f) {
  const auto good = Guarding(f, {});
  Require(good.window_row_reads_in_guard > 0, f.name + ": the guard must see evictions of rows that still exist");
  Require(good.boundaries_match && good.written_covered && good.evicted_gone_afterwards,
          f.name + ": guard calls do not match the prefill plan");
  const auto bad = Guarding(f, {.no_guard = true});
  std::cout << "  " << f.name << ": " << good.calls << " BeforeOverwrite calls for a 3-pass prefill; "
            << good.window_row_reads_in_guard << " window ranges read back inside the guard (rows still valid), then "
            << "refused after the pass; token-component passes equal PlanPrefill\n    twin no-guard: " << bad.calls
            << " calls\n";
  Require(bad.calls == 0 && bad.window_row_reads_in_guard == 0 && !bad.boundaries_match,
          f.name + ": the no-guard twin must be caught");
}

// ---- layout ---------------------------------------------------------------------------------------------------------

void Layout(const TinyFixture& f) {
  Rig r(f);
  const Config& c = r.model->config();
  const auto comps = r.adapter->Components();
  Require(comps.size() == 1 + 2 * c.block_count + 2, "component count");
  std::size_t window = 0, append = 0, priv = 0;
  for (const auto& comp : r.adapter->Layout()) {
    const auto& d = comp.descriptor;
    if (d.kind == cache::ComponentKind::kWindowRows) {
      ++window;
      Require(c.IsSliding(comp.layer) && d.window_rows == c.sliding_window - 1 && d.rows_per_chunk == 4 &&
                  d.row_bytes == std::size_t{c.KvWidth()} * 4, "window component");
    } else if (d.kind == cache::ComponentKind::kAppendRows) {
      ++append;
    } else {
      ++priv;
    }
  }
  const std::size_t sliding = static_cast<std::size_t>(std::ranges::count(c.sliding, true));
  Require(window == 2 * sliding && append == 1 + 2 * (c.block_count - sliding) && priv == 2, "kinds");
  Require(!r.adapter->BackingPlan(std::size_t{1} << 30).empty(), "backing plan");
  Rig other(f, nullptr, kContext - 16);
  Require(other.adapter->CompatibilityIdentity() != r.adapter->CompatibilityIdentity(),
          "identity must change with the context");
  std::cout << "  " << f.name << ": " << comps.size() << " components (" << window << " window, " << append
            << " append, " << priv << " private), identity changes with the context\n";
}

}  // namespace

int main() {
  return Run([] {
    for (const auto& f : TinyFixtures()) {
      Layout(f);
      CheckpointAndRestore(f);
      RewindChecks(f);
      FailureChecks(f);
      GuardChecks(f);
    }
  });
}

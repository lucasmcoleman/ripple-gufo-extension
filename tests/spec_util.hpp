// Helpers of tests/speculative_test.cpp, copied (that file stays untouched where it can): the models, the
// drafter setup, plain decoding as the truth, one drafted run compared with it, the proposal check against
// DrafterReference. Shared by tests/speculative_followup_test.cpp (docs/DRAFTER-S4.md, "Follow-ups").
#ifndef RICURSIVE_TESTS_SPEC_UTIL_HPP_
#define RICURSIVE_TESTS_SPEC_UTIL_HPP_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <utility>

#include "tests/draft_rig.hpp"
#include "tests/draft_util.hpp"

#include "hip/drafter_engine.hpp"
#include "src/drafter_engine_cpu.hpp"
#include "src/runner.hpp"

namespace ricursive_test {
namespace core = gufo::core;
namespace sampling = gufo::sampling;
namespace speculative = gufo::speculative;
namespace {

constexpr std::uint32_t kContext = 160;
constexpr std::uint32_t kPrefillRows = 20;
constexpr std::uint32_t kChunk = 4;
constexpr std::int32_t kStop = 510;  // tinyA/tinyB: eos and eot

bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

// ---- models -------------------------------------------------------------------------

std::shared_ptr<EngineModel> PlainModel(const TinyFixture& f, const DraftBackend& be, std::uint32_t rows = kPrefillRows) {
  return LoadDraftModel(f.path, be, kContext, rows, kChunk);
}

/// Verification rows only (the mock tests need no features).
std::shared_ptr<EngineModel> VerifyModel(const TinyFixture& f, const DraftBackend& be) {
  return LoadDraftModel(f.path, be, kContext, kPrefillRows, kChunk, [](ModelOptions& o) { o.verify_rows = kMaxVerifyRows; });
}

/// RICURSIVE_TEST_DRAFTER=ops: the proposals come from the drafter executor (hip/drafter_executor.cpp) instead
/// of the scalar CPU reference (the default, as in stage 4); the checks are the same, a proposal that differs
/// from the reference's only at a near-tie of the two choices' scores is counted and reported.
bool OpsDrafter() {
  const char* env = std::getenv("RICURSIVE_TEST_DRAFTER");
  const std::string name = env != nullptr ? env : "ref";
  Require(name == "ref" || name == "ops", "RICURSIVE_TEST_DRAFTER must be ref or ops, not " + name);
  return name == "ops";
}

std::size_t g_proposal_ties = 0;  ///< proposals that differed from the reference's at a near-tie only

std::shared_ptr<const SpeculativeSetup> Setup(const TinyFixture& f, const std::string& dir, std::uint32_t d) {
  std::string error;
  std::shared_ptr<const core::GgufReader> reader = core::GgufReader::OpenFile(f.path, &error);
  Require(reader != nullptr, "open " + f.path + ": " + error);
  auto loaded = LoadSpeculativeSetup(dir + "/drafter-" + f.name + "-q8_0.gguf", *reader, d, &error);
  Require(loaded != nullptr, "drafter of " + f.name + ": " + error);
  auto setup = std::make_shared<SpeculativeSetup>(*loaded);
  if (OpsDrafter()) {
    // The drafter of docs/DRAFTER-S2.md: the GPU drafter when the model runs on the device, else the same
    // executor on the host operations. F32 or F16 context storage as the model's cache type.
    setup->engine_factory = [](const SpeculativeSetup& s, const EngineModel& m) -> std::unique_ptr<DrafterEngine> {
      if (m.backend().uses_device) return MakeDefaultDrafterEngine(s, m);
      DrafterOptions o = s.drafter_options;
      o.kv_type = m.backend().kv_type;
      return MakeHostOpsDrafterEngine(s.weights, m.weights(), o);
    };
  } else {
    // The stage 4 drafter, whatever the build would choose by default (the GPU drafter on a device model).
    setup->engine_factory = [](const SpeculativeSetup& s, const EngineModel& m) -> std::unique_ptr<DrafterEngine> {
      return std::make_unique<CpuDrafterEngine>(s.weights, m.weights(), s.engine_faults);
    };
  }
  return setup;
}

std::shared_ptr<const SpeculativeSetup> WithDraftLength(const SpeculativeSetup& base, std::uint32_t d) {
  auto copy = std::make_shared<SpeculativeSetup>(base);
  copy->max_draft_tokens = d;
  return copy;
}

std::shared_ptr<EngineModel> ArmedModel(const TinyFixture& f, const DraftBackend& be, const SpeculativeSetup& setup,
                                        std::uint32_t rows = kPrefillRows) {
  return LoadDraftModel(f.path, be, kContext, rows, kChunk, [&](ModelOptions& o) { ArmForSpeculation(setup, o); });
}

// ---- plain decoding: the truth -------------------------------------------------------

struct Truth {
  std::vector<std::int32_t> prompt, gen;
  bool stopped{false};         ///< gen ends before `limit` because the model chose a stop token
  std::vector<float> rows;     ///< logits after position P-1 (the prompt), then after each generated token
  std::vector<float> taps;     ///< [position][tap][hidden] when captured
  std::size_t vocab{0}, tap_width{0};

  [[nodiscard]] std::span<const float> Row(std::size_t position) const {
    const std::size_t index = position + 1 - prompt.size();
    Require(position + 1 >= prompt.size() && (index + 1) * vocab <= rows.size(), "truth has no logits row for position " + std::to_string(position));
    return std::span<const float>(rows).subspan(index * vocab, vocab);
  }
  [[nodiscard]] std::span<const float> Taps(std::size_t position) const {
    Require((position + 1) * tap_width <= taps.size(), "truth has no taps for position " + std::to_string(position));
    return std::span<const float>(taps).subspan(position * tap_width, tap_width);
  }
  /// Prompt, generated tokens, and the stop token when that is how the run ended.
  [[nodiscard]] std::vector<std::int32_t> Sequence() const {
    auto s = prompt;
    s.insert(s.end(), gen.begin(), gen.end());
    if (stopped) s.push_back(kStop);
    return s;
  }
};

/// Plain greedy decoding: Sync(prompt), then one-row steps, at most `limit` tokens, until a stop token.
Truth RunTruth(const std::shared_ptr<EngineModel>& model, const std::vector<std::int32_t>& prompt, std::size_t limit) {
  Truth t;
  t.prompt = prompt;
  t.vocab = model->VocabSize();
  t.tap_width = model->executor_config().capture_layers.size() * model->weights().config.embedding_length;
  std::string error;
  auto s = NewSession(model);
  const auto take = [&](std::size_t first, std::size_t rows) {
    auto taps = s->TakeTaps();
    if (t.tap_width == 0) return;
    Require(taps.first_position == first && taps.rows == rows && taps.first_position * t.tap_width == t.taps.size(), "truth: taps out of order");
    t.taps.insert(t.taps.end(), taps.data.begin(), taps.data.end());
  };
  Require(s->Sync(prompt, &error), "truth sync: " + error);
  take(0, prompt.size());
  t.rows.insert(t.rows.end(), s->Logits().begin(), s->Logits().end());
  while (t.gen.size() < limit) {
    const std::int32_t token = Argmax(s->Logits());
    if (model->IsStopToken(token)) {
      t.stopped = true;
      break;
    }
    t.gen.push_back(token);
    const std::size_t position = s->Position();
    Require(s->Evaluate(token, &error), "truth evaluate: " + error);
    take(position, 1);
    t.rows.insert(t.rows.end(), s->Logits().begin(), s->Logits().end());
  }
  return t;
}

/// A prompt of `length` tokens whose plain run ends in a stop token (want_stop, between 6 and
/// limit-6 tokens in) or never reaches one within `limit` tokens.
Truth FindTruth(const std::shared_ptr<EngineModel>& model, std::size_t length, std::uint64_t seed0, std::size_t limit,
                bool want_stop) {
  for (std::uint64_t seed = seed0; seed < seed0 + 4000; ++seed) {
    Truth t = RunTruth(model, SyntheticTokens(seed, length), limit);
    if (want_stop ? (t.stopped && t.gen.size() >= 6 && t.gen.size() + 6 <= limit) : (!t.stopped && t.gen.size() == limit)) return t;
  }
  Require(false, "no prompt of " + std::to_string(length) + " tokens " + (want_stop ? "ends" : "never ends") + " in a stop token");
  return {};
}

// ---- one drafted run ---------------------------------------------------------------------

struct Drafted {
  std::string error;  ///< "" = every comparison held, else the first that did not
  std::size_t steps{0}, drafted{0}, accepted{0}, compared{0};
  std::vector<std::int32_t> out;
  bool stop{false};
};

using AfterStep = std::function<std::string(const DecodeResult&)>;

/// Drives `dec` until `limit` tokens or a stop; compares tokens, the logits behind every committed
/// position and the frontier with `truth` after every step. `after` may add checks.
Drafted RunDrafted(SpeculativeDecoder& dec, const Truth& truth, std::size_t limit, const AfterStep& after = {}) {
  Drafted run;
  Session& s = dec.session();
  struct Seen {
    std::uint32_t first;
    std::vector<float> rows;
  };
  std::vector<Seen> seen;
  dec.target().SetObserver([&](std::uint32_t first, std::span<const float> rows) { seen.push_back({first, {rows.begin(), rows.end()}}); });
  sampling::SamplerState sampler;
  const std::size_t vocab = truth.vocab;
  try {
    while (run.out.size() < limit && !run.stop) {
      seen.clear();
      const DecodeResult r = dec.Step(limit - run.out.size(), sampler, true);
      ++run.steps;
      run.drafted += r.draft_tokens;
      run.accepted += r.accepted_tokens;
      run.stop = r.stop;
      if (r.tokens.size() > limit - run.out.size()) return run.error = "a step emitted more than its budget", run;
      run.out.insert(run.out.end(), r.tokens.begin(), r.tokens.end());
      const std::uint32_t end = s.Position();
      for (const Seen& o : seen) {
        for (std::size_t i = 0; i < o.rows.size() / vocab; ++i) {
          const std::size_t position = o.first + i;
          if (position >= end) continue;  // a rejected row
          if (!SameBits(std::span<const float>(o.rows).subspan(i * vocab, vocab), truth.Row(position))) {
            return run.error = "logits row of position " + std::to_string(position) + " differs from plain decoding", run;
          }
          ++run.compared;
        }
      }
      if (!SameBits(s.Logits(), truth.Row(end - 1))) return run.error = "frontier logits at position " + std::to_string(end - 1) + " differ from plain", run;
      ++run.compared;
      if (after) {
        std::string e = after(r);
        if (!e.empty()) return run.error = std::move(e), run;
      }
    }
  } catch (const std::exception& e) {
    run.error = std::string("exception: ") + e.what();
    return run;
  }
  const bool want_stop = truth.stopped && truth.gen.size() < limit;
  const std::size_t want = want_stop ? truth.gen.size() : limit;
  if (run.out.size() != want || !std::equal(run.out.begin(), run.out.end(), truth.gen.begin())) {
    run.error = "emitted tokens differ from plain greedy decoding (" + std::to_string(run.out.size()) + " of " + std::to_string(want) + ")";
  } else if (run.stop != want_stop) {
    run.error = want_stop ? "the stop token was not reported" : "a stop was reported that plain decoding does not make";
  }
  const auto& stats = dec.verifier().GetStats();
  if (run.error.empty() && (stats.total_draft_tokens != run.drafted || stats.total_accepted_tokens != run.accepted || run.accepted > run.drafted)) {
    run.error = "the accounting differs from the verifier's: drafted " + std::to_string(run.drafted) + "/" +
                std::to_string(stats.total_draft_tokens) + ", accepted " + std::to_string(run.accepted) + "/" +
                std::to_string(stats.total_accepted_tokens);
  }
  return run;
}

// ---- 3 proposals equal the reference ---------------------------------------------------------

/// DrafterReference driven directly with plain decoding's taps. After a step in which the backend
/// proposed, its block must equal the reference's block for the same anchor, and must have read
/// only rows from `base` up to the anchor.
class ProposalCheck {
 public:
  ProposalCheck(const SpeculativeSetup& setup, const EngineModel& model, const Truth& truth, std::uint32_t base)
      : ref_(setup.weights, model.weights()), truth_(&truth), base_(base) {
    ref_.ResetTo(base);
  }
  std::string Check(DraftedSession& ds) {
    auto& backend = ds.backend();
    if (backend.proposals() == seen_) return "";
    seen_ = backend.proposals();
    const std::uint32_t p = backend.last_anchor_position();
    const auto sequence = truth_->Sequence();
    if (p >= sequence.size() || backend.last_anchor_token() != static_cast<sampling::TokenId>(sequence[p])) {
      return "the anchor token of position " + std::to_string(p) + " is not plain decoding's";
    }
    while (ref_.ContextRows() < p) {
      ref_.AppendContext(truth_->Taps(ref_.ContextRows()), 1);
    }
    const auto expect = ref_.DraftStep(sequence[p], p);
    const auto& got = backend.last_block();
    ++compared;
    if (got.context_first < base_ || got.context_end > p || got.context_first != expect.context_first || got.context_end != expect.context_end) {
      ++range_wrong;
      return "the drafter read rows [" + std::to_string(got.context_first) + ", " + std::to_string(got.context_end) + ") for the anchor at " +
             std::to_string(p) + " (rows given from " + std::to_string(base_) + ")";
    }
    // The engine may return fewer tokens than the reference's whole block (the draft-length hint): compare those.
    std::size_t row = 0;
    const std::size_t shared = std::min(got.tokens.size(), expect.tokens.size());
    while (row < shared && got.tokens[row] == expect.tokens[row]) ++row;
    if (row < shared || got.tokens.size() > expect.tokens.size()) {
      // A different token is acceptable only at a near-tie of the reference's own scores (another engine rounds
      // differently); the rows after it follow another path and are not compared.
      const std::size_t KT = expect.candidates.size() / std::max<std::size_t>(expect.tokens.size(), 1);
      const auto score_of = [&](std::int32_t id) -> std::optional<float> {
        for (std::size_t i = 0; i < KT && row < expect.tokens.size(); ++i)
          if (expect.candidates[row * KT + i] == id) return expect.scores[row * KT + i];
        return std::nullopt;
      };
      const auto a = row < shared ? score_of(got.tokens[row]) : std::nullopt, b = row < shared ? score_of(expect.tokens[row]) : std::nullopt;
      if (OpsDrafter() && a && b && std::fabs(static_cast<double>(*a) - *b) <= 2e-3 * (1.0 + std::fabs(static_cast<double>(*b)))) {
        ++g_proposal_ties;
        return "";
      }
      ++mismatched;
      return "the block proposed at position " + std::to_string(p) + " differs from the reference's";
    }
    return "";
  }
  /// Proposals made before this check began (a drafter that outlives requests) are not its business.
  void Skip(DraftedSession& ds) { seen_ = ds.backend().proposals(); }
  std::size_t compared{0}, mismatched{0}, range_wrong{0};

 private:
  DrafterReference ref_;
  const Truth* truth_;
  std::uint32_t base_;
  std::size_t seen_{0};
};

/// The invariants of the drafter's context after every step: it ends exactly at the session's
/// position, from `base`.
std::string ContextLedger(DraftedSession& ds, std::uint32_t base) {
  auto& e = ds.engine();
  if (e.End() != ds.decoder().session().Position() || e.Base() != base) {
    return "the drafter holds rows [" + std::to_string(e.Base()) + ", " + std::to_string(e.End()) + ") but the session is at " +
           std::to_string(ds.decoder().session().Position()) + " (expected from " + std::to_string(base) + ")";
  }
  if (ds.backend().desyncs() != 0 || ds.ingest().failures() != 0 || ds.health().failures() != 0 || ds.health().disabled()) {
    return "the drafter's bookkeeping reported " + std::to_string(ds.backend().desyncs()) + " desyncs, " + std::to_string(ds.ingest().failures()) +
           " ingest failures and " + std::to_string(ds.health().failures()) + " drafter failures (a real drafter bug must not hide behind the fallback)";
  }
  return "";
}


}  // namespace
}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_SPEC_UTIL_HPP_

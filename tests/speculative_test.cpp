// Drafted decoding through gufo's own SpeculativeVerifier (docs/DRAFTER-S4.md), on the CPU
// executor and the host-ops executor (or, through tests/run-gpu-gates.sh, the GPU executor),
// tinyA (window 8) and tinyB (window 16). The property every check protects: the emitted
// tokens, and the logits behind them, are bit for bit what plain 1-row decoding produces,
// whatever the draft is.
//
//   mock    1. mock draft back ends (always right, always wrong, random, right then wrong at every
//              split point, a stop token inside a draft) x every draft length 1..15 x prompts that
//              put the generation across W, 2W+3, the ring size and the ring wrap, the generation
//              limit inside a draft, a plain greedy run that really ends in a stop token
//           7. twin: the adapter keeps one more row than the verifier decided
//   real    2. the stage 1 tiny drafter end to end, every draft length: exactness, accounting,
//              the drafter's context ends exactly at the session's position after every step
//           3. proposals equal DrafterReference driven directly with the taps of plain decoding;
//              twins (anchor position, anchor token, context row short): output still exact,
//              proposals wrong
//           4. prefix reuse, a restored checkpoint, a full recompute after a drafted rejection
//              (the band of docs/DRAFTER-S4.md): output equals plain, test 3 holds again, the
//              drafter never reads a row it was not given
//           6. tap memory: one pass of rows at a time through a multi-pass Sync
//   driver     `gufo prompt --speculative dflash2` (RunRicursivePrompt, CPU): the plain run's text
//   runner  5. the serving runner (prefill, preview, DecodeStep, prefix reuse): the same tokens as
//              a runner without speculation; sampled, penalised requests draft nothing
//
// Argument: "all", "<group>:<fixture>" or a comma separated list of them (groups mock, real, runner, driver; fixtures tinyA, tinyB), then
// the directory of the tiny drafters (drafter-tinyA-q8_0.gguf, drafter-tinyB-q8_0.gguf).
// RICURSIVE_TEST_DRAFTER=ops: the proposals come from the drafter executor (docs/DRAFTER-S2.md, host operations,
// or the GPU drafter on a device backend) instead of the scalar CPU reference.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>
#include <map>
#include <optional>
#include <set>
#include <utility>

#include "tests/draft_rig.hpp"
#include "tests/draft_util.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::puts("SKIP: this build has no speculative verifier (ENGINE_ENABLE_HIP)");
  return 77;
}
#else

#include "hip/drafter_engine.hpp"
#include "src/drafter_engine_cpu.hpp"
#include "src/runner.hpp"

using namespace ricursive_test;
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

/// Plain decoding of `second` given the same call sequence as the drafted session saw: `first`'s prompt, its generated tokens
/// one row at a time (a drafted request ends with every token evaluated), then Sync(second) and one-row steps. Taps cover every
/// position of `second` (below the rewind they are the first run's).
Truth RunTruthAfter(const std::shared_ptr<EngineModel>& model, const Truth& first, std::size_t first_used,
                     const std::vector<std::int32_t>& second, std::size_t limit) {
  Truth t;
  t.prompt = second;
  t.vocab = model->VocabSize();
  t.tap_width = model->executor_config().capture_layers.size() * model->weights().config.embedding_length;
  std::string error;
  auto s = NewSession(model);
  const auto take = [&] {
    auto taps = s->TakeTaps();
    if (t.tap_width == 0 || taps.rows == 0) return;
    Require(taps.first_position * t.tap_width <= t.taps.size(), "truth: a hole in the taps");
    t.taps.resize(taps.first_position * t.tap_width);
    t.taps.insert(t.taps.end(), taps.data.begin(), taps.data.end());
  };
  Require(s->Sync(first.prompt, &error), "truth sync: " + error);
  take();
  for (std::size_t i = 0; i < first_used; ++i) {
    Require(s->Evaluate(first.gen[i], &error), "truth evaluate: " + error);
    take();
  }
  Require(s->Sync(second, &error), "truth second sync: " + error);
  take();
  t.rows.insert(t.rows.end(), s->Logits().begin(), s->Logits().end());
  while (t.gen.size() < limit) {
    const std::int32_t token = Argmax(s->Logits());
    if (model->IsStopToken(token)) {
      t.stopped = true;
      break;
    }
    t.gen.push_back(token);
    Require(s->Evaluate(token, &error), "truth evaluate: " + error);
    take();
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

// ---- 1 mock draft back ends --------------------------------------------------------------

enum class Mode { kRight, kWrong, kRandom, kSplit, kStopInside };

const char* ModeName(Mode m) {
  switch (m) {
    case Mode::kRight: return "always right";
    case Mode::kWrong: return "always wrong";
    case Mode::kRandom: return "random";
    case Mode::kSplit: return "right then wrong at every split";
    case Mode::kStopInside: return "a stop token inside the draft";
  }
  return "?";
}

class MockDraft final : public speculative::IDraftBackend {
 public:
  MockDraft(std::vector<std::int32_t> truth, Mode mode, std::uint64_t seed) : truth_(std::move(truth)), mode_(mode), rng_(seed) {}
  [[nodiscard]] std::string_view Name() const noexcept override { return "mock"; }
  [[nodiscard]] speculative::DraftProposal Propose(std::span<const sampling::TokenId> tokens, std::uint32_t position,
                                                   std::uint32_t max_tokens) override {
    speculative::DraftProposal p;
    p.start_pos = position;
    // What the verifier hands over: the sequence including the anchor, which is the plain token.
    if (tokens.size() != position + 1 || position >= truth_.size() || static_cast<std::int32_t>(tokens.back()) != truth_[position]) {
      ++bad_inputs;
    }
    const std::size_t split = step_ % (max_tokens + 1), stop_at = step_ % std::max<std::uint32_t>(max_tokens, 1);
    ++step_;
    for (std::uint32_t i = 0; i < max_tokens; ++i) {
      const std::size_t index = std::size_t{position} + 1 + i;
      if (index >= truth_.size()) break;  // beyond what plain decoding produces: nothing to draft
      const std::int32_t right = truth_[index];
      const std::int32_t wrong = static_cast<std::int32_t>((right + 1 + i) % 256);  // never the plain token, never a stop token
      std::int32_t token = right;
      switch (mode_) {
        case Mode::kRight: break;
        case Mode::kWrong: token = wrong; break;
        case Mode::kRandom: token = rng_.Below(2) == 0 ? right : wrong; break;
        case Mode::kSplit: token = i < split ? right : wrong; break;
        case Mode::kStopInside: token = i == stop_at ? kStop : right; break;
      }
      p.tokens.push_back(static_cast<sampling::TokenId>(token));
    }
    drafted += p.tokens.size();
    return p;
  }
  std::size_t bad_inputs{0}, drafted{0};

 private:
  std::vector<std::int32_t> truth_;
  Mode mode_;
  Rng rng_;
  std::size_t step_{0};
};

struct MockCase {
  const Truth* truth;
  Mode mode;
  std::uint32_t d;
  std::size_t limit;
};

Drafted RunMock(const std::shared_ptr<EngineModel>& model, const MockCase& c, SessionSpeculativeTarget::Faults faults) {
  auto session = NewSession(model);
  std::string error;
  Require(session->Sync(c.truth->prompt, &error), error);
  auto mock = std::make_unique<MockDraft>(c.truth->Sequence(), c.mode, 0xD1CEULL + c.d * 31 + static_cast<unsigned>(c.mode));
  MockDraft* raw = mock.get();
  SpeculativeDecoder dec(*session, std::move(mock), c.d, faults);
  Drafted run = RunDrafted(dec, *c.truth, c.limit);
  if (run.error.empty() && raw->bad_inputs != 0) run.error = "the verifier handed the draft a sequence that is not plain decoding's";
  const bool clean_stop = !c.truth->stopped;
  if (run.error.empty() && c.mode == Mode::kWrong && run.accepted != 0) run.error = "a wrong draft was accepted";
  if (run.error.empty() && c.mode == Mode::kRight && clean_stop && run.accepted != run.drafted) run.error = "a right draft was not accepted";
  return run;
}

void MockTests(const TinyFixture& f, const DraftBackend& be) {
  const auto model = VerifyModel(f, be);
  const std::size_t limit = 22;
  // Prompts: 5 (the generation crosses W and 2W+3), 23 (the ring and its wrap); and one whose
  // plain run ends in a stop token.
  std::vector<Truth> truths;
  for (const std::size_t length : {std::size_t{5}, std::size_t{23}}) {
    truths.push_back(FindTruth(model, length, 100 + length * 7, limit + 18, false));
  }
  truths.push_back(FindTruth(model, 11, 9000, limit, true));
  std::size_t runs = 0, compared = 0, drafted = 0, accepted = 0;
  for (const Mode mode : {Mode::kRight, Mode::kWrong, Mode::kRandom, Mode::kSplit, Mode::kStopInside}) {
    for (std::uint32_t d = 1; d <= kMaxVerifyRows - 1; ++d) {
      // Every draft length on the CPU executor; the host ops (the same orchestration the GPU runs, an order of
      // magnitude slower here) take a spread of lengths including both ends. The time budget is minutes.
      if (be.hostops && d != 1 && d != 2 && d != 3 && d != 5 && d != 8 && d != 15) continue;
      for (const Truth& truth : truths) {
        const Drafted run = RunMock(model, {&truth, mode, d, limit}, {});
        Require(run.error.empty(), be.name + "/" + f.name + " mock '" + ModeName(mode) + "' d=" + std::to_string(d) + " prompt " +
                                       std::to_string(truth.prompt.size()) + (truth.stopped ? " (stops)" : "") + ": " + run.error);
        ++runs;
        compared += run.compared;
        drafted += run.drafted;
        accepted += run.accepted;
      }
    }
  }
  // The generation limit inside a draft, explicitly: budgets of 1..4 tokens with 15 drafts on offer.
  for (std::size_t small = 1; small <= 4; ++small) {
    const Drafted run = RunMock(model, {&truths[1], Mode::kRight, 15, small}, {});
    Require(run.error.empty(), be.name + "/" + f.name + " limit " + std::to_string(small) + ": " + run.error);
    Require(run.out.size() == small, "the generation limit was not honoured");
    ++runs;
  }
  // The stop token must really have been inside a draft in a right-draft run on the stopping prompt.
  const Drafted stop_run = RunMock(model, {&truths.back(), Mode::kRight, 15, limit}, {});
  Require(stop_run.error.empty() && stop_run.stop && stop_run.drafted > 0, "the stopping prompt did not stop inside a draft");
  std::printf("  %s/%s: 1. mock drafts: %zu runs (5 modes x %s draft lengths x 3 prompts + 4 limits), %zu logits rows compared, %zu tokens drafted, "
              "%zu accepted: output and logits bit-identical to plain decoding in every run; the stopping prompt ends in a stop token inside a draft\n",
              be.name.c_str(), f.name.c_str(), runs, be.hostops ? "6 of the 15" : "all 15", compared, drafted, accepted);

  // 7. twin: the adapter reports one more accepted row than the verifier decided.
  std::size_t twin_runs = 0, twin_failed = 0;
  for (std::uint32_t d : {1u, 2u, 4u, 8u, 15u}) {
    for (const Mode mode : {Mode::kWrong, Mode::kRandom}) {
      const Drafted clean = RunMock(model, {&truths[1], mode, d, limit}, {});
      const Drafted twin = RunMock(model, {&truths[1], mode, d, limit}, {.accept_one_more = true});
      Require(clean.error.empty(), "the clean run of the twin case failed: " + clean.error);
      ++twin_runs;
      if (!twin.error.empty()) ++twin_failed;
      Require(mode != Mode::kWrong || !twin.error.empty(), be.name + "/" + f.name + ": the twin 'one more accepted row' passed an always-wrong draft at d=" +
                                                              std::to_string(d));
    }
  }
  Require(twin_failed >= twin_runs / 2, "the twin 'one more accepted row' passed too many runs");
  std::printf("  %s/%s: 7. twin 'the adapter keeps one more row than the verifier decided': %zu of %zu runs fail (every always-wrong run)\n",
              be.name.c_str(), f.name.c_str(), twin_failed, twin_runs);
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
  if (ds.backend().desyncs() != 0 || ds.ingest().failures() != 0) {
    return "the drafter's bookkeeping reported " + std::to_string(ds.backend().desyncs()) + " desyncs and " +
           std::to_string(ds.ingest().failures()) + " failures";
  }
  return "";
}

// ---- 2, 3 the real tiny drafter ------------------------------------------------------------

void RealTests(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  const auto base_setup = Setup(f, dir, 4);
  const auto model = ArmedModel(f, be, *base_setup);
  const std::size_t limit = 22;
  std::size_t runs = 0, compared = 0, drafted = 0, accepted = 0, proposals = 0, checked = 0;
  const std::size_t lengths[] = {5, 13, 23, 29};
  std::vector<Truth> truths;
  for (const std::size_t len : lengths) truths.push_back(FindTruth(model, len, 300 + len * 3, limit + 18, false));
  for (std::uint32_t d = 1; d <= kMaxVerifyRows - 1; ++d) {
    const Truth& truth = truths[d % 4];
    const auto setup = WithDraftLength(*base_setup, d);
    auto session = NewSession(model);
    std::string error;
    DraftedSession ds(*session, *setup);  // attached before the first row, as the runner binds it
    Require(session->Sync(truth.prompt, &error), error);
    ProposalCheck check(*setup, *model, truth, 0);
    const Drafted run = RunDrafted(ds.decoder(), truth, limit, [&](const DecodeResult&) {
      std::string e = ContextLedger(ds, 0);
      return e.empty() ? check.Check(ds) : e;
    });
    Require(run.error.empty(), be.name + "/" + f.name + " real drafter d=" + std::to_string(d) + ": " + run.error);
    ++runs;
    compared += run.compared;
    drafted += run.drafted;
    accepted += run.accepted;
    proposals += ds.backend().proposals();
    checked += check.compared;
    Require(ds.backend().proposals() > 0 && check.compared == ds.backend().proposals(), "every proposal must have been compared with the reference");
  }
  std::printf("  %s/%s: 2. real tiny drafter, d 1..15: %zu runs, %zu logits rows compared, %zu tokens drafted, %zu accepted; exactness and accounting hold, "
              "the drafter's context ends at the session's position after every step\n",
              be.name.c_str(), f.name.c_str(), runs, compared, drafted, accepted);
  std::printf("  %s/%s: 3. all %zu proposals (%zu compared) equal DrafterReference driven with plain decoding's taps%s\n", be.name.c_str(), f.name.c_str(),
              proposals, checked, OpsDrafter() ? (" (drafter executor; " + std::to_string(g_proposal_ties) + " proposals so far differ from the reference's at a near-tie only)").c_str() : "");

  // 3, twins: the output stays exact (the verifier checks every token) and the proposals go wrong.
  struct Twin {
    const char* name;
    std::function<void(SpeculativeSetup&)> apply;
  };
  std::vector<Twin> twins = {
      {"anchor position off by one", [](SpeculativeSetup& s) { s.faults.anchor_position_off_by_one = true; }},
      {"anchor token off by one", [](SpeculativeSetup& s) { s.faults.anchor_token_off_by_one = true; }},
  };
  // The scalar reference's own fault; the executor's twins are in tests/drafter_ops_test.cpp.
  if (!OpsDrafter()) twins.push_back({"context one row short", [](SpeculativeSetup& s) { s.engine_faults.context_one_row_short = true; }});
  for (const Twin& twin : twins) {
    std::size_t twin_proposals = 0, twin_wrong = 0, exact_failures = 0;
    for (const std::uint32_t d : {1u, 3u, 7u}) {
      auto setup = std::make_shared<SpeculativeSetup>(*base_setup);
      setup->max_draft_tokens = d;
      twin.apply(*setup);
      const Truth& truth = truths[1 + d % 3];
      auto session = NewSession(model);
      std::string error;
      DraftedSession ds(*session, *setup);
      Require(session->Sync(truth.prompt, &error), error);
      ProposalCheck check(*setup, *model, truth, 0);
      std::string first_wrong;
      const Drafted run = RunDrafted(ds.decoder(), truth, limit, [&](const DecodeResult&) {
        const std::string e = check.Check(ds);
        if (!e.empty() && first_wrong.empty()) first_wrong = e;
        return std::string();  // keep going: the output must stay exact
      });
      if (!run.error.empty()) ++exact_failures;
      twin_proposals += ds.backend().proposals();
      twin_wrong += check.mismatched + check.range_wrong;
    }
    Require(exact_failures == 0, be.name + "/" + f.name + ": the twin '" + twin.name + "' changed the output (the verifier must absorb a bad draft)");
    Require(twin_wrong > 0, be.name + "/" + f.name + ": the twin '" + twin.name + "' passed the proposal check");
    Require(twin_wrong * 2 >= twin_proposals, be.name + "/" + f.name + ": the twin '" + twin.name + "' left most proposals equal to the reference's");
    std::printf("  %s/%s: 3. twin '%s': output still exact, %zu of %zu proposals differ from the reference\n", be.name.c_str(), f.name.c_str(), twin.name,
                twin_wrong, twin_proposals);
  }
}

// ---- 4 prefix reuse, restore, the recompute band --------------------------------------------

std::vector<std::int32_t> Concat(std::vector<std::int32_t> a, const std::vector<std::int32_t>& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

void ReuseAndRestore(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  const auto base_setup = Setup(f, dir, 6);
  const std::size_t limit = 22;
  const auto model = ArmedModel(f, be, *base_setup, 32);  // a ring with slack: a checkpoint right after a rewind is capturable
  std::string error;

  // 4a. prefix reuse: drafted generation, then a text sharing the first half, drafted again. The truth is plain decoding given the SAME
  // call sequence (Sync, one-row steps, Sync of the new text, one-row steps), so this holds on the production route as well.
  {
    Rig rig(model);
    DraftedSession ds(rig.session(), *base_setup);
    const Truth a = FindTruth(model, 30, 700, limit + 18, false);
    Require(rig.session().Sync(a.prompt, &error), error);
    ProposalCheck check_a(*base_setup, *model, a, 0);
    Drafted run = RunDrafted(ds.decoder(), a, limit, [&](const DecodeResult&) {
      std::string e = ContextLedger(ds, 0);
      return e.empty() ? check_a.Check(ds) : e;
    });
    Require(run.error.empty(), be.name + "/" + f.name + " 4a first request: " + run.error);
    // The next text shares the first 24 tokens (and diverges at once), as a conversation that was edited.
    const auto a_sequence = a.Sequence();
    auto b_prompt = std::vector<std::int32_t>(a_sequence.begin(), a_sequence.begin() + 24);
    for (std::size_t i = 0; i < 9; ++i) b_prompt.push_back((a_sequence[24 + i] + 5) % 256);
    const Truth b = RunTruthAfter(model, a, limit, b_prompt, limit);
    Require(rig.session().Sync(b_prompt, &error), error);
    ds.decoder().Restart();
    Require(ContextLedger(ds, 0).empty(), be.name + "/" + f.name + " 4a: the drafter's context after the prefix-reuse Sync: " + ContextLedger(ds, 0));
    ProposalCheck check_b(*base_setup, *model, b, 0);
    run = RunDrafted(ds.decoder(), b, limit, [&](const DecodeResult&) {
      std::string e = ContextLedger(ds, 0);
      return e.empty() ? check_b.Check(ds) : e;
    });
    Require(run.error.empty(), be.name + "/" + f.name + " 4a second request: " + run.error);
    Require(check_b.compared > 0, "4a: nothing was proposed after the prefix reuse");
  }
  {
    std::printf("  %s/%s: 4a. prefix reuse after drafted steps: output equals plain, the drafter's context is the session's, proposals equal the reference\n",
                be.name.c_str(), f.name.c_str());
  }

  // 4b. a restored checkpoint into a slot whose drafter holds an old branch.
  {
    Rig source(model);
    const Truth c = FindTruth(model, 40, 1700, limit + 18, false);
    Require(source.session().Sync(c.prompt, &error), error);
    const Checkpoint ck = Capture(source);
    Rig rig(model);
    DraftedSession ds(rig.session(), *base_setup);
    const Truth a = FindTruth(model, 30, 700, limit + 18, false);
    Require(rig.session().Sync(a.prompt, &error), error);
    Drafted run = RunDrafted(ds.decoder(), a, 12);
    Require(run.error.empty(), "4b first branch: " + run.error);
    Require(ds.engine().End() == rig.session().Position() && ds.engine().End() > 40, "4b: the old branch must reach past the restore position");
    Require(Restore(rig, ck), "4b: restore refused");
    ds.decoder().Restart();
    Require(rig.session().Position() == 40, "4b: restored position");
    Require(ds.engine().Base() == 40 && ds.engine().End() == 40, "4b: a restore must leave the drafter with no rows below the restored position; it holds [" +
                                                                    std::to_string(ds.engine().Base()) + ", " + std::to_string(ds.engine().End()) + ")");
    ProposalCheck check(*base_setup, *model, c, 40);
    std::size_t first_empty = 0;
    run = RunDrafted(ds.decoder(), c, limit, [&](const DecodeResult&) {
      std::string e = ContextLedger(ds, 40);
      if (!e.empty()) return e;
      e = check.Check(ds);
      if (e.empty() && check.compared == 1 && first_empty == 0) {
        first_empty = 1;
        const auto& blk = ds.backend().last_block();
        if (blk.context_first != blk.context_end) e = "the first proposal after a restore read context rows";
      }
      return e;
    });
    Require(run.error.empty(), be.name + "/" + f.name + " 4b after the restore: " + run.error);
    Require(check.compared > 0, "4b: nothing was proposed after the restore");
  }
  std::printf("  %s/%s: 4b. restored checkpoint into a used slot: the drafter's context restarts at the restored position, output equals plain, "
              "proposals equal the reference fed from that position, no row below it is read\n",
              be.name.c_str(), f.name.c_str());
}

/// 4c: the recompute band (docs/DRAFTER-S4.md). After rejections the executor's high-water mark stays
/// above the logical end, so a Sync whose common prefix lies in a band of up to 15 positions cannot
/// rewind (plain decoding could) and recomputes from 0 through the existing path.
void RecomputeBand(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  if (!be.exact) {  // a recompute in prefill passes and a rewind keep rows computed by different routes
    std::printf("  %s/%s: 4c. recompute band: skipped (needs an exact back end: rewound and recomputed rows differ in bits otherwise)\n",
                be.name.c_str(), f.name.c_str());
    return;
  }
  const auto setup = WithDraftLength(*Setup(f, dir, 15), 15);
  const auto model = ArmedModel(f, be, *setup);
  const std::uint32_t R = model->RingRows(), W = f.window;
  std::string error;
  const auto prompt = SyntheticTokens(2100, 40);
  auto drafted = NewSession(model);
  DraftedSession ds(*drafted, *setup);  // attached before the first row, as the runner binds it
  Require(drafted->Sync(prompt, &error), error);
  // Draft until the ring has wrapped and the drafted session's high-water mark is above the plain one's (a
  // verification whose rows were rejected evicted older ring rows than the accepted ones would have).
  sampling::SamplerState sampler;
  for (std::size_t steps = 0;; ++steps) {
    Require(steps < 200, "4c: no drafted step left a high-water mark above the end");
    (void)ds.decoder().Step(30, sampler, true);
    const std::uint32_t at = drafted->Position();
    if (at > R + W + 2 && drafted->executor().RetainedBegin() > at - R) break;
  }
  Drafted run;
  const std::uint32_t end = drafted->Position();
  Require(end > R + W, "4c: the generation must go past the ring");
  // Plain session at the same tokens.
  const auto tokens = drafted->tokens();
  auto plain = NewSession(model);
  Require(plain->Sync(Slice(tokens, 0, 40), &error), error);
  for (std::size_t i = 40; i < tokens.size(); ++i) Require(plain->Evaluate(tokens[i], &error), error);
  const std::uint32_t plain_begin = plain->executor().RetainedBegin(), drafted_begin = drafted->executor().RetainedBegin();
  Require(plain_begin == end - R, "4c: plain decoding's retained rows begin at the end minus the ring");
  Require(drafted_begin > plain_begin, "4c: the drafted session must have a higher retained-rows boundary (rejected rows evicted older ones)");
  const std::uint32_t common = plain_begin + W - 1;  // the lowest position plain decoding can rewind to
  Require(common < tokens.size() && drafted_begin > common - W + 1, "4c: the common prefix is in the band");
  auto text = Slice(tokens, 0, common);
  for (std::size_t i = 0; i < 9; ++i) text.push_back((tokens[common + i] + 7) % 256);  // diverges at `common`
  Require(text[common] != tokens[common], "4c: the new text must diverge at the common prefix");

  std::vector<WritePlan> plain_plans, plans;
  plain->SetWriteGuard([&](const WritePlan& p) { plain_plans.push_back(p); });
  drafted->SetWriteGuard([&](const WritePlan& p) { plans.push_back(p); });
  Require(plain->Sync(text, &error), "4c plain sync: " + error);
  Require(drafted->Sync(text, &error), "4c drafted sync: " + error);
  const WritePlan whole{{0, model->MaxContext()}, {0, 0}};
  Require(!plain_plans.empty() && !(plain_plans.front().written == whole.written), "4c: plain decoding should have rewound");
  Require(!plans.empty() && plans.front().written == whole.written, "4c: the drafted session must take the recompute path (guard told about the whole context first)");
  Require(SameBits(drafted->Logits(), plain->Logits()) && drafted->tokens() == plain->tokens(), "4c: after the Sync the drafted session differs from the plain one");
  drafted->SetWriteGuard({});
  ds.decoder().Restart();
  // The drafter's context was rebuilt from position 0 by the recompute.
  Require(ContextLedger(ds, 0).empty(), be.name + "/" + f.name + " 4c: " + ContextLedger(ds, 0));
  const Truth b = RunTruth(model, text, 20);
  ProposalCheck check(*setup, *model, b, 0);
  run = RunDrafted(ds.decoder(), b, 20, [&](const DecodeResult&) {
    std::string e = ContextLedger(ds, 0);
    return e.empty() ? check.Check(ds) : e;
  });
  Require(run.error.empty(), be.name + "/" + f.name + " 4c after the recompute: " + run.error);
  Require(check.compared > 0, "4c: nothing was proposed after the recompute");
  std::printf("  %s/%s: 4c. recompute band: common prefix %u of %u (ring %u): plain decoding rewinds, the drafted session (retained from %u instead of %u) recomputes "
              "from 0; output equals plain, the drafter's context was rebuilt from 0, proposals equal the reference\n",
              be.name.c_str(), f.name.c_str(), common, end, R, drafted_begin, plain_begin);
}

// ---- 6 tap memory ----------------------------------------------------------------------------

class CountingSink final : public TapSink {
 public:
  void Rows(std::uint32_t first, std::uint32_t rows, std::span<const float> data) noexcept override {
    max_rows = std::max<std::size_t>(max_rows, rows);
    contiguous = contiguous && first == total;
    total += rows;
    ++calls;
    sizes_ok = sizes_ok && data.size() == std::size_t{rows} * width;
  }
  void Rewound(std::uint32_t) noexcept override {}
  void Replaced(std::uint32_t) noexcept override {}
  std::size_t width{0}, total{0}, calls{0}, max_rows{0};
  bool contiguous{true}, sizes_ok{true};
};

void TapMemory(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  const auto setup = Setup(f, dir, 4);
  const auto model = ArmedModel(f, be, *setup);
  auto s = NewSession(model);
  CountingSink sink;
  sink.width = model->executor_config().capture_layers.size() * model->weights().config.embedding_length;
  s->SetTapSink(&sink);
  const auto prompt = SyntheticTokens(4242, 97);
  std::string error;
  Require(s->Sync(prompt, &error), error);
  const std::size_t passes = (prompt.size() + kPrefillRows - 1) / kPrefillRows;
  Require(sink.calls == passes && sink.total == prompt.size() && sink.max_rows <= kPrefillRows && sink.contiguous && sink.sizes_ok,
          "a multi-pass Sync must hand over its rows one pass at a time: " + std::to_string(sink.calls) + " calls, at most " + std::to_string(sink.max_rows) + " rows");
  Require(s->TakeTaps().rows == 0, "with a sink attached nothing may stay pending in the session");
  // A verification hands over only the rows it keeps, when it settles.
  Require(s->Verify(SyntheticTokens(1, 8), &error), error);
  Require(sink.total == prompt.size(), "the rows of an open verification must not reach the sink yet");
  Require(s->AcceptVerified(3, &error), error);
  Require(sink.total == prompt.size() + 3 && sink.contiguous, "an accepted verification hands over exactly the kept rows");
  s->SetTapSink(nullptr);
  std::printf("  %s/%s: 6. tap memory: a %zu-token Sync with a sink hands over %zu passes of at most %zu rows (the session keeps none pending); "
              "a verification delivers only its kept rows, when it settles\n",
              be.name.c_str(), f.name.c_str(), prompt.size(), sink.calls, sink.max_rows);
}

// ---- 5 the runner ----------------------------------------------------------------------------

using Tokens = std::vector<ext::TextRunnerToken>;

struct Side {
  std::shared_ptr<ext::TextModelRunner> runner;
  std::unique_ptr<Rig> rig;
  std::unique_ptr<ext::TextRunnerState> state;
};

Side MakeSide(const std::shared_ptr<EngineModel>& model, std::shared_ptr<const SpeculativeSetup> setup) {
  Side side;
  side.runner = CreateRunnerForModel(model, "", std::move(setup));
  side.rig = std::make_unique<Rig>(model);
  side.state = side.runner->BindComponentState(*side.rig->adapter, *side.rig->slot);
  side.state->SetStopAtEos(true);
  return side;
}

struct Generated {
  Tokens tokens;
  std::size_t drafted{0}, accepted{0};
  bool stop{false};
};

/// One request the way the serving pool drives a runner: (prefix reuse), prefill, preview, decode steps.
Generated Drive(Side& side, const Tokens& prompt, std::size_t cached, std::size_t limit, const sampling::SamplingConfig& config) {
  auto& runner = *side.runner;
  if (cached > 0) runner.PreparePrefixReuse(*side.state, std::span<const ext::TextRunnerToken>(prompt).first(cached));
  for (std::size_t offset = cached; offset < prompt.size();) {
    const auto step = runner.Prefill(*side.state, prompt, offset, std::size_t{1} << 20);
    Require(step.consumed_tokens > 0, "prefill made no progress");
    offset += step.consumed_tokens;
  }
  const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplerState sampler(config, history);
  Generated g;
  auto preview_sampler = sampler;
  const auto preview = runner.PreviewFirstToken(*side.state, preview_sampler);
  while (g.tokens.size() < limit && !g.stop) {
    const auto step = runner.DecodeStep(*side.state, limit - g.tokens.size(), sampler);
    Require(!step.selections.empty() || step.stop, "a decode step without a token or a stop");
    Require(step.selections.size() <= limit - g.tokens.size(), "a decode step beyond its budget");
    if (g.tokens.empty() && preview && !preview->stop && !step.selections.empty()) {
      Require(step.selections.front().token == preview->token, "the first-token preview disagrees with decoding");
    }
    for (const auto& sel : step.selections) {
      Require(!sel.stop, "a stop selection inside a step");
      sampler.Accept(sel.token);
      g.tokens.push_back(sel.token);
    }
    g.drafted += step.draft_tokens;
    g.accepted += step.draft_accepted_tokens;
    g.stop = step.stop;
  }
  return g;
}

void RunnerTests(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  const auto setup = Setup(f, dir, 4);
  const auto spec_model = ArmedModel(f, be, *setup);
  const auto plain_model = PlainModel(f, be);
  const std::size_t limit = 24;
  Side spec = MakeSide(spec_model, setup);
  Side plain = MakeSide(plain_model, nullptr);
  const auto plain_caps = plain.runner->Descriptor().capabilities, spec_caps = spec.runner->Descriptor().capabilities;
  Require(!plain_caps.multi_token_decode && plain_caps.final_token_advance_required && spec_caps.multi_token_decode && !spec_caps.final_token_advance_required,
          "the runner's capabilities must follow the speculation setting");
  Require(DraftingOf(*spec.state) != nullptr && DraftingOf(*plain.state) == nullptr, "drafting is bound to the speculative state only");

  const auto prompt_ints = SyntheticTokens(55 + f.window, 27);
  const Tokens prompt(prompt_ints.begin(), prompt_ints.end());
  struct Config {
    const char* name;
    sampling::SamplingConfig config;
    bool drafts;
  };
  std::vector<Config> configs;
  configs.push_back({"greedy", {}, true});
  {
    sampling::SamplingConfig c;
    c.temperature = 0.9F;
    c.top_k = 40;
    c.seed = 11;
    configs.push_back({"sampled (T 0.9, top-k 40)", c, false});
    sampling::SamplingConfig p;
    p.repeat_penalty = 1.25F;
    p.repeat_last_n = 16;
    configs.push_back({"greedy with a repeat penalty", p, false});
    sampling::SamplingConfig lp;
    lp.logprobs = 2;
    configs.push_back({"greedy with log-probabilities", lp, false});
  }
  for (const Config& c : configs) {
    // Fresh states for every configuration: nothing is carried over but the model.
    Side s = MakeSide(spec_model, setup);
    Side p = MakeSide(plain_model, nullptr);
    const Generated want = Drive(p, prompt, 0, limit, c.config);
    const Generated got = Drive(s, prompt, 0, limit, c.config);
    Require(got.tokens == want.tokens && got.stop == want.stop,
            be.name + "/" + f.name + " runner, " + c.name + ": the tokens differ from the runner without speculation");
    if (c.drafts) {
      Require(got.drafted > 0, "the greedy request must have drafted");
    } else {
      Require(got.drafted == 0 && got.accepted == 0, std::string(c.name) + " must draft nothing");
    }
    // SUPERSEDED (docs/DRAFTER-S4.md, "Follow-ups", item B): a drafted request that ends by its limit used to end with its last token pending
    // (checkpoint prompt + emitted - 1). It now ends where plain generation does, every token evaluated in a one-row pass: this holds in
    // production mode too, because the second request below follows the same call sequence on both sides.
    if (c.drafts && !got.stop) {
      Require(s.runner->CheckpointPosition(*s.state) == prompt.size() + got.tokens.size(), "a drafted request ends with every token advanced, as plain");
      Require(p.runner->CheckpointPosition(*p.state) == prompt.size() + want.tokens.size(), "a plain request ends with every token advanced");
      // A second request on the same state: the conversation continues (prefix reuse, the pending token in the prompt).
      Tokens next = prompt;
      next.insert(next.end(), got.tokens.begin(), got.tokens.end());
      for (std::size_t i = 0; i < 7; ++i) next.push_back((i * 13 + 5) % 256);
      const Generated want2 = Drive(p, next, p.runner->CheckpointPosition(*p.state), limit, c.config);
      const Generated got2 = Drive(s, next, s.runner->CheckpointPosition(*s.state), limit, c.config);
      Require(got2.tokens == want2.tokens && got2.stop == want2.stop, be.name + "/" + f.name + " runner: a second request with prefix reuse differs from the plain runner");
      Require(got2.drafted > 0, "the second request must have drafted");
    }
  }
  std::printf("  %s/%s: 5. runner: greedy requests (with a second request reusing the prefix and the pending token) give the tokens of the runner without "
              "speculation and draft; sampled, repeat-penalty and log-probability requests draft nothing and give the same tokens\n",
              be.name.c_str(), f.name.c_str());
}

// ---- the prompt driver ------------------------------------------------------------------------

/// `gufo prompt --speculative dflash2 --dflash-model F` (RunRicursivePrompt, on the CPU executor): the text it prints
/// equals the text printed without speculation.
void DriverTest(const TinyFixture& f, const std::string& dir) {
  const auto run = [&](bool speculative) {
    ext::TextPromptOptions opt;
    opt.model_path = f.path;
    opt.prompt_text = "tiny prompt for a drafted greedy run";
    opt.use_chat_template = false;
    opt.max_tokens = 30;
    opt.force_cpu = true;
    opt.verbose = true;
    opt.provider_options["--allow-chat-template-mismatch"] = true;  // the tiny files carry no chat template
    if (speculative) {
      opt.speculative_backend = "dflash2";
      opt.dflash_model_path = dir + "/drafter-" + f.name + "-q8_0.gguf";
      opt.draft_tokens = 4;
    }
    std::string error;
    std::shared_ptr<const core::GgufReader> reader = core::GgufReader::OpenFile(f.path, &error);
    Require(reader != nullptr, error);
    std::ostringstream captured;
    auto* old = std::cout.rdbuf(captured.rdbuf());
    int code = 1;
    try {
      code = RunRicursivePrompt(opt, *reader, std::chrono::steady_clock::now());
    } catch (...) {
      std::cout.rdbuf(old);
      throw;
    }
    std::cout.rdbuf(old);
    Require(code == 0, "RunRicursivePrompt failed (" + std::to_string(code) + ")");
    return captured.str();
  };
  // Only the generated text counts: the timing lines differ between runs.
  const auto text = [](const std::string& out) {
    const auto begin = out.find("--- Generation Output ---\n");
    const auto end = out.find("Prefill ");
    Require(begin != std::string::npos && end != std::string::npos && end > begin, "driver output has no generation section");
    return out.substr(begin, end - begin);
  };
  const std::string plain = run(false), drafted = run(true);
  Require(text(plain) == text(drafted) && text(plain).size() > 40, f.name + ": the driver's drafted text differs from the plain text");
  Require(drafted.find("Drafted ") != std::string::npos, "the drafted driver run must report its drafts");
  std::printf("  cpu/%s: driver: `prompt --speculative dflash2` prints the plain run's text (%zu characters) and reports drafted/accepted tokens\n", f.name.c_str(),
              text(plain).size());
}

}  // namespace

int main(int argc, char** argv) {
  const std::string filter = argc > 1 ? argv[1] : "all";
  const std::string dir = argc > 2 ? argv[2] : "/server/staging/ripple-gufo/perf/stage4";
  // "all", one "<group>:<fixture>", or several of them separated by commas.
  const auto wants = [&](const char* group, const TinyFixture& f) {
    const std::string want = std::string(group) + ":" + f.name;
    for (std::size_t at = 0; at <= filter.size();) {
      const std::size_t comma = std::min(filter.find(',', at), filter.size());
      if (filter == "all" || filter.substr(at, comma - at) == want) return true;
      at = comma + 1;
    }
    return false;
  };
  return Run([&] {
    std::size_t groups = 0;
    for (const auto& be : DraftBackends()) {
      for (const auto& f : TinyFixtures()) {
        if (wants("mock", f)) {
          MockTests(f, be);
          ++groups;
        }
        if (wants("real", f)) {
          RealTests(f, be, dir);
          ReuseAndRestore(f, be, dir);
          RecomputeBand(f, be, dir);
          TapMemory(f, be, dir);
          ++groups;
        }
        if (wants("runner", f)) {
          RunnerTests(f, be, dir);
          ++groups;
        }
        if (wants("driver", f) && !be.gpu && !be.hostops) {  // the driver builds its own CPU-executor model
          DriverTest(f, dir);
          ++groups;
        }
      }
    }
    Require(groups > 0, "no group matches '" + filter + "'");
  });
}

#endif  // ENGINE_ENABLE_HIP

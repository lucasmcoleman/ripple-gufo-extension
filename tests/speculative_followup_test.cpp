// Follow-ups to drafted decoding (docs/DRAFTER-S4.md, "Follow-ups", job 1692), on the CPU executor and the
// host-ops executor (or, through tests/run-gpu-gates.sh, the GPU executor), tinyA and tinyB.
//
//   reuse  A. the drafter's context survives from one request to the next on a live slot: every request binds
//             a NEW runner state over the same session, as gufo's component cache does on every acquire
//   end    B. a request that ends by its token limit (or by a stop token) leaves the session
//             and its checkpoint where plain generation leaves them, given the same call sequence
//   endcut B. a stop sequence or a cancellation leaves a checkpoint that extends plain's (exact back ends only: on the
//             production route the extra rows change the next request's pass boundaries)
//   fail   C. a drafter that throws costs speed, never the request
//   flow   D. flows through the runner as gufo's pool issues them (interleaving, cancel, cut by limit, restore)
//
// Requests are driven the way gufo's scheduler drives the runner pool (the budget arithmetic of
// StepMultiTokenDecode, the preview, Commit and Cancel with their checkpoint bound): see Serve::Run.
// Argument: "all", "<group>:<fixture>" or a comma separated list, then the directory of the tiny drafters.
// RICURSIVE_TEST_DRAFTER=ops: the proposals come from the drafter executor (docs/DRAFTER-S2.md).
#include "tests/spec_util.hpp"

#if !defined(ENGINE_ENABLE_HIP)
int main() {
  std::puts("SKIP: this build has no speculative verifier (ENGINE_ENABLE_HIP)");
  return 77;
}
#else

#include <iostream>
#include <set>
#include <sstream>

using namespace ricursive_test;

namespace {

using Tokens = std::vector<ext::TextRunnerToken>;

Tokens ToTokens(const std::vector<std::int32_t>& v) { return Tokens(v.begin(), v.end()); }

// ---- a cache slot served the way gufo serves it ----------------------------------------------------

struct Serve;
struct Hooks {
  std::function<void(Serve&)> after_acquire;  ///< a restore, before the state learns the position
  std::function<void(Serve&)> after_prefill;
  std::function<void(Serve&)> after_step;
  bool full_prefix{false};  ///< the slot may hold the whole prompt (a restored checkpoint at the prompt's end): no prefill pass, decoding starts from its frontier logits
};

enum class Ending { kLimit, kStopSequence, kCancel };

struct Result {
  Tokens published;  ///< what the client sees (the request's tokens)
  Tokens generated;  ///< what the pool recorded: every selection of every step
  bool stop{false};
  std::size_t cached{0}, checkpoint{0}, drafted{0}, accepted{0}, steps{0}, prefill_rows{0};
  Tokens checkpoint_tokens;  ///< prompt + generated, cut at the runner's checkpoint position
};

struct Serve {
  std::shared_ptr<ext::TextModelRunner> runner;
  std::unique_ptr<Rig> rig;
  std::unique_ptr<ext::TextRunnerState> state;
  bool multi{false};

  Serve(std::shared_ptr<ext::TextModelRunner> r, const std::shared_ptr<EngineModel>& model)
      : runner(std::move(r)), rig(std::make_unique<Rig>(model)) {
    multi = runner->Descriptor().capabilities.multi_token_decode;
  }
  Session& session() { return rig->session(); }
  std::vector<std::pair<std::size_t, std::size_t>> prefill;  ///< the prefill passes of the request in progress (readable from its hooks)

  /// Component cache Acquire: a NEW runner state over the slot's session, the old one gone first.
  void Acquire() {
    state.reset();
    state = runner->BindComponentState(*rig->adapter, *rig->slot);
    state->SetStopAtEos(true);
  }

  /// One request: acquire (the live slot is kept when its tokens are a prefix of the prompt, else cold), prefill,
  /// decode as gufo's scheduler does, then Commit (limit, stop token) or Cancel (stop sequence, cancellation).
  /// `ending_n`: the stop sequence completes at the n-th published token / the request is cancelled before step n.
  Result Run(const Tokens& prompt, std::size_t limit, Ending ending = Ending::kLimit, std::size_t ending_n = 0, const Hooks& hooks = {},
             const sampling::SamplingConfig& config = {}) {
    Result r;
    Acquire();
    if (hooks.after_acquire) hooks.after_acquire(*this);
    const auto& live = session().tokens();
    const bool prefix = (live.size() < prompt.size() || (hooks.full_prefix && live.size() == prompt.size())) && std::equal(live.begin(), live.end(), prompt.begin());
    if (!prefix && session().Position() != 0) Require(rig->adapter->Invalidate(*rig->slot), "cold start: invalidate");
    r.cached = prefix ? live.size() : 0;
    prefill.clear();
    runner->ReconcileComponentState(*state, r.cached);
    if (r.cached > 0) runner->PreparePrefixReuse(*state, std::span<const ext::TextRunnerToken>(prompt).first(r.cached));
    for (std::size_t offset = r.cached; offset < prompt.size();) {
      const auto step = runner->Prefill(*state, prompt, offset, std::size_t{1} << 20);
      Require(step.consumed_tokens > 0, "prefill made no progress");
      r.prefill_rows += step.consumed_tokens;
      prefill.emplace_back(offset, step.consumed_tokens);
      offset += step.consumed_tokens;
    }
    if (hooks.after_prefill) hooks.after_prefill(*this);
    const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
    sampling::SamplerState sampler(config, history);
    bool stop_sequence = false, cancelled = false;
    const auto publish = [&](ext::TextRunnerToken t) {
      r.published.push_back(t);
      if (ending == Ending::kStopSequence && r.published.size() >= ending_n) stop_sequence = true;
    };
    if (multi) {
      // StepMultiTokenDecode: PrepareFirstSnapshot (the preview), then DecodeStep(limit - published + preview).
      std::optional<ext::TextRunnerToken> preview_token;
      bool preview_stops = false;
      {
        auto preview_sampler = sampler;
        const auto preview = runner->PreviewFirstToken(*state, preview_sampler);
        if (preview && !preview->stop) {
          publish(preview->token);
          preview_token = preview->token;
        }
        preview_stops = preview && preview->stop;
      }
      if (preview_stops) {
        r.stop = true;
      } else if (stop_sequence) {
        cancelled = true;  // the stop sequence completed on the preview: CompleteSuccess(kStopSequence) = Cancel
      } else {
        while (true) {
          if (ending == Ending::kCancel && r.steps == ending_n) {
            cancelled = true;
            break;
          }
          const std::size_t remaining = limit - r.published.size() + (preview_token ? 1 : 0);
          const auto step = runner->DecodeStep(*state, remaining, sampler);
          ++r.steps;
          Require(step.selections.size() <= remaining && (!step.selections.empty() || step.stop), "an invalid decode step");
          if (preview_token && (step.selections.empty() || step.selections.front().token != *preview_token)) {
            throw std::runtime_error("first-token preview disagrees with decoding");
          }
          for (const auto& sel : step.selections) {
            Require(!sel.stop, "a stop selection inside a step");
            sampler.Accept(sel.token);
            r.generated.push_back(sel.token);
          }
          r.drafted += step.draft_tokens;
          r.accepted += step.draft_accepted_tokens;
          for (const auto& sel : step.selections) {
            if (preview_token) {
              preview_token.reset();
            } else {
              publish(sel.token);
            }
            if (stop_sequence) break;
          }
          if (hooks.after_step) hooks.after_step(*this);
          if (stop_sequence) {
            cancelled = true;
            break;
          }
          if (step.stop) {
            r.stop = true;
            break;
          }
          if (r.published.size() >= limit) break;
        }
      }
      // (the first preview token is in generated through the first step)
    } else {
      // PrepareDecode: SelectNext, publish, stop sequence -> Cancel (the token is not advanced), else Advance; the limit after it.
      while (true) {
        if (ending == Ending::kCancel && r.steps == ending_n) {
          cancelled = true;
          break;
        }
        const auto selection = runner->SelectNext(*state, sampler);
        if (selection.stop) {
          r.stop = true;
          break;
        }
        publish(selection.token);
        ++r.steps;
        sampler.Accept(selection.token);
        r.generated.push_back(selection.token);
        if (stop_sequence) {
          cancelled = true;
          break;
        }
        runner->Advance(*state, selection.token);
        if (hooks.after_step) hooks.after_step(*this);
        if (r.published.size() >= limit) break;
      }
    }
    (void)cancelled;  // Commit and Cancel bound the checkpoint the same way (CheckpointPosition); neither evaluates anything
    r.checkpoint = runner->CheckpointPosition(*state);
    r.checkpoint_tokens = prompt;
    r.checkpoint_tokens.insert(r.checkpoint_tokens.end(), r.generated.begin(), r.generated.end());
    Require(r.checkpoint >= r.cached && r.checkpoint <= r.checkpoint_tokens.size(),
            "the checkpoint is outside executed token history: position " + std::to_string(r.checkpoint) + ", cached " + std::to_string(r.cached) +
                ", prompt+generated " + std::to_string(r.checkpoint_tokens.size()));
    r.checkpoint_tokens.resize(r.checkpoint);
    return r;
  }
};

/// The session's tokens equal the checkpoint the pool would publish (what a live reuse trusts).
void RequireLive(Serve& s, const Result& r, const std::string& what) {
  Require(s.session().Position() == r.checkpoint && std::equal(s.session().tokens().begin(), s.session().tokens().end(), r.checkpoint_tokens.begin(),
                                                           r.checkpoint_tokens.end()),
          what + ": the session does not hold the checkpoint's tokens");
}

/// The passes wider than the 16 rows the production route computes row by row (a wider pass rounds activations across its rows).
std::vector<std::pair<std::size_t, std::size_t>> WidePasses(const std::vector<std::pair<std::size_t, std::size_t>>& passes) {
  std::vector<std::pair<std::size_t, std::size_t>> wide;
  for (const auto& p : passes) {
    if (p.second > 16) wide.push_back(p);
  }
  return wide;
}

Tokens Extend(Tokens t, std::size_t extra, std::size_t salt) {
  for (std::size_t i = 0; i < extra; ++i) t.push_back(static_cast<ext::TextRunnerToken>((i * 13 + salt) % 256));
  return t;
}

struct Models {
  std::shared_ptr<const SpeculativeSetup> setup;
  std::shared_ptr<EngineModel> spec, plain;
  std::shared_ptr<ext::TextModelRunner> spec_runner, plain_runner;
};

Models MakeModels(const TinyFixture& f, const DraftBackend& be, const std::string& dir, std::uint32_t d = 4,
                  std::shared_ptr<const SpeculativeSetup> given = nullptr) {
  Models m;
  m.setup = given ? given : Setup(f, dir, d);
  m.spec = ArmedModel(f, be, *m.setup);
  m.plain = PlainModel(f, be);
  m.spec_runner = CreateRunnerForModel(m.spec, "", m.setup);
  m.plain_runner = CreateRunnerForModel(m.plain, "", nullptr);
  return m;
}

// ---- A: the drafter's context survives from one request to the next ------------------------------------------

std::string Ledger(DraftedSession& ds, std::uint32_t base) {
  auto& e = ds.engine();
  if (ds.health().failures() != 0) return "the drafter reported failures";
  if (e.End() != ds.decoder().session().Position() || e.Base() != base) {
    return "the drafter holds rows [" + std::to_string(e.Base()) + ", " + std::to_string(e.End()) + ") but the session is at " +
           std::to_string(ds.decoder().session().Position()) + " (expected from " + std::to_string(base) + ")";
  }
  return "";
}

void ReuseTests(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  const Models m = MakeModels(f, be, dir);
  const std::size_t limit = 24;
  Serve a(m.spec_runner, m.spec), b(m.spec_runner, m.spec);  // two slots of one runner
  Serve pa(m.plain_runner, m.plain), pb(m.plain_runner, m.plain);
  const Tokens prompt1 = ToTokens(SyntheticTokens(55 + f.window, 27)), prompt_b = ToTokens(SyntheticTokens(901 + f.window, 21));

  // First turn of two conversations, interleaved; each slot has its own drafter.
  const Result a1 = a.Run(prompt1, limit), pa1 = pa.Run(prompt1, limit);
  const Result b1 = b.Run(prompt_b, limit), pb1 = pb.Run(prompt_b, limit);
  Require(a1.published == pa1.published && b1.published == pb1.published, "first turns differ from plain");
  Require(a1.drafted > 0, "the first turn must have drafted");

  // Second turn of conversation A: a new runner state over the same live session, the prompt extends the first conversation.
  const Tokens prompt2 = Extend(a1.checkpoint_tokens, 7, 5);
  // The truth for the second turn is plain decoding given the SAME call sequence as the live session saw (stage 4c, item 3): the first turn's
  // prompt, its generated tokens one row at a time, then the second turn's prefill. A fresh Sync of prompt 2 has other pass boundaries (the
  // taps of a pass wider than 16 rows differ on the production route, where wide passes round activations).
  const std::vector<std::pair<std::size_t, std::size_t>> first_passes = a.prefill;
  Truth first;
  first.prompt = std::vector<std::int32_t>(prompt1.begin(), prompt1.end());
  first.gen = std::vector<std::int32_t>(a1.checkpoint_tokens.begin() + static_cast<std::ptrdiff_t>(prompt1.size()), a1.checkpoint_tokens.end());
  const std::vector<std::int32_t> prompt2_ids(prompt2.begin(), prompt2.end());
  const Truth truth2 = RunTruthAfter(m.spec, first, first.gen.size(), prompt2_ids, limit);
  if (be.exact) {
    // Exact back end: also the fresh plain session (nothing in it depends on pass boundaries): tokens, every logits row and, as before stage 4c, the taps
    // the drafter's reference is fed (stage 4c-fix item 6: the check on the taps had been dropped when the oracle changed to `truth2`).
    const Truth fresh = RunTruth(m.spec, prompt2_ids, limit);
    Require(!truth2.taps.empty() && truth2.gen == fresh.gen && SameBits(truth2.rows, fresh.rows) && SameBits(truth2.taps, fresh.taps),
            be.name + "/" + f.name + " A: plain through the call sequence differs from a fresh plain session (tokens, logits or taps)");
  }
  std::unique_ptr<ProposalCheck> check;
  std::size_t compared = 0;
  Hooks hooks;
  hooks.after_prefill = [&](Serve& s) {
    DraftedSession* ds = DraftingOf(*s.state);
    Require(ds != nullptr, "the state has no drafting");
    const std::string e = Ledger(*ds, 0);
    Require(e.empty(), be.name + "/" + f.name + " A: after the second turn's prefill " + e);
    // The pass boundaries of the oracle are the live session's: every pass wider than 16 rows (where activations round on the production
    // route) covers the same positions.
    auto live = first_passes;
    live.insert(live.end(), s.prefill.begin(), s.prefill.end());
    Require(WidePasses(live) == WidePasses(truth2.passes), be.name + "/" + f.name + " A: the second turn's oracle has other prefill pass boundaries than the live session (" +
                                                              std::to_string(WidePasses(live).size()) + " wide passes live, " + std::to_string(WidePasses(truth2.passes).size()) + " in the oracle)");
    check = std::make_unique<ProposalCheck>(*m.setup, *m.spec, truth2, 0);
    check->Skip(*ds);
  };
  hooks.after_step = [&](Serve& s) {
    DraftedSession* ds = DraftingOf(*s.state);
    const std::string e = Ledger(*ds, 0);
    Require(e.empty(), be.name + "/" + f.name + " A: " + e);
    const std::string c = check->Check(*ds);
    Require(c.empty(), be.name + "/" + f.name + " A: " + c);
  };
  const Result a2 = a.Run(prompt2, limit, Ending::kLimit, 0, hooks);
  const Result pa2 = pa.Run(prompt2, limit);
  Require(a2.cached == a1.checkpoint && a2.cached > 0, "the second turn must continue the live slot");
  Require(a2.published == pa2.published, be.name + "/" + f.name + " A: the second turn's tokens differ from plain");
  Require(a2.drafted > 0 && check->compared > 0, "the second turn must have drafted, with proposals compared");
  compared = check->compared;

  // Conversation B's slot is untouched by A's turns.
  {
    DraftedSession* ds = nullptr;
    Hooks hb;
    hb.after_prefill = [&](Serve& s) {
      ds = DraftingOf(*s.state);
      const std::string e = Ledger(*ds, 0);
      Require(e.empty(), be.name + "/" + f.name + " A: the interleaved conversation's drafter: " + e);
    };
    const Tokens prompt_b2 = Extend(b1.checkpoint_tokens, 5, 9);
    const Result b2 = b.Run(prompt_b2, limit, Ending::kLimit, 0, hb);
    const Result pb2 = pb.Run(prompt_b2, limit);
    Require(b2.published == pb2.published, be.name + "/" + f.name + " A: the interleaved conversation's second turn differs from plain");
    Require(ds != nullptr, "no hook ran");
  }

  // A restore from the cache into the other slot resets its drafter to the restored position.
  {
    const Checkpoint ck = Capture(*a.rig);
    Hooks hr;
    hr.after_acquire = [&](Serve& s) { Require(Restore(*s.rig, ck), "restore refused"); };
    hr.after_prefill = [&](Serve& s) {
      const std::string e = Ledger(*DraftingOf(*s.state), static_cast<std::uint32_t>(ck.n));
      Require(e.empty(), be.name + "/" + f.name + " A: after a restore " + e);
    };
    const Tokens prompt3 = Extend(a2.checkpoint_tokens, 6, 11);
    const Result b3 = b.Run(prompt3, limit, Ending::kLimit, 0, hr);
    const Result p3 = pa.Run(prompt3, limit);
    Require(b3.cached == ck.n, "the restored position is the cached prefix");
    Require(b3.published == p3.published, be.name + "/" + f.name + " A: the turn restored from the cache differs from plain");
  }

  // The drafter's engine (its context rings and buffers) lives with the session: three requests on one live
  // slot construct exactly one.
  {
    auto engines = std::make_shared<std::size_t>(0);
    auto counted = std::make_shared<SpeculativeSetup>(*m.setup);
    auto shared = std::make_shared<std::size_t>(0);
    counted->shared_factory = [shared](const SpeculativeSetup&, const EngineModel&) -> std::shared_ptr<void> {
      ++*shared;
      return std::make_shared<int>(0);
    };
    const auto inner = m.setup->engine_factory;
    counted->engine_factory = [inner, engines](const SpeculativeSetup& st, const EngineModel& em) {
      ++*engines;
      return inner(st, em);
    };
    const Models cm = MakeModels(f, be, dir, 4, counted);
    Serve c(cm.spec_runner, cm.spec);
    Tokens prompt = prompt1;
    for (int turn = 0; turn < 3; ++turn) {
      const Result r = c.Run(prompt, 8);
      prompt = Extend(r.checkpoint_tokens, 5, 7 + turn);
    }
    Require(*engines == 1 && *shared == 1, be.name + "/" + f.name + " A: three requests on one live slot constructed " + std::to_string(*engines) +
                                              " drafter engines and " + std::to_string(*shared) + " shared drafters, not 1 and 1");
  }
  std::printf("  %s/%s: A. a second request on a live slot binds a new runner state over the same session: the drafter holds [0, position) after its prefill, "
              "%zu proposals equal the reference fed with all rows, output equals plain; an interleaved conversation's slot is untouched; a restore resets the drafter to the "
              "restored position\n",
              be.name.c_str(), f.name.c_str(), compared);
}

// ---- B: the state a request leaves ---------------------------------------------------------------------------------

/// Position, tokens, frontier logits and every checkpoint component (positions and bytes) of two slots.
void RequireSameState(Serve& plain, Serve& spec, const std::string& what) {
  Session &a = plain.session(), &b = spec.session();
  Require(a.Position() == b.Position(), what + ": the session is at " + std::to_string(b.Position()) + ", plain generation leaves it at " + std::to_string(a.Position()));
  Require(a.tokens() == b.tokens(), what + ": the session's tokens differ from plain's");
  Require(SameBits(a.Logits(), b.Logits()) && !a.Logits().empty(), what + ": the frontier logits differ from plain's");
  const Checkpoint ca = Capture(*plain.rig), cb = Capture(*spec.rig);
  Require(ca.n == cb.n && ca.positions.size() == cb.positions.size(), what + ": the checkpoints have another shape");
  for (std::size_t i = 0; i < ca.positions.size(); ++i) {
    Require(ca.positions[i].id == cb.positions[i].id && ca.positions[i].valid_rows == cb.positions[i].valid_rows, what + ": checkpoint frontier of component " + std::to_string(i));
    Require(ca.first[i] == cb.first[i] && ca.data[i] == cb.data[i], what + ": checkpoint component " + std::to_string(i) + " differs from plain's byte for byte");
  }
}

struct EndCase {
  std::string name;
  Tokens prompt;
  std::size_t limit;
  Ending ending;
  std::size_t n;
};

/// `cuts`: the stop-sequence and cancellation endings (group `endcut`, exact back ends only: the checkpoint they leave extends plain's by rows that came from
/// verification passes, so on the production route the next request's pass boundaries differ from plain's by construction, docs/DRAFTER-S4.md "Known
/// divergences from plain"); else the limit and stop-token endings (group `end`, every back end).
void EndTests(const TinyFixture& f, const DraftBackend& be, const std::string& dir, bool cuts) {
  if (cuts && !be.exact) {
    std::printf("  %s/%s: B (cuts). skipped: a stop sequence or a cancellation leaves more rows than plain does, so on the production route the next request's pass boundaries differ (exact back ends only)\n",
                be.name.c_str(), f.name.c_str());
    return;
  }
  const Models m = MakeModels(f, be, dir);
  const std::size_t limit = 22;
  const Truth running = FindTruth(m.spec, 23, 100 + 23 * 7, limit + 18, false);
  const Truth stopping = FindTruth(m.spec, 11, 9000, limit, true);
  std::vector<EndCase> cases = {
      {"limit 24", ToTokens(running.prompt), 24, Ending::kLimit, 0},
      {"limit 1", ToTokens(running.prompt), 1, Ending::kLimit, 0},
      {"limit 2", ToTokens(running.prompt), 2, Ending::kLimit, 0},
      {"limit 3", ToTokens(running.prompt), 3, Ending::kLimit, 0},
      {"limit 7", ToTokens(running.prompt), 7, Ending::kLimit, 0},
      {"stop token", ToTokens(stopping.prompt), 60, Ending::kLimit, 0},
      {"cancelled before step 2", ToTokens(running.prompt), 24, Ending::kCancel, 2},
      {"stop sequence at token 5", ToTokens(running.prompt), 24, Ending::kStopSequence, 5},
  };
  std::size_t checked = 0;
  for (const EndCase& c : cases) {
    if ((c.ending != Ending::kLimit) != cuts) continue;
    Serve spec(m.spec_runner, m.spec), plain(m.plain_runner, m.plain);
    const Result rs = spec.Run(c.prompt, c.limit, c.ending, c.n), rp = plain.Run(c.prompt, c.limit, c.ending, c.n);
    const std::string what = be.name + "/" + f.name + " B (" + c.name + ")";
    if (c.ending == Ending::kCancel) {
      // Cancelled after n steps: a drafted step carries more tokens than a plain one, so only the common part is compared.
      const std::size_t common = std::min(rs.published.size(), rp.published.size());
      Require(common > 0 && std::equal(rs.published.begin(), rs.published.begin() + common, rp.published.begin()),
              what + ": the request's tokens differ from plain's");
    } else {
      Require(rs.published == rp.published && rs.stop == rp.stop, what + ": the request's tokens differ from plain's");
    }
    RequireLive(spec, rs, what);
    RequireLive(plain, rp, what);
    if (c.name == "stop token") Require(rs.stop && rp.stop, what + ": the request must end by a stop token");
    const bool exact_end = c.ending == Ending::kLimit;  // limit and stop token end in plain's state
    if (exact_end) {
      RequireSameState(plain, spec, what);
    } else {
      // Cancelled and stop-sequence requests: plain leaves the unadvanced token out of its cache; the drafted request
      // has already evaluated the whole step it was cut in, so its checkpoint is at least as deep and extends plain's.
      Require(rs.checkpoint >= rp.checkpoint && std::equal(rp.checkpoint_tokens.begin(), rp.checkpoint_tokens.end(), rs.checkpoint_tokens.begin()),
              what + ": the drafted checkpoint must extend plain's");
    }
    // The next request: the conversation continues; with prefix reuse it gives plain's output.
    const Tokens next_plain = Extend(rp.checkpoint_tokens, 7, 3);
    const Tokens next_spec = exact_end ? next_plain : Extend(rs.checkpoint_tokens, 7, 3);
    const Result ns = spec.Run(next_spec, limit), np = plain.Run(next_spec, limit);
    Require(ns.published == np.published && ns.cached > 0, what + ": the next request (prefix reuse) differs from plain's");
    if (exact_end) RequireSameState(plain, spec, what + ", after the next request");
    if (!exact_end) {
      // The next prompt follows plain's cache, where the drafted slot holds more: a cold restart there, same output.
      const Result nc = spec.Run(next_plain, limit), pc = plain.Run(next_plain, limit);
      Require(nc.published == pc.published, what + ": a request that diverges inside the drafted checkpoint differs from plain's");
    }
    ++checked;
  }
  std::printf("  %s/%s: B%s. %zu request endings (%s): %s\n", be.name.c_str(), f.name.c_str(), cuts ? " (cuts)" : "", checked,
              cuts ? "a cancellation, a stop sequence" : "limits inside a draft, a stop token",
              cuts ? "the requests leave a checkpoint that extends plain's, the next request gives plain's output"
                   : "the requests leave the session, its frontier logits and every checkpoint component byte-equal to plain's, the next request with prefix reuse gives plain's output");
}

// ---- C: a drafter that fails costs speed, never the request ------------------------------------------------------

/// Which calls of the engine throw (0-based per kind), counted across every engine built from the setup.
struct FaultPlan {
  std::set<std::size_t> draft, append;
  std::size_t draft_from{SIZE_MAX};   ///< every Draft from this call on throws
  std::size_t resets_after_append{0};  ///< when an Append throws, this many following ResetTo calls throw too
  std::size_t resets_after_draft{0};   ///< the same after a Draft that throws
  std::size_t drafts{0}, appends{0}, resets{0}, pending_resets{0};
};

class FlakyEngine final : public DrafterEngine {
 public:
  FlakyEngine(std::unique_ptr<DrafterEngine> inner, std::shared_ptr<FaultPlan> plan) : inner_(std::move(inner)), plan_(std::move(plan)) {}
  const DrafterConfig& config() const noexcept override { return inner_->config(); }
  std::vector<std::uint32_t> TapLayers() const override { return inner_->TapLayers(); }
  std::uint32_t Base() const noexcept override { return inner_->Base(); }
  std::uint32_t End() const noexcept override { return inner_->End(); }
  void ResetTo(std::uint32_t base) override {
    ++plan_->resets;
    if (plan_->pending_resets > 0) {
      --plan_->pending_resets;
      throw std::runtime_error("injected ResetTo failure");
    }
    inner_->ResetTo(base);
  }
  void TruncateTo(std::uint32_t position) override { inner_->TruncateTo(position); }
  void Append(std::uint32_t first_position, std::span<const float> features) override {
    if (plan_->append.contains(plan_->appends++)) {
      plan_->pending_resets = plan_->resets_after_append;
      throw std::runtime_error("injected Append failure");
    }
    inner_->Append(first_position, features);
  }
  Block Draft(std::int32_t anchor_token, std::uint32_t anchor_position) override {
    Hit();
    return inner_->Draft(anchor_token, anchor_position);
  }
  Block Draft(std::int32_t anchor_token, std::uint32_t anchor_position, std::uint32_t max_tokens) override {
    Hit();
    return inner_->Draft(anchor_token, anchor_position, max_tokens);
  }
  std::uint64_t DeviceBytes() const noexcept override { return inner_->DeviceBytes(); }

 private:
  void Hit() {
    const std::size_t call = plan_->drafts++;
    if (plan_->draft.contains(call) || call >= plan_->draft_from) {
      plan_->pending_resets = plan_->resets_after_draft;
      throw std::runtime_error("injected Draft failure");
    }
  }
  std::unique_ptr<DrafterEngine> inner_;
  std::shared_ptr<FaultPlan> plan_;
};

std::shared_ptr<const SpeculativeSetup> Flaky(const SpeculativeSetup& base, const std::shared_ptr<FaultPlan>& plan) {
  auto setup = std::make_shared<SpeculativeSetup>(base);
  const auto inner = base.engine_factory;
  setup->engine_factory = [inner, plan](const SpeculativeSetup& s, const EngineModel& model) -> std::unique_ptr<DrafterEngine> {
    return std::make_unique<FlakyEngine>(inner(s, model), plan);
  };
  return setup;
}

void FailTests(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  const auto base = Setup(f, dir, 4);
  const std::size_t limit = 24;
  const Tokens prompt = ToTokens(SyntheticTokens(55 + f.window, 27));
  struct Case {
    const char* name;
    std::function<void(FaultPlan&)> arm;
    std::size_t failures;  ///< what the drafter's health must have counted after the first request
    bool disabled;
    std::size_t ingest{0};  ///< and the ingest's own counter (0: not checked)
    std::size_t reset_failures{0};  ///< item 5: the context resets that threw (a counter of its own; a failed step counts once, whichever calls failed inside it)
    std::size_t cold_failures{0};  ///< item 4: after a cold start on the same session (a new conversation): what the record counts after that request
    bool cold_disabled{false};
  };
  // {name, plan, failed steps, switched off, ingest failures (0: not checked), failed context resets, failures / switched off after a cold start}
  const std::vector<Case> cases = {
      {"Draft throws at calls 1 and 3", [](FaultPlan& p) { p.draft = {1, 3}; }, 2, false, 0, 0},
      {"Draft throws on every call", [](FaultPlan& p) { p.draft_from = 0; }, 3, true, 0, 0, 3, true},
      {"Draft throws from call 2 on", [](FaultPlan& p) { p.draft_from = 2; }, 3, true, 0, 0, 3, true},
      {"Append throws at calls 2, 3, 4 (the ingest path)", [](FaultPlan& p) { p.append = {2, 3, 4}; }, 3, true, 3, 0, 0, false},
      {"Append throws once and the two following ResetTo calls too", [](FaultPlan& p) {
         p.append = {3};
         p.resets_after_append = 2;
       }, 2, false, 1, 2},
      // one failed step counts once: the Draft and the ResetTo that follows it both throw, 1 failure and 1 failed reset
      {"Draft throws once and the ResetTo that follows it too", [](FaultPlan& p) {
         p.draft = {1};
         p.resets_after_draft = 1;
       }, 1, false, 0, 1},
      // the Draft throws on every call and every ResetTo after it too: each step still counts once
      {"Draft throws on every call and every ResetTo after it too", [](FaultPlan& p) {
         p.draft_from = 0;
         p.resets_after_draft = 1;
       }, 3, true, 0, 3, 3, true},
  };
  std::size_t checked = 0;
  for (const Case& c : cases) {
    const auto plan = std::make_shared<FaultPlan>();
    c.arm(*plan);
    const auto setup = Flaky(*base, plan);
    const auto spec_model = ArmedModel(f, be, *setup);
    const auto plain_model = PlainModel(f, be);
    Serve spec(CreateRunnerForModel(spec_model, "", setup), spec_model), plain(CreateRunnerForModel(plain_model, "", nullptr), plain_model);
    const std::string what = be.name + "/" + f.name + " C (" + c.name + ")";
    Result rs;
    try {
      rs = spec.Run(prompt, limit);
    } catch (const std::exception& e) {
      Require(false, what + ": the request failed: " + e.what());
    }
    const Result rp = plain.Run(prompt, limit);
    Require(rs.published == rp.published, what + ": the tokens differ from plain's");
    RequireSameState(plain, spec, what);
    DraftedSession* ds = DraftingOf(*spec.state);
    Require(ds->health().failures() == c.failures && ds->health().disabled() == c.disabled,
            what + ": the drafter counted " + std::to_string(ds->health().failures()) + " failures" + (ds->health().disabled() ? ", switched off" : "") +
                " (expected " + std::to_string(c.failures) + (c.disabled ? ", switched off" : ", still on") + ")");
    Require(ds->health().reset_failures() == c.reset_failures, what + ": the drafter counted " + std::to_string(ds->health().reset_failures()) + " failed context resets (expected " +
                                                                  std::to_string(c.reset_failures) + ")");
    if (c.ingest > 0) Require(ds->ingest().failures() == c.ingest, what + ": the ingest counted " + std::to_string(ds->ingest().failures()) + " failures");
    const std::size_t before = ds->health().failures();
    // The second turn of the conversation, with the engine still failing as planned.
    const Tokens next = Extend(rp.checkpoint_tokens, 7, 5);
    const Result ns = spec.Run(next, limit), np = plain.Run(next, limit);
    Require(ns.published == np.published, what + ": the second request differs from plain's");
    const DraftedSession* ds2 = DraftingOf(*spec.state);
    Require(ds2 == ds, what + ": the second request must keep the session's drafting");
    if (c.disabled) Require(ds2->health().failures() == before, what + ": a drafter switched off must not be tried again");
    if (c.disabled) {
      // Item 4 (stage 4c): "switched off" lives as long as the conversation. A restore to a position above 0 keeps it ...
      const Checkpoint ck = Capture(*spec.rig);
      Hooks hr;
      hr.after_acquire = [&](Serve& s) { Require(Restore(*s.rig, ck), "restore refused"); };
      const Tokens next3 = Extend(np.checkpoint_tokens, 7, 6);
      const std::size_t drafts_before = plan->drafts, appends_before = plan->appends;
      const Result rr = spec.Run(next3, limit, Ending::kLimit, 0, hr), pr = plain.Run(next3, limit);
      Require(rr.published == pr.published, what + ": the request after a restore differs from plain's");
      Require(DraftingHealthOf(*spec.state)->disabled() && DraftingHealthOf(*spec.state)->failures() == before && plan->drafts == drafts_before && plan->appends == appends_before,
              what + ": a restore to a position above 0 must keep drafting off (and not try the engine)");
      // ... a cold start (a new conversation on the slot) clears the record: drafting is tried again, with a fresh count.
      const Tokens cold = ToTokens(SyntheticTokens(901 + f.window, 21));
      const Result rc = spec.Run(cold, limit), pc = plain.Run(cold, limit);
      Require(rc.cached == 0 && rc.published == pc.published, what + ": the cold start's tokens differ from plain's");
      RequireSameState(plain, spec, what + ", after the cold start");
      const DrafterHealth* h = DraftingHealthOf(*spec.state);
      Require(plan->drafts + plan->appends > drafts_before + appends_before, what + ": after a cold start the drafter must be tried again");
      Require(h->failures() == c.cold_failures && h->disabled() == c.cold_disabled,
              what + ": after a cold start the record counts " + std::to_string(h->failures()) + " failures" + (h->disabled() ? ", switched off" : "") + " (expected " +
                  std::to_string(c.cold_failures) + (c.cold_disabled ? ", switched off" : ", on") + ")");
      if (!c.cold_disabled) Require(rc.drafted > 0, what + ": after a cold start drafting must work again");
    }
    ++checked;
  }
  std::printf("  %s/%s: C. %zu failure plans (Draft, Append, ResetTo): every request completes with plain's tokens and plain's end state\n", be.name.c_str(), f.name.c_str(),
              checked);
}

/// Stage 4c-fix item 4: a drafter that is switched off is neither reset nor warned about at later binds. Its ingest ignores rows once it is off, so its
/// context is stale on purpose; the bind used to "repair" that with a ResetTo at every request (and, when the reset threw, one warning and one more
/// failed reset per request). Ten more requests after the switch-off, every ResetTo throwing: no further reset call, no warning line, the tokens and
/// the end state are plain's.
void DisabledBindTests(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  const auto base = Setup(f, dir, 4);
  const std::size_t limit = 16;
  const auto plan = std::make_shared<FaultPlan>();
  plan->draft_from = 0;  // every Draft throws: 3 failures, switched off
  const auto setup = Flaky(*base, plan);
  const auto spec_model = ArmedModel(f, be, *setup);
  const auto plain_model = PlainModel(f, be);
  Serve spec(CreateRunnerForModel(spec_model, "", setup), spec_model), plain(CreateRunnerForModel(plain_model, "", nullptr), plain_model);
  const std::string what = be.name + "/" + f.name + " C disabled binds";
  Tokens next = ToTokens(SyntheticTokens(55 + f.window, 27));
  Result rs = spec.Run(next, limit), rp = plain.Run(next, limit);
  Require(rs.published == rp.published, what + ": the first request differs from plain's");
  const DrafterHealth* health = DraftingHealthOf(*spec.state);
  Require(health != nullptr && health->disabled() && health->failures() == DrafterHealth::kLimit, what + ": drafting must be switched off after the first request");
  const std::size_t resets_before = plan->resets, reset_failures_before = health->reset_failures(), failures_before = health->failures();
  plan->pending_resets = 1000;  // from here on every context reset throws: the old code would count and warn at each bind
  std::ostringstream captured;
  std::streambuf* const old_buf = std::cerr.rdbuf(captured.rdbuf());
  std::string failure;
  try {
    for (int turn = 0; turn < 10; ++turn) {
      next = Extend(rp.checkpoint_tokens, 2, 5 + turn);  // short requests: ten turns must fit the tiny fixtures' context cap
      rs = spec.Run(next, 4);
      rp = plain.Run(next, 4);
      if (rs.published != rp.published) throw std::runtime_error("request " + std::to_string(turn + 2) + "'s tokens differ from plain's");
    }
  } catch (const std::exception& e) {
    failure = e.what();
  }
  std::cerr.rdbuf(old_buf);
  Require(failure.empty(), what + ": " + failure);
  RequireSameState(plain, spec, what);
  const std::string log = captured.str();
  const std::size_t warnings = static_cast<std::size_t>(std::count(log.begin(), log.end(), '\n'));
  Require(plan->resets == resets_before && health->reset_failures() == reset_failures_before && health->failures() == failures_before,
          what + ": ten binds after the switch-off made " + std::to_string(plan->resets - resets_before) + " ResetTo calls (" + std::to_string(health->reset_failures() - reset_failures_before) +
              " failed) and " + std::to_string(health->failures() - failures_before) + " more failures; expected none");
  Require(warnings == 0, what + ": ten binds after the switch-off printed " + std::to_string(warnings) + " warning lines; expected none");
  std::printf("  %s/%s: C disabled binds. ten requests after the switch-off: %zu ResetTo calls, %zu warning lines, tokens and end state equal plain's\n", be.name.c_str(), f.name.c_str(),
              plan->resets - resets_before, warnings);
}

/// Item 1 (stage 4c): a drafter that cannot be BUILT (the engine, or the shared device upload) costs speed, never the request.
/// `builds` counts the attempts of each kind across the setup; a build that fails in a row 3 times switches drafting off for the session
/// and nothing is attempted after that.
void BuildFailTests(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  const auto base = Setup(f, dir, 4);
  const std::size_t limit = 16;
  const Tokens prompt = ToTokens(SyntheticTokens(55 + f.window, 27));
  struct Counts {
    std::size_t engines{0}, uploads{0};
  };
  enum class Kind { kEngine, kUpload };
  struct Case {
    const char* name;
    Kind kind;
    std::size_t fail_first;  ///< the first n attempts throw
  };
  const std::vector<Case> cases = {
      {"the engine factory throws at every build", Kind::kEngine, SIZE_MAX},
      {"the shared upload throws at every build", Kind::kUpload, SIZE_MAX},
      {"the engine factory throws once, then builds", Kind::kEngine, 1},
  };
  std::size_t checked = 0;
  for (const Case& c : cases) {
    const auto counts = std::make_shared<Counts>();
    auto setup = std::make_shared<SpeculativeSetup>(*base);
    const auto inner = base->engine_factory;
    const std::size_t fail_first = c.fail_first;
    const Kind kind = c.kind;
    setup->engine_factory = [inner, counts, kind, fail_first](const SpeculativeSetup& s, const EngineModel& m) -> std::unique_ptr<DrafterEngine> {
      if (kind == Kind::kEngine && counts->engines++ < fail_first) throw std::runtime_error("injected engine build failure");
      return inner(s, m);
    };
    setup->shared_factory = [counts, kind, fail_first](const SpeculativeSetup&, const EngineModel&) -> std::shared_ptr<void> {
      if (kind == Kind::kUpload && counts->uploads++ < fail_first) throw std::runtime_error("the GPU drafter could not be uploaded: injected");
      return std::make_shared<int>(0);
    };
    const auto spec_model = ArmedModel(f, be, *setup);
    const auto plain_model = PlainModel(f, be);
    Serve spec(CreateRunnerForModel(spec_model, "", setup), spec_model), plain(CreateRunnerForModel(plain_model, "", nullptr), plain_model);
    const std::string what = be.name + "/" + f.name + " C build (" + c.name + ")";
    Tokens next = prompt;
    std::size_t drafted_after_first = 0;
    for (int turn = 0; turn < 5; ++turn) {
      Result rs, rp;
      try {
        rs = spec.Run(next, limit);
      } catch (const std::exception& e) {
        Require(false, what + ": request " + std::to_string(turn + 1) + " failed: " + e.what());
      }
      rp = plain.Run(next, limit);
      Require(rs.published == rp.published, what + ": request " + std::to_string(turn + 1) + "'s tokens differ from plain's");
      RequireSameState(plain, spec, what + ", request " + std::to_string(turn + 1));
      if (turn > 0) drafted_after_first += rs.drafted;
      next = Extend(rp.checkpoint_tokens, 7, 5 + turn);
    }
    const DrafterHealth* health = DraftingHealthOf(*spec.state);
    Require(health != nullptr, what + ": the session has no health record");
    const std::size_t attempts = kind == Kind::kEngine ? counts->engines : counts->uploads;
    if (c.fail_first == SIZE_MAX) {
      Require(attempts == DrafterHealth::kLimit, what + ": " + std::to_string(attempts) + " builds were attempted, not " + std::to_string(DrafterHealth::kLimit) +
                                                      " (after the third failure no further build is attempted)");
      Require(health->failures() == DrafterHealth::kLimit && health->disabled(),
              what + ": the health record counted " + std::to_string(health->failures()) + " failures" + (health->disabled() ? "" : ", still on"));
      Require(DraftingOf(*spec.state) == nullptr && drafted_after_first == 0, what + ": drafting must stay off");
      // Item 4: a cold start (a new conversation) clears the record. The request that cold-starts is bound before it knows (the bind ran first), so the
      // build is attempted again at the next bind, with a fresh count.
      const Tokens cold = ToTokens(SyntheticTokens(901 + f.window, 21));
      const Result rc = spec.Run(cold, limit), pc = plain.Run(cold, limit);
      Require(rc.cached == 0 && rc.published == pc.published, what + ": the cold start's tokens differ from plain's");
      Require(health->failures() == 0 && !health->disabled(), what + ": a cold start must clear the record (" + std::to_string(health->failures()) + " failures)");
      const Result rn = spec.Run(Extend(pc.checkpoint_tokens, 7, 5), limit), pn = plain.Run(Extend(pc.checkpoint_tokens, 7, 5), limit);
      Require(rn.published == pn.published, what + ": the request after the cold start differs from plain's");
      const std::size_t attempts_now = kind == Kind::kEngine ? counts->engines : counts->uploads;
      // Stage 4c-fix item 2 (supersedes "one more build is attempted afresh"): the run of failed BUILDS is not cleared by a cold start (gufo binds
      // before it knows the request is one, so that would re-arm the build, and the weights' upload, for every request of a drafter that cannot be built).
      Require(attempts_now == attempts && health->failures() == 0 && !health->disabled() && health->build_given_up() && DraftingOf(*spec.state) == nullptr,
              what + ": after a cold start no further build may be attempted (" + std::to_string(attempts_now) + " attempts after " + std::to_string(attempts) + ", " +
                  std::to_string(health->failures()) + " failures)");
    } else {
      Require(health->failures() == 1 && !health->disabled() && health->consecutive() == 0,
              what + ": the health record must show one failure, recovered (" + std::to_string(health->failures()) + " failures)");
      Require(DraftingOf(*spec.state) != nullptr && drafted_after_first > 0, what + ": a later bind must build the drafter and draft");
    }
    ++checked;
  }
  // Stage 4c-fix item 2: five requests that each START COLD (an unrelated prompt: the bind runs first, then the session is invalidated) with a factory that
  // always throws: exactly 3 build attempts in all (the cold start's Clear() must not re-arm the build), every request completes with plain's tokens and end
  // state, one warning (the third failure's) says the drafter is given up.
  for (const Kind kind : {Kind::kEngine, Kind::kUpload}) {
    const auto counts = std::make_shared<Counts>();
    auto setup = std::make_shared<SpeculativeSetup>(*base);
    const auto inner = base->engine_factory;
    setup->engine_factory = [inner, counts, kind](const SpeculativeSetup& s, const EngineModel& m) -> std::unique_ptr<DrafterEngine> {
      if (kind == Kind::kEngine) {
        ++counts->engines;
        throw std::runtime_error("injected engine build failure");
      }
      return inner(s, m);
    };
    setup->shared_factory = [counts, kind](const SpeculativeSetup&, const EngineModel&) -> std::shared_ptr<void> {
      if (kind == Kind::kUpload) {
        ++counts->uploads;
        throw std::runtime_error("the GPU drafter could not be uploaded: injected");
      }
      return std::make_shared<int>(0);
    };
    const auto spec_model = ArmedModel(f, be, *setup);
    const auto plain_model = PlainModel(f, be);
    Serve spec(CreateRunnerForModel(spec_model, "", setup), spec_model), plain(CreateRunnerForModel(plain_model, "", nullptr), plain_model);
    const std::string what = be.name + "/" + f.name + " C build, cold starts (" + (kind == Kind::kEngine ? "the engine factory" : "the shared upload") + " always throws)";
    std::ostringstream captured;
    std::streambuf* const old_buf = std::cerr.rdbuf(captured.rdbuf());
    struct Restore {
      std::streambuf* b;
      ~Restore() { std::cerr.rdbuf(b); }
    } restore_cerr{old_buf};
    for (int turn = 0; turn < 5; ++turn) {
      const Tokens cold = ToTokens(SyntheticTokens(300 + 17 * turn + f.window, 21 + turn));
      Result rs, rp;
      try {
        rs = spec.Run(cold, limit);
      } catch (const std::exception& e) {
        Require(false, what + ": request " + std::to_string(turn + 1) + " failed: " + e.what());
      }
      rp = plain.Run(cold, limit);
      Require(rs.cached == 0, what + ": request " + std::to_string(turn + 1) + " was meant to start cold");
      Require(rs.published == rp.published && rs.drafted == 0, what + ": request " + std::to_string(turn + 1) + "'s tokens differ from plain's");
      RequireSameState(plain, spec, what + ", request " + std::to_string(turn + 1));
    }
    std::cerr.rdbuf(old_buf);
    const std::string log = captured.str();
    std::size_t given_up_lines = 0;
    for (std::size_t at = log.find("no further build is attempted"); at != std::string::npos; at = log.find("no further build is attempted", at + 1)) ++given_up_lines;
    Require(given_up_lines == 1, what + ": " + std::to_string(given_up_lines) + " lines say no further build is attempted, expected exactly one");
    const std::size_t attempts = kind == Kind::kEngine ? counts->engines : counts->uploads;
    Require(attempts == DrafterHealth::kLimit, what + ": " + std::to_string(attempts) + " builds were attempted, not " + std::to_string(DrafterHealth::kLimit));
    const DrafterHealth* health = DraftingHealthOf(*spec.state);
    Require(health != nullptr && health->build_given_up() && health->build_failures() == DrafterHealth::kLimit, what + ": the build must be given up after the third failure");
    ++checked;
  }
  // Stage 4c-fix item 1: the endings a stop sequence and a cancellation give a request that runs WITHOUT a drafter (the build failed). Under --speculative
  // the runner sets multi_token_decode, so gufo always calls DecodeStep, and with no decoder that is ext::TextModelRunner::DecodeStep: select AND
  // advance, before the scheduler publishes. A stop sequence on the E-th token then leaves L + E rows where plain (PrepareDecode: publish, stop, no
  // advance) leaves L + E - 1: one extra row, the same class as the divergence docs/DRAFTER-S4.md records for drafted requests. A cancellation between
  // steps has advanced every published token on both sides (L + n). Exact back ends only: on the production route the extra row changes the next
  // request's pass boundaries (group `fail` has production-mode gate entries; this part skips there, as group `endcut` has none).
  if (!be.exact) {
    std::printf("  %s/%s: C build (stop sequence and cancellation without a drafter): skipped, exact back ends only (docs/DRAFTER-S4.md, \"Known divergences from plain\")\n", be.name.c_str(), f.name.c_str());
  } else {
    auto setup = std::make_shared<SpeculativeSetup>(*base);
    setup->engine_factory = [](const SpeculativeSetup&, const EngineModel&) -> std::unique_ptr<DrafterEngine> { throw std::runtime_error("injected engine build failure"); };
    const auto spec_model = ArmedModel(f, be, *setup);
    const auto plain_model = PlainModel(f, be);
    const Truth running = FindTruth(spec_model, 23, 100 + 23 * 7, limit + 18, false);
    const Tokens run_prompt = ToTokens(running.prompt);
    {  // a stop sequence at the 5th published token
      Serve spec(CreateRunnerForModel(spec_model, "", setup), spec_model), plain(CreateRunnerForModel(plain_model, "", nullptr), plain_model);
      const std::string what = be.name + "/" + f.name + " C build, stop sequence at token 5, no drafter";
      const Result rs = spec.Run(run_prompt, 24, Ending::kStopSequence, 5), rp = plain.Run(run_prompt, 24, Ending::kStopSequence, 5);
      Require(rs.published.size() == 5 && rs.published == rp.published && rs.drafted == 0, what + ": the published tokens must be plain's (5, no drafts)");
      RequireLive(spec, rs, what);
      RequireLive(plain, rp, what);
      // the documented state: plain L + E - 1, the fallback L + E (its checkpoint is plain's plus the last published token, advanced)
      Require(rp.checkpoint == run_prompt.size() + 4 && rs.checkpoint == run_prompt.size() + 5 &&
                  std::equal(rp.checkpoint_tokens.begin(), rp.checkpoint_tokens.end(), rs.checkpoint_tokens.begin()),
              what + ": plain must leave L + 4 = " + std::to_string(run_prompt.size() + 4) + " rows and the fallback L + 5 (plain's plus one); got plain " + std::to_string(rp.checkpoint) +
                  ", fallback " + std::to_string(rs.checkpoint));
      Require(DraftingOf(*spec.state) == nullptr, what + ": the build must have failed (no drafting)");
    }
    {  // a cancellation before step 2
      Serve spec(CreateRunnerForModel(spec_model, "", setup), spec_model), plain(CreateRunnerForModel(plain_model, "", nullptr), plain_model);
      const std::string what = be.name + "/" + f.name + " C build, cancelled before step 2, no drafter";
      const Result rs = spec.Run(run_prompt, 24, Ending::kCancel, 2), rp = plain.Run(run_prompt, 24, Ending::kCancel, 2);
      Require(!rs.published.empty() && rs.published == rp.published, what + ": the published tokens must equal plain's in full (" + std::to_string(rs.published.size()) + " against " + std::to_string(rp.published.size()) + ")");
      RequireLive(spec, rs, what);
      RequireLive(plain, rp, what);
      // the documented state: both have advanced every published token of two steps: L + 2 rows, plain's checkpoint
      Require(rs.checkpoint == run_prompt.size() + 2 && rp.checkpoint == run_prompt.size() + 2 && rs.checkpoint_tokens == rp.checkpoint_tokens,
              what + ": after two steps both must hold L + 2 = " + std::to_string(run_prompt.size() + 2) + " rows (plain " + std::to_string(rp.checkpoint) + ", fallback " + std::to_string(rs.checkpoint) + ")");
    }
    ++checked;
    std::printf("  %s/%s: C build (no drafter): a stop sequence leaves plain's checkpoint plus one row (the documented divergence), a cancellation leaves plain's\n", be.name.c_str(), f.name.c_str());
  }
  // Stage 4c-fix item 5: the message of the exception is copied into a fixed buffer, not built as a string inside the handler (a bad_alloc there ends the
  // process). A 20,000-character message is truncated to the buffer; the request completes as plain.
  {
    auto setup = std::make_shared<SpeculativeSetup>(*base);
    setup->engine_factory = [](const SpeculativeSetup&, const EngineModel&) -> std::unique_ptr<DrafterEngine> { throw std::runtime_error(std::string(20000, 'x')); };
    const auto spec_model = ArmedModel(f, be, *setup);
    const auto plain_model = PlainModel(f, be);
    Serve spec(CreateRunnerForModel(spec_model, "", setup), spec_model), plain(CreateRunnerForModel(plain_model, "", nullptr), plain_model);
    const std::string what = be.name + "/" + f.name + " C build (a 20,000-character exception message)";
    std::ostringstream captured;
    std::streambuf* const old_buf = std::cerr.rdbuf(captured.rdbuf());
    Result rs;
    std::string failure;
    try {
      rs = spec.Run(prompt, limit);
    } catch (const std::exception& e) {
      failure = e.what();
    }
    std::cerr.rdbuf(old_buf);
    Require(failure.empty(), what + ": the request failed: " + failure);
    const Result rp = plain.Run(prompt, limit);
    Require(rs.published == rp.published, what + ": the tokens differ from plain's");
    const DrafterHealth* health = DraftingHealthOf(*spec.state);
    Require(health != nullptr && health->failures() == 1 && health->last_error().size() < 300 && captured.str().size() < 600,
            what + ": the recorded message must be bounded (" + std::to_string(health != nullptr ? health->last_error().size() : 0) + " bytes in the record, " + std::to_string(captured.str().size()) + " on stderr)");
    ++checked;
  }
  // The drafter exists but its context cannot be brought to the session's position at the bind (ResetTo throws there): that request
  // runs as plain generation, is counted, and the next bind builds on what the sink kept.
  {
    const auto plan = std::make_shared<FaultPlan>();
    const auto setup = Flaky(*base, plan);
    const auto spec_model = ArmedModel(f, be, *setup);
    const auto plain_model = PlainModel(f, be);
    Serve spec(CreateRunnerForModel(spec_model, "", setup), spec_model), plain(CreateRunnerForModel(plain_model, "", nullptr), plain_model);
    const std::string what = be.name + "/" + f.name + " C build (ResetTo throws at the bind)";
    const Result r1 = spec.Run(prompt, limit), p1 = plain.Run(prompt, limit);
    Require(r1.published == p1.published && r1.drafted > 0, what + ": the first request must draft and equal plain");
    DraftedSession* ds = DraftingOf(*spec.state);
    Require(ds != nullptr, what + ": no drafting");
    Require(ds->engine().End() > 1, what + ": the engine holds too few rows");
    ds->engine().TruncateTo(ds->engine().End() - 1);  // a context that does not end at the session's position
    plan->pending_resets = 1;
    const Tokens next2 = Extend(p1.checkpoint_tokens, 7, 5);
    Result r2, p2;
    try {
      r2 = spec.Run(next2, limit);
    } catch (const std::exception& e) {
      Require(false, what + ": the request failed: " + e.what());
    }
    p2 = plain.Run(next2, limit);
    Require(r2.published == p2.published && r2.drafted == 0, what + ": the request must run as plain (no drafts) with plain's tokens");
    RequireSameState(plain, spec, what);
    Require(DraftingHealthOf(*spec.state)->failures() == 1 && DraftingHealthOf(*spec.state)->reset_failures() == 1 && DraftingOf(*spec.state) == nullptr,
            what + ": the failed bind must be counted once (and as one failed reset) and leave drafting off for it");
    const Tokens next3 = Extend(p2.checkpoint_tokens, 7, 6);
    const Result r3 = spec.Run(next3, limit), p3 = plain.Run(next3, limit);
    Require(r3.published == p3.published, what + ": the request after the failed bind differs from plain's");
    Require(r3.drafted > 0 && DraftingHealthOf(*spec.state)->failures() == 1,
            what + ": the next bind must draft again (drafted " + std::to_string(r3.drafted) + ", failures " + std::to_string(DraftingHealthOf(*spec.state)->failures()) + ")");
    RequireSameState(plain, spec, what + ", next request");
    ++checked;
  }
  std::printf("  %s/%s: C build. %zu plans (the engine factory throws, the shared upload throws, ResetTo throws at the bind): every request completes with plain's tokens and end state, the health record counts the "
              "failed builds, after the third in a row no build is attempted, a build that recovers drafts again\n",
              be.name.c_str(), f.name.c_str(), checked);
}

// ---- D: requests the way gufo's pool and scheduler issue them -------------------------------------------------------
// gufo's TextRunnerPool and ComponentTextCache live in gufo's serving library, which an out-of-tree extension may not
// link (cmake/GufoExtensionDiscovery.cmake rejects gufo_serving) and which no public header declares. Serve::Run is the
// closest this tree allows: a new runner state per acquire, the scheduler's budget and preview arithmetic, Commit/Cancel
// checkpoint bounds. What it cannot reach: the cache's own bookkeeping (snapshot publication, disk, leases); those are
// gate entries for the night run (docs/DRAFTER-S4.md, Follow-ups, item D).

void FlowTests(const TinyFixture& f, const DraftBackend& be, const std::string& dir) {
  const Models m = MakeModels(f, be, dir);
  const std::size_t limit = 20;
  const std::string tag = be.name + "/" + f.name + " D";
  const Tokens prompt_a = ToTokens(SyntheticTokens(55 + f.window, 27)), prompt_b = ToTokens(SyntheticTokens(901 + f.window, 21));
  std::size_t checked = 0;

  // 1. two sessions interleaved at step granularity: a whole request of B runs inside every step of A's second turn.
  {
    Serve a(m.spec_runner, m.spec), b(m.spec_runner, m.spec), pa(m.plain_runner, m.plain), pb(m.plain_runner, m.plain);
    const Result a1 = a.Run(prompt_a, limit), pa1 = pa.Run(prompt_a, limit);
    Require(a1.published == pa1.published, tag + " interleaved: first turn of A differs from plain");
    Tokens b_prompt = prompt_b, pb_prompt = prompt_b;
    std::size_t inner = 0;
    Hooks h;
    h.after_step = [&](Serve&) {
      const Result rb = b.Run(b_prompt, 6), rpb = pb.Run(pb_prompt, 6);
      Require(rb.published == rpb.published, tag + " interleaved: a request of B inside A's step differs from plain");
      // the conversation continues while it fits the context cap, then starts over (a cold start on the same slot)
      const bool fits = rb.checkpoint_tokens.size() + 4 + 6 <= 90;
      b_prompt = fits ? Extend(rb.checkpoint_tokens, 4, 3 + inner) : prompt_b;
      pb_prompt = fits ? Extend(rpb.checkpoint_tokens, 4, 3 + inner) : prompt_b;
      ++inner;
    };
    const Tokens p2 = Extend(a1.checkpoint_tokens, 6, 5);
    const Result a2 = a.Run(p2, limit, Ending::kLimit, 0, h), pa2 = pa.Run(p2, limit);
    Require(inner > 1 && a2.cached == a1.checkpoint && a2.published == pa2.published, tag + " interleaved: A's second turn differs from plain, or B did not interleave");
    DraftedSession* ds = DraftingOf(*a.state);
    Require(ds != nullptr && Ledger(*ds, 0).empty(), tag + " interleaved: A's drafter lost its context");
    ++checked;
  }

  // 2. a cancelled request, then a new one: the same conversation, then an unrelated prompt (cold), then a restore into another slot.
  {
    Serve a(m.spec_runner, m.spec), b(m.spec_runner, m.spec), pa(m.plain_runner, m.plain), pb(m.plain_runner, m.plain);
    const Truth running = FindTruth(m.spec, 23, 100 + 23 * 7, limit + 18, false);
    const Tokens p1 = ToTokens(running.prompt);
    const Result c1 = a.Run(p1, 24, Ending::kCancel, 2), pc1 = pa.Run(p1, 24, Ending::kCancel, 2);
    const std::size_t common = std::min(c1.published.size(), pc1.published.size());
    Require(common > 0 && std::equal(c1.published.begin(), c1.published.begin() + common, pc1.published.begin()), tag + " cancel: tokens differ from plain");
    RequireLive(a, c1, tag + " cancel");
    const Tokens next = Extend(c1.checkpoint_tokens, 5, 4);
    const Result n1 = a.Run(next, limit), pn1 = pa.Run(next, limit);
    Require(n1.published == pn1.published && n1.cached > 0, tag + " cancel: the request after a cancellation differs from plain");
    const Result u1 = a.Run(prompt_b, limit), pu1 = pa.Run(prompt_b, limit);  // unrelated: cold start on the same slot
    Require(u1.published == pu1.published && u1.cached == 0, tag + " cancel: an unrelated request after a cancellation differs from plain");
    // the checkpoint the cancelled request left is restored into another slot
    const Result c2 = b.Run(p1, 24, Ending::kCancel, 2);
    const Checkpoint ck = Capture(*b.rig);
    Hooks hr;
    hr.after_acquire = [&](Serve& s) { Require(Restore(*s.rig, ck), "restore refused"); };
    const Tokens next2 = Extend(c2.checkpoint_tokens, 5, 8);
    const Result r2 = a.Run(next2, limit, Ending::kLimit, 0, hr), pr2 = pb.Run(next2, limit);
    Require(r2.cached == ck.n && r2.published == pr2.published, tag + " cancel: a restore of a cancelled checkpoint differs from plain");
    ++checked;
  }

  // 3. a request cut by its limit inside a draft, its checkpoint restored into another slot, which continues the conversation.
  for (const std::size_t cut : {1u, 3u, 7u}) {
    Serve a(m.spec_runner, m.spec), b(m.spec_runner, m.spec), pa(m.plain_runner, m.plain), pb(m.plain_runner, m.plain);
    const Result r1 = a.Run(prompt_a, cut), p1 = pa.Run(prompt_a, cut);
    Require(r1.published == p1.published, tag + " cut " + std::to_string(cut) + ": tokens differ from plain");
    RequireSameState(pa, a, tag + " cut " + std::to_string(cut));
    const Checkpoint ck = Capture(*a.rig);
    Hooks hr;
    hr.after_acquire = [&](Serve& s) { Require(Restore(*s.rig, ck), "restore refused"); };
    hr.after_prefill = [&](Serve& s) { Require(Ledger(*DraftingOf(*s.state), static_cast<std::uint32_t>(ck.n)).empty(), tag + ": drafter after a restore"); };
    const Tokens next = Extend(r1.checkpoint_tokens, 6, 2);
    const Result r2 = b.Run(next, limit, Ending::kLimit, 0, hr), p2 = pb.Run(next, limit, Ending::kLimit, 0, {});
    Require(r2.cached == ck.n && r2.published == p2.published, tag + " cut " + std::to_string(cut) + ": the restored continuation differs from plain");
    ++checked;
  }
  // 4. a restore into a slot whose drafter holds an old branch past the restored position, then a request with NO prompt rows left to prefill (the whole
  // prompt is the restored checkpoint, decoding starts from its frontier logits). Nothing feeds the drafter's ingest before the first proposal, so the
  // drafter's context is the restore's own doing: it must be reset to the restored position by the session's `Replaced(n)`, and the first
  // proposal must find it there (no desync). A request with prompt rows after the restore would hide a missing reset: its first pass truncates or
  // resets the context by itself (stage 4b's mutant "Replaced is a no-op" survived for that reason).
  {
    Serve a(m.spec_runner, m.spec), b(m.spec_runner, m.spec), pa(m.plain_runner, m.plain), pb(m.plain_runner, m.plain);
    const Result r1 = a.Run(prompt_a, 7), p1 = pa.Run(prompt_a, 7);
    Require(r1.published == p1.published, tag + " restore without prompt rows: the first request differs from plain");
    const Checkpoint ck = Capture(*a.rig);
    const Result old_branch = b.Run(prompt_b, limit), p_old = pb.Run(prompt_b, limit);  // the drafter of b holds rows past the restored position
    Require(old_branch.published == p_old.published && b.session().Position() > ck.n, tag + " restore without prompt rows: the old branch must reach past the restored position");
    Hooks hr;
    hr.full_prefix = true;
    hr.after_acquire = [&](Serve& s) { Require(Restore(*s.rig, ck), "restore refused"); };
    Hooks hp;
    hp.full_prefix = true;
    const Result r2 = b.Run(r1.checkpoint_tokens, limit, Ending::kLimit, 0, hr), p2 = pa.Run(r1.checkpoint_tokens, limit, Ending::kLimit, 0, hp);
    Require(r2.cached == ck.n && r2.prefill_rows == 0 && p2.prefill_rows == 0, tag + " restore without prompt rows: the request must start from the restored frontier without a prefill pass");
    Require(r2.published == p2.published, tag + " restore without prompt rows: the tokens differ from plain's");
    Require(r2.drafted > 0, tag + " restore without prompt rows: the request must have drafted");
    DraftedSession* ds = DraftingOf(*b.state);
    Require(ds != nullptr, tag + " restore without prompt rows: no drafting");
    Require(ds->backend().desyncs() == 0 && ds->ingest().failures() == 0 && ds->health().failures() == 0,
            tag + " restore without prompt rows: the first proposal found the drafter's context off the restored position (" + std::to_string(ds->backend().desyncs()) +
                " desyncs): the session's Replaced(n) did not reset it");
    RequireSameState(pa, b, tag + " restore without prompt rows");
    ++checked;
  }
  std::printf("  %s/%s: D. %zu flows through the runner as gufo's pool issues them (two sessions interleaved inside each other's steps, a cancellation then a new request, "
              "a limit inside a draft then a restore into another slot, a restore with no prompt rows after it and no desync): tokens equal plain's\n",
              be.name.c_str(), f.name.c_str(), checked);
}

}  // namespace

int main(int argc, char** argv) {
  const std::string filter = argc > 1 ? argv[1] : "all";
  const std::string dir = argc > 2 ? argv[2] : "/server/staging/ripple-gufo/perf/stage4";
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
        if (wants("reuse", f)) {
          ReuseTests(f, be, dir);
          ++groups;
        }
        if (wants("end", f)) {
          EndTests(f, be, dir, false);
          ++groups;
        }
        if (wants("endcut", f)) {
          EndTests(f, be, dir, true);
          ++groups;
        }
        if (wants("flow", f)) {
          FlowTests(f, be, dir);
          ++groups;
        }
        if (wants("fail", f)) {
          FailTests(f, be, dir);
          BuildFailTests(f, be, dir);
          DisabledBindTests(f, be, dir);
          ++groups;
        }
      }
    }
    Require(groups > 0, "no group matches '" + filter + "'");
  });
}

#endif  // ENGINE_ENABLE_HIP

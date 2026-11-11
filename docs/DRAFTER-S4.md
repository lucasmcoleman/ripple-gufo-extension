# Drafter stage 4: the drafter joined to gufo's speculative verifier (CPU part)

Stage 4 of the DFlash2 drafter design (`/server/staging/ripple-gufo/perf/DRAFTER-GUFO-DESIGN.md` sections 1, 3.4, 6 and 7; job 1692).
Branch `drafter-s4` from `drafter-s3` (4878c73). Written 2026-11-11. **Everything here ran on the CPU executor and the host-ops back end
(plus the scalar CPU drafter reference); nothing ran on a GPU and no real model was loaded.** The GPU commands are in
`/server/staging/ripple-gufo/perf/STAGE4-GPU.md`.

With no `--speculative` option nothing changes: the runner builds the same state as before, `multi_token_decode` stays false, the engine
keeps `verify_rows` 0 and no capture (proof below).

## Part 0: `Rewind(0)` (commit "executor: Rewind(0) is refused ...")

DRAFTER-S3 item 4 was right about the mechanism and **too narrow about the trigger**. `Executor::Rewind(p)` tests
`needed = WindowFirst(p) < p && needed < RetainedBegin()`; at `p = 0` the first conjunct is false, so `Rewind(0)` returned true while the high
water mark and the import floor stayed. `RetainedBegin()` then stayed above 0 and the next pass from 0 failed with "window before the position is
not retained" (a prompt longer than `max_rows` fails on its second pass). It needs `Sync` with common prefix 0 and `RetainedBegin() > 0`, i.e. **a high
water mark above the ring size R (3,072 on the real model, not ~4,100 tokens)**, or **a restored checkpoint whose floor is above 0**.
A second, silent consequence on the same path: after `Rewind(0)` with high > R the forward plan said `written = [0, n)`, `evicted` empty, but
writing position q reuses the ring slot of old position q + R; rows of the old branch that a checkpoint had borrowed were overwritten without the
write guard being told.

**Fix** (in `Executor::Rewind`, the brief's): `position == 0 && RetainedBegin() > 0` returns false, state unchanged. Every caller already handles false: `SyncImpl`
takes its existing not-rewound path (guard told `{0, cap}`, `Reset`, `Fail`, recompute) and `AcceptVerified` (never at 0: a verification needs frontier logits)
resets and fails. I chose this over the reviewer's alternative (treat common == 0 as not rewound in `SyncImpl`, `Rewind` untouched) because the
hole is in the executor's contract, not in one caller, and a future caller of `Rewind(0)` would otherwise meet it; it does not zero the high water
mark or the floor (that would keep the write-guard hole). No position p > 0 is affected: for p >= 1 `WindowFirst(p) < p` (window >= 2), so the existing
test already covers it.

**Checks written first and run on the unfixed code** (`perf/stage4-part0-unfixed-*.log`):
- contract level, check 9 of `tests/executor_contract_test.cpp` on every back end the file covers (CPU executor, host ops f32 and f16; new checks only): a rewind to 0 after the
  ring wrapped, after an import (floor above 0), and where nothing is lost (high water within the ring: must still succeed); a refused rewind changes nothing and
  Reset + recompute equals a fresh executor. Unfixed: `FAIL ... rewind to 0 accepted after the ring wrapped (RetainedBegin 21)` on tinyA CPU and host ops (the run stops at the first failing check).
- session level, group `rewind0` of `tests/draft_session_test.cpp` (both fixtures, CPU executor and host ops): 73 tokens (longer than the ring), then `Sync` to an unrelated 30-token text in two
  passes, and a checkpoint restored at 61 then an unrelated 70-token text in three passes; logits and tokens equal a fresh session's, then 8 greedy 1-row steps, bit for bit; the write
  guard must hear `{0, max_context}` first. Unfixed, both scenarios on both fixtures fail with `ricursive executor: window before the position is not retained`
  (the CPU executor's run; the host-ops runs come after it and were not reached).
- Fixed: contract 26/26 (CPU) and 52/52 (host) checks pass, both `rewind0` groups pass on CPU and host ops, `plain_fingerprint` (golden hashes untouched), `engine`, `continuation` and the gate selftest pass.
The `rewind0` group adds two gate entries (`draft-rewind0-tinyA/B`, selftest counts 68 to 70).

## What was added (Part 1)

| file | what |
| --- | --- |
| `src/engine.{hpp,cpp}` | `TapSink` and `Session::SetTapSink`: with a sink attached the captured rows go to it, one pass at a time, instead of the pending buffer. A verification's rows go when it settles, without the rejected ones; `Rewound(p)` after a prefix-reuse rewind, `Replaced(p)` after a reset, failure, full recompute (p = 0) or restore (p = restored position). `Verify` settles a verification still open. Without a sink nothing changes |
| `src/drafter_reference.*` | `ResetTo(base)`, `TruncateTo(position)`, `ContextBase()`: the context is rows `[base, End)`, rows below `base` are never read; `Draft::context_first/end` (the rows a draft read); fault `context_one_row_short` |
| `src/drafter_backend.*` | `DrafterEngine` (interface: `Base/End`, `ResetTo`, `TruncateTo`, `Append`, `Draft`, no host assumption), `DrafterIngest` (`TapSink` to engine), `DFlash2DraftBackend : IDraftBackend`, `DraftBackendFaults` |
| `src/drafter_engine_cpu.*` | `CpuDrafterEngine`: the stage 1 reference behind the interface (in `gufo_ricursive_reference`) |
| `src/speculative_target.*` | `SessionSpeculativeTarget : ISpeculativeTargetExecutor` over `Session`; `SpeculativeDecoder` (the step loop around `SpeculativeVerifier`); empty without `ENGINE_ENABLE_HIP` |
| `src/runner.*` | `--speculative dflash2` accepted: `LoadSpeculativeSetup`, `ArmForSpeculation`, `CreateRunnerForModel`, `DraftedSession`; `multi_token_decode`, `DecodeStep`; the runner refuses everything but dflash2 |
| `src/drivers.cpp` | `prompt`, `chat` and `bench` take `--speculative dflash2 --dflash-model F` (drafted greedy generation, "Drafted N tokens, accepted M") |
| `src/provider.cpp`, `extension.json` | defaults `{dflash2, max 4 draft tokens, policy fixed}`; the provider's `speculative`, `serving.draft_options`, and the `--dflash-model` option, as Qwen declares them |
| `tools/dump_taps.cpp` | `--sweep-widths`: every pass width 2..16 against 1-row decoding on the real model (GPU gate, STAGE4-GPU.md section 2) |
| tests | `tests/speculative_test.cpp` (ctest `ricursive.speculative`), `tests/draft_rig.hpp` (the checkpoint helpers of `draft_session_test`, copied: that file is untouched), `tests/draft_util.hpp` (+ production GPU mode) |

## How one step flows through gufo's verifier

`SpeculativeDecoder::Step(max_tokens, sampler, stop_at_eos)` for a request the verifier would not treat as sampled (no temperature, penalty, grammar, reasoning budget, log-probabilities):

1. If the greedy token of the session's frontier logits has not been emitted yet (`published()` false), emit it. This is the token `PreviewFirstToken` showed.
2. `SpeculativeVerifier::VerifyStep(sequence = session tokens + the frontier token, position = session position, current token = the frontier token, ...)`:

| verifier call | lands in |
| --- | --- |
| `Propose(sequence, position, d)` (`PrepareStep`, `d = min(max draft, budget - 1)`) | `DFlash2DraftBackend::Propose`: drafts from the anchor with the context ending exactly at `position`; a context that does not end there is reset, nothing is proposed and `desyncs()` counts it |
| `SaveState(position)` | records the position |
| `ForwardVerificationChunk([current, d1..dn], position, hidden off, logits off)` | `Session::Verify` (n + 1 rows, one pass, all logits); predictions = the sampler's greedy rule on each row (the function plain decoding selects with) |
| accepted prefix, `UpdateTargetHidden` | not called: `RequiresTargetHiddenStates()` is false, the rows reach the drafter through the session's tap sink |
| partial accept: `RestoreState()` (no-op), `CommitVerificationChunk(committed, position)` | `Session::AcceptVerified(committed)`: the exact rewind, no accepted token is forwarded again; the kept rows' taps go to the sink |
| full accept: only `FinishVerification()` | nothing: the decoder calls `SessionSpeculativeTarget::Settle()` after `VerifyStep` (`AcceptVerified` of all rows) |
| no draft (budget 1, empty proposal): `AdvanceCommittedToken` | `ForwardToken` = `Session::Evaluate` |
| `AcceptFeedback(accepted, correction)` | no-op |

3. The emitted tokens are the accepted drafts and the correction token (a stop token ends the request and is not emitted). The correction token is **pending**: emitted, not yet in the target; it
   is the next step's first row. The session is then at `position + accepted + 1`, its frontier logits are the logits that chose the correction, `published()` is true.

Sampled requests, penalties, grammars, reasoning budgets and log-probabilities never reach the verifier: `Runner::DecodeStep` calls `TextModelRunner::DecodeStep` (select, advance), as without speculation.
`final_token_advance_required` is false with drafting (the last token of a request stays pending; `CheckpointPosition` is the session's position, so the cache keeps everything but it, as
Qwen's runner does; the next request's prompt carries it). `PreparePrefixReuse`, `Prefill` and `ReconcileComponentState` (after a restore) reset `published()` and the verifier's request state.

## Drafter context bookkeeping

The drafter holds the features of the contiguous positions `[Base, End)`; the invariant after every step is `Base == 0 && End == session position` (or `Base == restored position` after a restore), asserted by the tests.
The rows come only from the session's tap sink, so they are exactly the rows the session keeps:

| event | what the drafter does |
| --- | --- |
| prompt pass, one-row step, verification settled | `Append` the kept rows (a rejected row never reaches it) |
| `Sync` that rewinds to `common` | `TruncateTo(common)`, then the new rows |
| `Sync` that cannot rewind (full recompute, including the band below), `Reset`, a failed pass | `ResetTo(0)`, then the rows from 0 |
| cache restore at n | `ResetTo(n)`: no rows below n, so a restored long context drafts with a short context (lower acceptance, never wrong output) |
| a row at a position the context does not continue | rewind or hole: truncate or restart there |
| `Propose` with the context not ending at the anchor | reset, no proposal, counted |

The drafter never reads a row it was not given: a draft reads `[max(Base, anchor - context_window), anchor)`; the tests record the range of every draft and require it inside `[Base, anchor)`.
The backend is free of host assumptions: rows arrive in passes of at most one prefill pass, the context has the drafter's own bounded window (`context_window` 16384; the CPU reference does not free rows below it, a GPU engine can).

**Tap memory.** With a sink attached a multi-pass `Sync` hands over each pass's rows when the pass ends and keeps none: a 97-token prompt in 20-row passes gives 6 deliveries of at most 20 rows and
`TakeTaps()` stays empty (the 3.3 GB of a 32k-token prompt is never held; the runner's `Prefill` is also one pass per call).

## The recompute band: a known difference in cache reuse between drafted and plain sessions

*(This corrects DRAFTER-S3 item 4's sentence that stage 4 prefix reuse is not exposed; S3 itself is not rewritten.)*
After a rejected verification `AcceptVerified` moves only the executor's logical end back; the high water mark stays at p + k and the ring slots of the rejected rows overwrote older positions,
so `RetainedBegin()` is up to **15 rows** (high water p + k, logical end p + 1 when everything is rejected: k - 1 with k <= 16) higher than after plain decoding of the same tokens. A later `Sync` whose common prefix falls in that band (about R - W + 1 rows behind the end, 2,049 on the real model)
gets `Rewind` false and **recomputes from 0 where plain decoding would have rewound**. Output is unchanged (on the CPU a recompute is bit-identical; on the GPU it is as exact as any prefill), the cost is a longer prefill, and
the drafter's context is rebuilt from 0 by the same sink events (tested: `4c`). Removing it would need ring slack (rows beyond R) and a change of the retention ledger of the executor contract: a decision for the coordinator, not done.
The measured example (tests, `4c`): tinyB ring 36, end 55: plain retained from 19, drafted from 34; a Sync with common prefix 34 rewinds on the plain session and recomputes on the drafted one.

## Tests (`tests/speculative_test.cpp`, ctest `ricursive.speculative`; 3 min 41 s on 4 threads in the suite run, the whole new test)

CPU executor and host ops, both fixtures (tinyA window 8, ring 28; tinyB window 16, ring 36; 20-row prefill passes), the real `SpeculativeVerifier`, stage 1's tiny drafters
(`/server/staging/ripple-gufo/perf/stage4/drafter-tiny{A,B}-q8_0.gguf`, generated by `tests/fixtures_text/make_drafter_fixture.py <fixture> OUT --quant q8_0 --taps 4`; the tinyB file is byte-identical to stage 1's).
Every run compares the emitted tokens with plain greedy decoding and **every committed logits row of every verification, and the frontier after every step, bit for bit** with plain 1-row decoding of the same tokens.
Results of the run of 2026-11-11 on the final tree (`perf/stage4-spec-all2.log`; counts per back end and fixture, cpu/tinyA, cpu/tinyB, hostops/tinyA, hostops/tinyB):

1. **Hostile drafts** (mock back end, modes always right, always wrong, random right/wrong with a fixed seed, right then wrong at every split point, a stop token inside a draft; prompts of 5 and 23 tokens so
   the generation crosses W, 2W+3, the ring size and the ring wrap, and a prompt whose plain run ends in a stop token that falls inside a draft; budgets 1 to 4 with 15 drafts on offer, i.e. the generation limit inside a draft):
   **229 runs on the CPU executor** (all 15 draft lengths, 5 modes, 3 prompts, 4 limits) and **94 on the host ops** (6 draft lengths of 15: 1, 2, 3, 5, 8, 15; the host ops take several times longer here; this is a reduction against the brief's
   "every draft length 1 to 15" for that back end only, made for the 5-minute budget), per fixture: 5,623 / 6,484 / 2,310 / 2,667 logits rows compared, 10,329 / 12,447 / 3,076 / 3,669 tokens drafted,
   1,727 / 2,066 / 630 / 753 accepted. **0 differences.** The always-right runs accept every draft, the always-wrong runs none, the verifier's own statistics equal the decoder's accounting.
2. **The real tiny drafter, d = 1..15** (15 runs per back end and fixture, 22 tokens, prompts of 5, 13, 23, 29 tokens): exactness holds (630 rows compared per run set, 1,840 tokens drafted), accounting is consistent
   with the tokens emitted and the verifier's statistics, and the context invariant above holds after every step. **The random tiny drafter never has a draft accepted (0 accepted in all 60 runs)**, so the "accepted > 0" paths are exercised by the mock drafts only.
3. **Proposals equal the reference**: at every step of test 2 in which the backend proposed (300 proposals per back end and fixture) the 15 tokens equal `DrafterReference` driven directly with plain decoding's taps and the same anchor token. 300 of 300.
   **Twins** (each must fail test 3 while the output stays exact; the proposals of 3 draft lengths each, cpu/tinyA, cpu/tinyB, hostops/tinyA, hostops/tinyB): anchor position off by one: 60 of 60 differ in all four; anchor token off by one: 58 of 60, 59 of 59, 58 of 60, 59 of 59; context one row short: 60 of 60 in all four. None passed.
4. **Prefix reuse, restore and the band**: (4a) drafted generation, then a text sharing the first 24 tokens, drafted again: output equals plain, the context equals the session's, proposals equal the reference; (4b) a checkpoint restored
   into a slot whose drafter held an older branch: `Base == End == 40`, output equals plain, proposals equal the reference fed from position 40, the first proposal reads no context row, no row below 40 is read in any later one;
   (4c) the recompute band (above): the drafted session recomputes (guard told the whole context), output equals plain, the drafter's context was rebuilt from 0.
5. **Sampled requests draft nothing**: through the runner (prefill, preview, `DecodeStep`, a second request with prefix reuse on the same state): greedy requests give the tokens of the runner without speculation and draft; sampled (T 0.9, top-k 40, seed 11),
   repeat-penalty and log-probability requests draft nothing (0 drafted) and give the same tokens. A drafted request ends with its last token pending (`CheckpointPosition = prompt + emitted - 1`), a plain one with every token advanced; the second request
   reuses the prefix and carries the pending token. The first-token preview equals the first selection. The prompt driver (`prompt --speculative dflash2`, CPU, both fixtures) prints the plain run's text.
6. **Tap memory**: see above (97-token Sync, 6 deliveries of at most 20 rows, nothing pending; a verification delivers only its kept rows when it settles).
7. **Twin for test 1**: the adapter keeps one more row than the verifier decided (`SessionSpeculativeTarget::Faults::accept_one_more`): 10 of 10 runs fail on every back end and fixture (every always-wrong run, as it must).

GPU gate entries: 12 added to `engine_tests` (`spec-{mock,real,runner}-{tinyA,tinyB}` with `gpu-exact` and `spec-prod-...` with `RICURSIVE_TEST_BACKEND=gpu`, the production route), plus the 2 of Part 0; the selftest counts 68 to 82 (invocations and PASS rows), 69 to 83 (logs), 35 to 49 (NOT-RUN rows when the lock is refused).
Production mode skips 4a, 4c and the runner's second request (a resynced session and a fresh one round differently there by construction; said in the test and in the log).

**Plain generation is unchanged**: `ricursive.plain_fingerprint` passes with its golden hashes untouched; no existing test source was edited (`git diff 4878c73 -- tests/` shows only added files, the
`draft_util.hpp` production mode, the gate list and its self-test counts, and the new checks of Part 0); the engine's changes are inert without a sink (`SetTapSink` default null); a runner without `--speculative` builds the same state and capabilities as before
(tested: `multi_token_decode` false, `final_token_advance_required` true, no drafting object). Suite result below.

## Disagreements with the design, things it did not say, things that were surprising

1. **Features reach the drafter through the session's tap sink, not through `UpdateTargetHidden`/`AppendTargetContext`.** The design's route (verifier hands committed rows to the draft) would need the hidden rows in
   `VerificationChunkResult` for every verification and a second route for prompt passes; the session already knows exactly which rows it keeps after a rejection, a rewind, a recompute or a restore, so one route there
   is simpler and cannot disagree with the cache. The backend therefore reports `RequiresTargetHiddenStates() == false`.
2. **A full accept never calls `CommitVerificationChunk`** (design 3.3 maps commit to the rewind). The verification stays open until something settles it; `Settle()` after `VerifyStep` does (needed so that the sink has the rows before the next `Propose`).
3. **The verifier's declarations exist only with `ENGINE_ENABLE_HIP`** (`speculative_verifier.hpp` is empty otherwise); the design calls the headers public (true) but not that. Non-HIP builds get stand-ins and a refusal.
4. **Production has no drafter yet**: `MakeDrafterEngine` returns the CPU reference only where `RICURSIVE_WITH_CPU_BACKEND` is defined (the test build); a build without it refuses `--speculative dflash2`. With the CPU reference the output is exact and decoding is
   slower than plain (about 1 s per step). The runner prints that warning. The GPU drafter plugs in behind `DrafterEngine`.
5. DRAFTER-S3 item 4 (Part 0 above) and its claim that stage 4 prefix reuse is not exposed (the band) were wrong. `Executor::Rewind` was the right place for the first fix.
6. The cost of ingesting rows is paid for requests that never draft (sampled requests): the sink feeds the drafter every kept row. Not optimised (the GPU ingest is cheap; skipping it would need the request's sampler at prefill time).
7. The drafter's memory is not in `ResourceClaim`/`MeasuredResources` (the CPU reference is a test object); the GPU drafter's K/V (about 130 MiB per session at 16k, design 3.3) must be added there.
8. `GetEosTokenId()` returns the first id of `kStopTokens` (100265), which is outside the tiny fixtures' vocabulary; the verifier ORs it with `IsStopToken`, which uses the model's stop set, so behaviour is right.
9. On a non-exact GPU the next request of a conversation prefills the pending token inside a pass, where plain decoding evaluated it in a 1-row pass: the same chunk-versus-decode difference the runner already declares (`incremental_text_is_exact`), no new class.
10. Not done / not verified: the non-HIP stand-ins (`speculative_target.hpp`, `runner.hpp`) were not compiled (this tree is a HIP build); nothing on the GPU (the production-route claim that 2..16-row passes equal 1-row passes on the real model rests on the gates in STAGE4-GPU.md); the `chat` and `bench` driver paths are compiled but not run (the `prompt` driver is tested);
    the `--dflash-model` option declared in `extension.json` is also a core option (same name and type, which gufo allows); `gufo prompt`'s flag spelling in STAGE4-GPU.md section 3 was not checked.

## Suite result (2026-11-11, final tree)

`ctest -R '^ricursive\.' -L cpu` in `build-test`: **25 of 25 pass** (24 before, plus `ricursive.speculative`), 20 min 48 s wall (`perf/stage4-ctest-cpu.log`); `draft_session` 812 s (it gained the
`rewind0` group), `speculative` 221 s, `plain_fingerprint` 3 s with its golden hashes untouched. The existing tests were not edited except for the added checks of Part 0.

## Follow-ups (stage 4b, branch `drafter-s4b`, job 1692)

From the independent review of 4765c3b. Items A to D, one commit each. Test logs: `/server/staging/ripple-gufo/perf/stage4b-*.log`.
The new checks are `tests/speculative_followup_test.cpp` (ctest `ricursive.speculative_followup`, and `_ops` with the drafter executor), helpers copied
from `tests/speculative_test.cpp` into `tests/spec_util.hpp` (that file is untouched except where a statement below says so).

### Item A: the drafter's context survives from one request to the next (and its device state is made once)

**Superseded:** "Drafter context bookkeeping" above and the runner row of the table in "What was added" assumed one `DraftedSession` per runner state. gufo binds a
NEW state on every cache acquire (`component_text_cache.cpp`, `Lease::Impl`), so under `gufo serve` every request built a fresh drafter engine whose
context started at the session's position: output exact, acceptance lost on every turn after the first.
- `DraftedSession::For(session, setup)`: the drafting belongs to the session (`Session::SetAttachment`, destroyed with the session). A state finds the one attached
  for its setup and keeps it; a context that does not end at the session's position restarts there (it cannot, short of a missed event: the tap sink hears every
  reset, rewind, restore and recompute). A restore still resets it to the restored position through the sink. `Restart()` clears the pending token.
- Failing first (`perf/stage4b-A-unfixed.log`): `A: after the second turn's prefill the drafter holds rows [50, 57) but the session is at 57 (expected from 0)`.
  Fixed (`stage4b-A-fixed.log`): two requests the way serve issues them, second prompt extending the first; the drafter holds `[0, position)`, 22 proposals equal
  `DrafterReference` fed with all rows, output equals plain; an interleaved second slot is untouched; a restore into the other slot leaves `[n, n)`.
- Stage 2 addition (coordinator's review): the GPU drafter's device weights were kept alive only by weak pointers (`hip/drafter_gpu.cpp`, `Slot`), the engines being the only
  owners, so with no lease open the next request uploaded them again. `SpeculativeSetup::shared` (made once by `PinSharedDrafter`, called from the `DraftedSession` constructor)
  holds a strong reference for as long as the setup, i.e. the runner, lives. The per-session engine (context rings, buffers) now lives with the session: one allocation per session.
  Check: three requests on one live slot construct exactly one engine and one shared drafter (counting factories, `SpeculativeSetup::shared_factory` replaces the
  device upload in tests). **GPU path lines changed (not compiled here: `hip/drafter_gpu.cpp` is part of `gufo_ricursive_gpuexec`, a device build):** new `PinGpuDrafter` in
  `hip/drafter_gpu.{hpp,cpp}` (returns `Acquire(...)`, the existing function, as a strong `shared_ptr`), and its only caller, `PinSharedDrafter` in `src/runner.cpp`
  (`#if RICURSIVE_HAVE_GPU_EXECUTOR`, device models with no `engine_factory`). Nothing else in `hip/` changed. The night run must check that the second request does not upload
  again (command in STAGE4-GPU.md).

### Item B: a request that ends by its limit ends in plain generation's state

**Superseded:** "How one step flows" point 3 (the correction token stays pending when the step ends), "The last token of a request stays pending" in the runner row, test 5's
"a drafted request ends with its last token pending (`CheckpointPosition = prompt + emitted - 1`)" and disagreement 9 (the next request prefilled the pending token inside a pass)
hold only for a step that did not use its whole budget.
- What gufo's code does (read, not assumed): `StepMultiTokenDecode` passes `remaining = token_limit - published + (preview ? 1 : 0)` (`text_generation_scheduler.cpp` ~1583);
  the preview token is counted in `published` and re-emitted as the step's first token, so a step that returns `remaining` tokens has published exactly `token_limit`, and the next
  statement completes the request by length (`Commit`). The serial plan is the only one this runner declares, so no other call shortens a budget. "Budget used up" therefore means
  "request over", unless a stop sequence or a cancellation ends it first (those go through `Cancel`, which executes nothing). Plain generation advances (evaluates) the last token before
  `Commit`, so its checkpoint is L + E; a stop token is never advanced and all earlier tokens were (checkpoint L + E).
- Fix (`SpeculativeDecoder::Step`): when a step used its whole budget and did not stop, the pending token is evaluated in a one-row pass (`ForwardToken`, which settles first, so the drafter hears the row
  through the sink) and `published()` is cleared. Skipped at the context limit (plain would have stopped at selection).
- Cost: one extra one-row target pass (about one plain decode step) per request that ends by its limit. Wasted when the request ends anyway without a follow-up, when a stop sequence matched
  inside the step (plain would not have evaluated the token; `Cancel` keeps the rows, the checkpoint is longer than plain's) or when the client cancels the step it was in.
- Stop sequences and cancellation (tested): plain's `Cancel` leaves the unadvanced token out, checkpoint L + n - 1. The drafted step had already evaluated every token it accepted, and `generated` of the pool holds all of them,
  so the drafted checkpoint is longer: it extends plain's. A next prompt that continues past plain's checkpoint reuses the live slot as before; a prompt that diverges inside the drafted tail cannot reuse the live frontier and
  starts from a snapshot or cold (speed, not output). Next-request output equals plain in both cases.
- Failing first (`perf/stage4b-B-unfixed.log`): `B (limit 24): the session is at 46, plain generation leaves it at 47`. Fixed (`stage4b-B-fixed.log`): limits 1, 2, 3, 7, 24 (inside a draft), a stop token: position,
  tokens, frontier logits and every checkpoint component byte-equal to plain; the next request with prefix reuse gives plain's output; cancelled and stop-sequence requests extend plain's checkpoint.
- Production-mode gate entries: 4a and the runner's second request (stage 4 skipped them) now compare against plain given the same call sequence (`RunTruthAfter`; both sides evaluate the generated tokens in
  one-row passes, then Sync), and are enabled in production mode; they run here on the exact back ends and are unverified on the production route until the night run. 4c stays exact-mode only, and cannot be otherwise: a drafted
  session recomputes from 0 in prefill passes exactly where plain decoding rewinds (the recompute band above), so the same call sequence takes different routes on the two sides by construction.
  Existing statements checked: test 5's pending-token assertion was replaced (above); nothing else of `speculative_test.cpp` was loosened.

### Item C: a drafter failure costs speed, never the request

**Superseded:** "The drafter's output never changes what is emitted" held for wrong tokens only; an exception inside `Propose` used to fail the request.
- `DrafterHealth` (per session, shared by `DrafterIngest` and `DFlash2DraftBackend`). A throwing `Draft` becomes an empty proposal (the step runs plain), the context is reset to the current position
  (`ResetTo`; a `ResetTo` that throws is counted and swallowed, the desync path then retries on the next step), the failure is counted. After 3 consecutive failures drafting is switched off for the session with one
  warning line on stderr; the ingest then stops feeding rows. A failed `Append`/`TruncateTo`/`ResetTo` in the ingest counts too (it was already absorbed and reset); a proposal that succeeds ends the run
  of failures unless rows failed to arrive since the previous proposal, so a persistent ingest fault also switches drafting off. Failures of the target keep their existing handling.
  `DraftedSession::health()` exposes `failures()`, `consecutive()`, `disabled()`, `last_error()`.
- Failing first (`perf/stage4b-C-unfixed.log`): `C (Draft throws at calls 1 and 3): the request failed: injected Draft failure`. Fixed (`stage4b-C-fixed.log`): an engine wrapper that throws at chosen calls of
  `Draft`, `Append`, and `ResetTo` (5 plans: two isolated Draft failures: 2 counted, still on; every Draft, and from call 2 on: 3 counted, off, not tried again on the next request; three Append failures:
  3 ingest failures, 3 counted, off; one Append failure plus the two following `ResetTo` calls failing: counted, on): every request completes, tokens and end state equal plain's, also for the next request.
- The existing tests now assert zero failures (`ContextLedger` of `speculative_test.cpp` and `spec_util.hpp`, the twin loop, and the runner test): a real drafter bug cannot hide behind the fallback.

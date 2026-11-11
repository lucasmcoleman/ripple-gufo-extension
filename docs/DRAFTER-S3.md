# Drafter stage 3: the target side of speculative decoding (CPU part)

Stage 3 of the DFlash2 drafter design (`/server/staging/ripple-gufo/perf/DRAFTER-GUFO-DESIGN.md` sections 1.2, 1.3, 3 and 6; job 1692).
Branch `drafter-s3` from `drafter-s1` (241e7bf). Written 2026-11-11. **Everything here ran on the CPU executor and the host-ops
back end; nothing ran on a GPU and no real model was loaded.** The GPU gates and the cost test are prepared in
`/server/staging/ripple-gufo/perf/STAGE3-GPU.md`.

The property everything below protects: output drafted with k candidates, whatever number of them is accepted, is **bit for bit**
what plain 1-row decoding produces.

## What the target now offers (src/engine.{hpp,cpp})

Both features are **off by default** (`ModelOptions::verify_rows = 0`, `capture_layers` empty): with them off the executor holds
the last row's logits only, no capture buffer exists, and the engine is the one of `drafter-s1` (proof below). Stage 4 turns them on
(`verify_rows = kMaxVerifyRows`, `capture_layers = {1, 20, 39, 58, 77}`).

| call | meaning |
| --- | --- |
| `ModelOptions::verify_rows` | rows one verification may evaluate, 0 = drafting off, at most `kMaxVerifyRows` (16), clamped to the prefill rows. Becomes `ExecutorConfig::max_logit_rows` (1 when off) |
| `ModelOptions::capture_layers` | `ExecutorConfig::capture_layers` (stage 0), at most 5 layers |
| `Model::MaxVerifyRows()` | the effective cap |
| `Session::Verify(tokens, &error)` | evaluates 1 to `MaxVerifyRows()` tokens at the position p in ONE pass (`LogitRows::kAll`, the write guard first, like every pass). `VerifyLogits()` is `[k][vocab]`: row i = logits after `tokens[i]`. The session then holds p + k tokens and `Logits()` is the last row. Refused (nothing changes) for k outside the range, k above the executor's `batch_invariant_rows` (the exactness guard: 16 on the GPU unless exact), past the context cap, without frontier logits, or when the write guard throws. A failed forward empties the session like any other |
| `Session::AcceptVerified(a, &error)` | keeps the first a of the k tokens (0 <= a <= k): the session is exactly where plain decoding is at p + a (see below). `VerificationOpen()` is true between `Verify` and this call |
| `Session::TakeTaps()` | `TapRows{first_position, rows, data}`: the captured rows of the positions evaluated since the last call (prompt passes, 1-row steps, verification passes), `data` is `[row][tap][hidden]` float32 (the five layers' outputs concatenated: the drafter's feature input). Empty when capture is off |
| `Session::SetFaults(...)` | test-only defects (`rewind_from_end`, `stale_frontier`, `keep_rejected_taps`) |

The rows of a verification are the candidates **including the current token** as row 0: for gufo's verifier the inputs are
`[current_token, d drafts]`, so d = 15 is the 16-row pass. The brief's "k from 1 to 15 candidates plus whatever row the design needs
for the bonus token" and the test's "k from 1 to 15, every a from 0 to k" are covered by one API: k is the number of rows, the tests
run k = 1..16 and every a = 0..k. The bonus/correction token's logits are `VerifyLogits()` row a - 1 (or `Logits()` after
`AcceptVerified(a)`).

Any other call that changes the session (`Evaluate`, `Sync`, `Reset`, a restore) settles an open verification as "all k kept" and
drops `VerifyLogits()`. `TakeTaps()` during an open verification is refused (`std::logic_error`): the rows are not final.

## How the rewind works, per cache (no row is recomputed)

`AcceptVerified(a)` calls `Executor::Rewind(p + a)` (the contract's existing rewind) and sets the host state:

- **Full-attention layers (16)**: slot = position. The rejected rows p + a .. p + k - 1 sit above the new logical end and are
  overwritten by the next pass. Nothing is copied or cleared.
- **Sliding layers (64)**: ring slot = position mod R. The verification wrote the k rows into slots of p .. p + k - 1 and evicted
  positions up to p + k - 1 - R. A rewind to p + a is legal iff the window before it, [WindowFirst(p+a), p+a), is still retained, i.e.
  `p + a - W + 1 >= p + k - R`, i.e. `R >= W + k - a - 1`; the ring has `R >= MinRingRows = W + max_rows - 1` and k <= max_rows, so it
  always holds (R = 3072 against W + 15 = 1039 on the real model). The invariant the executor keeps (the first row's window is retained)
  carries the rest: the position p was legal, p + a >= p. `Rewind` returning false is treated as a bug: the session resets (nothing wrong
  is ever served) and the call fails.
- **Position counters**: the executor's logical end moves to p + a, its high water mark stays p + k (as the contract says, a later
  rewind is judged against the true ring); the session's token list is cut to p + a.
- **Frontier logits**: row a - 1 of the verification, or, for a = 0, the logits the session held before `Verify` (saved for that case).
  Plain decoding at p + a holds exactly these.
- **Anything the continuation adapter tracks** is derived from the session: `Positions()` from `Position()`, the tokens component from
  `tokens()`, the private logits component from `RawLogits()`, the K/V components from `Executor::ExportRows`. Nothing else was added, so
  nothing else needs to rewind. A checkpoint taken right after a rewind is capturable when the ring has slack
  (`R >= W + chunk - 2 + (k - a)`; the real model has 3072 against 1024 + 62 + 15); the test uses a ring with slack for that check.
- **Pending taps**: the rows at positions >= p + a are dropped from the session's pending taps.
- **Write guard / component cache**: the pass reports its plan (`written` [p, p+k), `evicted`) to the guard before it runs, like every
  other pass. The cache only ever sees states between steps.

## Taps through the session

The executor's stage-0 capture (`Executor::Forward(..., capture)`) is now driven by the session: every pass of `Sync` (prompt passes),
`Evaluate` and `Verify` captures when `capture_layers` is set, transposes `[tap][row][hidden]` to `[row][tap][hidden]` and appends to
a pending buffer that covers positions `[first, end)` and always ends at `Position()`. A rewind (a `Sync` that rewinds, an
`AcceptVerified`) trims it; a reset, a failure, a restore clears it (a restored cache has no taps: the drafter is not a cache
component in v1, as the design says). `TakeTaps()` hands over and empties it. `first_position` below what the caller already holds means
rows were rewound and the new ones replace the old from there (the test merges exactly like that).

**Memory**: untaken rows cost hidden x taps x 4 B = 100 KiB per position on the real model. `Sync` is one call, so a 32k-token prompt
accumulates 3.3 GB of host memory in the session before the caller can take anything. The design priced the prefill capture at 210 MB
(the device slots) and did not address this. Stage 4 must either accept it for short prompts, or add a per-pass sink to `Sync`
(a `std::function` called with each pass's rows); I did not add it (YAGNI: nothing consumes it yet, and the sink needs the drafter's
ingestion to define its shape).

## Tests (tests/draft_session_test.cpp, `ricursive.draft_session`; tests/plain_fingerprint_test.cpp, `ricursive.plain_fingerprint`)

CPU executor and host-ops executor, both tiny fixtures (tinyA window 8, tinyB window 16), a synthetic 94-token stream, prefill passes of
20 rows, ring 28 (tinyA) and 36 (tinyB). All comparisons are bitwise on the float bits, plus the arg-max token. Results of the run on the
tip (2026-11-11):

1. **Every k, every a, across the boundaries**: k = 1..16, a = 0..k (152 pairs), at four anchors per fixture chosen so the pass straddles
   W, 2W + 3, the ring size R and 2R. Each scenario: a fresh session, the prompt of p tokens, `Verify` k rows (the first a are the plain
   tokens, the rest deliberately wrong ones), `AcceptVerified(a)`, then 8 plain 1-row steps. Compared with plain 1-row decoding from
   the first token: the first a verification rows, the frontier after the rewind, the 8 steps' logits and tokens, the position, the tokens.
   **608 scenarios per backend and fixture, 2,432 in all, 8,736 logits rows compared per backend and fixture: 0 differences.**
   Also: a model with `verify_rows`/capture armed but unused gives the same logits as one without.
2. **Repeated cycles**: 64 verify/rewind cycles per backend and fixture (fixed seed, random k in 1..16 and a in 0..k, a plain row after a
   rejection-all or at random, restarts through `Sync` to a shorter prefix in 4 or 5 sweeps), every cycle's frontier compared, every
   observed position (241 to 289 per backend and fixture) compared with plain decoding at the end of each sweep, then 8 more plain steps.
   0 differences.
3. **Taps**: the same 64-cycle runs with layers 0, 2, 4 captured: the rows the "caller" holds (merged by `first_position`, 3 or 4
   hand-overs below what it held, i.e. rewinds) equal the taps of plain 1-row decoding at every position (341 to 432 positions per
   backend and fixture), bit for bit; `TakeTaps()` during an open verification is refused.
4. **Twins**, each must fail the check that covers it, and must fail exactly where it should:
   - rewind computed from the logical end (`rewind_from_end`: rewinds to end - a instead of p + a): **288 of 304 scenarios fail** (two
     anchors x 152), exactly the 144 of 152 (k, a) pairs with k != 2a, on all four backend/fixture pairs (it passes only where the two
     formulas coincide). First failure: "position after accepting 0 of 1 is 20, not 19".
   - stale frontier logits (`stale_frontier`): 272 of 304 fail, exactly the 136 pairs with a < k.
   - rejected rows' taps kept (`keep_rejected_taps`): 136 of 152 hand out wrong taps, exactly those with a < k.
   None passed, so no strengthening was needed.
5. **Snapshots and prefix reuse** (through the real `ContinuationAdapter`, a ring with slack): checkpoints at positions 40, 61 and 73 reached by
   three drafted routes each (everything rejected then a plain row; mixed; everything accepted) have **every component byte-equal** to the
   checkpoint after plain decoding; restoring them into a fresh slot and continuing 8 steps gives plain logits (9 restores per backend and
   fixture); a `Sync` to a text sharing the first half after the drafted steps equals the same `Sync` after plain decoding, and 8 further
   steps agree.
6. **Refusals**: empty and 17-row verifications, `Verify` before a prompt, `AcceptVerified` with nothing open or with a > k, a throwing
   write guard (session unchanged), the guard sees exactly `written = [p, p + k)`, `Evaluate`/`Sync` settle, rejecting all restores the
   frontier from before the pass, drafting is off by default (a 2-row `Verify` is refused, `TakeTaps` is empty), `verify_rows` above 16 is
   refused at load.
- **Plain generation unchanged**: `ricursive.plain_fingerprint` hashes every logits row and token of a plain schedule (prompt passes, 1-row
  decoding past the window and the ring wrap, a prefix-reuse sync, a recompute sync, 1-row decoding from the first token) on both backends
  and fixtures. It was written and run **on `drafter-s1` before any engine change** (commit c22b82a, whose `src/` is 241e7bf's); its four
  golden hashes were recorded there and the same file passes on the tip. Besides: the 22 existing tests pass without any edit to their
  sources (`git diff 241e7bf --stat -- tests/` shows only added files and the gate selftest's counts), and the executor contract
  (`src/executor_iface.cpp`, `src/cpu_executor.cpp`, `hip/executor.cpp`) is byte-identical to `drafter-s1`.
- `ctest -R '^ricursive\.' -L cpu`: **24 of 24 pass** (22 before + `plain_fingerprint` + `draft_session`). `draft_session` takes about 13 minutes
  on 4 threads (the whole suite ~17 minutes; it was 3.5 before): the matrix of check 1 is the cost. Its ctest timeout is 3000 s.

## GPU gates and the cost test

- `tests/run-gpu-gates.sh`: `engine_tests` gained four invocations of `ricursive_draft_session_test` with the GPU executor in exact mode
  (`draft-verify-tinyA/B`, `draft-rest-tinyA/B`); `tests/run-gpu-gates-selftest.sh` counts are +4 (64 -> 68 invocations, 65 -> 69 logs,
  31 -> 35 NOT-RUN rows when the lock is refused); the self-test passes.
- `tools/capture_cost.cpp` (`ricursive_capture_cost`, built on request): times one `Forward` of N rows with capture off and on, alternating,
  prints medians and the ratio, exit 0 when on/off <= 1.01. Smoke-tested on tinyB with the CPU backend only (it runs; the CPU's own
  noise was 2.2% at 8 rows, so the 1% figure means something only on the GPU). Commands: `perf/STAGE3-GPU.md`.

## Disagreements with the design, and things it did not say

1. **`max_logit_rows` 1 -> 16 is opt-in** (`verify_rows`, default 0), not a change of the default: brief rule 4 (with drafting unused,
   nothing may change; on the GPU the default would add 6.4 MB of logits). Stage 4 sets it when `--speculative dflash2` is on.
2. **The frontier logits** are part of "exactly the state plain decoding is in at p + a": the design's `SaveState(p)` / `RestoreState()`
   no-op mapping (3.3) says nothing about them. The session saves the pre-verification logits for a = 0.
3. **Tap memory in `Sync`** (above): the design priced device slots only.
4. **Latent bug found in the existing engine, not fixed** (rule 4: plain generation must not change, and the brief did not ask for it):
   `Session::Sync` to a text that shares **no** prefix with a longer session (common = 0) calls `Executor::Rewind(0)`, which always
   succeeds (nothing before position 0 is needed), but the executor keeps its high water mark; the next `Forward` whose first row's window
   starts below `H - R` is then refused ("window before the position is not retained"). Observed on tinyA: `Sync` of 73 tokens, then `Sync`
   of an unrelated 30-token prompt with 12-row passes fails after the first pass, and `Evaluate` after it fails too. The code path is
   unchanged from `drafter-s1` (`executor_iface.cpp` is byte-identical, `SyncImpl`'s rewind lines too); I did not run the `drafter-s1`
   binary itself. On the real model it needs a session holding more than ~4,100 tokens followed by an unrelated prompt (a new chat in the
   same slot) with no common prefix. The existing tests never sync to common = 0 (their divergence points start at 1). A fix would be one
   line in `Executor::Rewind` (position 0 resets the high water mark and floor) plus a test; it needs a decision because it changes plain
   behaviour from an error to success. Stage 4's prefix reuse (`Sync` after drafted steps) is not exposed to it: `AcceptVerified` never
   rewinds below p >= 1.
5. `Session::Verify` refuses passes wider than `batch_invariant_rows` rather than only `max_rows`: on the GPU in production (not exact)
   that is 16, so the design's "cap d + 1 at the largest verified invariant width" mitigation is already enforced in code.
6. Not done: nothing GPU (by rule); `batch_invariant_rows` refusal has no test of its own (no CPU executor has a width below `max_rows`);
   the 1% capture cost and the real-model bit-identity of 16-row passes against 1-row passes remain the GPU gates' business.

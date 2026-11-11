# Executor contract (host side <-> compute backend)

Code: `src/executor_iface.{hpp,cpp}` (the contract, including the shared position bookkeeping),
`src/cpu_executor.{hpp,cpp}` (CPU backend on the scalar `Reference`, F32 KV), tests
`tests/executor_contract_test.cpp` + `tests/executor_contract.hpp` + `tests/executor_backends_cpu.cpp`.
Written 2026-11-10 against gufo 0e2de43e (extension API 0.2.25). CPU only; nothing here ran on a GPU.

Abbreviations: UP = `/server/programming/gufo-ripple/gufo`, G4 = `/server/programming/gufo-ripple/ref-gemma4-ext`.

## 1. Shape

One `Executor` = one session's KV state plus the means to extend it (Gemma 4 splits this into `Executor` +
`KvCache`; here the GPU backend shares weights and scratch between its executors internally). The public
methods are **non-virtual**: the base class owns the logical end, the ring high water, the import floor
and coverage, and validates every call. A backend implements three data paths only:

| hook | must do |
| --- | --- |
| `DoForward(tokens, p0, which, logits, tap)` | write K/V rows of positions [p0, p0+n) in every layer (sliding: slot `p % ring_rows`, full: slot `p`) **before** any query reads; sliding query q reads [WindowFirst(q), q], full reads [0, q]; write nothing else; fill `logits`; return only after the device work finished |
| `DoExport(layer, part, rows, out)` | copy rows in position order, row layout `[kv_head][head_dim]`, element `kv_type` |
| `DoImport(layer, part, rows, bytes)` | the reverse |

So the rules that broke before in Laguna (LESSONS.md "Rewinding caches and handling failures": retention
from the logical end; failure leaving metadata unchanged after rows were written) exist once, for both
backends, and are covered by the twins below.

## 2. Methods and why each exists

| method | meaning | motivated by |
| --- | --- | --- |
| `Info()` | layers, window, sliding pattern, kv_width, vocab, max_context, max_rows, max_logit_rows, ring_rows, kv_type, batch_invariant_rows, `RowBytes()`, `WindowFirst(p)` | component descriptors need `row_bytes`, `kWindowRows`, `window_rows` (UP src/cache/types.hpp:34, :45; G4 continuation_adapter.cpp:327-345); runner `max_context` (UP src/ext/text.hpp:181); `max_rows`/`max_logit_rows` size the forward (G4 hip/executor.hpp:93-96, engine.cpp:313-316); ring (G4 continuation_adapter.cpp:314-316, hip/executor.cpp:187-190) |
| `State()` | kReady / kImporting / kFailed | slot validity and latched failure (UP src/cache/adapter.hpp:108 "refuse if failure is latched", :170 Validate latches); G4 `Session::valid_` (engine.cpp:434, :555) |
| `Position()` | valid tokens; next Forward starts here | `cache::Adapter::Positions` / `ReconcileComponentState(state, position)` (UP adapter.hpp:89; text.hpp:560); G4 continuation_adapter.cpp:481-490 |
| `RetainedBegin()` | first sliding row still exportable / rewindable = max(import floor, high water - ring) | `CopyRowsOut` may only read retained ring rows (G4 continuation_adapter.cpp:687-690 `occupant` check); LESSONS.md:205-213 |
| `PlanForward(rows)` | `written` and `evicted` ranges a Forward will overwrite | `MutationGuard::BeforeOverwrite` before any overwrite incl. ring eviction (UP adapter.hpp:28, :81-83); G4 continuation_adapter.cpp:220-242 (`first - ring .. end - ring`), executor.cpp:794 |
| `Forward(tokens, p0, which, logits, tap)` | evaluate a chunk at consecutive positions; logits of no row / last row / all rows | G4 hip/executor.hpp:118-128, engine.cpp:526 (chunks), :535 `ReadLogits`; host sampling `SamplerState::Sample(span<const float>)` (UP src/core/sampling.hpp:205) via `SelectNext` (text.hpp:614); all-row logits only for tests |
| `Rewind(p)` | lower Position() if the window before p is retained, else false; `Rewind(0)` is false too while `RetainedBegin() > 0` (high water above the ring, or an import floor): the engine resets and recomputes | prefix reuse in `Prefill(state, prompt, offset, ...)` (text.hpp:590-592), `CanContinue` (text.hpp:568); G4 engine.cpp:694-699 |
| `Reset()` | empty, ready, from any state | `Adapter::Invalidate` (UP adapter.hpp:176); G4 continuation_adapter.cpp:892-918, engine.cpp:399 |
| `ExportRows(layer, part, rows, out)` | rows as bytes, absolute positions | `CopyRowsOut` (UP adapter.hpp:122); G4 continuation_adapter.cpp:673-705 |
| `BeginImport(n)` | start a restore at frontier n | `BeginRestore` (UP adapter.hpp:108); G4 continuation_adapter.cpp:622-671 |
| `ImportRows(layer, part, rows, bytes)` | load rows | `CopyRowsIn` (UP adapter.hpp:137); G4 :706-747 |
| `CommitImport()` | check coverage, become ready; false latches kFailed | `Validate` (UP adapter.hpp:170); G4 :844-891 |
| `Tap` argument | per-stage host taps (Reference names); `l_out` per layer required | G4 hip/executor.hpp:202-207 (`TapSink`); RIC src/reference.hpp `TapFn`; also the failure-injection point |

Deliberately absent (no API fact forces them): async transfers / `Completion` (the adapter wraps the
synchronous calls; G4 does host memcpy for its token component the same way), `BorrowRowsView` (optional,
UP adapter.hpp:131, returning empty is allowed), batched multi-session forward, hidden-state / MTP state,
images, GPU sampling, lookahead rows, in-pass checkpoints.

## 3. State machine and error model

```
            Forward/Rewind/Export/PlanForward            ImportRows
  kReady  ------------------------------------> kReady    kImporting --------> kImporting
    |  ^                                                      |    CommitImport true -> kReady
    |  | Reset (any state)                                    |    CommitImport false -> kFailed
    |  +------------------------------------------------------+
    | exception from DoForward (incl. tap)          BeginImport(n) from any state -> kImporting
    v
  kFailed: Position() = RetainedBegin() = 0; only Reset and BeginImport accepted
```

- **Refusal** = `std::invalid_argument` (bad argument: position != Position(), rows > max_rows, token id out
  of range, wrong buffer size, rows outside the retained range, ...) or `std::logic_error` (wrong state).
  Thrown before anything changes: state, positions and every row are as before.
- **Failure** = anything thrown by `DoForward`/`DoImport` (device error, allocation, a throwing tap). The
  executor is kFailed until `Reset()` or a restore (`BeginImport` + `CommitImport`). Rows are undefined;
  we do not try to keep them ("unchanged"), because a GPU forward that failed after layer k has already
  overwritten ring slots in layers 0..k (LESSONS.md:214-222). The caller reads `State()` after catching.
- `Rewind` returning false is not an error: nothing changed; the engine resets and recomputes (or lets the
  cache restore a shallower checkpoint).
- Every call is synchronous (the GPU backend synchronizes its stream before returning), so a device error
  surfaces inside the call that caused it, never against a state the ledger already committed.
- Calls on one executor, and on executors sharing scratch, are serialized by the caller (G4 engine.cpp:436
  holds the model mutex).

## 4. Ring arithmetic

Notation: W = window (keys a sliding query sees, itself included; `ricursive.attention.sliding_window`),
M = max_rows, R = ring_rows, H = high water (largest position ever written + 1, **not** Position()),
F = import floor (0 unless restored), E = Position(). `WindowFirst(p) = p >= W ? p - W + 1 : 0`.

- **Minimum ring:** R >= W + M - 1 (`MinRingRows`). A Forward writes all its rows first; its first row p
  reads back to p - W + 1, its last write (p + M - 1) evicts p + M - 1 - R, which must be below p - W + 1.
  The constructor refuses a smaller ring. The component cache adds R >= window_rows + rows_per_chunk - 1
  with window_rows = W - 1 (UP adapter.hpp:81); the adapter picks R >= both and may round it up.
- **Retained** sliding rows: [RetainedBegin, E) with RetainedBegin = max(F, H > R ? H - R : 0). Full layers
  keep [0, E). Rows in [E, H) belong to a rewound branch: not exportable, overwritten by the next Forward.
- **Rewind to p** (p <= E) succeeds iff [WindowFirst(p), p) is retained, i.e. WindowFirst(p) >= RetainedBegin
  (or the range is empty). It changes only E. H stays, so a second rewind is judged against the true ring.
- **PlanForward(n)**: written = [E, E + n); evicted = [RB, RB') with RB' = max(F, max(H, E + n) - R).
  After the Forward, RetainedBegin == RB'. The adapter guards `written` in every K/V component and
  `evicted` in the sliding ones.
- **Import at n**: sliding rows below n - R cannot be placed (the ring would alias them) and are refused.
  CommitImport needs every full layer covered over [0, n) and every sliding layer over [WindowFirst(n), n);
  F becomes the highest contiguous coverage start over all sliding layers and parts. After an import with
  the cache's chunk-aligned ranges (from `FirstChunkRow`), F <= WindowFirst(n).

### Worked example, W = 8 (tinyA), M = 12 -> R = 19
- Prefill 40 tokens in chunks 12, 12, 12, 4: H = 40, retained [21, 40).
- `PlanForward(12)` at 40: written [40, 52), evicted [21, 33); afterwards retained [33, 52).
- Instead rewind to 38: needs [31, 38), OK. Then rewind to 28: needs [21, 28), OK (lowest legal).
  Rewind to 27 instead: needs row 20, which position 39 overwrote (slot 20 % 19 = 1 = 39 % 19): refused.
  A rule computed from the logical end (38 - 19 = 19) would accept 26 and 27 and silently read row 20.
  The test measures that twin at 5.47 max abs logit error.
- Checkpoint at 40 with window_rows 7 and rows_per_chunk 4: `WindowFirstRow` = 33, `FirstChunkRow` = 32;
  the adapter exports sliding rows [32, 40), full rows [0, 40). Requires R >= 7 + 4 - 1 = 10 (19 holds).

### Worked example, W = 1024 (Ripple-65B), prefill chunk 2048 + 16-row tail merge -> M = 2064
- R_min = 1024 + 2064 - 1 = 3087; component cache with rows_per_chunk 64 needs >= 1023 + 63 = 1086;
  rounded to 64: R = 3136. (Gemma rounds W + M to 256: 3328.)
- Prompt of 5000 tokens, Gemma-style plan: passes end at 2048, 4096, 5000. Retained [1864, 5000).
  Lowest legal rewind: 1864 + 1023 = 2887.
- Checkpoint at 5000: `WindowFirstRow` = 3977, `FirstChunkRow` = 3968: sliding rows [3968, 5000).
- One decode token at 5000: written [5000, 5001), evicted [1864, 1865).
- Memory at F16 (8 KV heads x 128): one K or V row 2 KiB; sliding layers 3136 x 4 KiB x 64 layers = 784 MiB,
  independent of context; full layers 64 KiB per token (16 layers), 2 GiB at a 32768 context.

## 5. Call sequences (engine / adapter)

- **Prefill** (G4 engine.cpp:422-563 without lookahead/images): for each pass end from the adapter's
  `PlanPrefill` (both must use the same chunking): `plan = PlanForward(n)`; `guard.BeforeOverwrite(id,
  plan.written)` for every K/V component and `(id, plan.evicted)` for sliding ones; `Forward(chunk,
  start, last ? kLast : kNone, logits)`. Keep the last row's logits on the host (private component).
- **Decode**: the same with n = 1 and kLast; `SelectNext` samples the host logits.
- **Checkpoint** (`CopyRowsOut`): `ExportRows(layer, part, {first, end}, bytes)`; refuses rows below
  RetainedBegin. Tokens, logits and metadata are the adapter's own host components.
- **Restore**: `BeginRestore` guards [0, context) then `BeginImport(n)`; each `CopyRowsIn` ->
  `ImportRows`; `Validate` -> adapter coverage checks and `CommitImport()` (false -> Validate false);
  `ReconcileComponentState` checks `Position() == n`. `Invalidate` -> `Reset()`.
- **Rewind / prefix reuse**: common prefix c (at most prompt size - 1, so the last prompt token is
  evaluated again for its logits, G4 engine.cpp:689); `Rewind(c)`; on false `Reset()` and prefill from 0.
- **Failure**: catch, read `State()`; kFailed -> mark the session invalid (its tokens/logits too), then
  `Reset()` or restore. kReady after a throw means the call was refused and nothing changed.

Notes for the engine/adapter worker:
- Do not copy G4's rewind rule `tokens_.size() - common + sliding_window > ring` (engine.cpp:697): it
  measures from the logical end, the pattern of LESSONS.md:205-213. By reading (not run), after a re-prefill
  that ends below the old high water it can accept a rewind whose keys were overwritten. Use `Rewind()`.
- After a rewind, RetainedBegin can exceed `FirstChunkRow(Position())`, so a checkpoint right there is
  refused by `ExportRows`; treat it as not capturable (or refuse such rewinds).
- Cache identity (UP docs/extension-api.md:60-63) must include kv_type, `exact`, max_rows (pass widths
  change rounding; G4 continuation_adapter.cpp:365-366), ring_rows and max_context.
- `batch_invariant_rows` tells the engine which passes are bit-identical to single-token decode; on the GPU
  in production a 2048-row pass is not, so `incremental_text_is_exact` (text.hpp:161) must stay false unless
  `exact` is set.

## 6. Precision modes

| mode | kv_type | exact | batch_invariant_rows | judged against |
| --- | --- | --- | --- | --- |
| production GPU | F16 | false | 16 (W8A8 prefill above) | llama.cpp `--kv-f16`, 2e-2 max abs, top-1 100% (SPEC S2) |
| exact GPU test mode | F32 | true | max_rows | llama.cpp F32 KV (`*.llama.f32`), 2e-3, top-1 100% |
| CPU backend | F32 only | ignored | max_rows | the same; measured 2.1e-5 / 1.1e-5 |

## 7. Contract tests

`tests/executor_contract_test.cpp` runs, for every backend returned by `ContractBackends()` and both tiny
F32 fixtures (tinyA W = 8, 10 layers; tinyB W = 16, 5 layers; 64-token sequences; max_rows = W + 4, so
R = 19 / 35 and the ring wraps), eight checks. Each runs on the backend (must pass) and on a broken twin
(must fail; numeric twins must also move the result by >= 1e-3). Comparisons are exact (0) when every
Forward involved had at most `batch_invariant_rows` rows, else `rounding_tol` (backend parameter).

| check | what it asserts | twin(s) |
| --- | --- | --- |
| 1 chunked == one-shot | plans {1}, {3}, {M}, {W-3, 6, M}, {1, W+1, W-1} equal one 64-row Forward, all rows | RoPE positions restart per chunk; ring stores one slot fewer than declared |
| 2 decode == prefill | prefill W+3 then 1-row kLast decodes equal the 64-row run | RoPE per chunk |
| 3 vs llama.cpp | 4 sequences, max abs <= `llama_max_abs`, top-1 100% | window one key short |
| 4 restore | at n in {W-1, W, W+1, 2W+3, 40}: export the rows a kWindowRows adapter loads (`gufo::cache::FirstChunkRow`, chunk 4), import into a fresh executor, commit; exported bytes equal before and after continuing 64-n tokens; logits equal | import skips a sliding layer (commit refused); import starts one row above WindowFirst(n) (refused); no coverage check + skipped layer (silent corruption, measured) |
| 5 rewinds | at H = 40: rewind to 38 then to the lowest legal position, continue with other tokens == fresh run; separately 38 then one below the legal position: refused, position and bytes unchanged, continuation == fresh; beyond Position: invalid_argument | retention from the logical end |
| 6 failure | tap throws at layer L/2 inside a Forward at 30: exception propagates, kFailed, Position 0; Forward, Rewind, Export, PlanForward, ImportRows, CommitImport refused with logic_error; restore by import == uninterrupted; Reset + prefill == uninterrupted | failed Forward stays ready (then: rewind to 20 and continue, measured) |
| 7 write plans | for 10 mixed-size Forwards with a rewind: `written` exact, `evicted` starts at RetainedBegin, RetainedBegin afterwards == evicted.end, every retained row outside the plan byte-identical | ring one slot short |
| 8 taps and rows | every emitted tap equals `Reference::ForwardFull`'s (same name/layer), `l_out` for every layer present; exported K/V rows (F16 decoded) equal Reference `Kcur`/`Vcur` | compared one layer off |

Results, CPU backend (2026-11-10, `ricursive_executor_contract_test`, 25 s, deterministic over two runs;
log `/server/staging/ripple-gufo/executor-contract-run2.log`). Correct runs: max abs logit difference.
Twins: the error they produce (max abs logits; for check 7 the number of corrupted retained rows; 0 for
the refusal twins, which fail by refusing).

| line | tinyA | tinyB | verdict |
| --- | --- | --- | --- |
| 1 chunked == one-shot | 0 | 0 | pass |
| 1 twin: RoPE per chunk | 8.42 | 6.26 | fails |
| 1 twin: ring one slot short | 8.72 | 3.11 | fails |
| 2 decode == prefill | 0 | 0 | pass |
| 2 twin: RoPE per chunk | 8.02 | 5.30 | fails |
| 3 vs llama.cpp (max abs) | 2.12e-5 | 1.12e-5 | pass (top-1 100%) |
| 3 twin: window one key short | 7.38 | 3.25 | fails (top-1 down to 25% / 69%) |
| 4 restore at W-1, W, W+1, 2W+3, 40 | 0 (all 5) | 0 (all 5) | pass |
| 4 twin: import skips a layer | refused | refused | fails |
| 4 twin: import one window row short | refused | refused | fails |
| 4 twin: no coverage check + skipped layer | 6.30 | 5.48 | fails |
| 5 rewinds | 0 | 0 | pass |
| 5 twin: retention from logical end | 5.47 | 2.14 | fails |
| 6 failure, refusal, recovery | 0 | 0 | pass |
| 6 twin: failed forward stays ready | 10.7 | 6.95 | fails |
| 7 write plans, untouched rows | 0 rows | 0 rows | pass |
| 7 twin: ring one slot short | 96 rows | 24 rows | fails |
| 8 taps and K/V rows (193 / 98 sets) | 0 | 0 | pass |
| 8 twin: one layer off | 9.89 | 17.1 | fails |

On the CPU, checks 1, 2 and 8 are bitwise by construction (the backend runs the Reference's own layer
code); their value is as the harness for the GPU backend. The independent check is 3 (llama.cpp).

### Plugging in the GPU backend (no edit to the test logic)
Add `tests/executor_backends_gpu.cpp` defining `ContractBackends()` with entries for the production mode
(F16, `rounding_tol` measured, `llama_suffix` of the `--kv-f16` logits, `llama_max_abs` 2e-2) and the exact
mode (F32, `exact`, `llama_max_abs` 2e-3), `make_twin` empty (twins prove the harness, done on the CPU). Add a
target `ricursive_executor_contract_gpu_test` (sources `tests/executor_contract_test.cpp` + that file,
labels `gpu`, `SKIP_RETURN_CODE 77`, `make` returning a skip when no device) and run it through the GPU lock.

## 8. Open questions for the GPU worker (confirm or report)
1. `batch_invariant_rows` = 16 in production: are 2..16-row passes bit-equal to 1-row passes end to end
   (projections are by the Gemma tests, docs/S2-PROJECT.md; attention, norms and the gate kernels must be
   too)? If not, report the true value; checks 1, 2 then use `rounding_tol`.
2. Attention must read keys only after all rows of the pass are written (or prove the per-row order gives
   the same ring requirement); otherwise R_min changes.
3. Synchronous calls: a stream sync at the end of each Forward/Export/Import. Acceptable for correctness;
   say if it costs more than a few percent of prefill.
4. Failure semantics on a sticky HIP error: kFailed + Reset is not enough if the device context is lost;
   `DeviceUsable()`/`PollDevice()` (text.hpp:627-630) must then report it. Not modelled here.
5. llama.cpp `--kv-f16` logits for tinyA/tinyB do not exist in `oracle/out/` yet (only F32 KV); the F16
   mode needs them (SPEC S2) and a measured `rounding_tol`.
6. Taps cost a device-to-host copy each; emit them only while a tap is set, and at least `l_out` per layer.
7. Exported row layout `[kv_head][head_dim]`, binary16 for F16 (check 8 decodes it); K after q/k-norm and
   RoPE, V raw.

## 9. Changes needed in files this work does not own
- `src/reference.hpp`: add `friend class CpuExecutor;` to `Reference`. Until then `cpu_executor.cpp` takes the
  private member addresses (`Embed`, `Layer`, `Head`, `AttendRow`, `EffectiveWindow`) through explicit
  template instantiation, which the standard exempts from access checks. It is legal and compiles warning-free
  with g++ 15, but it is a workaround; with the friend line the `Expose` block (about 40 lines) goes away.
- `src/text/**`, `hip/project.*`, `hip/embed.*`: no change needed; they built inside the extension unchanged.

## 10. Not verified
- No GPU run of anything: the GPU tests (`ricursive.embed_gpu`, `ricursive.projection_ops.*`) are
  registered (label `gpu`, skip code 77) but were not run.
- `hip/project.cpp`, `hip/embed.cpp` compiled as plain C++ (g++), as Gemma compiles its executor; the
  standalone S2 build compiled them as HIP. Built and linked, not executed on a device.
- The GPU-side claims in sections 6 and 8 are requirements, not measurements.
- The note on G4's rewind rule (section 5) is from reading; no G4 run.

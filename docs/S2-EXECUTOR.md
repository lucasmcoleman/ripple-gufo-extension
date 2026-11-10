# S2 work package C: GPU executor (orchestration, host ops back end, GPU back end, gates)

Branch `s2-exec`. Written 2026-11-10 against gufo 0e2de43e (extension API 0.2.25), correctness first, no speed tuning.
**Nothing here ran on a GPU.** The GPU lock was refused all day (exit 75); the GPU path is compiled, linked and
desk-checked, and everything about its orchestration was run on the CPU through the host ops. Section 8 says exactly
what ran where.

## 1. Shape: one orchestration, two op back ends

```
hip/ops.hpp            Ops interface (HIP-free): buffers, 12 operators; WeightView, ModelHandles; binary16 helpers
hip/executor.{hpp,cpp} OpsExecutor: the forward pass, written ONCE against Ops (HIP-free, ~420 lines)
hip/ops_host.cpp       HostOps + HostOpsExecutor: host memory, scalar loops, guarded buffers (CPU test back end)
hip/ops_gpu.cpp        GpuOps + GpuExecutor: hipMalloc, the real launch functions, one stream and scratch per executor
hip/device_model.*     DeviceModel: weights uploaded through the public WeightUpload / WeightAllocator (Gemma's Uploader)
```

`OpsExecutor` is an `Executor` (src/executor_iface.hpp): it implements `DoForward`, `DoExport`, `DoImport` and nothing
else; the ledger (positions, retained rows, import coverage, failure latch) is the contract's own. Pointers handed to
`Ops` are backend memory; the orchestration does the offset arithmetic, so buffer sizes, offsets, ring slots, row counts,
call order and scratch reuse are exercised by whichever back end runs.

`HostOps` is not the `Reference`. It is written against the operator signatures with the kernels' conventions (guards
replicated: head_dim 128, group 1/2/10, `rotary_dim` 0|128, `capacity >= tokens`, `capacity >= MinCapacity`,
`x != branch`, `clamp` finite and > 0; ring/window/GQA/RoPE index math from `hip/kernels_math.hpp`; projections are a
double-accumulated dot product over `DequantizeRow` rows; K/V stored as binary16 (RNE, own `FloatToHalf`, checked against
`_Float16` on 17 M floats) or float exactly as the device stores them). It also bites: every buffer is its own aligned
allocation with 64-byte guards and NaN-filled on allocation; every operator checks that each pointer argument covers
the bytes it will touch inside one live buffer (the weights, which are the mapped GGUF, are exempt); a stray write is
caught at the next `Synchronize`. Projections ignore `exact` (the host never rounds activations).

### Forward order (per layer, exactly SPEC)
`RmsNorm(h, attn_norm) -> a`; Q, K, gate projected from `a`; `ValueMix(x0, a, lambda) -> vin` (its own buffer, x0 kept for
the whole pass), V projected from `vin`; `QkNormRopeAppend` (rotary 128 on sliding layers, 0 on full layers; all rows
appended before any attends); `Attention` (window ring on sliding layers, full otherwise, scale `1/sqrt(128)`);
`SigmoidGate(o, gate)`; output projection; `ResidualPostNorm(h, post_attention_norm)`; `RmsNorm(ffn_norm)`; gate/up
projections; `ClampedSwiGlu(clamp[il])`; down projection; `ResidualPostNorm(h, post_ffw_norm)`. Embedding: `EmbedRows`
(scale 1) -> `ScaleRows(embedding_scale)` -> `x0` copy. Head: `GatherRows` (last row only) -> `RmsNorm(output_norm)` on the
requested rows -> `Project(output)` -> read back. Taps (same names as `Reference`): `embd_scaled, attn_norm, vres_in,
Qcur_normed, Kcur_normed, Qcur, Kcur, Vcur, attn_out, attn_gate, attn_gated, attn_post_norm, ffn_inp, ffn_norm,
ffn_swiglu, ffn_out, ffn_post_norm, l_out, result_norm, result_output`. The fused kernels hide some of them, so tap mode
recomputes them on copies (`RmsNorm` on q/k copies, `SigmoidGate` with ones, `ResidualPostNorm` into zeros, `Kcur` read back
from the cache); the production path (no tap) does none of that.

Supported: F16 KV (production) and F32 KV (exact test mode); `ExecutorConfig::exact` -> `Project(..., exact)`; logits for
none / last / all rows; F32 weights (tiny fixtures; per-row GEMV on the GPU). Refused up front
(`std::invalid_argument`, "not supported"): `head_dim != 128`, GQA group not 1/2/10, any layer with `swiglu_clamp` 0 or not
finite (the real model has 20 everywhere; `ClampedSwiGlu` has no no-clamp mode). `max_context == 0` means
`min(kDefaultContextCap = 32768, GGUF context)`, never the GGUF's 2,097,152.

Memory: one scratch set per executor (never shared between streams), caches allocated at construction (sliding layers
`ring_rows`, full layers `max_context`); the Q8_1 projection scratch is reserved up front. A failed `DoForward` first
synchronizes the stream, then rethrows (the base latches kFailed).

### Orchestration twins (`ExecFaults`, tests only)
`x0_before_scale`, `value_mix_into_qk`, `gate_from_residual`, `rope_on_full`, `ring_shortfall`, `swap_post_norms`,
`wrong_logit_row`, `stale_scratch` (value-mix buffer computed in layer 0 only), plus the contract's `chunk_local_rope`,
`window_shortfall` and the three ledger faults.

## 2. Files

| file | lines | role |
| --- | --- | --- |
| `hip/ops.hpp` | 204 | Ops interface, WeightView, ModelHandles, binary16 |
| `hip/executor.hpp`, `hip/executor.cpp` | 139 + 404 | orchestration, `NormalizeConfig`, `ExecFaults`, `HostOpsExecutor`/`GpuExecutor` declarations |
| `hip/ops_host.cpp` | 529 | host ops, host model binding |
| `hip/ops_gpu.cpp`, `hip/device_model.{hpp,cpp}` | 161 + 48 + 150 | GPU ops, weight upload |
| `tests/executor_test.cpp`, `tests/executor_test_backend.hpp` | 629 + 47 | executor test (header: new, shared by the two back-end files) |
| `tests/executor_backends_host.cpp`, `tests/executor_backends_gpu.cpp` | 74 + 124 | contract backends + executor test backend |
| `tests/run-gpu-gates.sh`, `tests/run-gpu-gates-selftest.sh` | 187 + 94 | the gate script and its self-test (selftest: new) |

## 3. Build and ctest additions (exact)

`extension.cmake`: one block `# --- WP-C executor begin/end ---`, inserted just before `# Tool, built on request:`. It adds
targets `gufo_ricursive_exec` (hip/executor.cpp), `gufo_ricursive_hostops` (hip/ops_host.cpp, OpenMP, -O2),
`gufo_ricursive_gpuexec` (hip/device_model.cpp, hip/ops_gpu.cpp and, as a HIP source with
`-O3 -fno-fast-math -ffp-contract=off -Wno-unused-command-line-argument`, hip/kernels.hip.cpp), the function
`_ricursive_hip_test` (kernel_ops and attention_ops need HIP language mode for `tests/check.hpp`'s device helpers), and
tests: `ricursive.kernels_math` (cpu), `ricursive.executor_contract_host` and `ricursive.executor_host` (cpu),
`ricursive.kernel_ops`, `ricursive.attention_ops.{small,real-sliding-f32,real-sliding-f16,real-full-f32,real-full-f16}`,
`ricursive.executor_contract_gpu`, `ricursive.executor_gpu.<group:fixture>` (17) (all `gpu`, `SKIP_RETURN_CODE 77`,
`RESOURCE_LOCK ricursive_gpu`; they must only be run through the admission lock).
`extension.json`: entries appended after `ricursive_projection_ops_test` in `targets`: `gufo_ricursive_exec`,
`gufo_ricursive_hostops` (testing), `ricursive_executor_contract_host_test`, `ricursive_executor_host_test`,
`ricursive_kernels_math_test` (testing, test), `gufo_ricursive_gpuexec` (hip), `ricursive_kernel_ops_test`,
`ricursive_attention_ops_test`, `ricursive_executor_contract_gpu_test`, `ricursive_executor_gpu_test` (hip, testing, test).
Merge note: `kernels.hip.cpp` lives in `gufo_ricursive_gpuexec`, not in the existing `gufo_ricursive_hip`, so no existing
line was touched; it can move there at merge.

## 4. CPU results (host ops), 2026-11-10

Logs: `/server/staging/ripple-gufo/s2-exec/{contract-host-final,executor-host-final,ctest-cpu}.log`. Build:
`source env.sh; nice -n 10 ninja -j 2 ricursive_executor_host_test ricursive_executor_contract_host_test
ricursive_kernels_math_test` in `build-test` (no compiler warnings). ctest `-R ricursive.(executor_host|
executor_contract_host|kernels_math)`: 3/3 pass, 266 s on a loaded host (OMP_NUM_THREADS 4).

**Contract suite on the host ops** (`ricursive_executor_contract_host_test`, `tests/executor_contract_test.cpp` unedited;
backends `hostops` F32 KV with all twins, `hostops-f16` F16 KV without twins). Correct runs: checks 1, 2, 4 (5 restore points),
5, 6, 7 all 0 difference (bitwise) on tinyA and tinyB, in both KV modes. Check 3 vs llama.cpp: F32 KV 1.7e-5 (tinyA) / 1.1e-5
(tinyB), top-1 100%; F16 KV vs `--kv-f16` 3.3e-3 / 9.1e-4, top-1 100%. Check 8 (taps and K/V rows vs `Reference`): F32 2.1e-5 /
1.9e-5 over 193 / 98 tap and row sets; F16 1.3e-2 / 5.9e-3. Twins (max abs logit error, or rows corrupted; all fail as
required): RoPE per chunk 15.1 / 11.5 (check 1), 13.4 / 9.6 (2); ring one slot short 8.7 / 3.1 (1), 192 / 24 rows (7);
window one key short 7.4 / 3.3 (3); import skips a layer, one window row short: refused; no coverage check 6.3 / 5.5;
retention from the logical end 5.5 / 2.1; failed forward stays ready 10.7 / 7.0; compared one layer off 9.9 / 17.1.

**Executor test on the host ops** (`ricursive_executor_host_test`, 19 invocations, all PASS):
- `taps`: all 20 tap names of every layer against `Reference::ForwardFull` (F32 KV, 64 rows, one Forward): 193 / 88 taps, first
  stage with any difference `attn_norm` layer 0 (rel 1.5e-7 / 8.2e-8), worst stage `ffn_swiglu` layer 5 / layer 4 (rel 3.9e-6 /
  2.2e-6, abs 1.4e-5 / 1.9e-5); tolerance rel 1e-4 (25x above the worst, five orders below any twin).
- `llama-f32`: max abs 1.7e-5 (tinyA, seqs 1.4-1.7e-5) / 1.1e-5 (tinyB), mean KL <= 3.8e-12, top-1 100% on all 8 sequences (limit 2e-3).
- `llama-f16`: max abs 3.3e-3 / 9.1e-4, mean KL <= 1.0e-7, top-1 100% (limit 2e-2).
- `chunk`: five chunkings vs one-shot, prefill + single-token decodes, kLast/kNone mixes: 0 in F32 exact and in F16 production.
- `rewind`: rewind then different tokens == fresh run: 0 (several highs and distances; a rewind outside the ring is refused and changes nothing).
- `quant` (tinyB Q4_K_M, Q5_K_M, Q6_K, Q8_0; exact mode, F32 KV, vs `Reference` on the same file): max abs 7.5e-6 .. 8.6e-6,
  mean KL 1.2e-12 .. 1.4e-12, top-1 100% (limit 1e-4): close to float rounding, as predicted. Production mode (F16 KV) vs
  `Reference`: max abs 2.9e-3 .. 3.4e-3, mean KL 2e-7 .. 2.4e-7, top-1 100%. Exact mode vs llama.cpp on the same file (report
  only): max abs 0.18-0.29, mean KL 7.6e-4 .. 1.6e-3, top-1 92-100% per sequence (the same gap the S1 `Reference` has: it is
  llama.cpp's activation quantization, not the executor). Batch invariance in production mode with quantized weights: 16-row
  passes vs 1-row passes 0 on all four files (on the host ops this is by construction; it is the check to run on the GPU).
- `twins` (orchestration level; margin = twin max abs error / the 2e-3 threshold; control runs 1.7e-5 / 1.1e-5; each must be >= 10x):
  x0 before the scale 7.6 / 4.4 (3.8e3x / 2.2e3x); value mix into Q/K 10.5 / 6.3; gate from the residual 5.5 / 3.4; RoPE on full
  layers 3.9 / 2.5 (**weakest: 1.97e3x / 1.24e3x**); post-norm weights swapped 8.6 / 5.0; logits from the wrong row 12 / 13.9;
  stale value-mix buffer 7.7 / 3.8; ring capacity one row short: refused by the kernels' `MinCapacity` guard (replicated in the
  host ops), and with the guard off 7.6 / 3.1.
- `limits` (clamp 0 refused: "layer 2 has swiglu_clamp 0 ... not supported"; context cap 0 -> 32768 for a 2,097,152 GGUF;
  too many rows / bad token id refused with the executor still ready), `half` (binary16 helpers vs `_Float16`, 17 M floats and
  all 65,536 halves), `guards` (the host ops refuse an oversized operator, an aliased `ResidualPostNorm`, a pointer outside
  every buffer, and report a stray write past a buffer).
Not caught by the host ops by design: anything about rounding order on a device, W8A8, stream ordering, device errors.

## 5. Answers to the contract's open questions (docs/EXECUTOR-CONTRACT.md section 8)

Line numbers are in `hip/kernels.hip.cpp` (KC), `hip/kernels.hpp` (KH), `hip/kernels_math.hpp` (KM), `hip/project.cpp` (PC).
All four are answered by reading; none is measured on a GPU.

1. **`batch_invariant_rows` = 16: expected true, not measured.** Norms are one block per row with a fixed summation order
   independent of the row count (RmsNormKernel KC:119-127, ResidualPostNormKernel KC:130-140, `BlockSumSquares` KC:61-81);
   elementwise kernels are per element (KC:142-181); q/k norm, RoPE and append are one block per (row, head)
   (KC:202-256); attention's key tiles start at the first visible key, so a row's sum order depends only on its position and
   the window (KM:46-57, KH:100-101, KH:9-10). Projections: row 1 is the GEMV, rows 2..16 the small-batch kernel, "bit-equal
   to the GEMV per row" (`hip/project.hpp:61-63`, PC:128-131, projection test in docs/S2-PROJECT.md), `exact` forces that
   path at any width (PC:113-121), F32/BF16 weights are per-row GEMV at any width (PC:106-112); only rows > 16 take the
   rounding W8A8 GEMM (PC:122-127). So 16 holds if the projection tests pass on the GPU. Executor-level check, to run: `quant:tinyB-*` compares
   16-row passes with 1-row passes (production mode, quantized weights; must be 0) and reports 20-row passes (W8A8).
   `ExecutorInfo::batch_invariant_rows` is `kSplitRows` (16) unless `exact`, then `max_rows`. If the check shows a difference
   at <= 16 rows, the contract's checks 1 and 2 need `rounding_tol` for the production entry.
2. **Keys are read only after all rows of the pass are written; `R_min` is unchanged.** Each layer is one
   `QkNormRopeAppend` launch writing all n rows (KC:202-256) followed by one `Attention128` launch on the same stream, so the
   append finishes before any query reads (stream order); the header demands exactly this (KH:90-91) and `Attention128`
   refuses `capacity < MinCapacity = window + tokens - 1` (KC:552-553, KM:68-77). The executor passes `ring_rows >=
   MinRingRows(window, max_rows)` (the contract's constructor guard) so the guard never trips; `ring_shortfall` shows it
   would (host twin "ring capacity one row short": refused by the guard; with the guard off: 7.6 / 3.1 max abs logits).
3. **Synchronous calls.** One `hipStreamSynchronize` at the end of each Forward (plus the logits copy, which has to wait
   anyway) is one sync per pass: for a 2048-row prefill chunk that is negligible, for decode it is the sync the sampler needs
   anyway. `ExportRows`/`ImportRows` sync once per contiguous run (<= 2 per layer part). Taps add copies only while a tap is
   set. Not measured; if it ever matters the place is `OpsExecutor::Run` (executor.cpp), the only per-pass sync.
4. **Sticky HIP error.** `OpsExecutor::DeviceUsable()` (-> `GpuOps::Usable()`: stream sync + `hipGetLastError`) is the hook for
   `DeviceUsable()/PollDevice()`; `kFailed + Reset()` does not clear a lost device context, `DeviceUsable() == false` then
   tells the caller to drop the executor. A failing `DoForward` synchronizes before rethrowing so no kernel runs against
   buffers the caller resets next. Not testable on the CPU (the host ops cannot lose a device); untested.

Other items of section 8 of the contract: (5) `oracle/out/*.llama-kvf16.f32` exist now for tinyA, tinyB and the four
quantized files; measured host-ops F16 vs them: tinyA 3.3e-3, tinyB 9.1e-4 max abs (limit 2e-2), top-1 100%. (6) taps are
only emitted while a tap is set; `l_out` for every layer. (7) Row layout `[kv_head][head_dim]`, binary16 or float, K after
norm and RoPE, V raw: contract check 8 passes on the host ops (K/V rows equal the Reference's `Kcur`/`Vcur` taps).

## 6. The gate script (`tests/run-gpu-gates.sh`)

37 invocations in dependency order, each one lock acquisition (`GPU_ADMISSION_LOCK=/server/state/gpu-admission/gpu.lock
gpu-lock-exec -w 120 -- timeout 180 CMD`), one log each in `/server/staging/ripple-gufo/gpu-gates/<timestamp>/`, a table
(gate, invocation, status, seconds, first failure line) printed and saved as `summary.txt`:

| gate | invocations | needs |
| --- | --- | --- |
| kernel_ops | 1 | |
| attention_ops | 5 (small, real-sliding/full x f32/f16) | kernel_ops |
| projection_ops | 11 (every shape of tests/project/run-gpu-tests.sh) | |
| embed | 1 (`embed_test gpu`) | |
| executor_test | 17 (taps, llama-f32, llama-f16, chunk, rewind, twins, quant x4 ...) | all of the above |
| executor_contract | 2 (backend gpu-f32-exact, gpu-f16; the contract main has no filter, the backends file reads `RICURSIVE_CONTRACT_BACKEND`) | executor_test |

Statuses PASS / FAIL / SKIP(77) / LOCKED(75) / NOT-RUN (prerequisite not PASS). Exit 0 all PASS, 1 any FAIL, 2 nothing
failed but something is unverified. Exit 75 is never retried: after the first LOCKED(75) the independent gates are listed
as LOCKED "not tried" and the dependent ones as NOT-RUN. A test exceeding `INVOCATION_TIMEOUT` (180 s; the design budget is
20 s) is terminated by `timeout` (its own process, not anything else) and counts as FAIL. `LIST=1` prints the invocations,
`ONLY=<regex on gate:invocation>` selects.

Tested paths: `tests/run-gpu-gates-selftest.sh` (fake wrapper and fake binaries, 18 checks: all PASS; FAIL cutting off exactly
the dependent gates; one executor invocation FAIL; SKIP(77); lock refused on the first call (one wrapper call, no retry);
lock lost mid-run; timeout; ONLY; missing binary; LIST). The real wrapper was exercised once on the refused path
(`ONLY='^kernel_ops:' GPU_LOCK_WAIT=5`, 14:33): exit 75 after 6.0 s, table showed LOCKED(75) with the wrapper's line,
exit status 2 (log `/server/staging/ripple-gufo/gpu-gates/20261110-143349`). The whole script was then run once at 14:43 with the
default 120 s wait: **not run on GPU: lock held** (kernel_ops LOCKED(75) after 120.0 s, "GPU admission lock held (Scriberr GPU
job or another heavy model)", the other 36 invocations LOCKED "not tried" or NOT-RUN, exit 2, log
`/server/staging/ripple-gufo/gpu-gates/20261110-144344`). Nothing else was attempted.

## 7. Notes for other owners (nothing in `src/**` or other owners' files was edited)

- **`tests/executor_contract_test.cpp` (not mine; test logic untouched): a NaN passes its comparisons.** `MaxDiff` propagates NaN
  but `o.measure = std::max(o.measure, d)` drops it (`std::max(0, NaN)` is 0) and `if (d > tol)` is false for NaN, so checks 1, 2, 4
  (continuation), 5 and 6 accept logits that are NaN in both runs being compared, and a twin that produces NaN "passes" (found
  when the host ops' NaN-poisoned memory made the "RoPE per chunk" twin pass). Suggested fix: `if (!(d <= tol))` and a
  NaN-propagating max. Workaround here: the host contract twins run on zero-filled memory (`HostOpsExecutor::Options::
  poison_memory = false`); the correct runs stay NaN-poisoned and check 3 (llama.cpp, `Compare` has a finite check) would
  catch a NaN. The GPU backends cannot poison, so on the GPU this hole is open.
- **The contract test's `main()` has no filter**, so the GPU contract binary cannot run "one fixture per invocation": the
  backends file reads `RICURSIVE_CONTRACT_BACKEND` (gpu-f32-exact | gpu-f16) and the gate script runs one backend (both
  fixtures) per invocation. Whether that stays under 20 s is unmeasured; a fixture argument in `main()` would split it further.
- `hip/project.cpp` `Scratch::Reserve` frees the old buffer before allocating the new one (LESSONS: "allocate the new buffer
  before freeing the old one"). It sets `q8_` to null and `capacity_` to 0 between the two, so an allocation failure leaves it
  consistent (no dangling pointer), and the executor reserves the maximum once at construction, so it never runs inside a pass.
- No change needed in `src/**`. `hip/embed.cpp` stays in `gufo_ricursive_hip` (HIP target); the host ops therefore do their
  own `DequantizeRow` lookup instead of calling `EmbedRowsHost` (an independent implementation of the same 10 lines).
- Memory per executor at production size (max_rows 2064, 32768 context): scratch about 1.2 GB (g/u 2 x 338 MB, activations
  6 x 42 MB + 3 x 85 MB, head rows), caches 0.78 GB sliding + 2 GiB full, Q8_1 scratch about 0.1 GB. Executors do not share
  scratch (LESSONS: every stream its own scratch); the engine owner sizes sessions accordingly.

## 8. What ran where / unverified

Ran on the CPU today (host ops, the same orchestration the GPU uses): everything in section 4, the gate script's self-test,
the gate script against the real wrapper on the refused path, `kernels_math_test` (983,027 index checks, PASS).
Compiled for the GPU but **never run**: `hip/kernels.hip.cpp` (gfx1151, strict flags, no warnings), `hip/ops_gpu.cpp`,
`hip/device_model.cpp` (both as ordinary C++ against the HIP headers, like Gemma's executor), the three kernel test programs,
`ricursive_executor_gpu_test`, `ricursive_executor_contract_gpu_test`. GPU lock: refused (exit 75, "Scriberr GPU job or
another heavy model") at 14:33 and 14:43; nothing was loaded, unloaded or killed.

Everything below is unverified until a GPU run:
- all of `GpuOps` and `DeviceModel` (stream creation, hipMalloc, async copies and stream order against the null-stream upload
  that is closed by `hipDeviceSynchronize`, `WeightUpload::CopyMapped` on an opened fixture, the zeroed tail margin);
- every kernel and projection result in the executor's context, F16 KV on the device, `exact`, the F32 per-row GEMV at
  the tiny shapes, `Attention128` with group 10 at kv_heads 1 and 2 inside a real forward;
- batch invariance up to 16 rows with quantized weights (open question 1) and the size of the W8A8 envelope for 17+ rows;
- thresholds that were set from host measurements, not GPU ones: GPU tap tolerance 2e-4 (exact) and 5e-2 (F16, 3.8x the host
  F16 measurement 1.3e-2); GPU quantized exact-vs-Reference limit 2e-3 (host measured 8.6e-6); chunk/decode/rewind equality
  tolerance 0 in both modes (expected by construction; if the first GPU run shows 1e-7 differences, find out why before
  loosening); the 0.5 sanity bound on 20-row W8A8 passes (provisional);
- that every invocation stays under 20 s and 4 GB (tiny models; unmeasured), in particular the contract GPU binary (two
  fixtures per invocation) and `twins`/`quant` (host-side Reference runs);
- sticky-error recovery (`DeviceUsable`), `exit 77` of the GPU test programs without a device, the real-size path (uploading
  40 GB, W8A8 at 2048 rows and K = 40960, per-executor memory), and the gate script's behaviour with an admitted lock
  (only the refused path ran for real).

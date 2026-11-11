# Per-stage timing of the executor (RICURSIVE_PROFILE)

Added for job 1692 (where does gufo's decode time go?). Branch `perf-profile`.
Nothing here has run on the GPU: the only runs are the CPU tests and the
host-ops back end on a tiny fixture (see "What was verified").

## Switch

`RICURSIVE_PROFILE=1` in the environment, read once in the `OpsExecutor`
constructor (`hip/executor.cpp`), so it covers `GpuExecutor` and
`HostOpsExecutor`. Any other value, or unset, leaves `prof_` null. The
scalar `CpuExecutor` (`RICURSIVE_BACKEND=cpu`) has no profiler.

Off: no allocation, no extra device wait, no clock read. Each added line is
`Lap(...)` (inline, `hip/executor.hpp`: `if (prof_) LapSlow(...)`) or
`if (prof_) prof_->BeginPass/EndPass`; one pointer test.

On: at every stage boundary `LapSlow` calls `ops_->Synchronize()` (the same
`hipStreamSynchronize` the executor uses before it reads results back; a no-op
on the host ops), then `StageProfile::Mark` adds the `steady_clock` time since
the previous boundary to the stage. The only effect on the work is extra
waits; the operations and their order are unchanged, so outputs are
bit-identical (checked below).

Cost of the switch itself: one device wait per stage, about 17 per layer plus
6 per pass: roughly 1,370 waits per 80-layer pass. Each wait also stops the
GPU from overlapping launch overhead with the previous kernel, so with the
switch on a pass is slower than with it off. That is why the stage times are
real kernel times (plus launch and wait latency) and not a model of the
unprofiled run; compare the total with the switch off
(`GPU-COMMANDS.md` runs both).

## Stages

Defined in `hip/profile.hpp` (`LayerStage`, `PassStage`), placed in
`OpsExecutor::Layer` and `OpsExecutor::Run` (`hip/executor.cpp`). Layer stages
are kept apart for sliding-window layers (`sw.`, 64 in the real model) and
global layers (`gl.`, 16).

| stage | covers (op in `hip/ops.hpp`) |
|---|---|
| `attn_norm` | `RmsNorm` of the residual into `a` |
| `value_mix` | `ValueMix`: x0 + sigmoid(lambda)(a - x0) |
| `proj_q`, `proj_k`, `proj_gate`, `proj_v` | `Project` of attn_q, attn_k, attn_gate, attn_v (matmul stages) |
| `qknorm_rope_kv` | `QkNormRopeAppend`: fused Q/K norm, RoPE (sliding layers), K/V append into the cache or ring slot. The KV write has no separate stage because it is inside this kernel |
| `attention` | `Attention` (`Attention128`): the decode or prompt attention kernel |
| `sigmoid_gate` | `SigmoidGate`: o * sigmoid(gate) |
| `proj_attn_out` | `Project` of attn_output |
| `post_attn_residual` | `ResidualPostNorm` after attention (post-norm and add) |
| `ffn_norm` | `RmsNorm` before the FFN |
| `proj_ffn_gate`, `proj_ffn_up` | `Project` of ffn_gate, ffn_up |
| `swiglu` | `ClampedSwiGlu` (activation and clamp) |
| `proj_ffn_down` | `Project` of ffn_down |
| `post_ffn_residual` | `ResidualPostNorm` after the FFN |
| `embed` | `EmbedRows`: host table lookup and upload (`hip/embed.cpp`) |
| `embed_scale` | `ScaleRows` and the `CopyInternal` of x0 |
| `row_gather` | last-row index upload and `GatherRows` (only for `LogitRows::kLast`, which the engine always uses) |
| `final_norm` | `RmsNorm` with output_norm |
| `output_proj` | `Project` of the output matrix to logits (matmul stage) |
| `logits_readback` | `CopyOut` of the logits to the host |

Report-only rows: `sw.sum`, `gl.sum`, `pass.sum` (sum of the pass-level
stages), `matmul.all` (all matmul stages together, with bytes and GB/s),
`unattributed` = `pass.total` minus the three sums, `pass.total`.

"Weight MB" is the sum of `WeightView::bytes` of the matrices a stage reads
(1 MB = 10^6 bytes), per pass; GB/s = total bytes / total stage time (1 GB =
10^9 bytes). It is the weight traffic of one read of each matrix; it ignores
activations, the KV cache and any re-reads inside a kernel.

## Buckets

Chosen at `BeginPass(rows, context)` with `context = first position + rows`
(tokens in the cache including the one being processed).

- `prompt`: passes of more than one row (any size).
- `gen_lt1024`, `gen_1024_2047`, `gen_2048_4095`, `gen_ge4096`: passes of
  exactly one row, by that context length. A 1-row pass at the end of a
  prefill counts as generation.

A pass joins its bucket only when it completes; a pass that throws leaves no
trace. Buckets with no pass are not printed.

## Report

To stderr when the executor is destroyed (once per executor; skipped if
`OpsExecutor::ProfileReport()` already returned those passes) and on demand
through `OpsExecutor::ProfileReport()`; `ricursive_decode_bench` prints it on
stdout. Every line starts with the bucket tag, so
`grep '^gen_lt1024 '` gives one bucket. Columns: bucket, stage, total_ms,
mean_ms (per pass), share% (of `pass.total`), weight_MB (per pass), GB/s.
The last three lines of a bucket are `mean_ms_per_pass`, `tokens_per_s`
(rows / seconds, so rows per second for `prompt`) and
`sync_points_per_pass` (profile waits, one per `Mark`; the executor's own
`CopyIn`/`CopyOut`/`EmbedRows` waits are in addition).

## Driver

`ricursive_decode_bench MODEL.gguf --tokens FILE.i32 --prompt-tokens N
--generate M [--backend cpu|host|gpu] [--rows R] [--context-cap N]
[--logits-out FILE] [--allow-template-mismatch]` (`tools/decode_bench.cpp`,
`tools/decode_bench_host.cpp`; built only on request, like
`ricursive_gpu_logits`). No existing tool did prompt-then-generate:
`ricursive_gpu_logits` is teacher-forced, `gufo prompt`/`bench` go through the
provider and a sampler.

It loads with `Model::Load`, creates a `Session` (the type the runner's
`Prefill`/`Advance` use, `src/runner.cpp`), runs the first N tokens with
`Session::Sync` (passes of at most `--rows` rows, default 2048 = serving), then
M times takes the arg-max of `Session::Logits()` and feeds it back with
`Session::Evaluate` (one 1-row pass each, so exactly M generation passes). Stop
tokens are ignored. `--backend host` is the executor orchestration on the host
ops (float32 caches), the CPU stand-in for the GPU path; `cpu` is the scalar
reference. `--logits-out` writes the logits after the prompt and after the last
fed-back token (float32 [2][vocab]) for byte comparison.

## What the tool cannot see

- Time outside `OpsExecutor::Run`: `Executor::Forward` argument checks and
  ledger (`src/executor_iface.cpp`), `Session::ForwardChunks` (the
  per-pass `std::vector<float>(vocab)`, `PlanForward`, the write guard), the
  driver's arg-max, and the sampler and HTTP layers of `gufo serve`. Compare
  the driver's wall-clock ms/token with `mean_ms_per_pass` of the generation
  buckets: the difference is that outside time.
- Overlap. With the switch on every stage is isolated by a device wait, so
  launch latency and the gaps between kernels that the real run overlaps are
  charged to stages. Stage times add up to more than the unprofiled pass if
  launches normally hide behind running kernels, and they cannot show which
  kernel the GPU was idle after.
- Inside a kernel (occupancy, bandwidth per wave): the GB/s column is weight
  bytes over stage wall time, not a hardware counter.
- Tapped passes (`Session::SetTap`, tests): the tap's copies and callback run
  between stages and land in the next stage. Profile without a tap.
- Warm-up. The weight upload happens at session creation (counted in the
  driver's `load:` line, not in a bucket), but the first pass may still carry
  one-time costs (code-object loading, page faults on mapped data). That is
  unmeasured; if the `prompt` bucket's first pass stands out, it is in the
  prompt mean.

## What was verified (2026-11-11, CPU only)

- `ctest -R '^ricursive\.' -L cpu` in `build-test`: 20 of 20 pass (the 19
  existing tests plus the new `ricursive.profile`); log
  `/server/staging/ripple-gufo/perf/ctest-cpu-after.log`. The test binaries were
  rebuilt with `ricursive_tests` first (they include `hip/executor.hpp`).
- `tests/profile_test.cpp` (`ricursive.profile`): bucket boundaries, stage
  names, accumulation, aborted-pass handling and the report columns with a fake
  clock. No sleeping, no device.
- `ricursive_decode_bench --backend host` on `chat-tiny.gguf` (made by the
  chat fixture: 5 layers, 4 sliding with window 8, tiny random weights; the
  300-token prompt wraps the ring), `--prompt-tokens 300 --generate 16
  --rows 128`, once without and once with `RICURSIVE_PROFILE=1`
  (files in `/server/staging/ripple-gufo/perf/tiny/`): the 16 generated
  token ids are identical and the `--logits-out` files are identical byte for
  byte (`cmp`, 802,816 bytes each). Without the switch the run prints no
  report.
- The contract suite `ricursive_executor_contract_host_test` run with
  `RICURSIVE_PROFILE=1` (log `contract-host-profile-on.log` in the same
  directory): prints `PASS`, exit 0, with a
  profile report per executor on stderr (the deliberately wrong twins fail as
  they must). Same pass/fail result as without the switch (ctest above).
- With the switch off, no existing statement changed behaviour: the only
  removed line of the whole diff is `OpsExecutor::~OpsExecutor() = default;`,
  replaced by a destructor whose body is skipped when `prof_` is null.

Not verified: any GPU run (`GpuOps::Synchronize` is a real
`hipStreamSynchronize`; that it adds only waits is by reading `ops_gpu.cpp`,
not by running it), the speed cost of the switch on the real model, and the
destructor's stderr print on a GPU executor.

## Sample report

Tiny fixture (`chat-tiny.gguf`: 5 layers, 4 sliding with window 8, 1 global,
embedding 128, random weights), host ops, `--prompt-tokens 300 --generate 16
--rows 128`, `RICURSIVE_PROFILE=1`. Only the generation bucket is shown; the
`prompt` bucket (3 passes of 128, 128 and 44 rows) has the same shape. The
GB/s are tiny because the host ops are scalar loops; on the GPU the column is
the quantity to read.

```
# bucket gen_lt1024: 16 passes, 16 rows (1 row, context = tokens including this one)
bucket         stage                   total_ms    mean_ms  share%   weight_MB     GB/s
gen_lt1024     sw.attn_norm               0.105     0.0065    0.00           -        -
gen_lt1024     sw.value_mix               0.112     0.0070    0.00           -        -
gen_lt1024     sw.proj_q                529.163    33.0727    9.66        0.52      0.0
gen_lt1024     sw.proj_k                532.171    33.2607    9.71        0.26      0.0
gen_lt1024     sw.proj_gate             518.300    32.3938    9.46        0.52      0.0
gen_lt1024     sw.proj_v                522.585    32.6616    9.54        0.26      0.0
gen_lt1024     sw.qknorm_rope_kv          1.193     0.0746    0.02           -        -
gen_lt1024     sw.attention               0.587     0.0367    0.01           -        -
gen_lt1024     sw.sigmoid_gate            0.143     0.0089    0.00           -        -
gen_lt1024     sw.proj_attn_out         517.348    32.3342    9.44        0.52      0.0
gen_lt1024     sw.post_attn_residual       0.114     0.0071    0.00           -        -
gen_lt1024     sw.ffn_norm                0.088     0.0055    0.00           -        -
gen_lt1024     sw.proj_ffn_gate         555.177    34.6986   10.13        0.52      0.0
gen_lt1024     sw.proj_ffn_up           537.627    33.6017    9.81        0.52      0.0
gen_lt1024     sw.swiglu                  0.181     0.0113    0.00           -        -
gen_lt1024     sw.proj_ffn_down         528.974    33.0609    9.65        0.52      0.0
gen_lt1024     sw.post_ffn_residual       0.122     0.0076    0.00           -        -
gen_lt1024     gl.attn_norm               0.024     0.0015    0.00           -        -
gen_lt1024     gl.value_mix               0.029     0.0018    0.00           -        -
gen_lt1024     gl.proj_q                134.012     8.3758    2.45        0.13      0.0
gen_lt1024     gl.proj_k                134.092     8.3808    2.45        0.07      0.0
gen_lt1024     gl.proj_gate             138.309     8.6443    2.52        0.13      0.0
gen_lt1024     gl.proj_v                133.619     8.3512    2.44        0.07      0.0
gen_lt1024     gl.qknorm_rope_kv          0.063     0.0040    0.00           -        -
gen_lt1024     gl.attention               3.383     0.2114    0.06           -        -
gen_lt1024     gl.sigmoid_gate            0.038     0.0024    0.00           -        -
gen_lt1024     gl.proj_attn_out         134.214     8.3884    2.45        0.13      0.0
gen_lt1024     gl.post_attn_residual       0.028     0.0018    0.00           -        -
gen_lt1024     gl.ffn_norm                0.023     0.0014    0.00           -        -
gen_lt1024     gl.proj_ffn_gate         133.471     8.3419    2.44        0.13      0.0
gen_lt1024     gl.proj_ffn_up           138.875     8.6797    2.53        0.13      0.0
gen_lt1024     gl.swiglu                  0.048     0.0030    0.00           -        -
gen_lt1024     gl.proj_ffn_down         144.109     9.0068    2.63        0.13      0.0
gen_lt1024     gl.post_ffn_residual       0.030     0.0018    0.00           -        -
gen_lt1024     embed                      0.188     0.0118    0.00           -        -
gen_lt1024     embed_scale                0.023     0.0015    0.00           -        -
gen_lt1024     row_gather                 0.035     0.0022    0.00           -        -
gen_lt1024     final_norm                 0.033     0.0021    0.00           -        -
gen_lt1024     output_proj              140.442     8.7776    2.56       51.38      5.9
gen_lt1024     logits_readback            0.430     0.0269    0.01           -        -
gen_lt1024     sw.sum                  4243.989   265.2493   77.45           -        -
gen_lt1024     gl.sum                  1094.367    68.3979   19.97           -        -
gen_lt1024     pass.sum                 141.152     8.8220    2.58           -        -
gen_lt1024     matmul.all              5472.489   342.0306   99.87       55.97      0.2
gen_lt1024     unattributed               0.018     0.0011    0.00           -        -
gen_lt1024     pass.total              5479.526   342.4704  100.00           -        -
gen_lt1024     mean_ms_per_pass 342.470
gen_lt1024     tokens_per_s 2.920
gen_lt1024     sync_points_per_pass 91.0
```

# Drafter stage 2: the DFlash2 drafter on the operations layer (host operations tested, GPU kernels written, never run)

Stage 2 of the DFlash2 drafter design (`/server/staging/ripple-gufo/perf/DRAFTER-GUFO-DESIGN.md` sections 4 to 6, job 1692). Branch `drafter-s2`
from `drafter-s4` (4765c3b). Written 2026-11-11. The drafter is written once against `Ops` (`hip/ops.hpp`), like the target's executor, and runs
on the host operations (`hip/ops_host.cpp`, the CPU test backend: **everything below that is called tested ran there**) and on the GPU
(`hip/ops_gpu.cpp` and `hip/drafter_kernels.hip.cpp`: **compiled for gfx1151, never executed**). No GPU, no lock, no gate script was used.
The GPU commands are in `/server/staging/ripple-gufo/perf/STAGE2-GPU.md`.

## What was added

| file | what |
| --- | --- |
| `hip/ops.hpp` | five new `Ops` operations (below), `CreateHostOps` / `CreateGpuOps`, `HostOpsOptions` (the host-only convolution twin), `EncodingOf` |
| `hip/ops_host.cpp`, `hip/ops_gpu.cpp` | their host and GPU implementations (`ops_gpu.cpp` forwards to the kernels' launch functions) |
| `hip/drafter_kernels.{hpp,hip.cpp}` | the GPU kernels, adapted from gufo's Qwen DFlash2 drafter (`THIRD_PARTY.md`) |
| `hip/drafter_math.hpp`, `hip/dequant_math.hpp` | index math (visible keys of a block row, selector constants) and element access into encoded codebook rows, shared by the host operations and the kernels (the host test of each tests the code the GPU runs) |
| `hip/drafter_handles.hpp` | `DrafterHandles` (the drafter's weights in backend memory) and `DrafterTarget` (the target's token table and output head, which the drafter shares) |
| `hip/drafter_executor.{hpp,cpp}` | `DrafterExecutor`: the drafter on `Ops`: per-layer context K/V rings, the context pass, the 16-row block pass, the head and selector for the rows a draft needs, the stage callbacks of the reference, `RICURSIVE_PROFILE=1` |
| `hip/drafter_engine.{hpp,cpp}` | `OpsDrafterEngine : DrafterEngine` over it, `MakeHostOpsDrafterEngine` |
| `hip/drafter_gpu.{hpp,cpp}` | `MakeGpuDrafterEngine` (production: the target's device copy supplies the head) and `MakeGpuDrafterEngineWithHead` (tools: the head alone), the drafter's weights uploaded once per drafter (`DeviceDrafter`, `hip/device_model.cpp`, which also gained `DeviceModel::UploadHead`) |
| `src/runner.{hpp,cpp}` | `MakeDrafterEngine`: the factory of `SpeculativeSetup` (tests), else `MakeDefaultDrafterEngine`: **the GPU drafter when the model runs on the device**, else the CPU reference. The resource claim includes the drafter (weights, per-session bytes), `MeasuredResources` its engine's `DeviceBytes()` |
| `src/drafter_backend.{hpp,cpp}` | `DrafterEngine::Draft(token, position, max_tokens)` (the hint), `DeviceBytes()`; `Propose` passes its budget. The CPU engine ignores the hint, so what it returns is unchanged |
| `tools/drafter_bench.cpp` | `ricursive_drafter_bench` (built on request) and its `--drafts-out` mode; `tools/drafter_compare.py` |
| tests | `tests/drafter_ops_test.cpp` (ctest `ricursive.drafter_ops`), `ricursive.speculative_ops` (the stage 4 tests with the executor as the drafter), `tests/fixtures_text/make_drafter_fixture.py` (`--top-k`, `--rank`, `--codebooks`, `--selector-scale`; the default output is byte-identical to stage 1's), 10 gate entries |

### The operations and where each kernel came from

| `Ops` method | host | GPU kernel | origin |
| --- | --- | --- | --- |
| `DraftQkNormRope` | per-head RMSNorm, float32 rotary, store of K (norm, rotary) and V into a ring slot | `DraftQkNormRopeKernel` | written here from our `QkNormRopeAppendKernel`; any head dimension up to 256 (the target's is 128 only), float32 frequencies as the reference computes them, row t into slot `(slot_start + t) % capacity`, rotary switchable per call (**per layer**: the last layer has none), `heads == 0` for context rows (K and V only) |
| `DraftAttention` | non-causal attention over `[context ring rows ; block rows]` with a window (the sliding layers' `sliding_window`, 0 = the full layer, which sees the whole context window) | `DraftAttentionKernel` | **replaced**, not adapted (see below); the online-softmax tile loop of our `AttentionKernel` over two key segments |
| `DynamicConv` | grouped dynamic causal convolution (kernel 4, group 16, two sides) | `DraftDynamicConvKernel` | `dflash_grouped_dynamic_conv_kernel`, plus an `accumulate` flag so that `h += conv(x)` needs no add kernel |
| `SiluMul` | `gate = silu(gate) * up`, no clamp | `DraftSiluMulKernel` | `dflash_silu_mul_kernel`, unchanged |
| `DraftSelect` | top-k up to 32 over the logits rows, the walk left to right with the transition score | `DraftPartialTopKKernel` + `DraftFinalizeKernel` | `dflash_selector_partial_topk_kernel` and `FinalizeSelector`: top-k 16 -> 32, greedy only, codebooks decoded element by element (F32, BF16, Q8_0, Q4_K) |

**Why the vendored attention was replaced**: gufo's non-causal attention keeps one fp32 score per attended key in dynamic shared memory
(`(history + block) * 4` bytes per head). The full layer sees 16,384 context rows + 16: 65,600 bytes, more than a workgroup may use (64 KB on gfx1151; gufo's
launcher compares the size with that limit only to choose its two-head variant, and its Qwen configurations do not reach it). Our online-softmax loop has no such limit, reads the F16 or F32 ring directly
and handles any head dimension (the tiny fixtures have 32, the target 128). The per-row key ranges are `DraftVisibleKeys` (`hip/drafter_math.hpp`,
tested by brute force against the definition `|p - j| < window`).

**Not new kernels**: embedding rows (`EmbedRows`, the target's raw table rows, scale 1), RMSNorm (`RmsNorm`), every projection (`Project`: Q4_K, Q8_0 and
the target's Q6_K output head), the head and the selector's hidden projection run through the target's existing, tested routes. No existing kernel,
`Ops` method or call site was changed (proof at the end).

## The executor

Per session: `DrafterExecutor` owns an `Ops` (its own stream and projection scratch on the GPU, so it never shares scratch with the target's executor), the
rings and the activations. Weights are shared (`DrafterHandles`, `DeviceDrafter`).

**Context store.** For every layer a ring of K rows and V rows, `kv_width = 8 x 128` elements per row, F16 (production) or F32 (`DrafterOptions::kv_type`,
the model's cache type), position `p` in slot `p % capacity`. A sliding layer keeps `min(context_window, sliding_window)` rows, the full layer `context_window`
(16,384; 0 in the GGUF = `max_context`, 32,768), each plus `ring_slack` (1,024) rows. `AppendContext` runs `fc`, the encoder norm, and per layer the K and V
projections of the new rows (at most `max_append_rows` = 512 per pass) and one `DraftQkNormRope` into the ring. A row appended at position `p` is written
at its slot at once; nothing is copied afterwards. `TruncateTo(p)` sets the end (a plain truncation needs no copy). `ResetTo(base)` sets base and end.
The slack is what makes truncation work with a bounded ring: after `TruncateTo(p)` the rows `[p - window, p)` must still be in the ring, which holds
the last `capacity` rows below the **highest row ever held** since the last reset; a truncation that drops `ring_slack` rows or more below that mark
resets the context at `p` instead (`Base() == p`, the drafter then drafts with a shorter context until rows below `p`... never return; acceptance drops, output **[SUPERSEDED, stage 2b item 4: such a truncation now keeps the rows the rings still hold (the base rises to the oldest row every layer has); the context is emptied only when no row is left below `p`. See "Stage 2b".]**
never changes, the same degradation as a restored cache in stage 4). One row less than the slack still serves an anchor one row below the new end.
`Draft(anchor <= End())` reads the rows `[max(anchor - window, Base(), ring begin), anchor)` and reports them in `Block::context_first / end`.

**Block pass.** 16 rows `[anchor, MASK x 15]` embedded from the target's table; per layer: `RmsNorm`, `conv_proj` (`Project`, 16 rows), the first convolution (`DynamicConv`
side 0), q / k / v projections, `DraftQkNormRope` (q in place; the block's K and V rows go to an fp32 block buffer, slot t, not into the ring), `DraftAttention`, `attn_output`,
`DynamicConv` side 1 **accumulating into the residual**, `RmsNorm`, the FFN convolution, gate and up, `SiluMul`, down, the second convolution accumulating. Then the final norm, the
**output head and the selector's hidden projection for rows 1..d only** (`Project` of d rows), `DraftSelect` for d rows. Block K and V stay fp32; only the context ring is F16
(a difference to the CPU reference, which keeps context K/V in float32; measured below).
Projections are called with 16 rows at most for a block pass: `rocm::Project` runs its FP32 small-batch kernels there (bit-equal per row to the GEMV). Context passes of more than 16 rows run the **[Stage 2b item 2: `DrafterOptions::q8` can send any group of these projections down the 8-bit route (`Ops::ProjectQ8`) instead; the default is this text.]**
W8A8 GEMM (activation rounding) unless `DrafterOptions::exact` (set for the exact test backends).

**Memory on the real model** (Ripple-65B-Preview-DFlash2, Q4_K_M or Q8_0; computed from the configuration by `DrafterSessionBytes`, the same table the executor allocates from):

| item | size |
| --- | --- |
| weights, device, once per process | Q4_K_M: 2,387,245,664 bytes (file; 2.22 GiB) + 4 KiB tail per tensor (81); Q8_0: 4,295,047,776 bytes (4.00 GiB) |
| the target's output head and token table | shared with the target (0.42 GB device); `ricursive_drafter_bench` uploads the head alone |
| per session: context rings, F16 | sliding layers 5,120 rows x 4 KiB = 20.0 MiB each (4 layers), the full layer 17,408 rows = 68.0 MiB: **148.0 MiB** (128.0 MiB without the slack) |
| per session: the context pass (512 rows: features 50 MiB, projections, K/V) | 74.0 MiB |
| per session: the block pass (16 rows of hidden, q, o, 2 x 16 x 40,960 FFN, logits of 15 rows 5.7 MiB) and the selector's scratch | 14.0 MiB + 0.2 MiB |
| per session, total | **236.2 MiB** (the design estimated 130 MiB for the rings alone; it forgot the activations of a 512-row pass) | **[SUPERSEDED, stage 2b item 4: 252.1 MiB with F16 rings; the figure omitted the projection scratch and the MASK rows. See "Stage 2b".]**

## Where to look first on the GPU (every place the GPU code differs from the host code that was tested)

The host operations and the kernels have the same loops, bounds, indexing and parameter names; they differ in these ways, most likely culprits first.

1. **`DraftAttention`**: the host computes scores, the softmax and the output in double with a global maximum; the kernel streams tiles of `head_dim` keys with an
   online softmax in fp32 (`expf`, `fmaf` for q.k and p.v, wave shuffles for the tile reductions, LDS `probs`). A different summation order per row: expect 1e-6 relative differences, not bits.
   It reads the F16 ring through 16-byte loads (`uint4`, 8 halves) and the block rows through fp32 `float4` pairs: **rows must be 16-byte aligned** (`kv_width x 2` bytes, true for 64 and 1,024).
   `head_dim` must be a multiple of 32 on the GPU (one thread per dimension, wave32); the host takes any even head dimension. Untested on the GPU: a context longer than 1,024 keys (the tiny
   walk reaches 87 rows, the real p02 prompt 4,351).
2. **`DraftSelect`**: the host merges the partial lists by sequential insertion; the GPU by per-thread lists and a tree merge in LDS (32 KB, 273 B of private arrays per thread). Equal logits
   at the 32nd place (or equal final scores) can be resolved differently (host: lowest token id; GPU: the lower slot of the merge). No such tie occurred on the tiny fixtures or the real prompts. The transition **[SUPERSEDED, stage 2b item 1, "Tie rules".]**
   sum has the same lanes (4 per candidate, strided ranks, added in lane order) and the same decode function (`DequantAt`, bit-equal to `quant::DequantizeRow`, tested on the host).
   `DraftFinalizeKernel` is one block, 15 rows in sequence: its latency, not its throughput, is what a step pays (expected well under 1 ms).
3. **`DraftQkNormRope`**: device `powf`, `cosf`, `sinf` (a few ulp from the host libm) and a wave-shuffle sum of squares (the host sums in double): relative 1e-7 differences, and the rotary angle
   `float(position) * inv` of positions above 4,096 loses what float32 loses, as the reference does.
   **[Stage 2d, 2026-11-12, job 1692]** The device `powf` differed from glibc's by 1 ulp in 13 of the 64 pairs of head_dim 128 (3 of 16 at 32), and the position multiplied that into the angle
   (q/K 4.9e-4 relative at position 4,351, 1.9e-3 at 16,384+; the first GPU run failed 10 and 11 of 40 checks). The launcher now computes the factors `inv` on the host with the host
   operation's expression and passes them by value (`RopeInv`); angles are then bit-equal and cos/sin within 1 ulp (q/K 3.6e-7 at worst on the gate).
4. **Projections**: `ops.Project` on the host dequantizes and sums in double and ignores `exact`; on the GPU rows <= 16 use the FP32 small-batch kernels and rows > 16 (context passes of 17 to 512 rows)
   the W8A8 GEMM unless `exact`. The gates' `gpu-exact` drafter has `exact` set; the production route (`gpu`, F16) and the real-weights comparison run the W8A8 route for long context passes, which
   changes the context K/V by the activation rounding (stage 3's measurement on the target: about 0.2 maximum absolute, mean KL 8e-4) and so can flip near-tie drafts: **a few token rows differing from
   the oracle on the GPU is expected; the criterion is 95% of rows and acceptance within 2%** (STAGE2-GPU.md).
5. **`DraftDynamicConv`, `DraftSiluMul`**: the host expressions are the vendored ones; `expf` on the device vs `std::exp` on the host (float).
6. **Ring writes**: `DraftQkNormRope` writes F16 with `__float2half` (round to nearest even), the host with `FloatToHalf` (the same rounding).
7. **Launch limits** the host cannot show: LDS 4,352 bytes (attention, two heads), 33,284 bytes (finalize); 1 thread block per (row, head pair) = 16 x 40 blocks for the real shapes (the machine has 40 CUs, so a
   step fills the GPU once per kernel); the selector's partial kernel 49 x 15 blocks. None of this was measured.
8. **Host-only twin**: `HostOpsOptions::conv_across_block_start` (the convolution twin of test 4) does not exist on the GPU, so that twin is skipped in the GPU gates; the window and rotary twins are executor faults and run on both.

The kernels were compiled for gfx1151 with the extension's flags (`-O3 -fno-fast-math -ffp-contract=off`): no scratch spills (the selector kernels use 272 bytes of private arrays), 175 VGPRs at most (finalize), LDS as above, wave size 32.

## Tests (2026-11-11, host operations; `ricursive.drafter_ops`, `ricursive.speculative_ops`, `ricursive.drafter_cpu`)

Logs: `/server/staging/ripple-gufo/perf/stage2-*.log`. Fixtures: stage 1's tiny drafters (`perf/stage1/tiny/drafter-q8_0.gguf`, `drafter-q4_k.gguf`; top-k 8, rank 16), stage 2's
`perf/stage2/tiny/drafter-topk32.gguf` (top-k 32, rank 256, predecessor codebook Q8_0, successor codebook Q4_K, scale 0.25 so that the transition term decides between candidates; made by
`make_drafter_fixture.py <tinyB.gguf> OUT --quant q8_0 --taps 4 --top-k 32 --rank 256 --codebooks mixed --selector-scale 0.25`), all paired with tinyB and its 88-position tap dump.

**1. Stage by stage against `DrafterReference`** (`drafter_ops_test tiny`, 12 s on 4 threads): the executor on the host operations and the reference are fed the same rows and drafted at every
anchor of a walk over positions 8 to 87 (80 anchors: past the 64-row context window and the 8-row sliding window; the rings, with `ring_slack` 8, hold 16 and 72 rows and wrap). 56 named stages
(`ctx_h`, `ctx_k.L`, `ctx_v.L` for 5 layers, and per layer `attn_norm`, `attn_conv_in`, `attn_o`, `h_attn`, `ffn_norm`, `mlp_conv_in`, `mlp_o`, `h`; `emb`, `final`, `hproj`, `logits`) plus the
candidate scores, worst relative RMSE over all anchors:

| fixture, context storage | tolerance | worst stage | drafted tokens equal (of 80 anchors; of 1,200 rows) | candidate sets equal |
| --- | --- | --- | --- | --- |
| Q8_0, F32 | 1e-5 | 1.02e-6 (`mlp_o.4`) | 80; 1,200 | 1,200 of 1,200 |
| Q4_K, F32 | 1e-5 | 1.38e-6 (`mlp_o.4`) | 80; 1,200 | 1,200 of 1,200 |
| Q8_0, F16 | 1e-3 | 2.9e-4 (`ctx_k.1`) | 80; 1,200 | 1,197 of 1,200 (the 32nd candidate, within F16 noise) |
| top-k 32, F32 | 1e-5 | 1.12e-6 (`mlp_o.4`) | 80; 1,200 | 1,200 of 1,200 |
| top-k 32, F16 | 1e-3 | 2.8e-4 (`ctx_v.4`) | 77; 1,173 (**3 blocks differ, all near-ties**: the two choices' scores are within 3.4e-4 of each other) | 1,170 of 1,176 |

**2. Top-k 32**: the fixture above (rows above). **Twin** (the selector keeps 16 candidates): 37 of 80 blocks differ (848 of 1,200 rows equal before the first difference), none a near-tie: it fails, by tokens only
(every hidden stage is unaffected, as for stage 1's selector twin).

**3. Bookkeeping**: the engine and the reference get the same operations: appends of 1, 3, 16, 17, 5, 2, 24 and 20 rows (88 rows, more than the full ring's 72; the engine chunks passes at 5 rows), a draft at the end
and one row below it after each; truncations inside the slack (88 -> 86 -> 81, the edge), a rewrite, a truncation one row past the slack (80, which resets the context there: `Base() == 80`, and the reference is reset to the same base), **[SUPERSEDED, stage 2b item 4: truncating to 80 now keeps rows 72 to 79.]**
a reset to base 40 and 48 appended rows, a deep truncation onto that base (60), a reset to 0, a context shorter than both windows, truncation to the base (an empty context) and to the end (no change): 31 drafts, all with the
reference's tokens and the same rows read (`Block::context_first/end`), and `Base()` / `End()` equal the reference's after every operation.

**4. Twins that must fail test 1** (F32; each ran through the same comparison): the attention window one row wider on the sliding layers: 0 of 80 blocks equal, worst stage `score` 0.84 (first failing stages `attn_norm.1` ...);
rotary on the no-rotary layer (layer 4): 2 of 80 equal, `ctx_k.4` 1.34; the convolution reading across the block start (host operations fault: rows before the start read from the block's end): 0 of 80, `attn_conv_in.0` ...
None passed, none had to be strengthened.

**Also**: the draft-length hint (84 hinted drafts of 1, 2, 3, 4, 6, 15 and more than 15 tokens equal the first tokens of the whole block's, and read the same context rows); `DequantAt` (the function the GPU selector decodes codebook
elements with) equals `quant::DequantizeRow` bit for bit on 33,280 random elements of F32, BF16, Q8_0 and Q4_K rows; `DraftVisibleKeys` against the definition on 1,764 combinations.

**5. The stage 4 end-to-end tests with the executor in place of the CPU reference** (`ricursive.speculative_ops`: groups `real` and `runner` of tinyA and tinyB on the CPU executor and on the host operations,
`RICURSIVE_TEST_DRAFTER=ops`, 97 s on 4 threads): output and every committed logits row bit-identical to plain decoding in every run; the drafter's context ends at the session's position after every step; **all 300 proposals per back
end and fixture (2 back ends x 2 fixtures = 1,200) equal `DrafterReference`'s tokens** (0 near-ties: this is F32 storage); the twins (anchor position, anchor token) still fail the proposal check (60 of 60 / 53-58 of 60 differ) with the output still exact;
prefix reuse, restored checkpoint, the recompute band, tap memory and the runner behave as in stage 4. The `context one row short` twin is a fault of the CPU reference's and is skipped in this mode. The mock groups do not use a drafter and are
unchanged (`ricursive.speculative`).

**6. Real weights on the host operations** (`drafter_ops_test real`, F16 context storage, 4 threads, `free -g` showed 112 to 113 GB available before each run), against the oracle's stage files of stage 1 (variant R):

| drafter | dump (positions) | anchors | all 15 tokens equal | token rows equal | context build | per anchor |
| --- | --- | --- | --- | --- | --- | --- |
| Q8_0 | p07-B-sky (136) | 12 (26..130) | 12 | 180 of 180 | 2.9 s (130 rows) | 8.8 s |
| Q8_0 | p01-A-build-510 (766) | 12 (100..700) | 12 | 180 of 180 | 15.2 s (700 rows) | 9.6 s |
| Q4_K_M | p07-B-sky | 7 | 7 | 105 of 105 | 2.8 s | 9.9 s |

31 anchors of two prompts, **100% of the token rows**, brief criterion 95%. The host path costs 9 s per anchor (double-precision dot products over dequantized rows; the CPU reference needed 1 s on 16 threads); it exists
to be tested, not to be fast. Per-anchor logs: `perf/stage2-real.log`.

**`ricursive_drafter_bench` smoke** (host operations, tiny drafter, `RICURSIVE_PROFILE=1`, `--contexts 40,80`): prints the table, the append times and the stage profile as specified; the drafts-out mode and `tools/drafter_compare.py` were
checked with the oracle's own drafts (PASS) and with corrupted ones (FAIL) on the host. Not run on real weights or the GPU.

**CPU suite**: `ctest -R '^ricursive\.' -L cpu` in `build-test`: **27 of 27 pass** (the 25 of stage 4, plus `ricursive.drafter_ops` and `ricursive.speculative_ops`), 22 min 32 s wall (stage 4: 20 min 48 s), on the commit before this document
(the same code as the branch tip, which adds only this file and `THIRD_PARTY.md`): `draft_session` 811 s, `speculative` 220 s, `speculative_ops` 94 s, `drafter_ops` 11.7 s, `drafter_cpu` 2 s, `plain_fingerprint` 3 s with its golden hashes untouched
(`perf/stage2-ctest-cpu.log`). The new tests stay far under the 5 minutes the brief allows (106 s together). The build of every test target, of the GPU test binaries (`ricursive_executor_gpu_test`,
`ricursive_executor_contract_gpu_test`, `ricursive_kernel_ops_test`, `ricursive_attention_ops_test`, `ricursive_projection_ops_test`, `ricursive_embed_test`), of `gufo_ricursive_gpuexec` (the new kernels, compiled for gfx1151) and of the tools
(`ricursive_drafter_bench`, `ricursive_decode_bench`, `ricursive_dump_taps`, `ricursive_capture_cost`, `ricursive_gpu_logits`, `ricursive_kernel_bench`) succeeded (`perf/stage2-build-final.log`); the gate script's self-test (`tests/run-gpu-gates-selftest.sh`) passes with the counts raised by exactly the 10 entries added.

## Disagreements with the design, things it did not say, things that were surprising

1. **The vendored attention could not be used** (design section 4 table: "vendor `LaunchDFlashNonCausalAttention`, add the full-layer 16384 window"): one fp32 score per key in LDS needs 65,600 bytes for the full layer's 16,400 keys.
   Replaced by an online-softmax kernel (above). It also made the head dimension generic (the tiny fixtures are 32 wide; the design's tiny drafter "hidden 128, 2 layers" is not what stage 1 built either).
2. **`QkNormRopeAppend` / `Attention` of the target do not serve the drafter** (design section 4: "supports `rotary_dim` 0 or 128 ... writes the draft cache layout"): they are written for head dimension 128, GQA group 1/2/10, double-precision rotary angles and
   a ring slot of `pos % capacity` for the K/V they append to. The drafter needs the block's K/V outside the ring, float32 frequencies like the reference (a double-precision angle differs by about `pos x 6e-8` rad), any head dimension for the tiny
   fixtures, and no q for context rows. `DraftQkNormRope` is new.
3. **Top-k 32 is more than raising a constant** (design: "raise `kMaxSelectorTopK` 16 -> 32"): the per-thread lists, the merge buffers, the transition lanes (128 threads / 32 = 4 lanes per candidate instead of 8) and the LDS (32 KB) follow;
   the real files' codebooks are Q8_0 and Q4_K, not BF16, so the selector decodes elements (`DequantAt`).
4. **The 16-row W8A8 route is not available** (design section 4: "the drafter need not be bit-exact, so it may use the fastest width route"; section 5 leaves its speed at 16 rows open): `rocm::Project` switches to W8A8 above 16 rows, so a 16-row block pass **[SUPERSEDED, stage 2b item 2: a drafter-only route exists, `Ops::ProjectQ8`; whether it pays is for the bench.]**
   runs the FP32 small-batch kernels, whose cost on the target is 3.08 x the 1-row time (design 5: 41 ms if the drafter behaves like the target's). Changing `kSplitRows` or the route is a change to an existing kernel path (rule 5): **a decision for the coordinator**
   if the bench says the 16-row projections dominate (STAGE2-GPU.md).
5. **Session memory is 236 MiB, not 130 MiB** (design 3.3: rings only; the 512-row context pass adds 74 MiB of activations and the block pass 14 MiB). The per-pass rows (`max_append_rows`) trade activation memory against the number of passes. **[SUPERSEDED, stage 2b item 4: 252.1 MiB, see "Stage 2b".]**
6. **`TruncateTo` may reset** (DRAFTER-S4: "TruncateTo drops the rows at positions >= p"): with a bounded ring a truncation more than `ring_slack` (1,024) rows below the highest row held cannot keep the rows below it; the engine then **[SUPERSEDED, stage 2b item 4.]**
   resets there, `Base() == p`. A conversation edited more than 1,023 rows back loses its older context for drafting (lower acceptance, never a different output; the same degradation as after a cache restore). The CPU reference engine keeps all rows.
   A larger slack costs 4 KiB per layer per row. Whether 1,024 is enough is a measurement for stage 6 / the coordinator.
7. **F16 context storage flips a few near-tie tokens** on random tiny drafters (3 of 80 blocks on the top-k 32 fixture, 0 of 80 on the others), each a near-tie of the executor's own scores (gap <= 3.4e-4), reported by the tests; on the real weights (31 anchors) it flipped none.
8. **Things the design did not say**: the `DrafterEngine::Draft` hint changes the length of the block `DFlash2DraftBackend::last_block()` holds for an engine that honours it, so the proposal check compares the common prefix and accepts a different token only at a near-tie of
   the reference's own scores, in `RICURSIVE_TEST_DRAFTER=ops` mode only; the CPU reference mode compares whole blocks as before.
9. **Not done / not verified**: nothing ran on the GPU, including the bench on real weights; the kernels' real speed, LDS and occupancy behaviour, the W8A8 route for long context passes and the tie behaviour of the GPU selector are open (section "Where to look first");
   the `chat` / `bench` drivers were not run with the new engine; the non-HIP stand-ins were not compiled (this tree is a HIP build)

## Proof that the target's path is unchanged

- **No existing kernel, projection, executor, embedding, engine or session file changed**: `git diff 4765c3b --stat -- hip/kernels.hip.cpp hip/kernels.hpp hip/kernels_math.hpp hip/project.* hip/executor.* hip/embed.* hip/profile.* hip/ggml src/engine.* src/executor_iface.* src/cpu_executor.cpp src/reference.cpp`
  is empty; so is the diff of `tests/plain_fingerprint_test.cpp`, `executor_contract_test.cpp`, `executor_test.cpp`, `engine_test.cpp`, `kernel_ops_test.cpp`, `attention_ops_test.cpp`, `projection_ops_test.cpp`.
- In `hip/` and `src/` the only removed lines are: the `HostOps` constructor and its one call (now takes `HostOpsOptions{lenient, poison}`; the same two booleans), the `Draft` call in `DFlash2DraftBackend::Propose` (now passes its budget; the default implementation calls the old function), and the
  runner's drafter selection, resource claim, measured resources and two messages (the claim and measurement add the drafter's bytes only when a drafter is configured).
  `Ops` gained five pure virtual methods; the target's executor calls none of them.
- `ricursive.plain_fingerprint` passes with its golden hashes untouched (suite result above); `ricursive.executor_contract*`, `ricursive.executor_host`, `ricursive.engine`, `ricursive.continuation`, `ricursive.draft_session` pass.
- Tests: no existing check was loosened or removed. In `tests/` the removed lines are the stage 1 generator's constants (the default output is byte-identical, checked with `cmp`), the gate self-test counts (raised by exactly the 10 entries added: 82 -> 92, 83 -> 93, 49 -> 59 not-run rows),
  and in `speculative_test.cpp` the proposal comparison (the same strict comparison for the CPU reference, a prefix comparison with near-tie reporting only in `RICURSIVE_TEST_DRAFTER=ops` mode), the twin list (the reference's own twin only in the default mode) and the filter (a comma list added).


## Stage 2b (2026-11-11, branch `drafter-s2b` from `drafter-s4b` 72ced1a, job 1692): test reach before the first GPU run, a faster projection route, bookkeeping

An independent review of stage 2 found no way for the drafter to change the target's output, but found that tonight's gates would not reach the code that only real shapes exercise, and
several avoidable costs per draft step. **Statements above that this section replaces are marked SUPERSEDED, not deleted.** Everything ran on the CPU (host operations); the GPU code compiles for
gfx1151 (`gufo_ricursive_gpuexec`, the GPU test targets and the tools were built) and **has not run**. One commit per item (items 1 and 5 have later `fixup!` commits that could not be folded: later items touch the same file).

### Item 1: the operation-level test at the real shapes

`tests/drafter_op_shapes_test.cpp` (ctest `ricursive.drafter_op_shapes`, gate `drafter_op_shapes`, 2 invocations: `gpu-exact` F32 rings, `gpu` F16 rings; gate list 124 -> 126, self-test counts adjusted and the self-test run: all pass).
It drives each new operation alone through `Ops`, with seeded random inputs at the real shapes (hidden 5,120, head dimension 128, 80 query heads on 8 K/V heads, FFN 40,960, vocabulary 100,352 = 49 partial lists, top-k 32, rank 256).
Side A is under test, side B the reference: cpu mode A = host operations, B = a naive implementation in the test (doubles, brute-force key sets, its own binary16 conversion and `DequantizeRow`, no code shared with `ops_host.cpp`);
`RICURSIVE_TEST_BACKEND=gpu-exact|gpu`: A = GPU operations, B = host operations. The header of the file states every tolerance and derives it from the arithmetic. Coverage: `DraftQkNormRope` at 15 situations (block pass at positions 0 to 32,760;
context passes of 1, 16 and 512 rows into the 5,120-row sliding ring and the 17,408-row full ring, ending at, 100 rows before, 1 row before and just after the wrap, and a second lap); `DraftAttention` at 14 context situations (87, 500, 4,096 rows,
ring just full, one past the sliding wrap, second lap, 16,384 rows, one past the full ring's wrap with the 16,384-row clamp, position 30,000; contexts of 4,096 rows and more run 4 block rows instead of 16 because the sequential host operation takes 50 to 80 s
for 16 rows at 16,384 keys); `DynamicConv` (both sides, plain and accumulating); `SiluMul`; `DraftSelect` with three codebook type pairs, six equal logits straddling the cut at 32 in every row, and a designed tie of final scores; `ProjectQ8` (item 2) on four real shapes.
**Result (cpu mode, 4 threads): 70 checks, 0 failed, 197.7 s** (`perf/stage2b-op-shapes.log`); worst attention error 8.9e-7 against the tolerance 4e-6, worst convolution 2.0e-7 (1e-6), silu 2.3e-7 (3e-6), selector candidate lists equal in 75 of 75 rows, 0 near-ties,
worst candidate score error 0.02 of its bound. **Twins** (one wrong ring index in the naive side: a slot off by one past the wrap, in the attention read and in the ring write): both fail the comparison, as they must.
The test first failed on its own tolerance for F16 K rows (a result near zero whose float rounding differs in sign) and on its run time (11 min); both were fixed in the test, not in the code under test.

**Tie rules.** Before stage 2b: the host top-k took the lowest token id for equal logits (strict insertion in id order), but for equal *final scores* the first candidate in logit order (so the higher logit, not the lower id); the GPU merged the per-thread lists keeping the *lower thread slot* for equal logits
(not the lower id: thread t sees ids t, t+128, ...), and for equal final scores also the first in logit order; `DrafterReference` takes the lowest id in both cases. After: `rocm::DraftRanksAbove(score, id, score2, id2)` (`hip/drafter_math.hpp`: larger score, equal scores the lower id) is
used for the host insertion and final choice and by the kernels' `InsertCandidate`, the tree merge and the finalize choice (three comparisons changed in `drafter_kernels.hip.cpp`, compiled, **not run**): all three implementations now choose the lowest token id at every tie. The test builds both tie kinds (the designed final-score tie is
one candidate with logit 4 plus a transition term of exactly 1 against one with logit 5 and term 0: the lowest id must win; "first in logit order" would not).
The check no longer grades itself: a differing choice is forgiven only if, in B's scores, the two choices are within the sum of their two score tolerances (5e-6 x (|logit| + sum of the 256 |terms|) each); near-ties are counted and printed, the run fails above 2% of the compared rows, and the built exact ties get no forgiveness.

### Item 2: the 8-bit activation route at any row count

`Ops::ProjectQ8(w, x, rows, y)` (new; `Project` and its callers unchanged): GPU `rocm::ProjectW8A8` in `hip/project.cpp` = Q8_1 quantization of the activations plus `LaunchBatchedQuantGEMMPreQuantized` at any row count for quantized weights (dense F32/BF16 weights take `Project`'s per-row GEMV); it refuses to run when
the scratch is too small instead of reallocating inside a pass. `DrafterOptions::q8` = `DrafterRoutes{block, head, context}` (default none = today); `exact` overrides every route. Selectable by `RICURSIVE_DRAFTER_Q8=block,head,context|all|none` (the runner, via `EffectiveDrafterOptions`),
`ricursive_drafter_bench --q8 SET/SET...`, `RICURSIVE_TEST_Q8` (the real-weights check). **The host has no model of the Q8_1 activation rounding** (the only W8A8 material in the tree is the GPU test's noise model), so on the host a routed projection computes `Project`'s result and the route is tested by call counting:
`CheckRoutes` (16 configurations: 8 subsets x exact or not) finds exactly the chosen groups' matrices on the route (counts, rows, and the head's two weights by identity). The tiny walk with every group routed gives the same tokens as the reference (host: the same arithmetic), so the "95% of rows on real weights" report
**is not available on the host and was not produced**; it needs the GPU (STAGE2-GPU.md steps 3c and 6). The GPU route's arithmetic is checked tonight by the op-level gate (relative RMSE per row <= 1.2e-2, derived in the test header: about 0.5% expected).

### Item 3: avoidable work per draft step (final code)

Construction: the 15 MASK rows are embedded once (one `EmbedRows`, which waits for the device) into a buffer the block pass reads. Per draft step, in this order: `CopyIn` of the anchor token (4 bytes; waits: the stream is idle), `EmbedRows` of one row (20,480 bytes up, waits),
`CopyInternal` of the 15 MASK rows into the block input (307,200 bytes on the device, asynchronous), then only kernel launches (no wait: scratch is reserved at construction), then one `CopyOut` of 3,904 bytes (tokens, candidate ids and scores in one buffer, whatever `d` is; waits). **Three waits and two host-to-device copies per step** (before: five waits, one of them in the middle of the step
before the selector's launch, and 16 embedded rows, about 327 KB, up per step; three read-backs). The context range read is known on the host and is not copied. No asynchrony was added between the drafter and the target. `HostOpsCounters` counts every call on the host and `CheckTransfers` asserts the list above.
A context pass: one `CopyIn` of its feature rows (waits), no read-back. On the GPU `CopyIn`, `CopyOut` and `EmbedRows` synchronize the stream; `Project`/`ProjectQ8` and the drafter kernels do not.

### Item 4: bookkeeping and accounting

*Deep truncation.* `TruncateTo(p)` with `p + ring_slack <= highest row held` no longer empties the context: every layer's ring holds the rows `[highest - capacity, highest)`, so the base rises to `highest - (the smallest ring)` (the 5,120-row sliding rings: rows from `highest - 5120`);
if no row is left below `p` the context is empty at `p`. Tested against the reference given the same rows: the bookkeeping script (truncation to 80 now keeps rows 72 to 79) and a sweep of every position from 40 to 87 of 88 rows across the slack edge (slack 8, smallest ring 16): 8 truncations kept rows, 33 left none, 63 drafts equal the reference's.
*Accounting.* The projection scratch is now planned from what is used (`rocm::Q8ActivationBytes`; the Q8_1 tile layout of `prefill_quant_gemm.hpp`; only the context pass above 16 rows or on the route, the block pass and the head when routed; none with `exact`), the MASK rows are in the plan, `DeviceBytes()` adds `Ops::ProjectionScratchBytes()` (the real `Scratch` capacity on the GPU),
and the runner's claim uses `EffectiveDrafterOptions` (the cache type and exactness of the backend, as the engine does). Per session, real configuration: **F16 rings 252.1 MiB** (rings 148.0, context pass 74.0, block pass 14.27, selector 0.18, scratch 15.6; the figure of stage 2, 236.2 MiB, left out the scratch and the MASK rows,
and the engine really held 261.2 MiB: its scratch was reserved at (512, 40,960) = 25.0 MiB); **F32 rings 400.1 MiB** (the old claim assumed F16 rings: 236.2 against 409.2 allocated); `exact`: 236.5 / 384.5 MiB (no scratch); block route scratch 0.78 MiB, head route 0.10 MiB (below the context pass's 15.6 MiB). Tested on the host: `DeviceBytes() == DrafterSessionBytes` in 96 configurations
(F16/F32, exact or not, 8 route subsets, 3 ring/pass sizes), and the runner's `ResourceClaim()` equals its state's `MeasuredResources()` in `ricursive.speculative_ops`. On the GPU `DeviceBytes()` reads the real scratch allocation, so the same check in the tiny gates tests the formula against ggml's.

### Item 5: the bench and tonight's commands

`ricursive_drafter_bench`: a 16,384-row context in the default list, `--engine head|factory` (the production `MakeGpuDrafterEngine` over the target's device copy; uploads the whole target first), `--q8 SET/SET...` (route sets in one process, the drafter's weights pinned between them), `--attention NAME` (`DrafterOptions::attention_kernel`,
validated at construction; only `default` exists; `Ops::DraftAttention` takes no kernel yet: stage 2c adds the argument **[SUPERSEDED, stage 2c: `default` and `tiled` exist; stage 2c added a second method, `Ops::DraftAttentionTiled`, instead of an argument, and `--attention` takes a list `default/tiled` for an A/B in one process. See "Stage 2c".]**), the cost of creating an engine (first, and a second from the resident weights), allocated against claimed session bytes, medians with p10/p90/min/max and the per-stage profile as before. `drafter_ops_test real` now also compares every named stage with the oracle's stage file (relative RMSE table);
`drafter_ops_test long` writes (host) or compares (GPU) drafts at 5,121 / 16,385 / 17,409 / 20,000 rows on real weights. **Not run**: both need 20 GB of free host memory (35 available) and `free -g` showed 12 GiB at the time; they were built and syntax-checked only. The ordered run list is `/server/staging/ripple-gufo/perf/STAGE2-GPU.md`.

### Proof that the target's path is unchanged

`git diff 72ced1a --stat` is empty for `hip/kernels.*`, `hip/kernels_math.hpp`, `hip/executor.*`, `hip/embed.*`, `hip/profile.*`, `hip/ggml/`, `src/engine.*`, `src/executor_iface.*`, `src/cpu_executor.cpp`, `src/reference.cpp`, and the tests `plain_fingerprint`, `executor_contract`, `executor`, `engine`, `kernel_ops`, `attention_ops`, `projection_ops`.
`hip/project.cpp` only gained `ProjectW8A8` (nothing removed). `Ops` gained `ProjectQ8` and `ProjectionScratchBytes`; the target's executor calls neither; the host `Project` body was moved into a helper behind a call counter, bit for bit. `ReserveProjection` on the host now records a size (no allocation). Existing test lines removed are the gate-count constants (124 -> 126), the stage 2 bookkeeping expectations that item 4 supersedes,
and the two engine-factory option lines (now `EffectiveDrafterOptions`); no check was loosened.

## Stage 2c: the second attention kernel (2026-11-11, branch `drafter-s2c` from `drafter-s4d` 5a54398, job 1692)

A second kernel for `DraftAttention`, **`tiled`**, in which each K/V tile of a K/V head is loaded from memory once and used by every query that
sees it (16 block rows x 10 query heads). The existing kernel (`DraftAttentionKernel`) is unchanged and stays the default; `tiled` is chosen by name
(`DrafterOptions::attention_kernel`, `ricursive_drafter_bench --attention default/tiled`, `RICURSIVE_TEST_ATTENTION=tiled` in the tests). **Compiled for gfx1151,
never run on a GPU.** Everything called tested below ran on the CPU, on its structural twin. Logs: `/server/staging/ripple-gufo/perf/stage2c-*.log`.

| file | what |
| --- | --- |
| `hip/drafter_math.hpp` | the shared index arithmetic: `DraftTiledSplits` (the key plan), `DraftTiledSplitKeys`, `DraftTiledSlot` (ring slot without a modulo), `DraftTiledThreads`, `DraftTiledScratchFloats`, `DraftTiledRecord`; constants `kDraftTiledKeys` = 32, `kDraftTiledMaxSplits` = 16 |
| `hip/drafter_kernels.{hpp,hip.cpp}` | `DraftTiledAttentionKernel<kD, CacheT>` (kD 32, 64, 128; F16 or F32 ring), `DraftTiledCombineKernel`, the launcher `rocm::DraftAttentionTiled` |
| `hip/ops.hpp`, `hip/ops_gpu.cpp`, `hip/ops_host.cpp` | `Ops::DraftAttentionTiled` (a new method: `Ops::DraftAttention` and its two implementations are untouched); the host implementation is the **structural twin** `TiledAttentionWorkgroup` / `TiledAttentionCombine`; `HostOpsOptions::tiled_fault` (three test-only defects) |
| `hip/drafter_executor.{hpp,cpp}` | `kDraftAttentionKernels = {default, tiled}`; with `tiled` the executor allocates the partials scratch (in `DrafterSessionBytes`) and calls `DraftAttentionTiled`; refuses a head dimension other than 32/64/128 at construction |
| tests, gates, bench | below |

### Design

**Geometry.** Grid `(splits + 1, kv_heads)`, block `DraftTiledThreads(tokens, group)` = 2 threads per query rounded up to whole waves of 32: **320 threads (10 waves) for the
real shapes** (16 rows x 10 heads), 128 for the tiny drafters (16 x 4). Workgroup `(split, kvh)` serves every query `qi = t x group + g` of K/V head `kvh` over one key range:
- **Key numbering.** The context keys of all rows are numbered `n = 0 .. ctx_keys - 1` from the first row's first visible context row: every row's range ends at the anchor and its
  begin `max(ctx_first, p + 1 - window)` does not decrease with the row, so the union of the rows' ranges is the first row's (`DraftTiledSplits`; checked by brute force against
  `DraftVisibleKeys` on 414 plans, below). Each query keeps its own visible range `[vis_lo, vis_hi)` in that numbering, from `DraftVisibleKeys` (the function the old kernel and the host
  operation use), and masks the keys outside it: a ring that has wrapped, a context shorter than the window, a base above 0 and the sliding window's per-row start are all
  `DraftVisibleKeys`'s.
- **Splits.** `tiles = ceil(ctx_keys / 32)`, `tiles_per_split = ceil(tiles / 16)`, `splits = ceil(tiles / tiles_per_split)` (at most 16, never empty, whole tiles but the last);
  split `splits` is the block's own 16 rows (fp32, one tile). 500 context rows: 16 splits of 1 tile; 4,096: 16 of 8 tiles; 16,384: 16 of 32 tiles. **136 workgroups per
  layer** for the real shapes (17 x 8), each of 10 waves.
- **Ring slot without a modulo per key**: one modulo per workgroup (`slot0 = (ctx_begin + n0) % capacity`), then `DraftTiledSlot(slot0, rel)` = `slot0 + rel` minus `capacity` if
  it is not below it. Exact because `rel < ctx_keys <= capacity` (the launcher refuses `ctx_keys > capacity`, as the old one refused a row's range larger than the ring).
- **Per tile (32 keys)**: the workgroup stages the tile's K and V rows of its head as fp32 into LDS (chunks of 8 elements, `LoadKeys8` as the old kernel: 16-byte loads; keys past
  the split are zeros), one barrier; every thread scores the 32 keys with its half of q (dimensions `d = 4 (2 j + half) + e`: the two halves of a query are interleaved by 16 bytes)
  and adds its partner's half with `__shfl_xor(p, 1)` (half 0 forms `p0 + p1`, half 1 `p1 + p0`: the same float); both threads of a pair therefore hold the same scores, max and sum;
  then the online-softmax update of its 64 output dimensions; one barrier before the next tile is staged. No block-wide reduction at all.
- **Partials and combine.** Each query writes `acc[head_dim], m, l` to the scratch (`[kv_heads][splits + 1][tokens x group][head_dim + 2]` floats, scalar stores); a second
  kernel, grid `(tokens, heads)`, block `head_dim`, combines the splits in order 0 .. splits (block rows last) and writes `out`. Scratch: `(16 + 1) x tokens x heads x (head_dim + 2)`
  floats = **10.8 MiB** per session at the real shapes, one buffer for all layers, in `DrafterSessionBytes` only when `tiled` is chosen.

**Softmax, and why it is exact up to rounding.** Within a split, per query, after every tile: `m` = the largest score seen, `l = sum exp(s - m)`, `acc = sum exp(s - m) v` over the
visible keys seen so far. A tile with largest score `t_max`: `m' = max(m, t_max)`, `alpha = exp(m - m')`, `p_i = exp(s_i - m')`, `l' = l alpha + sum p_i`, `acc' = acc alpha + sum p_i v_i`:
multiplying the old sums by `exp(m - m')` re-expresses them relative to `m'`, so the invariant holds after the tile in exact arithmetic. Combine: `M = max_s m_s`, `w_s = exp(m_s - M)`,
`out = sum_s w_s acc_s / sum_s w_s l_s = sum_all exp(s - M) v / sum_all exp(s - M)` = softmax(s) V, exactly in real arithmetic; in fp32 the only errors are the rounding of the dot
products, `expf`, the products and the sums (every exponent argument is <= 0, so nothing overflows). Edge cases: a query that sees no key of a tile and none before keeps its state
(`m' = -inf`: the update is skipped; it would compute `exp(-inf - -inf)` = NaN); a split in which a query saw no key has `m = -inf`, `l = 0` and weight 0; every query sees at least
its own block row (`|p - p| = 0 < window`, or window 0), so `M` is finite and `l > 0`.

### Numbers (computed from the configuration by a script, not measured; 4 sliding layers with window 4,096, 1 full layer with 16,384; 16 rows; F16 ring, fp32 block rows)

| per draft step | 500 context rows | 4,096 | 16,384 |
| --- | --- | --- | --- |
| K/V bytes read, old kernel (each K/V head's keys read by 16 rows x 5 head pairs = 80 workgroups) | 831.2 MiB | 6,439.4 MiB | 10,279.4 MiB |
| K/V bytes read, `tiled` (once) | 10.4 MiB | 80.6 MiB | 128.6 MiB |
| `tiled`'s other traffic: partials written and read by the combine | 107.9 MiB | 107.9 MiB | 107.9 MiB |
| `tiled`'s other traffic: q rows (every workgroup reads its 160 rows; old: 2 rows per workgroup, 3.1 MiB) | 53.1 MiB | 53.1 MiB | 53.1 MiB |
| multiply-adds (q.k and p.v, 128 each per key), old | 0.845 G | 6.726 G | 10.752 G |
| multiply-adds executed by `tiled` (tiles padded to 32 keys, masked keys included) | 0.891 G (+5.4%) | 6.763 G (+0.6%) | 10.790 G (+0.3%) |
| workgroups per layer | old 640 x 128 threads; `tiled` 136 x 320 threads + combine 1,280 x 128 | same | same |
| LDS per workgroup | old 4,352 B (two heads); `tiled` 32,768 B (2 x 32 keys x 128 x 4) | same | same |

**What limits it.** Memory traffic falls from 0.8 to 10 GiB per step to 0.17 to 0.29 GiB (at 500 rows the partials and q rows are 94% of `tiled`'s traffic: see "left on the table").
The multiply-adds are the same as the old kernel's, so the limit becomes the arithmetic: every fp32 FMA takes one operand from LDS (each thread reads the tile's K and V rows of its
half: 16 `float4` LDS reads per key for 64 FMAs each, the 16 lanes of a half-wave reading the same address) and the compiler places the kernel at 8 waves per SIMD (192 VGPRs). Which of FMA issue, LDS reads
or the 136-workgroup granularity binds is for tonight's measurement. For orientation only (an extrapolation, not a measurement): the target's attention kernel, the same loop structure as the
old drafter kernel, took 285.9 us for 1 row x 80 heads x 4,100 keys (`docs/PERF-KERNELS.md`); 16 rows of it would be about 4.6 ms per sliding layer at 4,096 rows and about 18 ms for the
full layer at 16,384, about 37 ms of a step's 41 ms budget. No time is predicted for `tiled`.
**Why this geometry** (from the measurements in the repository): `docs/PERF-KERNELS.md` ("Multi-row attention") found the target's multi-row attention cost to be per workgroup and per tile
(loads, syncs, block reductions), not per product, and fewer, larger workgroups faster; `tiled` has 136 workgroups, two barriers per 32-key tile and no block-wide reduction. The same
document measured a K tile staged in LDS **2x slower** for the target's kernel (row stride 17 chunks, 44 KB LDS): there one staged K row served at most 10 heads of one row, here 160 queries;
whether staging pays here is exactly what the A/B measures. Head dimension 128 in two halves keeps q (64) and the accumulator (64) in registers: 192 VGPRs, no spill.
**Left on the table** (simplicity first): fewer splits at short contexts (a minimum split length would cut the partials and q traffic that dominate at 500 rows, at the cost of fewer
workgroups); the matrix units (WMMA) for q.k and p.v; F16 K/V in LDS (half the LDS reads, a conversion per read); several queries per thread (fewer LDS reads per FMA); the combine folded
into the main kernel (needs a cross-workgroup completion count: atomics, which the first night should not debug); a 16-key tile (16 KiB LDS, fewer registers).

### Compiler report (gfx1151, the extension's flags `-O3 -fno-fast-math -ffp-contract=off`, `-Rpass-analysis=kernel-resource-usage` and the device assembly; `perf/stage2c-resource-usage.log`)

| kernel | VGPRs | SGPRs | scratch (spills) | LDS bytes | occupancy (waves/SIMD) | wave size |
| --- | --- | --- | --- | --- | --- | --- |
| `DraftTiledAttentionKernel<128, __half>` and `<128, float>` (the real shapes) | 192 | 37 | 0 (VGPR spills 0, SGPR spills 0) | 32,768 | 8 | 32 (`.amdhsa_wavefront_size32 1`) |
| `DraftTiledAttentionKernel<64, *>` | 125 | 37-38 | 0 | 16,384 | 10 | 32 |
| `DraftTiledAttentionKernel<32, *>` (the tiny drafters) | 102 | 38 | 0 | 8,192 | 12 | 32 |
| `DraftTiledCombineKernel` | 8 | 23 | 0 | 0 | 16 | 32 |

LDS limit per workgroup: 64 KB on gfx1151 per this document (stage 2, "Why the vendored attention was replaced"); 32,768 B is half of it. **The existing kernels' device code did not change**: the
device assembly of all 10 kernels that were in `hip/drafter_kernels.hip.cpp` at 5a54398 (`DraftAttentionKernel` x 4, `DraftQkNormRopeKernel` x 2, conv, silu, partial top-k, finalize) is
identical, line for line (block labels normalised), compiled with the same command from the base sources and from this branch (`perf/stage2c-isa-existing-kernels.log`).

### Every way the kernel differs from its structural twin (what the night run should suspect first)

The twin (`hip/ops_host.cpp`, `TiledAttentionWorkgroup<kD>` / `TiledAttentionCombine`) has the kernel's statements in the kernel's order and calls the same `drafter_math.hpp` functions; a GPU
thread is an iteration over `tid`, a barrier is the end of a loop over all threads, the shuffle reads the partner's partial from the array the loop just filled.
1. **Concurrency.** The twin cannot show a race: a missing or misplaced `__syncthreads()` (two per tile: after staging, before the next staging), or a shuffle in divergent code (the
   kernel executes `__shfl_xor` in every lane, active or not, outside any branch). Suspect first if results differ between runs or only for long contexts (many tiles).
2. **Lanes of a pair in one wave.** The partner of thread `tid` is `tid ^ 1`; true for any wave size (compiled wave32). The twin assumes it.
3. **Loads.** The kernel reads K/V in 16-byte vectors (`uint4` of 8 halves, two `float4`): ring rows, block rows and head offsets must be 16-byte aligned (true for `kv_width x 2` bytes and head
   dimensions that are multiples of 32, as for the old kernel); the twin reads scalars. q loads and record stores are scalar in both (a record is `head_dim + 2` floats, not 16-byte aligned).
4. **LDS layout.** The kernel's `float4 ks[32 x kD / 4]` at index `i x kD/4 + 2 c8 (+1)` (staging) and `i x kD/4 + 2 j + half` (reading) are the twin's floats `i x kD + 8 c8 .. + 7` and
   `i x kD + 4 (2 j + half) .. + 3`.
5. **Staging loop.** The kernel's threads take chunks `c = tid, tid + blockDim.x, ...`; the twin takes every chunk once in order. The same set iff `blockDim.x` is what
   `DraftTiledThreads` returns (the launcher passes exactly that).
6. **`expf`** on the device vs `std::exp` (float) on the host: a few ulp. `fmaf` / `std::fma` (both correctly rounded), `fmaxf` / `std::fmax`, and the plain `+ * /` are the same IEEE
   operations (the kernel file has no FP contraction; the host build has no `-march`, so GCC cannot contract into FMA instructions). The twin is therefore not bit-identical, only within rounding.
7. **Index checks.** The twin checks every ring slot (`slot < capacity`, else it loads zeros and the operation fails: the wrap twin's failure mode) and the host checks every pointer range,
   including the part of the scratch the workgroups write; the kernel checks nothing per element (the launcher validates the arguments as the twin does).
8. **Workgroup order.** The twin runs the workgroups on OpenMP threads, the combine after all of them (the second launch on the same stream on the GPU); no result depends on the order.
9. **Inactive threads** (`2 x tokens x group` below the rounded block size; none at the real shapes' 16 rows, 16 of 96 at the op-level test's 4-row cases): both form the pointer to
   q row 0 but read nothing through it (`active ? qrow[...] : 0.0F`), hold zeros, see no key and write nothing.

### Tests (2026-11-11, CPU; the structural twin is what ran)

- **The operation-level test at the real shapes** (`RICURSIVE_TEST_ATTENTION=tiled ricursive_drafter_op_shapes_test`, ctest `ricursive.drafter_op_shapes_tiled`, attention only): the twin
  (A) against the existing host operation `DraftAttention` (B), at all 14 context situations of the test (87, 500 and 4,096 rows, the sliding ring just full, one past the sliding wrap, a
  second lap with the block across a slot boundary, position 20,000, full layer at 87, 500, 4,096 rows, a base above 0, 16,384 rows, one past the full ring's wrap with the 16,384-row clamp,
  position 30,000), F32 and F16 rings: **28 of 28 within the tolerance 5e-5** (the GPU's attention tolerance of the test header: the twin computes in fp32 as the GPU does); worst
  3.69e-6 (sliding, ring just full, F16), largest F32 3.21e-6 (full, 16,384 rows). 31 checks, 0 failed, 161.1 s (`perf/stage2c-op-shapes-tiled.log`). **Added after the review (finding 3):** one tiled-only case at the real 16 block
  rows with several tiles per split (sliding, 4,096 context rows: 16 splits of 8 tiles, 320 threads; before it every case of 4,096 rows or more ran 4 rows and the 16-row cases had at
  most one tile per split): F32 2.75e-6, F16 2.41e-6; the rerun has 30 attention cases, worst 3.21e-6, 33 checks, 0 failed, 87.8 s (`perf/stage2cfix-ctest-affected.log`; the twins'
  numbers there differ from the ones below because the extra case shifts the random inputs; all three fail as before). In the GPU modes each tiled invocation runs 15 checks.
- **Twins that must fail it** (through the structural twin, `HostOpsOptions::tiled_fault`, on three cases: sliding 500 rows, one past the sliding wrap, full at position 30,000):
  every tile one key late (a tile boundary off by one key): fails, worst 1.22 / 0.468 / 0.176; the splits combined without rescaling to the common maximum: fails, 1.31 / 1.23 / 1.29;
  the ring slot without its wrap check: fails on both wrapping cases by the host's slot check ("a context key's ring slot is not below the capacity"; on the GPU this would read past
  the ring), and on the non-wrapping case (500 rows) it passes with 1.91e-6, as it must (no slot wraps there). **This twin fails through the host's own slot guard, which the kernel does
  not have: it is evidence that the guard catches a missing wrap, not numeric evidence that the comparison would** (the comparison never sees the wrong values). The first two twins fail
  by the comparison itself. None passed; none had to be strengthened.
- **The tiny walks with the twin** (`RICURSIVE_TEST_ATTENTION=tiled ricursive_drafter_ops_test tiny`, ctest `ricursive.drafter_ops_tiled`; `perf/stage2c-drafter-ops-tiled.log`; since the
  review (finding 6) the log names the kernel in use, `backend host, attention kernel tiled (...)`: `perf/stage2cfix-ctest-affected.log`): every
  stage-1 check against `DrafterReference` passes as with the default kernel: Q8_0 F32 80 of 80 anchors, worst stage 1.05e-6 (1e-5); Q4_K F32 80, 1.42e-6; Q8_0 F16 80, 2.89e-4 (1e-3);
  top-k 32 F32 80, 1.17e-6; top-k 32 F16 77 of 80 with 3 near-ties (gap at most 3.33e-4; stage 2's table above reports 3 near-tie blocks for the default kernel too; whether they are the same blocks was not compared); the bookkeeping, the slack sweep, the accounting
  (`DeviceBytes() == DrafterSessionBytes` in 96 configurations, now with the tiled scratch) and the twins of stage 2 (window off by one, rotary, convolution: all fail) as before.
  The key plan by brute force (`CheckTiledSplits`, host part of the same test): 414 plans (windows 0 to 4,096, 1 to 16 rows, anchors to 30,000, bases to 13,616): the union equals the first
  row's range and contains every row's, the splits are consecutive, non-empty, whole tiles, at most 16; `DraftTiledSlot` equals `position % capacity` on 2,517,591 (key, capacity) pairs.
- **The gate list** takes the kernel: 4 entries in two gates of their own, `drafter_op_shapes_tiled` and `drafter_ops_tiled` (`gpu-exact`, `gpu` each; prerequisites as their
  default-kernel twins; **no gate depends on them**, so a tiled failure never marks a default-kernel invocation NOT-RUN; review of stage 2c, finding 2, which superseded the first
  version: the entries were inside `drafter_op_shapes` / `drafter_ops`, on which `engine_tests` depends). `LIST=1 | wc -l`: **130 before, 134 after**; per gate 2 / 2 / 2 / 2 for
  `drafter_op_shapes`, `drafter_ops` and the two tiled gates; 4 entries carry `RICURSIVE_TEST_ATTENTION=tiled`. Self-test counts raised by exactly those (130 -> 134 PASS rows and wrapper
  calls, 131 -> 135 logs, 97 -> 101 NOT-RUN after a refused lock; 33 LOCKED unchanged: the tiled gates depend on `kernel_ops`); the fake test binaries now see `RICURSIVE_TEST_ATTENTION`
  and one check was added: a failure of every tiled invocation fails exactly the two tiled gates and leaves `drafter_ops` and `engine_tests` PASS with nothing NOT-RUN. **22 checks, all pass**
  (`perf/stage2cfix-selftest.log`; 21 before).
- **The bench**: `--attention default/tiled` runs the whole table per kernel in one process (smoke on the host ops with the tiny drafter: both tables print, allocated equals claimed
  bytes, 9.1 MiB default and 9.4 MiB tiled; `perf/stage2c-bench-host-smoke.log`). Not run on the GPU.
- **CPU suite**: `ctest -R '^ricursive\.' -L cpu` in `build-test`, started 17:02 on 9fe94f2 (the code of the three commits above; the later commits change the bench, these
  tests and documents): **32 of 32 passed, 0 failed, total test time 2181.62 s** (`ctest exit=0`; no `Failed` or `Timeout` line), including `plain_fingerprint` (5.55 s, golden hashes untouched),
  `drafter_ops_tiled` 15.41 s and `drafter_op_shapes_tiled` 80.21 s (`perf/stage2c-ctest-cpu.log`).

### Proof that the target's path and the existing kernel are unchanged

`git diff 5a54398 --stat` is empty for `hip/kernels.*`, `hip/kernels_math.hpp`, `hip/project.*`, `hip/executor.*`, `hip/embed.*`, `hip/profile.*`, `hip/ggml/`, `src/` and the tests
`plain_fingerprint`, `executor_contract`, `executor`, `engine`, `kernel_ops`, `attention_ops`, `projection_ops`. `Ops::DraftAttention`, its host implementation and its GPU launcher are
unchanged (a new method was added beside them), and the existing kernels' device assembly is identical (above). Lines removed outside the documents: the executor's one
`DraftAttention` call (now inside `if (tiled_) ... else` with the same arguments), the unknown-name message (now lists `tiled`), `kDraftAttentionKernels`, the `OpsSide` constructor and
attention call of the op-level test (same call when not tiled), its "only default" guard (now default or tiled), its header line, `EffectiveOptions`' kernel line (a check that names its
own kernel now keeps it, so the unknown-name check still runs under `RICURSIVE_TEST_ATTENTION=tiled`), the gate self-test counts, the bench's `--attention` parsing and help text. No
check was loosened or removed.

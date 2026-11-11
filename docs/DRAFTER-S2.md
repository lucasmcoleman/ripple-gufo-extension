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

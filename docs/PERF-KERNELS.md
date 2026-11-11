# PERF-KERNELS: attention and ffn_down for 1-row passes (job 1692, step 2)

Status 2026-11-11: **stopped before any measurement.** The GPU wrapper (`perf/gpu-run.sh`) refused three times in
a row (02:07:20 twice, 02:10:04 once; reason each time: a model, `gufo-npu/qwen3.8-flash-next`, was loaded in the
router). Per the brief, all GPU work stopped. No kernel change in this branch is measured, gated or kept.

## What is on this branch

- `tools/kernel_bench.cpp`, CMake target `ricursive_kernel_bench` (EXCLUDE_FROM_ALL, declared in `extension.json`):
  `attn KEYS [sliding|full]`, `proj TYPE M K`, `attn-suite`, `proj-suite`. Synthetic tensors of the real shapes
  (80 heads / 8 KV heads / 128, F16 caches, 16 rotating layer copies; Q4_K/Q5_K/Q6_K/Q8_0 weights, 3-6 rotating
  copies, <1 GiB), median per call over batches of 8 launches, and an FNV-1a hash of the output bits (equal hash =
  identical bits, the quickest parity check while iterating). It builds (`ninja -C build-test -j 3
  ricursive_kernel_bench`); it has **never run** (no GPU slot).

## Untested ideas (patch kept outside the repo: `perf/kernels-wip-untested.patch`, not committed)

Both are hypotheses read from the code, not measurements.

1. Attention: `AttentionKernel` launches `(tokens, kv_heads)` = 8 blocks of 128 threads for a 1-row pass, on a
   40-CU GPU, and walks tiles serially with a 16-step dependent K-load loop. Idea A: one block per query head
   (`grid.y = heads`) when `tokens <= 4`. Each head's arithmetic is independent of the others, so the output bits
   cannot change (hash check in the bench; chunk invariance and the existing gates stay valid). Idea B: unroll the
   K chunk loop and the V loop so loads are in flight together (same arithmetic order).
2. `ffn_down` (K = 40960): the single-row GEMV re-reads the 160 KB activation vector from L2 for every row
   (5120 rows x 160 KB = 840 MB of L2 traffic against 118 MB of weights); `ffn_gate`/`ffn_up` (K = 5120, 20 KB)
   stay in L1. The small-batch kernels already stage x in LDS and are documented as bit-equal to the GEMV;
   `LaunchKQuantSmallBatchDoubleStage` accepts batch 1 and is documented as suited to long K. Idea: route 1-row
   K-quant projections with long K to it. Same per-lane order, so bits should not change (bench hash, the
   projection gate's "rows 2..16 bit-equal to the single row" check).

## Numbers

None measured. Baselines from the profile run remain those in `docs/PERF-PROFILE.md`.
GPU time used by this task: one 20 s `true` probe; nothing else ran.

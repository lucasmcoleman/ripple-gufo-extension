# PERF-KERNELS: attention and ffn_down for 1-row passes (job 1692, step 2)

Branch `perf-kernels` (from `perf-profile` d698a61). Measured 2026-11-11 02:13-02:35 on the shared GPU through
`perf/gpu-run.sh`. Earlier the wrapper refused three times (a model loaded by the coordinator); the coordinator
reset the counter. Raw outputs: `perf/gpu-out/kernels-bench/`, `perf/gpu-out/kernels-unroll16/` (final),
`perf/gpu-out/kernels-final/` (intermediate, unroll 8).

## Changes (all bit-identical to the old kernels)

1. **Attention, <= 4 rows**: grid `(tokens, heads / HPB)` with HPB = 2 query heads per block (was a whole KV group,
   10 heads, so only 8 blocks for a 1-row pass). Rows > 4 keep the whole-group block (prefill unchanged). Each
   head's arithmetic is a separate set of registers/LDS rows, so the per-element operation order is unchanged.
   Then the K chunk loop is fully unrolled and the V loop unrolled 16 so loads are in flight together (same
   operation order). Bench output hashes equal the old kernel's for all four cases.
2. **ffn_down**: 1-row Q4_K/Q5_K projections with K >= 16384 call `LaunchKQuantSmallBatchDoubleStage` at batch 1
   instead of the GEMV. That kernel stages the activation vector in LDS (the GEMV re-reads 160 KB per row from L2)
   and is documented and tested bit-equal to the GEMV (same per-lane sub-block order, same butterfly); bench
   hashes are equal for every shape. Q6_K was slower there (757 -> 823 us) and stays on the GEMV.

Arithmetic order per output element: unchanged in both. The real-model logits files are byte-identical (`cmp`)
to the baseline, so the largest absolute logit difference is 0.

## Micro-benchmark (`ricursive_kernel_bench`, median per call, rotating weight/cache copies)

| case | before | after | |
|---|---|---|---|
| attn sliding 540 keys | 147.8 us | 41.0 us | 3.6x |
| attn sliding 1024 keys | 299.5 us | 69.5 us | 4.3x |
| attn full 540 keys | 146.0 us | 39.0 us | 3.7x |
| attn full 4100 keys | 1281 us | 285.9 us | 4.5x |
| Q4_K 5120x40960 | 664.6 us (177 GB/s) | 529.3 us (223 GB/s) | +26% |
| Q5_K 5120x40960 | 825.1 us (175 GB/s) | 676.3 us (213 GB/s) | +22% |
| Q6_K 5120x40960 | 757 us | 757 us (unchanged route) | |
| other projections | | unchanged (same code path) | |

Steps measured: HPB 10 / 1 / 2 / 5 (no unroll) at 4100 keys: 1281 / 652 / 650 / 678 us; with K unroll and V unroll 8:
HPB 1 426, HPB 2 394; V unroll 16: 286; V unroll 32: 639 (worse, dropped). `ffn_down` variants: GEMV 664.6,
batched-FP32 route 735.6 (slower, dropped), double-stage 529.3 us.

## Real model (Ripple-65B Q4_K_M, ms per generated token, profile on, 64 tokens)

| stage | ~540 ctx before | after | ~4,100 ctx before | after |
|---|---|---|---|---|
| sliding attention (64 layers) | 12.53 | 3.63 | 22.49 | 5.80 |
| global attention (16 layers) | 3.12 | 0.91 | 21.97 | 5.27 |
| sliding proj_ffn_down | 47.28 (173 GB/s) | 38.51 (212) | 47.23 | 38.56 (212) |
| global proj_ffn_down | 11.81 | 9.63 | 11.80 | 9.64 |

Targets: attention at 4,100 sliding 5.80 <= 7.5 and global 5.27 <= 7.5; at 540 total 4.53 <= 5; ffn_down
212 GB/s >= 205. Tokens/s with the profile off: ~540 ctx 4.49 -> 4.96 (222.7 -> 201.7 ms/token); ~4,100 ctx
3.99 -> 4.80 (250.9 -> 208.4 ms/token). llama.cpp is about 193 ms. Prefill speed unchanged (113-117 tok/s at 4096).

## Verification

- Real model, both prompts, profile on and off: `generated:` ids equal the baseline; logits files byte-identical.
- Full GPU gate set after each kept change: 9 gates, 44 invocations PASS (also after the final unroll change).
- CPU suite `ctest -R '^ricursive\.' -L cpu`: 20/20 PASS (run after the projection and first attention change; the
  final commit only changes an unroll factor in the GPU kernel).
- No tolerance changed, no test removed or skipped.
- GPU time (perf/gpu-use.log, 14 runs): about 5.7 minutes.

## Not done / next

Target 3 (proj_q, proj_gate, proj_k, proj_v) not started. Remaining gap to llama.cpp is mostly the rest of the
projections (proj_q/gate 186 GB/s, proj_k/v 148 GB/s) and `unattributed`/launch time.

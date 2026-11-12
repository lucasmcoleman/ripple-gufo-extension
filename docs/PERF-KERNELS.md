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

## Follow-up (review items A-D, then target 3)

A. `projection_ops` has 12 new cases `route_{q4k,q5k}_{47,5123}x{16640,16384,16128}` (in `--list`, the ctest list and
`run-gpu-gates.sh`; the gate script now has 56 invocations and its selftest counts were updated, nothing loosened).
Per case: 1-row `Project` memcmp-equal to the plain GEMV for 16 different input rows, rows 2..16 bit-equal to the 1-row
results, output guard words intact, input pointers misaligned by 4/8/12 bytes bit-equal. All 12 PASS.
B. `Project` checks `x % 16 == 0` before the staged route and falls back to the GEMV. Negative control: with the
check removed the misaligned cases still passed on this gfx1151 (unaligned global loads work), so the test shows the
fallback is bit-equal but does not prove the check is needed; it is defensive.
C. Dead 5-head instantiation and the repeated group check removed, two stale comments fixed; bench hashes and times
unchanged (4100 keys 286.9 us, 1024 window 69.7 us).
D. Full gates after each step: 9 gates, 56 invocations PASS; CPU suite 20/20; profile-off real-model rerun after
A-C: ids and logits identical, 201.2 ms/token (540) and 208.2 ms/token (4,100).

### Target 3

Micro-benchmark (cold weights: >= 192 MB of distinct copies per case). 27 geometries of the staged kernel at batch 1
(waves 4/8/16 x rows 2/3/4 x tiles 1/2/4) on Q4_K 10240x5120: base GEMV 147.2 us; the earlier route (16,3,2) 136-137;
best (4,2,4) 132.1 us (+11.5 %); (4,3,4) 132.7-135. (4,2,4) also wins elsewhere (all hashes identical):
Q4_K 40960x5120 580 -> 512 us, Q5_K 40960x5120 657 -> 628, Q4_K 5120x40960 529 -> 525, Q5_K 5120x40960 676 -> 638,
Q5_K 10240x5120 170 -> 161 (+5 %, kept because it is the same rule), Q4_K 5120x5120 77 -> 70, 1024x5120 20 -> 17.7.
The rule is now: every 1-row Q4_K/Q5_K projection with an aligned input uses (4,2,4); a function of type only.
Q6_K spills in this kernel and stays on the GEMV. Caveat: inside the full `proj-suite` run the same case measured
144.7 us (standalone 132 us); allocation order matters for this kernel, the real-model numbers below are what counts.

`proj_k`/`proj_v` (Q8_0 1024x5120): a one-row-per-warp Q8_0 GEMV with the 2-row kernel's arithmetic (2/4/8 waves per
block) gave 27.6-27.8 us against 27.8 us for the existing kernel (identical hashes): nothing to gain, **dropped, not
kept**. In isolation the kernel already runs at 200 GB/s; the 142-149 GB/s of the profile is the per-stage
event/sync overhead on a 38 us stage (~5.6 MB), so 190 GB/s is not reachable by a kernel change.

Real model (stage ms per token, GB/s; profile on; before = after the attention/ffn_down work):

| stage | before | after |
|---|---|---|
| proj_q / proj_gate sliding (64 layers) | 10.19 / 10.22 ms (185 GB/s) | 9.15 / 9.17 ms (206 GB/s) |
| proj_ffn_gate / up sliding | 39.2 ms (208.6) | 36.3 ms (225.3) |
| proj_ffn_down sliding | 38.5 ms (212.4) | 37.4 ms (218.8) |
| proj_k / proj_v sliding | 2.45 / 2.40 ms (146 / 149) | 2.50 / 2.39 ms (142 / 149), unchanged |

Tokens/s with the profile off: ~540 context 4.96 -> 5.26 (190.3 ms/token), ~4,100 context 4.80 -> 5.08 (196.8 ms/token);
llama.cpp is about 193 ms. Generated ids and logits files identical to the original baseline (`cmp`), both prompts,
profile on and off. Gates 56/56, CPU 20/20.

## GPU time

`perf/gpu-use.log` at the end of this work: 33 wrapped runs, 11.4 minutes, plus 3 refusals (the coordinator's own
model load). The gate script's lock-wrapped runs are not in that log (about 2 minutes per full set, 4 sets).

## Round 3: review items F, then long prompts (target 4)

F1. 8 small-shape route cases `route_{q4k,q5k}_{1,7}x{256,512}` (RunRoute; gate list 56 -> 64, self-test counts raised by
exactly 8 invocations and the dependent sums). F2. The stage-count comment is corrected (128 sub-blocks = 2048 inputs per
stage: K = 16384 is 8 stages, 16640 is 8 plus a 16-sub-block stage, 16128 is 7 plus a 112-sub-block stage). F3. `Project`
checks 16-byte alignment on the 2..16-row paths too (and on the `exact` path) and falls back to row-by-row plain GEMV
(each batched row equals the GEMV bit for bit); every route case now also runs 4 rows at +4 bytes, normal and exact, and compares with the aligned
1-row results (as with B, the misaligned case also passes without the check on this hardware, so it proves the fallback
is bit-equal, not that the check is needed). Rows > 16 (Q8_1 activation quantisation) were not touched. Gates 64/64, CPU 20/20.

### qknorm_rope_kv (kept)

Cause: `QkNormRopeAppendKernel` launched one block per (row, head) and every one of the 80 + 8 head blocks of a row
recomputed the RoPE angle in double precision (`pow`, `cos`, `sin`, 2 threads per pair), about 23 million double
transcendental evaluations per 2048-row call. The cos/sin depend on (position, pair) only. For more than 4 rows a block
now serves all 88 "heads" of one row (loop over heads; the angle is evaluated once per thread per row); for <= 4 rows
the old one-head-per-block layout stays (decode is unchanged). The value each element receives is identical.
Micro-benchmark (q restore subtracted, hashes over q, K cache and V cache identical): 2048 rows 24,985 -> 1,300 us (19x),
510 rows 6,109 -> 216 us (28x). Real model, `prompt` bucket, ms per pass: sw.qknorm_rope_kv 1906 -> 89.6 (2048 rows) and
436.7 -> 24.4 (510 rows); gl 104 -> 17. 

Prompt speed with the profile off: 510 tokens 152.7 -> 171.2 tokens/s; 4096 tokens 113.4 (116.8 in the first baseline
file) -> 128.8 tokens/s (llama.cpp reference 139 / 118). Generation with the profile off unchanged: 5.263 tokens/s
(190.0 ms) at ~540 and 5.081 (196.8 ms) at ~4,100. Ids and logits files identical to `off-510.*` / `off-4096.*` (and
profile-on files), gates 64/64 PASS, CPU 20/20.

### Multi-row attention (nothing kept)

Bench (`ricursive_kernel_bench rows-suite`, us): sliding 510 rows 2,019; 2048 rows 25,133 (start 0) and 28,774 (start 2048);
global 510 rows 2,023; 2048 rows 48,630 (start 0) and 204,324 (start 2048, up to 4,096 keys). Findings, all with identical hashes:
- Heads per block (10 / 5 / 2): time scales inversely with heads per block (5 heads = 2x, 2 heads = 5x slower), so the
  kernel's cost is per block and tile (loads, syncs, reductions), not per product; 10 stays best. Occupancy is already 10
  waves/SIMD, no spills (kernel-resource-usage remark).
- 16-byte LDS reads of q and of the probabilities (4 products per LDS read): global 2048 rows 42.7 / 181 ms (-12 %), sliding
  -7 % to +6 %: below the 15 % bar, dropped.
- K tile staged into LDS with coalesced loads (row stride 17 chunks, 44 KB LDS per block): 2x SLOWER (sliding 2048 rows
  54.6 ms, global 232 ms at start 2048); dropped.
Not tried (would change bits or needs a rewrite): matrix-unit (WMMA) scoring, sharing a K/V tile between several rows
(global layers only: sliding-window tile starts differ per row), splitting the key range across blocks. The attention
stage at 2048 rows is about 5 % of the FP32 peak; a rewrite is the next step if more prompt speed is wanted. The
sw.attention stage in the real-model runs moved 1739 -> 1918 ms per pass with unchanged code (run-to-run variation).

GPU time at the end of round 3: see `perf/gpu-use.log` (46 wrapped runs, 20.2 minutes, plus the gate script's own lock time); no
refusals in round 3.

## 9 to 16 rows in one weight pass: measured, not adopted (job 1692, 2026-11-12, branch `drafter-s2e`)

Question: `Project` sends 9..16 rows to `LaunchBatchedQuantGEMMFp32`, which reads the weights in 8-row passes. Would
`LaunchKQuantSmallBatchDoubleStage` (16 waves x 3 rows, one tile per stage past 8 rows; vendored, never dispatched above
one row) be faster in one pass? Bench mode `ricursive_kernel_bench gemm16 [REPS]` (`tools/kernel_bench.cpp`; exit 4 on a
difference): every distinct K-quant (type, shape) the DFlash2 Q4_K_M drafter (GGUF header) and the target's verify
multiply, synthetic weights (random bytes, fp16 scales 2^-9..2^-6), rotating >= 192 MB of weight copies, split and one
pass alternating in one process (order swapped each repetition), each call synchronised, after a warm-up over every copy.
Logs: `perf/gemm16-20261112/02-gemm16-before.log` (40 reps) and `03-gemm16-repeat.log` (60 reps).

Bit-equality: 14 (type, shape) x rows 9, 12, 15, 16 x 3 seeds = 168 cases, outputs `memcmp`-equal in all 168
(activations with zeros, denormals, values up to 1e30, an all-zero row, an all-denormal row; outputs poisoned with two
different patterns).

Speed, rows 16 (head also 15), median us, run 1 / run 2 (positive = one pass faster):

| type | shape (M x K) | split | one pass | one pass vs split |
|---|---|---|---|---|
| Q4_K | 1024 x 5120 | 74.1 / 74.0 | 59.1 / 59.3 | +20.2 / +19.9 % |
| Q4_K | 10240 x 5120 | 388.6 / 407.5 | 378.3 / 390.3 | +2.6 / +4.2 % |
| Q4_K | 256 x 5120 | 52.5 / 53.2 | 42.6 / 42.3 | +18.9 / +20.5 % |
| Q4_K | 2560 x 5120 | 105.9 / 105.1 | 120.0 / 119.9 | -13.3 / -14.1 % |
| Q4_K | 40960 x 5120 | 1553.9 / 1548.6 | 1377.9 / 1364.0 | +11.3 / +11.9 % |
| Q4_K | 5120 x 10240 | 502.1 / 452.5 | 380.1 / 362.9 | +24.3 / +19.8 % |
| Q4_K | 5120 x 25600 | 1264.8 / 1247.1 | 872.1 / 852.2 | +31.0 / +31.7 % |
| Q4_K | 5120 x 40960 | 1958.4 / 1977.6 | 1393.4 / 1416.8 | +28.8 / +28.4 % |
| Q6_K | 1024 x 5120 | 58.3 / 58.6 | 69.8 / 69.7 | -19.7 / -19.0 % |
| Q6_K | 5120 x 40960 | 1995.6 / 1935.7 | 2081.7 / 2020.5 | -4.3 / -4.4 % |
| Q6_K | 5120 x 10240 | 492.7 / 488.9 | 498.6 / 485.1 | -1.2 / +0.8 % |
| Q6_K | 100352 x 5120, 15 rows | 4189.3 / 4046.6 | 3878.7 / 3582.6 | +7.4 / +11.5 % |
| Q6_K | 100352 x 5120, 16 rows | 4200.6 / 4061.8 | 3908.8 / 3665.2 | +6.9 / +9.8 % |
| Q5_K | 40960 x 5120 | 1574.0 / 1512.1 | 1615.3 / 1538.2 | -2.6 / -1.7 % |
| Q5_K | 5120 x 40960 | 1619.5 / 1610.7 | 1871.1 / 1804.0 | -15.5 / -12.0 % |

Rule fixed before measuring: a type moves to one pass only if every bit-equality case passes AND its median is at least
5 % faster on every shape of that type the models use. Bit-equality holds for all three types; speed fails for each:
Q4_K on 2560 x 5120 (-13 %) and 10240 x 5120 (+3 to +4 %), Q6_K on 1024 x 5120, 5120 x 40960 and 5120 x 10240, Q5_K on
both shapes. **Nothing changed in `Project`.**

Why the expected gain is not there: on the shapes of 5120 x 10240 and larger the 16-row one pass streams the weights at only 75..118 GB/s
(40960 x 5120 Q4_K: 16 rows x 2 FLOP x 210 M weights in 1.37 ms = 4.9 TFLOP/s), so at 16 rows both routes are limited by the kernel's FP32 dequantise-and-FMA
work, not by reading the weights twice. The split route's times match the real-model profile (gate/up 1.55 ms x 10 =
15.5 ms against 14.90 measured; down 9.9 against 10.22), so the bench reproduces the stage. Even switching only the shapes
that win would save about 4.2 ms of the block (3 Q4_K + 2 Q6_K layers, from these deltas) plus 0.3..0.5 ms of head, not
the 10..19 ms estimated from the traffic.

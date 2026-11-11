# Drafter stage 1: the DFlash2 drafter on the CPU, checked against the Python oracle

Stage 1 of the drafter design (`/server/staging/ripple-gufo/perf/DRAFTER-GUFO-DESIGN.md` section 6, job 1692). CPU only: no
GPU back end, no lock, no gate script was used. Nothing in the engine, the runner or the executors changed; with no drafter
configured none of this code runs (it is not referenced by any production path). Stage 0 (`DRAFTER-STAGE0.md`) settled the
recipe; this stage implements it in C++ and proves it equals the oracle.

## Files

| file | what |
|---|---|
| `src/drafter_config.{hpp,cpp}` | `ParseDrafterConfig(reader, target_config, &error)`: the `dflash.*` metadata, validated; refuses a mismatch with the target |
| `src/drafter_weights.{hpp,cpp}` | `ResolveDrafterWeights(reader, config, target_config, &error)`: the 81 tensors as `TensorRef` views, shapes and types checked |
| `src/drafter_reference.{hpp,cpp}` | `DrafterReference`: scalar CPU implementation, `AppendContext` (target features of new positions) and `DraftStep` (one block) |
| `tests/drafter_cpu_test.cpp` | ctest `ricursive.drafter_cpu` (label `cpu`): tiny fixtures against the oracle, four deliberate-defect twins; `real` mode for the real weights |
| `tools/drafter_oracle.py` (`stages`) | extension of the oracle: named stages of variant R, one anchor per forward through the upstream modules' hooks |
| `tools/drafter_stage1_fixtures.sh` | regenerates the oracle side of the tiny fixtures |
| `tests/fixtures_text/make_drafter_fixture.py` | existing from stage 0; now `--quant q8_0|q4_k` and `--taps N` |

`drafter_config` and `drafter_weights` are in `gufo_ricursive`; `drafter_reference` is in `gufo_ricursive_reference` (validation
targets, built with `-O2` and OpenMP like `reference.cpp`). The matrix product and dot product repeat the 25 lines of
`reference.cpp` (they live in its anonymous namespace; that file is untouched).

## GGUF keys (arch `dflash`, read by `ParseDrafterConfig`)

`dflash.` + `block_count` (5), `context_length`, `embedding_length` (5120, must equal the target's), `feed_forward_length` (40960),
`attention.head_count` (80), `attention.head_count_kv` (8), `attention.key_length`/`value_length` (128, equal),
`attention.causal` (must exist and be false), `rope.freq_base` (10000), `attention.layer_norm_rms_epsilon`, `block_size` (16),
`conv_kernel_size` (4), `conv_group_size` (16), `selector_rank` (256), `selector_top_k` (32), `target_layers`
([2,21,40,59,78], stored = target block index + 1; the reader decodes it to taps [1,20,39,58,77] = zero-based target blocks
whose output, before the final norm, is the feature), `attention.sliding_window` (4096),
`attention.sliding_window_pattern` and `attention.rope_pattern` (per layer, [T,T,T,T,F]: the last layer is full attention
and has no rotary), `attention.context_window` (16384, 0 = all rows), and `tokenizer.ggml.mask_token_id` (100278,
must be inside the target's vocabulary). Arrays may be bool or integer arrays of any width (gguf-py writes a list of
Python ints as int32). Also checked: every tap is inside the target's blocks, 1 to 5 distinct taps (the executor's capture
limit), `block_size >= conv_kernel_size - 1` (the pinned convolution cannot run shorter blocks) and the tensor count
(6 + 15 per layer, so an unused tensor is refused). Tensors: `fc.weight` [25600 -> 5120], `enc.output_norm.weight`,
`output_norm.weight`, `selector_{hidden,predecessor,successor}.weight`, and per layer `attn_{norm,q,k,v,output,q_norm,k_norm}`,
`ffn_{norm,gate,up,down}`, `attn_conv_{base,proj}`, `ffn_conv_{base,proj}`. Matrices: F32, BF16, Q8_0, Q4_K, Q5_K,
Q6_K or IQ4_XS (the real Q8_0 and Q4_K_M files use Q8_0 and Q4_K for every matrix and the codebooks); norms and the
convolution bases (`[2 sides][4 taps][5120]`, GGUF dims 5120x4x2) are F32.

## The forward pass as implemented

Oracle references: `ZL` = the pinned upstream file (`ref/dflash2/zlab-07ebd93-dflash_model.py`, sha256 `f55b7fe0...651e`),
`OR` = `tools/drafter_oracle.py`. The C++ follows ZL's classes; OR only binds weights and records stages.

Per anchor (the committed token at position `s`; the bonus token of the last verify): `DraftStep(bonus, s)`.

1. **Context** (`AppendContext`, once per position, cached): `ctx = RMSNorm(fc(features), enc.output_norm)` with the five
   tapped layer outputs concatenated (ZL `DFlashDraftModel.forward` 573-598, OR `stage_ctx` 473). Per layer
   `k = k_norm(k_proj(ctx))`, rotary at the row's absolute position unless the layer has no rotary, `v = v_proj(ctx)`
   (ZL `Qwen3DFlashAttention.forward` 369-422; OR `build_ctx` 428). Stages `ctx_h`, `ctx_k.L`, `ctx_v.L`.
2. **Block**: 16 rows `[bonus, MASK x 15]` at positions `s .. s+15`, the target's token-embedding rows, **not scaled**
   (OR `embed` 417, `draft_block` 454). Stage `emb`.
3. **Each of the 5 layers** (ZL `Qwen3DFlashDecoderLayer.forward` 433-476; OR hooks in `stage_block` 486):
   - `a = RMSNorm(h, attn_norm)` (`attn_norm.L`); `dyn = a @ attn_conv_proj^T` ([16][2*4*320], index `side*K*G + tap*G + group`);
     `x0 = conv(a, base side 0, dyn side 0)` (`attn_conv_in.L`). Convolution (ZL `_grouped_dynamic_convolve` 478-490): within
     the 16 block rows only, `out[t][c] = sum_{o<4} (base[o][c] + dyn[t][o][c/16]) * x[t-o][c]`, rows before the block start
     are zero; accumulated per tap as `out += base*x; out += dyn*x`.
   - `q = q_norm(Wq x0)` (per head), `k, v` from `x0`; rotary on q and the block's k at the block positions, except on the
     layer whose `rope_pattern` entry is false; attention over `[context rows ; block rows]`, GQA 10, scale `1/sqrt(128)`,
     **non-causal**, softmax in double (as `reference.cpp`); sliding layers see keys with `|q_pos - k_pos| < sliding_window`
     (ZL `_attention_mask` 157-171), the full layer sees everything; the context rows are the last `context_window` rows
     below `s` for every layer (OR `draft_block` 458-459). `attn_o.L` = `Wo @ attention`.
   - `h += conv(attn_o, base side 1, dyn side 1)` with the same `dyn` (side 1 coefficients of the same projection). `h_attn.L`.
   - `f = RMSNorm(h, ffn_norm)` (`ffn_norm.L`); a second dynamic convolution (`ffn_conv_*`): `x1 = conv(f, side 0)`
     (`mlp_conv_in.L`); `mlp = down(silu(gate(x1)) * up(x1))`, no clamp (`mlp_o.L`); `h += conv(mlp, side 1)`. `h.L`.
4. **Final**: `RMSNorm(h, output_norm)` (`final`); logits of rows 1..15 with the **target's** output head (multiplier 1, no
   softcap): `logits` (ZL `compute_logits` 599-605).
5. **Selector** (ZL `CandidateSelector.select` 524-547): per row the top-32 logits (the candidates are reported sorted by
   token id), `hproj = selector_hidden(final)` (`hproj`); walk from `prev = bonus`: `score_k = logit_k + (pred_cb[prev] * hproj) . succ_cb[cand_k]`,
   argmax, `prev = that candidate`. `Draft.tokens` are the 15 choices, `candidates` and `scores` are `[15][32]`.
   `Draft.confidence` = `softmax(scores)[chosen]` (temperature 1). The oracle's greedy path has no confidence, so it is
   checked against the oracle's *scores* (the `conf` record is computed from them in the oracle); nothing consumes it yet.

Rotary: NeoX half-split, frequencies computed in float32 exactly as torch does (`inv = 1/base^(2i/D)`, `angle = float(pos)*inv`,
`cosf/sinf`); using double here would differ from the oracle by about `pos * 6e-8` rad.

## Tests

`ctest -R '^ricursive\.drafter_cpu$'` (about 3 s). Fixtures: `/server/staging/ripple-gufo/perf/stage1/tiny/` (override via the
CMake arguments): `drafter-q8_0.gguf` and `drafter-q4_k.gguf` (random tiny drafters: 5 layers, 8 heads / 2 KV, head 32,
FF 256, rank 16, top-k 8, sliding window 8, context window 64, 4 taps = target layers 0..3, written by
`make_drafter_fixture.py <tinyB.gguf> OUT --quant q8_0|q4_k --taps 4`; the Q4_K blocks are random bytes with a zero-mean scale
layout), `dump/` (ids and the layer 0..3 outputs of `perf/stage0-tiny/cpu-dump`, sliced), `stages-*.bin` (oracle output).
Regenerate the oracle side with `tools/drafter_stage1_fixtures.sh` (through `drafter_oracle.sh` only).

The features the C++ drafter reads are **captured by the CPU executor** with `capture_layers` = `DrafterReference::TapLayers()`
(the configuration's decoded layers), so the stored-plus-one decoding is tested through the real capture path; they equal
the dump the oracle read bit for bit (max abs 0).

## Results (2026-11-11)

**Tiny fixtures, C++ against the oracle** (`/server/staging/ripple-gufo/perf/stage1/tiny/test-output.txt`). 57 named stages
(ctx_h, ctx_k/ctx_v x5, and per layer attn_norm, attn_conv_in, attn_o, h_attn, ffn_norm, mlp_conv_in, mlp_o, h; emb, final,
hproj, logits, score, conf), compared as relative RMSE `|got - ref|_2 / |ref|_2`, worst over all anchors, tolerance 1e-5:

| fixture | anchors | worst stage error | draft tokens | candidate sets |
|---|---|---|---|---|
| Q8_0, positions 8..86 (79 anchors; contexts of 8..86 rows, past the 64-row context window and the 8-key sliding window) | 79 | 2.4e-6 (`conf`; every other stage <= 1.5e-6) | 79 of 79 identical | 79 of 79 identical |
| Q4_K, positions 60..75 | 16 | 1.9e-6 | 16 of 16 | 16 of 16 |

**Twins**, the same comparison with `DrafterFaults` switched on (each must fail; all do, by orders of magnitude):

| defect | draft tokens identical | worst stage | first failing stages |
|---|---|---|---|
| rotary on the no-rotary layer | 2 of 79 | ctx_k.4 0.94 | attn_o.4, ctx_k.4, ffn_norm.4, final, h.4 |
| convolution taps shifted by one | 0 of 79 | mlp_o.2 1.13 | attn_conv_in.0..4, attn_norm.1 |
| selector predecessor = candidate's own row | 0 of 79 | score 0.92 | score, conf only (every hidden stage is unaffected, so the score/token checks carry this twin) |
| tap layer one too late (capture of layers 1..4 instead of 0..3) | 0 of 79 | score 0.82 | attn_conv_in.1.., attn_norm.1.. |

None of the twins had to be strengthened.

**Real weights** (Q8_0 and Q4_K_M DFlash2 files, target `Ripple-65B-Preview-AD-Q4_K_M.gguf`, stage 0 GPU dumps, oracle variant R;
`free -g` showed 113-114 GB available before each run). Tolerances: stage 1e-4, logits max abs 1e-3, tokens identical:

| drafter | dump | anchors | worst stage (except `conf`) | `conf` | logits max abs | 15 tokens identical | scalar time per anchor |
|---|---|---|---|---|---|---|---|
| Q8_0 | p07-B-sky (136 positions) | 12 (26..130) | 2.8e-6 | 1.1e-5 | 1.3e-4 | 12 of 12 | 0.8 s (16 threads) |
| Q8_0 | p01-A-build-510 (766 positions) | 12 (100..700) | 5.0e-6 | 1.1e-5 | 4.3e-4 | 12 of 12 | 1.0 s (16 threads), 1.4 s (4 threads) |
| Q4_K_M | p07-B-sky | 7 | 3.0e-6 | 6.3e-6 | 1.3e-4 | 7 of 7 | 1.4 s (16 threads) |

24 Q8_0 anchors and 7 Q4_K_M anchors, all 31 give the same 15 tokens as the oracle. The whole p01 run (context of 700 rows plus 12
anchors) takes 11 s on 16 threads. Commands: `perf/stage1/real/oracle-stages.sh` (the oracle side, ~15 s per file after a ~15 s
load, 20 GB RAM) and `ricursive_drafter_cpu_test real DRAFTER.gguf TARGET.gguf DUMP_DIR STAGES.bin [max_seconds]`; logs
`perf/stage1/real/cpp-*.log`.

**Candidate sets on real weights.** On 30 of 31 anchors the 32 candidates per row are identical to the oracle's. At p07 anchor 130
(Q4_K_M) one row lists a different 32nd candidate: the 32nd and 33rd logits differ by less than the 1.3e-4 numeric noise of the
logits. The draft tokens were unaffected. The real-mode comparison therefore accepts a one-sided candidate when its logit is within
the logits tolerance (1e-3) of the oracle's 32nd largest, compares scores on the common candidates and compares `conf` (a
softmax over all 32) only on rows with identical sets. The tiny fixtures keep exact set equality. Stage 2 (the GPU drafter) should
expect this and not assert set identity.

**CPU suite**: `ctest -R '^ricursive\.' -L cpu` in `build-test`: 22 of 22 pass (the 21 of stage 0 plus `ricursive.drafter_cpu`),
`perf/stage1-ctest-cpu.log`.

## Where the design and the oracle disagree, and other findings

The oracle wins in each case. No stage of the design's section 2 contradicts variant R (57 stages agree to 1e-5 on the tiny
fixtures, 1e-4 on the real weights). What the design left open or got wrong:

1. **Fixture size**: the design says "hidden 128, 2 layers, top-k 8"; the tiny targets are 512 wide with 5 layers, and the fixture keeps
   5 draft layers (4 sliding, one full without rotary) because the NoPE twin needs that layer. The fixture drafter taps 4 target
   layers (0..3): the tiny target has only 5 layers, so a tap one too late (layers 1..4) must still be a layer of the target.
2. **Anchor count**: the design says 50 real anchors, the brief 20; 24 (Q8_0) plus 7 (Q4_K_M) were run.
3. **GGUF detail**: `dflash.target_layers` is an int32 array (gguf-py writes Python ints that way), not uint32; the reader accepts both
   widths. The design did not say the convolution projection output is `side * K * G + tap * G + group` (side major); the oracle fixes it.
4. **Confidence**: the design says confidence exists only in sampled mode (the softmax `q`). The brief wants a confidence with the
   greedy draft; it is defined here as the temperature-1 softmax of the 32 scores at the chosen candidate. The oracle has no
   such value for greedy (its `conf` record is computed from the oracle's scores), so it is validated only as a function of the scores.
5. **Candidate ties**: upstream takes the top-32 with `sorted=False`, so its argmax tie rule over candidates is "first in torch's
   internal order"; the C++ orders candidates by token id and resolves a score tie to the lowest id, and a logit tie at the 32nd place
   to the lowest id. No such tie occurred in any run; exact ties are the one behaviour the oracle does not define.
6. **Rotary numerics**: the design is silent on how the frequencies are rounded. The oracle uses float32 frequencies and float32
   `pos * inv`; the C++ does the same (a double-precision version would differ by about `pos * 6e-8` rad, i.e. up to 3e-4 at position 4096).
7. **Not covered by this stage**: the 16384-row context window and the 4096 sliding window on real weights (the real prompts have at
   most 766 positions) are only exercised on the tiny fixtures (windows 64 and 8); the batched oracle path used by stage 0's acceptance
   runs is not compared (`stages` runs one anchor per forward, the upstream mask path); tie behaviour (5); the drafter at other
   block sizes than 16.

## Regenerating

Tiny fixtures (`RICURSIVE_STAGE=/server/staging/ripple-gufo`; the generator imports `gguf` and numpy from the oracle venv, so it
is run with that interpreter in isolated mode, like the wrapper does):

```
V=/server/staging/ripple-gufo/venv-oracle/bin/python; T=$RICURSIVE_STAGE/perf/stage1/tiny
for q in q8_0 q4_k; do $V -I tests/fixtures_text/make_drafter_fixture.py $RICURSIVE_STAGE/fixtures/tinyB.gguf $T/drafter-$q.gguf --quant $q --taps 4; done
# $T/dump: ids.i32 + taps.f32 of layers 0..3 cut from perf/stage0-tiny/cpu-dump (ricursive_dump_taps --backend cpu), meta.json tap_layers [0,1,2,3]
tools/drafter_stage1_fixtures.sh        # the oracle stage files, through drafter_oracle.sh
```

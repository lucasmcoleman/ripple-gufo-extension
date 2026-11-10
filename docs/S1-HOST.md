# S1 host side: evidence (2026-11-10)

Branch `s1-host`. CPU only; no GPU, no model loaded beyond the 100-210 MB tiny fixtures. Logs: `/server/staging/ripple-gufo/s1-host/*.log`.

## Build
- `WITH_RICURSIVE=1 /server/programming/gufo-ripple/configure.sh /server/programming/gufo-ripple/build-test -DBUILD_TESTING=ON` (5 s).
- `source env.sh; nice -n 10 ninja -j 3 ricursive_config_test ricursive_weights_test ricursive_oracle_test ricursive_quant_test ricursive_incremental_test ricursive_ref_logits` (50 s wall: core was already partly cached by the 62-step graph). Target `ricursive_tests` builds all tests.
- `ctest --test-dir build-test -R ricursive`: 5/5 pass, 41.7 s total (OMP_NUM_THREADS=4 set by the test properties).
- Binaries: `build-test/extensions/build/ricursive/build/ricursive_*_test`, `ricursive_ref_logits`.

## Files
`src/config.*`, `src/weights.*`, `src/reference.*`, `tests/{config,weights,oracle,quant,incremental}_test.cpp` (+ helpers), `tools/ref_logits.cpp`, `extension.cmake`, `extension.json`, `THIRD_PARTY.md`.

## Reference vs llama.cpp, F32 fixtures (threshold max abs <= 2e-3 and top-1 100%)
tinyA seq0-3 max abs 1.91e-5, 1.76e-5, 2.00e-5, 2.12e-5; tinyB seq0-3 1.08e-5, 1.12e-5, 1.08e-5, 1.07e-5. Top-1 100% on all 8 (64 positions each). Mean KL 2e-12 to 6e-12. Also equals oracle/np_ref.py output to 2.1e-5 (CLI run on tinyA seq0).

## Negative controls (max abs diff over the 4 sequences; each must fail)
rope_full 3.93 / 2.47 (tinyA / tinyB); win_minus1 7.38 / 3.25; win_plus1 8.11 / 3.57; vres_on_v_layer0 11.9 / 6.99; vres_seed_prescale 7.64 / 4.42; sym_gate_clamp 0.710 / 0.115 (weakest); drop_post_attn_norm 10.7 / 7.51; drop_post_ffn_norm 4.99 / 5.12; gate_from_resid 5.54 / 3.43; swap_kv 17.6 / 12.6; non_causal 14.1 / 9.67; gqa_modulo 2.1e-5 (cannot fail) / 9.32.
- gqa_modulo cannot fail on tinyA: head_count_kv = 1, so h % 1 = 0 = h / 10. It fails on tinyB (kv = 2). The test asserts this structural reason.
- vres_on_v_equiv (lerp on V outputs of the same layer) is mathematically equal to the spec; the test requires it to PASS (2.4e-5 / 1.2e-5), as a positive twin.

## Incremental vs full (tolerance 1e-5; measured max abs 0, bitwise equal)
Chunkings on both fixtures: 1 at a time, 3, 7, 16, prefill 40 + 24 singles, two window-straddling plans, one chunk of 64. Cache sizes asserted (sliding layers = window + max_rewind rows, full layers = T). Taps: 173 / 88 named taps, chunked session equals full exactly.
Rewind (fed 52, rewound to W-3, W-1, W, W+2, 2W+1, 40, 51; different tokens fed; vs fresh full forward): 0 difference at every point on both fixtures; plus double rewind, rewind to 0, no-op rewind. Rewind beyond `max_rewind` and past the end throw.
Broken twins (all must differ): window-1 session 7.2 / 2.7; ring of W-1 rows 7.7 / 2.9; continuing without rewind 3.0 to 12.5.
Caveat: full and incremental share the layer code (projections, norms, RoPE, FFN) and AttendRow; they differ in storage (full arrays vs ring), visibility bounds and chunking. Bitwise equality is therefore expected; the independent check is the llama.cpp comparison.

## Quantized tinyB (reference on dequantized weights vs llama.cpp on the same file; report only)
| file | max abs | mean abs | mean KL | top-1 |
|---|---|---|---|---|
| Q4_K_M | 0.285 | 0.0464 | 1.54e-3 | 95.3% |
| Q5_K_M | 0.274 | 0.0461 | 1.55e-3 | 96.1% |
| Q6_K | 0.270 | 0.0465 | 1.51e-3 | 96.1% |
| Q8_0 | 0.197 | 0.0334 | 7.8e-4 | 97.7% |
All finite. (Means over the 4 sequences; per sequence in quant_test.log.) For scale, llama.cpp's own quantized logits differ from its F32 logits by mean 0.425 / 0.228 / 0.124 / 0.049 (Q4_K_M / Q5_K_M / Q6_K / Q8_0), so the dequant path tracks llama.cpp's quantized result much better than F32 does. The remaining gap does not shrink with bit width as expected for dequant error; the likely cause is llama.cpp's activation quantization (not isolated here: I did not emulate it).

## Weights test
Synthetic: all 7 accepted types resolve in every matrix role; 5 rejected types (F16, Q4_0, Q2_K, IQ3_S, Q3_K) fail per matrix role with tensor name and type; every norm / q_norm / k_norm / lambda role rejects BF16, F16, Q8_0; each missing tensor and each wrong shape reports its name.
Real: stand-ins `/server/staging/ripple-gufo/s1-host/standin-{plain,ad}-q4km.gguf` (sparse, 16 MB on disk) and the real `Ripple-65B-Preview-AD-Q4_K_M.gguf` (opened read-only, metadata and tensor infos only; test skips if absent): 1203/1203 tensors resolve, every file tensor claimed exactly once.
- plain Q4_K_M: F32 561, Q4_K 561, Q6_K 81.
- AD-Q4_K_M (stand-in and real file identical): F32 561, Q4_K 310, Q5_K 90, Q6_K 81, Q8_0 160, IQ4_XS 1. Per role: attn_q, attn_gate Q4_K x80; attn_k, attn_v Q8_0 x80; attn_output Q6_K x80; ffn_gate/up/down Q4_K 50 + Q5_K 30 each; token_embd IQ4_XS; output Q6_K; all norms, q/k norm, lambda F32.
- Broken twin: copy of each head with blk.7.attn_q type patched to Q2_K fails with "tensor blk.7.attn_q.weight has unsupported type Q2_K".

## Deviations / decisions
- `embedding_scale` and `swiglu_clamp` are REQUIRED here; llama.cpp treats both as optional (absent = no scale / no clamp). Strict on purpose: a silent missing scale ruins the model. Clamp 0 means no clamp (as llama.cpp and np_ref).
- `output.weight` is required (llama.cpp falls back to tied token_embd).
- Dot products use 8 float partial sums (not double); np_ref uses numpy float32. Result equals np_ref to 2e-5.
- Extra variants beyond np_ref: swap_kv, non_causal, gqa_modulo; np_ref's gate_from_attn_out is not implemented (not in the brief). vres_on_v_equiv added as a must-pass twin.
- Ring: capacity = window + max_rewind (constructor argument) with per-token write-then-attend, so no chunk-size term. `Rewind` throws beyond max_rewind from the highest position written.
- Taps are written by the CLI as 2-D .npy [rows][heads*dim] (np_ref writes 3-D for q/k/v).
- Test paths default to the staging tree (env overrides RICURSIVE_FIXTURES, RICURSIVE_ORACLE_OUT, RICURSIVE_S1_DIR, RICURSIVE_REAL_MODEL).
- The skeleton's `extension.cpp` still returns "not implemented"; no runner was in scope.
- `extension.json` was rewritten by a script (arrays expanded); `src/text/` will need its sources added to `extension.cmake` and targets to `extension.json` when merged.
- Public API: nothing missing. Used `GgufReader` (OpenFile/OpenMemory/FindTensor/metadata) and `quant::DequantizeRow/EncodedSizeBytes` through `gufo/ext/experimental/ggml_dequant.hpp`, linking only gufo_ext_runtime / gufo_ext_core.

## Not run / unverified
- tools/ref_logits was run on tinyA seq0 only (plain and win_minus1); taps readable by numpy (shape 64x256).
- No sanitizer build; no run under `ctest -j`; no test of the real file's tensor data (deliberately).
- Spec, np_ref.py and llama.cpp (master_ricursive.cpp) disagreed nowhere I found; the only differences are the optional-key behaviours listed above.

# S3: real weights vs llama.cpp (2026-11-10 night run)

Model: `Ripple-65B-Preview-AD-Q4_K_M.gguf` (40,426,079,712 bytes). Host: MasterServer (gfx1151, GTT 116 GiB).
gufo extension: `integration` at 138ff86. Reference: llama.cpp b12252 (a2c942d), HIP build, F16 KV.
Run: `STEP=all /server/staging/ripple-gufo/s3/run-s3.sh`, 23:01:11 to 00:03:38, exit 0.
Evidence: `/server/staging/ripple-gufo/s3/night-20261110/` (logits, score logs, smoke and speed JSON,
`status.tsv`, `window-split.txt`) and `s3/driver-20261110.log`.

Load during the run: a restic backup ran beside it (CPU only, up to one core). Correctness is unaffected;
the speed figures are measured under that load and have not been repeated on an idle machine.

## Logits: gufo vs llama.cpp

KL(llama.cpp || gufo) per position, pooled over 16,384 positions per set. Floor = llama.cpp against itself
at a second batch size (256 vs 16, and 2048 vs 16): llama.cpp HIP switches kernel path between 16 and 32
rows, so this is its own batch-to-batch noise on this model.

| set | pair | mean KL | p99 KL | max KL (text, position) | top-1 agreement |
|---|---|---|---|---|---|
| 64 x 256 | floor: llama.cpp b256 vs b16 | 6.5196e-04 | 5.6690e-03 | 0.840 (t34-nonenglish-de, 156) | 99.0356% |
| 64 x 256 | gufo rows 256 vs llama.cpp b256 | 5.9048e-04 | 5.8186e-03 | 0.187 (t07-prose, 68) | 99.0967% |
| 64 x 256 | gufo rows 16 vs llama.cpp b256 | 5.6428e-04 | 4.3485e-03 | 0.956 (t34-nonenglish-de, 156) | 99.0662% |
| 8 x 2048 | floor: llama.cpp b2048 vs b16 | 2.6831e-03 | 1.6909e-02 | 3.526 (t06-symbols-json, 602) | 97.8455% |
| 8 x 2048 | gufo rows 256 vs llama.cpp b2048 | 2.5191e-03 | 1.6889e-02 | 3.509 (t06-symbols-json, 602) | 97.8516% |
| 8 x 2048 | gufo rows 16 vs llama.cpp b2048 | 2.4154e-03 | 1.5036e-02 | 3.093 (t06-symbols-json, 602) | 97.9492% |

All four gufo verdicts: PASS, 0 outlier positions. The scorer's rule allows 10 x the floor; that slack was
not needed: every gufo aggregate (mean KL, p99 KL, top-1 disagreement) is between 0.77 and 1.03 times the
floor. Reading: gufo differs from
llama.cpp batch 256/2048 by as much as llama.cpp batch 16 does. A comparison against llama.cpp cannot show
more than that, because llama.cpp's own noise on this quantized model is this large (top-1 flips on 1 to 2%
of positions, single positions with KL above 0.8).

### Across the 1024-token window (8 x 2048 set, `tools/window_split.py`)

| positions | floor mean KL | gufo rows 256 mean KL | gufo rows 16 mean KL |
|---|---|---|---|
| 0-511 | 2.0126e-03 | 2.1612e-03 | 1.8782e-03 |
| 512-1023 | 3.6105e-03 | 3.3784e-03 | 3.2686e-03 |
| 1024-1535 | 2.8811e-03 | 2.4515e-03 | 2.5175e-03 |
| 1536-2047 | 2.2283e-03 | 2.0855e-03 | 1.9975e-03 |
| 1024-1047 (192 positions) | 3.0212e-03 | 4.0985e-03 | 2.5190e-03 |

gufo rows 256 / floor: 0.985 before the window, 0.888 after. No step at 1024. The 24 positions right after
the boundary are a small sample (192 positions) and sit within the spread of the other buckets.

## Servers: smoke

Same client and prompts for both (`s3/tools/chat_probe.py`, `/v1/chat/completions`, greedy).
llama.cpp: `serve.sh --port 4903` (`-c 32768 -ngl 99 -fa on`). gufo: `gufo serve llm -m MODEL
--served-model-name ripple-65b --port 4904` with `RICURSIVE_BACKEND=gpu`.

Both PASS all five checks: "Paris" with `stop`; "144" with `stop` and no literal end marker; greedy repeat
identical; `get_weather` tool call emitted with `{"city": "Paris"}`; tool result used in the final answer.

## Servers: speed (medians of 3 timed runs, warm-up dropped, 256 generated tokens)

| engine | prompt 510 tokens | prompt 4096 tokens | decode after 510 | decode after 4096 | load | GTT after | Tctl max |
|---|---|---|---|---|---|---|---|
| llama.cpp HIP | 117.5 t/s | 137.1 t/s | 5.32 t/s | 5.18 t/s | 17.2 s | 40.7 GiB | 73.9 C |
| gufo | 120.0 t/s | 109.8 t/s | 4.47 t/s | 4.01 t/s | 6.4 s | 42.5 GiB | 74.5 C |

Logits tools (no HTTP): llama.cpp 1.7 s per 256-token text and 19.2 s per 2048-token text (batch = text
length, flash attention off); gufo rows 256: 1.74 s and 16.0 s (128 tokens/s); gufo rows 16: 9.5 s and
78.4 s; llama.cpp batch 16: 38.3 s per 2048-token text.

gufo decode is 16% (short context) to 23% (4096 tokens) slower than llama.cpp, and its decode slows more
with context. Its 4096-token prompt speed is below its own 2048-token logits-tool speed and below llama.cpp.
Neither has been profiled. The kernels are the first working version (vendored quantized matmul, Laguna
attention with group 10); see `S2-KERNELS.md`.

## Not tested

- Speeds on an idle machine; contexts beyond 4096 tokens; multi-turn sessions with cache reuse past the
  window; the DFlash2 drafter; Vulkan (`ripple_logits_vulkan` was never started; HIP showed no fault).
- `gpu_logits` NaN/Inf branch (no input produced one).
- gufo through llama-swap (it ran from the work tree only).

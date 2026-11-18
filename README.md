# ripple-gufo-extension

An out-of-tree [gufo](https://github.com/gufo-org/gufo) extension, `ricursive`, that runs
[Ripple-65B-Preview](https://huggingface.co/ricursiveAI/Ripple-65B-Preview) GGUF files (architecture
`ricursive`), including drafted decoding with the published DFlash2 drafter.

Status: preview. Built and tested on one machine only: AMD Strix Halo (gfx1151), ROCm 7.2.2, against gufo main
`0e2de43` (extension API 0.2.25), with `Ripple-65B-Preview-AD-Q4_K_M.gguf` and the DFlash2 `Q4_K_M` drafter.

## Build

Configure gufo with this directory as an extension directory and add `ricursive` to the extension list. The
tested configuration (gufo's core extensions plus this one):

```sh
cmake -S gufo -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENGINE_ENABLE_HIP=ON \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DGPU_TARGETS=gfx1151 \
  -DGUFO_EXTENSION_DIRS=/path/to/ripple-gufo-extension \
  "-DGUFO_EXTENSIONS=deepseek_v4_flash;embedding_gemma2;minimax_h3;qwen;qwen38_flash_next;qwen3_asr;qwen3_tts;qwen_image_21;ricursive"
ninja -C build gufo
```

Build the `gufo` target (or named `ricursive_*` targets). The extension's `api_version` is `>= 0.2.0`; gufo
requires the minor version to match the host exactly, so a gufo with a new extension API minor needs a rebuild
and recheck.

## Run

```sh
RICURSIVE_BACKEND=gpu gufo serve llm -m Ripple-65B-Preview-AD-Q4_K_M.gguf -c 32768 -j 1 \
  --speculative dflash2 --dflash-model Ripple-65B-Preview-DFlash2-Q4_K_M.gguf
```

OpenAI-compatible `/v1/completions` and `/v1/chat/completions`. Without `--speculative dflash2` it decodes
one token at a time.

## What was checked

- Logits on the real weights match llama.cpp within llama.cpp's own batch-size noise (`docs/S3-RESULTS.md`).
- GPU gate tests on small fixtures: `tests/run-gpu-gates.sh`; CPU tests through ctest.
- Speed on the machine above (2026-11-17; 16 prompts, 128 tokens, greedy, three alternating runs): 10.5 tokens/s
  with the drafter, against 9.6 for llama.cpp with the same drafter; prompt processing about equal.

## Notes

`docs/` holds the design and stage notes as written during development. They refer to paths on the original
machine, and their commit ids predate a metadata-only rewrite before publication: `docs/COMMIT-MAP.txt` maps them.

## Licence

MIT (`LICENSE`). Portions adapted from the Gemma 4 gufo extension and from gufo itself are listed in
`THIRD_PARTY.md`, with the Gemma 4 extension's licence in `LICENSES/`.

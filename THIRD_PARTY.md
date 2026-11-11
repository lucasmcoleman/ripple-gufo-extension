# Third-party material

- `tests/gguf_builder.hpp`: in-memory GGUF v3 serializer. Its structure (Pod/Text/Align helpers, metadata
  variant visitor, tensor-info/offset layout) is derived from `tests/gguf_builder.hpp` of the Gemma 4 extension
  (franzmoca/gemma4-gufo-extension, commit a31ce42b13f57c53eed3fa5f08e5a20d880e3aba, MIT, Copyright Francesco Moca).
  Changed: namespace, `U32Array`/`F32Array` instead of `I32Array`/`StringArray`.
- `tests/check.hpp`: the `Require`/`Run` idea of the same extension's `tests/check.hpp`, with an added `RequireContains`.
- `extension.cmake`: the `_ricursive_test` function and the OpenMP/-O2 reference target follow the pattern of the
  same extension's `extension.cmake`.
- `src/reference.cpp`, `src/config.*`, `src/weights.*`: written for this repository; the layout (config / weights /
  scalar reference split) follows the Gemma 4 extension, no code copied.
- WP-D (engine), from the same Gemma 4 extension (a31ce42b, MIT, Copyright Francesco Moca); everything else in
  these files is written for this repository:
  - `src/continuation_adapter.cpp`: `BackingPlan` (page and block arithmetic) and `Covered` (coverage check) are
    copied nearly verbatim from `continuation_adapter.cpp`; the component list, the restore/validate/invalidate
    sequence and the slot state follow its structure, with a synchronous host copy instead of its device gathers,
    and no image, hidden-state or MTP parts.
  - `src/runner.cpp`: structure and several members follow `runner.cpp` (State, Descriptor, ResourceClaim,
    BuildConstraintVocabulary, SelectNext, the cache-resource hooks); the Gemma-specific parts (images, MTP, thought
    header, in-pass checkpoints) are dropped.
  - `src/drivers.cpp`: `PrintModelLoadTime`, `ComputeStats`, the benchmark table and loop, and the chat loop follow
    `drivers.cpp`.
  - `src/provider.cpp`: the registration call follows `provider.cpp`.

The MIT licence of the Gemma 4 extension requires its notice with copied portions:

    MIT License. Copyright (c) Francesco Moca. Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, subject to the conditions of the full licence text in
    /server/programming/gufo-ripple/ref-gemma4-ext/LICENSE (to be copied into LICENSES/ before release).

## Drafter stage 2: kernels adapted from gufo's Qwen DFlash2 drafter

Source: gufo (gufo-org/gufo), checkout `/server/programming/gufo-ripple/gufo`, commit 0e2de43, file
`src/families/qwen/kernels/dflash_kernels.hip` (MIT, Copyright (c) 2026 gufo contributors; the file has no
licence header of its own; gufo's `LICENSE` applies, and this repository's `LICENSE` is the same MIT licence with the
same copyright line; the source path, commit and licence are named in the header of `hip/drafter_kernels.hip.cpp`).
Read-only: the gufo
checkout was not touched. The adapted code is in `hip/drafter_kernels.hip.cpp`; every change (the code is also
described operation by operation in `docs/DRAFTER-S2.md`):

- `dflash_grouped_dynamic_conv_kernel` -> `DraftDynamicConvKernel`: one parameter `accumulate` (`output[i] = accumulate ?
  output[i] + sum : sum`, so that `h += conv(x)` needs no separate add); the body is otherwise the original (kernel size,
  group size and side are runtime parameters in both).
- `dflash_silu_mul_kernel` -> `DraftSiluMulKernel`: unchanged apart from names.
- `dflash_selector_partial_topk_kernel`, `MergeSelectorLists`, `InsertSelectorCandidate`, `FinalizeSelector` ->
  `DraftPartialTopKKernel`, `MergeLists`, `InsertCandidate`, `DraftFinalizeKernel`: `kMaxSelectorTopK` 16 -> 32 (the
  per-thread lists, the merge buffers and the transition lanes follow: 128 / 32 = 4 lanes per candidate instead of 8); only
  the greedy path is kept (no temperature, sampling, penalties, draft vocabularies, batch of sequences); the rows of one
  draft are the steps of the single block; the codebooks are read element by element through `hip/dequant_math.hpp`
  (F32, BF16, Q8_0, Q4_K) instead of BF16 only; the finalize step is one kernel body with the thread-zero work in an
  `if` (the vendored `FinalizeSelector` is a device function that returns early for the other threads); the candidate
  lists and scores are written out for the tests.
- `dflash_noncausal_attention_wave_kernel` / `_kernel` -> `DraftAttentionKernel` (**replaced, not adapted**): the
  vendored kernels keep one fp32 score per key in dynamic shared memory (the full layer's 16,400 keys need 65,600
  bytes, more than a workgroup may use), read the history from a flat fp32 array, and take the window from the first
  query. The replacement is the online-softmax tile loop of this repository's `AttentionKernel` (`hip/kernels.hip.cpp`,
  itself adapted from our Laguna port) over two key segments (the context ring, the block's own rows), generic head
  dimension, F16 or F32 ring, `DraftVisibleKeys` for the per-row key ranges.
- `DraftQkNormRopeKernel` (norm, rotary and store of q / k / v with any head dimension, float32 rotary frequencies):
  written for this repository, structure from `QkNormRopeAppendKernel`; gufo's counterpart is its generic
  `LaunchBatchedFusedQKNormRoPEKvWrite`, which was not vendored.

The host operations (`hip/ops_host.cpp`) and `hip/drafter_math.hpp` are written for this repository.

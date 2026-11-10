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

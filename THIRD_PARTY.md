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

The MIT licence of the Gemma 4 extension requires its notice with copied portions:

    MIT License. Copyright (c) Francesco Moca. Permission is hereby granted, free of charge, to any person
    obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, subject to the conditions of the full licence text in
    /server/programming/gufo-ripple/ref-gemma4-ext/LICENSE (to be copied into LICENSES/ before release).

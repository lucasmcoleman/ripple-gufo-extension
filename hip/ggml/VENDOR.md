# Vendored kernels: hip/ggml

Quantized GEMV, small-batch and W8A8 prefill GEMM kernels for Q4_K, Q5_K, Q6_K, Q8_0 and the F32/BF16
GEMV, as shipped by the Gemma 4 extension for Gufo.

- Source repository: https://github.com/franzmoca/gemma4-gufo-extension (local checkout
  `/server/programming/gufo-ripple/ref-gemma4-ext`)
- Commit: `a31ce42b13f57c53eed3fa5f08e5a20d880e3aba` (a31ce42b), directory `hip/ggml/`
- Licence: MIT, Copyright (c) 2026 Francesco Moca (notice below). The Gemma 4 files are themselves
  excerpts of Gufo (gufo-org/gufo, MIT, Copyright (c) 2026 gufo contributors) `src/families/qwen/`
  kernels; each file's header comment names its upstream file.

## Files

Vendored (8): `gemm.hpp`, `quant_ops.hpp`, `small_batch_gemm.hpp`, `prefill_quant_gemm.hpp`,
`gemv.hip.cpp`, `gemv_quant.hip.cpp`, `prefill_quant_gemm.hip.cpp`, `prefill_quant_wave64.hip.cpp`.

Not vendored (not needed for projections): `argmax.hip.cpp`, `prefill_embed.hip.cpp` (GPU embedding
gather, no IQ4_XS decode), `token.hpp`.

## Changes made

One mechanical rename, `sed 's/gemma4/ricursive/g; s/GEMMA4/RICURSIVE/g'`:

- namespace `gufo::models::gemma4::ggml` -> `gufo::models::ricursive::ggml` (and `...::ggml::detail`,
  one fully qualified use in `prefill_quant_gemm.hip.cpp`);
- include guards `GUFO_MODELS_GEMMA4_HIP_GGML_*` -> `GUFO_MODELS_RICURSIVE_HIP_GGML_*`;
- the sentence "in the gemma4 extension namespace" in the seven file header comments (and `gemm.hpp`,
  `quant_ops.hpp`) now reads "ricursive".

Nothing else. Check: reversing the rename (`sed 's/ricursive/gemma4/g; s/RICURSIVE/GEMMA4/g'`) gives files
that `diff` reports identical (0 lines) to the a31ce42b originals. `#include "hip/ggml/..."` paths are
unchanged (relative to the extension root). The kernels are unmodified, so a shape that fails is
reported, not patched.

## Difference from upstream Gufo (measured 2026-11-10)

Compared with `/server/programming/gufo-ripple/gufo` (HEAD 0e2de43e, path
`src/families/qwen/{kernels,hip_common}/`), namespace normalised (`gufo::hip` / ours -> `NS`).
Raw diffs: `/server/staging/ripple-gufo/s2-project/vendor-diff/*.diff`.

| file (upstream path) | ours / upstream lines | lines only ours / only upstream |
| --- | --- | --- |
| gemm.hpp (kernels/ops/gemm.hpp) | 91 / 329 | 38 / 276 |
| quant_ops.hpp (hip_common/quant_ops.hpp) | 470 / 855 | 36 / 421 |
| gemv.hip.cpp (kernels/gemv.hip) | 203 / 198 | 36 / 31 |
| gemv_quant.hip.cpp (kernels/gemv_quant.hip) | 150 / 199 | 9 / 58 |
| prefill_quant_gemm.hpp | 565 / 575 | 23 / 33 |
| prefill_quant_gemm.hip.cpp (kernels/prefill_quant_gemm.hip) | 898 / 2000 | 226 / 1328 |
| prefill_quant_wave64.hip.cpp | 43 / 47 | 5 / 9 |
| small_batch_gemm.hpp | 352 / 745 | 158 / 551 |

Findings:

1. There is no drift to an "older upstream commit" for these eight files. Upstream's versions at
   1c75887b (the commit the Gemma headers cite) and at HEAD 0e2de43e are identical except one comment
   line (a doc path in `prefill_quant_gemm.hip`); the per-file counts above are the same against both.
2. The Gemma copy is a pruned excerpt plus additions. Only upstream: Q8_K, Q3_K, IQ4_NL, IQ4_XS and
   IQ3_S decode (`quant_ops.hpp`, GEMV, prefill tiles, wave64 dispatch), BF16 activation GEMMs and
   dequantize-to-BF16, hipBLASLt plan database, dual-gate/up GEMM and fused SwiGLU / RMSNorm / SSM Q8_1
   epilogues, Qwen 27B wave64 verification tables, the GEMV route resolver (`QwenGemmRoute`) and
   telemetry plumbing, and `kPinnedOrder` / `StoreActivationSum` template parameters (the Gemma code
   behaves as `kPinnedOrder = false`, activation sums always stored). Upstream does have an IQ4_XS GPU decode; the Gemma
   excerpt does not, so IQ4_XS silently does nothing here (Project rejects it; the embedding is host side).
3. Only ours (Gemma fork additions): `LaunchKQuantSmallBatchDoubleStage`, `LaunchQ8_0SmallBatchWide`,
   in-kernel Q4_0 decode (`Q4_0Block`, Q4_0 instantiations), `static_assert`s of
   `FitsSmallBatch32BitIndices` at Gemma's shapes, and `LaunchGEMV` as a direct type switch instead of
   the route resolver (`abort()` for other types or K not a block multiple). Present in both: the
   `FitsSmallBatch32BitIndices` guard with the `Narrow` kernel parameter and the Q6_K vocabulary-head
   kernel for batch 7/8 at M >= 65536 (relevant here: M = 100352, widths 7 and 8 take that kernel).
4. Not diffed line by line: the ~1,600 differing lines of `prefill_quant_gemm.hip.cpp` and ~800 of
   `small_batch_gemm.hpp`. Their shared kernels (`W8A8BlockedWmmaGEMMKernel`, the small-batch exact
   kernels) were not compared for arithmetic equality with upstream; what Project relies on is checked
   by tests/projection_ops_test.cpp against FP64 instead.

## MIT licence notice (Gemma 4 extension)

    MIT License

    Copyright (c) 2026 Francesco Moca

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

The upstream Gufo code the Gemma 4 files were excerpted from is MIT, Copyright (c) 2026 gufo
contributors (same terms).

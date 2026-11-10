# S2 work package B: elementwise and attention GPU kernels

Branch `s2-kernels` (worktree `/server/programming/gufo-ripple/ricursive-kernels`), branched from `d1ef579`.
Files: `hip/kernels.hpp`, `hip/kernels.hip.cpp`, `hip/kernels_math.hpp`, `tests/kernel_ops_test.cpp`,
`tests/attention_ops_test.cpp`, `tests/kernels_math_test.cpp`, `tests/check.hpp`, `tests/kernels/CMakeLists.txt`,
`tests/kernels/run-gpu-tests.sh`, this file. Nothing under `src/`, `extension.cmake` or `extension.json` was touched.

## For the integrator (extension.cmake snippet for worker D)

- Source: `hip/kernels.hip.cpp` (one translation unit). Headers: `hip/kernels.hpp`, `hip/kernels_math.hpp`
  (no gufo header, no other include; the include root must be the extension root because the headers include each
  other as `"hip/kernels_math.hpp"`).
- Language/arch: HIP, `gfx1151`, C++20. Blocks are 128 or 256 threads and the reductions use `warpSize` at run time
  (wave32 is the gfx11 default); not built or tried with `-mwavefrontsize64`.
- Flags: `-O3 -fno-fast-math -ffp-contract=off` (plus `-Wno-unused-command-line-argument` as Gemma 4 does). This is
  the Gemma 4 extension's strict-FP unit set (`attention_wmma`, `gemv`, ...) rather than its default own-kernel set
  (`-O3 -ffast-math -fno-finite-math-only`): Gemma's comment in `extension.cmake` says fast math makes a row depend on
  the rest of its launch under this Clang, and the attention here promises bit-identical results across chunkings.
  The file also carries `#pragma clang fp contract(off)`.
- Link: the HIP runtime only (`hip::host` / `amdhip64`; `gufo_ext_hip` already brings it). The kernels need no other
  library. Standalone use: `tests/kernels/CMakeLists.txt` builds `ricursive_kernels` as a static HIP library from this
  one file.
- Namespace `gufo::models::ricursive::rocm`.

```
source /server/programming/gufo-ripple/env.sh
cmake -S tests/kernels -B build-kernels -G Ninja -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.2/lib/llvm/bin/clang++ -DCMAKE_BUILD_TYPE=Release
nice -n 10 cmake --build build-kernels -j2
build-kernels/ricursive_kernels_math_test                # host only, no GPU
tests/kernels/run-gpu-tests.sh build-kernels             # every GPU binary through the admission lock, one by one
```

## Operators (all fp32 row-major unless said; every function returns bool, enqueues on the stream, never syncs)

See `hip/kernels.hpp` for every argument. Compact list:

```
RmsNorm(x, weight, out, rows, dim, eps, stream, error*)                       // out may alias x; per-head norm = rows*heads, dim 128
ResidualPostNorm(x, branch, weight, rows, dim, eps, stream, error*)           // x += weight * branch * rsqrt(mean(branch^2)+eps)
ScaleRows(in, out, rows, dim, scale, stream, error*)
ValueMix(x0, a, lambda_dev, out, rows, dim, stream, error*)                   // x0 + sigmoid(lambda) * (a - x0)
QkNormRopeAppend(q, k, v, q_gamma, k_gamma, k_cache, v_cache, KvType, tokens, heads, kv_heads, eps,
                 start_pos, capacity, rotary_dim /*128 or 0*/, theta, stream, error*)
Attention128(q, k_cache, v_cache, KvType, out, tokens, heads, kv_heads, window /*0 = full*/, start_pos,
             capacity, scale, stream, error*)                                 // groups 1, 2, 10
SigmoidGate(attn, gate, out, rows, dim, stream, error*)
ClampedSwiGlu(gate, up, out, rows, dim, clamp, stream, error*)
GatherRows(src, ids_dev, dst, n_ids, src_rows, dim, stream, error*)
```

`KvType::kF16` is the production cache (binary16, round to nearest even); `KvType::kF32` is the test mode.
`hip/kernels_math.hpp` holds `RingSlot`, `FirstVisibleKey`, `KeyVisible`, `TileCount/TileStart/TileKeys`, `KvHeadOf`,
`MinCapacity`, `RopePair/RopePartner/RopeAngle`; the kernels call exactly these.

### Contracts the executor must respect

- Ring capacity: a launch of `tokens` rows appends all its K/V before any row attends, so the ring needs
  `capacity >= window + tokens - 1` (sliding) or `start_pos + tokens` (full, never wraps). `Attention128` rejects less
  (`MinCapacity`); `QkNormRopeAppend` rejects `capacity < tokens` only (it does not know the window). Decode rows
  (`tokens = 1`) need exactly `window` rows. Laguna used `window + max_chunk`; this is the same rule, tight.
- `rotary_dim` is 128 on sliding layers and 0 on full layers; anything else is rejected. Order per layer:
  projections -> `QkNormRopeAppend` (q normed/rotated in place, K/V appended) -> `Attention128` -> `SigmoidGate`.
- Chunking never changes a row: the sum order of a row depends only on its position and the window.

## Decisions

- **Clamp `c <= 0` or non-finite is rejected** by `ClampedSwiGlu` (returns false, launches nothing). Why: the model's
  `swiglu_clamp` may be per layer, and an unreviewed "0 = no clamp" would silently change numbers if a config or a
  parser produced 0 or NaN. Note the disagreement this exposes: `oracle/np_ref.py` (`if c > 0:` clamp, else none) and
  recon C-gpu-inventory 3b ("0 = no clamp") treat `c <= 0` as "no clamp". The brief overrides here; if some layer
  legitimately has no clamp, the executor must call a plain SwiGLU explicitly (not provided: YAGNI until a model needs it).
  NaN inputs give NaN outputs (comparisons, not `fminf`, so a NaN is not replaced by the clamp). A `-inf` gate input
  also gives NaN (`-inf * sigmoid(-inf) = -inf * 0`), as in numpy; mathematically silu(-inf) = 0. Not handled: it
  cannot occur from finite activations.
- **RoPE angle in double.** `pos * theta^(-2 pair / 128)` and its cos/sin are evaluated in double and narrowed after.
  A float angle at position 3000 is off by up to 1.2e-4 rad, far more than the 2e-5 attention tolerance; llama.cpp and
  np_ref use float angles, which only matters beyond a few hundred positions (their tiny-fixture sequences are 64 long,
  where the angle difference is ~4e-6). Cost: double sin/cos per element, accepted (no speed tuning).
- **Strict floating point for the whole file** (see flags above).
- Per-head norm is `RmsNorm` with `rows = tokens * heads`; no separate kernel. No V norm.

## What differs from Laguna's kernels (beyond the inventory's list: groups, rotary_dim=0, scale)

1. K/V cache element type is a template parameter (F16 or F32), loads via overloads (`LoadKeys8`, `LoadCache`).
2. RoPE: Laguna's YaRN fields (`corr_low/high`, `ext_factor`, `mscale`, `freq_scale`) and its repeated
   `theta *= theta_scale` float loop are gone; angle is the closed form in double. `rotary_dim` is runtime 0 or 128.
3. Attention grid is `(tokens, kv_heads)` instead of `(kv_heads, tokens)`: tokens in `blockIdx.x` (limit 2^31)
   rather than `blockIdx.y` (65535 on AMD), so long prefill chunks cannot overflow the grid.
4. `DecodeRow` table (batched multi-sequence decode) removed: not asked for, no multi-session batching in scope.
5. Index and tile arithmetic moved into `kernels_math.hpp` (host and device), including `MinCapacity`.
6. `GateLogitsKernel`/`GateKernel` (softplus per-head gate) not carried over: replaced by `SigmoidGate`.
7. Compile flags (strict FP, above); Laguna had only the pragma.
No change to the online-softmax structure, tile size 128, or the thread mapping.

## Copied material and licences

- Laguna code (our own fork, `/server/staging/ripple-gufo/ref/laguna/`): `QkNormRopeAppendKernel`, `AttentionKernel`
  structure, adapted as above. Our own code.
- GLM 5.3 extension (our own): clamped SwiGLU formula (`Activate` in `moe.hip.cpp`), re-written, not copied.
- Gemma 4 extension (MIT, Copyright Francesco Moca): no kernel or test code copied. `tests/check.hpp` keeps the
  `Require`/`Run` idea of its `tests/check.hpp` (the S1 branch's `check.hpp`, same namespace `ricursive_test`, is the
  base; this branch adds RNG, error statistics, twin bookkeeping and HIP buffer helpers behind `#if __HIPCC__`).
  The S1 `THIRD_PARTY.md` already carries the MIT notice for that idea. The strict-FP flag set and test layout
  (standalone executables, exit 77 for no GPU) follow Gemma 4's `extension.cmake` and tests; no text copied.
  Merge note: S1 and this branch both own `tests/check.hpp`; this version is a superset of S1's (same `Require`,
  `RequireContains`, `Run`), take this one.

## Tests

Host only (no GPU): `ricursive_kernels_math_test`. GPU (through `gpu-lock-exec`, each under 20 s by design):
`ricursive_kernel_ops_test`; `ricursive_attention_ops_test` with modes `small` (tiny-fixture shapes 10/1 and 20/2,
windows 8 and 16, full layers, groups 1 and 2, both cache types, every row checked), `real-sliding-f32|f16`,
`real-full-f32|f16` (80 heads, 8 KV heads, window 1024 / full, T = 3300 positions, 32-row ring chunks so the 1055-row
ring wraps 3.1 times, rows sampled around window, ring and 128-multiple boundaries).

RESULTS_PLACEHOLDER

# S2 work package A: projections and embedding

Branch `s2-project` (worktree `/server/programming/gufo-ripple/ricursive-project`). Files: `hip/ggml/**`
(vendored, see `hip/ggml/VENDOR.md`), `hip/project.{hpp,cpp}`, `hip/embed.{hpp,cpp}`,
`tests/projection_ops_test.cpp`, `tests/embed_test.cpp`, `tests/project/{CMakeLists.txt,check.hpp,run-gpu-tests.sh}`.
Nothing under `src/`, `extension.cmake` or `extension.json` was touched.

## Interfaces

```cpp
namespace gufo::models::ricursive::rocm {        // hip/project.hpp
inline constexpr int kSplitRows = 16;
inline constexpr std::size_t kWeightTailBytes = 4096;
struct DeviceTensor {                            // does not own `data`
  const void* data;         // device pointer; kWeightTailBytes readable bytes must follow `bytes`
  core::GgmlType type;      // F32, BF16, Q8_0, Q4_K, Q5_K, Q6_K
  std::uint32_t rows;       // M, output features (GGUF ne[1])
  std::uint32_t cols;       // K, input features  (GGUF ne[0], contiguous)
  std::size_t bytes;        // == rows * RowBytes(type, cols)
};
std::size_t RowBytes(core::GgmlType, std::size_t cols);     // 0 if unsupported / not block aligned
bool IsProjectable(core::GgmlType);
class Scratch { void Reserve(rows, cols); ... };            // Q8_1 buffer for rows > 16
void Project(const DeviceTensor& w, const float* x, int rows, float* y, Scratch&, hipStream_t,
             bool exact = false);                           // y[rows][M] = x[rows][K] * W[M][K]^T
}
namespace gufo::models::ricursive {              // hip/embed.hpp
struct EmbeddingTable { const void* data; core::GgmlType type; size_t vocab, hidden; };
size_t EmbeddingRowBytes(core::GgmlType, size_t hidden);
void EmbedRowsHost(const EmbeddingTable&, std::span<const int32_t> tokens, float scale, float* out);
}
namespace gufo::models::ricursive::rocm {        // only with ENGINE_ENABLE_HIP
struct EmbedStaging { std::vector<float> host; };
void EmbedTokens(const EmbeddingTable&, std::span<const int32_t> tokens, float scale,
                 float* device_out, hipStream_t, EmbedStaging&);  // synchronizes the stream
}
```

`Project` dispatch (mirrors Gemma 4 `Executor::Project`, without WMMA/grouped/half branches):

| weights | rows | kernel |
| --- | --- | --- |
| Q4_K, Q5_K, Q6_K, Q8_0 | 1 | `ggml::LaunchGEMV` |
| same | 2..16 | `ggml::LaunchBatchedQuantGEMMFp32` (FP32 activations, bit-equal to the GEMV per row by the Gemma tests) |
| same | > 16 | `LaunchQuantizeActivationQ8_1FromFp32` into `scratch`, then `LaunchBatchedQuantGEMMPreQuantized` (W8A8 WMMA, activations rounded to int8 blocks of 32) |
| same, `exact = true` | any | 16-row groups of `LaunchBatchedQuantGEMMFp32` (slow; test aid) |
| F32, BF16 | any | one `LaunchGEMV` per row (no batched dense kernel exists) |
| anything else | any | `std::runtime_error` naming the type (IQ4_XS, F16, Q4_0, Q3_K, ...) |

It also throws for null pointers, rows < 1, cols not a whole number of blocks, and `bytes` that do not equal
`rows * RowBytes`. The Gemma-only fast paths `LaunchKQuantSmallBatchDoubleStage` and
`LaunchQ8_0SmallBatchWide` are vendored but not dispatched (speed tuning; not needed for correctness).
Q4_0 has kernels but is rejected: the target file has none and nothing tests it here. `Scratch::Reserve`
allocates (and synchronizes); reserve the maximum before a graph capture. After the launch `Project` calls
`hipGetLastError` and throws on a launch error. Kernel execution errors surface at the caller's next sync.

Embedding: `EmbedRowsHost` validates every id first (id < 0 or >= vocab throws, naming the id, before any
write), then per token `gufo::quant::DequantizeRow` of its row from the mapped table and one FP32 multiply by
`scale` (the CPU reference does `raw * embedding_scale`). Supported table types: IQ4_XS, Q4_K, Q5_K, Q6_K,
Q8_0, F32, F16, BF16. It touches only the requested rows' pages. `EmbedTokens` stages into host memory,
`hipMemcpyAsync`, then `hipStreamSynchronize` (staging can be reused at once; a pinned, double-buffered version
would be a later optimization).

## For the integrator (extension.cmake snippet; not applied)

Sources and per-source flags (the Gemma 4 extension's for the same files, in their own target):

```cmake
gufo_extension_target(NAME gufo_ricursive_project   # or fold into the executor's hip target
  SOURCES hip/project.cpp hip/embed.cpp
  LINK_LIBRARIES gufo_ricursive gufo_ext_runtime gufo_ext_core gufo_ext_hip roc::hipblas Threads::Threads)
if(TARGET gufo_ricursive_project)
  gufo_extension_hip_sources(gufo_ricursive_project SOURCES
    hip/ggml/gemv.hip.cpp hip/ggml/gemv_quant.hip.cpp)             # default HIP flags, no fast math
  gufo_extension_hip_sources(gufo_ricursive_project SOURCES
    hip/ggml/prefill_quant_gemm.hip.cpp OPTIONS -fno-slp-vectorize)
  gufo_extension_hip_sources(gufo_ricursive_project SOURCES
    hip/ggml/prefill_quant_wave64.hip.cpp OPTIONS -mwavefrontsize64 -fno-slp-vectorize)
  target_compile_options(gufo_ricursive_project PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)
  target_include_directories(gufo_ricursive_project PUBLIC "$<BUILD_INTERFACE:${GUFO_EXTENSION_BUILD_ROOT}>")
endif()
```

Do NOT put `-ffast-math` on any of them (decode GEMV and the small-batch kernels must round identically).
Includes in the sources are extension-root relative (`"hip/ggml/gemm.hpp"`, `"hip/project.hpp"`). The
vendored headers need `ENGINE_ENABLE_HIP` (set by `gufo_ext_hip`), `<gufo/ext/experimental/ggml_dequant.hpp>`,
`hip_utils.hpp` and `dispatch_telemetry.hpp`. The standalone build compiled `project.cpp`/`embed.cpp` as HIP
language sources; compiling them as ordinary C++ (as Gemma's `executor.cpp` is) is expected to work but was not
built here. A device model must allocate every weight with `kWeightTailBytes` zeroed bytes after it (Gemma's
`Uploader::Allocate`), because the K-quant kernels over-read the last row.

Standalone build (`tests/project/CMakeLists.txt`): HIP compiler `/opt/rocm-7.2.2/lib/llvm/bin/clang++`,
`gfx1151`, flags from the gufo build (`-O3 -DNDEBUG -std=c++20 --offload-arch=gfx1151 -Wall -Wextra -Wpedantic`,
`-DENGINE_ENABLE_HIP=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1`), includes
`<gufo>` and `<gufo>/include` (the public headers forward to `src/core/...`). The public library is not
separate: `gufo::quant::DequantizeRow` and the GGUF reader live in `build-release/libgufo_core.a`, which was
linked as a rescanned group with `libgufo_hip_host.a`, `src/core/platform/libgufo_platform.a`,
`src/core/rust/cargo/release/libgufo_rust.a`, plus `hip::host`, `roc::hipblas`, `OpenSSL::Crypto`, pthread, dl.
That linked and ran. The tests add `-ffp-contract=off -fno-fast-math` to keep the reference exact.

```
source /server/programming/gufo-ripple/env.sh
cmake -S tests/project -B /server/staging/ripple-gufo/s2-project/build -G Ninja
nice -n 10 cmake --build /server/staging/ripple-gufo/s2-project/build -j2
```

## Tests

`tests/embed_test.cpp` (host part needs no GPU; `embed_test gpu` also copies rows to the device and back):

1. `DequantizeRow` against independent decoders written in the test from the ggml block definitions
   (IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0, F32; own binary16 decoder): 24 random rows of 5120 per type, bit-equal.
2. Real file `Ripple-65B-Preview-AD-Q4_K_M.gguf`: `token_embd` is IQ4_XS [5120 x 100352],
   `ricursive.embedding_scale` 71.5542; 200 seeded random ids (seed 20261110) plus ids 0 and 100351, read through
   the mapping, all bit-equal to the independent decode times the scale; process RSS 67 -> 90 MiB (the file
   is 40 GB; only the metadata and those 202 rows were touched).
3. Tiny fixtures: all 512 rows of `tinyA` (F32), `tinyB` (F32), `tinyB-Q4_K_M` (Q4_K), `-Q5_K_M`, `-Q6_K`,
   `-Q8_0`, bit-equal.
4. Twins (32 rows, max|diff|/max|expected|): correct lookup 0 (exact); unscaled 0.956 (fixture) / 0.986 (real);
   token id off by one 1.51 / 0.71. Ids -1, vocab, vocab+1000, INT_MAX, INT_MIN are rejected before any write.

`tests/projection_ops_test.cpp` cases (`--list`): `q4k_10240x5120` (attn_q, attn_gate), `q8_0_1024x5120`
(attn_k, attn_v), `q6k_5120x10240` (attn_output), `q4k_40960x5120` and `q5k_40960x5120` (ffn_gate, ffn_up),
`q4k_5120x40960` and `q5k_5120x40960` (ffn_down), `q6k_100352x5120` (head), `f32-tiny` (F32 at the tiny
fixtures' shapes: 1280/2560 query, 256/512 hidden, 2048/4096 FFN, 512 vocab, both directions, plus two BF16),
`twins`, `reject`. One shape per invocation, rows 1..16, 17 and 512 inside it (rows 2..16 are all run, not just
2, 8, 9, 16). Per shape (details at the top of the test file):

- 512 single-row GEMV results; rows 0 and 15 against an FP64 dot of `DequantizeRow` weights on ~100 sampled
  outputs (first 16, last 16, power-of-two boundaries +-1, random). Relative to sum|w x|, bound 2e-6: FP32
  accumulation depth up to ~700 sequential adds per lane gives a random-walk error ~1e-6 and a worst case
  depth*u ~ 4e-5 (u = 6e-8); wrong data is off by O(1), five orders above.
- one-hot probes at columns 0, K-1, K-32, K-256 (last column/block checks) through GEMV, small batch, exact.
- rows 2..16 bit-equal to the single rows, every output of every row (Gemma's tests assert the same).
- rows 17 and 512 with `exact = true` bit-equal to the single rows.
- rows 17 and 512 normal mode: every element within 0.06 x row RMS of the exact result; whole-matrix RMS
  relative error <= 1.5e-2 (expected ~5.5e-3); sampled outputs within 6 sigma_o + 2e-6 sum|w x| of FP64,
  sigma_o^2 = sum_b d_b^2 sum_c w_c^2 / 12, d_b = amax_b/127 (per-output int8 rounding noise).
- y poisoned with a NaN pattern, 4096 guard words on both sides and after the expected end; unwritten, non
  finite or out-of-range writes fail with the first bad row and column. x has NaN guards (over-reads of x give
  NaN). Weights have the 4 KiB zero tail followed by 64 KiB of 0xFF (NaN scales).
- `twins` (1024 x 5120 per type, weights allocated 1.7x so wrong reads stay in bounds): Q4_K read as Q5_K,
  Q5_K as Q4_K, Q6_K as Q8_0, Q8_0 as Q6_K (raw GEMV); `Project` refusing the same confusion on declared size;
  rows 8 instead of 9 (row 8 unwritten); M-1 (last output unwritten); transposed dims (same bytes).
- `reject`: IQ4_XS, F16, Q4_0, Q3_K, IQ3_S rejected naming the type at rows 1, 4 (exact) and 17; wrong size,
  unaligned cols, rows 0, null weight rejected. No kernel launches; run it through the lock anyway.

`--cpu` runs the same harness against a host implementation of `Project` on shapes cut to M <= 392, rows <= 40.
It proves the harness (reference code, envelope, sentinels, twins), says nothing about the GPU kernels.

GPU invocations: `tests/project/run-gpu-tests.sh` runs each case (and `embed_test gpu`) through
`GPU_ADMISSION_LOCK=/server/state/gpu-admission/gpu.lock gpu-lock-exec -w 120`, retries once after
`RETRY_SLEEP` (240 s) on exit 75, prints a PASS/FAIL/SKIP table. Every case keeps device memory under
3.5 GB (asserted in the test) and prints its wall time (warns above 20 s; not measured on GPU yet).

## Results (2026-11-10)

- Host: `embed_test` PASS (0.17 s), output in `/server/staging/ripple-gufo/s2-project/embed_test.host.out`.
- Harness self-check on the CPU backend: all ten cases PASS
  (`/server/staging/ripple-gufo/s2-project/cpu-selfcheck.txt`); rows=17 normal mode with the noise model:
  RMS relative error 0.0043..0.0056, max|diff|/row RMS 0.021..0.025 (limit 0.06), rms z 0.94..1.14 (the
  sigma_o model matches), excess 0.43..0.58 (limit 1). Twins on the CPU backend: type confusion non finite,
  transposed dims error 0.057..0.069 against 2e-9 correct, row/output off-by-one detected as unwritten.
- GPU: NOT RUN. The admission lock was held by a Scriberr job (`gpu.lock.owner`: parakeet_rocm_transcribe.py,
  pid 489, since 2026-09-30) at 13:35 and 13:37-13:39 (`gpu-lock-exec -w 30` and `-w 120` on the `reject`
  invocation: exit 75 both times). Per the brief nothing else was tried. Not run on GPU: lock held.

## Unverified

- Every GPU path: all eight real shapes (in particular 40960 as M and as K, Q6_K at K = 10240 and M = 100352,
  Q8_0 at M = 1024, the batch 7/8 head kernel), the F32/BF16 GEMV at the tiny shapes, `Project` rejection
  under the real runtime, the device copy in `embed_test gpu`. The test harness is proven on the CPU backend
  only; the 0.06 x RMS and 1.5e-2 envelope constants were calibrated on that backend's noise model, so a real
  kernel with extra rounding (fp16 activation scales) could need them revisited, not silenced.
- Test wall time under 20 s per invocation on the GPU (host data generation at K = 40960 and the 421 MB head
  matrix is the part that could be slow; GPU time should be milliseconds per launch).
- `project.cpp` / `embed.cpp` built as ordinary C++ inside the extension build (built as HIP sources only).
- Differences of the shared W8A8/small-batch kernels from upstream (not diffed line by line; see VENDOR.md).

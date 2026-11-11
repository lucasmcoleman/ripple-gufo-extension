# Drafter stage 0: hidden-state capture, dump tool, CPU oracle

Stage 0 of the DFlash2 drafter design (`/server/staging/ripple-gufo/perf/DRAFTER-GUFO-DESIGN.md`, job 1692): the
kill experiment. It decides whether building the drafter into this extension can beat llama.cpp's drafter. This
document covers what is in the repository; the questions it answers and the GPU commands are in
`/server/staging/ripple-gufo/perf/STAGE0-NOTES.md` and `STAGE0-GPU.md`.

## Capture (executor level)

`ExecutorConfig::capture_layers` (at most 5 distinct zero-based layers, default empty = off) makes the executor copy
the output of those layers, `l_out`, the residual stream after the layer and before the final norm, to a buffer the
caller gives `Executor::Forward(..., tap, capture)`:

- Layout `[tap k][row t][hidden]` float32, `capture.size() == capture_layers.size() * rows * hidden`. Slot k is
  `capture_layers[k]`, in the order given (the drafter wants 1, 20, 39, 58, 77).
- An empty `capture` span (the default argument) means no capture for that pass, also on an executor that has
  `capture_layers`. A buffer on an executor without `capture_layers`, or of the wrong size, is refused
  (`std::invalid_argument`) before anything changes. Bad `capture_layers` (more than 5, duplicate, out of range)
  are refused when the executor is built.
- Host ops and GPU executor (`hip/executor.cpp`, one orchestration): the layer loop copies the n rows of `b.h` into a
  device slot right after the layer's last `ResidualPostNorm` (one `CopyInternal`) and the pass reads the slots back
  after the logits. The slot buffer `[tap][max_rows][hidden]` is allocated only when `capture_layers` is set.
- CPU executor: the Reference's own `l_out` tap, copied into the buffer.
- With capture off nothing changes: the only added work in a pass is one `bool` store and one `bool` test per layer
  in `OpsExecutor`, one `std::span` test in `CpuExecutor`, and `Executor::Forward` checks `capture.empty()`. The
  profiling switch (`RICURSIVE_PROFILE=1`) is not extended to the capture copies.

`tests/capture_test.cpp` (`ricursive.capture`) on tinyA and tinyB, host ops and CPU executor: captured rows against
`Reference::ForwardFull` `l_out` (existing host tolerance 1e-4 relative; measured 1.1e-6 host, 0 CPU), capture
equals the `l_out` tap callback of the same Forward bit for bit in any layer order, logits are bit-identical
without capture, with `capture_layers` unused, and capturing, 16-row passes reproduce 1-row passes (logits row
hashes and captured rows), and the refusals.

## Dump tool: `ricursive_dump_taps` (named target, built on request)

`ricursive_dump_taps MODEL.gguf --tokens F.i32 --prompt-tokens P --generate G --out DIR [--backend cpu|host|gpu]`
or `--jobs JOBS.txt --generate G` (one model load for many prompts; finished jobs, those with `OUTDIR/meta.json`,
are skipped, so a cut-off run is resumed by repeating the command). Per job: (a) greedy generation as serving does
it (prompt passes of at most `--rows` 2048, then 1-row passes), the argmax and a 64-bit FNV-1a hash of every logits
row; (b) from scratch, the prompt in the same passes with capture on, then the G generated tokens in `--verify-rows`
(16) passes with every row's logits and capture on. Output: `ids.i32`, `taps.f32` (`[P+G][5][hidden]` float32,
position major: one row of 25,600 floats is the drafter's feature input), `meta.json` (written last). The tool
prints `EXACT: n of m` (hash equality of 16-row pass rows with the 1-row decode rows at the same positions): the
exactness precondition of the design, measured, not asserted.

## Oracle: `tools/drafter_oracle.py` (run through `tools/drafter_oracle.sh`)

The pinned upstream model file (z-lab/dflash `dflash/model.py` at 07ebd93, sha256 `f55b7fe0...651e`, the file gufo
pins in `tools/qwen27b/dflash_reference.py`) with the drafter GGUF's weights dequantized to float32 and the target's
embedding rows and output head. Variants R (recipe), Rs (embedding scaled), L (llama.cpp-like). Self-checks:
`selfcheck`; per-anchor drafts and accepted counts: `run`; one-anchor timing: `timing`; `tools/drafter_report.py`
prints the three acceptance tests. Details, results and limits: `STAGE0-NOTES.md`.

`drafter_oracle.sh` is the only way to run the oracle: it executes the downloaded upstream file, so the interpreter
is isolated (`-I`), the environment is empty (offline, no tokens), CPU torch only, in the venv
`/server/staging/ripple-gufo/venv-oracle`. `tests/fixtures_text/make_drafter_fixture.py` writes a tiny random drafter
GGUF (same tensor names and keys as the real one) for the plumbing check.

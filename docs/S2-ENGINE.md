# S2 engine: model, session, component-cache adapter, runner, drivers (work package D)

Branch `s2-engine`, written 2026-11-10 against gufo 0e2de43e (extension API 0.2.25). CPU only: nothing here ran on a
GPU and the real 40 GB model was never loaded. The compute interface is `src/executor_iface.hpp`
(docs/EXECUTOR-CONTRACT.md); everything below programs against `Executor` only.

## Files

| file | role |
| --- | --- |
| `src/engine.{hpp,cpp}` | `Model` (config, weights table, tokenizer, template hash, context cap, ring arithmetic) and `Session` (tokens, executor, frontier logits, sync) |
| `src/executor_factory.{hpp,cpp}` | GPU or CPU backend choice |
| `src/continuation_adapter.{hpp,cpp}` | `cache::Adapter` over a session |
| `src/runner.{hpp,cpp}` | `TextModelRunner`, provider options |
| `src/provider.cpp`, `src/extension.cpp` | registration (provider; tool dialect) |
| `src/drivers.cpp` | `gufo prompt`, `chat`, `bench` |
| `src/chat_prompt.{hpp,cpp}` | render + tokenize, message text cannot inject control tokens |
| `src/chat_request.{hpp,cpp}` | host `ChatRequest` -> the renderer's JSON (new file, not in the brief's list) |
| `src/output_parser.{hpp,cpp}` | streaming reasoning/content/tool-call split, host scanner, dialect |
| `tests/{engine,continuation,output_parser,chat_e2e}_test.cpp`, `tests/fixtures_text/` | tests, fixture generator |

## Backend choice (`executor_factory`)

- `RICURSIVE_BACKEND` unset or `gpu`: GPU. `cpu`: the scalar CPU backend. Anything else is an error that names both.
  `gufo prompt --cpu` also selects the CPU backend.
- The CPU backend is for tests and tiny models only (every weight is decoded on the fly, caches are F32); the error
  and help text say so (`BackendHelp()`). It is built into `gufo_ricursive` when the CMake option
  `RICURSIVE_CPU_BACKEND` is ON (default: `BUILD_TESTING`), which links `gufo_ricursive_reference`. The component
  cache is not served on the CPU backend (`serve` needs the GPU one; the error says so).
- **GPU executor signature expected** (declared in `src/executor_factory.hpp`, compiled in when the build defines
  `RICURSIVE_HAVE_GPU_EXECUTOR`; until then the error is "GPU executor not built"):

      std::unique_ptr<gufo::models::ricursive::Executor>
      gufo::models::ricursive::CreateGpuExecutor(const ModelWeights& weights, const ExecutorConfig& config,
                                                 std::string* error);

  Assumptions that must hold: called once per session, many times for one `weights` (the first call uploads, later
  calls share the device weights, released with the last executor); returns nullptr with `*error` and never throws;
  `config.kv_type` is F16 (F32 + `exact` only as a test mode); the executor honours executor_iface.hpp exactly. The
  runner creates an `ext::DeviceProbe` (device memory!) only when the backend `uses_device`; a lost device is then
  reported through `PollDevice`/`DeviceUsable`.
- CMake: the GPU worker adds `target_compile_definitions(gufo_ricursive PRIVATE RICURSIVE_HAVE_GPU_EXECUTOR=1)` and
  links its library into `gufo_ricursive`.

## Engine

- Load: `ParseConfig`, `ResolveWeights`, tokenizer from the GGUF (`tokenizer.ggml.*`), SHA-256 of
  `tokenizer.chat_template` against `kRippleChatTemplateSha256`. A different or missing template is refused with both
  hashes in the message unless `--allow-chat-template-mismatch` (then prompts may be wrong).
- EOS set: {100265, 100257} plus the GGUF's eos/eot ids (the same in the real file), each only if inside the vocab. No
  BOS is ever added.
- **Context cap**: provider option `--context-cap`, else the host's `--context` (serve), else
  `min(32768, the GGUF's context_length)`. The GGUF's 2,097,152 is never a default; a cap above it is an error.
- **Memory** (F16 caches on the GPU; the CPU backend stores F32, twice as much): full layers 16 x (K+V) x 8 x 128 x 2 B
  = **64 KiB per token**: 32768 tokens = 2 GiB, the nominal 2,097,152 = 128 GiB. Sliding layers 64 x 2 x R x 2 KiB with
  the ring R = `round_up_64(max(window + max_rows - 1, window - 1 + 63))` = 3072 rows for 2048-row passes = **768 MiB**
  independent of context, plus the frontier logits (392 KiB). One session at the default cap is about 2.77 GiB. (The
  contract's worked example, 3136, used 2064-row passes with a 16-row tail merge; passes here are plain 2048-row.)
  `Model::SessionBytes` computes it; `engine_test` asserts the real-model numbers.
- Prefill in passes of at most `max_rows` (2048); **every pass returns the logits of its last row** (one head
  projection), so a checkpoint can be taken after any pass and the cache's private logits component is always valid.
- `Session::Sync(target)`: common prefix with the session's tokens, less one token (the last needs logits);
  `Executor::Rewind(common)`; if it returns false the engine **recomputes from position 0** (a shallower position
  needs even older rows, so there is no partial retention), after telling the cache (`[0, cap)` guard).
  Failure: a throw from `Forward` with the executor kFailed empties the session; kReady means it was refused and
  nothing changed. A prompt over the cap is refused with "prompt of N tokens exceeds the context cap of M tokens".
- Before every `Forward` the session calls `PlanForward` and hands the plan to a write guard (the adapter's
  `MutationGuard`); not under the session mutex (the guard waits for readers that take it).

## Component-cache adapter

Components (ID = index + 1): tokens (append rows, 4 B); per layer K and V: sliding layers `kWindowRows`
(`window_rows = W - 1`, 64-row chunks), full layers append rows (256-row chunks); frontier logits (private,
vocab floats); metadata (private: version, position, cache element type). **No hidden-state component.** Transfers are
synchronous host copies through `Executor::ExportRows/ImportRows`; rows are copied, never lent (`BorrowRowsView` is
empty), so the cache's `copy_observed_rows` hook is a stub that throws.
- Save-before-overwrite: `BeforeOverwrite(id, written)` for every row component and `(id, evicted)` for window
  components, from `Executor::PlanForward`, before each pass; `[0, context)` before a restore or a full recompute;
  `BeforeRelease` on invalidate and destruction.
- Restore: `BeginRestore` (frontier check, guards, `Executor::BeginImport`), `CopyRowsIn` -> `ImportRows`, private
  loads (pieces allowed), `Validate`: per-component coverage from `FirstChunkRow` to n, metadata check, then
  `CommitImport` (the executor's own coverage check). Any failed load latches `failed`; `Invalidate` clears it.
- Identity (appended to the caller's artifact/template identity): layout version, context cap, pass width, ring,
  chunk rows, cache element type, exactness.
- One mutex per session serializes executor calls against row transfers from other threads.
- Rewind rule: the executor's (`Rewind()`), never one computed from the logical end (see the twin below).

## Runner, provider, drivers

- Runner base: `ext::TextModelRunner`, **not** `HipTextModelRunner`: the HIP base owns an `ext::DeviceProbe` that
  allocates device memory at construction, and the CPU backend must not touch the GPU. The runner builds a probe
  itself only for the GPU backend.
- Descriptor: `incremental_prefill`, `prefix_reuse`; `incremental_text_is_exact` only on an exact backend (the CPU);
  no speculative decoding, no in-pass checkpoints, no multi-token decode, serial plans only.
- **Sampling defaults: none.** The model card (`ref/hf/README.md`) names no temperature/top-p/top-k (the only
  `temperature` in the vendor files is a request example, `inference/server/README.md:61`, value 0); gufo's defaults
  apply (`sampling_defaults` unset).
- **Reasoning**: `ReasoningDelimiters()` = `<think>` / `</think>`; `ReasoningBudgetTags()` empty (the tags are plain text
  of several tokens, and there is no switch: the generation prompt never opens `<think>`, so `InitialOutputState` is
  `kAuto`; `--think` is ignored with a note).
- **Tool calls**: a registered `ToolDialect` (id `ricursive`): marker `<tool_call>`, syntax open `<tool_call>`,
  name prefix `\n{"name": "`, name suffix `", "arguments": `, close `}\n</tool_call>`, JSON arguments, plus
  `scan_tool_calls` (`ScanToolCalls`, including the partial mode for streaming: closed arguments and an open string only
  grow). The grammar compiled from this syntax accepts the template's own call text byte for byte
  (`output_parser_test`).
- Drivers: `prompt` and `chat` print reasoning, content and tool calls apart (`OutputParser`), chat keeps the RAW model
  text in the history (re-renders to the produced tokens); `bench` is Gemma's pp/tg table.

## Chat wiring and control-token safety

`RenderChat` returns one string. `src/chat_prompt.cpp` replaces every caller string (content, reasoning, tool names,
call arguments, tool schemas, including object keys) by a unique marker, renders, puts the originals back (JSON
escaped inside `tojson` output) and records where the template's own `<|im_start|>`, `<|im_end|>`, `<|endoftext|>` are.
The ids are the text cut at those positions, the cuts becoming the control ids, the rest encoded with
`parse_special = false` (so `|||...|||` user tokens still match, as in llama.cpp). A content that starts with `<think>`
keeps its marker prefix so the template's `lstrip().startswith('<think>')` decision is unchanged. Upstream's Qwen
path instead cuts only spans that contain a special spelling; the result is the same ids.
Needed from `src/text` instead of this trick: `RenderChat` returning content-span offsets (see end).

## Tests (all CPU, ctest label `cpu`; build by name, run 2026-11-10)

| test | time | what | twins (must fail) |
| --- | --- | --- | --- |
| `ricursive.output_parser` | 0.1 s | 27 cases x (unsplit, every two-way cut, single bytes) = 1468 runs; 400 random fragment streams; scanner; 148-prefix partial scan; grammar vs the template's call | close call on first `</tool_call>`: 2 of 27 cases wrong; no hold-back before `\n</think>`: 88 of 1468 runs wrong and 17 cut-dependent fuzz streams; call header without the space: refused at byte 36 |
| `ricursive.engine` | 14 s | tinyA/tinyB: 48 greedy tokens == scalar Reference argmax chain; passes of 1, 3, W-3, W+4, 59 rows equal one pass (max abs 0); sync at 10 divergence points around the window and the ring edge == fresh session (4 rewound, 6 recomputed); two syncs below the high water; cap of 32; real-model memory numbers | window-1: 48/48 (A) and 47/48 (B) tokens differ; RoPE per pass: 5.2 / 3.75 max abs; retention from the logical end: 0.508 / 0.258; cap 64 accepts 33 tokens |
| `ricursive.continuation` | 5 s | layout; checkpoint/restore at W-1, W, W+1, 2W+3, 40: rows byte-identical, frontier logits and next 16 tokens equal; rewinds (inside, outside, three consecutive); failed forward; guard calls == PlanPrefill and rows readable inside the guard | skipped layer: Validate refuses; skipped layer without coverage check: 20 / 10 row sets differ, next 16 diverge; retention-from-end: 1.32 / 0.191 max abs, stale row served; failed forward stays ready: 3.2 / 1.56 max abs; no guard: 0 of 132 / 68 calls |
| `ricursive.chat_e2e` (+ `chat_fixture`) | 4 s + 5 s | real tokenizer and template, 111 MB fixture (vocab 100352, hidden 128, generated by `tests/fixtures_text/make_chat_fixture.py` into the build dir): 23 goldens reproduced byte for byte, 22 ASCII conversations equal llama-tokenize id for id (up to 278 tokens); prompt of 29 tokens -> 32 greedy tokens == Reference chain, 194 bytes of text; injection | naive whole-string special parsing: 38 control ids vs the template's 13; window-1: 32/32 differ; file without the template refused |

`ctest -R 'ricursive\.(output_parser|engine|continuation|chat_fixture|chat_e2e)'`: 5/5 pass, 27 s.

## `gufo prompt` smoke (CPU)

`gufo` linked from a COPY of this tree (`scratch-engine/ext-copy`) whose manifest drops `requires: testing` from
`gufo_ricursive_reference`, with only `ricursive` as extension, `-DRICURSIVE_CPU_BACKEND=ON`, build dir
`build-engine-gufo` (a normal BUILD_TESTING build of core needs all 8 extensions). Source read first: `RunPrompt` opens
the GGUF, probes, calls the provider driver; no HIP call before it. Run:

    HIP_VISIBLE_DEVICES=-1 RICURSIVE_BACKEND=cpu strace -f -e trace=openat -o strace.txt \
      gufo prompt --model chat-tiny.gguf --prompt "What is 2+2?" --max-tokens 16 --verbose

Result: 15 prompt tokens, 16 generated tokens of text (random weights, so gibberish), 0.08 s load; `grep -c
'kfd\|/dev/dri\|renderD' strace.txt` = 0. `serve`, `chat`, `bench` were not run.

## Exact additions to the shared files

`extension.cmake`: one block between `# --- WP-D engine begin ---` and `# --- WP-D engine end ---`, appended at the
end (option `RICURSIVE_CPU_BACKEND`; `target_sources`/links/definitions on `gufo_ricursive`; tests `output_parser`,
`engine`, `continuation`, `chat_e2e`, ctest fixture `chat_fixture`). `extension.json`: `options` gets
`--context-cap` (integer) and `--allow-chat-template-mismatch` (boolean); `targets` gets `ricursive_output_parser_test`,
`ricursive_engine_test`, `ricursive_continuation_test`, `ricursive_chat_e2e_test` (after the contract test entry).
`src/extension.cpp` (not in my list) was rewritten: the skeleton's provider/factory moved to `provider.cpp`;
`ricursive_register` now registers the tool dialect.

## Needed in files I do not own

- `src/text/chat_template.*`: `RenderChat` returning content-span offsets would replace the marker substitution.
- `extension.json`: `gufo_ricursive_reference` has `requires: testing`, so the CPU backend (and `gufo prompt` on a tiny
  model) exists only in BUILD_TESTING builds; dropping that requirement would let any build have it.
- `src/reference.hpp`: the `friend class CpuExecutor` line (already in the contract doc).

## Deviations, unverified

- Passes are plain 2048 rows (no 16-row tail merge); ring 3072, not 3136. Every pass computes last-row logits.
- Unverified: any GPU run; the host's own reasoning/tool parsers against `<think>\n` framing and my delimiters; `serve`
  and the HIP component-cache resources (compiled and linked only); `--think` handling; ICU >= 76 exactness
  (`UnicodeDataIsCurrent` is not checked at load); a real-model template/tokenizer load beyond vocab-only.

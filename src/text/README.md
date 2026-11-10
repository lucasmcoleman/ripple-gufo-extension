# src/text: tokenizer and chat template for Ripple-65B-Preview (`ricursive`)

Plain host C++20, no GPU, no gufo dependency. Build inputs: `SOURCES.md`. Output-format notes for the later
parser stage: `OUTPUT-FORMAT.md`. Provenance: `VENDOR.md` (nothing copied). Branch `s1-text`.

## API

`tokenizer.hpp`: `Tokenizer::Create(tokens, merges, token_types, SpecialTokenIds, TokenizerOptions)`;
`Encode(text, parse_special)`; `Decode(ids, SpecialDecode)`; `TokenBytes`; `IsSpecial`; `IsStopToken`;
`StreamDecoder` (Push/Flush); `PreTokenizeLengths` (regex pieces, for tests); `UnicodeDataIsCurrent()`.
`chat_template.hpp`: `RenderChat(messages, tools*, {add_generation_prompt, clear_thinking}, &error)` and
`kRippleChatTemplateSha256`. `json_value.hpp`: order-preserving `JsonValue` + `ParseJson` / `DumpPythonJson`
(Python `json.loads` / `json.dumps(ensure_ascii=False)` rules: `", "`/`": "` separators, repr floats, big ints,
NaN/Infinity).

## Design decisions

* Own code. Unicode classes via ICU `libicuuc` (already a gufo core dependency, nothing new to link). A hand-made
  table would have to be generated from one fixed Unicode version, and the version turned out to matter (below).
* Pre-tokenizer: hand-written equivalent of the `dbrx` regex (digits grouped by 3; the optional prefix before
  letters is any char except CR/LF/L/N; marks are NOT letters, unlike the Qwen rule). Contractions are
  case-insensitive with Unicode simple folding (U+017F matches `'s`, as in Hugging Face's engine).
* Unicode cutoff (`TokenizerOptions::unicode_max_major = 16`): the Hugging Face `tokenizers` 0.23.3 engine
  knows Unicode 16; ICU 78.2 knows 17. The 4,657 letters/numbers added in Unicode 17 are "punctuation" to HF, so
  they are treated as unassigned here. Measured, not assumed: classifying all code points through HF's
  pre-tokenizer vs ICU gives exactly those 4,644 L + 13 N differences. Set 0 to trust ICU fully.
* Special tokens: from GGUF `token_type` (CONTROL/UNKNOWN/USER_DEFINED = special). `parse_special=false` keeps only
  the three USER_DEFINED `|||...|||` tokens active (matches llama.cpp and the goldens). One left-to-right pass,
  leftmost then longest (Hugging Face semantics). No BOS/EOS is ever added. `<think>`, `<tool_call>` etc. are text.
* BPE: merge ranks by (left id, right id), priority queue + linked list, O(n log n) per pre-token.
* Decode: `Decode` returns raw bytes. CONTROL/UNKNOWN/USER_DEFINED tokens render their literal text by default
  (so `Decode(Encode(s, true)) == s`); `SpecialDecode::kSkip` drops them. The 74 padding ids 100278..100351 (the
  GGUF gives them placeholder strings `[PAD100278]`, type UNUSED) and ids beyond the vocabulary decode to the empty
  string in both modes. `StreamDecoder` holds back an incomplete UTF-8 tail, replaces bytes that cannot be valid
  UTF-8 by U+FFFD (never triggers on valid streams), and turns a truncated tail at `Flush()` into one U+FFFD.
* Chat template: hand-written for this one template. Variables read: `messages`, `tools`,
  `add_generation_prompt`, `clear_thinking`. Jinja error behaviour is mirrored (`messages` empty, null/non-string
  content where a string is concatenated, tool call without name/arguments, ...); verified differentially.

## Results (all from `make -C tests/text test`, log in `/server/staging/ripple-gufo/s1-text/test-run.log`)

Tokenizer, ids (default and `parse_special=false`), `Decode(Encode(s)) == s` and streaming per case:
* 221 goldens x 2 modes: 0 mismatches, 0 round-trip failures.
* 7,886 seeded random strings (mixed scripts, emoji/ZWJ/modifiers, marks, digit runs, every whitespace kind,
  contractions in mixed case, code, special-token texts and near-misses, every odd code point in
  discriminating contexts): vs llama.cpp (llama_tok_batch.cpp, in-process `llama_tokenize`, checked first against
  the 221 goldens = 0 differences): 27 differ, all explained (below), 0 unexplained; vs Hugging Face: 0 differ in
  both modes; round trip and stream: 0 failures.
* One-off larger runs (not in `make test`): 58,411 strings, seed 7: 76 differ from llama.cpp (71 Unicode>=16,
  5 overlapping specials), 0 from HF; 4,448,256 strings = every Unicode scalar value in 4 contexts: 0 differences
  from llama.cpp and HF.
* Pre-tokenizer pieces vs HF's regex engine (`pieces`): every scalar value x 8 contexts = 8,896,512 strings,
  0 differences. Same run with the cutoff disabled: 23,311 differences (negative control).

llama.cpp vs Hugging Face (reported, not picked silently; we follow HF, the training-time definition):
* llama.cpp b12252's Unicode tables are older than HF's. Example `"\x01" "U+10D40" "358"` (Garay digit, Unicode 16):
  llama.cpp `189 172 238 113 222 17112`, HF and ours `189 172 238 113 222 1758 23`. 27 of 7,886 strings.
* Partially overlapping special tokens: llama.cpp takes the longest token first, HF the leftmost. Example
  `|||PHONE_NUMBER|||||EMAIL_ADDRESS|||`: llama.cpp `8651 91 76098 19925 8651 100262`, HF and ours
  `100261 8651 37004 20739 8651 91`. Only the three `|||` tokens can overlap.
* `'` followed by U+017F: HF folds it to `'s`; llama.cpp's source lower-cases ASCII only. No id-level difference
  was observed in my strings (BPE gave the same ids), so this is from reading the source plus HF piece equality.

Negative controls (`make negative`, each exits 1 as required): Qwen-style single-digit `\p{N}` (118 id mismatches on
goldens), ASCII-only letter class (10), no optional prefix char (161), control tokens parsed with
`parse_special=false` (15), no Unicode cutoff (10 HF mismatches, 23,311 piece differences); template: no newline
after `<think>` (3 of 23 goldens differ), `","` separator (10), `ensure_ascii` on (1), no newline before
`</tool_response>` (5), lstrip without Unicode spaces (2 of 1000), float repr threshold off by one (10 of 1000).
The script refuses a mutation that changes nothing, and treats a non-1 exit as a broken control.

Chat template: 23/23 goldens byte-identical. 1,000 seeded conversations (seed 20261110) rendered by jinja2
the way the golden generator does it: 1,000/1,000 agree, including the 83 where jinja2 raises (we error too);
one-off seed 99 x 6,000: 6,000/6,000 (546 errors agree). Covered: system or none, mid-conversation system, multi-turn,
reasoning_content present/null/empty/absent, `<think>` already in content with Unicode leading whitespace,
0..3 tool calls (wrapped and flat form), nested arguments with unicode/escapes/control chars/ints/big ints/floats
(1e-5, 1e16, -0.0, 1e22, NaN, Infinity), arguments given as a JSON string, nested tool schemas, 1..3 consecutive
tool results, null/empty/missing content, unknown roles, `add_generation_prompt` and `clear_thinking` on/off/absent.

Template hash: the GGUF `tokenizer.chat_template` is byte-identical to `ref/hf/chat_template.jinja` (2,652 bytes
each), sha256 `877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3` (the constant in the header).
(The minimal ChatML template in `tokenizer_config.json` is a different, tools-less one; spec 3.12.)

Speed (`bench`, g++ -O2, one core, this host under load): 1,000,068 bytes of mixed text -> 411,308 tokens, encode
74.7 ms best of 5 (13.4 MB/s, 5.5 Mtok/s); decode 6 ms; one 200,000-byte word without whitespace: 48 ms.

## Differences from Jinja that the renderer reports as errors or does not model

`tools`/`tool_calls` that are not arrays (Jinja would iterate a string's characters or a dict's keys), non-string
`reasoning_content`, non-object messages (Jinja treats them as all-undefined; we skip them only because they match
no role), lone-surrogate `\u` escapes (become U+FFFD), non-UTF-8 input. `add_generation_prompt`/`clear_thinking`
use Python truthiness. Only the jinja2 sandboxed environment of the golden generator was compared; transformers'
own `apply_chat_template` and llama.cpp's minja renderer were not run.

## Not verified / not done

* Hugging Face `tokenizers` 0.23.3 only (pip into a private venv `/server/staging/ripple-gufo/s1-text/hfvenv`);
  other versions or a different Oniguruma Unicode level could move the cutoff. ICU 78.2 only on this host; an ICU
  older than 76 fails `UnicodeDataIsCurrent()` and would misclassify Unicode 16 characters.
* Invalid UTF-8 as encoder input is passed through byte-wise (each bad byte counts as an "other symbol");
  not compared with any oracle.
* No gufo build, no wiring into `extension.cmake`/`extension.json` (other worker). Message-content-as-text spans (so
  a user's literal `<|im_end|>` is not a control id) are not implemented; see `OUTPUT-FORMAT.md`.
* The exhaustive all-code-point id comparison (4.4M strings) was run by hand (llama in 4 parallel parts), not
  through make; the piece-level one is in `make test`.

## Reproduce

    make -C tests/text -j3 test        # builds, regenerates seeded data under /server/staging/ripple-gufo/s1-text, runs everything
    make -C tests/text negative        # negative controls only
Tools in `tools/text/`: `gen_tok_prop.py`, `gen_chat_prop.py`, `gen_cp_probes.py`, `hf_tok_batch.py`,
`hf_pieces_batch.py`, `make_hf_nospecial.py`, `llama_tok_batch.cpp`.

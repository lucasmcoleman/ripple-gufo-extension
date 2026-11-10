# Ripple output format (notes for the streaming reasoning / tool-call parser, a later stage)

Sources: `ref/hf/chat_template.jinja`, `ref/hf/inference/server/ripple_format.py` (the vendor's SGLang
parsers, which define the format in executable form), spec section 3.12. No code here.

## What the model writes

The generation prompt is `<|im_start|>assistant\n`. It does NOT open `<think>`; there is no thinking
switch (`force_reasoning` is ignored by the vendor parser). The model decides what its first characters are.
Every turn ends with `<|im_end|>` (id 100265, EOS); `<|endoftext|>` (100257) is the second stop id.

    A  plain answer      {content}<|im_end|>
    B  with reasoning    <think>\n{reasoning}\n</think>\n{content}<|im_end|>
    C  tool call(s)      [<think>\n{reasoning}\n</think>\n][{text}]\n<tool_call>\n{"name": "f", "arguments": {...}}\n</tool_call>
                         ...repeated per call...<|im_end|>
       reasoning + calls, empty text:  <think>\nR\n</think>\n\n<tool_call>\n{...}\n</tool_call><|im_end|>   (blank line)

* Output is reasoning only if it STARTS with `<think>` (no leading whitespace); otherwise all of it is content.
* The template adds one separator newline after `<think>`, one before `</think>`, one after `</think>`, and one
  before each `<tool_call>`. The vendor parser strips exactly those so an assistant message sent back unchanged
  re-renders to the tokens the model produced (keeping them doubles the newlines each turn).
* Call JSON layout is fixed by the template: `{"name": <json string>, "arguments": <json value>}` with
  `", "` and `": "` separators, one call per `<tool_call>` block. Arguments should be an object (the template's
  own example says so) but the template renders any JSON value, so a parser must not assume an object.
* `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<tool_response>` are plain text, tokenised by BPE into
  several ids whose boundaries do not align with the tags. Only `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`
  are single special ids.
* `<|im_end|>` / `<|endoftext|>` must be detected as ids (`Tokenizer::IsStopToken`) and never sent as text
  (`StreamDecoder` with `SpecialDecode::kSkip`).

## What a streaming parser needs

1. Work on decoded text from `StreamDecoder` (complete UTF-8 only), never on token ids; tags are split across tokens.
2. A small state machine, as in the vendor parser: `start -> lead -> think -> after -> content`.
   * `start`: buffer while the text so far is a strict prefix of `<think>`; commit to reasoning if it begins with
     it, otherwise to content. This holds back at most 6 characters of an ordinary answer.
   * `think`: emit reasoning but hold back any suffix that could still become `\n</think>` or `</think>`
     (the newline before `</think>` is a separator, not reasoning). After `</think>` drop one `\n`.
   * End of stream while in `lead`/`think` = truncated reasoning (max_tokens): flush the buffer as reasoning
     (or as content if the API wants a non-empty content fallback).
3. Tool calls are looked for in the CONTENT stream after the reasoning split. Hold back a suffix that could become
   `<tool_call>`; one `\n` directly before the first `<tool_call>` is the template separator (strip it, keep
   other whitespace of the text before it).
4. Inside a call: after `<tool_call>\n{"name": ` read a JSON string, then `, "arguments": `, then stream the
   arguments value with a string-aware bracket matcher (escapes, `"` inside strings, nested objects, unicode) up
   to the closing `}` and `\n</tool_call>`. Stream argument text as deltas; the name is known before arguments.
   Several calls may follow back to back; whitespace between them is the template newline.
5. Failure modes to handle: truncated call at max_tokens (do not emit a half call as complete), invalid JSON,
   `</think>\n<tool_call>` with a single newline, text after a call (the vendor parser drops it; the template
   cannot render it back), `<tool_call>` appearing inside reasoning (stays reasoning), unbalanced `<think>`.
6. Constrained decoding for `tool_choice=required`/strict tools: the vendor structure is begin
   `<tool_call>\n{"name": "<fn>", "arguments": ` end `}\n</tool_call>`, trigger `<tool_call>`.

## Consequences for the engine

* Re-rendering history is only byte-stable for what the template can express. Arguments are re-serialised with
  Python `json.dumps` rules (`RenderChat`): a model-written `1e2` or `1.0` comes back as `100.0`/`1.0`, key order
  is kept, `ensure_ascii` is off. If the stored assistant turn is the raw model text instead, prefix-cache reuse is
  exact; if it is rebuilt from parsed JSON, expect the prompt prefix to differ in number formatting only.
* `clear_thinking=true` drops `reasoning_content` when rendering history; default keeps it.
* A user/tool message that contains `<|im_end|>` text is tokenised as the control id when the rendered prompt is
  encoded with `parse_special=true` (same as Hugging Face). Upstream gufo's Qwen path encodes message content spans
  as plain text to prevent this injection; that is not implemented here (would need content spans out of
  `RenderChat`) and is a scope decision for the wiring stage.

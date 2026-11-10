// Hand-written renderer for the Ripple-65B-Preview chat template
// (ref/hf/chat_template.jinja). It is not a Jinja engine: it reproduces the
// output of that one template, byte for byte, as rendered by jinja2 in the
// transformers-style environment (sandboxed, trim_blocks, lstrip_blocks,
// `tojson` = json.dumps(ensure_ascii=False) with default separators).
//
// Template variables read: `messages`, `tools`, `add_generation_prompt`,
// `clear_thinking` (all others are unused by the template).
#ifndef RICURSIVE_TEXT_CHAT_TEMPLATE_HPP_
#define RICURSIVE_TEXT_CHAT_TEMPLATE_HPP_

#include <optional>
#include <string>
#include <string_view>

#include "json_value.hpp"

namespace ricursive::text {

/// sha256 of the bytes of chat_template.jinja that this renderer implements.
/// The engine should refuse a GGUF whose `tokenizer.chat_template` hashes to
/// anything else (the GGUF in hand carries a different one, see README.md).
inline constexpr std::string_view kRippleChatTemplateSha256 =
    "877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3";

struct ChatRenderOptions {
  bool add_generation_prompt{false};  ///< template: add_generation_prompt
  bool clear_thinking{false};         ///< template: clear_thinking | default(false)
};

/// Renders `messages` (JSON array of objects with role/content/reasoning_content/
/// tool_calls) and optional `tools` (JSON array, each rendered verbatim with
/// tojson). Returns nullopt and sets *error where jinja2 would raise
/// (null/non-string content where a string is concatenated, empty `messages`,
/// tool call without name/arguments, ...). Inputs whose Jinja behaviour depends
/// on duck typing (a non-array `tools`/`tool_calls`, which Jinja would iterate
/// as a string or dict) are reported as errors; see README.md.
[[nodiscard]] std::optional<std::string> RenderChat(
    const JsonValue& messages, const JsonValue* tools,
    const ChatRenderOptions& options = {}, std::string* error = nullptr);

}  // namespace ricursive::text

#endif  // RICURSIVE_TEXT_CHAT_TEMPLATE_HPP_

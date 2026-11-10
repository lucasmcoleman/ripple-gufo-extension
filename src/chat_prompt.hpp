#ifndef RICURSIVE_CHAT_PROMPT_HPP_
#define RICURSIVE_CHAT_PROMPT_HPP_

// Chat prompt = src/text's renderer + tokenizer, with message text that cannot
// inject control tokens.
//
// RenderChat gives one string and no span boundaries, so the spans are found
// from its output: every string the caller supplied (message content,
// reasoning, tool names and arguments, tool schemas) is replaced by a unique
// marker before rendering, the template runs on the markers, and the markers
// in its output are replaced by the originals again. What is left between the
// markers is the template's own text; the control tokens found there
// (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`) are the only ones the
// tokenizer is told to emit. The rest, between them, is encoded as plain text,
// so a user's literal "<|im_end|>" becomes ordinary byte tokens.
//
// Cutting at those control tokens gives the same ids as encoding the whole
// string with special-token parsing, as long as no supplied text contains one
// (tested against the goldens and llama-tokenize).

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "text/chat_template.hpp"
#include "text/tokenizer.hpp"

namespace gufo::models::ricursive {

struct RenderedChat {
  std::string text;  ///< byte-identical to RenderChat on the original input
  struct Special {
    std::size_t offset{0};  ///< in `text`; inside the template's own text
    std::size_t length{0};
    std::string text;
  };
  std::vector<Special> specials;  ///< ascending by offset
};

/// Same inputs and errors as ::ricursive::text::RenderChat.
[[nodiscard]] std::optional<RenderedChat> RenderChatProtected(
    const ::ricursive::text::JsonValue& messages,
    const ::ricursive::text::JsonValue* tools,
    const ::ricursive::text::ChatRenderOptions& options, std::string* error);

/// Token ids of a rendered chat: control tokens only where the template put
/// them. Text between them is encoded without special-token parsing.
[[nodiscard]] std::vector<::ricursive::text::TokenId> TokenizeChat(
    const ::ricursive::text::Tokenizer& tokenizer, const RenderedChat& chat);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_CHAT_PROMPT_HPP_

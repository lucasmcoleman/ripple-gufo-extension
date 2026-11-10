#ifndef RICURSIVE_CHAT_REQUEST_HPP_
#define RICURSIVE_CHAT_REQUEST_HPP_

// Host chat types -> the JSON the template renderer reads.

#include <gufo/ext/text.hpp>
#include <optional>
#include <span>
#include <string>

#include "engine.hpp"

namespace gufo::models::ricursive {

/// messages / tools as the template sees them. The Ripple template knows the
/// roles system, user, assistant and tool; a developer message is rendered as
/// a system message. Throws std::invalid_argument for images.
[[nodiscard]] ::ricursive::text::JsonValue MessagesToJson(std::span<const ext::ChatMessage> messages);
/// Null-less: an empty array when there are no tools.
[[nodiscard]] ::ricursive::text::JsonValue ToolsToJson(std::span<const ext::ChatTool> tools);

/// Renders and tokenizes `request` (tool_choice none sends no tools). Throws
/// std::invalid_argument with the template's error.
[[nodiscard]] ChatPrompt PrepareRequest(const Model& model, const ext::ChatRequest& request);
/// The rendered text, for diagnostics.
[[nodiscard]] std::string RenderRequest(const ext::ChatRequest& request);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_CHAT_REQUEST_HPP_

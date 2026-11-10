#ifndef RICURSIVE_OUTPUT_PARSER_HPP_
#define RICURSIVE_OUTPUT_PARSER_HPP_

// Streaming split of Ripple's output into reasoning, content and tool calls
// (format: src/text/OUTPUT-FORMAT.md) and the scanner the host's tool dialect
// calls. Works on decoded text, never on token ids.
//
// Aggregate output (the concatenation of every Push/Finish result) depends
// only on the whole text, never on where it was cut: a decision is taken only
// when later bytes cannot change it, and whatever might still start a tag is
// held back. Rules:
//  - Output is reasoning only if it STARTS with `<think>` (nothing before it);
//    `<think>` + one optional "\n" opens, one optional "\n" before `</think>`
//    and one after it are template separators and dropped.
//  - `<tool_call>` is looked for in the content after the reasoning. One "\n"
//    before it is the template's separator. Between calls, whitespace-only
//    text is dropped. Text after a call is kept as content (the vendor's
//    parser drops it; losing output silently is worse).
//  - A call is `<tool_call>` ws JSON-object ws `</tool_call>` with a string
//    "name" and an "arguments" member. The object end is found by a
//    string-aware bracket matcher, so tags inside JSON strings are data.
//  - A malformed block is released as content, starting with its opening tag.
//    An unterminated one (end of stream inside a call) is not a call and is
//    reported in `unterminated`; unterminated reasoning is flushed as reasoning.

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gufo/ext/text.hpp"

namespace gufo::models::ricursive {

struct ParsedToolCall {
  std::string name;
  /// Python-style json.dumps of the "arguments" value (any JSON value).
  std::string arguments_json;
  bool operator==(const ParsedToolCall&) const = default;
};

struct ParserEvents {
  std::string reasoning;
  std::string content;
  std::vector<ParsedToolCall> calls;
  /// Finish() only: raw text of a `<tool_call>` block that never ended.
  std::string unterminated;
  /// `<tool_call>` blocks released as content because they were malformed.
  std::size_t malformed{0};
  void Append(ParserEvents&& other);
  bool operator==(const ParserEvents&) const = default;
};

class OutputParser {
public:
  /// Deliberate defects for tests (all off in normal use).
  struct Faults {
    /// A call ends at the first `</tool_call>` even inside a JSON string.
    bool close_call_on_tag{false};
    /// Reasoning is not held back: `\n</think>` can leak its newline.
    bool no_think_holdback{false};
  };
  OutputParser() = default;
  explicit OutputParser(Faults faults) : faults_(faults) {}

  [[nodiscard]] ParserEvents Push(std::string_view text);
  /// End of stream: flushes everything held back. Call once.
  [[nodiscard]] ParserEvents Finish();

private:
  enum class State { kStart, kThinkLead, kThink, kAfterThink, kContent, kCall };
  void Run(bool finishing, ParserEvents* out);
  void EmitContent(std::string_view text, ParserEvents* out);

  Faults faults_;
  State state_{State::kStart};
  std::string buffer_;
  bool calls_seen_{false};
  bool emitted_since_call_{false};
};

/// Result of reading one call block at the start of `text`.
struct CallBlock {
  enum class Kind { kIncomplete, kMalformed, kComplete };
  Kind kind{Kind::kIncomplete};
  std::size_t end{0};  ///< kComplete: bytes up to and including `</tool_call>`
  ParsedToolCall call;
  /// kComplete: "arguments" is a JSON object, so it fits the host's
  /// name/arguments call shape.
  bool arguments_object{false};
};
/// `text` must start with `<tool_call>`.
[[nodiscard]] CallBlock ReadCallBlock(std::string_view text,
                                      bool close_on_tag = false);

inline constexpr std::string_view kToolDialectId = "ricursive";
inline constexpr std::string_view kToolCallOpen = "<tool_call>";
inline constexpr std::string_view kToolCallClose = "</tool_call>";
inline constexpr std::string_view kThinkOpen = "<think>";
inline constexpr std::string_view kThinkClose = "</think>";

/// `ext::ToolDialect::scan_tool_calls`.
void ScanToolCalls(std::string_view text,
                   std::span<const std::string_view> markers,
                   std::vector<ext::ScannedToolCall>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans,
                   bool partial);

/// The dialect the extension registers: `<tool_call>` + "\n{"name": " ...
/// `}\n</tool_call>`, the shape the chat template renders.
[[nodiscard]] ext::ToolDialect MakeToolDialect();
/// `<think>` / `</think>`.
[[nodiscard]] ext::ReasoningMarkup MakeReasoningDelimiters();

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_OUTPUT_PARSER_HPP_

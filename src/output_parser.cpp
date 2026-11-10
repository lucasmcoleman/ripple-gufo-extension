#include "output_parser.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>

#include "text/json_value.hpp"

namespace gufo::models::ricursive {
namespace {

namespace rt = ::ricursive::text;

constexpr std::size_t kNeedMore = std::string_view::npos;
constexpr std::size_t kBad = std::string_view::npos - 1;

bool IsWs(char c) {
  return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}
bool AllWs(std::string_view s) {
  return std::ranges::all_of(s, IsWs);
}
std::size_t SkipWs(std::string_view t, std::size_t p) {
  while (p < t.size() && IsWs(t[p])) ++p;
  return p;
}

/// Longest suffix of `buffer`, shorter than `pattern`, that is a prefix of it.
std::size_t SuffixPrefix(std::string_view buffer, std::string_view pattern) {
  const std::size_t most = std::min(buffer.size(), pattern.size() - 1);
  for (std::size_t k = most; k > 0; --k) {
    if (pattern.starts_with(buffer.substr(buffer.size() - k))) return k;
  }
  return 0;
}

/// `t[pos]` is '"': index after the closing quote, kNeedMore if it has not
/// arrived.
std::size_t StringEnd(std::string_view t, std::size_t pos) {
  bool escape = false;
  for (std::size_t i = pos + 1; i < t.size(); ++i) {
    if (escape) {
      escape = false;
    } else if (t[i] == '\\') {
      escape = true;
    } else if (t[i] == '"') {
      return i + 1;
    }
  }
  return kNeedMore;
}

/// `t[pos]` is '{' or '[': index after the matching closer (string-aware),
/// kNeedMore, or kBad for a mismatched closer.
std::size_t BracketEnd(std::string_view t, std::size_t pos) {
  std::string stack;
  bool in_string = false;
  bool escape = false;
  for (std::size_t i = pos; i < t.size(); ++i) {
    const char c = t[i];
    if (in_string) {
      if (escape) {
        escape = false;
      } else if (c == '\\') {
        escape = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
    } else if (c == '{' || c == '[') {
      stack.push_back(c == '{' ? '}' : ']');
    } else if (c == '}' || c == ']') {
      if (stack.empty() || stack.back() != c) return kBad;
      stack.pop_back();
      if (stack.empty()) return i + 1;
    }
  }
  return kNeedMore;
}

std::optional<rt::JsonValue> Parse(std::string_view text) {
  rt::JsonValue value;
  if (!rt::ParseJson(text, &value)) return std::nullopt;
  return value;
}

/// Decodes a string body (without quotes) that may end inside an escape: an
/// incomplete escape and a trailing high surrogate are held back, so the
/// result only grows as text arrives.
std::string UnescapePrefix(std::string_view raw) {
  std::size_t keep = raw.size();
  // Trailing partial escape: backslash run of odd length, or \uXXXX cut short.
  std::size_t back = 0;
  while (back < raw.size() && raw[raw.size() - 1 - back] == '\\') ++back;
  if (back % 2 == 1) keep = raw.size() - 1;
  const std::size_t u = raw.substr(0, keep).rfind("\\u");
  if (u != std::string_view::npos && keep - u < 6) keep = u;
  // A complete \uD8xx..\uDBxx escape at the end waits for its low half.
  if (keep >= 6 && raw[keep - 6] == '\\' && raw[keep - 5] == 'u' &&
      (raw[keep - 4] == 'd' || raw[keep - 4] == 'D') &&
      std::strchr("89abAB", raw[keep - 3]) != nullptr) {
    keep -= 6;
  }
  std::string quoted = "\"";
  quoted.append(raw.substr(0, keep));
  quoted.push_back('"');
  if (auto v = Parse(quoted); v && v->is_string()) return v->text();
  return {};
}

}  // namespace

void ParserEvents::Append(ParserEvents&& other) {
  reasoning += other.reasoning;
  content += other.content;
  for (auto& c : other.calls) calls.push_back(std::move(c));
  unterminated += other.unterminated;
  malformed += other.malformed;
}

CallBlock ReadCallBlock(std::string_view text, bool close_on_tag) {
  CallBlock block;
  std::size_t begin = kToolCallOpen.size();
  std::size_t json_end = 0;
  std::size_t end = 0;
  if (close_on_tag) {
    const std::size_t close = text.find(kToolCallClose, begin);
    if (close == std::string_view::npos) return block;
    json_end = close;
    end = close + kToolCallClose.size();
  } else {
    begin = SkipWs(text, begin);
    if (begin >= text.size()) return block;
    if (text[begin] != '{') {
      block.kind = CallBlock::Kind::kMalformed;
      return block;
    }
    json_end = BracketEnd(text, begin);
    if (json_end == kNeedMore) return block;
    if (json_end == kBad) {
      block.kind = CallBlock::Kind::kMalformed;
      return block;
    }
    const std::size_t after = SkipWs(text, json_end);
    const std::string_view rest = text.substr(after);
    if (rest.size() < kToolCallClose.size()) {
      if (kToolCallClose.starts_with(rest)) return block;
      block.kind = CallBlock::Kind::kMalformed;
      return block;
    }
    if (!rest.starts_with(kToolCallClose)) {
      block.kind = CallBlock::Kind::kMalformed;
      return block;
    }
    end = after + kToolCallClose.size();
  }
  const auto object = Parse(text.substr(begin, json_end - begin));
  const rt::JsonValue* name = object ? object->Find("name") : nullptr;
  const rt::JsonValue* arguments = object ? object->Find("arguments") : nullptr;
  if (!object || !object->is_object() || name == nullptr ||
      !name->is_string() || arguments == nullptr) {
    block.kind = CallBlock::Kind::kMalformed;
    return block;
  }
  block.kind = CallBlock::Kind::kComplete;
  block.end = end;
  block.call = {name->text(), rt::DumpPythonJson(*arguments)};
  block.arguments_object = arguments->is_object();
  return block;
}

void OutputParser::EmitContent(std::string_view text, ParserEvents* out) {
  if (text.empty()) return;
  if (calls_seen_ && !emitted_since_call_ && AllWs(text)) return;
  out->content += text;
  emitted_since_call_ = true;
}

void OutputParser::Run(bool finishing, ParserEvents* out) {
  for (;;) {
    switch (state_) {
      case State::kStart:
        if (buffer_.starts_with(kThinkOpen)) {
          buffer_.erase(0, kThinkOpen.size());
          state_ = State::kThinkLead;
          continue;
        }
        if (!finishing && kThinkOpen.starts_with(buffer_)) return;
        state_ = State::kContent;
        continue;
      case State::kThinkLead:
        if (buffer_.empty()) return;
        if (buffer_[0] == '\n') buffer_.erase(0, 1);
        state_ = State::kThink;
        continue;
      case State::kThink: {
        const std::size_t pos = buffer_.find(kThinkClose);
        if (pos != std::string::npos) {
          std::string_view text(buffer_.data(), pos);
          if (text.ends_with('\n')) text.remove_suffix(1);
          out->reasoning += text;
          buffer_.erase(0, pos + kThinkClose.size());
          state_ = State::kAfterThink;
          continue;
        }
        std::size_t hold = 0;
        if (!finishing && !faults_.no_think_holdback) {
          hold = std::max(SuffixPrefix(buffer_, "\n</think>"),
                          SuffixPrefix(buffer_, kThinkClose));
        }
        out->reasoning.append(buffer_, 0, buffer_.size() - hold);
        buffer_.erase(0, buffer_.size() - hold);
        return;
      }
      case State::kAfterThink:
        if (buffer_.empty()) return;
        if (buffer_[0] == '\n') buffer_.erase(0, 1);
        state_ = State::kContent;
        continue;
      case State::kContent: {
        const std::size_t pos = buffer_.find(kToolCallOpen);
        if (pos != std::string::npos) {
          std::string_view text(buffer_.data(), pos);
          if (text.ends_with('\n')) text.remove_suffix(1);
          EmitContent(text, out);
          buffer_.erase(0, pos);
          state_ = State::kCall;
          continue;
        }
        std::size_t hold = 0;
        if (!finishing) {
          hold = std::max(SuffixPrefix(buffer_, "\n<tool_call>"),
                          SuffixPrefix(buffer_, kToolCallOpen));
        }
        const std::size_t n = buffer_.size() - hold;
        const std::string_view head(buffer_.data(), n);
        if (calls_seen_ && !emitted_since_call_ && AllWs(head)) {
          // Whitespace after a call may still be the separator of the next.
          if (finishing) buffer_.clear();
          return;
        }
        EmitContent(head, out);
        buffer_.erase(0, n);
        return;
      }
      case State::kCall: {
        const CallBlock block = ReadCallBlock(buffer_, faults_.close_call_on_tag);
        if (block.kind == CallBlock::Kind::kComplete) {
          out->calls.push_back(block.call);
          buffer_.erase(0, block.end);
          state_ = State::kContent;
          calls_seen_ = true;
          emitted_since_call_ = false;
          continue;
        }
        if (block.kind == CallBlock::Kind::kMalformed) {
          ++out->malformed;
          EmitContent(kToolCallOpen, out);
          buffer_.erase(0, kToolCallOpen.size());
          state_ = State::kContent;
          continue;
        }
        if (finishing) {
          out->unterminated = buffer_;
          buffer_.clear();
        }
        return;
      }
    }
  }
}

ParserEvents OutputParser::Push(std::string_view text) {
  ParserEvents out;
  buffer_.append(text);
  Run(false, &out);
  return out;
}

ParserEvents OutputParser::Finish() {
  ParserEvents out;
  Run(true, &out);
  return out;
}

namespace {

std::vector<ext::ScannedToolArgument> Arguments(const ParsedToolCall& call) {
  std::vector<ext::ScannedToolArgument> args;
  const auto value = Parse(call.arguments_json);
  if (!value || !value->is_object()) return args;
  for (const auto& [key, member] : value->object()) {
    if (member.is_string()) {
      args.push_back({key, member.text(), true});
    } else {
      args.push_back({key, rt::DumpPythonJson(member), false});
    }
  }
  return args;
}

/// The call a text ending inside it could still complete: its name once
/// closed, the arguments whose value has ended, and a string value still open.
void PartialCall(std::string_view t, std::vector<ext::ScannedToolCall>* calls) {
  std::size_t p = SkipWs(t, kToolCallOpen.size());
  const auto expect = [&](std::string_view literal) {
    p = SkipWs(t, p);
    if (!t.substr(p).starts_with(literal)) return false;
    p += literal.size();
    return true;
  };
  if (!expect("{") || !expect("\"name\"") || !expect(":")) return;
  p = SkipWs(t, p);
  if (p >= t.size() || t[p] != '"') return;
  const std::size_t name_end = StringEnd(t, p);
  if (name_end == kNeedMore) return;
  const auto name = Parse(t.substr(p, name_end - p));
  if (!name || !name->is_string()) return;
  p = name_end;
  if (!expect(",") || !expect("\"arguments\"") || !expect(":") || !expect("{")) {
    return;
  }
  ext::ScannedToolCall call{.name = name->text(), .valid = true};
  while (true) {
    p = SkipWs(t, p);
    if (p >= t.size() || t[p] == '}') break;
    if (t[p] == ',') {
      ++p;
      continue;
    }
    if (t[p] != '"') {
      call.valid = false;
      break;
    }
    const std::size_t key_end = StringEnd(t, p);
    if (key_end == kNeedMore) break;
    const auto key = Parse(t.substr(p, key_end - p));
    if (!key || !key->is_string()) break;
    p = key_end;
    if (!expect(":")) break;
    p = SkipWs(t, p);
    if (p >= t.size()) break;
    if (t[p] == '"') {
      const std::size_t end = StringEnd(t, p);
      if (end == kNeedMore) {
        call.unfinished = ext::ScannedToolArgument{
            key->text(), UnescapePrefix(t.substr(p + 1)), true};
        break;
      }
      const auto value = Parse(t.substr(p, end - p));
      if (!value) break;
      call.arguments.push_back({key->text(), value->text(), true});
      p = end;
      continue;
    }
    std::size_t end = kNeedMore;
    if (t[p] == '{' || t[p] == '[') {
      end = BracketEnd(t, p);
    } else {
      // A bare scalar may still grow until a delimiter follows it.
      const std::size_t stop = t.find_first_of(",}] \n\r\t", p);
      end = stop == std::string_view::npos ? kNeedMore : stop;
    }
    if (end == kNeedMore || end == kBad) break;
    const auto value = Parse(t.substr(p, end - p));
    if (!value) break;
    call.arguments.push_back({key->text(), rt::DumpPythonJson(*value),
                              false});
    p = end;
  }
  if (call.valid) calls->push_back(std::move(call));
}

}  // namespace

void ScanToolCalls(std::string_view text,
                   std::span<const std::string_view> markers,
                   std::vector<ext::ScannedToolCall>* calls,
                   std::vector<std::pair<std::size_t, std::size_t>>* spans,
                   bool partial) {
  const std::string_view open = markers.empty() ? kToolCallOpen : markers.front();
  std::size_t cursor = 0;
  while (true) {
    const std::size_t at = text.find(open, cursor);
    if (at == std::string_view::npos) return;
    const CallBlock block = ReadCallBlock(text.substr(at));
    if (block.kind == CallBlock::Kind::kMalformed) {
      cursor = at + open.size();  // stays content
      continue;
    }
    if (block.kind == CallBlock::Kind::kIncomplete) {
      if (partial) PartialCall(text.substr(at), calls);
      return;
    }
    if (spans != nullptr) spans->emplace_back(at, at + block.end);
    ext::ScannedToolCall scanned{.name = block.call.name,
                                 .arguments = Arguments(block.call),
                                 .valid = block.arguments_object,
                                 .complete = true};
    calls->push_back(std::move(scanned));
    cursor = at + block.end;
  }
}

ext::ToolDialect MakeToolDialect() {
  sampling::JsonConstraint::ToolSyntax syntax;
  syntax.open = std::string(kToolCallOpen);
  syntax.name_prefix = "\n{\"name\": \"";
  syntax.name_suffix = "\", \"arguments\": ";
  syntax.close = "}\n</tool_call>";
  syntax.name_delimiters = "\"";
  return {
      .markers = {kToolCallOpen},
      .reasoning_markers = {},
      .closers = {},
      .scan_tool_calls = &ScanToolCalls,
      .id = std::string(kToolDialectId),
      .syntax = std::move(syntax),
  };
}

ext::ReasoningMarkup MakeReasoningDelimiters() {
  return {std::string(kThinkOpen), std::string(kThinkClose)};
}

}  // namespace gufo::models::ricursive

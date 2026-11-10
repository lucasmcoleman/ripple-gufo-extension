#include "chat_prompt.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <random>

namespace gufo::models::ricursive {
namespace {

namespace rt = ::ricursive::text;
using rt::JsonValue;

constexpr std::string_view kControlTexts[] = {"<|im_start|>", "<|im_end|>",
                                              "<|endoftext|>"};

// Python str.isspace() code points, as the template's content.lstrip().
bool PyIsSpace(std::uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 ||
         c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
         c == 0x3000;
}

bool LstripStartsWith(const std::string& s, std::string_view prefix) {
  std::size_t i = 0;
  while (i < s.size()) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    const std::size_t len = b0 < 0x80 ? 1 : (b0 >> 5) == 6 ? 2 : (b0 >> 4) == 14 ? 3 : (b0 >> 3) == 30 ? 4 : 1;
    if (i + len > s.size()) break;
    std::uint32_t cp = len == 1 ? b0 : (b0 & (0xFFU >> (len + 1)));
    for (std::size_t k = 1; k < len; ++k) {
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3FU);
    }
    if (!PyIsSpace(cp)) break;
    i += len;
  }
  return std::string_view(s).substr(i).starts_with(prefix);
}

struct Entry {
  std::string original;
  bool json{false};   ///< rendered inside a JSON string (tojson): escape it
  bool think{false};  ///< the marker carries a leading "<think>" (see below)
};

class Marker {
 public:
  Marker(const JsonValue& messages, const JsonValue* tools) {
    const std::string haystack =
        rt::DumpPythonJson(messages) + (tools != nullptr ? rt::DumpPythonJson(*tools) : "");
    std::mt19937_64 rng(std::random_device{}());
    do {
      nonce_ = "RICPL" + std::to_string(rng()) + "x";
    } while (haystack.find(nonce_) != std::string::npos);
  }
  [[nodiscard]] const std::string& nonce() const { return nonce_; }
  [[nodiscard]] const std::vector<Entry>& entries() const { return entries_; }

  JsonValue Replace(const std::string& text, bool json, bool think = false) {
    if (text.empty()) return JsonValue::String(text);  // emptiness is semantic
    entries_.push_back({text, json, think});
    return JsonValue::String(Token(entries_.size() - 1, think));
  }
  /// Every string and every key below `value` (tools, call arguments).
  JsonValue Deep(const JsonValue& value) {
    switch (value.type()) {
      case JsonValue::Type::kString:
        return Replace(value.text(), true);
      case JsonValue::Type::kArray: {
        JsonValue::Array out;
        for (const auto& v : value.array()) out.push_back(Deep(v));
        return JsonValue::MakeArray(std::move(out));
      }
      case JsonValue::Type::kObject: {
        JsonValue::Object out;
        for (const auto& [k, v] : value.object()) {
          out.emplace_back(k.empty() ? k : Replace(k, true).text(), Deep(v));
        }
        return JsonValue::MakeObject(std::move(out));
      }
      default:
        return value;
    }
  }
  [[nodiscard]] std::string Token(std::size_t index, bool think) const {
    // The template tests content.lstrip().startswith('<think>') and only then
    // skips reasoning_content; a marker must not change that decision.
    return std::string(think ? "<think>" : "") + "@@" + nonce_ + "_" +
           std::to_string(index) + "@@";
  }

 private:
  std::string nonce_;
  std::vector<Entry> entries_;
};

JsonValue MarkFunction(Marker& marker, const JsonValue& fn) {
  if (!fn.is_object()) return fn;
  JsonValue out = fn;
  if (const JsonValue* name = fn.Find("name"); name != nullptr && name->is_string()) {
    out.Set("name", marker.Replace(name->text(), true));
  }
  if (const JsonValue* args = fn.Find("arguments"); args != nullptr) {
    out.Set("arguments", marker.Deep(*args));
  }
  return out;
}

// The template reads call.function.* when `function` exists, else the call.
JsonValue MarkCall(Marker& marker, const JsonValue& call) {
  if (!call.is_object()) return call;
  if (const JsonValue* fn = call.Find("function"); fn != nullptr) {
    JsonValue out = call;
    out.Set("function", MarkFunction(marker, *fn));
    return out;
  }
  return MarkFunction(marker, call);
}

JsonValue MarkMessage(Marker& marker, const JsonValue& message) {
  if (!message.is_object()) return message;
  JsonValue out = message;
  const JsonValue* role = message.Find("role");
  const bool assistant = role != nullptr && role->is_string() && role->text() == "assistant";
  if (const JsonValue* c = message.Find("content"); c != nullptr && c->is_string()) {
    out.Set("content", marker.Replace(c->text(), false,
                                      assistant && LstripStartsWith(c->text(), "<think>")));
  }
  if (const JsonValue* r = message.Find("reasoning_content"); r != nullptr && r->is_string()) {
    out.Set("reasoning_content", marker.Replace(r->text(), false));
  }
  if (const JsonValue* calls = message.Find("tool_calls"); calls != nullptr && calls->is_array()) {
    JsonValue::Array marked;
    for (const auto& call : calls->array()) marked.push_back(MarkCall(marker, call));
    out.Set("tool_calls", JsonValue::MakeArray(std::move(marked)));
  }
  return out;
}

std::string EscapedBody(const std::string& text) {
  std::string dumped = rt::DumpPythonJson(JsonValue::String(text));
  return dumped.substr(1, dumped.size() - 2);
}

}  // namespace

std::optional<RenderedChat> RenderChatProtected(
    const JsonValue& messages, const JsonValue* tools,
    const rt::ChatRenderOptions& options, std::string* error) {
  if (!messages.is_array()) {
    (void)rt::RenderChat(messages, tools, options, error);  // reports the error
    return std::nullopt;
  }
  Marker marker(messages, tools);
  JsonValue::Array marked_messages;
  for (const auto& m : messages.array()) marked_messages.push_back(MarkMessage(marker, m));
  const JsonValue marked_messages_value = JsonValue::MakeArray(std::move(marked_messages));
  std::optional<JsonValue> marked_tools;
  if (tools != nullptr && tools->is_array()) marked_tools = marker.Deep(*tools);
  const auto rendered = rt::RenderChat(
      marked_messages_value,
      marked_tools ? &*marked_tools : tools, options, error);
  if (!rendered) return std::nullopt;

  const std::string head = "@@" + marker.nonce() + "_";
  RenderedChat out;
  const auto scan_scaffold = [&](std::size_t from, std::size_t to) {
    // `text` already holds the scaffold at [from, to).
    for (const std::string_view control : kControlTexts) {
      for (std::size_t at = out.text.find(control, from);
           at != std::string::npos && at + control.size() <= to;
           at = out.text.find(control, at + 1)) {
        out.specials.push_back({at, control.size(), std::string(control)});
      }
    }
  };
  std::size_t cursor = 0;
  while (true) {
    std::size_t at = rendered->find(head, cursor);
    if (at == std::string::npos) break;
    std::size_t digits = at + head.size();
    std::size_t end = digits;
    while (end < rendered->size() && std::isdigit(static_cast<unsigned char>((*rendered)[end])) != 0) ++end;
    if (end == digits || rendered->compare(end, 2, "@@") != 0) {
      if (error != nullptr) *error = "internal: damaged chat marker";
      return std::nullopt;
    }
    const std::size_t index = std::stoull(rendered->substr(digits, end - digits));
    if (index >= marker.entries().size()) {
      if (error != nullptr) *error = "internal: unknown chat marker";
      return std::nullopt;
    }
    const Entry& entry = marker.entries()[index];
    std::size_t begin = at;
    if (entry.think) {
      if (at < 7 || rendered->compare(at - 7, 7, "<think>") != 0 || at - 7 < cursor) {
        if (error != nullptr) *error = "internal: chat marker lost its think prefix";
        return std::nullopt;
      }
      begin = at - 7;
    }
    const std::size_t scaffold_from = out.text.size();
    out.text.append(*rendered, cursor, begin - cursor);
    scan_scaffold(scaffold_from, out.text.size());
    out.text += entry.json ? EscapedBody(entry.original) : entry.original;
    cursor = end + 2;
  }
  const std::size_t scaffold_from = out.text.size();
  out.text.append(*rendered, cursor, std::string::npos);
  scan_scaffold(scaffold_from, out.text.size());
  std::ranges::sort(out.specials, {}, &RenderedChat::Special::offset);
  if (out.text.find("@@" + marker.nonce()) != std::string::npos) {
    if (error != nullptr) *error = "internal: chat marker left in the prompt";
    return std::nullopt;
  }
  return out;
}

std::vector<rt::TokenId> TokenizeChat(const rt::Tokenizer& tokenizer,
                                      const RenderedChat& chat) {
  std::vector<rt::TokenId> tokens;
  std::size_t cursor = 0;
  const auto plain = [&](std::size_t to) {
    if (to > cursor) {
      const auto ids = tokenizer.Encode(std::string_view(chat.text).substr(cursor, to - cursor), false);
      tokens.insert(tokens.end(), ids.begin(), ids.end());
    }
  };
  for (const auto& special : chat.specials) {
    const rt::TokenId id = tokenizer.FindToken(special.text);
    if (id == rt::kInvalidTokenId) continue;  // stays plain text
    plain(special.offset);
    tokens.push_back(id);
    cursor = special.offset + special.length;
  }
  plain(chat.text.size());
  return tokens;
}

}  // namespace gufo::models::ricursive

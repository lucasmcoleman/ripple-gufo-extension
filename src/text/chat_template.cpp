#include "chat_template.hpp"

#include <cstdint>

namespace ricursive::text {
namespace {

struct RenderError {
  std::string message;
};
[[noreturn]] void Fail(std::string m) { throw RenderError{std::move(m)}; }

// Jinja getattr/getitem on a dict: nullptr == Undefined (also for non-objects).
const JsonValue* Get(const JsonValue& v, std::string_view key) {
  return v.Find(key);
}

bool RoleIs(const JsonValue& msg, std::string_view role) {
  const JsonValue* r = Get(msg, "role");
  return r != nullptr && r->is_string() && r->text() == role;
}

// Python str.isspace() code points (what str.lstrip() removes).
bool PyIsSpace(std::uint32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 ||
         c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// content.lstrip().startswith(prefix)
bool LstripStartsWith(const std::string& s, std::string_view prefix) {
  std::size_t i = 0;
  while (i < s.size()) {
    auto b0 = static_cast<unsigned char>(s[i]);
    std::size_t len = b0 < 0x80 ? 1 : (b0 >> 5) == 6 ? 2 : (b0 >> 4) == 14 ? 3 : (b0 >> 3) == 30 ? 4 : 1;
    if (i + len > s.size()) break;
    std::uint32_t cp = len == 1 ? b0 : (b0 & (0xFF >> (len + 1)));
    for (std::size_t k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    if (!PyIsSpace(cp)) break;
    i += len;
  }
  return std::string_view(s).substr(i).starts_with(prefix);
}

// `a + b` where b must be a Python str.
const std::string& MustBeString(const JsonValue* v, const char* what) {
  if (v == nullptr || !v->is_string())
    Fail(std::string("TypeError/UndefinedError: ") + what + " is not a string");
  return v->text();
}

std::string RenderImpl(const JsonValue& messages, const JsonValue* tools,
                       const ChatRenderOptions& opt) {
  if (!messages.is_array()) Fail("messages is not an array");
  const auto& msgs = messages.array();
  std::string out;

  // messages[0]['role']: indexing an empty list yields Undefined, whose
  // subscript raises UndefinedError.
  if (msgs.empty()) Fail("UndefinedError: messages[0] is undefined");
  const bool first_is_system = RoleIs(msgs[0], "system");

  if (tools != nullptr && tools->Truthy()) {
    if (!tools->is_array()) Fail("tools is not an array");
    out += "<|im_start|>system\n";
    if (first_is_system) {
      out += MustBeString(Get(msgs[0], "content"), "messages[0].content");
      out += "\n\n";
    }
    out += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
           "You are provided with function signatures within <tools> XML tags:\n<tools>";
    for (const auto& tool : tools->array()) {
      out += "\n";
      AppendPythonJson(out, tool);
    }
    out += "\n</tools>\n\nFor each function call, return exactly one valid JSON object with function name and "
           "arguments inside XML tool-call tags. The arguments field must be a JSON object, not a string, and "
           "strings must use valid JSON escaping:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": "
           "{\"command\": \"pwd\"}}\n</tool_call><|im_end|>\n";
  } else if (first_is_system) {
    out += "<|im_start|>system\n";
    out += MustBeString(Get(msgs[0], "content"), "messages[0].content");
    out += "<|im_end|>\n";
  }

  for (std::size_t i = 0; i < msgs.size(); ++i) {
    const JsonValue& msg = msgs[i];
    const bool is_user = RoleIs(msg, "user");
    const bool is_assistant = RoleIs(msg, "assistant");
    const bool is_system = RoleIs(msg, "system");
    const bool is_tool = RoleIs(msg, "tool");

    // {% set content = message.content or '' %}
    const JsonValue* raw_content = Get(msg, "content");
    const JsonValue empty = JsonValue::String("");
    const JsonValue* content_v = (raw_content != nullptr && raw_content->Truthy()) ? raw_content : &empty;
    std::string content_storage;
    bool content_is_string = content_v->is_string();
    if (content_is_string) content_storage = content_v->text();

    if (is_assistant) {
      const JsonValue* rc = Get(msg, "reasoning_content");
      if (rc != nullptr && rc->Truthy() && !opt.clear_thinking) {
        if (!content_is_string) Fail("content.lstrip on a non-string");
        if (!LstripStartsWith(content_storage, "<think>")) {
          std::string c = "<think>\n";
          c += MustBeString(rc, "reasoning_content");
          c += "\n</think>\n";
          c += content_storage;
          content_storage = std::move(c);
        }
      }
    }

    const JsonValue* tool_calls = Get(msg, "tool_calls");
    const bool has_tool_calls = tool_calls != nullptr && tool_calls->Truthy();

    if (is_user || (is_system && i != 0) || (is_assistant && !has_tool_calls)) {
      if (!content_is_string) Fail("TypeError: content is not a string");
      out += "<|im_start|>";
      out += msg.Find("role")->text();
      out += "\n";
      out += content_storage;
      out += "<|im_end|>\n";
    } else if (is_assistant) {
      out += "<|im_start|>assistant";
      if (!content_is_string) Fail("TypeError: content is not a string");
      if (!content_storage.empty()) {
        out += "\n";
        out += content_storage;
      }
      if (!tool_calls->is_array()) Fail("tool_calls is not an array");
      for (const JsonValue& call_outer : tool_calls->array()) {
        // {% if tool_call.function is defined %}{% set tool_call = tool_call.function %}
        const JsonValue* call = &call_outer;
        if (const JsonValue* fn = Get(call_outer, "function")) call = fn;
        const JsonValue* name = Get(*call, "name");
        const JsonValue* args = Get(*call, "arguments");
        if (name == nullptr) Fail("TypeError: tool_call.name is undefined (not JSON serializable)");
        out += "\n<tool_call>\n{\"name\": ";
        AppendPythonJson(out, *name);
        out += ", \"arguments\": ";
        if (args == nullptr) Fail("TypeError: tool_call.arguments is undefined (not JSON serializable)");
        AppendPythonJson(out, *args);
        out += "}\n</tool_call>";
      }
      out += "<|im_end|>\n";
    } else if (is_tool) {
      if (i == 0 || !RoleIs(msgs[i - 1], "tool")) out += "<|im_start|>user";
      out += "\n<tool_response>\n";
      out += MustBeString(raw_content, "tool message content");
      out += "\n</tool_response>";
      if (i + 1 == msgs.size() || !RoleIs(msgs[i + 1], "tool")) out += "<|im_end|>\n";
    }
  }

  if (opt.add_generation_prompt) out += "<|im_start|>assistant\n";
  return out;
}

}  // namespace

std::optional<std::string> RenderChat(const JsonValue& messages, const JsonValue* tools,
                                      const ChatRenderOptions& options, std::string* error) {
  try {
    return RenderImpl(messages, tools, options);
  } catch (const RenderError& e) {
    if (error != nullptr) *error = e.message;
    return std::nullopt;
  }
}

}  // namespace ricursive::text

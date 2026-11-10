#include "chat_request.hpp"

#include <stdexcept>

namespace gufo::models::ricursive {
namespace {

namespace rt = ::ricursive::text;
using rt::JsonValue;

const char* RoleName(ext::ChatRole role) {
  switch (role) {
    case ext::ChatRole::kSystem:
    case ext::ChatRole::kDeveloper: return "system";
    case ext::ChatRole::kUser: return "user";
    case ext::ChatRole::kAssistant: return "assistant";
    case ext::ChatRole::kTool: return "tool";
  }
  return "user";
}

JsonValue Argument(const ext::ChatMessage::ToolArgument& argument) {
  if (argument.is_string) return JsonValue::String(argument.value);
  JsonValue value;
  std::string error;
  if (!rt::ParseJson(argument.value, &value, &error)) {
    throw std::invalid_argument("tool call argument '" + argument.name + "' is not JSON: " + error);
  }
  return value;
}

struct Inputs {
  JsonValue messages;
  JsonValue tools;
  bool with_tools{false};
  rt::ChatRenderOptions options;
};

Inputs MakeInputs(const ext::ChatRequest& request) {
  Inputs in;
  in.options.add_generation_prompt = true;
  in.options.clear_thinking = request.reasoning.preserve_thinking.has_value() &&
                              !*request.reasoning.preserve_thinking;
  in.messages = MessagesToJson(request.messages);
  in.with_tools = request.tool_choice != ext::ChatRequest::ToolChoice::kNone && !request.tools.empty();
  if (in.with_tools) in.tools = ToolsToJson(request.tools);
  return in;
}

}  // namespace

JsonValue MessagesToJson(std::span<const ext::ChatMessage> messages) {
  JsonValue::Array out;
  for (const auto& message : messages) {
    if (!message.images.empty()) throw std::invalid_argument("model does not support image input");
    JsonValue::Object object;
    object.emplace_back("role", JsonValue::String(RoleName(message.role)));
    object.emplace_back("content", JsonValue::String(message.content + message.framing_suffix));
    if (message.role == ext::ChatRole::kAssistant) {
      if (!message.thought.empty()) {
        object.emplace_back("reasoning_content", JsonValue::String(message.thought));
      }
      if (!message.tool_calls.empty()) {
        JsonValue::Array calls;
        for (const auto& call : message.tool_calls) {
          JsonValue::Object arguments;
          for (const auto& argument : call.arguments) {
            arguments.emplace_back(argument.name, Argument(argument));
          }
          JsonValue::Object function;
          function.emplace_back("name", JsonValue::String(call.name));
          function.emplace_back("arguments", JsonValue::MakeObject(std::move(arguments)));
          calls.push_back(JsonValue::MakeObject(std::move(function)));
        }
        object.emplace_back("tool_calls", JsonValue::MakeArray(std::move(calls)));
      }
    }
    out.push_back(JsonValue::MakeObject(std::move(object)));
  }
  return JsonValue::MakeArray(std::move(out));
}

JsonValue ToolsToJson(std::span<const ext::ChatTool> tools) {
  JsonValue::Array out;
  for (const auto& tool : tools) {
    JsonValue definition;
    std::string error;
    if (!tool.definition_json.empty()) {
      if (!rt::ParseJson(tool.definition_json, &definition, &error)) {
        throw std::invalid_argument("tool '" + tool.name + "' definition is not JSON: " + error);
      }
    } else {
      JsonValue parameters;
      const std::string& text = tool.parameters_json.empty() ? std::string("{}") : tool.parameters_json;
      if (!rt::ParseJson(text, &parameters, &error)) {
        throw std::invalid_argument("tool '" + tool.name + "' parameters are not JSON: " + error);
      }
      JsonValue::Object function;
      function.emplace_back("name", JsonValue::String(tool.name));
      function.emplace_back("description", JsonValue::String(tool.description));
      function.emplace_back("parameters", std::move(parameters));
      JsonValue::Object wrapper;
      wrapper.emplace_back("type", JsonValue::String("function"));
      wrapper.emplace_back("function", JsonValue::MakeObject(std::move(function)));
      definition = JsonValue::MakeObject(std::move(wrapper));
    }
    out.push_back(std::move(definition));
  }
  return JsonValue::MakeArray(std::move(out));
}

ChatPrompt PrepareRequest(const Model& model, const ext::ChatRequest& request) {
  const Inputs in = MakeInputs(request);
  std::string error;
  auto prompt = model.PrepareChat(in.messages, in.with_tools ? &in.tools : nullptr, in.options, &error);
  if (!prompt) throw std::invalid_argument("ricursive chat template: " + error);
  return std::move(*prompt);
}

std::string RenderRequest(const ext::ChatRequest& request) {
  const Inputs in = MakeInputs(request);
  std::string error;
  const auto rendered = RenderChatProtected(in.messages, in.with_tools ? &in.tools : nullptr, in.options, &error);
  if (!rendered) throw std::invalid_argument("ricursive chat template: " + error);
  return rendered->text;
}

}  // namespace gufo::models::ricursive

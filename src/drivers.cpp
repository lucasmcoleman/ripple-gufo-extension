// `gufo prompt`, `gufo chat` and `gufo bench` for ricursive on one session,
// outside the serving pool.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "chat_request.hpp"
#include "output_parser.hpp"
#include "runner.hpp"

namespace gufo::models::ricursive {
namespace {

using Clock = std::chrono::steady_clock;
namespace rt = ::ricursive::text;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void PrintModelLoadTime(Clock::time_point start, bool success = true) {
  auto& out = success ? std::cout : std::cerr;
  out << "[Model Load]: " << SecondsSince(start) << " s" << (success ? "" : " (failed)") << '\n';
}

std::shared_ptr<Model> LoadModel(const std::string& path, const ext::ParsedOptions& provider_options,
                                 bool force_cpu, std::uint32_t at_least_context,
                                 Clock::time_point start) {
  ModelOptions options;
  try {
    ApplyProviderOptions(provider_options, options);
  } catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return nullptr;
  }
  if (force_cpu) options.backend = CpuBackend();
  // The default cap suffices unless the benchmark needs more positions.
  if (options.max_context == 0 && at_least_context > kDefaultContextCap) {
    options.max_context = at_least_context;
  }
  std::string error;
  auto model = Model::Load(path, options, &error);
  if (!model) {
    std::cerr << "Error loading ricursive model: " << error << '\n';
    PrintModelLoadTime(start, false);
    return nullptr;
  }
  PrintModelLoadTime(start);
  return model;
}

/// The reply as the user sees it: reasoning, content and tool calls apart.
class Display {
 public:
  explicit Display(bool parse) : parse_(parse) {}
  void Push(const std::string& text) {
    if (!parse_) {
      std::cout << text << std::flush;
      return;
    }
    Show(parser_.Push(text));
  }
  void Finish() {
    if (parse_) Show(parser_.Finish());
    if (in_reasoning_) std::cout << "\n[/reasoning]";
    std::cout << '\n';
  }

 private:
  void Show(const ParserEvents& ev) {
    if (!ev.reasoning.empty()) {
      if (!in_reasoning_) std::cout << "[reasoning]\n";
      in_reasoning_ = true;
      std::cout << ev.reasoning;
    }
    if (!ev.content.empty()) {
      if (in_reasoning_) std::cout << "\n[/reasoning]\n";
      in_reasoning_ = false;
      std::cout << ev.content;
    }
    for (const auto& call : ev.calls) {
      if (in_reasoning_) std::cout << "\n[/reasoning]";
      in_reasoning_ = false;
      std::cout << "\n[tool call] " << call.name << ' ' << call.arguments_json << '\n';
    }
    if (!ev.unterminated.empty()) std::cout << "\n[unfinished tool call] " << ev.unterminated;
    std::cout << std::flush;
  }
  bool parse_;
  bool in_reasoning_{false};
  OutputParser parser_;
};

void WarnThinking(const ext::TextPromptOptions& opt) {
  if (opt.reasoning_mode != "auto" || opt.reasoning_effort != "auto") {
    std::cerr << "note: Ripple has no thinking switch (the generation prompt never opens "
                 "<think>); --think and --reasoning-effort are ignored\n";
  }
}

/// Generates one reply after `prompt`; `reply` receives the raw model text.
int Generate(const ext::TextPromptOptions& opt, Model& model, Session& session,
             std::span<const std::int32_t> prompt, bool parse, std::string* reply) {
  if (prompt.empty()) {
    std::cerr << "ricursive prompt produced no tokens\n";
    return 1;
  }
  if (prompt.size() + opt.max_tokens > model.MaxContext()) {
    std::cerr << "ricursive prompt (" << prompt.size() << " tokens) and output (" << opt.max_tokens
              << " tokens) exceed the " << model.MaxContext()
              << "-token context cap (--context-cap raises it)\n";
    return 1;
  }
  std::string error;
  const auto prefill_start = Clock::now();
  if (!session.Sync(prompt, &error)) {
    std::cerr << "ricursive prefill failed: " << error << '\n';
    return 1;
  }
  const double prefill_seconds = SecondsSince(prefill_start);
  if (opt.verbose) {
    std::cout << "[Engine]: ricursive (" << (model.backend().uses_device ? "GPU" : "CPU")
              << " backend)\nModel: " << model.ModelName() << "\nPrompt tokens: " << prompt.size()
              << "\n--- Generation Output ---\n";
  }
  const std::vector<sampling::TokenId> history(prompt.begin(), prompt.end());
  sampling::SamplerState sampler(opt.sampling, history);
  rt::StreamDecoder decoder(model.tokenizer());
  Display display(parse);
  const auto decode_start = Clock::now();
  std::size_t generated = 0;
  while (generated < opt.max_tokens) {
    const auto token = static_cast<std::int32_t>(sampler.Sample(session.Logits()));
    if (model.IsStopToken(token)) break;
    sampler.Accept(static_cast<sampling::TokenId>(token));
    ++generated;
    const std::string piece = decoder.Push(static_cast<rt::TokenId>(token));
    if (reply != nullptr) reply->append(model.TokenText(token));
    display.Push(piece);
    if (generated < opt.max_tokens && !session.Evaluate(token, &error)) {
      std::cerr << "\nricursive decode failed: " << error << '\n';
      return 1;
    }
  }
  display.Push(decoder.Flush());
  display.Finish();
  if (opt.verbose && generated > 0) {
    const double seconds = SecondsSince(decode_start);
    std::cout << "Prefill " << prompt.size() << " tokens in " << prefill_seconds << " s ("
              << static_cast<double>(prompt.size()) / prefill_seconds << " tok/s)\nGenerated "
              << generated << " tokens in " << seconds << " s ("
              << static_cast<double>(generated) / seconds << " tok/s)\n";
  }
  return 0;
}

std::vector<std::int32_t> ChatTokens(const Model& model, std::span<const ext::ChatMessage> history) {
  rt::ChatRenderOptions options;
  options.add_generation_prompt = true;
  const rt::JsonValue messages = MessagesToJson(history);
  std::string error;
  auto prompt = model.PrepareChat(messages, nullptr, options, &error);
  if (!prompt) throw std::invalid_argument("chat template: " + error);
  return std::move(prompt->tokens);
}

struct BenchStats {
  double mean{0};
  double stddev{0};
};

BenchStats ComputeStats(const std::vector<double>& values) {
  if (values.empty()) return {};
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
  if (values.size() <= 1) return {mean, 0.0};
  double squares = 0.0;
  for (const double v : values) squares += (v - mean) * (v - mean);
  return {mean, std::sqrt(squares / static_cast<double>(values.size() - 1))};
}

}  // namespace

int RunRicursivePrompt(const ext::TextPromptOptions& opt, const core::GgufReader&,
                       Clock::time_point load_start) {
  auto model = LoadModel(opt.model_path, opt.provider_options, opt.force_cpu, 0, load_start);
  if (!model) return 1;
  std::string error;
  auto session = model->CreateSession(&error);
  if (!session) {
    std::cerr << "ricursive session creation failed: " << error << '\n';
    return 1;
  }
  try {
    WarnThinking(opt);
    std::vector<std::int32_t> prompt;
    if (opt.use_chat_template) {
      std::vector<ext::ChatMessage> messages;
      if (!opt.system_prompt.empty()) messages.emplace_back(ext::ChatRole::kSystem, opt.system_prompt);
      messages.emplace_back(ext::ChatRole::kUser, opt.prompt_text);
      prompt = ChatTokens(*model, messages);
    } else {
      prompt = model->Tokenize(opt.prompt_text);
    }
    return Generate(opt, *model, *session, prompt, opt.use_chat_template, nullptr);
  } catch (const std::exception& e) {
    std::cerr << "ricursive prompt failed: " << e.what() << '\n';
    return 1;
  }
}

int RunRicursiveChat(const ext::TextPromptOptions& opt, const core::GgufReader&,
                     Clock::time_point load_start) {
  if (!opt.use_chat_template) {
    std::cerr << "ricursive interactive chat requires chat framing; use prompt --raw for raw text\n";
    return 1;
  }
  auto model = LoadModel(opt.model_path, opt.provider_options, opt.force_cpu, 0, load_start);
  if (!model) return 1;
  std::string error;
  auto session = model->CreateSession(&error);
  if (!session) {
    std::cerr << "ricursive session creation failed: " << error << '\n';
    return 1;
  }
  WarnThinking(opt);
  std::vector<ext::ChatMessage> history;
  if (!opt.system_prompt.empty()) history.emplace_back(ext::ChatRole::kSystem, opt.system_prompt);
  std::cout << "=== Gufo Interactive Chat (ricursive) ===\nType 'exit' or Ctrl+D to quit.\n\n";
  for (std::string input;;) {
    std::cout << ">>> User: " << std::flush;
    if (!std::getline(std::cin, input) || input == "exit" || input == "quit") break;
    if (input.empty()) continue;
    history.emplace_back(ext::ChatRole::kUser, input);
    std::vector<std::int32_t> prompt;
    try {
      prompt = ChatTokens(*model, history);
    } catch (const std::exception& e) {
      std::cerr << "ricursive chat template failed: " << e.what() << '\n';
      return 1;
    }
    std::cout << "<<< Assistant: ";
    std::string reply;
    if (Generate(opt, *model, *session, prompt, true, &reply) != 0) return 1;
    // The raw text, so the next prompt re-renders to the tokens the model
    // produced (src/text/OUTPUT-FORMAT.md).
    history.emplace_back(ext::ChatRole::kAssistant, std::move(reply));
    std::cout << '\n';
  }
  return 0;
}

int RunRicursiveBench(const ext::TextBenchOptions& options,
                      const std::shared_ptr<const core::GgufReader>& reader,
                      Clock::time_point load_start) {
  if (options.concurrency != std::vector<std::size_t>{1}) {
    std::cerr << "Error: ricursive bench supports C1; use the serving benchmark for concurrent requests\n";
    return 1;
  }
  const auto max_or_zero = [](const std::vector<std::size_t>& v) {
    return v.empty() ? std::size_t{0} : *std::max_element(v.begin(), v.end());
  };
  const std::size_t depth_max = max_or_zero(options.n_depths);
  const std::size_t required = std::max({std::size_t{4096}, depth_max + max_or_zero(options.n_prompts) + 1,
                                         std::max<std::size_t>(depth_max, 16) + max_or_zero(options.n_gens) + 1});
  if (required > UINT32_MAX) {
    std::cerr << "Error: ricursive benchmark context is out of range\n";
    return 1;
  }
  auto model = LoadModel(options.model_path, options.provider_options, false,
                         static_cast<std::uint32_t>(required), load_start);
  if (!model) return 1;
  std::vector<std::int32_t> tokens(required);
  {
    const auto pattern = model->Tokenize(
        "The quick brown fox jumps over the lazy dog. "
        "Strix Halo executes this deterministic benchmark sequence. ");
    if (pattern.empty()) {
      std::cerr << "Error: benchmark pattern produced no tokens\n";
      return 1;
    }
    for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = pattern[i % pattern.size()];
  }
  const double size_gib = static_cast<double>(reader->GetSize()) / (1024.0 * 1024.0 * 1024.0);
  std::ostringstream size_text;
  size_text << std::fixed << std::setprecision(2) << size_gib << " GiB";
  const std::string backend = model->backend().uses_device ? "ROCm (HIP)" : "CPU";
  std::cout << "| " << std::left << std::setw(32) << "model" << " | " << std::right << std::setw(10)
            << "size" << " | " << std::left << std::setw(10) << "backend" << " | " << std::right
            << std::setw(18) << "test" << " | " << std::setw(21) << "t/s" << " |\n"
            << "| " << std::string(32, '-') << " | " << std::string(10, '-') << " | "
            << std::string(10, '-') << " | " << std::string(18, '-') << " | " << std::string(21, '-')
            << " |\n";
  const auto print = [&](const std::string& test, const BenchStats& stats) {
    std::ostringstream throughput;
    throughput << std::fixed << std::setprecision(2) << stats.mean << " +- " << stats.stddev;
    std::cout << "| " << std::left << std::setw(32) << model->ModelName() << " | " << std::right
              << std::setw(10) << size_text.str() << " | " << std::left << std::setw(10) << backend
              << " | " << std::right << std::setw(18) << test << " | " << std::setw(21)
              << throughput.str() << " |\n" << std::flush;
  };
  const auto name = [](std::string_view prefix, std::size_t count, std::size_t depth) {
    return std::string(prefix) + std::to_string(count) + (depth > 0 ? " @ d" + std::to_string(depth) : "");
  };
  std::string error;
  for (const std::size_t depth : options.n_depths) {
    for (const std::size_t prompt_length : options.n_prompts) {
      std::vector<double> runs;
      // The first repetition is an untimed warm-up.
      for (std::size_t rep = 0; rep <= options.repetitions; ++rep) {
        auto session = model->CreateSession(&error);
        if (!session || (depth > 0 && !session->Sync(std::span(tokens).first(depth), &error))) {
          std::cerr << "Error preparing depth: " << error << '\n';
          return 1;
        }
        const auto start = Clock::now();
        if (!session->Sync(std::span(tokens).first(depth + prompt_length), &error)) {
          std::cerr << "Error running prefill: " << error << '\n';
          return 1;
        }
        if (rep > 0) runs.push_back(static_cast<double>(prompt_length) / SecondsSince(start));
      }
      print(name("pp", prompt_length, depth), ComputeStats(runs));
    }
    for (const std::size_t gen : options.n_gens) {
      const std::size_t prefix = depth > 0 ? depth : 16;
      std::vector<double> runs;
      for (std::size_t rep = 0; rep < options.repetitions; ++rep) {
        auto session = model->CreateSession(&error);
        if (!session || !session->Sync(std::span(tokens).first(prefix), &error)) {
          std::cerr << "Error preparing generation: " << error << '\n';
          return 1;
        }
        const std::vector<sampling::TokenId> history(tokens.begin(),
                                                     tokens.begin() + static_cast<std::ptrdiff_t>(prefix));
        sampling::SamplerState sampler(options.sampling, history);
        const auto start = Clock::now();
        for (std::size_t i = 0; i < gen; ++i) {
          const auto token = static_cast<std::int32_t>(sampler.Sample(session->Logits()));
          sampler.Accept(static_cast<sampling::TokenId>(token));
          if (!session->Evaluate(token, &error)) {
            std::cerr << "Error running ricursive decode: " << error << '\n';
            return 1;
          }
        }
        runs.push_back(static_cast<double>(gen) / SecondsSince(start));
      }
      print(name("tg", gen, depth), ComputeStats(runs));
    }
  }
  std::cout << '\n';
  return 0;
}

}  // namespace gufo::models::ricursive

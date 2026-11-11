// ricursive_decode_bench MODEL.gguf --tokens FILE.i32 --prompt-tokens N --generate M
//     [--backend cpu|host|gpu] [--rows R] [--context-cap N] [--logits-out FILE]
//     [--allow-template-mismatch]
//
// No HTTP. Loads the GGUF through the engine (Model::Load), runs the first N
// tokens of the token file as the prompt (Session::Sync, in passes of at most R
// rows: the path a `gufo serve llm` prefill takes) and then generates M tokens
// greedily (arg-max of Session::Logits(), fed back with Session::Evaluate: one
// 1-row pass per token, KV cache and sliding ring as in serving). Stop tokens are
// ignored so exactly M tokens are produced. Prints prompt and generation speed and
// the generated token ids (so two runs can be compared).
//
// Backends: gpu = the production executor; cpu = the scalar reference executor
// (tiny models only); host = the executor's own orchestration on the host ops
// (hip/ops_host.cpp, float32 caches), the CPU stand-in for the GPU path.
// RICURSIVE_PROFILE=1 adds the per-stage report (docs/PERF-PROFILE.md) on stdout.
// --logits-out writes float32 [2][vocab]: the logits after the prompt, then after
// the last generated token was fed back. Exit 0 ok, 1 failure, 2 usage.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "src/engine.hpp"
#include "src/executor_factory.hpp"

namespace gufo::models::ricursive {
// tools/decode_bench_host.cpp
std::unique_ptr<Executor> MakeHostOpsExecutor(const ModelWeights&, const ExecutorConfig&,
                                              std::string* error);
std::string StageProfileReport(Executor& executor);
}  // namespace gufo::models::ricursive

using namespace gufo::models::ricursive;
using Clock = std::chrono::steady_clock;

namespace {

double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

void Usage() {
  std::cerr << "usage: ricursive_decode_bench MODEL.gguf --tokens FILE.i32 --prompt-tokens N "
               "--generate M [--backend cpu|host|gpu] [--rows R] [--context-cap N] "
               "[--logits-out FILE] [--allow-template-mismatch]\n"
               "  --backend defaults to gpu; --rows defaults to "
            << kDefaultPrefillRows << " (prompt rows per pass, as in serving).\n"
               "  RICURSIVE_PROFILE=1 prints the per-stage timing report after the run.\n";
}

bool ParseUnsigned(const std::string& text, std::uint32_t* out, bool allow_zero = false) {
  char* end = nullptr;
  const unsigned long v = std::strtoul(text.c_str(), &end, 10);
  if (text.empty() || *end != '\0' || (v == 0 && !allow_zero) || v > 0xFFFFFFFFUL) return false;
  *out = static_cast<std::uint32_t>(v);
  return true;
}

std::int32_t ArgMax(std::span<const float> logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<std::int32_t>(best);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, tokens_path, backend_arg = "gpu", logits_out;
  std::uint32_t prompt_tokens = 0, generate = 0, rows = kDefaultPrefillRows, context_cap = 0;
  bool allow_mismatch = false;
  bool have_prompt = false, have_generate = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    if (a == "--tokens" && more) {
      tokens_path = argv[++i];
    } else if (a == "--prompt-tokens" && more) {
      have_prompt = ParseUnsigned(argv[++i], &prompt_tokens);
      if (!have_prompt) return Usage(), 2;
    } else if (a == "--generate" && more) {
      have_generate = ParseUnsigned(argv[++i], &generate, true);
      if (!have_generate) return Usage(), 2;
    } else if (a == "--backend" && more) {
      backend_arg = argv[++i];
    } else if (a == "--rows" && more) {
      if (!ParseUnsigned(argv[++i], &rows)) return Usage(), 2;
    } else if (a == "--context-cap" && more) {
      if (!ParseUnsigned(argv[++i], &context_cap)) return Usage(), 2;
    } else if (a == "--logits-out" && more) {
      logits_out = argv[++i];
    } else if (a == "--allow-template-mismatch") {
      allow_mismatch = true;
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "unknown option " << a << '\n';
      return Usage(), 2;
    } else if (model_path.empty()) {
      model_path = a;
    } else {
      return Usage(), 2;
    }
  }
  if (model_path.empty() || tokens_path.empty() || !have_prompt || !have_generate ||
      (backend_arg != "cpu" && backend_arg != "host" && backend_arg != "gpu")) {
    return Usage(), 2;
  }

  try {
    const auto t_start = Clock::now();
    std::ifstream in(tokens_path, std::ios::binary | std::ios::ate);
    if (!in) {
      std::cerr << "cannot open " << tokens_path << '\n';
      return 1;
    }
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (bytes == 0 || bytes % 4 != 0) {
      std::cerr << tokens_path << ": size " << bytes << " is not a positive multiple of 4\n";
      return 1;
    }
    if (bytes / 4 < prompt_tokens) {
      std::cerr << tokens_path << " holds " << bytes / 4 << " tokens, fewer than --prompt-tokens "
                << prompt_tokens << '\n';
      return 1;
    }
    std::vector<std::int32_t> prompt(prompt_tokens);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(prompt.data()),
            static_cast<std::streamsize>(prompt.size() * sizeof(std::int32_t)));
    if (!in) {
      std::cerr << "cannot read " << tokens_path << '\n';
      return 1;
    }

    ModelOptions options;
    options.prefill_rows = rows;
    options.allow_template_mismatch = allow_mismatch;
    options.max_context = context_cap != 0 ? context_cap : 0;
    if (context_cap == 0 && prompt_tokens + generate > kDefaultContextCap) {
      std::cerr << "prompt + generate exceeds the default context cap " << kDefaultContextCap
                << "; pass --context-cap\n";
      return 1;
    }
    if (backend_arg == "gpu") {
      options.backend = GpuBackend();
    } else {
      options.backend = CpuBackend();  // float32 caches, exact
      if (backend_arg == "host") {
        options.executor_override = MakeHostOpsExecutor;
      }
    }
    std::string error;
    auto model = Model::Load(model_path, options, &error);
    if (!model) {
      std::cerr << "load: " << error << '\n';
      return 1;
    }
    auto session = model->CreateSession(&error);
    if (!session) {
      std::cerr << "session: " << error << '\n';
      return 1;
    }
    const std::uint32_t vocab = model->VocabSize();
    if (std::any_of(prompt.begin(), prompt.end(), [&](std::int32_t t) { return t < 0 || static_cast<std::uint32_t>(t) >= vocab; })) {
      std::cerr << "a prompt token is outside the vocabulary (" << vocab << ")\n";
      return 1;
    }
    const auto t_loaded = Clock::now();
    std::printf("ricursive_decode_bench: backend=%s rows/pass=%u prompt=%u generate=%u context-cap=%u\n",
                backend_arg.c_str(), model->PrefillRows(), prompt_tokens, generate, model->MaxContext());
    std::printf("load: %.2f s (open, resolve, session and executor%s)\n", Seconds(t_start, t_loaded),
                backend_arg == "gpu" ? ", weight upload" : "");
    std::fflush(stdout);

    const auto t0 = Clock::now();
    if (!session->Sync(prompt, &error)) {
      std::cerr << "prompt failed: " << error << '\n';
      return 1;
    }
    const auto t1 = Clock::now();
    std::vector<float> first_logits(session->Logits().begin(), session->Logits().end());
    const double prompt_s = Seconds(t0, t1);
    std::printf("prompt: %u tokens in %.3f s = %.2f tokens/s\n",
                prompt_tokens, prompt_s, prompt_s > 0 ? prompt_tokens / prompt_s : 0.0);
    std::fflush(stdout);

    std::vector<std::int32_t> generated;
    generated.reserve(generate);
    const auto g0 = Clock::now();
    for (std::uint32_t i = 0; i < generate; ++i) {
      const auto logits = session->Logits();
      if (logits.empty()) {
        std::cerr << "no logits before generated token " << i << '\n';
        return 1;
      }
      const std::int32_t token = ArgMax(logits);
      generated.push_back(token);
      if (!session->Evaluate(token, &error)) {
        std::cerr << "decode failed at token " << i << ": " << error << '\n';
        return 1;
      }
    }
    const auto g1 = Clock::now();
    const double gen_s = Seconds(g0, g1);
    std::printf("generate: %u tokens in %.3f s = %.3f tokens/s (%.1f ms/token; argmax, Evaluate and "
                "readback included)\n",
                generate, gen_s, gen_s > 0 ? generate / gen_s : 0.0,
                generate > 0 ? 1000.0 * gen_s / generate : 0.0);
    std::printf("generated:");
    for (const std::int32_t t : generated) std::printf(" %d", t);
    std::printf("\n");

    if (!logits_out.empty()) {
      std::ofstream out(logits_out, std::ios::binary);
      out.write(reinterpret_cast<const char*>(first_logits.data()),
                static_cast<std::streamsize>(first_logits.size() * sizeof(float)));
      const auto last = session->Logits();
      out.write(reinterpret_cast<const char*>(last.data()),
                static_cast<std::streamsize>(last.size() * sizeof(float)));
      out.close();
      if (!out) {
        std::cerr << "cannot write " << logits_out << '\n';
        return 1;
      }
    }
    const std::string report = StageProfileReport(session->executor());
    if (!report.empty()) {
      std::printf("\n%s", report.c_str());
    } else if (const char* env = std::getenv("RICURSIVE_PROFILE"); env != nullptr && *env != '\0') {
      std::printf("RICURSIVE_PROFILE is set but the %s executor produced no report\n", backend_arg.c_str());
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

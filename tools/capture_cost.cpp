// ricursive_capture_cost MODEL.gguf --rows N --reps R [--layers a,b,c,d,e] [--backend cpu|host|gpu]
//     [--base P] [--context-cap C] [--allow-template-mismatch]
//
// The cost test of hidden-state capture (docs/DRAFTER-S3.md): the wall time of
// ONE Forward of N rows at position P, with capture off and with capture on
// (the five tap layers copied to the host), alternating off/on R times each, on
// the same executor, rewinding to P before every pass. N of 1 or a prompt-sized
// pass returns the last row's logits (as serving does); 2 to 16 rows return every
// row's logits (a verification pass). Prints the median and the spread of each
// and the ratio on/off; exit 0 when the ratio is within 1% (design section 6,
// stage 3), 1 when it is not or on failure, 2 usage.
//
// The model is positioned at P (default 2048) with one prompt pass of P tokens
// (synthetic ids), untimed. A pass of N rows needs P + N <= the context cap.
// Backend gpu only through the GPU lock wrapper (tests/run-gpu-gates.sh style,
// perf/gpu-run.sh); cpu and host are for tiny fixtures.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
}  // namespace gufo::models::ricursive

using namespace gufo::models::ricursive;
using Clock = std::chrono::steady_clock;

namespace {

bool ParseUnsigned(const std::string& text, std::uint32_t* out) {
  char* end = nullptr;
  const unsigned long v = std::strtoul(text.c_str(), &end, 10);
  if (text.empty() || *end != '\0' || v == 0 || v > 0xFFFFFFFFUL) return false;
  *out = static_cast<std::uint32_t>(v);
  return true;
}

void Usage() {
  std::cerr << "usage: ricursive_capture_cost MODEL.gguf --rows N --reps R [--layers a,b,c,d,e] "
               "[--backend cpu|host|gpu] [--base P] [--context-cap C] [--allow-template-mismatch]\n"
               "  --layers defaults to 1,20,39,58,77; --backend to gpu; --base to 2048.\n";
}

struct Stat {
  double median{0}, min{0}, max{0};
};

Stat Summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return {v[v.size() / 2], v.front(), v.back()};
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, backend_arg = "gpu", layers_arg = "1,20,39,58,77";
  std::uint32_t rows = 0, reps = 0, base = 2048, context_cap = 0;
  bool allow_mismatch = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    if (a == "--rows" && more) {
      if (!ParseUnsigned(argv[++i], &rows)) return Usage(), 2;
    } else if (a == "--reps" && more) {
      if (!ParseUnsigned(argv[++i], &reps)) return Usage(), 2;
    } else if (a == "--base" && more) {
      if (!ParseUnsigned(argv[++i], &base)) return Usage(), 2;
    } else if (a == "--context-cap" && more) {
      if (!ParseUnsigned(argv[++i], &context_cap)) return Usage(), 2;
    } else if (a == "--layers" && more) {
      layers_arg = argv[++i];
    } else if (a == "--backend" && more) {
      backend_arg = argv[++i];
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
  if (model_path.empty() || rows == 0 || reps == 0 ||
      (backend_arg != "cpu" && backend_arg != "host" && backend_arg != "gpu")) {
    return Usage(), 2;
  }
  std::vector<std::uint32_t> layers;
  for (std::size_t at = 0; at < layers_arg.size();) {
    const std::size_t comma = layers_arg.find(',', at);
    std::uint32_t v = 0;
    const std::string item = layers_arg.substr(at, comma == std::string::npos ? comma : comma - at);
    char* end = nullptr;
    const unsigned long n = std::strtoul(item.c_str(), &end, 10);
    if (item.empty() || *end != '\0') return Usage(), 2;
    v = static_cast<std::uint32_t>(n);
    layers.push_back(v);
    if (comma == std::string::npos) break;
    at = comma + 1;
  }

  try {
    ModelOptions options;
    options.prefill_rows = std::max<std::uint32_t>(rows, 2048);
    options.allow_template_mismatch = allow_mismatch;
    options.max_context = context_cap != 0 ? context_cap : base + rows;
    options.verify_rows = rows > 1 && rows <= kMaxVerifyRows ? rows : 0;
    options.capture_layers = layers;
    if (backend_arg == "gpu") {
      options.backend = GpuBackend();
    } else {
      options.backend = CpuBackend();
      if (backend_arg == "host") options.executor_override = MakeHostOpsExecutor;
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
    const auto token = [&](std::uint32_t i) { return static_cast<std::int32_t>((i * 7919U + 13U) % vocab); };
    if (base + rows > model->MaxContext()) {
      std::cerr << "base + rows exceeds the context cap " << model->MaxContext() << '\n';
      return 1;
    }
    std::vector<std::int32_t> prompt(base);
    for (std::uint32_t i = 0; i < base; ++i) prompt[i] = token(i);
    if (!session->Sync(prompt, &error)) {
      std::cerr << "prompt failed: " << error << '\n';
      return 1;
    }
    Executor& exec = session->executor();
    std::vector<std::int32_t> pass(rows);
    for (std::uint32_t i = 0; i < rows; ++i) pass[i] = token(base + i);
    const LogitRows which = rows == 1 || rows > kMaxVerifyRows ? LogitRows::kLast : LogitRows::kAll;
    std::vector<float> logits((which == LogitRows::kAll ? rows : 1U) * std::size_t{vocab});
    std::vector<float> capture(layers.size() * std::size_t{rows} * exec.Info().hidden);
    const auto run = [&](bool capturing) {
      if (!exec.Rewind(base)) throw std::runtime_error("cannot rewind to the base position");
      const auto t0 = Clock::now();
      exec.Forward(pass, base, which, logits, {}, capturing ? std::span<float>(capture) : std::span<float>());
      return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    };
    (void)run(false);  // warm-up (kernels, allocator), untimed
    (void)run(true);
    std::vector<double> off, on;
    for (std::uint32_t r = 0; r < reps; ++r) {  // off, on, on, off, ...: no drift bias
      if (r % 2 == 0) {
        off.push_back(run(false));
        on.push_back(run(true));
      } else {
        on.push_back(run(true));
        off.push_back(run(false));
      }
    }
    const Stat a = Summarize(off), b = Summarize(on);
    const double ratio = b.median / a.median;
    std::printf("capture cost: backend=%s rows=%u base=%u reps=%u layers=%s logits=%s\n", backend_arg.c_str(), rows,
                base, reps, layers_arg.c_str(), which == LogitRows::kAll ? "all rows" : "last row");
    std::printf("  capture off: median %.3f ms (min %.3f, max %.3f)\n", a.median, a.min, a.max);
    std::printf("  capture on : median %.3f ms (min %.3f, max %.3f)\n", b.median, b.min, b.max);
    std::printf("  on/off = %.4f  (%+.2f%%)  %s\n", ratio, (ratio - 1.0) * 100.0,
                ratio <= 1.01 ? "WITHIN 1%" : "OVER 1%");
    return ratio <= 1.01 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "capture cost failed: " << e.what() << '\n';
    return 1;
  }
}

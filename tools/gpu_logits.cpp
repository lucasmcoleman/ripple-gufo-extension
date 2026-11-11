// ricursive_gpu_logits MODEL.gguf TOKENS.i32 OUT.f32
//     [--rows N per pass] [--kv f16|f32] [--exact] [--context-cap N] [--last-only]
// ricursive_gpu_logits MODEL.gguf --list PAIRS.txt [same options]
//
// List mode: one model load and one executor for many texts. PAIRS.txt has one "TOKENS.i32 OUT.f32"
// pair per line (whitespace separated; blank lines and lines starting with # ignored). The executor
// is sized for the longest text (--context-cap overrides, and must cover it), Reset() before each
// text, and each text is run exactly as the single-file mode runs it (chunks of min(rows, T)).
// Per text it prints the same summary lines as the single mode (so a text's output equals what a
// separate single-file run prints), the load line once. The first failing text stops the run with
// exit 1; outputs written so far stay.
//
// Teacher-forced passes through the production executor (src/executor_factory):
// the GPU executor, or the scalar CPU backend with RICURSIVE_BACKEND=cpu (tests
// and tiny models only). Writes float32 logits [T][vocab] in the file format of
// oracle/bin/ripple_logits and tools/ref_logits; with --last-only the file holds
// only the logits after the final token ([1][vocab]) and the other passes skip
// the head. Prints load time, tokens per second for the passes, and device
// memory in use (GPU backend). Exit 0 = written and finite, 1 = failure or any
// NaN/Inf logit (the file is still written for inspection), 2 = usage.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "hip/gpu_executor_factory.hpp"
#include "src/executor_factory.hpp"

using namespace gufo::models::ricursive;
using Clock = std::chrono::steady_clock;

namespace {

double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

void Usage() {
  std::cerr << "usage: ricursive_gpu_logits MODEL.gguf TOKENS.i32 OUT.f32 [--rows N per pass] "
               "[--kv f16|f32] [--exact] [--context-cap N] [--last-only]\n"
               "       ricursive_gpu_logits MODEL.gguf --list PAIRS.txt [options as above]\n"
               "  RICURSIVE_BACKEND=cpu runs the scalar CPU backend (tiny models only).\n"
               "  --kv defaults to f16 on the GPU (f32 only with --exact is a meaningful "
               "comparison against an F32-KV reference); the CPU backend stores f32.\n"
               "  --rows defaults to 256; --context-cap defaults to the token count.\n";
}

bool ParseUnsigned(const std::string& text, std::uint32_t* out) {
  char* end = nullptr;
  const unsigned long v = std::strtoul(text.c_str(), &end, 10);
  if (text.empty() || *end != '\0' || v == 0 || v > 0xFFFFFFFFUL) return false;
  *out = static_cast<std::uint32_t>(v);
  return true;
}

double Mib(std::size_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  std::uint32_t rows = 256, context_cap = 0;
  std::string kv_arg;
  bool exact = false, last_only = false;
  std::string list_path;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--list" && i + 1 < argc) {
      list_path = argv[++i];
    } else if (a == "--rows" && i + 1 < argc) {
      if (!ParseUnsigned(argv[++i], &rows)) return Usage(), 2;
    } else if (a == "--context-cap" && i + 1 < argc) {
      if (!ParseUnsigned(argv[++i], &context_cap)) return Usage(), 2;
    } else if (a == "--kv" && i + 1 < argc) {
      kv_arg = argv[++i];
      if (kv_arg != "f16" && kv_arg != "f32") return Usage(), 2;
    } else if (a == "--exact") {
      exact = true;
    } else if (a == "--last-only") {
      last_only = true;
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "unknown option " << a << '\n';
      return Usage(), 2;
    } else {
      pos.push_back(a);
    }
  }
  if (pos.size() != (list_path.empty() ? 3u : 1u)) return Usage(), 2;

  try {
    const auto t_start = Clock::now();
    std::string error;
    const auto backend = BackendFromEnvironment(&error);
    if (!backend) {
      std::cerr << error << '\n';
      return 1;
    }
    const bool gpu = backend->backend == Backend::kGpu;

    // Work list: one (tokens, output) pair, or the pairs of the list file.
    std::vector<std::pair<std::string, std::string>> jobs;
    if (list_path.empty()) {
      jobs.emplace_back(pos[1], pos[2]);
    } else {
      std::ifstream lf(list_path);
      if (!lf) {
        std::cerr << "cannot open list " << list_path << '\n';
        return 1;
      }
      std::string line;
      while (std::getline(lf, line)) {
        std::istringstream ls(line);
        std::string a, b, extra;
        if (!(ls >> a) || a[0] == '#') continue;
        if (!(ls >> b) || (ls >> extra)) {
          std::cerr << "list line needs exactly TOKENS.i32 OUT.f32: " << line << '\n';
          return 1;
        }
        jobs.emplace_back(a, b);
      }
      if (jobs.empty()) {
        std::cerr << "list has no pairs: " << list_path << '\n';
        return 1;
      }
    }
    const bool list_mode = !list_path.empty();

    // Read every token file first: a bad file should fail before the model load.
    std::vector<std::vector<std::int32_t>> all_tokens;
    std::size_t max_tokens = 0;
    for (const auto& job : jobs) {
      std::ifstream in(job.first, std::ios::binary | std::ios::ate);
      if (!in) {
        std::cerr << "cannot open " << job.first << '\n';
        return 1;
      }
      const auto bytes = static_cast<std::size_t>(in.tellg());
      if (bytes == 0 || bytes % 4 != 0) {
        std::cerr << job.first << ": size " << bytes << " is not a positive multiple of 4\n";
        return 1;
      }
      std::vector<std::int32_t> tokens(bytes / 4);
      in.seekg(0);
      in.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(bytes));
      if (!in) {
        std::cerr << "cannot read " << job.first << '\n';
        return 1;
      }
      max_tokens = std::max(max_tokens, tokens.size());
      all_tokens.push_back(std::move(tokens));
    }

    const std::shared_ptr<const gufo::core::GgufReader> reader =
        gufo::core::GgufReader::OpenFile(pos[0], &error);
    if (!reader) {
      std::cerr << "open " << pos[0] << ": " << error << '\n';
      return 1;
    }
    const auto config = ParseConfig(*reader, &error);
    if (!config) {
      std::cerr << error << '\n';
      return 1;
    }
    const auto weights = ResolveWeights(*reader, *config, &error);
    if (!weights) {
      std::cerr << error << '\n';
      return 1;
    }
    const std::uint32_t T_max = static_cast<std::uint32_t>(max_tokens);
    const std::uint32_t vocab = weights->config.vocab_size;
    if (context_cap == 0) context_cap = T_max;
    if (T_max > context_cap) {
      std::cerr << T_max << " tokens exceed --context-cap " << context_cap << '\n';
      return 1;
    }

    // The CPU backend stores float32 caches and never rounds activations.
    if (!gpu && kv_arg == "f16") {
      std::cerr << "the CPU backend stores float32 caches; --kv f16 needs the GPU backend\n";
      return 1;
    }
    const std::uint32_t pass_rows_max = std::min(rows, T_max);
    ExecutorConfig ec;
    ec.max_context = context_cap;
    ec.max_rows = pass_rows_max;
    ec.max_logit_rows = last_only ? 1 : pass_rows_max;
    ec.kv_type = !gpu || kv_arg == "f32" ? KvType::kF32 : KvType::kF16;
    ec.exact = exact || !gpu;

    RegisterGpuWeightsReader(*weights, reader);
    auto executor = CreateExecutor(*backend, *weights, ec, &error);
    if (!executor) {
      std::cerr << "executor: " << error << '\n';
      return 1;
    }
    const auto t_loaded = Clock::now();
    std::size_t mem_after_load = 0, mem_after_passes = 0, mem_total = 0;
    std::string mem_error;
    bool have_mem = false;
    if (gpu) have_mem = GpuMemoryUsage(&mem_after_load, &mem_total, &mem_error);
    if (list_mode) {
      std::printf("load: %.2f s (open, resolve, executor%s) for %zu texts, longest %u tokens\n",
                  Seconds(t_start, t_loaded), gpu ? ", weight upload" : "", jobs.size(), T_max);
      std::fflush(stdout);
    }

    for (std::size_t ji = 0; ji < jobs.size(); ++ji) {
      const std::vector<std::int32_t>& tokens = all_tokens[ji];
      const std::string& out_path = jobs[ji].second;
      const std::uint32_t T = static_cast<std::uint32_t>(tokens.size());
      const std::uint32_t pass_rows = std::min(rows, T);
      if (ji > 0) executor->Reset();  // empty sequence for every text; rows are rewritten before read

      const std::size_t out_rows = last_only ? 1 : T;
      std::vector<float> logits(out_rows * vocab);
      const auto t0 = Clock::now();
      std::uint32_t passes = 0;
      for (std::uint32_t at = 0; at < T; at += pass_rows, ++passes) {
        const std::uint32_t n = std::min(pass_rows, T - at);
        const bool final_pass = at + n == T;
        const auto chunk = std::span<const std::int32_t>(tokens).subspan(at, n);
        if (last_only) {
          if (final_pass) {
            executor->Forward(chunk, at, LogitRows::kLast, std::span<float>(logits));
          } else {
            executor->Forward(chunk, at, LogitRows::kNone, std::span<float>());
          }
        } else {
          executor->Forward(chunk, at, LogitRows::kAll,
                            std::span<float>(logits).subspan(std::size_t{at} * vocab,
                                                             std::size_t{n} * vocab));
        }
      }
      const auto t1 = Clock::now();
      if (gpu && have_mem) have_mem = GpuMemoryUsage(&mem_after_passes, &mem_total, &mem_error);

      std::size_t bad = 0;
      for (const float x : logits) bad += std::isfinite(x) ? 0 : 1;
      std::ofstream out(out_path, std::ios::binary);
      out.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * sizeof(float)));
      out.close();
      if (!out) {
        std::cerr << "cannot write " << out_path << '\n';
        return 1;
      }

      const double pass_s = Seconds(t0, t1);
      if (list_mode) std::printf("text %zu/%zu: ", ji + 1, jobs.size());
      std::printf("ricursive_gpu_logits: T=%u vocab=%u backend=%s kv=%s exact=%d rows/pass=%u passes=%u "
                  "context-cap=%u%s -> %s\n",
                  T, vocab, gpu ? "gpu" : "cpu", ec.kv_type == KvType::kF16 ? "f16" : "f32",
                  ec.exact ? 1 : 0, pass_rows, passes, context_cap,
                  last_only ? " last-only" : "", out_path.c_str());
      if (!list_mode) {
        std::printf("load: %.2f s (open, resolve, executor%s; %.2f s since start)\n",
                    Seconds(t_start, t_loaded), gpu ? ", weight upload" : "", Seconds(t_start, t1));
      }
      std::printf("passes: %u tokens in %.3f s = %.1f tokens/s (batch_invariant_rows %u)\n", T,
                  pass_s, pass_s > 0 ? T / pass_s : 0.0, executor->Info().batch_invariant_rows);
      if (!gpu) {
        std::printf("GPU memory: n/a (CPU backend)\n");
      } else if (have_mem) {
        std::printf("GPU memory in use (device-wide hipMemGetInfo, includes other processes; no "
                    "per-process peak API): %.0f MiB after load, %.0f MiB after passes, of %.0f MiB\n",
                    Mib(mem_after_load), Mib(mem_after_passes), Mib(mem_total));
      } else {
        std::printf("GPU memory: unavailable (%s)\n", mem_error.c_str());
      }
      std::fflush(stdout);
      if (bad != 0) {
        std::cerr << "ERROR: " << bad << " of " << logits.size() << " logits are NaN or Inf in "
                  << out_path << "; the output file is not usable\n";
        return 1;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

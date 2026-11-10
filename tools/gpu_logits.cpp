// ricursive_gpu_logits MODEL.gguf TOKENS.i32 OUT.f32
//     [--rows N per pass] [--kv f16|f32] [--exact] [--context-cap N] [--last-only]
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
#include <string>
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
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--rows" && i + 1 < argc) {
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
  if (pos.size() != 3) return Usage(), 2;

  try {
    const auto t_start = Clock::now();
    std::string error;
    const auto backend = BackendFromEnvironment(&error);
    if (!backend) {
      std::cerr << error << '\n';
      return 1;
    }
    const bool gpu = backend->backend == Backend::kGpu;

    std::ifstream in(pos[1], std::ios::binary | std::ios::ate);
    if (!in) {
      std::cerr << "cannot open " << pos[1] << '\n';
      return 1;
    }
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (bytes == 0 || bytes % 4 != 0) {
      std::cerr << pos[1] << ": size " << bytes << " is not a positive multiple of 4\n";
      return 1;
    }
    std::vector<std::int32_t> tokens(bytes / 4);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(bytes));
    if (!in) {
      std::cerr << "cannot read " << pos[1] << '\n';
      return 1;
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
    const std::uint32_t T = static_cast<std::uint32_t>(tokens.size());
    const std::uint32_t vocab = weights->config.vocab_size;
    if (context_cap == 0) context_cap = T;
    if (T > context_cap) {
      std::cerr << T << " tokens exceed --context-cap " << context_cap << '\n';
      return 1;
    }

    // The CPU backend stores float32 caches and never rounds activations.
    if (!gpu && kv_arg == "f16") {
      std::cerr << "the CPU backend stores float32 caches; --kv f16 needs the GPU backend\n";
      return 1;
    }
    const std::uint32_t pass_rows = std::min(rows, T);
    ExecutorConfig ec;
    ec.max_context = context_cap;
    ec.max_rows = pass_rows;
    ec.max_logit_rows = last_only ? 1 : pass_rows;
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
    std::ofstream out(pos[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * sizeof(float)));
    out.close();
    if (!out) {
      std::cerr << "cannot write " << pos[2] << '\n';
      return 1;
    }

    const double pass_s = Seconds(t0, t1);
    std::printf("ricursive_gpu_logits: T=%u vocab=%u backend=%s kv=%s exact=%d rows/pass=%u passes=%u "
                "context-cap=%u%s -> %s\n",
                T, vocab, gpu ? "gpu" : "cpu", ec.kv_type == KvType::kF16 ? "f16" : "f32",
                ec.exact ? 1 : 0, pass_rows, passes, context_cap,
                last_only ? " last-only" : "", pos[2].c_str());
    std::printf("load: %.2f s (open, resolve, executor%s; %.2f s since start)\n",
                Seconds(t_start, t_loaded), gpu ? ", weight upload" : "", Seconds(t_start, t1));
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
    if (bad != 0) {
      std::cerr << "ERROR: " << bad << " of " << logits.size()
                << " logits are NaN or Inf; the output file is not usable\n";
      return 1;
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

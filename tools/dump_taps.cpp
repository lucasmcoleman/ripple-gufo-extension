// ricursive_dump_taps MODEL.gguf --tokens FILE.i32 --prompt-tokens P --generate G --out DIR
//     [--backend cpu|host|gpu] [--layers a,b,c,d,e] [--rows R] [--verify-rows V]
//     [--context-cap N]
// ricursive_dump_taps MODEL.gguf --jobs JOBS.txt --generate G [same options]
//
// Stage 0 of the DFlash2 drafter design (docs/DRAFTER-STAGE0.md). No HTTP.
//
//  (a) Greedy generation as serving does it: the prompt (first P tokens of the
//      token file) in passes of at most R rows (last row's logits), then G
//      tokens, each fed back in a 1-row pass. The argmax of every logits row
//      that produced a token is kept, with a hash of the row.
//  (b) The same sequence (prompt + generated) evaluated again from scratch with
//      the capture on: the prompt in the same passes (taps of every prompt
//      position), then the G generated tokens in passes of V rows (default 16)
//      with every row's logits. It saves the captured outputs of the
//      configured layers for all P+G positions and the hash and argmax of each
//      generated position's logits row.
//
// Exactness precondition (design acceptance test 2): the logits row at the
// position that followed generated token i-1 must hash equal in (a) (a 1-row
// pass) and in (b) (a row of a V-row pass). The tool compares them and prints
// `EXACT: n of m`; the numbers are in meta.json for the report script.
//
// Jobs mode: one model load for many prompts. JOBS.txt has one job per line,
// "TOKENS.i32 PROMPT_TOKENS OUTDIR" (whitespace separated; blank lines and
// lines starting with # ignored); every job generates G tokens. A job whose
// OUTDIR/meta.json already exists is skipped, so a run cut off by a time limit
// is resumed by running the same command again. The executor is sized for the
// longest job; a job that fails stops the run (exit 1), finished jobs stay.
//
// Output (OUTDIR, created if missing; meta.json is written last, so its
// presence means the dump is complete):
//   ids.i32       int32 [P+G]            prompt then generated token ids
//   taps.f32      float32 [P+G][K][H]    layer outputs (l_out), position major,
//                                        tap k = layers[k]; one row of K*H
//                                        floats is the drafter's feature input
//   meta.json     sizes, layers, backend, hashes (hex FNV-1a 64 of the float
//                 bytes of a logits row) and argmaxes
// Backends: gpu = production executor (only through the GPU wrapper); cpu =
// scalar reference executor (tiny models only); host = the executor's own
// orchestration on host ops (float32 caches), the CPU stand-in for the GPU path.
// Exit 0 written, 1 failure, 2 usage. Exactness mismatches do not change the
// exit code: they are the measurement.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "hip/executor.hpp"
#include "hip/gpu_executor_factory.hpp"
#include "src/executor_factory.hpp"
#include "src/row_hash.hpp"

using namespace gufo::models::ricursive;
using Clock = std::chrono::steady_clock;

namespace {

double Seconds(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

void Usage() {
  std::cerr
      << "usage: ricursive_dump_taps MODEL.gguf --tokens FILE.i32 --prompt-tokens P --generate G "
         "--out DIR [options]\n"
         "       ricursive_dump_taps MODEL.gguf --jobs JOBS.txt --generate G [options]\n"
         "  JOBS.txt lines: TOKENS.i32 PROMPT_TOKENS OUTDIR (jobs with OUTDIR/meta.json are skipped)\n"
         "  options: [--backend cpu|host|gpu] [--layers a,b,c,d,e] [--rows R] [--verify-rows V] "
         "[--context-cap N] [--stop ID,ID|none] [--sweep-widths]\n"
         "  --sweep-widths: also check every pass width 2..V against the 1-row run (prints WIDTH w lines)\n"
         "  --stop defaults to 100265,100257 (the engine's stop tokens): generation ends after one of them, the id kept;\n"
         "  --generate is the most tokens generated.\n"
         "  --backend defaults to gpu; --rows defaults to 2048 (prompt rows per pass, as in serving); "
         "--verify-rows defaults to 16;\n"
         "  --layers defaults to 1,20,39,58,77 (zero-based; the output of each layer).\n";
}

bool ParseUnsigned(const std::string& text, std::uint32_t* out, bool allow_zero = false) {
  char* end = nullptr;
  const unsigned long v = std::strtoul(text.c_str(), &end, 10);
  if (text.empty() || *end != '\0' || (v == 0 && !allow_zero) || v > 0xFFFFFFFFUL) return false;
  *out = static_cast<std::uint32_t>(v);
  return true;
}

bool ParseLayers(const std::string& text, std::vector<std::uint32_t>* out) {
  out->clear();
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    std::uint32_t v = 0;
    if (!ParseUnsigned(item, &v, true)) return false;
    out->push_back(v);
  }
  return !out->empty();
}

std::int32_t ArgMax(std::span<const float> logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<std::int32_t>(best);
}

std::string Hex(std::uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

std::uint32_t RoundUp(std::uint32_t value, std::uint32_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

template<class T>
void WriteVector(const std::filesystem::path& path, const std::vector<T>& v) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()),
            static_cast<std::streamsize>(v.size() * sizeof(T)));
  out.close();
  if (!out) throw std::runtime_error("cannot write " + path.string());
}

struct Job {
  std::string tokens_path;
  std::uint32_t prompt_tokens{0};
  std::string out_dir;
  std::vector<std::int32_t> prompt;  ///< read before the model load
};

/// Everything a job needs that does not change between jobs.
struct Setup {
  std::string model_path, backend;
  std::uint32_t generate{0}, hidden{0}, vocab{0};
  std::vector<std::uint32_t> layers;
  std::vector<std::int32_t> stop;  ///< generation ends after one of these ids (the id is kept)
  bool sweep_widths{false};        ///< also check every pass width 2..verify_rows against the 1-row run
  ExecutorConfig ec;
};

/// One job: (a) plain greedy generation, (b) the capture pass, then the files. Returns false on a
/// failure that stops the run.
void RunJob(Executor& executor, const Setup& s, Job& job, std::size_t index, std::size_t count) {
  const auto t_job = Clock::now();
  const std::uint32_t P = job.prompt_tokens, G_max = s.generate, vocab = s.vocab, hidden = s.hidden;
  const std::size_t K = s.layers.size();
  const std::uint32_t rows = s.ec.max_rows, verify_rows = s.ec.max_logit_rows;
  std::vector<std::int32_t> ids = job.prompt;
  std::vector<float> row(vocab);
  std::printf("job %zu/%zu: %s prompt=%u generate<=%u -> %s\n", index + 1, count,
              job.tokens_path.c_str(), P, G_max, job.out_dir.c_str());
  std::fflush(stdout);

  // (a) plain greedy generation, capture off (no buffer is given).
  executor.Reset();
  std::vector<std::uint64_t> gen_hash;
  gen_hash.reserve(G_max);
  std::vector<std::int32_t> generated;
  generated.reserve(G_max);
  const auto a0 = Clock::now();
  for (std::uint32_t at = 0; at < P; at += rows) {
    const std::uint32_t n = std::min(rows, P - at);
    const bool last = at + n == P;
    executor.Forward(std::span<const std::int32_t>(ids).subspan(at, n), at,
                     last ? LogitRows::kLast : LogitRows::kNone,
                     last ? std::span<float>(row) : std::span<float>());
  }
  const auto a1 = Clock::now();
  const std::uint64_t prefill_hash = RowHash(row);
  bool stopped = false;
  for (std::uint32_t i = 0; i < G_max; ++i) {
    gen_hash.push_back(RowHash(row));
    const std::int32_t token = ArgMax(row);
    generated.push_back(token);
    if (std::find(s.stop.begin(), s.stop.end(), token) != s.stop.end()) {
      stopped = true;
      break;
    }
    if (i + 1 < G_max) {
      const std::int32_t one[] = {token};
      executor.Forward(one, P + i, LogitRows::kLast, row);
    }
  }
  const auto a2 = Clock::now();
  const auto G = static_cast<std::uint32_t>(generated.size()), total = P + G;
  std::printf("  (a) prompt %.2f s, %u generated tokens%s in %.2f s (1-row passes, capture off)\n",
              Seconds(a0, a1), G, stopped ? " (ended with a stop token)" : "", Seconds(a1, a2));
  std::fflush(stdout);
  ids.insert(ids.end(), generated.begin(), generated.end());

  // (b) re-evaluation with the capture on.
  executor.Reset();
  std::vector<float> taps(std::size_t{total} * K * hidden);
  std::vector<float> capture(K * std::size_t{rows} * hidden);
  auto store = [&](std::uint32_t at, std::uint32_t n) {
    for (std::size_t k = 0; k < K; ++k) {
      for (std::uint32_t t = 0; t < n; ++t) {
        std::copy_n(capture.data() + (k * n + t) * hidden, hidden,
                    taps.data() + ((std::size_t{at} + t) * K + k) * hidden);
      }
    }
  };
  const auto b0 = Clock::now();
  std::uint64_t prefill_hash_b = 0;
  for (std::uint32_t at = 0; at < P; at += rows) {
    const std::uint32_t n = std::min(rows, P - at);
    const bool last = at + n == P;
    executor.Forward(std::span<const std::int32_t>(ids).subspan(at, n), at,
                     last ? LogitRows::kLast : LogitRows::kNone,
                     last ? std::span<float>(row) : std::span<float>(), {},
                     std::span<float>(capture).first(K * std::size_t{n} * hidden));
    store(at, n);
    if (last) prefill_hash_b = RowHash(row);
  }
  const auto b1 = Clock::now();
  std::vector<std::uint64_t> verify_hash;
  std::vector<std::int32_t> verify_argmax;
  verify_hash.reserve(G);
  std::vector<float> rows_logits(std::size_t{verify_rows} * vocab);
  for (std::uint32_t at = 0; at < G; at += verify_rows) {
    const std::uint32_t n = std::min(verify_rows, G - at);
    executor.Forward(std::span<const std::int32_t>(ids).subspan(P + at, n), P + at, LogitRows::kAll,
                     std::span<float>(rows_logits).first(std::size_t{n} * vocab), {},
                     std::span<float>(capture).first(K * std::size_t{n} * hidden));
    store(P + at, n);
    for (std::uint32_t t = 0; t < n; ++t) {
      const std::span<const float> r(rows_logits.data() + std::size_t{t} * vocab, vocab);
      verify_hash.push_back(RowHash(r));
      verify_argmax.push_back(ArgMax(r));
    }
  }
  const auto b2 = Clock::now();
  std::printf("  (b) prompt %.2f s (capture on), %u generated tokens in %.2f s (%u-row passes, capture on)\n",
              Seconds(b0, b1), G, Seconds(b1, b2), verify_rows);

  // Exactness: verify row j is the logits at position P+j, which produced generated token j+1 in (a).
  std::size_t equal = 0, compared = 0, argmax_equal = 0;
  for (std::uint32_t j = 0; j + 1 < G; ++j) {
    ++compared;
    equal += verify_hash[j] == gen_hash[j + 1] ? 1 : 0;
    argmax_equal += verify_argmax[j] == generated[j + 1] ? 1 : 0;
  }
  std::printf("  EXACT: %zu of %zu logits rows hash-equal between %u-row passes and 1-row passes "
              "(argmax equal %zu of %zu); prompt-pass last row %s\n",
              equal, compared, verify_rows, argmax_equal, compared,
              prefill_hash == prefill_hash_b ? "equal" : "DIFFERENT");

  // --sweep-widths: the same generated tokens evaluated in passes of every width 2..verify_rows (capture off, as a
  // serving verification runs), from the same prompt state (rewound, not recomputed), each row compared with the
  // 1-row run's. This is the exactness a drafted step rests on: a k-row verification equals k 1-row steps.
  if (s.sweep_widths) {
    std::size_t bad_widths = 0;
    for (std::uint32_t w = 2; w <= verify_rows; ++w) {
      if (!executor.Rewind(P)) {
        std::printf("  WIDTH %u: cannot rewind to the prompt end (the ring no longer holds its window)\n", w);
        ++bad_widths;
        continue;
      }
      std::size_t eq = 0, cmp = 0;
      for (std::uint32_t at = 0; at < G; at += w) {
        const std::uint32_t n = std::min(w, G - at);
        executor.Forward(std::span<const std::int32_t>(ids).subspan(P + at, n), P + at, LogitRows::kAll,
                         std::span<float>(rows_logits).first(std::size_t{n} * vocab));
        for (std::uint32_t t = 0; t < n; ++t) {
          const std::uint32_t j = at + t;  // logits at position P+j produced generated token j+1 in (a)
          if (j + 1 >= G) continue;
          ++cmp;
          eq += RowHash(std::span<const float>(rows_logits.data() + std::size_t{t} * vocab, vocab)) == gen_hash[j + 1] ? 1 : 0;
        }
      }
      std::printf("  WIDTH %u: %zu of %zu logits rows hash-equal to 1-row passes\n", w, eq, cmp);
      if (eq != cmp) ++bad_widths;
    }
    std::printf("  WIDTHS: %s\n", bad_widths == 0 ? "all pass widths 2.. equal 1-row passes" : "SOME WIDTH DIFFERS FROM 1-ROW PASSES");
  }

  // Output.
  std::filesystem::create_directories(job.out_dir);
  const std::filesystem::path dir(job.out_dir);
  WriteVector(dir / "ids.i32", ids);
  WriteVector(dir / "taps.f32", taps);
  auto array = [](const auto& v, auto fmt) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + fmt(v[i]);
    return out + "]";
  };
  std::ostringstream meta;
  meta << "{\n"
       << "  \"format\": 1,\n"
       << "  \"model\": \"" << std::filesystem::path(s.model_path).filename().string() << "\",\n"
       << "  \"backend\": \"" << s.backend << "\",\n"
       << "  \"prompt_len\": " << P << ",\n"
       << "  \"generated\": " << G << ",\n"
       << "  \"stopped\": " << (stopped ? "true" : "false") << ",\n"
       << "  \"stop_tokens\": " << array(s.stop, [](std::int32_t v) { return std::to_string(v); }) << ",\n"
       << "  \"hidden\": " << hidden << ",\n"
       << "  \"vocab\": " << vocab << ",\n"
       << "  \"tap_layers\": " << array(s.layers, [](std::uint32_t v) { return std::to_string(v); })
       << ",\n"
       << "  \"tap_point\": \"l_out (layer output, before the final norm), zero-based layer\",\n"
       << "  \"prefill_rows\": " << rows << ",\n"
       << "  \"verify_rows\": " << verify_rows << ",\n"
       << "  \"ids_file\": \"ids.i32\",\n"
       << "  \"taps_file\": \"taps.f32\",\n"
       << "  \"taps_layout\": \"float32 [prompt_len+generated][len(tap_layers)][hidden]\",\n"
       << "  \"hash\": \"FNV-1a 64 over the float bytes of one logits row\",\n"
       << "  \"gen_hash\": \"gen_hash[i] = hash of the logits row at position prompt_len-1+i (1-row passes; "
          "prompt_len-1 is the last prompt pass), which produced generated token i\",\n"
       << "  \"gen_hash_values\": "
       << array(gen_hash, [](std::uint64_t v) { return "\"" + Hex(v) + "\""; }) << ",\n"
       << "  \"verify_hash\": \"verify_hash[j] = hash of the logits row at position prompt_len+j from a "
          "verify_rows-row pass; equals gen_hash[j+1] when passes are bit-exact\",\n"
       << "  \"verify_hash_values\": "
       << array(verify_hash, [](std::uint64_t v) { return "\"" + Hex(v) + "\""; }) << ",\n"
       << "  \"verify_argmax\": "
       << array(verify_argmax, [](std::int32_t v) { return std::to_string(v); }) << ",\n"
       << "  \"prefill_hash_a\": \"" << Hex(prefill_hash) << "\",\n"
       << "  \"prefill_hash_b\": \"" << Hex(prefill_hash_b) << "\",\n"
       << "  \"exact_equal\": " << equal << ",\n"
       << "  \"exact_compared\": " << compared << ",\n"
       << "  \"seconds\": {\"a\": " << Seconds(a0, a2) << ", \"b\": " << Seconds(b0, b2) << "}\n"
       << "}\n";
  {
    std::ofstream m(dir / "meta.json.tmp");
    m << meta.str();
    m.close();
    if (!m) throw std::runtime_error("cannot write meta.json");
  }
  std::filesystem::rename(dir / "meta.json.tmp", dir / "meta.json");
  std::printf("  wrote %s (ids %zu bytes, taps %zu bytes) in %.2f s\n", job.out_dir.c_str(),
              ids.size() * 4, taps.size() * 4, Seconds(t_job, Clock::now()));
  std::fflush(stdout);
}

bool ReadTokens(Job* job, std::string* error) {
  std::ifstream in(job->tokens_path, std::ios::binary | std::ios::ate);
  if (!in) {
    *error = "cannot open " + job->tokens_path;
    return false;
  }
  const auto bytes = static_cast<std::size_t>(in.tellg());
  if (bytes == 0 || bytes % 4 != 0 || bytes / 4 < job->prompt_tokens) {
    *error = job->tokens_path + ": " + std::to_string(bytes) +
             " bytes is not a multiple of 4 holding at least " +
             std::to_string(job->prompt_tokens) + " tokens";
    return false;
  }
  job->prompt.resize(job->prompt_tokens);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(job->prompt.data()),
          static_cast<std::streamsize>(job->prompt.size() * sizeof(std::int32_t)));
  if (!in) {
    *error = "cannot read " + job->tokens_path;
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, tokens_path, out_dir, jobs_path, backend_arg = "gpu";
  std::uint32_t prompt_tokens = 0, generate = 0, rows = 2048, verify_rows = 16, context_cap = 0;
  std::vector<std::uint32_t> layers = {1, 20, 39, 58, 77};
  std::vector<std::int32_t> stop = {100265, 100257};  // <|im_end|>, <|endoftext|>: the engine's stop tokens
  bool have_prompt = false, have_generate = false, sweep_widths = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const bool more = i + 1 < argc;
    if (a == "--tokens" && more) {
      tokens_path = argv[++i];
    } else if (a == "--prompt-tokens" && more) {
      have_prompt = ParseUnsigned(argv[++i], &prompt_tokens);
      if (!have_prompt) return Usage(), 2;
    } else if (a == "--generate" && more) {
      have_generate = ParseUnsigned(argv[++i], &generate);
      if (!have_generate) return Usage(), 2;
    } else if (a == "--out" && more) {
      out_dir = argv[++i];
    } else if (a == "--jobs" && more) {
      jobs_path = argv[++i];
    } else if (a == "--backend" && more) {
      backend_arg = argv[++i];
    } else if (a == "--layers" && more) {
      if (!ParseLayers(argv[++i], &layers)) return Usage(), 2;
    } else if (a == "--rows" && more) {
      if (!ParseUnsigned(argv[++i], &rows)) return Usage(), 2;
    } else if (a == "--verify-rows" && more) {
      if (!ParseUnsigned(argv[++i], &verify_rows)) return Usage(), 2;
    } else if (a == "--sweep-widths") {
      sweep_widths = true;
    } else if (a == "--stop" && more) {
      std::vector<std::uint32_t> v;
      const std::string list = argv[++i];
      if (list != "none" && !ParseLayers(list, &v)) return Usage(), 2;
      stop.assign(v.begin(), v.end());
    } else if (a == "--context-cap" && more) {
      if (!ParseUnsigned(argv[++i], &context_cap)) return Usage(), 2;
    } else if (!a.empty() && a[0] == '-') {
      std::cerr << "unknown option " << a << '\n';
      return Usage(), 2;
    } else if (model_path.empty()) {
      model_path = a;
    } else {
      return Usage(), 2;
    }
  }
  const bool single = !tokens_path.empty() || !out_dir.empty() || have_prompt;
  if (model_path.empty() || !have_generate || (backend_arg != "cpu" && backend_arg != "host" && backend_arg != "gpu") ||
      (jobs_path.empty() == !single) ||
      (single && (tokens_path.empty() || out_dir.empty() || !have_prompt))) {
    return Usage(), 2;
  }

  try {
    const auto t_start = Clock::now();
    std::vector<Job> jobs;
    if (single) {
      jobs.push_back({tokens_path, prompt_tokens, out_dir, {}});
    } else {
      std::ifstream lf(jobs_path);
      if (!lf) {
        std::cerr << "cannot open jobs file " << jobs_path << '\n';
        return 1;
      }
      std::string line;
      while (std::getline(lf, line)) {
        std::istringstream ls(line);
        std::string tokens, plen, out, extra;
        if (!(ls >> tokens) || tokens[0] == '#') continue;
        std::uint32_t p = 0;
        if (!(ls >> plen >> out) || (ls >> extra) || !ParseUnsigned(plen, &p)) {
          std::cerr << "job line needs exactly TOKENS.i32 PROMPT_TOKENS OUTDIR: " << line << '\n';
          return 1;
        }
        jobs.push_back({tokens, p, out, {}});
      }
      if (jobs.empty()) {
        std::cerr << "no jobs in " << jobs_path << '\n';
        return 1;
      }
    }
    // Jobs already done are skipped; read every remaining token file before the model load.
    std::vector<Job> todo;
    std::uint32_t longest = 0;
    std::string error;
    for (Job& job : jobs) {
      if (std::filesystem::exists(std::filesystem::path(job.out_dir) / "meta.json")) {
        std::printf("skip (done): %s\n", job.out_dir.c_str());
        continue;
      }
      if (!ReadTokens(&job, &error)) {
        std::cerr << error << '\n';
        return 1;
      }
      longest = std::max(longest, job.prompt_tokens + generate);
      todo.push_back(std::move(job));
    }
    if (todo.empty()) {
      std::printf("nothing to do\n");
      return 0;
    }

    const std::shared_ptr<const gufo::core::GgufReader> reader =
        gufo::core::GgufReader::OpenFile(model_path, &error);
    if (!reader) {
      std::cerr << "open " << model_path << ": " << error << '\n';
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
    Setup s;
    s.model_path = model_path;
    s.backend = backend_arg;
    s.generate = generate;
    s.sweep_widths = sweep_widths;
    s.vocab = weights->config.vocab_size;
    s.hidden = weights->config.embedding_length;
    s.layers = layers;
    s.stop = stop;
    for (const Job& job : todo) {
      for (const std::int32_t t : job.prompt) {
        if (t < 0 || static_cast<std::uint32_t>(t) >= s.vocab) {
          std::cerr << job.tokens_path << ": a prompt token is outside the vocabulary (" << s.vocab << ")\n";
          return 1;
        }
      }
    }
    const std::uint32_t cap = context_cap != 0 ? context_cap : longest;
    if (longest > cap) {
      std::cerr << "prompt + generate = " << longest << " exceeds the context cap " << cap << '\n';
      return 1;
    }

    const bool gpu = backend_arg == "gpu";
    ExecutorConfig& ec = s.ec;
    ec.max_context = cap;
    ec.max_rows = std::min(std::max(rows, verify_rows), cap);
    ec.max_logit_rows = std::min(verify_rows, ec.max_rows);
    ec.kv_type = gpu ? KvType::kF16 : KvType::kF32;
    ec.exact = !gpu;
    ec.capture_layers = layers;
    // The ring serving uses (engine.cpp): a multiple of 64 holding a full pass.
    const std::uint32_t window = weights->config.sliding_window;
    ec.ring_rows = RoundUp(std::max(MinRingRows(window, ec.max_rows), window - 1 + 64 - 1), 64);
    std::unique_ptr<Executor> executor;
    if (backend_arg == "host") {
      executor = std::make_unique<HostOpsExecutor>(*weights, ec);
    } else {
      if (gpu) RegisterGpuWeightsReader(*weights, reader);
      executor = CreateExecutor(gpu ? GpuBackend() : CpuBackend(), *weights, ec, &error);
      if (!executor) {
        std::cerr << "executor: " << error << '\n';
        return 1;
      }
    }
    std::printf("ricursive_dump_taps: backend=%s jobs=%zu generate=%u rows/pass=%u verify-rows=%u layers=",
                backend_arg.c_str(), todo.size(), generate, ec.max_rows, verify_rows);
    for (std::size_t k = 0; k < layers.size(); ++k) std::printf("%s%u", k ? "," : "", layers[k]);
    std::printf(" context-cap=%u\nload: %.2f s\n", cap, Seconds(t_start, Clock::now()));
    std::fflush(stdout);

    for (std::size_t i = 0; i < todo.size(); ++i) RunJob(*executor, s, todo[i], i, todo.size());
    std::printf("done: %zu jobs in %.2f s\n", todo.size(), Seconds(t_start, Clock::now()));
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

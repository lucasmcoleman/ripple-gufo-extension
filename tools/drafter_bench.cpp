// ricursive_drafter_bench DRAFTER.gguf TARGET.gguf --dump DIR [--contexts 500,4096,16384] [--draft 1,2,3,4,6,15]
//     [--steps 20] [--warmup 3] [--appends 5] [--kv f16|f32] [--backend gpu|host] [--append-rows 512]
//     [--engine head|factory] [--q8 none/block/head/context/block,head,context/all ...] [--attention default]
//     [--drafts-out FILE.i32 --from A --to B]
//
// Stage 2b switches (docs/DRAFTER-S2.md): --engine head (default) uploads the target's output head alone (0.42 GB);
// --engine factory is the production path, MakeGpuDrafterEngine over the target's device copy (the whole 40 GB target is
// uploaded first: its time is printed). --q8 takes route sets separated by '/' (groups block, head, context of
// DrafterOptions::q8; "none" = today's behaviour) and runs the whole table once per set, in one process (the drafter's weights
// stay resident between sets). --attention NAME picks the attention kernel by name (only "default" exists; stage 2c adds a
// second). The cost of creating an engine is printed for the first set: the first engine (which uploads the drafter's weights
// unless pinned) and a second one made from the resident weights.
//
// With --drafts-out the tool does not time anything: for every anchor s in [A, B) of the dump it drafts the
// whole block (15 tokens, the context being the dump's rows 0 .. s-1) and writes the tokens as int32
// [B - A][15] to FILE (tools/drafter_compare.py compares them with the oracle's R drafts of stage 0).
//
// Times the DFlash2 drafter alone (docs/DRAFTER-S2.md): loads the drafter and, from the target, only the
// token table and the output head (not the layers); fills a context of N rows from a stage 0 dump
// (ricursive_dump_taps output: ids.i32, taps.f32, meta.json; a dump with fewer rows than N is repeated
// cyclically, which changes nothing about the cost of a step); then, for every context N,
//   - times Draft (one DraftStep: the block through the 5 layers, the output head and the selector for
//     d rows) for every d: median, 10th and 90th percentile, min and max of --steps steps after --warmup
//     warm-up steps, each step ending with the host reading the tokens back (the device is waited for);
//   - times appending 1 and 4 context rows (median of --appends repetitions) and, when N >= 2048,
//     appending 2048 rows into an empty context (once).
// With RICURSIVE_PROFILE=1 it then prints where a step's time goes (the device is waited for at each
// stage boundary, so the profile run is slower than the table's; it is a separate pass).
// Backend gpu only through the GPU wrapper (perf/gpu-run.sh); host is for tiny fixtures.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hip/drafter_engine.hpp"
#include "hip/drafter_gpu.hpp"
#include "hip/gpu_executor_factory.hpp"
#include "src/config.hpp"
#include "src/drafter_config.hpp"
#include "src/drafter_weights.hpp"
#include "src/weights.hpp"

using namespace gufo::models::ricursive;
using Clock = std::chrono::steady_clock;

namespace {

void Usage() {
  std::cerr << "usage: ricursive_drafter_bench DRAFTER.gguf TARGET.gguf --dump DIR [--contexts 500,4096] "
               "[--draft 1,2,3,4,6,15] [--steps 20] [--warmup 3] [--appends 5] [--kv f16|f32] "
               "[--backend gpu|host] [--append-rows 512] [--engine head|factory] [--q8 SET/SET...] [--attention NAME]\n";
}

std::vector<std::uint32_t> ParseList(const std::string& text) {
  std::vector<std::uint32_t> out;
  std::stringstream ss(text);
  for (std::string item; std::getline(ss, item, ',');) out.push_back(static_cast<std::uint32_t>(std::stoul(item)));
  return out;
}

template<class T>
std::vector<T> ReadBinary(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in.good()) throw std::runtime_error("cannot open " + path);
  const auto bytes = static_cast<std::size_t>(in.tellg());
  std::vector<T> v(bytes / sizeof(T));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
  return v;
}

std::string Slurp(const std::string& path) {
  std::ifstream in(path);
  if (!in.good()) throw std::runtime_error("cannot open " + path);
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}

struct Stats {
  double median{0}, p10{0}, p90{0}, min{0}, max{0};
};

Stats Summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const auto at = [&](double q) { return v[std::min(v.size() - 1, static_cast<std::size_t>(q * static_cast<double>(v.size())))]; };
  return {at(0.5), at(0.1), at(0.9), v.front(), v.back()};
}

double Ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // a timeout or a GPU fault must not lose the line that explains it (review of stage 2b, finding 6)
  if (argc < 3) {
    Usage();
    return 2;
  }
  std::string drafter_path = argv[1], target_path = argv[2], dump_dir, backend = "gpu", kv = "f16";
  std::vector<std::uint32_t> contexts{500, 4096, 16384}, drafts{1, 2, 3, 4, 6, 15};
  std::string engine_kind = "head", attention = "default";
  std::vector<DrafterRoutes> route_sets{DrafterRoutes{}};
  std::uint32_t steps = 20, warmup = 3, appends = 5, append_rows = 512, from = 0, to = 0;
  std::string drafts_out;
  try {
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      const auto next = [&]() -> std::string {
        if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
        return argv[++i];
      };
      if (a == "--dump") dump_dir = next();
      else if (a == "--contexts") contexts = ParseList(next());
      else if (a == "--draft") drafts = ParseList(next());
      else if (a == "--steps") steps = static_cast<std::uint32_t>(std::stoul(next()));
      else if (a == "--warmup") warmup = static_cast<std::uint32_t>(std::stoul(next()));
      else if (a == "--appends") appends = static_cast<std::uint32_t>(std::stoul(next()));
      else if (a == "--append-rows") append_rows = static_cast<std::uint32_t>(std::stoul(next()));
      else if (a == "--drafts-out") drafts_out = next();
      else if (a == "--from") from = static_cast<std::uint32_t>(std::stoul(next()));
      else if (a == "--to") to = static_cast<std::uint32_t>(std::stoul(next()));
      else if (a == "--engine") engine_kind = next();
      else if (a == "--attention") attention = next();
      else if (a == "--q8") {
        route_sets.clear();
        const std::string list = next();
        std::stringstream ss(list);
        for (std::string item; std::getline(ss, item, '/');) route_sets.push_back(ParseDrafterRoutes(item));
        if (route_sets.empty()) route_sets.push_back(DrafterRoutes{});
      }
      else if (a == "--kv") kv = next();
      else if (a == "--backend") backend = next();
      else throw std::invalid_argument("unknown argument " + a);
    }
    if (!drafts_out.empty() && !(from < to)) throw std::invalid_argument("--drafts-out needs --from A --to B with A < B");
    if (dump_dir.empty() || (kv != "f16" && kv != "f32") || (backend != "gpu" && backend != "host") || (engine_kind != "head" && engine_kind != "factory") || (engine_kind == "factory" && backend != "gpu") || steps < 1 || appends < 1) {
      Usage();
      return 2;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    Usage();
    return 2;
  }
  try {
    std::string error;
    std::shared_ptr<const gufo::core::GgufReader> target_reader(gufo::core::GgufReader::OpenFile(target_path, &error));
    if (!target_reader) throw std::runtime_error("open " + target_path + ": " + error);
    const auto target_config = ParseConfig(*target_reader, &error);
    if (!target_config) throw std::runtime_error(target_path + ": " + error);
    const auto target = ResolveWeights(*target_reader, *target_config, &error);
    if (!target) throw std::runtime_error(target_path + ": " + error);
    std::shared_ptr<const gufo::core::GgufReader> drafter_reader(gufo::core::GgufReader::OpenFile(drafter_path, &error));
    if (!drafter_reader) throw std::runtime_error("open " + drafter_path + ": " + error);
    const auto drafter_config = ParseDrafterConfig(*drafter_reader, *target_config, &error);
    if (!drafter_config) throw std::runtime_error(drafter_path + ": " + error);
    const auto drafter = ResolveDrafterWeights(*drafter_reader, *drafter_config, *target_config, &error);
    if (!drafter) throw std::runtime_error(drafter_path + ": " + error);

    const std::string meta = Slurp(dump_dir + "/meta.json");
    const auto ids = ReadBinary<std::int32_t>(dump_dir + "/ids.i32");
    const auto taps = ReadBinary<float>(dump_dir + "/taps.f32");
    const std::size_t F = drafter->config.FeatureWidth(), rows_available = ids.size();
    if (rows_available == 0 || taps.size() != rows_available * F) {
      throw std::runtime_error("the dump's taps are not [rows][" + std::to_string(F) + "] floats for its " + std::to_string(rows_available) +
                               " ids (is its tap layer list the drafter's?)");
    }
    if (meta.find("\"tap_layers\"") == std::string::npos) throw std::runtime_error("meta.json has no tap_layers");

    // The factory path's target copy and the drafter's weights stay resident for the whole run, so that every engine after the
    // first is made from them (the cost of creating an engine is then the buffers, the scratch and the MASK rows).
    std::shared_ptr<const DeviceModel> target_device;
    std::shared_ptr<void> pinned;
    if (backend == "gpu") {
      if (engine_kind == "factory") {
        const auto t0 = Clock::now();
        RegisterGpuWeightsReader(*target, target_reader);
        target_device = AcquireGpuTargetModel(*target, &error);
        if (!target_device) throw std::runtime_error("uploading the target: " + error);
        std::printf("target model uploaded to the device (factory path): %.1f s\n", Ms(Clock::now() - t0) / 1e3);
      }
      const auto t0 = Clock::now();
      pinned = PinGpuDrafter(*drafter, drafter_reader, &error);
      if (!pinned) throw std::runtime_error("uploading the drafter: " + error);
      std::printf("drafter weights uploaded: %.2f s, %.2f GiB resident\n", Ms(Clock::now() - t0) / 1e3,
                  static_cast<double>(GpuDrafterWeightBytes(*drafter)) / (1024.0 * 1024.0 * 1024.0));
    }
    const auto make_engine = [&](const DrafterOptions& o) -> std::unique_ptr<DrafterEngine> {
      std::unique_ptr<DrafterEngine> e;
      if (backend == "gpu") {
        e = engine_kind == "factory" ? MakeGpuDrafterEngine(*drafter, drafter_reader, *target, o, &error)
                                     : MakeGpuDrafterEngineWithHead(*drafter, drafter_reader, *target, target_reader, o, &error);
        if (!e) throw std::runtime_error(error);
      } else {
        e = MakeHostOpsDrafterEngine(*drafter, *target, o);
      }
      return e;
    };

    bool first_set = true;
    for (const DrafterRoutes& routes : route_sets) {
      DrafterOptions options;
      options.kv_type = kv == "f16" ? KvType::kF16 : KvType::kF32;
      options.max_append_rows = append_rows;
      options.q8 = routes;
      options.attention_kernel = attention;
      const auto t_make = Clock::now();
      std::unique_ptr<DrafterEngine> engine = make_engine(options);
      const double first_ms = Ms(Clock::now() - t_make);
      if (first_set) {
        const auto t_second = Clock::now();
        auto second = make_engine(options);
        const double second_ms = Ms(Clock::now() - t_second);
        second.reset();
        std::printf("creating an engine: %.1f ms the first, %.1f ms a second one from the resident weights (%s head)\n", first_ms, second_ms,
                    engine_kind.c_str());
      }
      auto* ops_engine = dynamic_cast<OpsDrafterEngine*>(engine.get());
      DrafterExecutor& exec = ops_engine->executor();
      std::printf("\n==== drafter %s (%u layers, top-k %u, context window %u), target %s (%s), backend %s, context storage %s, "
                  "%u context rows per pass; projection routes on the 8-bit path: %s; attention kernel %s\n"
                  "     device bytes of one session: %.1f MiB allocated (DeviceBytes), %.1f MiB claimed (DrafterSessionBytes)\n",
                  drafter_path.c_str(), drafter->config.block_count, drafter->config.selector_top_k, drafter->config.context_window,
                  target_path.c_str(), engine_kind.c_str(), backend.c_str(), kv.c_str(), append_rows, ToString(options.q8).c_str(), attention.c_str(),
                  static_cast<double>(engine->DeviceBytes()) / (1024.0 * 1024.0),
                  static_cast<double>(DrafterSessionBytes(drafter->config, target->output.rows, options)) / (1024.0 * 1024.0));

      // Row i of a context is dump row i % rows_available (a longer context repeats the dump).
      const auto feature_rows = [&](std::size_t first, std::size_t n) {
        std::vector<float> out(n * F);
        for (std::size_t r = 0; r < n; ++r) std::copy_n(&taps[((first + r) % rows_available) * F], F, &out[r * F]);
        return out;
      };
      const auto sync = [&] { exec.ops().Synchronize(); };

      if (!drafts_out.empty()) {
        if (to > rows_available) throw std::runtime_error("--to " + std::to_string(to) + " is beyond the dump's " + std::to_string(rows_available) + " rows");
        engine->ResetTo(0);
        std::vector<std::int32_t> out;
        const auto t0 = Clock::now();
        for (std::uint32_t s = from; s < to; ++s) {
          if (engine->End() < s) engine->Append(engine->End(), feature_rows(engine->End(), s - engine->End()));
          const auto block = engine->Draft(ids[s], s);
          if (block.tokens.size() != drafter->config.block_size - 1) throw std::runtime_error("a draft did not return the whole block");
          out.insert(out.end(), block.tokens.begin(), block.tokens.end());
        }
        sync();
        std::ofstream file(drafts_out, std::ios::binary);
        file.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(std::int32_t)));
        if (!file.good()) throw std::runtime_error("cannot write " + drafts_out);
        std::printf("%u anchors [%u, %u) drafted in %.1f ms (%.2f ms per anchor, context appends included), written to %s\n", to - from, from, to,
                    Ms(Clock::now() - t0), Ms(Clock::now() - t0) / (to - from), drafts_out.c_str());
        return 0;
      }

      for (const std::uint32_t n : contexts) {
        if (n > rows_available) {
          std::printf("\ncontext %u rows: the dump holds %zu rows, so its rows are repeated (timing only)\n", n, rows_available);
        } else {
          std::printf("\ncontext %u rows (from the dump's first %u)\n", n, n);
        }
        engine->ResetTo(0);
        const auto t0 = Clock::now();
        for (std::uint32_t done = 0; done < n;) {  // in prefill-sized pieces, as the session delivers them
          const std::uint32_t piece = std::min<std::uint32_t>(2048, n - done);
          engine->Append(done, feature_rows(done, piece));
          done += piece;
        }
        sync();
        std::printf("  building the context: %.1f ms for %u rows (first use of every kernel included)\n", Ms(Clock::now() - t0), n);
        const std::int32_t anchor_token = ids[n % rows_available];
        std::printf("  %-4s %10s %10s %10s %10s %10s   (ms per Draft, %u steps after %u warm-up)\n", "d", "median", "p10", "p90", "min", "max", steps,
                    warmup);
        for (const std::uint32_t d : drafts) {
          for (std::uint32_t i = 0; i < warmup; ++i) (void)engine->Draft(anchor_token, n, d);
          std::vector<double> ms;
          for (std::uint32_t i = 0; i < steps; ++i) {
            const auto a = Clock::now();
            (void)engine->Draft(anchor_token, n, d);  // returns after the tokens are on the host
            ms.push_back(Ms(Clock::now() - a));
          }
          const Stats s = Summarize(ms);
          std::printf("  %-4u %10.3f %10.3f %10.3f %10.3f %10.3f\n", d, s.median, s.p10, s.p90, s.min, s.max);
        }
        // Appending rows: truncate to just below the end and append them again (inside the ring slack).
        for (const std::uint32_t k : {1U, 4U}) {
          std::vector<double> ms;
          for (std::uint32_t i = 0; i < appends + 1; ++i) {
            engine->TruncateTo(n - k);
            const auto rows = feature_rows(n - k, k);
            sync();
            const auto a = Clock::now();
            engine->Append(n - k, rows);
            sync();
            if (i > 0) ms.push_back(Ms(Clock::now() - a));  // the first repetition is a warm-up
          }
          const Stats s = Summarize(ms);
          std::printf("  append %u row%s: median %.3f ms (min %.3f, max %.3f, %u repetitions)\n", k, k == 1 ? "" : "s", s.median, s.min, s.max, appends);
        }
        if (n >= 2048) {
          engine->ResetTo(0);
          const auto rows = feature_rows(0, 2048);
          sync();
          const auto a = Clock::now();
          engine->Append(0, rows);
          sync();
          std::printf("  append 2048 rows into an empty context: %.3f ms (once)\n", Ms(Clock::now() - a));
          for (std::uint32_t done = 2048; done < n;) {  // restore the context for the profile below
            const std::uint32_t piece = std::min<std::uint32_t>(2048, n - done);
            engine->Append(done, feature_rows(done, piece));
            done += piece;
          }
          sync();
        }
        if (exec.Profiling()) {
          (void)exec.ProfileReport();  // forget everything before this pass
          for (std::uint32_t i = 0; i < steps; ++i) (void)engine->Draft(anchor_token, n, 15);
          std::printf("\n  profile, d = 15, context %u rows:\n", n);
          engine->TruncateTo(n - 4);
          engine->Append(n - 4, feature_rows(n - 4, 4));
          const std::string report = exec.ProfileReport();
          std::printf("%s", report.c_str());
        }
      }
      first_set = false;
    }
  } catch (const std::exception& e) {
    std::cerr << "ricursive_drafter_bench: " << e.what() << "\n";
    return 1;
  }
  return 0;
}

// Shared by tests/plain_fingerprint_test.cpp and tests/draft_session_test.cpp:
// engine models on the scalar CPU executor and on the host-ops executor (the
// orchestration the GPU executor shares, run on the CPU).
#ifndef RICURSIVE_TESTS_DRAFT_UTIL_HPP_
#define RICURSIVE_TESTS_DRAFT_UTIL_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tests/draft_hostops.hpp"
#include "tests/fixtures_text/engine_util.hpp"

namespace ricursive_test {

/// "cpu": CpuExecutor. "hostops": HostOpsExecutor (F32 caches, exact, like the
/// CPU backend's BackendChoice, so both are comparable bit for bit).
struct DraftBackend {
  std::string name;
  bool hostops{false};
};

inline std::vector<DraftBackend> DraftBackends() { return {{"cpu", false}, {"hostops", true}}; }

/// `configure` may change the ModelOptions after the defaults below (prefill
/// rows, context, window chunk; template check off).
inline std::shared_ptr<EngineModel> LoadDraftModel(
    const std::string& path, const DraftBackend& be, std::uint32_t context, std::uint32_t rows,
    std::uint32_t chunk, const std::function<void(ModelOptions&)>& configure = {}) {
  ModelOptions options;
  options.max_context = context;
  options.prefill_rows = rows;
  options.window_chunk_rows = chunk;
  options.allow_template_mismatch = true;
  options.backend = CpuBackend();
  if (be.hostops) {
    options.executor_override = [](const ModelWeights& w, const ExecutorConfig& c,
                                   std::string*) -> std::unique_ptr<Executor> {
      return MakeHostOpsExecutor(w, c);
    };
  }
  if (configure) configure(options);
  std::string error;
  auto model = EngineModel::Load(path, options, &error);
  Require(model != nullptr, "load " + path + ": " + error);
  return model;
}

/// `count` token ids in [0, 256) from a fixed seed (the tiny fixtures' vocabulary
/// is larger; the token files hold only 64 tokens).
inline std::vector<std::int32_t> SyntheticTokens(std::uint64_t seed, std::size_t count) {
  Rng rng(seed);
  std::vector<std::int32_t> out(count);
  for (auto& t : out) t = static_cast<std::int32_t>(rng.Below(256));
  return out;
}

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_DRAFT_UTIL_HPP_

// Half of ricursive_decode_bench that needs hip/executor.hpp. It cannot share a
// translation unit with src/engine.hpp (both declare kDefaultContextCap).
#include <exception>
#include <memory>
#include <string>

#include "hip/executor.hpp"

namespace gufo::models::ricursive {

/// The executor's orchestration on the host ops (float32 caches), or nullptr and *error.
std::unique_ptr<Executor> MakeHostOpsExecutor(const ModelWeights& weights,
                                              const ExecutorConfig& config, std::string* error) {
  try {
    return std::make_unique<HostOpsExecutor>(weights, config);
  } catch (const std::exception& e) {
    if (error != nullptr) *error = std::string("host-ops executor: ") + e.what();
    return nullptr;
  }
}

/// The per-stage report of an OpsExecutor with RICURSIVE_PROFILE=1; empty otherwise.
std::string StageProfileReport(Executor& executor) {
  auto* ops = dynamic_cast<OpsExecutor*>(&executor);
  return ops != nullptr ? ops->ProfileReport() : std::string();
}

}  // namespace gufo::models::ricursive

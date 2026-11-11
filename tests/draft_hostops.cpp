// The host-ops executor for the engine-level drafting tests. Separate unit:
// hip/executor.hpp and src/engine.hpp both define kDefaultContextCap, so no
// translation unit can include both.
#include "hip/executor.hpp"
#include "tests/draft_hostops.hpp"

namespace ricursive_test {

std::unique_ptr<gufo::models::ricursive::Executor> MakeHostOpsExecutor(
    const gufo::models::ricursive::ModelWeights& weights,
    const gufo::models::ricursive::ExecutorConfig& config) {
  gufo::models::ricursive::HostOpsExecutor::Options o;
  o.poison_memory = false;
  return std::make_unique<gufo::models::ricursive::HostOpsExecutor>(weights, config, o);
}

}  // namespace ricursive_test

// MakeHostOpsDrafterEngine lives with the host operations (gufo_ricursive_hostops), not in gufo_ricursive_exec:
// the release gufo binary links the executor library but not the host operations, and a reference to
// CreateHostOps from the executor library breaks its link (found 2026-11-11, first release build since stage 2).
#include "hip/drafter_engine.hpp"

namespace gufo::models::ricursive {

std::unique_ptr<OpsDrafterEngine> MakeHostOpsDrafterEngine(const DrafterWeights& drafter, const ModelWeights& target,
                                                           const DrafterOptions& options,
                                                           const DrafterExecFaults& faults,
                                                           const HostOpsOptions& host) {
  auto exec = std::make_unique<DrafterExecutor>(CreateHostOps(host), BuildHostDrafterHandles(drafter),
                                                BuildHostDrafterTarget(target), nullptr, options, faults);
  return std::make_unique<OpsDrafterEngine>(std::move(exec));
}

}  // namespace gufo::models::ricursive

// The host-ops backend (hip/ops_host.cpp) for the executor contract test, with
// every contract twin, and for tests/executor_test.cpp.
#include "hip/executor.hpp"
#include "tests/executor_contract.hpp"
#include "tests/executor_test_backend.hpp"

namespace ricursive_test {

using namespace gufo::models::ricursive;

std::vector<ContractBackend> ContractBackends() {
  ContractBackend host;
  host.name = "hostops";
  host.make = [](const Model& m, const ExecutorConfig& c) {
    return std::make_unique<HostOpsExecutor>(m.weights, c);
  };
  host.make_twin = [](const Model& m, const ExecutorConfig& c, Twin twin) {
    HostOpsExecutor::Options o;
    o.poison_memory = false;  // the contract test cannot compare NaN (see docs)
    switch (twin) {
      case Twin::kChunkLocalRope: o.faults.chunk_local_rope = true; break;
      case Twin::kRingOneShort:
        o.faults.ring_shortfall = 1;
        o.lenient_capacity = true;  // the kernels' own guard would refuse it
        break;
      case Twin::kWindowMinus1: o.faults.window_shortfall = 1; break;
      case Twin::kRetentionFromEnd:
        o.faults.ledger = Executor::LedgerFault::kRetentionFromEnd;
        break;
      case Twin::kFailureKeepsReady:
        o.faults.ledger = Executor::LedgerFault::kFailureKeepsReady;
        break;
      case Twin::kNoImportCoverage:
        o.faults.ledger = Executor::LedgerFault::kNoImportCoverage;
        break;
    }
    return std::make_unique<HostOpsExecutor>(m.weights, c, o);
  };
  host.kv_type = KvType::kF32;
  host.tap_tol = 2e-4;  // measured 2.1e-5 (docs/S2-EXECUTOR.md); the twin moves taps by 9.9
  return {host};
}

ExecBackend TestBackend() {
  ExecBackend b;
  b.name = "hostops";
  b.make = [](const Model& m, const ExecutorConfig& c, const ExecFaults& f,
              bool lenient) -> std::unique_ptr<Executor> {
    HostOpsExecutor::Options o;
    o.faults = f;
    o.lenient_capacity = lenient;
    return std::make_unique<HostOpsExecutor>(m.weights, c, o);
  };
  b.tap_rel_tol = 1e-4;
  b.exact_equal_tol = 0;
  b.production_equal_tol = 0;  // host ops never round activations
  b.quant_exact_max_abs = 1e-4;  // measured 8.6e-6 (float rounding)
  return b;
}

}  // namespace ricursive_test

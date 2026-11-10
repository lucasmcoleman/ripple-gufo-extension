// The CPU backend for the executor contract test, with every twin.
#include "src/cpu_executor.hpp"
#include "tests/executor_contract.hpp"

namespace ricursive_test {

using namespace gufo::models::ricursive;

std::vector<ContractBackend> ContractBackends() {
  ContractBackend cpu;
  cpu.name = "cpu";
  cpu.make = [](const Model& m, const ExecutorConfig& c) {
    return std::make_unique<CpuExecutor>(m.weights, c);
  };
  cpu.make_twin = [](const Model& m, const ExecutorConfig& c, Twin twin) {
    CpuExecutor::Faults f;
    switch (twin) {
      case Twin::kChunkLocalRope: f.chunk_local_rope = true; break;
      case Twin::kRingOneShort: f.ring_shortfall = 1; break;
      case Twin::kWindowMinus1: f.variant = Variant::kWinMinus1; break;
      case Twin::kRetentionFromEnd:
        f.ledger = Executor::LedgerFault::kRetentionFromEnd;
        break;
      case Twin::kFailureKeepsReady:
        f.ledger = Executor::LedgerFault::kFailureKeepsReady;
        break;
      case Twin::kNoImportCoverage:
        f.ledger = Executor::LedgerFault::kNoImportCoverage;
        break;
    }
    return std::make_unique<CpuExecutor>(m.weights, c, f);
  };
  cpu.kv_type = KvType::kF32;
  return {cpu};
}

}  // namespace ricursive_test

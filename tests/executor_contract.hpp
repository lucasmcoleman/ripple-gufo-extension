#ifndef RICURSIVE_TESTS_EXECUTOR_CONTRACT_HPP_
#define RICURSIVE_TESTS_EXECUTOR_CONTRACT_HPP_

// Backends the executor contract test runs. tests/executor_contract_test.cpp
// holds the test logic and never names a backend; each test binary links one
// file that defines ContractBackends() (CPU: executor_backends_cpu.cpp; a GPU
// binary adds its own file and target, without editing the test).

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "src/executor_iface.hpp"
#include "tests/oracle_common.hpp"

namespace ricursive_test {

/// Deliberate defects the test needs to prove that each check can fail.
enum class Twin {
  kChunkLocalRope,     ///< RoPE positions restart at every Forward
  kRingOneShort,       ///< ring stores one slot fewer than it declares
  kWindowMinus1,       ///< attention window one key short
  kRetentionFromEnd,   ///< retention computed from the logical end
  kFailureKeepsReady,  ///< failed Forward leaves the executor ready
  kNoImportCoverage,   ///< CommitImport accepts incomplete imports
};

struct ContractBackend {
  std::string name;
  /// Builds one session's executor; throws if the config is unsupported.
  std::function<std::unique_ptr<gufo::models::ricursive::Executor>(
      const Model&, const gufo::models::ricursive::ExecutorConfig&)>
      make;
  /// Same with a defect; empty when the backend has no twins (twin checks
  /// then run only on backends that have them).
  std::function<std::unique_ptr<gufo::models::ricursive::Executor>(
      const Model&, const gufo::models::ricursive::ExecutorConfig&, Twin)>
      make_twin;
  gufo::models::ricursive::KvType kv_type{
      gufo::models::ricursive::KvType::kF32};
  /// Allowed difference between two runs of the same tokens when one of them
  /// used a Forward wider than Info().batch_invariant_rows (else 0).
  double rounding_tol{0};
  /// llama.cpp logits to compare with: "<fixture>.seqN<suffix>", limit on the
  /// max abs logit difference (top-1 must agree everywhere).
  std::string llama_suffix{".llama.f32"};
  double llama_max_abs{2e-3};
  /// Allowed difference of taps and exported K/V rows from the Reference.
  double tap_tol{0};
};

[[nodiscard]] std::vector<ContractBackend> ContractBackends();

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_EXECUTOR_CONTRACT_HPP_

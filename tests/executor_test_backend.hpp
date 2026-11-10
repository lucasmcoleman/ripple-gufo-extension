#ifndef RICURSIVE_TESTS_EXECUTOR_TEST_BACKEND_HPP_
#define RICURSIVE_TESTS_EXECUTOR_TEST_BACKEND_HPP_

// The backend tests/executor_test.cpp runs: defined by exactly one of
// tests/executor_backends_host.cpp (HostOpsExecutor) and
// tests/executor_backends_gpu.cpp (GpuExecutor), like ContractBackends() for
// the contract test.

#include <functional>
#include <memory>
#include <string>

#include "hip/executor.hpp"
#include "tests/oracle_common.hpp"

namespace ricursive_test {

struct ExecBackend {
  std::string name;  ///< "hostops" or "gpu"
  bool gpu{false};
  /// Builds an executor on `model` (kept alive by the caller). `lenient`: the
  /// host ops skip the ring-capacity guards (host only; the GPU backend
  /// returns nullptr for it).
  std::function<std::unique_ptr<gufo::models::ricursive::Executor>(
      const Model&, const gufo::models::ricursive::ExecutorConfig&,
      const gufo::models::ricursive::ExecFaults&, bool lenient)>
      make;
  /// Tap comparison against the Reference: max abs difference relative to the
  /// largest |value| of that tap (F32 KV, exact projections).
  double tap_rel_tol{0};
  /// Allowed difference between runs that must agree (chunked vs one-shot,
  /// decode vs prefill, rewind vs fresh) in exact mode; 0 = bitwise.
  double exact_equal_tol{0};
  /// The same in production mode (F16 KV, non-exact projections).
  double production_equal_tol{0};
  /// Quantized files, exact mode: max abs logit difference to the Reference
  /// run on the same file.
  double quant_exact_max_abs{0};
  /// Returns false when the backend cannot run here (no GPU device).
  std::function<bool()> available{[] { return true; }};
};

[[nodiscard]] ExecBackend TestBackend();

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_EXECUTOR_TEST_BACKEND_HPP_

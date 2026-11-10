// The GPU backend (hip/ops_gpu.cpp, hip/device_model.cpp) for the executor
// contract test and for tests/executor_test.cpp. Needs a device; without one
// the binaries exit 77. Run only through the GPU admission lock
// (tests/run-gpu-gates.sh). Twins are not registered here: they prove the
// harness and run on the host ops (docs/S2-EXECUTOR.md).
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "hip/device_model.hpp"
#include "hip/executor.hpp"
#include "tests/executor_contract.hpp"
#include "tests/executor_test_backend.hpp"

namespace ricursive_test {

using namespace gufo::models::ricursive;

namespace {

bool HaveDevice() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

/// One upload per opened fixture, shared by every executor built on it. The
/// copy lives in the Model: it holds host pointers into that model's mapping
/// (the embedding table), so it must die with it. A process-wide map keyed by
/// the reader's address handed tinyA's upload to tinyB when the allocator
/// reused the address (first GPU run of the contract suite, 2026-11-10:
/// segfault in EmbedRowsHost). Fixtures are locals of main, so the hipFree
/// runs while the HIP runtime is alive, never during static destruction.
std::shared_ptr<const DeviceModel> ModelFor(const Model& m) {
  if (!m.backend_state) {
    std::string error;
    std::shared_ptr<const DeviceModel> d =
        DeviceModel::Upload(m.weights, *m.reader, &error);
    if (!d) throw std::runtime_error("device model upload: " + error);
    m.backend_state = std::move(d);
  }
  return std::static_pointer_cast<const DeviceModel>(m.backend_state);
}

std::unique_ptr<Executor> MakeGpu(const Model& m, ExecutorConfig c, const ExecFaults& f,
                                  bool force_exact) {
  if (force_exact) {
    c.kv_type = KvType::kF32;
    c.exact = true;
  }
  // Refuse a model the executor cannot run before uploading anything.
  (void)NormalizeConfig(m.weights.config, c);
  return std::make_unique<GpuExecutor>(ModelFor(m), c, f);
}

}  // namespace

std::vector<ContractBackend> ContractBackends() {
  if (!HaveDevice()) {
    std::printf("SKIP: no GPU device\n");
    std::exit(77);  // the contract test's main has no skip path
  }
  // Production: F16 KV, activation rounding allowed above 16 rows. The tiny
  // fixtures have F32 weights (per-row GEMV, nothing rounds), so 0 is
  // expected; the first GPU run records the measured value.
  ContractBackend prod;
  prod.name = "gpu-f16";
  prod.make = [](const Model& m, const ExecutorConfig& c) {
    ExecutorConfig x = c;
    x.kv_type = KvType::kF16;
    x.exact = false;
    return MakeGpu(m, x, {}, false);
  };
  prod.kv_type = KvType::kF16;
  prod.rounding_tol = 0;
  prod.llama_suffix = ".llama-kvf16.f32";
  prod.llama_max_abs = 2e-2;
  prod.tap_tol = 5e-2;  // host-ops F16 measurement, docs/S2-EXECUTOR.md
  // Exact test mode: F32 KV, FP32 activations at every width.
  ContractBackend exact;
  exact.name = "gpu-f32-exact";
  exact.make = [](const Model& m, const ExecutorConfig& c) {
    return MakeGpu(m, c, {}, true);
  };
  exact.kv_type = KvType::kF32;
  exact.rounding_tol = 0;
  exact.llama_suffix = ".llama.f32";
  exact.llama_max_abs = 2e-3;
  exact.tap_tol = 2e-4;
  // The contract test's main() takes no filter; the gate script runs one
  // backend per invocation through this variable.
  const char* only = std::getenv("RICURSIVE_CONTRACT_BACKEND");
  std::vector<ContractBackend> all = {exact, prod};
  if (only != nullptr && *only != '\0') {
    std::vector<ContractBackend> picked;
    for (auto& b : all)
      if (b.name == only) picked.push_back(std::move(b));
    if (picked.empty()) {
      std::printf("FAIL: RICURSIVE_CONTRACT_BACKEND=%s matches no backend (gpu-f32-exact, gpu-f16)\n", only);
      std::exit(1);
    }
    return picked;
  }
  return all;
}

ExecBackend TestBackend() {
  ExecBackend b;
  b.name = "gpu";
  b.gpu = true;
  b.available = HaveDevice;
  b.make = [](const Model& m, const ExecutorConfig& c, const ExecFaults& f,
              bool lenient) -> std::unique_ptr<Executor> {
    if (lenient) return nullptr;  // the kernels' guards cannot be switched off
    return MakeGpu(m, c, f, false);
  };
  b.tap_rel_tol = 1e-4;
  b.exact_equal_tol = 0;
  b.production_equal_tol = 0;
  b.quant_exact_max_abs = 2e-3;  // provisional; host ops measured 8.6e-6
  return b;
}

}  // namespace ricursive_test

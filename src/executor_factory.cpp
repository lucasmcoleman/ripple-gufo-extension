#include "executor_factory.hpp"

#include <cstdlib>
#include <exception>
#include <string_view>

#if defined(RICURSIVE_WITH_CPU_BACKEND)
#include "cpu_executor.hpp"
#endif

namespace gufo::models::ricursive {

BackendChoice GpuBackend() {
  return {.backend = Backend::kGpu, .kv_type = KvType::kF16, .exact = false,
          .uses_device = true};
}

BackendChoice CpuBackend() {
  return {.backend = Backend::kCpu, .kv_type = KvType::kF32, .exact = true,
          .uses_device = false};
}

std::string BackendHelp() {
  return "ricursive backend: the GPU executor is the production backend; "
         "RICURSIVE_BACKEND=cpu selects the scalar CPU backend, which is for "
         "tests and tiny models only (it decodes weights on the fly and keeps "
         "float32 caches; a 65B model does not run on it)";
}

std::optional<BackendChoice> BackendFromEnvironment(std::string* error) {
  const char* value = std::getenv("RICURSIVE_BACKEND");
  const std::string_view name = value == nullptr ? "" : value;
  if (name.empty() || name == "gpu") return GpuBackend();
  if (name == "cpu") return CpuBackend();
  if (error != nullptr) {
    *error = "RICURSIVE_BACKEND=" + std::string(name) +
             " is not a backend (gpu or cpu). " + BackendHelp();
  }
  return std::nullopt;
}

std::unique_ptr<Executor> CreateExecutor(const BackendChoice& choice,
                                         const ModelWeights& weights,
                                         const ExecutorConfig& config,
                                         std::string* error) {
  const auto fail = [&](std::string message) -> std::unique_ptr<Executor> {
    if (error != nullptr) *error = std::move(message);
    return nullptr;
  };
  if (choice.backend == Backend::kCpu) {
#if defined(RICURSIVE_WITH_CPU_BACKEND)
    try {
      return std::make_unique<CpuExecutor>(weights, config);
    } catch (const std::exception& e) {
      return fail(std::string("CPU executor: ") + e.what());
    }
#else
    return fail("the CPU backend is not built into this extension "
                "(CMake option RICURSIVE_CPU_BACKEND, default ON with "
                "BUILD_TESTING). " + BackendHelp());
#endif
  }
#if defined(RICURSIVE_HAVE_GPU_EXECUTOR)
  return CreateGpuExecutor(weights, config, error);
#else
  (void)weights;
  (void)config;
  return fail("GPU executor not built into this extension (the GPU backend is "
              "written separately and not merged yet). " + BackendHelp());
#endif
}

}  // namespace gufo::models::ricursive

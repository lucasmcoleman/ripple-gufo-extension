#ifndef RICURSIVE_EXECUTOR_FACTORY_HPP_
#define RICURSIVE_EXECUTOR_FACTORY_HPP_

// Chooses the compute backend behind `Executor` (src/executor_iface.hpp).
//
//  - GPU: the production backend, written separately (hip/executor.*). It is
//    linked in when the build defines RICURSIVE_HAVE_GPU_EXECUTOR and provides
//    CreateGpuExecutor below. Until then the GPU choice reports an error.
//  - CPU: the scalar Reference (src/cpu_executor.*), selected with the
//    environment variable RICURSIVE_BACKEND=cpu. It exists for tests and tiny
//    models: it decodes every weight on the fly, so a 65B model would take
//    minutes per token and its caches are float32. It is built in when
//    RICURSIVE_WITH_CPU_BACKEND is defined (CMake option RICURSIVE_CPU_BACKEND).

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "executor_iface.hpp"
#include "weights.hpp"

namespace gufo::models::ricursive {

enum class Backend : std::uint8_t { kGpu, kCpu };

struct BackendChoice {
  Backend backend{Backend::kGpu};
  /// Cache element type this backend stores (GPU: F16, CPU: F32).
  KvType kv_type{KvType::kF16};
  /// No activation rounding at any width (CPU; the GPU in its test mode only).
  bool exact{false};
  /// Whether the backend touches a GPU (device probe, device memory).
  bool uses_device{true};
};

[[nodiscard]] BackendChoice GpuBackend();
[[nodiscard]] BackendChoice CpuBackend();

/// Reads RICURSIVE_BACKEND: unset or "gpu" = GPU, "cpu" = CPU. Anything else
/// sets *error and returns nullopt.
[[nodiscard]] std::optional<BackendChoice> BackendFromEnvironment(
    std::string* error);

/// One executor (one session's cache) for `weights`. The GPU backend shares
/// the uploaded weights between the executors it returns for the same
/// `weights` object. Returns nullptr and sets *error when the backend is not
/// built in or refuses the configuration.
[[nodiscard]] std::unique_ptr<Executor> CreateExecutor(
    const BackendChoice& choice, const ModelWeights& weights,
    const ExecutorConfig& config, std::string* error);

/// What a user needs to know about the backend choice, for errors and --help.
[[nodiscard]] std::string BackendHelp();

#if defined(RICURSIVE_HAVE_GPU_EXECUTOR)
/// PROVIDED BY THE GPU EXECUTOR (hip/executor.cpp). Expected behaviour:
///  - config.kv_type is kF16 (or kF32 with config.exact, the test mode);
///  - callable many times for one `weights` (one call per session): the first
///    call uploads the weights, later calls reuse them, and they are released
///    when the last returned executor and the `weights` object are gone;
///  - returns nullptr with *error set (never throws) when it cannot run
///    (no device, out of memory, unsupported tensor type);
///  - the returned executor honours executor_iface.hpp: synchronous calls,
///    kFailed after a throw, Rewind that returns false rather than lying.
std::unique_ptr<Executor> CreateGpuExecutor(const ModelWeights& weights,
                                            const ExecutorConfig& config,
                                            std::string* error);
#endif

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_EXECUTOR_FACTORY_HPP_

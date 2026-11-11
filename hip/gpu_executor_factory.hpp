#ifndef RICURSIVE_HIP_GPU_EXECUTOR_FACTORY_HPP_
#define RICURSIVE_HIP_GPU_EXECUTOR_FACTORY_HPP_

// Glue between the engine (src/executor_factory.hpp) and the GPU executor
// (hip/executor.hpp, hip/device_model.hpp). HIP builds only: this unit is part
// of gufo_ricursive_gpuexec and gufo_ricursive defines RICURSIVE_HAVE_GPU_EXECUTOR
// when it is linked. docs/INTEGRATION.md lists the mismatches it absorbs.
//
// The factory itself is
//   std::unique_ptr<Executor> CreateGpuExecutor(const ModelWeights&,
//       const ExecutorConfig&, std::string* error);
// declared in src/executor_factory.hpp. What the engine's signature cannot
// carry (the GGUF reader the weights were resolved from, which the device
// upload and the host-side embedding table need) is handed over once per
// model through RegisterGpuWeightsReader.

#include <cstddef>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <string>

#include "hip/device_model.hpp"
#include "src/weights.hpp"

namespace gufo::models::ricursive {

/// Tells the GPU factory which reader `weights` was resolved from. Call once
/// after resolving (Model::Load does); repeating it with the same reader is a
/// no-op, a different reader forgets the uploaded model of that `weights`
/// object. The factory keeps the reader alive as long as any executor built
/// on it exists, and holds no strong reference otherwise.
void RegisterGpuWeightsReader(const ModelWeights& weights,
                              std::shared_ptr<const core::GgufReader> reader);

/// The uploaded copy of `weights` (uploading it when no executor holds it yet), shared with the
/// executors; nullptr and *error when the weights were not registered or the upload fails. The
/// drafter takes the output head from it.
[[nodiscard]] std::shared_ptr<const DeviceModel> AcquireGpuTargetModel(const ModelWeights& weights,
                                                                       std::string* error);

/// Number of uploaded device models that are currently alive (one per model
/// whose weights are resident, however many executors share them).
[[nodiscard]] std::size_t GpuSharedModelCount();

/// Device-wide memory in use (total - free, includes other processes) and the
/// device total, from hipMemGetInfo. False with *error on failure.
[[nodiscard]] bool GpuMemoryUsage(std::size_t* used_bytes,
                                  std::size_t* total_bytes, std::string* error);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_HIP_GPU_EXECUTOR_FACTORY_HPP_

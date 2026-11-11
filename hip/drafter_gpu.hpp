#ifndef RICURSIVE_HIP_DRAFTER_GPU_HPP_
#define RICURSIVE_HIP_DRAFTER_GPU_HPP_

// The GPU drafter engine (HIP builds; part of gufo_ricursive_gpuexec). Docs: docs/DRAFTER-S2.md.

#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <string>

#include "hip/drafter_engine.hpp"
#include "src/weights.hpp"

namespace gufo::models::ricursive {

/// A drafter engine for one session of the target model `target` (registered with
/// RegisterGpuWeightsReader; its device copy, the output head included, is shared with the
/// executors). The drafter's own weights are uploaded once per (`drafter`, `reader`) and shared by
/// every engine made from them. nullptr and *error on failure.
[[nodiscard]] std::unique_ptr<DrafterEngine> MakeGpuDrafterEngine(
    const DrafterWeights& drafter, std::shared_ptr<const core::GgufReader> reader, const ModelWeights& target,
    const DrafterOptions& options, std::string* error, const DrafterExecFaults& faults = {});

/// The same without a target model: only the output head and the token table of the target are
/// used (uploaded here, 0.4 GB for the Q6_K head), for tools that time the drafter alone.
[[nodiscard]] std::unique_ptr<DrafterEngine> MakeGpuDrafterEngineWithHead(
    const DrafterWeights& drafter, std::shared_ptr<const core::GgufReader> reader, const ModelWeights& target,
    std::shared_ptr<const core::GgufReader> target_reader, const DrafterOptions& options, std::string* error,
    const DrafterExecFaults& faults = {});

/// A strong reference to the shared device copy of the drafter's weights (uploaded here if it is not alive).
/// The engines are otherwise its only owners, so it would be freed when the last engine goes and uploaded again
/// for the next one; the runner's setup holds this for as long as it lives (docs/DRAFTER-S4.md, "Follow-ups").
[[nodiscard]] std::shared_ptr<void> PinGpuDrafter(const DrafterWeights& drafter,
                                                  std::shared_ptr<const core::GgufReader> reader, std::string* error);

/// Device bytes the drafter's weights hold once uploaded (0 until the first engine is made).
[[nodiscard]] std::size_t GpuDrafterWeightBytes(const DrafterWeights& drafter);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_HIP_DRAFTER_GPU_HPP_

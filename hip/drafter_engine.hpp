#ifndef RICURSIVE_HIP_DRAFTER_ENGINE_HPP_
#define RICURSIVE_HIP_DRAFTER_ENGINE_HPP_

// DrafterEngine (src/drafter_backend.hpp) over the drafter executor of hip/drafter_executor.hpp:
// the host-ops back end here, the GPU one in hip/drafter_gpu.hpp. Docs: docs/DRAFTER-S2.md.

#include <memory>

#include "hip/drafter_executor.hpp"
#include "src/drafter_backend.hpp"

namespace gufo::models::ricursive {

class OpsDrafterEngine final : public DrafterEngine {
 public:
  explicit OpsDrafterEngine(std::unique_ptr<DrafterExecutor> executor) : exec_(std::move(executor)) {}

  [[nodiscard]] DrafterExecutor& executor() noexcept { return *exec_; }
  [[nodiscard]] const DrafterConfig& config() const noexcept override { return exec_->config(); }
  [[nodiscard]] std::vector<std::uint32_t> TapLayers() const override { return exec_->config().tap_layers; }
  [[nodiscard]] std::uint32_t Base() const noexcept override { return exec_->Base(); }
  [[nodiscard]] std::uint32_t End() const noexcept override { return exec_->End(); }
  void ResetTo(std::uint32_t base) override { exec_->ResetTo(base); }
  void TruncateTo(std::uint32_t position) override { exec_->TruncateTo(position); }
  void Append(std::uint32_t first_position, std::span<const float> features) override;
  using DrafterEngine::Draft;
  /// The whole block (block_size - 1 tokens).
  [[nodiscard]] Block Draft(std::int32_t anchor_token, std::uint32_t anchor_position) override {
    return Draft(anchor_token, anchor_position, exec_->config().block_size - 1);
  }
  /// The first `max_tokens` tokens of the block: the output head and the selector run for those rows only.
  [[nodiscard]] Block Draft(std::int32_t anchor_token, std::uint32_t anchor_position,
                            std::uint32_t max_tokens) override;
  [[nodiscard]] std::uint64_t DeviceBytes() const noexcept override { return exec_->DeviceBytes(); }

 private:
  std::unique_ptr<DrafterExecutor> exec_;
};

/// A drafter engine on the host operations (tests, and the CPU check of the GPU code's structure).
/// `drafter` and `target` must outlive the engine.
[[nodiscard]] std::unique_ptr<OpsDrafterEngine> MakeHostOpsDrafterEngine(
    const DrafterWeights& drafter, const ModelWeights& target, const DrafterOptions& options = {},
    const DrafterExecFaults& faults = {}, const HostOpsOptions& host = {});

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_HIP_DRAFTER_ENGINE_HPP_

#ifndef RICURSIVE_DRAFTER_ENGINE_CPU_HPP_
#define RICURSIVE_DRAFTER_ENGINE_CPU_HPP_

// DrafterEngine over the stage 1 scalar CPU reference (validation and tiny models only;
// part of gufo_ricursive_reference).

#include "drafter_backend.hpp"
#include "drafter_reference.hpp"

namespace gufo::models::ricursive {

class CpuDrafterEngine final : public DrafterEngine {
 public:
  /// `drafter` and `target` must outlive the engine.
  CpuDrafterEngine(const DrafterWeights& drafter, const ModelWeights& target, DrafterFaults faults = {})
      : ref_(drafter, target, faults) {}

  [[nodiscard]] const DrafterConfig& config() const noexcept override { return ref_.config(); }
  [[nodiscard]] std::vector<std::uint32_t> TapLayers() const override { return ref_.TapLayers(); }
  [[nodiscard]] std::uint32_t Base() const noexcept override { return static_cast<std::uint32_t>(ref_.ContextBase()); }
  [[nodiscard]] std::uint32_t End() const noexcept override { return static_cast<std::uint32_t>(ref_.ContextRows()); }
  void ResetTo(std::uint32_t base) override { ref_.ResetTo(base); }
  void TruncateTo(std::uint32_t position) override { ref_.TruncateTo(position); }
  void Append(std::uint32_t first_position, std::span<const float> features) override;
  [[nodiscard]] Block Draft(std::int32_t anchor_token, std::uint32_t anchor_position) override;

 private:
  DrafterReference ref_;
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_DRAFTER_ENGINE_CPU_HPP_

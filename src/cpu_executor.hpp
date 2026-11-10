#ifndef RICURSIVE_CPU_EXECUTOR_HPP_
#define RICURSIVE_CPU_EXECUTOR_HPP_

// Executor backend on the scalar CPU Reference: float32 KV, no activation
// rounding at any width (batch_invariant_rows == max_rows). The test double
// for the engine and the cache adapter, and the in-tree oracle the GPU
// backend is compared with. Built only with tests (gufo_ricursive_reference).

#include <cstdint>
#include <span>
#include <vector>

#include "executor_iface.hpp"
#include "reference.hpp"
#include "weights.hpp"

namespace gufo::models::ricursive {

class CpuExecutor final : public Executor {
public:
  /// Deliberate defects for contract-test twins; all off in normal use.
  struct Faults {
    Variant variant{Variant::kNone};  ///< Reference twin (not kNonCausal)
    bool chunk_local_rope{false};     ///< RoPE positions restart per Forward
    /// The ring stores this many slots fewer than Info().ring_rows declares.
    std::uint32_t ring_shortfall{0};
    LedgerFault ledger{LedgerFault::kNone};
  };

  /// `weights` must outlive the executor. config.kv_type must be kF32;
  /// config.exact is accepted and changes nothing.
  CpuExecutor(const ModelWeights& weights, const ExecutorConfig& config,
              Faults faults);
  CpuExecutor(const ModelWeights& weights, const ExecutorConfig& config)
      : CpuExecutor(weights, config, Faults{}) {}

private:
  void DoForward(std::span<const std::int32_t> tokens,
                 std::uint32_t first_position, LogitRows which,
                 std::span<float> logits, const Tap& tap) override;
  void DoExport(std::uint32_t layer, KvPart part, RowRange rows,
                std::span<std::byte> out) const override;
  void DoImport(std::uint32_t layer, KvPart part, RowRange rows,
                std::span<const std::byte> bytes) override;
  [[nodiscard]] float* Row(std::uint32_t layer, KvPart part,
                           std::uint32_t position);
  [[nodiscard]] const float* Row(std::uint32_t layer, KvPart part,
                                 std::uint32_t position) const;

  Reference ref_;
  bool chunk_local_rope_;
  std::uint32_t slots_;  ///< sliding-layer ring slots actually stored
  std::vector<std::vector<float>> k_, v_;  ///< per layer [slot][kv_width]
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_CPU_EXECUTOR_HPP_

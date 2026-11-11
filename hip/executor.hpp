#ifndef RICURSIVE_HIP_EXECUTOR_HPP_
#define RICURSIVE_HIP_EXECUTOR_HPP_

// The `ricursive` forward pass, written once against the operations of
// hip/ops.hpp and instantiated twice: GpuExecutor (ops_gpu.cpp) and
// HostOpsExecutor (ops_host.cpp, the CPU test backend that exercises the same
// orchestration). Contract: src/executor_iface.hpp. Docs: docs/S2-EXECUTOR.md.
// This header needs no HIP header.

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "hip/ops.hpp"
#include "hip/profile.hpp"
#include "src/executor_iface.hpp"
#include "src/weights.hpp"

namespace gufo::models::ricursive {

/// Context cap used when an ExecutorConfig leaves max_context at 0
/// (SPEC.md: "default 32768"); never the GGUF's nominal 2,097,152.
inline constexpr std::uint32_t kDefaultContextCap = 32768;

/// Deliberate orchestration defects for tests (all off in normal use). Each
/// must make the executor's output differ from the correct one.
struct ExecFaults {
  bool x0_before_scale{false};     ///< value-residual seed taken before the embedding scale
  bool value_mix_into_qk{false};   ///< Q/K projected from the value mix, V from the normed input
  bool gate_from_residual{false};  ///< attention gate projected from the residual, not the normed input
  bool rope_on_full{false};        ///< RoPE requested on full-attention layers
  std::uint32_t ring_shortfall{0};  ///< ring capacity handed to the ops is this many rows short
  bool swap_post_norms{false};     ///< attention and FFN post-norm weights swapped
  bool wrong_logit_row{false};     ///< kLast logits taken from the row before the last
  bool stale_scratch{false};       ///< the value-mix buffer is computed in layer 0 only
  bool chunk_local_rope{false};    ///< K/V append and RoPE positions restart at 0 in every Forward
  std::uint32_t window_shortfall{0};  ///< sliding window handed to attention is this many keys short
  Executor::LedgerFault ledger{Executor::LedgerFault::kNone};
};

/// Fills defaults and validates what the orchestration (and the kernels it
/// calls) cannot do; throws std::invalid_argument naming the problem: head_dim
/// != 128, GQA group not 1/2/10, a layer whose swiglu_clamp is 0 or not
/// finite ("not supported": ClampedSwiGlu has no no-clamp mode), max_context 0
/// -> min(kDefaultContextCap, model context), max_logit_rows clamped to
/// max_rows.
[[nodiscard]] ExecutorConfig NormalizeConfig(const Config& model,
                                             ExecutorConfig config);

class OpsExecutor : public Executor {
 public:
  /// `ops` and `keep_alive` (whatever owns `model`) are owned for the
  /// executor's lifetime; every executor builds its own `ops` so streams never
  /// share scratch. `batch_invariant_rows`: see ExecutorInfo.
  OpsExecutor(std::unique_ptr<Ops> ops, const ModelHandles& model,
              std::shared_ptr<const void> keep_alive,
              const ExecutorConfig& config, std::uint32_t batch_invariant_rows,
              const ExecFaults& faults);
  ~OpsExecutor() override;

  /// False once the device reported a sticky error: Reset() cannot recover
  /// then, the process has to drop the executor (open question 4).
  [[nodiscard]] bool DeviceUsable() noexcept { return ops_->Usable(); }
  [[nodiscard]] Ops& ops() noexcept { return *ops_; }

  /// True when RICURSIVE_PROFILE=1 was set when this executor was created.
  [[nodiscard]] bool Profiling() const noexcept { return prof_ != nullptr; }
  /// The per-stage timing report (docs/PERF-PROFILE.md); empty when profiling
  /// is off or no pass ran. Asking for it also stops the destructor from
  /// printing the same passes to stderr again.
  [[nodiscard]] std::string ProfileReport();

 private:
  struct Buffers;
  void DoForward(std::span<const std::int32_t> tokens,
                 std::uint32_t first_position, LogitRows which,
                 std::span<float> logits, const Tap& tap) override;
  void DoExport(std::uint32_t layer, KvPart part, RowRange rows,
                std::span<std::byte> out) const override;
  void DoImport(std::uint32_t layer, KvPart part, RowRange rows,
                std::span<const std::byte> bytes) override;
  void Run(std::span<const std::int32_t> tokens, std::uint32_t p0,
           LogitRows which, std::span<float> logits, const Tap& tap);
  void Layer(std::uint32_t il, std::uint32_t n, std::uint32_t p0,
             const Tap& tap);
  void Emit(const Tap& tap, std::string_view name, int layer, const float* data,
            std::size_t rows, std::size_t width);
  void EnsureTapScratch();
  /// Slot of `position` in layer `layer`'s cache.
  [[nodiscard]] std::uint32_t Slot(std::uint32_t layer,
                                   std::uint32_t position) const noexcept;
  [[nodiscard]] std::size_t CacheRows(std::uint32_t layer) const noexcept;
  [[nodiscard]] void* CacheBase(std::uint32_t layer, KvPart part) const noexcept;

  /// Stage boundary: with profiling on, waits for the device and charges the
  /// time since the last boundary to `stage`; otherwise one boolean test.
  void Lap(std::uint32_t layer, profile::LayerStage stage, std::size_t bytes = 0) {
    if (prof_) LapSlow(profile::Id(Info().sliding[layer], stage), bytes);
  }
  void Lap(profile::PassStage stage, std::size_t bytes = 0) {
    if (prof_) LapSlow(profile::Id(stage), bytes);
  }
  void LapSlow(profile::StageId id, std::size_t bytes);

  std::unique_ptr<Ops> ops_;
  const ModelHandles& model_;
  std::shared_ptr<const void> keep_alive_;
  ExecFaults faults_;
  bool exact_;
  std::unique_ptr<profile::StageProfile> prof_;  ///< null unless RICURSIVE_PROFILE=1
  std::uint64_t profile_reported_{0};            ///< passes already in a ProfileReport()
  std::unique_ptr<Buffers> buf_;  ///< destroyed before ops_ (declared after)
};

// ---- the two instantiations ------------------------------------------------

class DeviceModel;  // hip/device_model.hpp

/// Production backend: the orchestration on the GPU (ops_gpu.cpp). Shares the
/// uploaded weights; owns its own stream, scratch activations and caches.
/// KV type from config.kv_type (F16 production, F32 exact test mode);
/// batch_invariant_rows = 16 (kSplitRows) unless config.exact.
class GpuExecutor final : public OpsExecutor {
 public:
  GpuExecutor(std::shared_ptr<const DeviceModel> model,
              const ExecutorConfig& config, const ExecFaults& faults);
  GpuExecutor(std::shared_ptr<const DeviceModel> model,
              const ExecutorConfig& config)
      : GpuExecutor(std::move(model), config, ExecFaults{}) {}
};

/// CPU test backend: the orchestration on host memory (ops_host.cpp). No GPU,
/// no HIP header; weights are read from the mapped GGUF.
class HostOpsExecutor final : public OpsExecutor {
 public:
  struct Options {
    ExecFaults faults{};
    /// Test only: the host ops skip the kernels' ring-capacity guards, so an
    /// undersized ring corrupts results instead of being refused.
    bool lenient_capacity{false};
    /// New buffers are filled with NaN patterns (so a read of something never
    /// written shows); false fills zeros, which turns such a bug into finite
    /// wrong numbers for twin checks that cannot compare NaN.
    bool poison_memory{true};
  };
  HostOpsExecutor(const ModelWeights& weights, const ExecutorConfig& config,
                  Options options);
  HostOpsExecutor(const ModelWeights& weights, const ExecutorConfig& config)
      : HostOpsExecutor(weights, config, Options{}) {}

 private:
  HostOpsExecutor(std::shared_ptr<const ModelHandles> model,
                  const ExecutorConfig& config, Options options);
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_HIP_EXECUTOR_HPP_

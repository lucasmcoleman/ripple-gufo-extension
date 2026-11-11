#ifndef RICURSIVE_HIP_PROFILE_HPP_
#define RICURSIVE_HIP_PROFILE_HPP_

// Per-stage timing for the executor's forward pass (docs/PERF-PROFILE.md).
// Pure host code (no HIP header): the executor (hip/executor.cpp) waits for the
// device at each stage boundary and calls Mark; this class only does the
// bookkeeping and the text report, so it is testable with a fake clock.
// Off by default: the executor creates one only when RICURSIVE_PROFILE=1.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>

namespace gufo::models::ricursive::profile {

/// Steps inside one transformer layer (hip/executor.cpp, OpsExecutor::Layer).
enum class LayerStage : std::uint8_t {
  kAttnNorm,
  kValueMix,
  kProjQ,
  kProjK,
  kProjGate,
  kProjV,
  kQkNormRopeKv,  ///< fused Q/K norm + RoPE + K/V append (the KV write and ring slot)
  kAttention,
  kSigmoidGate,
  kProjAttnOut,
  kPostAttnResidual,
  kFfnNorm,
  kProjFfnGate,
  kProjFfnUp,
  kSwiGlu,  ///< clamped SwiGLU activation
  kProjFfnDown,
  kPostFfnResidual,
};
inline constexpr std::size_t kLayerStageCount = 17;

/// Steps outside the layer loop (OpsExecutor::Run).
enum class PassStage : std::uint8_t {
  kEmbed,           ///< host table lookup and upload
  kEmbedScale,      ///< embedding scale and the x0 copy
  kRowGather,       ///< last-row index upload and gather (LogitRows::kLast)
  kFinalNorm,
  kOutputProj,      ///< output projection to logits
  kLogitsReadback,  ///< logits copied to the host
};
inline constexpr std::size_t kPassStageCount = 6;

/// Flat stage index: sliding layers, then global layers, then pass stages.
inline constexpr std::size_t kStageCount = 2 * kLayerStageCount + kPassStageCount;
using StageId = std::uint32_t;

[[nodiscard]] constexpr StageId Id(bool sliding, LayerStage s) noexcept {
  return static_cast<StageId>((sliding ? 0 : kLayerStageCount) + static_cast<std::size_t>(s));
}
[[nodiscard]] constexpr StageId Id(PassStage s) noexcept {
  return static_cast<StageId>(2 * kLayerStageCount + static_cast<std::size_t>(s));
}
/// "sw.attention", "gl.proj_q", "output_proj", ...
[[nodiscard]] std::string StageName(StageId id);
/// Stages that read a weight matrix (the report adds bytes and GB/s to them).
[[nodiscard]] bool IsMatmul(StageId id) noexcept;

/// Passes of more than one row are prompt passes; 1-row passes are generation,
/// split by the context length (tokens including the one being processed).
enum class Bucket : std::uint8_t { kPrompt, kGenLt1024, kGen1024, kGen2048, kGen4096 };
inline constexpr std::size_t kBucketCount = 5;
[[nodiscard]] Bucket BucketFor(std::uint32_t rows, std::uint32_t context_len) noexcept;
[[nodiscard]] const char* BucketTag(Bucket b) noexcept;

using ClockFn = std::function<std::int64_t()>;  ///< nanoseconds, monotonic
[[nodiscard]] std::int64_t SteadyNowNs();

struct BucketStats {
  std::uint64_t passes{0};
  std::uint64_t rows{0};
  std::uint64_t syncs{0};  ///< Mark calls (one device wait each)
  std::int64_t total_ns{0};
  std::array<std::int64_t, kStageCount> ns{};
  std::array<std::uint64_t, kStageCount> bytes{};
};

class StageProfile {
 public:
  explicit StageProfile(ClockFn clock = SteadyNowNs) : clock_(std::move(clock)) {}

  /// Starts a pass; the stage clock starts now. Anything unfinished from an
  /// earlier, aborted pass is dropped.
  void BeginPass(std::uint32_t rows, std::uint32_t context_len);
  /// Adds the time since the previous Mark (or BeginPass) to `id`; `weight_bytes`
  /// is what the stage read from weights.
  void Mark(StageId id, std::uint64_t weight_bytes = 0);
  /// Closes the pass: its stages and total join the bucket's accumulators.
  void EndPass();

  [[nodiscard]] const BucketStats& Stats(Bucket b) const noexcept {
    return buckets_[static_cast<std::size_t>(b)];
  }
  [[nodiscard]] std::uint64_t TotalPasses() const noexcept;
  /// One table per bucket that saw a pass; empty when none did.
  [[nodiscard]] std::string Report() const;

 private:
  ClockFn clock_;
  std::array<BucketStats, kBucketCount> buckets_{};
  // The pass in flight (merged into its bucket by EndPass).
  BucketStats cur_{};
  Bucket cur_bucket_{Bucket::kPrompt};
  std::uint32_t cur_rows_{0};
  std::int64_t pass_start_{0};
  std::int64_t last_{0};
  bool open_{false};
};

}  // namespace gufo::models::ricursive::profile

#endif  // RICURSIVE_HIP_PROFILE_HPP_

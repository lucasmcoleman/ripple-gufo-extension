#ifndef RICURSIVE_HIP_DRAFTER_EXECUTOR_HPP_
#define RICURSIVE_HIP_DRAFTER_EXECUTOR_HPP_

// The DFlash2 drafter (docs/DRAFTER-S1.md, the recipe; src/drafter_reference.* the ground truth)
// written once against the operations of hip/ops.hpp, like hip/executor.hpp is for the target: run
// on the host ops (the CPU test backend) or on the GPU. One executor holds one session's drafter
// state: the per-layer ring of context K/V, bounded by the context window, plus its own
// activations, stream and scratch. Needs no HIP header. Docs: docs/DRAFTER-S2.md.
//
// The drafter does not have to equal the reference bit for bit (it only changes how many drafts
// the verifier accepts, never what is emitted); it is deterministic.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "hip/drafter_handles.hpp"
#include "hip/ops.hpp"
#include "src/drafter_weights.hpp"
#include "src/weights.hpp"

namespace gufo::models::ricursive {

/// Which groups of the drafter's projections run on the 8-bit activation route (`Ops::ProjectQ8`, docs/DRAFTER-S2.md "Stage 2b")
/// instead of `Ops::Project` (FP32 small-batch kernels up to 16 rows). All false = today's behaviour. `DrafterOptions::exact`
/// overrides every route (FP32 activations everywhere).
struct DrafterRoutes {
  bool block{false};    ///< the block pass's 9 matrices per layer (16 rows): conv_proj, q, k, v, output, ffn conv_proj, gate, up, down
  bool head{false};     ///< the output head and the selector's hidden projection (the d drafted rows)
  bool context{false};  ///< the context pass: fc and the K and V projections of the appended rows (any row count)
  [[nodiscard]] bool Any() const noexcept { return block || head || context; }
  friend bool operator==(const DrafterRoutes&, const DrafterRoutes&) = default;
};
/// "block,head,context" in any order and any subset; "all"; "none" or "" (nothing). Throws std::invalid_argument otherwise.
[[nodiscard]] DrafterRoutes ParseDrafterRoutes(std::string_view list);
/// ParseDrafterRoutes of the environment variable RICURSIVE_DRAFTER_Q8 (unset: nothing).
[[nodiscard]] DrafterRoutes DrafterRoutesFromEnvironment();
[[nodiscard]] std::string ToString(const DrafterRoutes& routes);

struct DrafterOptions {
  KvType kv_type{KvType::kF16};          ///< context K/V storage (F32: exact comparison with the reference)
  std::uint32_t max_append_rows{512};    ///< context rows per pass (activations scale with it)
  /// Rows a ring holds beyond the window it serves: a truncation that drops ring_slack rows or more
  /// below the highest row ever held resets the context there (fewer rows keep it, and a draft
  /// anchored one row below the new end still sees its whole window).
  std::uint32_t ring_slack{1024};
  /// Rows kept when the configuration's context window is 0 (all rows).
  std::uint32_t max_context{32768};
  /// FP32 activations in every projection whatever the row count (Ops::Project `exact`; the GPU's exact
  /// test mode, where the context passes of more than 16 rows would otherwise round activations to int8
  /// blocks). The host operations ignore it.
  bool exact{false};
  DrafterRoutes q8;  ///< groups of projections on the 8-bit activation route (default: none)
};

/// Test-only deliberate defects (each a one-line change that must make the checks fail).
struct DrafterExecFaults {
  bool window_off_by_one{false};      ///< the sliding layers' window is one row wider
  bool rope_on_nope_layer{false};     ///< rotary positions on the layer that has none
  std::uint32_t selector_top_k_limit{0};  ///< the selector keeps this many candidates at most (0 = off)
};

/// Rows of the stage callback: as the reference's StageFn (src/drafter_reference.hpp).
using DrafterStageFn = std::function<void(std::string_view name, int layer, const float* data,
                                          std::size_t rows, std::size_t width)>;

/// Handles into the mapped drafter / target files (the host ops).
[[nodiscard]] std::shared_ptr<DrafterHandles> BuildHostDrafterHandles(const DrafterWeights& drafter);
[[nodiscard]] DrafterTarget BuildHostDrafterTarget(const ModelWeights& target);

/// Device memory one session's executor needs, from the configuration alone (what the runner
/// adds to a session's resource claim): rings, block buffers, selector scratch, the context pass.
[[nodiscard]] std::uint64_t DrafterSessionBytes(const DrafterConfig& config, std::uint32_t vocab_size,
                                                const DrafterOptions& options);

class DrafterExecutor {
 public:
  /// `keep_alive` owns whatever `drafter`, `target` and the weights behind them point into.
  /// Throws std::invalid_argument for a drafter the operations cannot run (head_dim, a codebook
  /// type, a window the rings cannot hold).
  DrafterExecutor(std::unique_ptr<Ops> ops, std::shared_ptr<const DrafterHandles> drafter,
                  const DrafterTarget& target, std::shared_ptr<const void> keep_alive,
                  const DrafterOptions& options, const DrafterExecFaults& faults = {});
  ~DrafterExecutor();
  DrafterExecutor(const DrafterExecutor&) = delete;
  DrafterExecutor& operator=(const DrafterExecutor&) = delete;

  [[nodiscard]] const DrafterConfig& config() const noexcept { return drafter_->config; }
  [[nodiscard]] Ops& ops() noexcept { return *ops_; }
  [[nodiscard]] std::uint64_t DeviceBytes() const noexcept;

  /// The context is the features of the positions [Base(), End()).
  [[nodiscard]] std::uint32_t Base() const noexcept { return base_; }
  [[nodiscard]] std::uint32_t End() const noexcept { return end_; }
  /// Empties the context; the next row appended is position `base`.
  void ResetTo(std::uint32_t base);
  /// Drops the rows at positions >= `position`. A position at or below Base() empties the context there. A position
  /// ring_slack or more rows below the highest row ever held since the last reset (the rings no longer hold a full window
  /// below it) keeps the rows the rings still hold: Base() rises to the oldest row every layer has (the highest row held
  /// minus the smallest ring), so the next drafts have a shorter context rather than none; if that leaves no row below
  /// `position`, the context is empty there (Base() == `position`). SUPERSEDED statement (stage 2): such a truncation
  /// always emptied the context.
  void TruncateTo(std::uint32_t position);
  /// Appends the features of `rows` positions starting at End(): [rows][taps * hidden] float32,
  /// in passes of at most max_append_rows. Stages ctx_h, ctx_k, ctx_v as the reference emits them
  /// (the K and V rows are read back from the ring, so F16 storage shows its rounding).
  void AppendContext(std::span<const float> features, std::size_t rows,
                     const DrafterStageFn& stage = {});

  struct Draft {
    std::vector<std::int32_t> tokens;      ///< the first `draft_tokens` of the block's 15
    std::vector<std::int32_t> candidates;  ///< [draft][top_k], by descending logit
    std::vector<float> scores;             ///< [draft][top_k], in candidate order
    std::uint32_t top_k{0};
    std::uint32_t context_first{0}, context_end{0};  ///< the context rows the full layer read
  };
  /// Drafts the block [anchor_token, MASK x (block_size - 1)] whose first row sits at
  /// `anchor_position` (<= End()); the context is the rows below it (the last context window of
  /// them, never below Base()). The block runs all its rows through the layers; the output head and
  /// the selector run for the first `draft_tokens` (1 .. block_size - 1; 0 returns nothing).
  /// Stages as the reference's DraftStep (emb, per layer attn_norm .. h, final, hproj and logits
  /// with `draft_tokens` rows).
  [[nodiscard]] Draft DraftStep(std::int32_t anchor_token, std::uint32_t anchor_position,
                                std::uint32_t draft_tokens, const DrafterStageFn& stage = {});

  /// True when RICURSIVE_PROFILE=1 was set when this executor was created: every stage of a draft
  /// step and of a context pass is then timed (the device is waited for at each boundary, which
  /// slows the step down; use it to see where the time goes, not for the step's own time).
  [[nodiscard]] bool Profiling() const noexcept { return prof_ != nullptr; }
  /// Mean milliseconds per stage over the draft steps / context passes since the last report
  /// (empty when profiling is off).
  [[nodiscard]] std::string ProfileReport();

 private:
  struct Buffers;
  /// One projection: the 8-bit route when `q8` (and not `exact`), else Ops::Project.
  void Proj(bool q8, const WeightView& w, const float* x, int rows, float* y);
  void Emit(const DrafterStageFn& stage, std::string_view name, int layer, const float* data,
            std::size_t rows, std::size_t width);
  void EmitRing(const DrafterStageFn& stage, std::string_view name, int layer, const void* ring,
                std::uint32_t capacity, std::uint32_t first_position, std::size_t rows);
  [[nodiscard]] bool LayerHasRope(std::size_t layer) const noexcept;
  struct Profile;
  void Lap(const char* stage) {
    if (prof_) LapSlow(stage);
  }
  void LapSlow(const char* stage);
  void LapStart(bool context_pass);

  std::unique_ptr<Ops> ops_;
  std::shared_ptr<const DrafterHandles> drafter_;
  DrafterTarget target_;
  std::shared_ptr<const void> keep_alive_;
  DrafterOptions options_;
  DrafterExecFaults faults_;
  std::uint32_t base_{0}, end_{0};
  std::uint32_t high_{0};  ///< highest End() since the last reset
  std::uint32_t append_rows_{0};
  std::uint32_t window_{0};  ///< rows of context the full layer sees
  std::vector<std::uint32_t> capacity_;  ///< per layer ring rows
  std::unique_ptr<Profile> prof_;  ///< null unless RICURSIVE_PROFILE=1
  std::unique_ptr<Buffers> buf_;  ///< destroyed before ops_
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_HIP_DRAFTER_EXECUTOR_HPP_

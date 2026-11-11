#ifndef RICURSIVE_DRAFTER_REFERENCE_HPP_
#define RICURSIVE_DRAFTER_REFERENCE_HPP_

// Scalar CPU reference of one DFlash2 draft step, in the style of reference.hpp:
// readable, not fast; weights are dequantized row by row through
// quant::DequantizeRow, activations are float32, dot products use 8 float
// partial sums. Written from the recipe in docs/DRAFTER-S1.md and checked stage
// by stage against tools/drafter_oracle.py (variant R).
//
// State: the drafter's context K/V cache, one row per committed position.
// AppendContext adds the target features of new positions; DraftStep drafts the
// block that follows position `anchor_position`. Nothing is shared with the
// engine; with no drafter configured none of this code runs.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "drafter_weights.hpp"
#include "reference.hpp"

namespace gufo::models::ricursive {

/// Deliberate defects for negative controls (tests only), each a one-line change.
struct DrafterFaults {
  bool rope_on_nope_layer{false};   ///< rotary positions also on the no-rotary layer
  bool conv_taps_shifted{false};    ///< convolution tap o reads the row t - o - 1
  bool selector_own_row{false};     ///< predecessor = the candidate itself, not the previous row's choice
  bool tap_layer_plus_one{false};   ///< TapLayers() are one target layer too late
  bool context_one_row_short{false};  ///< DraftStep ignores the context row just below the anchor
};

class DrafterReference {
 public:
  /// Named intermediates, `data` is [rows][width] row-major; `layer` is -1 for
  /// global stages. Context stages (emitted by AppendContext, rows in position
  /// order): ctx_h, ctx_k, ctx_v (K after k-norm and RoPE, [rows][kv_heads*head_dim]).
  /// Draft stages (DraftStep, 16 block rows unless noted): emb, attn_norm,
  /// attn_conv_in, attn_o, h_attn, ffn_norm, mlp_conv_in, mlp_o, h, final,
  /// hproj (15 rows), logits (15 rows).
  using StageFn = TapFn;

  /// `drafter` and `target` must outlive the object. `target` supplies the
  /// token embedding rows and the output head the drafter shares.
  DrafterReference(const DrafterWeights& drafter, const ModelWeights& target,
                   DrafterFaults faults = {});

  [[nodiscard]] const DrafterConfig& config() const noexcept { return w_.config; }

  /// Zero-based target layers whose outputs are the features, in feature order.
  [[nodiscard]] std::vector<std::uint32_t> TapLayers() const;

  /// Drops every context row; the next row appended is position 0.
  void Reset();
  /// Drops every context row; the next row appended is position `base`, and
  /// nothing below it is ever read (a restored cache, a bounded drafter window).
  void ResetTo(std::size_t base);
  /// Drops the context rows at positions >= `position` (<= ContextRows()); a
  /// position below ContextBase() empties the context at that base.
  void TruncateTo(std::size_t position);
  /// Positions held: rows ContextBase() .. ContextRows()-1.
  [[nodiscard]] std::size_t ContextRows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t ContextBase() const noexcept { return base_; }
  /// Appends `rows` positions' features, at position ContextRows() onward, [rows][taps*hidden] float32 (the five
  /// tapped layer outputs of one position, concatenated in TapLayers() order).
  void AppendContext(std::span<const float> features, std::size_t rows,
                     const StageFn& stage = {});

  struct Draft {
    std::vector<std::int32_t> tokens;      ///< block_size - 1 drafted tokens
    std::vector<float> confidence;         ///< softmax of the chosen candidate's score
    std::vector<std::int32_t> candidates;  ///< [draft][top_k], sorted by token id
    std::vector<float> scores;             ///< [draft][top_k], in candidate order
    /// The context rows this draft read: positions [context_first, context_end).
    std::size_t context_first{0};
    std::size_t context_end{0};
  };

  /// Drafts the block [bonus_token, MASK x (block_size - 1)] whose first row sits
  /// at position `anchor_position`. The context is the rows below it (the last
  /// `context_window` of them); `anchor_position` <= ContextRows().
  [[nodiscard]] Draft DraftStep(std::int32_t bonus_token, std::size_t anchor_position,
                                const StageFn& stage = {}) const;

 private:
  void MatMul(const TensorRef& w, const float* x, std::size_t n, float* y) const;
  void EmbedRows(std::span<const std::int32_t> tokens, std::vector<float>& out) const;
  void Rope(float* rows, std::size_t n, std::size_t first_position, std::size_t heads) const;
  void Conv(const float* x, std::size_t n, const TensorRef& base, std::size_t side,
            const float* dyn, float* out) const;
  [[nodiscard]] bool LayerHasRope(std::size_t layer) const noexcept;

  const DrafterWeights& w_;
  const ModelWeights& target_;
  DrafterFaults faults_;
  std::size_t base_{0};  ///< position of the first row held
  std::size_t rows_{0};  ///< one past the last position held
  std::vector<std::vector<float>> k_, v_;  ///< per layer [row - base_][kv_width]
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_DRAFTER_REFERENCE_HPP_

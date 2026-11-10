#ifndef RICURSIVE_REFERENCE_HPP_
#define RICURSIVE_REFERENCE_HPP_

// Scalar CPU reference forward for `ricursive`. Readable, not fast: weights are
// dequantized row by row through gufo::quant::DequantizeRow, activations and
// storage are float32, dot products use 8 float partial sums in a fixed order
// (so a row's result does not depend on how many rows are processed with it).
// Written from the spec (recon/B-reference-spec.md section 3), with
// oracle/np_ref.py as the numeric cross-check.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "weights.hpp"

namespace gufo::models::ricursive {

/// Deliberately wrong variants for negative controls (tests only). Every one
/// must make the comparison against llama.cpp fail, except kVresOnVEquiv,
/// which is mathematically equal to the spec (attn_v is linear, no bias) and
/// must still agree.
enum class Variant {
  kNone,
  kRopeFull,           ///< RoPE also on full-attention layers
  kWinMinus1,          ///< window - 1
  kWinPlus1,           ///< window + 1
  kVresOnVLayer0,      ///< value residual mixes with layer 0's V output
  kVresOnVEquiv,       ///< lerp on V outputs of the same layer (== spec)
  kVresSeedPrescale,   ///< value-residual seed before the embedding scale
  kSymGateClamp,       ///< FFN gate clamp symmetric
  kDropPostAttnNorm,   ///< post-attention norm dropped
  kDropPostFfnNorm,    ///< post-FFN norm dropped
  kGateFromResid,      ///< attention gate from the un-normed residual
  kSwapKV,             ///< scores use V rows, output mixes K rows
  kNonCausal,          ///< future keys visible (full forward only)
  kGqaModulo,          ///< query head h reads KV head h % kv_heads
};

[[nodiscard]] std::string_view VariantName(Variant v) noexcept;
[[nodiscard]] std::optional<Variant> ParseVariant(std::string_view name);
/// Every variant except kNone, in declaration order.
[[nodiscard]] std::span<const Variant> AllVariants() noexcept;

/// Receives a named intermediate: `data` is [rows][width] row-major. `layer`
/// is -1 for global taps. Names (as in np_ref.py): embd_scaled, attn_norm,
/// vres_in, Qcur_normed, Kcur_normed, Qcur, Kcur, Vcur, attn_out, attn_gate,
/// attn_gated, attn_post_norm, ffn_inp, ffn_norm, ffn_swiglu, ffn_out,
/// ffn_post_norm, l_out, result_norm, result_output. In an incremental chunk
/// `rows` is the chunk length. Q/K/V taps are flattened [heads*head_dim].
using TapFn = std::function<void(std::string_view name, int layer,
                                 const float* data, std::size_t rows,
                                 std::size_t width)>;

class Reference {
 public:
  /// `weights` must outlive the Reference.
  explicit Reference(const ModelWeights& weights, Variant variant = Variant::kNone);

  [[nodiscard]] const ModelWeights& weights() const noexcept { return w_; }
  [[nodiscard]] Variant variant() const noexcept { return variant_; }

  /// Full-sequence forward: logits [tokens.size()][vocab], positions 0..T-1,
  /// full-length K/V arrays and an explicit mask (no ring).
  [[nodiscard]] std::vector<float> ForwardFull(
      std::span<const std::int32_t> tokens, const TapFn& tap = {}) const;

  /// Incremental decoding state: sliding layers keep a ring of the last
  /// `window + max_rewind` rows, full layers keep every row.
  class Session {
   public:
    /// `max_rewind` bounds how far Rewind may go back from the highest
    /// position ever written. `ring_rows_for_test` (tests only, 0 = off)
    /// overrides the ring capacity to show that an undersized ring is caught.
    Session(const Reference& reference, std::size_t max_rewind,
            std::size_t ring_rows_for_test = 0);

    [[nodiscard]] std::size_t position() const noexcept { return pos_; }
    /// Appends a chunk of any size (>= 1); returns logits [n][vocab], or
    /// [1][vocab] for the last token when `last_only`.
    [[nodiscard]] std::vector<float> Append(
        std::span<const std::int32_t> tokens, bool last_only = false,
        const TapFn& tap = {});
    /// Drops every token at position >= `pos`. Throws std::runtime_error if
    /// the ring no longer holds the rows the next token needs.
    void Rewind(std::size_t pos);
    /// Rows resident in layer `il`'s cache (ring capacity for sliding layers).
    [[nodiscard]] std::size_t CacheRows(std::size_t il) const;

   private:
    struct LayerCache {
      bool ring{false};
      std::size_t capacity{0};  ///< ring slots; 0 for full layers
      std::vector<float> k, v;  ///< [slot or position][kv_width]
    };
    const Reference& ref_;
    std::size_t max_rewind_;
    std::size_t pos_{0};
    std::size_t high_water_{0};
    std::vector<LayerCache> cache_;
  };

 private:
  friend class Session;
  void Layer(std::size_t il, std::size_t n, std::size_t p0, std::vector<float>& h,
             const std::vector<float>& x0, std::vector<float>& v_layer0,
             const std::function<void(const float*, const float*, const float*,
                                      float*)>& attend,
             const TapFn& tap) const;
  void MatMul(const TensorRef& w, const float* x, std::size_t n, float* y) const;
  [[nodiscard]] std::vector<float> Embed(std::span<const std::int32_t> tokens,
                                         std::vector<float>* x0) const;
  [[nodiscard]] std::vector<float> Head(const std::vector<float>& h,
                                        std::size_t n, bool last_only,
                                        const TapFn& tap) const;
  /// Window used by sliding layers, after the variant offset.
  [[nodiscard]] std::size_t EffectiveWindow() const noexcept;
  /// Attention of one query row (all heads) over keys/values `key(j)`,`val(j)`
  /// for positions lo..hi inclusive.
  void AttendRow(const float* q, std::size_t lo, std::size_t hi,
                 const std::function<const float*(std::size_t)>& key,
                 const std::function<const float*(std::size_t)>& val,
                 float* out) const;

  const ModelWeights& w_;
  Variant variant_;
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_REFERENCE_HPP_

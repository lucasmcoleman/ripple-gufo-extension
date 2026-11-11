#ifndef RICURSIVE_DRAFTER_CONFIG_HPP_
#define RICURSIVE_DRAFTER_CONFIG_HPP_

// Configuration of a DFlash2 drafter GGUF (general.architecture "dflash"),
// read from its `dflash.*` metadata and validated against the target model.
// docs/DRAFTER-S1.md lists every key.

#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <optional>
#include <string>
#include <vector>

#include "config.hpp"

namespace gufo::models::ricursive {

struct DrafterConfig {
  std::uint32_t block_count{0};          ///< draft layers (5)
  std::uint32_t context_length{0};       ///< nominal (131072)
  std::uint32_t embedding_length{0};     ///< equals the target's (5120)
  std::uint32_t feed_forward_length{0};  ///< 40960
  std::uint32_t head_count{0};           ///< 80
  std::uint32_t head_count_kv{0};        ///< 8
  std::uint32_t head_dim{0};             ///< key_length == value_length (128)
  std::uint32_t block_size{0};           ///< rows per draft block: anchor + MASKs (16)
  std::uint32_t conv_kernel_size{0};     ///< taps of the dynamic convolution (4)
  std::uint32_t conv_group_size{0};      ///< channels sharing a coefficient (16)
  std::uint32_t selector_rank{0};        ///< 256
  std::uint32_t selector_top_k{0};       ///< candidates per row (32)
  std::uint32_t sliding_window{0};       ///< |query - key| < window on sliding layers (4096)
  /// Last `context_window` committed rows every layer sees; 0 = all rows (16384).
  std::uint32_t context_window{0};
  std::uint32_t mask_token_id{0};        ///< 100278
  float rms_eps{0.0F};                   ///< 1e-6
  float rope_freq_base{0.0F};            ///< 10000
  /// Per draft layer: window-limited attention.
  std::vector<bool> sliding;
  /// Per draft layer: RoPE applied. False = the NoPE layer (layer 4).
  std::vector<bool> rope;
  /// `dflash.target_layers` as stored: target LAYER-INPUT numbering, so the
  /// output of target block i is stored as i + 1 ([2, 21, 40, 59, 78]).
  std::vector<std::uint32_t> target_layers_stored;
  /// Zero-based target blocks whose output (`l_out`, before the final norm)
  /// the drafter reads: stored - 1 ([1, 20, 39, 58, 77]).
  std::vector<std::uint32_t> tap_layers;

  [[nodiscard]] std::uint32_t QueryWidth() const noexcept { return head_count * head_dim; }
  [[nodiscard]] std::uint32_t KvWidth() const noexcept { return head_count_kv * head_dim; }
  [[nodiscard]] std::uint32_t GqaGroup() const noexcept { return head_count / head_count_kv; }
  /// Number of convolution coefficient groups: hidden / conv_group_size.
  [[nodiscard]] std::uint32_t ConvGroups() const noexcept {
    return embedding_length / conv_group_size;
  }
  /// Floats in one row of target features: taps x hidden.
  [[nodiscard]] std::size_t FeatureWidth() const noexcept {
    return std::size_t{embedding_length} * tap_layers.size();
  }
};

/// Reads and validates `dflash.*` metadata and checks it against `target`
/// (hidden size, mask token inside the vocabulary, every tap layer inside the
/// target's layers). On failure returns nullopt and `*error` names the key and
/// the values that disagree. The attention must be non-causal (the only form
/// the reference implements); a causal file is refused.
[[nodiscard]] std::optional<DrafterConfig> ParseDrafterConfig(
    const core::GgufReader& reader, const Config& target, std::string* error);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_DRAFTER_CONFIG_HPP_

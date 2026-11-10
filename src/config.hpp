#ifndef RICURSIVE_CONFIG_HPP_
#define RICURSIVE_CONFIG_HPP_

#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <optional>
#include <string>
#include <vector>

namespace gufo::models::ricursive {

/// Architecture parameters of a `ricursive` GGUF (Ripple-65B family). Every
/// value is read from `ricursive.*` metadata and validated; see ParseConfig.
struct Config {
  std::uint32_t block_count{0};        ///< 80
  std::uint32_t context_length{0};     ///< 2097152 (nominal; the runner caps)
  std::uint32_t embedding_length{0};   ///< 5120
  std::uint32_t feed_forward_length{0};  ///< 40960
  std::uint32_t head_count{0};         ///< 80 query heads
  std::uint32_t head_count_kv{0};      ///< 8 KV heads
  std::uint32_t head_dim{0};           ///< key_length == value_length == rope dims
  std::uint32_t vocab_size{0};         ///< 100352
  std::uint32_t sliding_window{0};     ///< 1024 keys including the query
  float rms_eps{0.0F};                 ///< 1e-6
  float rope_freq_base{0.0F};          ///< 10000
  float embedding_scale{0.0F};         ///< 71.5541763305664, runtime multiply
  /// Per layer: true = sliding window (and RoPE), false = full attention (NoPE).
  std::vector<bool> sliding;
  /// Per layer SwiGLU clamp c: gate = min(gate, c), up = clamp(up, -c, c).
  /// 0 disables the clamp for that layer (as in llama.cpp and the HF code).
  std::vector<float> swiglu_clamp;

  [[nodiscard]] std::uint32_t QueryWidth() const noexcept {
    return head_count * head_dim;
  }
  [[nodiscard]] std::uint32_t KvWidth() const noexcept {
    return head_count_kv * head_dim;
  }
  /// Query heads per KV head; query head h reads KV head h / GqaGroup().
  [[nodiscard]] std::uint32_t GqaGroup() const noexcept {
    return head_count / head_count_kv;
  }
  [[nodiscard]] bool IsSliding(std::uint32_t layer) const noexcept {
    return sliding[layer];
  }
  /// RoPE exactly on the sliding layers (rope_pattern is validated equal).
  [[nodiscard]] bool HasRope(std::uint32_t layer) const noexcept {
    return sliding[layer];
  }
};

/// Reads and validates `ricursive.*` metadata. On failure returns nullopt and
/// sets `*error` to a message that names the offending key.
///
/// Required: block_count, context_length, embedding_length,
/// feed_forward_length, attention.{head_count, head_count_kv, key_length,
/// value_length, layer_norm_rms_epsilon, sliding_window}, rope.{freq_base,
/// dimension_count}, vocab_size, embedding_scale, swiglu_clamp (scalar or
/// per-layer array). Optional: attention.sliding_window_pattern (default:
/// every 5th layer full), attention.rope_pattern (default and only legal
/// value: equal to the sliding pattern).
[[nodiscard]] std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                                std::string* error);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_CONFIG_HPP_

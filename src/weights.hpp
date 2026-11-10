#ifndef RICURSIVE_WEIGHTS_HPP_
#define RICURSIVE_WEIGHTS_HPP_

#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "config.hpp"

namespace gufo::models::ricursive {

/// Non-owning view of one GGUF tensor. `cols` (ne[0]) is the contiguous
/// reduction dimension, `rows` (ne[1], 1 for vectors) the output dimension.
/// `data` points into the reader's mapping; resolving a table never reads it.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{1};
  std::string_view name;

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  /// Encoded bytes of one row of `cols` elements.
  [[nodiscard]] std::size_t RowBytes() const noexcept;
};

struct LayerWeights {
  TensorRef attn_norm;         ///< F32 [E]
  TensorRef attn_q;            ///< [E -> H*D]
  TensorRef attn_k;            ///< [E -> KV*D]
  TensorRef attn_v;            ///< [E -> KV*D], input is the value-residual mix
  TensorRef attn_output;       ///< [H*D -> E]
  TensorRef attn_q_norm;       ///< F32 [D], shared by all heads
  TensorRef attn_k_norm;       ///< F32 [D]
  TensorRef attn_gate;         ///< [E -> H*D]
  TensorRef attn_vres_lambda;  ///< F32 [1]; value-residual weight = sigmoid
  TensorRef post_attention_norm;  ///< F32 [E]
  TensorRef ffn_norm;          ///< F32 [E]
  TensorRef ffn_gate;          ///< [E -> FF]
  TensorRef ffn_up;            ///< [E -> FF]
  TensorRef ffn_down;          ///< [FF -> E]
  TensorRef post_ffw_norm;     ///< F32 [E]
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;   ///< [E -> V]
  TensorRef output_norm;  ///< F32 [E]
  TensorRef output;       ///< [E -> V], separate from token_embd
  std::vector<LayerWeights> layers;
};

/// True for the projection/embedding/head storage types this extension loads:
/// F32, BF16, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS.
[[nodiscard]] bool IsSupportedMatrixType(core::GgmlType type) noexcept;

/// Number of tensors a complete file holds: 3 + 15 per layer.
[[nodiscard]] constexpr std::uint64_t ExpectedTensorCount(
    std::uint32_t block_count) noexcept {
  return 3ULL + 15ULL * block_count;
}

/// Finds every tensor by GGUF name, checks shapes against `config` and gates
/// the storage type per role (matrices: IsSupportedMatrixType; norms, q/k
/// norm and attn_vres_lambda: F32 only). The returned table references
/// `reader`, which must outlive it. On failure returns nullopt and `*error`
/// names the tensor and, for a type problem, the type.
[[nodiscard]] std::optional<ModelWeights> ResolveWeights(
    const core::GgufReader& reader, const Config& config, std::string* error);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_WEIGHTS_HPP_

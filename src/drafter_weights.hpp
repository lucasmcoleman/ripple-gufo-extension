#ifndef RICURSIVE_DRAFTER_WEIGHTS_HPP_
#define RICURSIVE_DRAFTER_WEIGHTS_HPP_

// Tensor table of a DFlash2 drafter GGUF: the same non-owning TensorRef views
// the target uses (weights.hpp), resolved and shape-checked against the
// drafter and target configurations. Matrices may be F32, BF16, Q8_0, Q4_K,
// Q5_K, Q6_K or IQ4_XS (the Q8_0 and Q4_K_M files of the model use Q8_0 and
// Q4_K); norms and convolution bases are F32.

#include <optional>
#include <string>
#include <vector>

#include "drafter_config.hpp"
#include "weights.hpp"

namespace gufo::models::ricursive {

struct DrafterLayerWeights {
  TensorRef attn_norm;       ///< F32 [E]
  TensorRef attn_q;          ///< [E -> H*D]
  TensorRef attn_k;          ///< [E -> KV*D]
  TensorRef attn_v;          ///< [E -> KV*D]
  TensorRef attn_output;     ///< [H*D -> E]
  TensorRef attn_q_norm;     ///< F32 [D]
  TensorRef attn_k_norm;     ///< F32 [D]
  TensorRef attn_conv_base;  ///< F32, rows = side * kernel + tap, cols = E
  TensorRef attn_conv_proj;  ///< [E -> 2 * kernel * E/group]; output index = side*K*G + tap*G + group
  TensorRef ffn_norm;        ///< F32 [E]
  TensorRef ffn_gate;        ///< [E -> FF]
  TensorRef ffn_up;          ///< [E -> FF]
  TensorRef ffn_down;        ///< [FF -> E]
  TensorRef ffn_conv_base;   ///< as attn_conv_base
  TensorRef ffn_conv_proj;   ///< as attn_conv_proj
};

struct DrafterWeights {
  DrafterConfig config;
  TensorRef fc;                    ///< [taps*E -> E]
  TensorRef enc_output_norm;       ///< F32 [E], norm of the projected taps
  TensorRef output_norm;           ///< F32 [E], final norm
  TensorRef selector_hidden;       ///< [E -> rank]
  TensorRef selector_predecessor;  ///< [rank] per token: V rows
  TensorRef selector_successor;    ///< [rank] per token: V rows
  std::vector<DrafterLayerWeights> layers;
};

/// Tensors a complete file holds: 6 + 15 per layer.
[[nodiscard]] constexpr std::uint64_t ExpectedDrafterTensorCount(
    std::uint32_t block_count) noexcept {
  return 6ULL + 15ULL * block_count;
}

/// Finds every tensor by GGUF name and checks shapes and storage types against
/// `config` and the target's vocabulary. A file with a tensor the model does not
/// use is refused. The table references `reader`, which must outlive it. On
/// failure returns nullopt and `*error` names the tensor.
[[nodiscard]] std::optional<DrafterWeights> ResolveDrafterWeights(
    const core::GgufReader& reader, const DrafterConfig& config, const Config& target,
    std::string* error);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_DRAFTER_WEIGHTS_HPP_

#ifndef RICURSIVE_HIP_DRAFTER_HANDLES_HPP_
#define RICURSIVE_HIP_DRAFTER_HANDLES_HPP_

// What the drafter's executor (hip/drafter_executor.hpp) reads of the drafter model, in
// backend memory: the host mapping (host ops) or a device copy (hip/device_model.hpp,
// DeviceDrafter). Needs no HIP header.

#include <vector>

#include "hip/ops.hpp"
#include "src/drafter_config.hpp"

namespace gufo::models::ricursive {

struct DrafterLayerHandles {
  WeightView attn_q, attn_k, attn_v, attn_output, attn_conv_proj;
  WeightView ffn_gate, ffn_up, ffn_down, ffn_conv_proj;
  /// F32 vectors: norms [E] and [head_dim], the convolution bases [2 sides][kernel][E].
  const float *attn_norm{nullptr}, *attn_q_norm{nullptr}, *attn_k_norm{nullptr},
      *attn_conv_base{nullptr}, *ffn_norm{nullptr}, *ffn_conv_base{nullptr};
};

struct DrafterHandles {
  DrafterConfig config;
  WeightView fc, selector_hidden, selector_predecessor, selector_successor;
  const float *enc_output_norm{nullptr}, *output_norm{nullptr};
  std::vector<DrafterLayerHandles> layers;
};

/// The part of the target model the drafter shares: the token table (host mapping; the rows it
/// reads are copied to the device) and the output head (backend memory).
struct DrafterTarget {
  EmbeddingTable table;
  WeightView output;
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_HIP_DRAFTER_HANDLES_HPP_

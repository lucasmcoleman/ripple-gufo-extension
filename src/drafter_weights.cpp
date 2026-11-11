#include "drafter_weights.hpp"

#include <gufo/ext/experimental/ggml_dequant.hpp>

namespace gufo::models::ricursive {
namespace {

struct Fail {
  std::string message;
};

enum class Role { kMatrix, kF32 };

std::string Dims(const std::vector<std::uint64_t>& d) {
  std::string s = "[";
  for (std::size_t i = 0; i < d.size(); ++i) s += (i ? "x" : "") + std::to_string(d[i]);
  return s + "]";
}

/// Tensor with GGUF dimensions `dims` (ne[0] first). The view has cols = ne[0] and
/// rows = the product of the remaining dimensions (1 for a vector).
TensorRef Find(const core::GgufReader& r, const std::string& name, Role role,
               std::vector<std::uint64_t> dims) {
  const core::GgufTensorInfo* t = r.FindTensor(name);
  if (t == nullptr) throw Fail{"missing tensor " + name};
  if (t->dimensions != dims) {
    throw Fail{"tensor " + name + " has shape " + Dims(t->dimensions) + ", expected " + Dims(dims)};
  }
  const bool ok = role == Role::kF32 ? t->type == core::GgmlType::kF32
                                     : IsSupportedMatrixType(t->type);
  if (!ok) {
    throw Fail{"tensor " + name + " has unsupported type " + std::string(core::ToString(t->type)) +
               (role == Role::kF32 ? " (this role must be F32)"
                                   : " (supported: F32, BF16, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS)")};
  }
  TensorRef ref;
  ref.data = t->data;
  ref.type = t->type;
  ref.cols = dims[0];
  ref.rows = 1;
  for (std::size_t i = 1; i < dims.size(); ++i) ref.rows *= dims[i];
  ref.name = t->name;
  if (!t->sized || ref.RowBytes() == 0 || t->size_bytes != ref.RowBytes() * ref.rows) {
    throw Fail{"tensor " + name + " (" + std::string(core::ToString(t->type)) + ") has a row of " +
               std::to_string(dims[0]) +
               " elements that its block format cannot hold, or an unexpected size"};
  }
  return ref;
}

}  // namespace

std::optional<DrafterWeights> ResolveDrafterWeights(const core::GgufReader& reader,
                                                    const DrafterConfig& c, const Config& target,
                                                    std::string* error) {
  try {
    DrafterWeights w;
    w.config = c;
    const std::uint64_t E = c.embedding_length, D = c.head_dim, Q = c.QueryWidth(),
                        KV = c.KvWidth(), FF = c.feed_forward_length, V = target.vocab_size,
                        K = c.conv_kernel_size, G = c.ConvGroups(), R = c.selector_rank;
    if (c.selector_top_k > V) {
      throw Fail{"dflash.selector_top_k (" + std::to_string(c.selector_top_k) +
                 ") exceeds the target's vocabulary"};
    }
    if (reader.GetTensorCount() != ExpectedDrafterTensorCount(c.block_count)) {
      throw Fail{"the file has " + std::to_string(reader.GetTensorCount()) +
                 " tensors, the model uses " +
                 std::to_string(ExpectedDrafterTensorCount(c.block_count)) +
                 " (a tensor this drafter does not use, or a missing one)"};
    }
    w.fc = Find(reader, "fc.weight", Role::kMatrix, {c.FeatureWidth(), E});
    w.enc_output_norm = Find(reader, "enc.output_norm.weight", Role::kF32, {E});
    w.output_norm = Find(reader, "output_norm.weight", Role::kF32, {E});
    w.selector_hidden = Find(reader, "selector_hidden.weight", Role::kMatrix, {E, R});
    // GGUF [rank, vocab]: a token's codebook row is `rank` contiguous values.
    w.selector_predecessor = Find(reader, "selector_predecessor.weight", Role::kMatrix, {R, V});
    w.selector_successor = Find(reader, "selector_successor.weight", Role::kMatrix, {R, V});
    w.layers.resize(c.block_count);
    for (std::uint32_t il = 0; il < c.block_count; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      auto& l = w.layers[il];
      auto mat = [&](const char* n, std::uint64_t cols, std::uint64_t rows) {
        return Find(reader, p + n + ".weight", Role::kMatrix, {cols, rows});
      };
      auto vec = [&](const char* n, std::uint64_t len) {
        return Find(reader, p + n + ".weight", Role::kF32, {len});
      };
      l.attn_norm = vec("attn_norm", E);
      l.attn_q = mat("attn_q", E, Q);
      l.attn_k = mat("attn_k", E, KV);
      l.attn_v = mat("attn_v", E, KV);
      l.attn_output = mat("attn_output", Q, E);
      l.attn_q_norm = vec("attn_q_norm", D);
      l.attn_k_norm = vec("attn_k_norm", D);
      l.attn_conv_base = Find(reader, p + "attn_conv_base", Role::kF32, {E, K, 2});
      l.attn_conv_proj = mat("attn_conv_proj", E, 2 * K * G);
      l.ffn_norm = vec("ffn_norm", E);
      l.ffn_gate = mat("ffn_gate", E, FF);
      l.ffn_up = mat("ffn_up", E, FF);
      l.ffn_down = mat("ffn_down", FF, E);
      l.ffn_conv_base = Find(reader, p + "ffn_conv_base", Role::kF32, {E, K, 2});
      l.ffn_conv_proj = mat("ffn_conv_proj", E, 2 * K * G);
    }
    return w;
  } catch (const Fail& f) {
    if (error != nullptr) *error = "dflash drafter: " + f.message;
    return std::nullopt;
  }
}

}  // namespace gufo::models::ricursive

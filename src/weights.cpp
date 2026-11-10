#include "weights.hpp"

#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <sstream>

namespace gufo::models::ricursive {
namespace {

struct Fail {
  std::string message;
};

enum class Role { kMatrix, kF32Vector };

std::string Dims(const std::vector<std::uint64_t>& d) {
  std::string s = "[";
  for (std::size_t i = 0; i < d.size(); ++i) {
    s += (i ? "x" : "") + std::to_string(d[i]);
  }
  return s + "]";
}

/// `cols` x `rows` matrix (rows == 0: a vector with `cols` elements).
TensorRef Find(const core::GgufReader& r, const std::string& name,
               Role role, std::uint64_t cols, std::uint64_t rows) {
  const core::GgufTensorInfo* t = r.FindTensor(name);
  if (t == nullptr) throw Fail{"missing tensor " + name};
  std::vector<std::uint64_t> want{cols};
  if (rows != 0) want.push_back(rows);
  if (t->dimensions != want) {
    throw Fail{"tensor " + name + " has shape " + Dims(t->dimensions) +
               ", expected " + Dims(want)};
  }
  const bool ok = role == Role::kF32Vector ? t->type == core::GgmlType::kF32
                                           : IsSupportedMatrixType(t->type);
  if (!ok) {
    throw Fail{"tensor " + name + " has unsupported type " +
               std::string(core::ToString(t->type)) +
               (role == Role::kF32Vector ? " (this role must be F32)"
                                         : " (supported: F32, BF16, Q8_0, "
                                           "Q4_K, Q5_K, Q6_K, IQ4_XS)")};
  }
  TensorRef ref;
  ref.data = t->data;
  ref.type = t->type;
  ref.cols = cols;
  ref.rows = rows == 0 ? 1 : rows;
  ref.name = t->name;
  if (!t->sized || ref.RowBytes() == 0 ||
      t->size_bytes != ref.RowBytes() * ref.rows) {
    throw Fail{"tensor " + name + " (" + std::string(core::ToString(t->type)) +
               ") has a row of " + std::to_string(cols) +
               " elements that its block format cannot hold, or an unexpected size"};
  }
  return ref;
}

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  return quant::EncodedSizeBytes(type, static_cast<std::size_t>(cols));
}

bool IsSupportedMatrixType(core::GgmlType type) noexcept {
  switch (type) {
    case core::GgmlType::kF32:
    case core::GgmlType::kBF16:
    case core::GgmlType::kQ8_0:
    case core::GgmlType::kQ4_K:
    case core::GgmlType::kQ5_K:
    case core::GgmlType::kQ6_K:
    case core::GgmlType::kIQ4_XS:
      return true;
    default:
      return false;
  }
}

std::optional<ModelWeights> ResolveWeights(const core::GgufReader& reader,
                                           const Config& c,
                                           std::string* error) {
  try {
    ModelWeights w;
    w.config = c;
    const std::uint64_t E = c.embedding_length, D = c.head_dim,
                        Q = c.QueryWidth(), KV = c.KvWidth(),
                        FF = c.feed_forward_length, V = c.vocab_size;
    w.token_embd = Find(reader, "token_embd.weight", Role::kMatrix, E, V);
    w.output_norm = Find(reader, "output_norm.weight", Role::kF32Vector, E, 0);
    w.output = Find(reader, "output.weight", Role::kMatrix, E, V);
    w.layers.resize(c.block_count);
    for (std::uint32_t il = 0; il < c.block_count; ++il) {
      const std::string p = "blk." + std::to_string(il) + ".";
      auto& l = w.layers[il];
      auto mat = [&](const char* n, std::uint64_t cols, std::uint64_t rows) {
        return Find(reader, p + n + ".weight", Role::kMatrix, cols, rows);
      };
      auto vec = [&](const char* n, std::uint64_t len) {
        return Find(reader, p + n + ".weight", Role::kF32Vector, len, 0);
      };
      l.attn_norm = vec("attn_norm", E);
      l.attn_q = mat("attn_q", E, Q);
      l.attn_k = mat("attn_k", E, KV);
      l.attn_v = mat("attn_v", E, KV);
      l.attn_output = mat("attn_output", Q, E);
      l.attn_q_norm = vec("attn_q_norm", D);
      l.attn_k_norm = vec("attn_k_norm", D);
      l.attn_gate = mat("attn_gate", E, Q);
      l.attn_vres_lambda = vec("attn_vres_lambda", 1);
      l.post_attention_norm = vec("post_attention_norm", E);
      l.ffn_norm = vec("ffn_norm", E);
      l.ffn_gate = mat("ffn_gate", E, FF);
      l.ffn_up = mat("ffn_up", E, FF);
      l.ffn_down = mat("ffn_down", FF, E);
      l.post_ffw_norm = vec("post_ffw_norm", E);
    }
    return w;
  } catch (const Fail& f) {
    if (error != nullptr) *error = "ricursive: " + f.message;
    return std::nullopt;
  }
}

}  // namespace gufo::models::ricursive

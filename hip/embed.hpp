// Host-side embedding lookup for ricursive: token_embd is IQ4_XS in the target file and no GPU kernel
// decodes it, so each token's row is dequantized on the host from the memory-mapped GGUF, multiplied by
// the embedding scale, and copied to the device. Work package A of S2 (docs/S2-PROJECT.md).
#ifndef GUFO_MODELS_RICURSIVE_HIP_EMBED_HPP_
#define GUFO_MODELS_RICURSIVE_HIP_EMBED_HPP_

#include <cstddef>
#include <cstdint>
#include <gufo/ext/gguf_reader.hpp>
#include <span>
#include <vector>

namespace gufo::models::ricursive {

/// Non-owning view of the token table in the mapped GGUF: `vocab` rows of `hidden` contiguous elements
/// (GGUF dims [hidden, vocab]). Resolving it reads no pages.
struct EmbeddingTable {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kIQ4_XS};
  std::size_t vocab{0};
  std::size_t hidden{0};
};

/// Encoded bytes of one table row, or 0 for a type DequantizeRow cannot decode (supported here: F32,
/// F16, BF16, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS) or a hidden size that is not block aligned.
[[nodiscard]] std::size_t EmbeddingRowBytes(core::GgmlType type, std::size_t hidden) noexcept;

/// Dequantizes the rows of `tokens` into `out[tokens.size()][hidden]` and multiplies each element by
/// `scale` (one FP32 multiply, as the CPU reference does: raw * embedding_scale). Throws
/// std::runtime_error naming the bad id for an id outside [0, vocab), and for an unsupported type.
/// Touches only the pages of the requested rows.
void EmbedRowsHost(const EmbeddingTable& table, std::span<const std::int32_t> tokens, float scale,
                   float* out);

}  // namespace gufo::models::ricursive

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace gufo::models::ricursive::rocm {

/// Reusable host staging buffer for EmbedTokens.
struct EmbedStaging {
  std::vector<float> host;
};

/// EmbedRowsHost into `staging`, then a copy of [tokens.size()][hidden] FP32 to `device_out` on
/// `stream`. Returns after the copy has been consumed (the stream is synchronized), so `staging` can
/// be reused immediately. Does not read the table beyond the requested rows.
void EmbedTokens(const EmbeddingTable& table, std::span<const std::int32_t> tokens, float scale,
                 float* device_out, hipStream_t stream, EmbedStaging& staging);

}  // namespace gufo::models::ricursive::rocm
#endif

#endif  // GUFO_MODELS_RICURSIVE_HIP_EMBED_HPP_

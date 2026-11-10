#include "hip/embed.hpp"

#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <stdexcept>
#include <string>

namespace gufo::models::ricursive {

namespace {

using core::GgmlType;

[[noreturn]] void Fail(const std::string& message) {
  throw std::runtime_error("ricursive embedding: " + message);
}

}  // namespace

std::size_t EmbeddingRowBytes(GgmlType type, std::size_t hidden) noexcept {
  if (hidden == 0) {
    return 0;
  }
  switch (type) {
    case GgmlType::kF32:
      return hidden * 4;
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return hidden * 2;
    case GgmlType::kQ8_0:
    case GgmlType::kQ4_K:
    case GgmlType::kQ5_K:
    case GgmlType::kQ6_K:
    case GgmlType::kIQ4_XS:
      return quant::QuantizedRowBytes(type, hidden);  // 0 when not block aligned
    default:
      return 0;
  }
}

void EmbedRowsHost(const EmbeddingTable& table, std::span<const std::int32_t> tokens, float scale,
                   float* out) {
  const std::string type_name(core::ToString(table.type));
  const std::size_t row_bytes = EmbeddingRowBytes(table.type, table.hidden);
  if (row_bytes == 0) {
    Fail("unsupported table type " + type_name + " or hidden size " + std::to_string(table.hidden));
  }
  if (table.data == nullptr || table.vocab == 0) {
    Fail("empty table");
  }
  // Validate every id before writing anything.
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    const std::int32_t id = tokens[i];
    if (id < 0 || static_cast<std::size_t>(id) >= table.vocab) {
      Fail("token id " + std::to_string(id) + " at position " + std::to_string(i) +
           " is outside [0, " + std::to_string(table.vocab) + ")");
    }
  }
  const auto* base = static_cast<const std::uint8_t*>(table.data);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    float* dst = out + i * table.hidden;
    const std::uint8_t* row = base + static_cast<std::size_t>(tokens[i]) * row_bytes;
    if (!quant::DequantizeRow(table.type, row, dst, table.hidden)) {
      Fail("DequantizeRow rejected type " + type_name);
    }
    for (std::size_t c = 0; c < table.hidden; ++c) {
      dst[c] *= scale;
    }
  }
}

}  // namespace gufo::models::ricursive

#if defined(ENGINE_ENABLE_HIP)
namespace gufo::models::ricursive::rocm {

void EmbedTokens(const EmbeddingTable& table, std::span<const std::int32_t> tokens, float scale,
                 float* device_out, hipStream_t stream, EmbedStaging& staging) {
  if (tokens.empty()) {
    return;
  }
  staging.host.resize(tokens.size() * table.hidden);
  EmbedRowsHost(table, tokens, scale, staging.host.data());
  const std::size_t bytes = staging.host.size() * sizeof(float);
  hipError_t err = hipMemcpyAsync(device_out, staging.host.data(), bytes, hipMemcpyHostToDevice, stream);
  if (err == hipSuccess) {
    err = hipStreamSynchronize(stream);
  }
  if (err != hipSuccess) {
    throw std::runtime_error(std::string("ricursive embedding: copy to device failed: ") +
                             hipGetErrorString(err));
  }
}

}  // namespace gufo::models::ricursive::rocm
#endif

#ifndef RICURSIVE_ROW_HASH_HPP_
#define RICURSIVE_ROW_HASH_HPP_

// 64-bit FNV-1a over the bytes of a float row: the "hash of a logits row" that
// the stage 0 dump tool records and the capture test compares. It only has to
// tell bit-identical rows from different ones; it is not a security hash.

#include <cstddef>
#include <cstdint>
#include <span>

namespace gufo::models::ricursive {

[[nodiscard]] inline std::uint64_t RowHash(std::span<const float> row) noexcept {
  std::uint64_t h = 14695981039346656037ULL;
  const auto* bytes = reinterpret_cast<const unsigned char*>(row.data());
  for (std::size_t i = 0; i < row.size_bytes(); ++i) {
    h ^= bytes[i];
    h *= 1099511628211ULL;
  }
  return h;
}

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_ROW_HASH_HPP_

// Element access into encoded matrix rows, shared by the drafter's candidate selector on the
// host (hip/ops_host.cpp) and on the device (hip/drafter_kernels.hip.cpp): the selector reads
// single codebook elements of one token row, which no projection kernel does. Everything here
// compiles for the host and the device, so the host test of this file (tests/drafter_ops_test.cpp,
// against quant::DequantizeRow) tests the function the GPU runs. Formats: F32, BF16, Q8_0, Q4_K,
// the storage types of the drafter's codebooks in the real Q8_0 and Q4_K_M files.
// Layouts are ggml's (block_q8_0 34 bytes per 32 values; block_q4_K 144 bytes per 256).
#ifndef RICURSIVE_HIP_DEQUANT_MATH_HPP_
#define RICURSIVE_HIP_DEQUANT_MATH_HPP_

#include <cstdint>
#include <cstddef>

#include "hip/kernels_math.hpp"

namespace gufo::models::ricursive::rocm {

/// The encodings DequantAt knows (a subset of ggml's types; hip/ops.hpp maps core::GgmlType to it).
enum class Enc : std::uint8_t { kNone, kF32, kBF16, kQ8_0, kQ4_K };

/// Encoded bytes of one row of `cols` elements (0: unsupported type or cols not block aligned).
RICURSIVE_HD std::uint64_t DequantRowBytes(Enc type, std::uint64_t cols) {
  switch (type) {
    case Enc::kF32: return cols * 4;
    case Enc::kBF16: return cols * 2;
    case Enc::kQ8_0: return cols % 32 == 0 ? cols / 32 * 34 : 0;
    case Enc::kQ4_K: return cols % 256 == 0 ? cols / 256 * 144 : 0;
    default: return 0;
  }
}

/// IEEE binary16 bits -> float (exact), integer arithmetic only.
RICURSIVE_HD float HalfBitsToFloat(std::uint32_t h) {
  const std::uint32_t sign = (h & 0x8000U) << 16, exp = (h >> 10) & 0x1FU, man = h & 0x3FFU;
  std::uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;
    } else {
      int e = -1;
      std::uint32_t m = man;
      do {
        ++e;
        m <<= 1;
      } while ((m & 0x400U) == 0);
      bits = sign | ((127 - 15 - static_cast<std::uint32_t>(e)) << 23) | ((m & 0x3FFU) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000U | (man << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (man << 13);
  }
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}

RICURSIVE_HD float BitsToFloat(std::uint32_t bits) {
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}

/// ggml get_scale_min_k4: the 6-bit scale and minimum of sub-block `j` (0..7) of a Q4_K block.
RICURSIVE_HD void ScaleMinK4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

/// Element `i` of the encoded row at `row` (DequantRowBytes bytes), as quant::DequantizeRow
/// decodes it (the same float operations, in the same order).
RICURSIVE_HD float DequantAt(Enc type, const void* row, std::uint32_t i) {
  const auto* p = static_cast<const std::uint8_t*>(row);
  switch (type) {
    case Enc::kF32: {
      std::uint32_t bits = static_cast<std::uint32_t>(p[4 * i]) | (static_cast<std::uint32_t>(p[4 * i + 1]) << 8) |
                           (static_cast<std::uint32_t>(p[4 * i + 2]) << 16) |
                           (static_cast<std::uint32_t>(p[4 * i + 3]) << 24);
      return BitsToFloat(bits);
    }
    case Enc::kBF16: {
      const std::uint32_t bits = static_cast<std::uint32_t>(p[2 * i]) | (static_cast<std::uint32_t>(p[2 * i + 1]) << 8);
      return BitsToFloat(bits << 16);
    }
    case Enc::kQ8_0: {
      const std::uint8_t* block = p + std::size_t{i / 32} * 34;
      const float d = HalfBitsToFloat(static_cast<std::uint32_t>(block[0]) | (static_cast<std::uint32_t>(block[1]) << 8));
      return d * static_cast<float>(static_cast<std::int8_t>(block[2 + i % 32]));
    }
    case Enc::kQ4_K: {
      const std::uint8_t* block = p + std::size_t{i / 256} * 144;
      const std::uint32_t within = i % 256, group = within / 64, rest = within % 64, high = rest / 32, l = rest % 32;
      const float d = HalfBitsToFloat(static_cast<std::uint32_t>(block[0]) | (static_cast<std::uint32_t>(block[1]) << 8));
      const float dmin = HalfBitsToFloat(static_cast<std::uint32_t>(block[2]) | (static_cast<std::uint32_t>(block[3]) << 8));
      std::uint8_t sc = 0, m = 0;
      ScaleMinK4(static_cast<int>(2 * group + high), block + 4, &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = dmin * static_cast<float>(m);
      const std::uint8_t q = block[16 + 32 * group + l];
      return d1 * static_cast<float>(high != 0 ? (q >> 4) : (q & 0xF)) - m1;
    }
    default:
      return 0.0F;
  }
}

}  // namespace gufo::models::ricursive::rocm

#endif  // RICURSIVE_HIP_DEQUANT_MATH_HPP_

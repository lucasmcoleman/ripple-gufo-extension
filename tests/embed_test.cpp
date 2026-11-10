// Embedding lookup tests (work package A of S2).
//
//   embed_test            host-only part: needs no GPU and runs outside the GPU lock
//   embed_test gpu        additionally copies rows to the device and back (a few KB; run it through
//                         the lock wrapper)
//
// Host part:
//  1. gufo::quant::DequantizeRow (what EmbedRowsHost calls) against an independent decoder written here
//     from the ggml block definitions (ggml-quants.c dequantize_row_*), bit for bit, on random blocks for
//     IQ4_XS, Q4_K, Q5_K, Q6_K, Q8_0 and on random F32.
//  2. 200 seeded random token ids of the real Ripple-65B AD-Q4_K_M token_embd (IQ4_XS, read through the
//     memory map, only those rows) against the independent decode times the file's embedding_scale.
//  3. Every row of the tiny fixtures against the same decoders.
//  4. Broken twins: unscaled rows and an off-by-one token id must differ from the expectation by a wide
//     margin; out-of-range ids must be rejected without writing.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <gufo/ext/gguf_reader.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "hip/embed.hpp"
#include "tests/project/check.hpp"

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

using gufo::core::GgmlType;
using gufo::models::ricursive::EmbeddingTable;
using gufo::models::ricursive::EmbedRowsHost;
using ricursive_test::Require;

namespace {

// ---- independent decoders (ggml-quants.c, block layouts of ggml-common.h) ---------------------------

/// IEEE binary16 to float, written from the format (no table, no hardware instruction).
float HalfToFloat(std::uint16_t h) {
  const int sign = (h >> 15) & 1;
  const int exp = (h >> 10) & 31;
  const int mant = h & 1023;
  double v;
  if (exp == 0) {
    v = std::ldexp(static_cast<double>(mant), -24);
  } else if (exp == 31) {
    v = mant == 0 ? std::numeric_limits<double>::infinity()
                  : std::numeric_limits<double>::quiet_NaN();
  } else {
    v = std::ldexp(1.0 + mant / 1024.0, exp - 15);
  }
  return static_cast<float>(sign ? -v : v);
}

std::uint16_t Le16(const std::uint8_t* p) { return static_cast<std::uint16_t>(p[0] | (p[1] << 8)); }

constexpr std::int8_t kValuesIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                          1,    13,   25,  38,  53,  69,  89,  113};

void ScaleMinK4(int j, const std::uint8_t* q, std::uint8_t* d, std::uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// block_iq4_xs: d (f16), scales_h (u16), scales_l[4], qs[128] = 136 bytes
void RefIq4Xs(const std::uint8_t* src, float* y, std::size_t k) {
  for (std::size_t i = 0; i < k / 256; ++i) {
    const std::uint8_t* b = src + i * 136;
    const float d = HalfToFloat(Le16(b));
    const unsigned scales_h = Le16(b + 2);
    const std::uint8_t* scales_l = b + 4;
    const std::uint8_t* qs = b + 8;
    for (int ib = 0; ib < 8; ++ib) {
      const int ls = ((scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((scales_h >> 2 * ib) & 3) << 4);
      const float dl = d * (ls - 32);
      for (int j = 0; j < 16; ++j) {
        y[j + 0] = dl * kValuesIq4Nl[qs[j] & 0xf];
        y[j + 16] = dl * kValuesIq4Nl[qs[j] >> 4];
      }
      y += 32;
      qs += 16;
    }
  }
}

// block_q4_K: d, dmin (f16), scales[12], qs[128] = 144 bytes
void RefQ4K(const std::uint8_t* src, float* y, std::size_t k) {
  for (std::size_t i = 0; i < k / 256; ++i) {
    const std::uint8_t* b = src + i * 144;
    const float d = HalfToFloat(Le16(b));
    const float min = HalfToFloat(Le16(b + 2));
    const std::uint8_t* scales = b + 4;
    const std::uint8_t* q = b + 16;
    int is = 0;
    std::uint8_t sc, m;
    for (int j = 0; j < 256; j += 64) {
      ScaleMinK4(is + 0, scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      ScaleMinK4(is + 1, scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

// block_q5_K: d, dmin (f16), scales[12], qh[32], qs[128] = 176 bytes
void RefQ5K(const std::uint8_t* src, float* y, std::size_t k) {
  for (std::size_t i = 0; i < k / 256; ++i) {
    const std::uint8_t* b = src + i * 176;
    const float d = HalfToFloat(Le16(b));
    const float min = HalfToFloat(Le16(b + 2));
    const std::uint8_t* scales = b + 4;
    const std::uint8_t* qh = b + 16;
    const std::uint8_t* ql = b + 48;
    int is = 0;
    std::uint8_t sc, m;
    std::uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
      ScaleMinK4(is + 0, scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      ScaleMinK4(is + 1, scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
  }
}

// block_q6_K: ql[128], qh[64], scales[16] (int8), d (f16) = 210 bytes
void RefQ6K(const std::uint8_t* src, float* y, std::size_t k) {
  for (std::size_t i = 0; i < k / 256; ++i) {
    const std::uint8_t* b = src + i * 210;
    const std::uint8_t* ql = b;
    const std::uint8_t* qh = b + 128;
    const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(b + 192);
    const float d = HalfToFloat(Le16(b + 208));
    for (int n = 0; n < 256; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const std::int8_t q1 = static_cast<std::int8_t>((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const std::int8_t q2 = static_cast<std::int8_t>((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const std::int8_t q3 = static_cast<std::int8_t>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const std::int8_t q4 = static_cast<std::int8_t>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l + 0] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

// block_q8_0: d (f16), qs[32] (int8) = 34 bytes
void RefQ80(const std::uint8_t* src, float* y, std::size_t k) {
  for (std::size_t i = 0; i < k / 32; ++i) {
    const std::uint8_t* b = src + i * 34;
    const float d = HalfToFloat(Le16(b));
    for (int j = 0; j < 32; ++j) {
      y[i * 32 + j] = static_cast<std::int8_t>(b[2 + j]) * d;
    }
  }
}

void RefF32(const std::uint8_t* src, float* y, std::size_t k) { std::memcpy(y, src, k * 4); }

using RefFn = void (*)(const std::uint8_t*, float*, std::size_t);

RefFn RefFor(GgmlType type) {
  switch (type) {
    case GgmlType::kIQ4_XS:
      return RefIq4Xs;
    case GgmlType::kQ4_K:
      return RefQ4K;
    case GgmlType::kQ5_K:
      return RefQ5K;
    case GgmlType::kQ6_K:
      return RefQ6K;
    case GgmlType::kQ8_0:
      return RefQ80;
    case GgmlType::kF32:
      return RefF32;
    default:
      return nullptr;
  }
}

// ---- helpers -----------------------------------------------------------------------------------------

std::size_t RowBytesOf(GgmlType type, std::size_t k) {
  const std::size_t r = gufo::models::ricursive::EmbeddingRowBytes(type, k);
  Require(r != 0, "row bytes for " + std::string(gufo::core::ToString(type)));
  return r;
}

/// max |a-b| / max |b|; infinite if any value is not finite.
double RelDiff(const float* a, const float* b, std::size_t n) {
  double worst = 0.0, scale = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<double>::infinity();
    worst = std::max(worst, std::fabs(double{a[i]} - double{b[i]}));
    scale = std::max(scale, std::fabs(double{b[i]}));
  }
  return scale == 0.0 ? worst : worst / scale;
}

long RssKb() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) return std::atol(line.c_str() + 6);
  }
  return -1;
}

/// Random blocks of `type` with bounded finite fp16 scale fields.
std::vector<std::uint8_t> RandomRows(GgmlType type, std::size_t rows, std::size_t k, std::mt19937_64& rng) {
  const std::size_t row_bytes = RowBytesOf(type, k);
  std::vector<std::uint8_t> data(rows * row_bytes);
  for (auto& b : data) b = static_cast<std::uint8_t>(rng());
  if (type == GgmlType::kF32) {
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    for (std::size_t i = 0; i < rows * k; ++i) {
      const float v = u(rng);
      std::memcpy(&data[i * 4], &v, 4);
    }
    return data;
  }
  std::size_t block_bytes = 0;
  std::vector<std::size_t> half_offsets;
  switch (type) {
    case GgmlType::kIQ4_XS: block_bytes = 136; half_offsets = {0}; break;
    case GgmlType::kQ4_K: block_bytes = 144; half_offsets = {0, 2}; break;
    case GgmlType::kQ5_K: block_bytes = 176; half_offsets = {0, 2}; break;
    case GgmlType::kQ6_K: block_bytes = 210; half_offsets = {208}; break;
    case GgmlType::kQ8_0: block_bytes = 34; half_offsets = {0}; break;
    default: Require(false, "RandomRows type");
  }
  std::uniform_real_distribution<float> scale(0.0005F, 0.004F);
  for (std::size_t b = 0; b < data.size() / block_bytes; ++b) {
    for (std::size_t off : half_offsets) {
      const float s = scale(rng);
      // round to nearest binary16 (normal range only: the scales are in [0.0005, 0.004])
      int e = std::ilogb(s);
      long mant = std::lround((std::ldexp(s, -e) - 1.0) * 1024.0);
      if (mant == 1024) {
        mant = 0;
        ++e;
      }
      const std::uint16_t best = static_cast<std::uint16_t>(((e + 15) << 10) | mant);
      data[b * block_bytes + off] = static_cast<std::uint8_t>(best & 0xFF);
      data[b * block_bytes + off + 1] = static_cast<std::uint8_t>(best >> 8);
    }
  }
  return data;
}

// ---- tests -------------------------------------------------------------------------------------------

void CheckDecodersOnRandomBlocks() {
  std::mt19937_64 rng(77);
  constexpr std::size_t kRows = 24, kHidden = 5120;
  for (GgmlType type : {GgmlType::kIQ4_XS, GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                        GgmlType::kQ8_0, GgmlType::kF32}) {
    const auto data = RandomRows(type, kRows, kHidden, rng);
    const std::size_t row_bytes = RowBytesOf(type, kHidden);
    std::vector<float> got(kHidden), want(kHidden);
    for (std::size_t r = 0; r < kRows; ++r) {
      Require(gufo::quant::DequantizeRow(type, data.data() + r * row_bytes, got.data(), kHidden),
              "DequantizeRow rejected a supported type");
      RefFor(type)(data.data() + r * row_bytes, want.data(), kHidden);
      Require(std::memcmp(got.data(), want.data(), kHidden * 4) == 0,
              std::string("DequantizeRow differs from the independent ") +
                  std::string(gufo::core::ToString(type)) + " decoder on random row " + std::to_string(r));
    }
    std::cout << "random blocks " << gufo::core::ToString(type) << ": " << kRows
              << " rows x " << kHidden << " bit-equal to the independent decoder\n";
  }
}

struct Opened {
  std::unique_ptr<gufo::core::GgufReader> reader;
  EmbeddingTable table;
  float scale{1.0F};
};

Opened OpenEmbedding(const std::string& path) {
  Opened o;
  std::string error;
  o.reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(o.reader != nullptr, "open " + path + ": " + error);
  const auto* t = o.reader->FindTensor("token_embd.weight");
  Require(t != nullptr, path + ": no token_embd.weight");
  Require(t->dimensions.size() == 2, path + ": token_embd is not 2-D");
  o.table.data = t->data;
  o.table.type = t->type;
  o.table.hidden = t->dimensions[0];
  o.table.vocab = t->dimensions[1];
  Require(t->size_bytes == o.table.vocab * RowBytesOf(t->type, o.table.hidden),
          path + ": token_embd size does not match vocab x row bytes");
  const auto scale = o.reader->GetMetadataFloat32("ricursive.embedding_scale");
  Require(scale.has_value(), path + ": no ricursive.embedding_scale");
  o.scale = *scale;
  return o;
}

/// Expected scaled rows from the independent decoder.
std::vector<float> Expected(const EmbeddingTable& t, float scale, const std::vector<std::int32_t>& ids) {
  const std::size_t row_bytes = RowBytesOf(t.type, t.hidden);
  const auto ref = RefFor(t.type);
  Require(ref != nullptr, "no independent decoder for table type");
  std::vector<float> out(ids.size() * t.hidden);
  for (std::size_t i = 0; i < ids.size(); ++i) {
    ref(static_cast<const std::uint8_t*>(t.data) + static_cast<std::size_t>(ids[i]) * row_bytes,
        &out[i * t.hidden], t.hidden);
    for (std::size_t c = 0; c < t.hidden; ++c) out[i * t.hidden + c] *= scale;
  }
  return out;
}

void CheckOutOfRange(const EmbeddingTable& t) {
  for (std::int32_t bad : {-1, static_cast<std::int32_t>(t.vocab), static_cast<std::int32_t>(t.vocab) + 1000,
                           std::numeric_limits<std::int32_t>::max(),
                           std::numeric_limits<std::int32_t>::min()}) {
    std::vector<float> out(2 * t.hidden, 12345.0F);
    const std::int32_t ids[2] = {0, bad};
    bool threw = false;
    try {
      EmbedRowsHost(t, ids, 1.0F, out.data());
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()).find(std::to_string(bad)) != std::string::npos;
    }
    Require(threw, "out-of-range id " + std::to_string(bad) + " not rejected with its value in the message");
    Require(std::all_of(out.begin(), out.end(), [](float v) { return v == 12345.0F; }),
            "output written despite a rejected id " + std::to_string(bad));
  }
  std::cout << "out-of-range ids (-1, vocab, vocab+1000, INT_MAX, INT_MIN) rejected before any write\n";
}

void CheckTwins(const EmbeddingTable& t, float scale, const std::vector<std::int32_t>& ids) {
  const std::size_t n = std::min<std::size_t>(ids.size(), 32);
  const std::vector<std::int32_t> head(ids.begin(), ids.begin() + n);
  const auto want = Expected(t, scale, head);
  std::vector<float> got(n * t.hidden);
  EmbedRowsHost(t, head, scale, got.data());
  const double good = RelDiff(got.data(), want.data(), got.size());
  Require(good == 0.0, "twin baseline: the correct lookup must equal the expectation exactly");
  // Unscaled embedding.
  EmbedRowsHost(t, head, 1.0F, got.data());
  const double unscaled = RelDiff(got.data(), want.data(), got.size());
  // Token id off by one (stay inside the vocabulary).
  std::vector<std::int32_t> shifted = head;
  for (auto& id : shifted) id = id + 1 < static_cast<std::int32_t>(t.vocab) ? id + 1 : id - 1;
  EmbedRowsHost(t, shifted, scale, got.data());
  const double off_by_one = RelDiff(got.data(), want.data(), got.size());
  std::cout << "twins vs the expectation (max|diff|/max|expected| over " << n << " rows): correct "
            << good << ", unscaled " << unscaled << ", token id off by one " << off_by_one << "\n";
  Require(unscaled > 0.5, "unscaled twin is not clearly different");
  Require(off_by_one > 0.5, "off-by-one-id twin is not clearly different");
}

void CheckFile(const std::string& path, const std::vector<std::int32_t>* fixed_ids, std::size_t random_ids,
               std::uint64_t seed, bool twins) {
  const long rss0 = RssKb();
  Opened o = OpenEmbedding(path);
  std::cout << path << ": token_embd " << gufo::core::ToString(o.table.type) << " [" << o.table.hidden
            << " x " << o.table.vocab << "], embedding_scale " << o.scale << "\n";
  std::vector<std::int32_t> ids;
  if (fixed_ids != nullptr) {
    ids = *fixed_ids;
    if (ids.empty()) {  // every row
      ids.resize(o.table.vocab);
      for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<std::int32_t>(i);
    }
  } else {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::int32_t> pick(0, static_cast<std::int32_t>(o.table.vocab) - 1);
    for (std::size_t i = 0; i < random_ids; ++i) ids.push_back(pick(rng));
    // the two edge rows are checked in addition to the seeded ones
    ids.push_back(0);
    ids.push_back(static_cast<std::int32_t>(o.table.vocab) - 1);
  }
  const auto want = Expected(o.table, o.scale, ids);
  std::vector<float> got(ids.size() * o.table.hidden);
  EmbedRowsHost(o.table, ids, o.scale, got.data());
  Require(std::memcmp(got.data(), want.data(), got.size() * 4) == 0,
          path + ": EmbedRowsHost differs from the independent decode (first bad id search below)");
  std::size_t nonzero_rows = 0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    bool any = false;
    for (std::size_t c = 0; c < o.table.hidden; ++c) {
      Require(std::isfinite(got[i * o.table.hidden + c]), "non-finite embedding value");
      any = any || got[i * o.table.hidden + c] != 0.0F;
    }
    nonzero_rows += any;
  }
  Require(nonzero_rows * 10 >= ids.size() * 9, path + ": most rows decode to zeros");
  std::cout << "  " << ids.size() << " rows bit-equal to the independent decode x scale (" << nonzero_rows
            << " non-zero rows); RSS " << rss0 / 1024 << " -> " << RssKb() / 1024 << " MiB\n";
  if (twins) {
    CheckTwins(o.table, o.scale, ids);
    CheckOutOfRange(o.table);
  }
}

#if defined(ENGINE_ENABLE_HIP)
void CheckDevice(const std::string& path) {
  Opened o = OpenEmbedding(path);
  const std::vector<std::int32_t> ids = {0, 1, static_cast<std::int32_t>(o.table.vocab) - 1, 7, 7, 3};
  const auto want = Expected(o.table, o.scale, ids);
  float* dev = nullptr;
  Require(hipMalloc(&dev, want.size() * 4) == hipSuccess, "hipMalloc");
  Require(hipMemset(dev, 0xFF, want.size() * 4) == hipSuccess, "hipMemset");
  gufo::models::ricursive::rocm::EmbedStaging staging;
  gufo::models::ricursive::rocm::EmbedTokens(o.table, ids, o.scale, dev, nullptr, staging);
  std::vector<float> got(want.size());
  Require(hipMemcpy(got.data(), dev, got.size() * 4, hipMemcpyDeviceToHost) == hipSuccess, "copy back");
  (void)hipFree(dev);
  Require(std::memcmp(got.data(), want.data(), got.size() * 4) == 0, "device rows differ from host rows");
  std::cout << "device copy: " << ids.size() << " rows of " << path << " round-trip bit-equal\n";
}
#endif

}  // namespace

int main(int argc, char** argv) {
  const bool gpu = argc > 1 && std::string(argv[1]) == "gpu";
  const char* real_env = std::getenv("RICURSIVE_REAL_GGUF");
  const std::string real = real_env != nullptr
      ? real_env
      : "/server/ai/models/AtomicChat/Ripple-65B-Preview-GGUF/Ripple-65B-Preview-AD-Q4_K_M.gguf";
  const char* fix_env = std::getenv("RICURSIVE_FIXTURES");
  const std::string fixtures = fix_env != nullptr ? fix_env : "/server/staging/ripple-gufo/fixtures";

  int skipped = 0;
  const int rc = ricursive_test::Run([&] {
    CheckDecodersOnRandomBlocks();
    // Tiny fixtures: every row of every file.
    const std::vector<std::int32_t> all;
    int seen = 0;
    for (const char* name : {"tinyA.gguf", "tinyB.gguf", "tinyB-Q4_K_M.gguf", "tinyB-Q5_K_M.gguf",
                             "tinyB-Q6_K.gguf", "tinyB-Q8_0.gguf"}) {
      const std::string path = fixtures + "/" + name;
      if (!std::filesystem::exists(path)) {
        std::cout << "SKIP fixture " << path << " (absent)\n";
        continue;
      }
      CheckFile(path, &all, 0, 0, std::string(name) == "tinyB-Q4_K_M.gguf");
      ++seen;
    }
    if (seen == 0) ++skipped;
    // The real model: 200 seeded random ids (+ the two edge rows).
    if (!std::filesystem::exists(real)) {
      std::cout << "SKIP real model " << real << " (absent)\n";
      ++skipped;
    } else {
      CheckFile(real, nullptr, 200, 20261110, true);
    }
#if defined(ENGINE_ENABLE_HIP)
    if (gpu) {
      int devices = 0;
      if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
        std::cout << "SKIP: no HIP device for the device-copy part\n";
        ++skipped;
      } else {
        const std::string tiny = fixtures + "/tinyB-Q4_K_M.gguf";
        if (std::filesystem::exists(tiny)) CheckDevice(tiny);
        if (std::filesystem::exists(real)) CheckDevice(real);
      }
    }
#else
    (void)gpu;
#endif
  });
  if (rc == 0 && skipped > 0 && !gpu) {
    std::cout << "note: " << skipped << " part(s) skipped\n";
  }
  return rc;
}

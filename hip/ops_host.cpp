// Host implementation of the executor's operations (hip/ops.hpp): ordinary
// memory and scalar loops with the argument conventions of the GPU kernels.
// This is NOT the Reference: it is written against the operator signatures,
// so it exercises the orchestration's buffer sizes, offsets, ring slots, row
// counts, call order and scratch reuse. To make that bite, every buffer is
// its own guarded allocation filled with NaN patterns, and every operator
// checks that each pointer argument lies, for the number of bytes the
// operator will touch, inside one live buffer (the weights, which are the
// mapped GGUF, are exempt). The kernels' own argument guards are replicated.
// Numerics: sums in double, K/V stored as binary16 (round to nearest even) or
// float exactly as the device stores them; projections ignore `exact` (the
// host never rounds activations).
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "hip/executor.hpp"

namespace gufo::models::ricursive {
namespace {

namespace km = rocm;  // hip/kernels_math.hpp (no HIP header needed)

[[noreturn]] void Fail(const char* op, const std::string& what) {
  throw std::runtime_error(std::string("ricursive host ops: ") + op + ": " +
                           what);
}

constexpr std::size_t kGuard = 64;
constexpr unsigned char kGuardByte = 0xA5;

float Sigmoid(float x) { return 1.0F / (1.0F + std::exp(-x)); }

class HostOps final : public Ops {
 public:
  HostOps(bool lenient_capacity, bool poison)
      : lenient_(lenient_capacity), poison_(poison) {}
  ~HostOps() override {
    for (auto& [ptr, block] : blocks_) std::free(block.raw);
  }

  const char* Name() const noexcept override { return "host"; }

  // ---- memory -----------------------------------------------------------
  void* Allocate(std::size_t bytes) override {
    const std::size_t padded = (std::max<std::size_t>(bytes, 1) + 63) & ~std::size_t{63};
    auto* raw = static_cast<unsigned char*>(std::aligned_alloc(64, padded + 2 * kGuard));
    if (raw == nullptr) throw std::runtime_error("ricursive host ops: out of memory");
    std::memset(raw, kGuardByte, kGuard);
    std::memset(raw + kGuard, poison_ ? 0xFF : 0x00, padded);  // 0xFF = NaN as float and binary16
    std::memset(raw + kGuard + padded, kGuardByte, kGuard);
    // The guard after the payload starts at the requested size's end so a
    // one-byte overrun is seen; padding bytes count as payload end.
    unsigned char* payload = raw + kGuard;
    blocks_[payload] = Block{raw, bytes, padded};
    return payload;
  }
  void Free(void* ptr) noexcept override {
    if (ptr == nullptr) return;
    const auto it = blocks_.find(static_cast<unsigned char*>(ptr));
    if (it == blocks_.end()) return;
    if (corruption_.empty()) corruption_ = Corruption(it->first, it->second);
    std::free(it->second.raw);
    blocks_.erase(it);
  }
  void Memset(void* dst, int byte, std::size_t bytes) override {
    Range("Memset", "dst", dst, bytes);
    std::memset(dst, byte, bytes);
  }
  void CopyIn(void* dst, const void* host_src, std::size_t bytes) override {
    Range("CopyIn", "dst", dst, bytes);
    std::memcpy(dst, host_src, bytes);
  }
  void CopyOut(void* host_dst, const void* src, std::size_t bytes) override {
    Range("CopyOut", "src", src, bytes);
    Synchronize();
    std::memcpy(host_dst, src, bytes);
  }
  void CopyInternal(void* dst, const void* src, std::size_t bytes) override {
    Range("CopyInternal", "dst", dst, bytes);
    Range("CopyInternal", "src", src, bytes);
    NoOverlap("CopyInternal", dst, bytes, src, bytes);
    std::memcpy(dst, src, bytes);
  }
  void Synchronize() override {
    std::string bad = corruption_;
    for (const auto& [ptr, block] : blocks_) {
      if (bad.empty()) bad = Corruption(ptr, block);
    }
    if (!bad.empty()) Fail("Synchronize", bad);
  }
  bool Usable() noexcept override { return true; }
  void ReserveProjection(std::size_t, std::size_t) override {}

  // ---- operators --------------------------------------------------------
  void EmbedRows(const EmbeddingTable& table,
                 std::span<const std::int32_t> tokens, float scale,
                 float* out) override {
    Range("EmbedRows", "out", out, tokens.size() * table.hidden * 4);
    // Written here, not through EmbedRowsHost (hip/embed.cpp lives in the HIP
    // target): one DequantizeRow per token row and one fp32 multiply.
    if (table.data == nullptr || table.vocab == 0 || table.hidden == 0)
      Fail("EmbedRows", "empty table");
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      if (tokens[i] < 0 || static_cast<std::size_t>(tokens[i]) >= table.vocab)
        Fail("EmbedRows", "token id " + std::to_string(tokens[i]) + " outside [0, " +
                              std::to_string(table.vocab) + ")");
    }
    using core::GgmlType;
    const std::size_t row_bytes =
        table.type == GgmlType::kF32 ? table.hidden * 4
        : (table.type == GgmlType::kF16 || table.type == GgmlType::kBF16)
            ? table.hidden * 2
            : quant::QuantizedRowBytes(table.type, table.hidden);
    if (row_bytes == 0) Fail("EmbedRows", "unsupported table type or width");
    const auto* base = static_cast<const std::uint8_t*>(table.data);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      float* dst = out + i * table.hidden;
      if (!quant::DequantizeRow(table.type,
                                base + static_cast<std::size_t>(tokens[i]) * row_bytes,
                                dst, table.hidden))
        Fail("EmbedRows", "DequantizeRow rejected the table type");
      for (std::size_t c = 0; c < table.hidden; ++c) dst[c] *= scale;
    }
  }

  void ScaleRows(const float* in, float* out, std::uint32_t rows,
                 std::uint32_t dim, float scale) override {
    const std::size_t n = std::size_t{rows} * dim;
    Range("ScaleRows", "in", in, n * 4);
    Range("ScaleRows", "out", out, n * 4);
    for (std::size_t i = 0; i < n; ++i) out[i] = in[i] * scale;
  }

  void RmsNorm(const float* x, const float* weight, float* out,
               std::uint32_t rows, std::uint32_t dim, float eps) override {
    if (dim == 0) Fail("RmsNorm", "dim must be > 0");
    const std::size_t n = std::size_t{rows} * dim;
    Range("RmsNorm", "x", x, n * 4);
    Range("RmsNorm", "out", out, n * 4);
    Weights("RmsNorm", weight, dim);
    if (out != x) NoOverlap("RmsNorm", out, n * 4, x, n * 4);
    for (std::uint32_t r = 0; r < rows; ++r) {
      const float* xr = x + std::size_t{r} * dim;
      float* yr = out + std::size_t{r} * dim;
      const float scale = InvRms(xr, dim, eps);
      for (std::uint32_t i = 0; i < dim; ++i) yr[i] = (xr[i] * scale) * weight[i];
    }
  }

  void ResidualPostNorm(float* x, const float* branch, const float* weight,
                        std::uint32_t rows, std::uint32_t dim,
                        float eps) override {
    if (dim == 0) Fail("ResidualPostNorm", "dim must be > 0");
    const std::size_t n = std::size_t{rows} * dim;
    Range("ResidualPostNorm", "x", x, n * 4);
    Range("ResidualPostNorm", "branch", branch, n * 4);
    Weights("ResidualPostNorm", weight, dim);
    if (x == branch) Fail("ResidualPostNorm", "x and branch must not alias");
    NoOverlap("ResidualPostNorm", x, n * 4, branch, n * 4);
    for (std::uint32_t r = 0; r < rows; ++r) {
      const float* br = branch + std::size_t{r} * dim;
      float* xr = x + std::size_t{r} * dim;
      const float scale = InvRms(br, dim, eps);
      for (std::uint32_t i = 0; i < dim; ++i) xr[i] += (br[i] * scale) * weight[i];
    }
  }

  void ValueMix(const float* x0, const float* a, const float* lambda,
                float* out, std::uint32_t rows, std::uint32_t dim) override {
    const std::size_t n = std::size_t{rows} * dim;
    Range("ValueMix", "x0", x0, n * 4);
    Range("ValueMix", "a", a, n * 4);
    Range("ValueMix", "out", out, n * 4);
    Weights("ValueMix", lambda, 1);
    const float s = Sigmoid(*lambda);
    for (std::size_t i = 0; i < n; ++i) {
      const float base = x0[i];
      out[i] = base + s * (a[i] - base);
    }
  }

  void Project(const WeightView& w, const float* x, int rows, float* y,
               bool /*exact*/) override {
    using core::GgmlType;
    const std::string type(core::ToString(w.type));
    switch (w.type) {
      case GgmlType::kF32: case GgmlType::kBF16: case GgmlType::kQ8_0:
      case GgmlType::kQ4_K: case GgmlType::kQ5_K: case GgmlType::kQ6_K:
        break;
      default:
        Fail("Project", "no projection path for type " + type);
    }
    if (rows < 1 || w.rows == 0 || w.cols == 0 || w.data == nullptr) {
      Fail("Project", "empty projection");
    }
    const std::size_t row_bytes =
        w.type == GgmlType::kF32 ? std::size_t{w.cols} * 4
        : w.type == GgmlType::kBF16 ? std::size_t{w.cols} * 2
                                    : quant::QuantizedRowBytes(w.type, w.cols);
    if (row_bytes == 0) Fail("Project", "cols not block aligned for " + type);
    if (w.bytes != row_bytes * w.rows) {
      Fail("Project", "byte size " + std::to_string(w.bytes) + " != rows*row bytes " +
                          std::to_string(row_bytes * w.rows));
    }
    const std::size_t n = static_cast<std::size_t>(rows), m = w.rows, k = w.cols;
    Range("Project", "x", x, n * k * 4);
    Range("Project", "y", y, n * m * 4);
    NoOverlap("Project", y, n * m * 4, x, n * k * 4);
    const auto* base = static_cast<const std::uint8_t*>(w.data);
    bool ok = true;
#pragma omp parallel
    {
      std::vector<float> row(k);
#pragma omp for schedule(static)
      for (std::size_t o = 0; o < m; ++o) {
        if (!quant::DequantizeRow(w.type, base + o * row_bytes, row.data(), k)) {
#pragma omp atomic write
          ok = false;
          continue;
        }
        for (std::size_t r = 0; r < n; ++r) {
          const float* xr = x + r * k;
          double acc = 0.0;
          for (std::size_t c = 0; c < k; ++c) {
            acc += static_cast<double>(row[c]) * static_cast<double>(xr[c]);
          }
          y[r * m + o] = static_cast<float>(acc);
        }
      }
    }
    if (!ok) Fail("Project", "cannot dequantize " + type);
  }

  void QkNormRopeAppend(float* q, const float* k, const float* v,
                        const float* q_gamma, const float* k_gamma,
                        void* k_cache, void* v_cache, KvType type,
                        std::uint32_t tokens, std::uint32_t heads,
                        std::uint32_t kv_heads, float eps,
                        std::uint32_t start_pos, std::uint32_t capacity,
                        std::uint32_t rotary_dim, float theta) override {
    const char* op = "QkNormRopeAppend";
    if (heads == 0 || kv_heads == 0 || heads % kv_heads != 0)
      Fail(op, "heads must be a positive multiple of kv_heads");
    if (rotary_dim != 0 && rotary_dim != km::kHeadDim)
      Fail(op, "rotary_dim must be 0 or 128");
    if (rotary_dim != 0 && !(std::isfinite(theta) && theta > 1.0F))
      Fail(op, "theta must be finite and > 1");
    if (capacity == 0 || tokens > capacity) Fail(op, "capacity must be >= tokens");
    if (std::uint64_t{start_pos} + tokens > (1ULL << 31)) Fail(op, "position out of range");
    const std::size_t D = km::kHeadDim, kvw = std::size_t{kv_heads} * D;
    const std::size_t esize = type == KvType::kF16 ? 2 : 4;
    Range(op, "q", q, std::size_t{tokens} * heads * D * 4);
    Range(op, "k", k, std::size_t{tokens} * kvw * 4);
    Range(op, "v", v, std::size_t{tokens} * kvw * 4);
    Range(op, "k_cache", k_cache, std::size_t{capacity} * kvw * esize);
    Range(op, "v_cache", v_cache, std::size_t{capacity} * kvw * esize);
    Weights(op, q_gamma, D);
    Weights(op, k_gamma, D);
    auto store = [&](void* cache, std::size_t index, float value) {
      if (type == KvType::kF16) {
        static_cast<std::uint16_t*>(cache)[index] = FloatToHalf(value);
      } else {
        static_cast<float*>(cache)[index] = value;
      }
    };
    std::vector<float> normed(D), rotated(D);
    for (std::uint32_t t = 0; t < tokens; ++t) {
      const std::uint32_t pos = start_pos + t;
      const std::size_t slot = km::RingSlot(pos, capacity);
      for (std::uint32_t hh = 0; hh < heads + kv_heads; ++hh) {
        const bool is_q = hh < heads;
        const std::uint32_t h = is_q ? hh : hh - heads;
        float* qrow = q + (std::size_t{t} * heads + h) * D;
        const float* src = is_q ? qrow : k + std::size_t{t} * kvw + h * D;
        const float scale = InvRms(src, D, eps);
        const float* gamma = is_q ? q_gamma : k_gamma;
        for (std::size_t d = 0; d < D; ++d) normed[d] = (src[d] * scale) * gamma[d];
        for (std::size_t d = 0; d < D; ++d) {
          float out = normed[d];
          if (rotary_dim != 0) {
            const std::uint32_t pair = km::RopePair(static_cast<std::uint32_t>(d), rotary_dim);
            const double angle = km::RopeAngle(pos, pair, rotary_dim, static_cast<double>(theta));
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));
            const float xa = normed[pair], xb = normed[pair + rotary_dim / 2];
            out = d < rotary_dim / 2 ? xa * c - xb * s : xa * s + xb * c;
          }
          rotated[d] = out;
        }
        if (is_q) {
          std::copy(rotated.begin(), rotated.end(), qrow);
        } else {
          for (std::size_t d = 0; d < D; ++d) store(k_cache, slot * kvw + h * D + d, rotated[d]);
        }
      }
      for (std::uint32_t h = 0; h < kv_heads; ++h) {
        for (std::size_t d = 0; d < D; ++d) {
          store(v_cache, slot * kvw + h * D + d, v[std::size_t{t} * kvw + h * D + d]);
        }
      }
    }
  }

  void Attention(const float* q, const void* k_cache, const void* v_cache,
                 KvType type, float* out, std::uint32_t tokens,
                 std::uint32_t heads, std::uint32_t kv_heads,
                 std::uint32_t window, std::uint32_t start_pos,
                 std::uint32_t capacity, float scale) override {
    const char* op = "Attention";
    if (heads == 0 || kv_heads == 0 || heads % kv_heads != 0)
      Fail(op, "heads must be a positive multiple of kv_heads");
    const std::uint32_t group = heads / kv_heads;
    if (group != 1 && group != 2 && group != 10)
      Fail(op, "group (heads / kv_heads) must be 1, 2 or 10");
    if (!std::isfinite(scale)) Fail(op, "scale must be finite");
    if (std::uint64_t{start_pos} + tokens > (1ULL << 31)) Fail(op, "position out of range");
    if (!lenient_ && capacity < km::MinCapacity(start_pos, tokens, window))
      Fail(op, "capacity < MinCapacity(start_pos, tokens, window)");
    if (capacity == 0) Fail(op, "capacity is 0");
    const std::size_t D = km::kHeadDim, kvw = std::size_t{kv_heads} * D;
    const std::size_t esize = type == KvType::kF16 ? 2 : 4;
    const std::size_t qn = std::size_t{tokens} * heads * D;
    Range(op, "q", q, qn * 4);
    Range(op, "out", out, qn * 4);
    Range(op, "k_cache", k_cache, std::size_t{capacity} * kvw * esize);
    Range(op, "v_cache", v_cache, std::size_t{capacity} * kvw * esize);
    NoOverlap(op, out, qn * 4, q, qn * 4);
    auto load = [&](const void* cache, std::size_t index) -> float {
      if (type == KvType::kF16) return HalfToFloat(static_cast<const std::uint16_t*>(cache)[index]);
      return static_cast<const float*>(cache)[index];
    };
    std::vector<double> p;
    for (std::uint32_t t = 0; t < tokens; ++t) {
      const std::uint32_t pos = start_pos + t;
      const std::uint32_t first = km::FirstVisibleKey(pos, window);
      p.assign(pos - first + 1, 0.0);
      for (std::uint32_t h = 0; h < heads; ++h) {
        const std::uint32_t kvh = km::KvHeadOf(h, group);
        const float* qh = q + (std::size_t{t} * heads + h) * D;
        double mx = -INFINITY;
        for (std::uint32_t j = first; j <= pos; ++j) {
          const std::size_t slot = km::RingSlot(j, capacity);
          double dot = 0.0;
          for (std::size_t d = 0; d < D; ++d)
            dot += static_cast<double>(qh[d]) * load(k_cache, slot * kvw + kvh * D + d);
          p[j - first] = static_cast<double>(static_cast<float>(dot) * scale);
          mx = std::max(mx, p[j - first]);
        }
        double sum = 0.0;
        for (double& s : p) {
          s = std::exp(s - mx);
          sum += s;
        }
        float* o = out + (std::size_t{t} * heads + h) * D;
        for (std::size_t d = 0; d < D; ++d) {
          double acc = 0.0;
          for (std::uint32_t j = first; j <= pos; ++j) {
            const std::size_t slot = km::RingSlot(j, capacity);
            acc += p[j - first] * load(v_cache, slot * kvw + kvh * D + d);
          }
          o[d] = static_cast<float>(acc / sum);
        }
      }
    }
  }

  void SigmoidGate(const float* attn, const float* gate, float* out,
                   std::uint32_t rows, std::uint32_t dim) override {
    const std::size_t n = std::size_t{rows} * dim;
    Range("SigmoidGate", "attn", attn, n * 4);
    Range("SigmoidGate", "gate", gate, n * 4);
    Range("SigmoidGate", "out", out, n * 4);
    for (std::size_t i = 0; i < n; ++i) out[i] = attn[i] * Sigmoid(gate[i]);
  }

  void ClampedSwiGlu(const float* gate, const float* up, float* out,
                     std::uint32_t rows, std::uint32_t dim, float clamp) override {
    if (!(std::isfinite(clamp) && clamp > 0.0F))
      Fail("ClampedSwiGlu", "clamp must be finite and > 0");
    const std::size_t n = std::size_t{rows} * dim;
    Range("ClampedSwiGlu", "gate", gate, n * 4);
    Range("ClampedSwiGlu", "up", up, n * 4);
    Range("ClampedSwiGlu", "out", out, n * 4);
    for (std::size_t i = 0; i < n; ++i) {
      float g = gate[i], u = up[i];
      g = g > clamp ? clamp : g;
      u = u > clamp ? clamp : (u < -clamp ? -clamp : u);
      out[i] = (g * Sigmoid(g)) * u;
    }
  }

  void GatherRows(const float* src, const std::int32_t* ids, float* dst,
                  std::uint32_t n_ids, std::uint32_t src_rows,
                  std::uint32_t dim) override {
    Range("GatherRows", "src", src, std::size_t{src_rows} * dim * 4);
    Range("GatherRows", "ids", ids, std::size_t{n_ids} * 4);
    Range("GatherRows", "dst", dst, std::size_t{n_ids} * dim * 4);
    NoOverlap("GatherRows", dst, std::size_t{n_ids} * dim * 4, src,
              std::size_t{src_rows} * dim * 4);
    for (std::uint32_t i = 0; i < n_ids; ++i) {
      const std::int32_t id = ids[i];
      const bool ok = id >= 0 && static_cast<std::uint32_t>(id) < src_rows;
      for (std::uint32_t d = 0; d < dim; ++d) {
        dst[std::size_t{i} * dim + d] =
            ok ? src[std::size_t{static_cast<std::uint32_t>(id)} * dim + d] : NAN;
      }
    }
  }

 private:
  struct Block {
    unsigned char* raw;
    std::size_t bytes;   // requested
    std::size_t padded;  // payload bytes incl. rounding
  };

  static float InvRms(const float* x, std::size_t dim, float eps) {
    double ss = 0.0;
    for (std::size_t i = 0; i < dim; ++i) ss += static_cast<double>(x[i]) * x[i];
    return 1.0F / std::sqrt(static_cast<float>(ss / static_cast<double>(dim)) + eps);
  }

  static std::string Corruption(unsigned char* payload, const Block& b) {
    const unsigned char* lo = b.raw;
    const unsigned char* hi = payload + b.padded;
    for (std::size_t i = 0; i < kGuard; ++i) {
      if (lo[i] != kGuardByte) return "buffer of " + std::to_string(b.bytes) + " bytes overwritten below its start";
      if (hi[i] != kGuardByte) return "buffer of " + std::to_string(b.bytes) + " bytes overwritten past its end";
    }
    return {};
  }

  /// [p, p+bytes) must lie inside one live buffer.
  void Range(const char* op, const char* what, const void* p, std::size_t bytes) const {
    if (p == nullptr) Fail(op, std::string(what) + " is null");
    const auto* c = static_cast<const unsigned char*>(p);
    auto it = blocks_.upper_bound(const_cast<unsigned char*>(c));
    if (it != blocks_.begin()) {
      --it;
      const unsigned char* start = it->first;
      if (c >= start && c < start + std::max<std::size_t>(it->second.bytes, 1)) {
        if (c + bytes > start + it->second.bytes)
          Fail(op, std::string(what) + " needs " + std::to_string(bytes) + " bytes at offset " +
                       std::to_string(c - start) + " of a " + std::to_string(it->second.bytes) +
                       "-byte buffer");
        return;
      }
    }
    Fail(op, std::string(what) + " is not inside any live buffer");
  }
  /// Weights / norm vectors live in the mapped GGUF: only a null check.
  static void Weights(const char* op, const float* p, std::size_t) {
    if (p == nullptr) Fail(op, "weight vector is null");
  }
  static void NoOverlap(const char* op, const void* a, std::size_t an, const void* b,
                        std::size_t bn) {
    const auto* x = static_cast<const unsigned char*>(a);
    const auto* y = static_cast<const unsigned char*>(b);
    if (x < y + bn && y < x + an) Fail(op, "output overlaps an input");
  }

  bool lenient_;
  bool poison_;
  std::map<unsigned char*, Block> blocks_;
  std::string corruption_;
};

/// Weight views straight into the mapped GGUF (host memory is the backend).
WeightView View(const TensorRef& t) {
  return WeightView{t.data, t.type, static_cast<std::uint32_t>(t.rows),
                    static_cast<std::uint32_t>(t.cols), t.RowBytes() * t.rows};
}
const float* F32(const TensorRef& t) {
  if (t.type != core::GgmlType::kF32) {
    throw std::runtime_error("ricursive host ops: " + std::string(t.name) + " must be F32");
  }
  return static_cast<const float*>(t.data);
}

std::shared_ptr<const ModelHandles> BuildHandles(const ModelWeights& w) {
  auto m = std::make_shared<ModelHandles>();
  m->config = w.config;
  m->table = EmbeddingTable{w.token_embd.data, w.token_embd.type,
                            static_cast<std::size_t>(w.token_embd.rows),
                            static_cast<std::size_t>(w.token_embd.cols)};
  m->output = View(w.output);
  m->output_norm = F32(w.output_norm);
  for (const LayerWeights& l : w.layers) {
    LayerHandles h;
    h.attn_q = View(l.attn_q);
    h.attn_k = View(l.attn_k);
    h.attn_v = View(l.attn_v);
    h.attn_output = View(l.attn_output);
    h.attn_gate = View(l.attn_gate);
    h.ffn_gate = View(l.ffn_gate);
    h.ffn_up = View(l.ffn_up);
    h.ffn_down = View(l.ffn_down);
    h.attn_norm = F32(l.attn_norm);
    h.attn_q_norm = F32(l.attn_q_norm);
    h.attn_k_norm = F32(l.attn_k_norm);
    h.post_attention_norm = F32(l.post_attention_norm);
    h.ffn_norm = F32(l.ffn_norm);
    h.post_ffw_norm = F32(l.post_ffw_norm);
    h.attn_vres_lambda = F32(l.attn_vres_lambda);
    m->layers.push_back(h);
  }
  return m;
}

}  // namespace

HostOpsExecutor::HostOpsExecutor(const ModelWeights& weights,
                                 const ExecutorConfig& config, Options options)
    : HostOpsExecutor(BuildHandles(weights), config, options) {}

HostOpsExecutor::HostOpsExecutor(std::shared_ptr<const ModelHandles> model,
                                 const ExecutorConfig& config, Options options)
    : OpsExecutor(std::make_unique<HostOps>(options.lenient_capacity, options.poison_memory), *model,
                  model, config,
                  /*batch_invariant_rows=*/config.max_rows, options.faults) {}

}  // namespace gufo::models::ricursive

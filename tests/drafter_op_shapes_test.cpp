// The drafter's operations one by one at the real model's shapes (docs/DRAFTER-S2.md, "Stage 2b", item 1).
//
//   drafter_op_shapes_test            (no arguments; RICURSIVE_TEST_BACKEND picks the sides)
//
// Why: the tiny fixtures have head dimension 32 (one wave), a 512-token vocabulary (one partial candidate list) and
// contexts of at most 87 rows, so the cross-wave sums of DraftQkNormRope and DraftAttention, the tree merge of 49 partial
// candidate lists, contexts above 87 rows, the wrap of the 5,120-row sliding rings and of the 17,408-row full ring and the
// 16,384-row window clamp never ran in a gate. Here every new operation is driven alone with seeded random inputs at the
// real shapes (hidden 5,120, head dimension 128, 80 query heads on 8 K/V heads, FFN 40,960, vocabulary 100,352, top-k 32,
// selector rank 256, F16 and F32 rings), through the `Ops` interface only.
//
// Sides. A is the implementation under test, B the one it is compared with:
//   RICURSIVE_TEST_BACKEND unset or cpu   A = the host operations (hip/ops_host.cpp), B = a deliberately naive
//                                         implementation written in this file from the definitions in
//                                         src/drafter_reference.cpp (it shares no helper with ops_host.cpp: doubles
//                                         everywhere, brute-force key sets, its own binary16 conversion, gufo's
//                                         DequantizeRow). Both ring types. About 2 to 3 minutes on 4 threads.
//   gpu-exact                             A = the GPU operations, B = the host operations; F32 rings.
//   gpu                                   A = the GPU operations, B = the host operations; F16 rings (production).
// The GPU modes run only through tests/run-gpu-gates.sh (gate drafter_op_shapes); the GPU kernels it exercises have never
// run before. The host side of a GPU run takes the same time as the cpu mode's host side (the naive side is not run).
//
// Tolerances (all derived from the arithmetic; "rel" = relative to the largest |reference| value of the same row):
//   DraftQkNormRope   q rows and F32 K rows: rel <= 2e-6 host vs naive, 1e-5 GPU vs host. The error is the sum of squares
//                     (128 fp32 additions, worst case 128 * 6e-8 = 7.7e-6 for sequential, about half for a tree) through a
//                     square root, plus cosf/sinf of an angle both sides form identically in float (a few ulp).
//                     F16 K rows: each element within one binary16 step; at most 0.5% (host) / 3% (GPU) of the elements
//                     may differ by that step (a relative error d moves a rounding decision with probability d / 2^-11).
//                     V rows are a plain conversion: bit-exact.
//   DraftAttention    rel <= 4e-6 host vs naive (the host rounds each dot to float before the scale: 6e-8 * |score| <= 6e-7
//                     on the weights, output about twice that), 5e-5 GPU vs host (fp32 dots: 128 * 6e-8 * |q||k| scale
//                     <= 1.3e-5 worst case on a score, an online softmax rescaled once per 128-key tile: 128 tiles at
//                     16,384 keys, each 6e-8). Queries are scaled so the scores have standard deviation 2 (softmax weights
//                     concentrate on a few dozen keys; a wrong key changes the output by far more than the tolerance).
//   DynamicConv       rel to the sum of the absolute terms (|base| + |dyn|) * |x| <= 1e-6 (four fp32 additions and a product each).
//   ProjectQ8         (item 2) cpu mode: the host operations against a double-precision product of the dequantized weights,
//                     relative RMSE per row <= 1e-6 (the host does not round activations). GPU modes: the W8A8 route
//                     against the host's exact activations, relative RMSE per row <= 1.2e-2: Q8_1 rounds each 32-element
//                     block of activations to int8 with step amax/127, an error of step/sqrt(12) per element, and
//                     amax of 32 Gaussian values is about 2.2 sigma, so the output's relative RMSE is about
//                     2.2 / (127 * sqrt(12)) = 0.5%, and the tolerance is 2.4 times that. Shapes: attention q (Q4_K
//                     10,240 x 5,120), a K projection (Q8_0 1,024 x 5,120), the FFN down projection (Q4_K 5,120 x
//                     40,960, the widest activation), the output head (Q6_K 100,352 x 5,120), rows 1, 3, 15 and 16.
//   SiluMul           element rel <= 3e-6 (fp32 expf and one division, a few ulp; the GPU's expf one or two more).
//   DraftSelect       candidate ids (by logit, equal logits by lowest token id) must be equal EXACTLY: logits are inputs.
//                     Candidate scores: |diff| <= 5e-6 * (|logit| + sum of the 256 |terms|) (68 sequential fp32 additions
//                     and three roundings per term). Chosen tokens: equal, except at a NEAR-TIE: the first row where the
//                     two sides differ is forgiven only if, in B's scores, the two choices are within the sum of their
//                     two score tolerances; near-ties are counted and printed, and the run fails when more than 2% of
//                     the compared rows are near-ties. Exact ties (built here) must be resolved to the lowest token id,
//                     with no forgiveness (docs/DRAFTER-S2.md, "Tie rules").
// Twins (cpu mode): the naive side with one wrong ring index (a slot off by one after the wrap, in the attention read
// and in the ring write) must fail the comparison.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "hip/ops.hpp"
#include "tests/check.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;
namespace core = gufo::core;
namespace quant = gufo::quant;

namespace {

using Clock = std::chrono::steady_clock;
double Seconds(Clock::time_point since) { return std::chrono::duration<double>(Clock::now() - since).count(); }

// ---- the real shapes (docs/DRAFTER-S1.md: the GGUF keys of Ripple-65B-Preview-DFlash2) ----------------------------------
constexpr std::uint32_t kE = 5120, kD = 128, kNH = 80, kNKV = 8, kKVW = kNKV * kD, kFF = 40960;
constexpr std::uint32_t kVocab = 100352;  ///< the target's output rows (projection_ops q6k_100352x5120); 49 partial lists
constexpr std::uint32_t kTopK = 32, kRank = 256, kBlock = 16, kRows = kBlock - 1;
constexpr std::uint32_t kConvK = 4, kConvGroup = 16;
constexpr std::uint32_t kSlideCap = 5120, kFullCap = 17408, kSlideWindow = 4096, kContextWindow = 16384;
constexpr float kEps = 1e-6F, kTheta = 10000.0F;

struct Backend {
  bool gpu{false};
  bool f16{false};
  std::string name{"cpu"};
};
const Backend& TheBackend() {
  static const Backend b = [] {
    const char* env = std::getenv("RICURSIVE_TEST_BACKEND");
    const std::string name = env != nullptr ? env : "cpu";
    Backend r;
    if (name == "gpu-exact") r = {true, false, "gpu-exact"};
    else if (name == "gpu") r = {true, true, "gpu"};
    else Require(name == "cpu", "RICURSIVE_TEST_BACKEND must be cpu, gpu-exact or gpu, not " + name);
    return r;
  }();
  return b;
}

// ---- binary16, written here (not shared with the host operations) ----------------------------------------------------------
std::uint16_t HalfBits(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  std::uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
float HalfValue(std::uint16_t b) {
  _Float16 h;
  std::memcpy(&h, &b, 2);
  return static_cast<float>(h);
}
std::size_t ElemBytes(KvType t) { return t == KvType::kF16 ? 2 : 4; }
float ReadElem(const std::uint8_t* base, KvType t, std::size_t i) {
  if (t == KvType::kF16) {
    std::uint16_t b;
    std::memcpy(&b, base + 2 * i, 2);
    return HalfValue(b);
  }
  float f;
  std::memcpy(&f, base + 4 * i, 4);
  return f;
}
void WriteElem(std::uint8_t* base, KvType t, std::size_t i, float v) {
  if (t == KvType::kF16) {
    const std::uint16_t b = HalfBits(v);
    std::memcpy(base + 2 * i, &b, 2);
  } else {
    std::memcpy(base + 4 * i, &v, 4);
  }
}

std::vector<float> Normals(Rng& rng, std::size_t n, double sigma, double mean = 0.0) {
  std::vector<float> v(n);
  for (auto& x : v) x = static_cast<float>(mean + sigma * rng.Normal());
  return v;
}
std::vector<std::uint8_t> RingBytes(const std::vector<float>& values, KvType t) {
  std::vector<std::uint8_t> out(values.size() * ElemBytes(t));
  for (std::size_t i = 0; i < values.size(); ++i) WriteElem(out.data(), t, i, values[i]);
  return out;
}

// ---- the cases ---------------------------------------------------------------------------------------------------------------
struct RopeCase {
  std::string what;
  KvType type{KvType::kF32};
  std::uint32_t tokens{0}, heads{0}, kv_heads{kNKV}, start_pos{0}, slot_start{0}, capacity{0};
  bool rope{true};
  std::vector<float> q, k, v, q_gamma, k_gamma;
  const std::uint8_t *ring_k{nullptr}, *ring_v{nullptr};  ///< initial contents, capacity * kKVW elements
};
struct RopeOut {
  std::vector<float> q;
  std::vector<std::uint8_t> ring_k, ring_v;
};
struct AttnCase {
  std::string what;
  KvType type{KvType::kF32};
  std::uint32_t capacity{0}, first{0}, anchor{0}, window{0};
  std::vector<float> q, k_blk, v_blk;
  const std::uint8_t *ring_k{nullptr}, *ring_v{nullptr};
};
struct ConvCase {
  std::uint32_t side{0};
  bool accumulate{false};
  std::vector<float> input, dyn, base, output_init;
};
struct SiluCase {
  std::vector<float> gate, up;
};
struct SelectCase {
  std::string what;
  core::GgmlType pred_type{core::GgmlType::kQ8_0}, succ_type{core::GgmlType::kQ4_K};
  std::uint32_t rows{kRows};
  std::int32_t anchor_token{0};
  std::vector<float> logits, hproj;
  std::vector<std::uint8_t> pred, succ;  ///< [kVocab] encoded rows
  bool exact_ties{false};                ///< built to contain exact ties: no near-tie forgiveness
};
struct ProjectCase {
  std::string what;
  core::GgmlType type{core::GgmlType::kQ4_K};
  std::uint32_t m{0}, k{0};
  int rows{0};
  std::vector<std::uint8_t> weights;
  std::vector<float> x;
};
struct SelectOut {
  std::vector<std::int32_t> tokens, ids;
  std::vector<float> scores;
};

class Side {
 public:
  virtual ~Side() = default;
  virtual RopeOut Rope(const RopeCase&) = 0;
  virtual std::vector<float> Attention(const AttnCase&) = 0;
  virtual std::vector<float> Conv(const ConvCase&) = 0;
  virtual std::vector<float> Silu(const SiluCase&) = 0;
  virtual SelectOut Select(const SelectCase&) = 0;
  virtual std::vector<float> ProjectQ8(const ProjectCase&) = 0;
};

// ---- a side that runs through Ops (host or GPU operations) --------------------------------------------------------------------
class OpsSide final : public Side {
 public:
  explicit OpsSide(std::unique_ptr<Ops> ops) : ops_(std::move(ops)) {}
  ~OpsSide() override { Release(); }

  RopeOut Rope(const RopeCase& c) override {
    const std::size_t es = ElemBytes(c.type), ring_bytes = std::size_t{c.capacity} * kKVW * es;
    float* q = c.heads != 0 ? Up(c.q) : nullptr;
    float *k = Up(c.k), *v = Up(c.v), *qg = Up(c.q_gamma), *kg = Up(c.k_gamma);
    auto* rk = static_cast<std::uint8_t*>(UpBytes(c.ring_k, ring_bytes));
    auto* rv = static_cast<std::uint8_t*>(UpBytes(c.ring_v, ring_bytes));
    ops_->DraftQkNormRope(q, k, v, qg, kg, rk, rv, c.type, c.tokens, c.heads, c.kv_heads, kD, kEps, c.start_pos, c.slot_start,
                          c.capacity, c.rope, kTheta);
    RopeOut o;
    if (c.heads != 0) o.q = Down<float>(q, c.q.size());
    o.ring_k = Down<std::uint8_t>(rk, ring_bytes);
    o.ring_v = Down<std::uint8_t>(rv, ring_bytes);
    Release();
    return o;
  }
  std::vector<float> Attention(const AttnCase& c) override {
    const std::size_t ring_bytes = std::size_t{c.capacity} * kKVW * ElemBytes(c.type);
    const float* q = Up(c.q);
    const float *kb = Up(c.k_blk), *vb = Up(c.v_blk);
    const void* rk = UpBytes(c.ring_k, ring_bytes);
    const void* rv = UpBytes(c.ring_v, ring_bytes);
    float* out = Alloc<float>(std::size_t{kBlock} * kNH * kD);
    ops_->DraftAttention(q, rk, rv, c.type, c.capacity, c.first, c.anchor, kb, vb, out, kBlock, kNH, kNKV, kD, c.window,
                         static_cast<float>(1.0 / std::sqrt(static_cast<double>(kD))));
    auto result = Down<float>(out, std::size_t{kBlock} * kNH * kD);
    Release();
    return result;
  }
  std::vector<float> Conv(const ConvCase& c) override {
    const float *in = Up(c.input), *dyn = Up(c.dyn), *base = Up(c.base);
    float* out = Up(c.output_init);
    ops_->DynamicConv(in, dyn, base, out, kBlock, kE, kConvK, kConvGroup, c.side, c.accumulate);
    auto result = Down<float>(out, c.output_init.size());
    Release();
    return result;
  }
  std::vector<float> Silu(const SiluCase& c) override {
    float* gate = Up(c.gate);
    const float* up = Up(c.up);
    ops_->SiluMul(gate, up, static_cast<std::uint32_t>(c.gate.size()));
    auto result = Down<float>(gate, c.gate.size());
    Release();
    return result;
  }
  std::vector<float> ProjectQ8(const ProjectCase& c) override {
    std::vector<std::uint8_t> padded = c.weights;
    padded.resize(c.weights.size() + 4096, 0);  // the uploader's zeroed tail
    WeightView w;
    w.data = UpBytes(padded.data(), padded.size());
    w.type = c.type;
    w.rows = c.m;
    w.cols = c.k;
    w.bytes = c.weights.size();
    const float* x = Up(c.x);
    float* y = Alloc<float>(std::size_t{static_cast<std::size_t>(c.rows)} * c.m);
    ops_->ReserveProjection(static_cast<std::size_t>(c.rows), c.k);
    ops_->ProjectQ8(w, x, c.rows, y);
    auto result = Down<float>(y, std::size_t{static_cast<std::size_t>(c.rows)} * c.m);
    Release();
    return result;
  }
  SelectOut Select(const SelectCase& c) override {
    const std::uint32_t P = (kVocab + 2047) / 2048;  // the partial lists of the operation's contract (DraftSelectPartials)
    const std::size_t R = c.rows;
    const float* logits = Up(c.logits);
    const float* hproj = Up(c.hproj);
    // The codebooks sit in backend memory with a zeroed tail (the uploader's convention).
    const auto weight = [&](const std::vector<std::uint8_t>& bytes, core::GgmlType type) {
      std::vector<std::uint8_t> padded = bytes;
      padded.resize(bytes.size() + 4096, 0);
      WeightView w;
      w.data = UpBytes(padded.data(), padded.size());
      w.type = type;
      w.rows = kVocab;
      w.cols = kRank;
      w.bytes = bytes.size();
      return w;
    };
    const WeightView pred = weight(c.pred, c.pred_type), succ = weight(c.succ, c.succ_type);
    auto* tokens = Alloc<std::int32_t>(R + 1);
    float* scores = Alloc<float>(R * kTopK);
    auto* ids = Alloc<std::int32_t>(R * kTopK);
    float* part_scores = Alloc<float>(R * P * kTopK);
    auto* part_ids = Alloc<std::int32_t>(R * P * kTopK);
    ops_->CopyIn(tokens, &c.anchor_token, sizeof(std::int32_t));
    ops_->DraftSelect(logits, hproj, pred, succ, tokens, scores, ids, part_scores, part_ids, c.rows, kVocab, kRank, kTopK);
    const auto all_tokens = Down<std::int32_t>(tokens, R + 1);
    SelectOut o;
    o.tokens.assign(all_tokens.begin() + 1, all_tokens.end());
    o.ids = Down<std::int32_t>(ids, R * kTopK);
    o.scores = Down<float>(scores, R * kTopK);
    Release();
    return o;
  }

 private:
  void* AllocBytes(std::size_t bytes) {
    void* p = ops_->Allocate(bytes);
    owned_.push_back(p);
    return p;
  }
  template<class T>
  T* Alloc(std::size_t n) {
    return static_cast<T*>(AllocBytes(std::max<std::size_t>(n, 1) * sizeof(T)));
  }
  void* UpBytes(const void* host, std::size_t bytes) {
    void* p = AllocBytes(bytes);
    ops_->CopyIn(p, host, bytes);
    return p;
  }
  template<class T>
  T* Up(const std::vector<T>& v) {
    return static_cast<T*>(UpBytes(v.data(), v.size() * sizeof(T)));
  }
  template<class T>
  std::vector<T> Down(const void* dev, std::size_t n) {
    std::vector<T> v(n);
    ops_->CopyOut(v.data(), dev, n * sizeof(T));
    return v;
  }
  void Release() {
    if (owned_.empty()) return;
    ops_->Synchronize();
    for (void* p : owned_) ops_->Free(p);
    owned_.clear();
  }
  std::unique_ptr<Ops> ops_;
  std::vector<void*> owned_;
};

// ---- the naive side: the definitions of src/drafter_reference.cpp, in doubles, brute force -------------------------------------
struct Faults {
  bool attention_slot{false};  ///< the slot of a context row past the wrap is one too high
  bool rope_slot{false};       ///< the same in the ring write
};

class NaiveSide final : public Side {
 public:
  explicit NaiveSide(Faults f = {}) : f_(f) {}

  RopeOut Rope(const RopeCase& c) override {
    const std::size_t es = ElemBytes(c.type), ring_bytes = std::size_t{c.capacity} * kKVW * es;
    RopeOut o;
    o.q = c.q;
    o.ring_k.assign(c.ring_k, c.ring_k + ring_bytes);
    o.ring_v.assign(c.ring_v, c.ring_v + ring_bytes);
    const std::size_t half = kD / 2;
    std::vector<float> inv(half);
    for (std::size_t i = 0; i < half; ++i) inv[i] = 1.0F / std::pow(kTheta, static_cast<float>(2 * i) / static_cast<float>(kD));
    // The definition: per head RMSNorm with the shared gamma (double sum of squares, scale in float), then NeoX rotation
    // with float32 frequencies, angle = float(position) * inv.
    const auto normalize = [&](const float* x, const std::vector<float>& gamma, float* y) {
      double ss = 0.0;
      for (std::size_t i = 0; i < kD; ++i) ss += static_cast<double>(x[i]) * x[i];
      const float scale = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(kD) + static_cast<double>(kEps)));
      for (std::size_t i = 0; i < kD; ++i) y[i] = x[i] * scale * gamma[i];
    };
    const auto rotate = [&](float* x, std::uint32_t position) {
      const float pos = static_cast<float>(position);
      for (std::size_t i = 0; i < half; ++i) {
        const float angle = pos * inv[i], cs = std::cos(angle), sn = std::sin(angle);
        const float x1 = x[i], x2 = x[i + half];
        x[i] = x1 * cs - x2 * sn;
        x[i + half] = x2 * cs + x1 * sn;
      }
    };
    std::vector<float> row(kD);
    for (std::uint32_t t = 0; t < c.tokens; ++t) {
      const std::uint32_t position = c.start_pos + t;
      std::size_t slot = (std::size_t{c.slot_start} + t) % c.capacity;
      if (f_.rope_slot && c.slot_start + t >= c.capacity) slot = (slot + 1) % c.capacity;
      for (std::uint32_t h = 0; h < c.heads; ++h) {
        float* qrow = &o.q[(std::size_t{t} * c.heads + h) * kD];
        normalize(qrow, c.q_gamma, row.data());
        if (c.rope) rotate(row.data(), position);
        std::copy(row.begin(), row.end(), qrow);
      }
      for (std::uint32_t h = 0; h < c.kv_heads; ++h) {
        normalize(&c.k[std::size_t{t} * kKVW + h * kD], c.k_gamma, row.data());
        if (c.rope) rotate(row.data(), position);
        for (std::size_t d = 0; d < kD; ++d) WriteElem(o.ring_k.data(), c.type, slot * kKVW + h * kD + d, row[d]);
        for (std::size_t d = 0; d < kD; ++d)
          WriteElem(o.ring_v.data(), c.type, slot * kKVW + h * kD + d, c.v[std::size_t{t} * kKVW + h * kD + d]);
      }
    }
    return o;
  }

  std::vector<float> Attention(const AttnCase& c) override {
    std::vector<float> out(std::size_t{kBlock} * kNH * kD);
    const double scale = 1.0 / std::sqrt(static_cast<double>(kD));
    const std::uint32_t group = kNH / kNKV;
    const auto slot_of = [&](std::uint32_t position) {
      std::size_t slot = position % c.capacity;
      if (f_.attention_slot && position >= c.capacity) slot = (slot + 1) % c.capacity;
      return slot;
    };
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int t = 0; t < static_cast<int>(kBlock); ++t) {
      for (int h = 0; h < static_cast<int>(kNH); ++h) {
        const std::int64_t p = std::int64_t{c.anchor} + t;
        const auto visible = [&](std::int64_t j) { return c.window == 0 || std::llabs(p - j) < static_cast<std::int64_t>(c.window); };
        const std::uint32_t kvh = static_cast<std::uint32_t>(h) / group;
        const float* q = &c.q[(static_cast<std::size_t>(t) * kNH + static_cast<std::size_t>(h)) * kD];
        std::vector<double> score;
        std::vector<std::pair<bool, std::size_t>> where;  // (context?, slot or block row)
        for (std::uint32_t j = c.first; j < c.anchor; ++j)
          if (visible(j)) where.emplace_back(true, slot_of(j));
        for (std::uint32_t i = 0; i < kBlock; ++i)
          if (visible(std::int64_t{c.anchor} + i)) where.emplace_back(false, i);
        score.resize(where.size());
        double mx = -INFINITY;
        for (std::size_t n = 0; n < where.size(); ++n) {
          double dot = 0.0;
          for (std::size_t d = 0; d < kD; ++d) {
            const double kv = where[n].first ? ReadElem(c.ring_k, c.type, where[n].second * kKVW + kvh * kD + d)
                                             : c.k_blk[where[n].second * kKVW + kvh * kD + d];
            dot += static_cast<double>(q[d]) * kv;
          }
          score[n] = dot * scale;
          mx = std::max(mx, score[n]);
        }
        double sum = 0.0;
        for (double& s : score) {
          s = std::exp(s - mx);
          sum += s;
        }
        float* o = &out[(static_cast<std::size_t>(t) * kNH + static_cast<std::size_t>(h)) * kD];
        for (std::size_t d = 0; d < kD; ++d) {
          double acc = 0.0;
          for (std::size_t n = 0; n < where.size(); ++n) {
            const double vv = where[n].first ? ReadElem(c.ring_v, c.type, where[n].second * kKVW + kvh * kD + d)
                                             : c.v_blk[where[n].second * kKVW + kvh * kD + d];
            acc += score[n] * vv;
          }
          o[d] = static_cast<float>(acc / sum);
        }
      }
    }
    return out;
  }

  std::vector<float> Conv(const ConvCase& c) override {
    std::vector<float> out = c.output_init;
    const std::uint32_t groups = kE / kConvGroup, per_token = 2 * kConvK * groups;
    for (std::uint32_t t = 0; t < kBlock; ++t) {
      for (std::uint32_t ch = 0; ch < kE; ++ch) {
        double sum = 0.0;
        for (std::uint32_t tap = 0; tap < kConvK && tap <= t; ++tap) {
          const double base = c.base[std::size_t{c.side * kConvK + tap} * kE + ch];
          const double dyn = c.dyn[std::size_t{t} * per_token + std::size_t{c.side * kConvK + tap} * groups + ch / kConvGroup];
          sum += (base + dyn) * c.input[std::size_t{t - tap} * kE + ch];
        }
        const std::size_t at = std::size_t{t} * kE + ch;
        out[at] = static_cast<float>((c.accumulate ? static_cast<double>(c.output_init[at]) : 0.0) + sum);
      }
    }
    return out;
  }

  std::vector<float> Silu(const SiluCase& c) override {
    std::vector<float> out(c.gate.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
      const double g = c.gate[i];
      out[i] = static_cast<float>(g / (1.0 + std::exp(-g)) * static_cast<double>(c.up[i]));
    }
    return out;
  }

  std::vector<float> ProjectQ8(const ProjectCase& c) override {
    const std::size_t n = static_cast<std::size_t>(c.rows), row_bytes = quant::QuantizedRowBytes(c.type, c.k);
    std::vector<float> y(n * c.m);
#pragma omp parallel
    {
      std::vector<float> row(c.k);
#pragma omp for schedule(static)
      for (std::int64_t o = 0; o < static_cast<std::int64_t>(c.m); ++o) {
        if (!quant::DequantizeRow(c.type, c.weights.data() + static_cast<std::size_t>(o) * row_bytes, row.data(), c.k)) std::abort();
        for (std::size_t r = 0; r < n; ++r) {
          double acc = 0.0;
          for (std::size_t i = 0; i < c.k; ++i) acc += static_cast<double>(row[i]) * c.x[r * c.k + i];
          y[r * c.m + static_cast<std::size_t>(o)] = static_cast<float>(acc);
        }
      }
    }
    return y;
  }

  SelectOut Select(const SelectCase& c) override {
    SelectOut o;
    const std::size_t R = c.rows;
    o.ids.resize(R * kTopK);
    o.scores.resize(R * kTopK);
    std::int32_t prev = c.anchor_token;
    std::vector<float> pred(kRank), succ(kRank);
    for (std::size_t r = 0; r < R; ++r) {
      const float* lg = &c.logits[r * kVocab];
      std::vector<std::int32_t> idx(kVocab);
      std::iota(idx.begin(), idx.end(), 0);
      std::partial_sort(idx.begin(), idx.begin() + kTopK, idx.end(),
                        [&](std::int32_t x, std::int32_t y) { return lg[x] > lg[y] || (lg[x] == lg[y] && x < y); });
      Row(c.pred, c.pred_type, static_cast<std::uint32_t>(prev), pred.data());
      double best = -INFINITY;
      std::int32_t best_id = 0;
      for (std::uint32_t k = 0; k < kTopK; ++k) {
        Row(c.succ, c.succ_type, static_cast<std::uint32_t>(idx[k]), succ.data());
        double t = 0.0;
        for (std::uint32_t i = 0; i < kRank; ++i) t += static_cast<double>(pred[i]) * c.hproj[r * kRank + i] * succ[i];
        const double score = static_cast<double>(lg[idx[k]]) + t;
        o.ids[r * kTopK + k] = idx[k];
        o.scores[r * kTopK + k] = static_cast<float>(score);
        if (score > best || (score == best && idx[k] < best_id)) {
          best = score;
          best_id = idx[k];
        }
      }
      o.tokens.push_back(best_id);
      prev = best_id;
    }
    return o;
  }

 private:
  static void Row(const std::vector<std::uint8_t>& book, core::GgmlType type, std::uint32_t id, float* out) {
    const std::size_t row_bytes = type == core::GgmlType::kF32 ? std::size_t{kRank} * 4 : quant::QuantizedRowBytes(type, kRank);
    Require(row_bytes != 0, "codebook row bytes");
    if (type == core::GgmlType::kF32) {
      std::memcpy(out, book.data() + std::size_t{id} * row_bytes, row_bytes);
      return;
    }
    Require(quant::DequantizeRow(type, book.data() + std::size_t{id} * row_bytes, out, kRank), "DequantizeRow");
  }
  Faults f_;
};

// ---- comparison helpers -----------------------------------------------------------------------------------------------------
/// Worst over rows of (max |got - want| / max |want|) for rows of `row` elements. NaN-safe (a NaN is infinity).
double WorstRowRel(const std::vector<float>& got, const std::vector<float>& want, std::size_t row) {
  Require(got.size() == want.size() && row != 0 && got.size() % row == 0, "compared vectors differ in size");
  double worst = 0.0;
  for (std::size_t r = 0; r < got.size() / row; ++r) {
    double mx = 0.0, diff = 0.0;
    bool bad = false;
    for (std::size_t i = 0; i < row; ++i) {
      const float g = got[r * row + i], w = want[r * row + i];
      if (!std::isfinite(g) || !std::isfinite(w)) {
        bad = true;
        break;
      }
      mx = std::max(mx, static_cast<double>(std::fabs(w)));
      diff = std::max(diff, std::fabs(static_cast<double>(g) - w));
    }
    const double rel = bad ? INFINITY : diff / std::max(mx, 1e-30);
    if (!(rel <= worst)) worst = rel;  // keeps NaN / infinity
  }
  return worst;
}

struct Verdict {
  bool ok{true};
  std::string line;
};

/// A one-line report; the check fails (returned in `ok`) when `worst` exceeds `tol`.
Verdict Judge(const std::string& what, double worst, double tol, const std::string& extra = {}) {
  Verdict v;
  v.ok = !Exceeds(worst, tol);
  v.line = "  " + what + ": worst " + Sci(worst) + " (tolerance " + Sci(tol) + ")" + (extra.empty() ? "" : ", " + extra) +
           (v.ok ? "" : "  -> FAILS");
  return v;
}

// ---- the scenarios -----------------------------------------------------------------------------------------------------------
struct Rings {
  std::vector<float> k, v;  ///< kFullCap * kKVW values (a ring of the sliding layers is the first kSlideCap rows)
  std::vector<std::uint8_t> kb[2], vb[2];  ///< [0] F32, [1] F16
  const std::uint8_t* K(KvType t) const { return kb[t == KvType::kF16 ? 1 : 0].data(); }
  const std::uint8_t* V(KvType t) const { return vb[t == KvType::kF16 ? 1 : 0].data(); }
};

struct Report {
  std::size_t checks{0}, failures{0};
  void Add(const Verdict& v) {
    ++checks;
    if (!v.ok) ++failures;
    std::printf("%s\n", v.line.c_str());
  }
};

struct Tolerances {
  double rope_rel, rope_half_share, attn_rel, conv_rel, silu_rel;
};
Tolerances TolFor(bool gpu) {
  return gpu ? Tolerances{1e-5, 0.03, 5e-5, 1e-6, 3e-6} : Tolerances{2e-6, 0.005, 4e-6, 1e-6, 3e-6};
}

/// Ring comparison for DraftQkNormRope: V bit-exact, K per the type (see the header).
Verdict CompareRope(const RopeCase& c, const RopeOut& a, const RopeOut& b, const Tolerances& tol) {
  Verdict v;
  std::string extra;
  double worst = 0.0;
  if (c.heads != 0) worst = WorstRowRel(a.q, b.q, kD);
  std::size_t half_steps = 0, elements = 0;
  bool bad = false;
  if (c.type == KvType::kF32) {
    std::vector<float> ka(c.capacity * kKVW), kb(c.capacity * kKVW);
    std::memcpy(ka.data(), a.ring_k.data(), ka.size() * 4);
    std::memcpy(kb.data(), b.ring_k.data(), kb.size() * 4);
    const double r = WorstRowRel(ka, kb, kD);
    if (!(r <= worst)) worst = r;
  } else {
    for (std::size_t i = 0; i < std::size_t{c.capacity} * kKVW; ++i) {
      const float x = ReadElem(a.ring_k.data(), c.type, i), y = ReadElem(b.ring_k.data(), c.type, i);
      if (!std::isfinite(x) || !std::isfinite(y)) {
        bad = true;
        break;
      }
      ++elements;
      const std::uint16_t ba = HalfBits(x), bb = HalfBits(y);
      if (ba != bb) {
        const bool sign_differs = ((ba ^ bb) & 0x8000) != 0, nonzero = ((ba | bb) & 0x7FFF) != 0;
        const int step = std::abs(static_cast<int>(ba & 0x7FFF) - static_cast<int>(bb & 0x7FFF));
        if ((sign_differs && nonzero) || step > 1) bad = true;  // more than one binary16 step
        ++half_steps;
      }
    }
  }
  const bool v_equal = a.ring_v == b.ring_v;
  const double share = elements != 0 ? static_cast<double>(half_steps) / static_cast<double>(elements) : 0.0;
  v.ok = !bad && v_equal && !Exceeds(worst, tol.rope_rel) && !Exceeds(share, tol.rope_half_share);
  v.line = "  rope+store " + c.what + ": q/K worst rel " + Sci(worst) + " (tolerance " + Sci(tol.rope_rel) + ")";
  if (c.type == KvType::kF16)
    v.line += ", " + std::to_string(half_steps) + " of " + std::to_string(elements) + " K elements one half-step apart (share " + Sci(share) +
              ", limit " + Sci(tol.rope_half_share) + ")";
  v.line += std::string(", V rings ") + (v_equal ? "bit-equal" : "DIFFER") + (v.ok ? "" : "  -> FAILS");
  return v;
}

RopeCase MakeRope(Rng& rng, const Rings& rings, KvType type, const std::string& what, std::uint32_t tokens, std::uint32_t heads,
                  std::uint32_t start_pos, std::uint32_t slot_start, std::uint32_t capacity, bool rope) {
  RopeCase c;
  c.what = what;
  c.type = type;
  c.tokens = tokens;
  c.heads = heads;
  c.start_pos = start_pos;
  c.slot_start = slot_start;
  c.capacity = capacity;
  c.rope = rope;
  c.q = Normals(rng, std::size_t{tokens} * heads * kD, 1.5);
  c.k = Normals(rng, std::size_t{tokens} * kKVW, 1.5);
  c.v = Normals(rng, std::size_t{tokens} * kKVW, 1.0);
  c.q_gamma = Normals(rng, kD, 0.2, 1.0);
  c.k_gamma = Normals(rng, kD, 0.2, 1.0);
  c.ring_k = rings.K(type);  // the first `capacity` rows: a smaller ring is a prefix
  c.ring_v = rings.V(type);
  return c;
}

void RunRope(Side& a, Side& b, Rng& rng, const Rings& rings, KvType type, Report& report, const Tolerances& tol) {
  const std::string t = type == KvType::kF16 ? "F16" : "F32";
  struct Spec {
    const char* name;
    std::uint32_t tokens, heads, start, slot, cap;
    bool rope;
  };
  const Spec specs[] = {
      // the block pass: 16 rows, 80 query heads, the block buffer (capacity 16, slot t), positions from 0 to past 32,000
      {"block pass, position 0", 16, kNH, 0, 0, 16, true},
      {"block pass, position 87", 16, kNH, 87, 0, 16, true},
      {"block pass, position 4351", 16, kNH, 4351, 0, 16, true},
      {"block pass, position 16384", 16, kNH, 16384, 0, 16, true},
      {"block pass, position 32760, no rotary layer", 16, kNH, 32760, 0, 16, false},
      // the context pass: K and V rows into a ring (heads 0); sliding ring 5,120 rows, full ring 17,408 rows
      {"context 1 row, sliding ring, position 87", 1, 0, 87, 87, kSlideCap, true},
      {"context 512 rows, sliding ring, from 0", 512, 0, 0, 0, kSlideCap, true},
      {"context 512 rows, sliding ring, ending exactly at the wrap", 512, 0, kSlideCap - 512, kSlideCap - 512, kSlideCap, true},
      {"context 512 rows, sliding ring, 100 rows before the wrap", 512, 0, kSlideCap - 100, kSlideCap - 100, kSlideCap, true},
      {"context 16 rows, sliding ring, 1 row before the wrap", 16, 0, 3 * kSlideCap - 1, kSlideCap - 1, kSlideCap, true},
      {"context 512 rows, sliding ring, just after the wrap", 512, 0, kSlideCap + 3, 3, kSlideCap, true},
      {"context 512 rows, full ring, from 16384", 512, 0, 16384, 16384, kFullCap, true},
      {"context 512 rows, full ring, 100 rows before the wrap", 512, 0, kFullCap - 100, kFullCap - 100, kFullCap, true},
      {"context 512 rows, full ring, second lap, no rotary layer", 512, 0, 2 * kFullCap + 5, 5, kFullCap, false},
      {"context 16 rows, full ring, 3 rows before the wrap", 16, 0, kFullCap - 3, kFullCap - 3, kFullCap, true},
  };
  for (const Spec& s : specs) {
    const RopeCase c = MakeRope(rng, rings, type, std::string(s.name) + ", " + t + " ring", s.tokens, s.heads, s.start, s.slot, s.cap, s.rope);
    const RopeOut oa = a.Rope(c), ob = b.Rope(c);
    report.Add(CompareRope(c, oa, ob, tol));
  }
}

struct AttnSpec {
  const char* name;
  std::uint32_t cap, window, anchor, first;
};

AttnCase MakeAttn(Rng& rng, const Rings& rings, KvType type, const AttnSpec& s) {
  AttnCase c;
  c.what = std::string(s.name) + ", " + (type == KvType::kF16 ? "F16" : "F32") + " ring";
  c.type = type;
  c.capacity = s.cap;
  c.window = s.window;
  c.anchor = s.anchor;
  c.first = s.first;
  c.q = Normals(rng, std::size_t{kBlock} * kNH * kD, 2.0);  // scores N(0, 4) with unit keys and the 1/sqrt(128) scale
  c.k_blk = Normals(rng, std::size_t{kBlock} * kKVW, 1.0);
  c.v_blk = Normals(rng, std::size_t{kBlock} * kKVW, 1.0);
  c.ring_k = rings.K(type);
  c.ring_v = rings.V(type);
  return c;
}

std::vector<AttnSpec> AttentionSpecs() {
  return {
      // sliding layers: window 4,096, ring 5,120 rows (4,096 + 1,024 slack)
      {"sliding, 87 context rows", kSlideCap, kSlideWindow, 87, 0},
      {"sliding, 500 context rows", kSlideCap, kSlideWindow, 500, 0},
      {"sliding, 4096 context rows", kSlideCap, kSlideWindow, 4096, 0},
      {"sliding, ring just full (5120 rows)", kSlideCap, kSlideWindow, kSlideCap, 0},
      {"sliding, one past the ring's wrap (position 5120 in slot 0)", kSlideCap, kSlideWindow, kSlideCap + 1, 1},
      {"sliding, second lap, block straddling a slot boundary", kSlideCap, kSlideWindow, 2 * kSlideCap + 7, kSlideCap + 7},
      {"sliding, position 20000", kSlideCap, kSlideWindow, 20000, 20000 - kSlideCap},
      // the full layer: no window, the last 16,384 rows, ring 17,408 rows
      {"full, 87 context rows", kFullCap, 0, 87, 0},
      {"full, 500 context rows", kFullCap, 0, 500, 0},
      {"full, 4096 context rows", kFullCap, 0, 4096, 0},
      {"full, base above 0 (rows 8900 to 8999)", kFullCap, 0, 9000, 8900},
      {"full, 16384 context rows (the window)", kFullCap, 0, kContextWindow, 0},
      {"full, one past the ring's wrap (position 17408 in slot 0), 16384-row clamp", kFullCap, 0, kFullCap + 1, kFullCap + 1 - kContextWindow},
      {"full, position 30000 (rows 13616 to 29999 wrap)", kFullCap, 0, 30000, 30000 - kContextWindow},
  };
}

void RunAttention(Side& a, Side& b, Rng& rng, const Rings& rings, KvType type, Report& report, const Tolerances& tol) {
  for (const AttnSpec& s : AttentionSpecs()) {
    const AttnCase c = MakeAttn(rng, rings, type, s);
    const auto t0 = Clock::now();
    const auto oa = a.Attention(c);
    const double ta = Seconds(t0);
    const auto t1 = Clock::now();
    const auto ob = b.Attention(c);
    char timing[96];
    std::snprintf(timing, sizeof timing, "A %.1f s, B %.1f s", ta, Seconds(t1));
    report.Add(Judge("attention " + c.what, WorstRowRel(oa, ob, kD), tol.attn_rel, timing));
  }
}

void RunConvSilu(Side& a, Side& b, Rng& rng, Report& report, const Tolerances& tol) {
  for (const std::uint32_t side : {0U, 1U}) {
    for (const bool accumulate : {false, true}) {
      ConvCase c;
      c.side = side;
      c.accumulate = accumulate;
      c.input = Normals(rng, std::size_t{kBlock} * kE, 1.0);
      c.dyn = Normals(rng, std::size_t{kBlock} * 2 * kConvK * (kE / kConvGroup), 0.3);
      c.base = Normals(rng, std::size_t{2} * kConvK * kE, 0.5);
      c.output_init = Normals(rng, std::size_t{kBlock} * kE, 1.0);
      const auto oa = a.Conv(c), ob = b.Conv(c);
      // The tolerance is relative to the sum of the absolute terms of an element; as a row measure use the row max of
      // |reference| scaled by the terms' typical growth: compare element by element against 1e-6 * (sum |terms| + |init|).
      double worst = 0.0;
      const std::uint32_t groups = kE / kConvGroup, per_token = 2 * kConvK * groups;
      for (std::uint32_t t = 0; t < kBlock; ++t) {
        for (std::uint32_t ch = 0; ch < kE; ++ch) {
          double bound = 0.0;
          for (std::uint32_t tap = 0; tap < kConvK && tap <= t; ++tap)  // |base| + |dyn|: the float sum base + dyn rounds relative to them
            bound += (std::fabs(static_cast<double>(c.base[std::size_t{side * kConvK + tap} * kE + ch])) +
                      std::fabs(static_cast<double>(c.dyn[std::size_t{t} * per_token + std::size_t{side * kConvK + tap} * groups + ch / kConvGroup]))) *
                     std::fabs(static_cast<double>(c.input[std::size_t{t - tap} * kE + ch]));
          const std::size_t at = std::size_t{t} * kE + ch;
          if (accumulate) bound += std::fabs(c.output_init[at]);
          const double diff = std::fabs(static_cast<double>(oa[at]) - ob[at]);
          const double rel = std::isfinite(oa[at]) && std::isfinite(ob[at]) ? diff / std::max(bound, 1e-30) : INFINITY;
          if (!(rel <= worst)) worst = rel;
        }
      }
      report.Add(Judge(std::string("dynamic convolution, side ") + std::to_string(side) + (accumulate ? ", accumulating" : ", plain") +
                           ", 16 x 5120",
                       worst, tol.conv_rel));
    }
  }
  SiluCase s;
  s.gate = Normals(rng, std::size_t{kBlock} * kFF, 2.0);
  s.up = Normals(rng, std::size_t{kBlock} * kFF, 1.0);
  const auto oa = a.Silu(s), ob = b.Silu(s);
  double worst = 0.0;
  for (std::size_t i = 0; i < oa.size(); ++i) {
    const double rel = std::isfinite(oa[i]) && std::isfinite(ob[i]) ? std::fabs(static_cast<double>(oa[i]) - ob[i]) / (std::fabs(ob[i]) + 1e-20)
                                                                    : INFINITY;
    if (!(rel <= worst)) worst = rel;
  }
  report.Add(Judge("silu(gate) * up, 16 x 40960", worst, tol.silu_rel));
}

// ---- the selector ---------------------------------------------------------------------------------------------------------------
/// Random encoded codebook rows (finite scales; the encodings' element range keeps terms of order one).
std::vector<std::uint8_t> RandomBook(Rng& rng, core::GgmlType type) {
  const std::size_t row_bytes = type == core::GgmlType::kF32 ? std::size_t{kRank} * 4 : quant::QuantizedRowBytes(type, kRank);
  Require(row_bytes != 0, "codebook type");
  std::vector<std::uint8_t> book(row_bytes * kVocab);
  const auto set_half = [&](std::size_t at, float x) {
    const std::uint16_t h = HalfBits(x);
    book[at] = static_cast<std::uint8_t>(h & 0xFF);
    book[at + 1] = static_cast<std::uint8_t>(h >> 8);
  };
  for (std::size_t r = 0; r < kVocab; ++r) {
    std::uint8_t* row = book.data() + r * row_bytes;
    if (type == core::GgmlType::kF32) {
      for (std::uint32_t i = 0; i < kRank; ++i) {
        const float x = static_cast<float>(rng.Normal() * 0.5);
        std::memcpy(row + 4 * i, &x, 4);
      }
    } else {
      for (std::size_t i = 0; i < row_bytes; ++i) book[r * row_bytes + i] = static_cast<std::uint8_t>(rng.Below(256));
      if (type == core::GgmlType::kQ8_0) {
        for (std::size_t b = 0; b < row_bytes; b += 34) set_half(r * row_bytes + b, 0.004F * static_cast<float>(1 + rng.Below(8)));
      } else {  // Q4_K: d and dmin
        set_half(r * row_bytes, 0.0008F * static_cast<float>(1 + rng.Below(8)));
        set_half(r * row_bytes + 2, 0.0008F * static_cast<float>(1 + rng.Below(8)));
      }
    }
  }
  return book;
}

SelectCase RandomSelect(Rng& rng, const std::string& what, core::GgmlType pred, core::GgmlType succ) {
  SelectCase c;
  c.what = what;
  c.pred_type = pred;
  c.succ_type = succ;
  c.anchor_token = static_cast<std::int32_t>(rng.Below(kVocab));
  c.logits = Normals(rng, std::size_t{kRows} * kVocab, 2.0);
  c.hproj = Normals(rng, std::size_t{kRows} * kRank, 0.5);
  c.pred = RandomBook(rng, pred);
  c.succ = RandomBook(rng, succ);
  return c;
}

/// Top-k membership around exact ties: in every row six tokens spread over different partial lists (and different
/// thread slots) get one logit value between the 28th and 29th largest, so the cut at 32 falls inside the tie.
void InjectLogitTies(SelectCase& c) {
  for (std::uint32_t r = 0; r < c.rows; ++r) {
    float* lg = &c.logits[std::size_t{r} * kVocab];
    std::vector<std::int32_t> idx(kVocab);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + 40, idx.end(), [&](std::int32_t x, std::int32_t y) { return lg[x] > lg[y]; });
    const float y = 0.5F * (lg[idx[27]] + lg[idx[28]]);
    Require(y < lg[idx[27]] && y > lg[idx[28]], "tie value between the 28th and 29th largest");
    std::vector<bool> top(kVocab, false);
    for (int i = 0; i < 40; ++i) top[idx[i]] = true;
    const std::uint32_t spots[6] = {3 + r, 2049 + 7 * r, 20000 + r, 40000 + 13 * r, 99999 - r, 100351 - 2 * r};
    for (std::uint32_t s : spots) {
      while (top[s]) ++s;  // never a token of the top 40 (the ranks above and below the tie stay what they are)
      lg[s] = y;
    }
  }
}

/// Row 0 of the case holds two candidates with different logits and exactly equal final scores: id 100 has logit 4 and a
/// transition term of exactly 1, id 5000 has logit 5 and a term of exactly 0; the lowest id (100) must win, whereas "the
/// first in logit order" would pick 5000. F32 codebooks make the terms exact on every implementation.
void InjectScoreTie(SelectCase& c) {
  c.exact_ties = true;
  c.pred_type = c.succ_type = core::GgmlType::kF32;
  c.pred.assign(std::size_t{kVocab} * kRank * 4, 0);
  c.succ.assign(std::size_t{kVocab} * kRank * 4, 0);
  const auto set = [](std::vector<std::uint8_t>& book, std::uint32_t row, std::uint32_t i, float x) {
    std::memcpy(book.data() + (std::size_t{row} * kRank + i) * 4, &x, 4);
  };
  c.anchor_token = 7;
  set(c.pred, 7, 0, 1.0F);
  for (std::uint32_t r = 0; r < c.rows; ++r) {
    std::fill(c.hproj.begin() + std::size_t{r} * kRank, c.hproj.begin() + std::size_t{r + 1} * kRank, 0.0F);
    c.hproj[std::size_t{r} * kRank] = 1.0F;
    float* lg = &c.logits[std::size_t{r} * kVocab];
    for (std::uint32_t i = 0; i < kVocab; ++i) lg[i] = -50.0F + 0.001F * static_cast<float>(i % 1000);
    lg[100] = 4.0F;
    lg[5000] = 5.0F;
    lg[77777] = -1.0F;
  }
  set(c.succ, 100, 0, 1.0F);  // term = pred[7][0] * hproj[0] * succ[100][0] = 1; every other successor row is 0
}

struct SelectStats {
  std::size_t rows_compared{0}, near_ties{0}, hard{0}, rows_ids_equal{0}, rows{0};
  double worst_score{0};
};

void CompareSelect(const SelectCase& c, const SelectOut& a, const SelectOut& b, SelectStats& st, Report& report) {
  // candidate ids: exactly equal in every row
  std::size_t ids_equal = 0;
  for (std::uint32_t r = 0; r < c.rows; ++r)
    ids_equal += std::equal(a.ids.begin() + r * kTopK, a.ids.begin() + (r + 1) * kTopK, b.ids.begin() + r * kTopK) ? 1 : 0;
  st.rows += c.rows;
  st.rows_ids_equal += ids_equal;
  // the bound of a score: 5e-6 * (|logit| + sum |terms|), the terms of B's chain
  std::vector<float> pred(kRank), succ(kRank);
  const auto row_of = [&](const std::vector<std::uint8_t>& book, core::GgmlType type, std::uint32_t id, float* out) {
    const std::size_t rb = type == core::GgmlType::kF32 ? std::size_t{kRank} * 4 : quant::QuantizedRowBytes(type, kRank);
    if (type == core::GgmlType::kF32) std::memcpy(out, book.data() + std::size_t{id} * rb, rb);
    else Require(quant::DequantizeRow(type, book.data() + std::size_t{id} * rb, out, kRank), "DequantizeRow");
  };
  const auto bound = [&](std::uint32_t r, std::int32_t prev, std::int32_t id) {
    row_of(c.pred, c.pred_type, static_cast<std::uint32_t>(prev), pred.data());
    row_of(c.succ, c.succ_type, static_cast<std::uint32_t>(id), succ.data());
    double sum = std::fabs(static_cast<double>(c.logits[std::size_t{r} * kVocab + static_cast<std::uint32_t>(id)]));
    for (std::uint32_t i = 0; i < kRank; ++i) sum += std::fabs(static_cast<double>(pred[i]) * c.hproj[std::size_t{r} * kRank + i] * succ[i]);
    return 5e-6 * sum;
  };
  std::size_t first_diff = c.rows;
  for (std::uint32_t r = 0; r < c.rows; ++r) {
    if (a.tokens[r] != b.tokens[r]) {
      first_diff = r;
      break;
    }
  }
  const std::uint32_t upto = std::min<std::size_t>(c.rows, first_diff + 1);  // rows whose predecessor chain is B's
  for (std::uint32_t r = 0; r < upto; ++r) {
    const std::int32_t prev = r == 0 ? c.anchor_token : b.tokens[r - 1];
    for (std::uint32_t k = 0; k < kTopK; ++k) {
      if (a.ids[r * kTopK + k] != b.ids[r * kTopK + k]) continue;  // reported by the ids check
      const double tol = bound(r, prev, b.ids[r * kTopK + k]);
      const double diff = std::fabs(static_cast<double>(a.scores[r * kTopK + k]) - b.scores[r * kTopK + k]);
      const double rel = diff / std::max(tol, 1e-30);
      if (!(rel <= st.worst_score)) st.worst_score = rel;
    }
  }
  st.rows_compared += upto;
  if (first_diff < c.rows) {
    const std::uint32_t r = static_cast<std::uint32_t>(first_diff);
    const std::int32_t prev = r == 0 ? c.anchor_token : b.tokens[r - 1];
    double sa = NAN, sb = NAN, tol = 0.0;
    for (std::uint32_t k = 0; k < kTopK; ++k) {
      if (b.ids[r * kTopK + k] == a.tokens[r]) {
        sa = b.scores[r * kTopK + k];
        tol += bound(r, prev, a.tokens[r]);
      }
      if (b.ids[r * kTopK + k] == b.tokens[r]) {
        sb = b.scores[r * kTopK + k];
        tol += bound(r, prev, b.tokens[r]);
      }
    }
    const bool near = !c.exact_ties && std::isfinite(sa) && std::isfinite(sb) && std::fabs(sa - sb) <= tol;
    if (near) {
      ++st.near_ties;
      std::printf("      near-tie, case '%s' row %u: A chose %d, B chose %d, B's scores %.9g and %.9g (gap %s, tolerance %s)\n", c.what.c_str(), r,
                  a.tokens[r], b.tokens[r], sa, sb, Sci(std::fabs(sa - sb)).c_str(), Sci(tol).c_str());
    } else {
      ++st.hard;
      std::printf("      DIFFERENT CHOICE, case '%s' row %u: A chose %d, B chose %d, B's scores %.9g and %.9g (tolerance %s)\n", c.what.c_str(), r,
                  a.tokens[r], b.tokens[r], sa, sb, Sci(tol).c_str());
    }
  }
  std::printf("  selector %s: candidate ids equal in %zu of %u rows, chosen tokens %s\n", c.what.c_str(), ids_equal, c.rows,
              first_diff == c.rows ? "all equal" : ("first differ at row " + std::to_string(first_diff)).c_str());
  (void)report;
}

void RunSelect(Side& a, Side& b, Rng& rng, Report& report) {
  const core::GgmlType q8 = core::GgmlType::kQ8_0, q4 = core::GgmlType::kQ4_K, f32 = core::GgmlType::kF32;
  SelectStats st;
  bool all_ok = true;
  const auto run_one = [&](SelectCase& c) {
    const auto oa = a.Select(c), ob = b.Select(c);
    const std::size_t before_hard = st.hard;
    CompareSelect(c, oa, ob, st, report);
    all_ok = all_ok && st.hard == before_hard;
    if (c.what.find("equal final scores") != std::string::npos) {
      // the designed tie: B resolves it by the rule, and so must A
      const bool b_rule = ob.tokens[0] == 100, a_rule = oa.tokens[0] == 100;
      std::printf("    designed score tie in row 0: lowest id (100) chosen by A: %s, by B: %s\n", a_rule ? "yes" : "NO", b_rule ? "yes" : "NO");
      all_ok = all_ok && a_rule && b_rule;
    }
  };
  {
    SelectCase c = RandomSelect(rng, "random logits, Q8_0 predecessor / Q4_K successor (the real files)", q8, q4);
    run_one(c);
  }
  {
    SelectCase c = RandomSelect(rng, "random logits, Q4_K predecessor / Q8_0 successor", q4, q8);
    run_one(c);
  }
  {
    SelectCase c = RandomSelect(rng, "random logits, F32 / F32 codebooks", f32, f32);
    run_one(c);
  }
  {
    SelectCase c = RandomSelect(rng, "six equal logits straddling the cut at 32 in every row", q8, q4);
    InjectLogitTies(c);
    run_one(c);
  }
  {
    SelectCase c = RandomSelect(rng, "equal final scores, different logits (lowest id must win)", f32, f32);
    InjectScoreTie(c);
    run_one(c);
  }
  const double share = st.rows_compared != 0 ? static_cast<double>(st.near_ties) / static_cast<double>(st.rows_compared) : 0.0;
  Verdict v;
  v.ok = all_ok && st.rows_ids_equal == st.rows && !Exceeds(st.worst_score, 1.0) && !Exceeds(share, 0.02);
  v.line = "  selector summary: " + std::to_string(st.rows_ids_equal) + " of " + std::to_string(st.rows) + " candidate lists equal, " +
           std::to_string(st.hard) + " different choices that are not near-ties, " + std::to_string(st.near_ties) + " near-ties among " +
           std::to_string(st.rows_compared) + " compared rows (share " + Sci(share) + ", limit 2e-2), worst candidate score error " +
           Sci(st.worst_score) + " of its bound" + (v.ok ? "" : "  -> FAILS");
  report.Add(v);
}

// ---- the 8-bit projection route (item 2) ---------------------------------------------------------------------------------------
/// Random encoded weight rows of Q4_K, Q8_0 or Q6_K (finite scales, elements of order 0.1 to 1).
std::vector<std::uint8_t> RandomMatrix(Rng& rng, core::GgmlType type, std::uint32_t m, std::uint32_t k) {
  const std::size_t row_bytes = quant::QuantizedRowBytes(type, k);
  Require(row_bytes != 0, "matrix type");
  std::vector<std::uint8_t> w(row_bytes * m);
  std::uint64_t bits = 0;
  for (std::size_t i = 0; i < w.size(); ++i) {
    if (i % 8 == 0) bits = rng.Next();
    w[i] = static_cast<std::uint8_t>(bits >> (8 * (i % 8)));
  }
  const auto set_half = [&](std::size_t at, float x) {
    const std::uint16_t h = HalfBits(x);
    w[at] = static_cast<std::uint8_t>(h & 0xFF);
    w[at + 1] = static_cast<std::uint8_t>(h >> 8);
  };
  for (std::uint32_t r = 0; r < m; ++r) {
    const std::size_t row = r * row_bytes;
    if (type == core::GgmlType::kQ8_0) {
      for (std::size_t b = 0; b < row_bytes; b += 34) set_half(row + b, 0.004F * static_cast<float>(1 + rng.Below(8)));
    } else if (type == core::GgmlType::kQ4_K) {
      for (std::size_t b = 0; b < row_bytes; b += 144) {
        set_half(row + b, 0.0008F * static_cast<float>(1 + rng.Below(8)));
        set_half(row + b + 2, 0.0008F * static_cast<float>(1 + rng.Below(8)));
      }
    } else {  // Q6_K: the scale d is the last two bytes of a 210-byte block
      for (std::size_t b = 0; b < row_bytes; b += 210) set_half(row + b + 208, 0.0002F * static_cast<float>(1 + rng.Below(8)));
    }
  }
  return w;
}

void RunProjectQ8(Side& a, Side& b, Rng& rng, Report& report, bool gpu) {
  struct Shape {
    const char* name;
    core::GgmlType type;
    std::uint32_t m, k;
  };
  const Shape shapes[] = {{"attention q, Q4_K 10240 x 5120", core::GgmlType::kQ4_K, 10240, kE},
                          {"K projection, Q8_0 1024 x 5120", core::GgmlType::kQ8_0, 1024, kE},
                          {"FFN down, Q4_K 5120 x 40960", core::GgmlType::kQ4_K, kE, kFF},
                          {"output head, Q6_K 100352 x 5120", core::GgmlType::kQ6_K, kVocab, kE}};
  const double tol = gpu ? 1.2e-2 : 1e-6;
  for (const Shape& sh : shapes) {
    ProjectCase c;
    c.type = sh.type;
    c.m = sh.m;
    c.k = sh.k;
    c.weights = RandomMatrix(rng, sh.type, sh.m, sh.k);
    double worst = 0.0;
    const bool head = sh.m == kVocab;
    for (const int rows : head ? std::vector<int>{15} : std::vector<int>{1, 3, 15, 16}) {
      c.rows = rows;
      c.what = std::string(sh.name) + ", " + std::to_string(rows) + " rows";
      c.x = Normals(rng, static_cast<std::size_t>(rows) * sh.k, 1.0);
      const auto oa = a.ProjectQ8(c), ob = b.ProjectQ8(c);
      for (int r = 0; r < rows; ++r) {
        double num = 0.0, den = 0.0;
        bool bad = false;
        for (std::uint32_t o = 0; o < sh.m; ++o) {
          const float g = oa[static_cast<std::size_t>(r) * sh.m + o], w = ob[static_cast<std::size_t>(r) * sh.m + o];
          if (!std::isfinite(g) || !std::isfinite(w)) {
            bad = true;
            break;
          }
          num += (static_cast<double>(g) - w) * (static_cast<double>(g) - w);
          den += static_cast<double>(w) * w;
        }
        const double rel = bad ? INFINITY : std::sqrt(num / std::max(den, 1e-30));
        if (!(rel <= worst)) worst = rel;
      }
    }
    report.Add(Judge(std::string("8-bit projection route, ") + sh.name + ", rows 1.." + (head ? "15" : "16"), worst, tol,
                     "relative RMSE per row"));
  }
}

// ---- the whole run --------------------------------------------------------------------------------------------------------------
void RunAll() {
  const Backend& be = TheBackend();
  const auto t_start = Clock::now();
  if (const char* kernel = std::getenv("RICURSIVE_TEST_ATTENTION")) {
    // The attention kernel by name (DrafterOptions::attention_kernel). Ops::DraftAttention takes no kernel yet: stage 2c adds the
    // argument and passes it in OpsSide::Attention; until then only "default" is accepted so that nothing is silently ignored.
    Require(std::string(kernel) == "default", std::string("RICURSIVE_TEST_ATTENTION=") + kernel + ": only 'default' exists");
    std::printf("  attention kernel: %s\n", kernel);
  }
  std::printf("  backend %s (A = %s, B = %s)\n", be.name.c_str(), be.gpu ? "the GPU operations" : "the host operations",
              be.gpu ? "the host operations" : "the naive implementation of this file");
  Rng rng(20261111);
  Rings rings;
  rings.k = Normals(rng, std::size_t{kFullCap} * kKVW, 1.0);
  rings.v = Normals(rng, std::size_t{kFullCap} * kKVW, 1.0);
  for (int i = 0; i < 2; ++i) {
    const KvType t = i == 0 ? KvType::kF32 : KvType::kF16;
    rings.kb[i] = RingBytes(rings.k, t);
    rings.vb[i] = RingBytes(rings.v, t);
  }
  std::unique_ptr<Side> a = std::make_unique<OpsSide>(be.gpu ? CreateGpuOps() : CreateHostOps());
  std::unique_ptr<Side> b = be.gpu ? std::unique_ptr<Side>(std::make_unique<OpsSide>(CreateHostOps())) : std::make_unique<NaiveSide>();
  const Tolerances tol = TolFor(be.gpu);
  std::vector<KvType> types;
  if (!be.gpu || !be.f16) types.push_back(KvType::kF32);
  if (!be.gpu || be.f16) types.push_back(KvType::kF16);

  Report report;
  for (const KvType type : types) {
    auto t0 = Clock::now();
    RunRope(*a, *b, rng, rings, type, report, tol);
    std::printf("  (rope+store, %s ring: %.1f s)\n", type == KvType::kF16 ? "F16" : "F32", Seconds(t0));
    t0 = Clock::now();
    RunAttention(*a, *b, rng, rings, type, report, tol);
    std::printf("  (attention, %s ring: %.1f s)\n", type == KvType::kF16 ? "F16" : "F32", Seconds(t0));
  }
  auto t0 = Clock::now();
  RunConvSilu(*a, *b, rng, report, tol);
  std::printf("  (convolution and silu: %.1f s)\n", Seconds(t0));
  t0 = Clock::now();
  RunSelect(*a, *b, rng, report);
  std::printf("  (selector: %.1f s)\n", Seconds(t0));
  t0 = Clock::now();
  RunProjectQ8(*a, *b, rng, report, be.gpu);
  std::printf("  (8-bit projection route: %.1f s)\n", Seconds(t0));

  if (!be.gpu) {
    // Twins: one wrong index in the naive side's ring arithmetic must fail the comparison.
    const KvType type = KvType::kF16;
    for (const bool in_attention : {true, false}) {
      Faults f;
      f.attention_slot = in_attention;
      f.rope_slot = !in_attention;
      NaiveSide twin(f);
      bool failed = false;
      std::string what;
      if (in_attention) {
        const AttnSpec s{"sliding, one past the ring's wrap", kSlideCap, kSlideWindow, kSlideCap + 1, 1};
        const AttnCase c = MakeAttn(rng, rings, type, s);
        failed = Exceeds(WorstRowRel(a->Attention(c), twin.Attention(c), kD), tol.attn_rel);
        what = "attention reads the slot of a context row past the wrap one too high";
      } else {
        const RopeCase c = MakeRope(rng, rings, type, "context 512 rows, 100 before the wrap", 512, 0, kSlideCap - 100, kSlideCap - 100, kSlideCap, true);
        failed = !CompareRope(c, a->Rope(c), twin.Rope(c), tol).ok;
        what = "the ring write puts rows past the wrap one slot too high";
      }
      std::printf("  twin '%s': %s\n", what.c_str(), failed ? "fails the comparison, as it must" : "PASSES (the test is too weak)");
      ++report.checks;
      if (!failed) ++report.failures;
    }
  }
  std::printf("  %zu checks, %zu failed, %.1f s in total\n", report.checks, report.failures, Seconds(t_start));
  Require(report.failures == 0, std::to_string(report.failures) + " of " + std::to_string(report.checks) + " checks failed");
}

}  // namespace

int main() {
  return ricursive_test::Run([] { RunAll(); });
}

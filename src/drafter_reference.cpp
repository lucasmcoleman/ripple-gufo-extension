#include "drafter_reference.hpp"

#include <algorithm>
#include <cmath>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace gufo::models::ricursive {
namespace {

// The helpers below repeat the few scalar primitives of reference.cpp (they
// live in its anonymous namespace; that file stays untouched).

float Dot(const float* a, const float* b, std::size_t k) {
  float acc[8] = {};
  std::size_t i = 0;
  for (; i + 8 <= k; i += 8) {
    for (int j = 0; j < 8; ++j) acc[j] += a[i + j] * b[i + j];
  }
  float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  for (; i < k; ++i) s += a[i] * b[i];
  return s;
}

/// y = weight * x / sqrt(mean(x^2) + eps), one row of `n` elements.
void RmsNormRow(const float* x, const float* weight, std::size_t n, float eps, float* y) {
  double ss = 0.0;
  for (std::size_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float scale = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(n) + eps));
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * scale * weight[i];
}

const float* F32Data(const TensorRef& t) {
  if (t.type != core::GgmlType::kF32) {
    throw std::runtime_error("drafter reference: " + std::string(t.name) + " must be F32");
  }
  return static_cast<const float*>(t.data);
}

void RmsNormRows(const std::vector<float>& x, std::size_t n, std::size_t width,
                 const TensorRef& weight, float eps, std::vector<float>& y) {
  const float* wt = F32Data(weight);
  y.resize(n * width);
  for (std::size_t t = 0; t < n; ++t) RmsNormRow(&x[t * width], wt, width, eps, &y[t * width]);
}

/// Per-head RMSNorm of [n][heads*dim] with a shared [dim] weight.
void HeadNorm(std::vector<float>& x, std::size_t n, std::size_t heads, std::size_t dim,
              const TensorRef& weight, float eps) {
  const float* wt = F32Data(weight);
  std::vector<float> tmp(dim);
  for (std::size_t t = 0; t < n; ++t) {
    for (std::size_t h = 0; h < heads; ++h) {
      float* p = &x[(t * heads + h) * dim];
      RmsNormRow(p, wt, dim, eps, tmp.data());
      std::copy(tmp.begin(), tmp.end(), p);
    }
  }
}

void Emit(const DrafterReference::StageFn& stage, std::string_view name, int layer,
          const std::vector<float>& d, std::size_t rows, std::size_t width) {
  if (stage) stage(name, layer, d.data(), rows, width);
}

float SiluOf(float g) {
  return g * static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(g))));
}

void DequantRow(const TensorRef& t, std::size_t row, float* out) {
  const auto* src = static_cast<const std::uint8_t*>(t.data) + row * t.RowBytes();
  if (!quant::DequantizeRow(t.type, src, out, t.cols)) {
    throw std::runtime_error("drafter reference: cannot dequantize " + std::string(t.name));
  }
}

}  // namespace

DrafterReference::DrafterReference(const DrafterWeights& drafter, const ModelWeights& target,
                                   DrafterFaults faults)
    : w_(drafter), target_(target), faults_(faults) {
  const auto& c = drafter.config;
  if (c.embedding_length != target.config.embedding_length ||
      target.token_embd.rows != drafter.selector_predecessor.rows ||
      target.output.rows != drafter.selector_predecessor.rows) {
    throw std::runtime_error("drafter reference: drafter and target disagree on hidden size or vocabulary");
  }
  k_.resize(c.block_count);
  v_.resize(c.block_count);
}

std::vector<std::uint32_t> DrafterReference::TapLayers() const {
  auto taps = w_.config.tap_layers;
  if (faults_.tap_layer_plus_one) {
    for (auto& t : taps) ++t;
  }
  return taps;
}

void DrafterReference::Reset() { ResetTo(0); }

void DrafterReference::ResetTo(std::size_t base) {
  base_ = base;
  rows_ = base;
  for (auto& k : k_) k.clear();
  for (auto& v : v_) v.clear();
}

void DrafterReference::TruncateTo(std::size_t position) {
  if (position >= rows_) return;
  if (position <= base_) {
    ResetTo(position);
    return;
  }
  const std::size_t kv = w_.config.KvWidth();
  for (auto& k : k_) k.resize((position - base_) * kv);
  for (auto& v : v_) v.resize((position - base_) * kv);
  rows_ = position;
}

bool DrafterReference::LayerHasRope(std::size_t layer) const noexcept {
  return w_.config.rope[layer] || faults_.rope_on_nope_layer;
}

/// y[t][r] = dot(W row r, x[t]); the row is dequantized once per call.
void DrafterReference::MatMul(const TensorRef& w, const float* x, std::size_t n, float* y) const {
  const std::size_t cols = w.cols;
  bool ok = true;
#pragma omp parallel
  {
    std::vector<float> row(cols);
#pragma omp for schedule(static)
    for (std::size_t r = 0; r < w.rows; ++r) {
      const auto* src = static_cast<const std::uint8_t*>(w.data) + r * w.RowBytes();
      if (!quant::DequantizeRow(w.type, src, row.data(), cols)) {
#pragma omp atomic write
        ok = false;
        continue;
      }
      for (std::size_t t = 0; t < n; ++t) y[t * w.rows + r] = Dot(row.data(), x + t * cols, cols);
    }
  }
  if (!ok) throw std::runtime_error("drafter reference: cannot dequantize " + std::string(w.name));
}

void DrafterReference::EmbedRows(std::span<const std::int32_t> tokens, std::vector<float>& out) const {
  const auto& te = target_.token_embd;
  const std::size_t E = te.cols;
  out.resize(tokens.size() * E);
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    if (tokens[t] < 0 || static_cast<std::uint64_t>(tokens[t]) >= te.rows) {
      throw std::runtime_error("drafter reference: token id " + std::to_string(tokens[t]) +
                               " out of range");
    }
    // The drafter reads the raw table rows: the target's embedding scale is NOT applied.
    DequantRow(te, static_cast<std::size_t>(tokens[t]), &out[t * E]);
  }
}

/// NeoX rotation of n rows of `heads` heads at positions first_position.. .
/// Frequencies as the oracle's torch code computes them, in float32:
/// inv = 1 / base^(2i/D), angle = float(position) * inv, cosf/sinf.
void DrafterReference::Rope(float* rows, std::size_t n, std::size_t first_position,
                            std::size_t heads) const {
  const auto& c = w_.config;
  const std::size_t D = c.head_dim, half = D / 2;
  std::vector<float> inv(half), cs(half), sn(half);
  for (std::size_t i = 0; i < half; ++i) {
    inv[i] = 1.0F / std::pow(c.rope_freq_base,
                             static_cast<float>(2 * i) / static_cast<float>(D));
  }
  for (std::size_t t = 0; t < n; ++t) {
    const float pos = static_cast<float>(first_position + t);
    for (std::size_t i = 0; i < half; ++i) {
      const float angle = pos * inv[i];
      cs[i] = std::cos(angle);
      sn[i] = std::sin(angle);
    }
    for (std::size_t h = 0; h < heads; ++h) {
      float* x = rows + (t * heads + h) * D;
      for (std::size_t i = 0; i < half; ++i) {
        const float x1 = x[i], x2 = x[i + half];
        x[i] = x1 * cs[i] - x2 * sn[i];
        x[i + half] = x2 * cs[i] + x1 * sn[i];
      }
    }
  }
}

/// Grouped dynamic causal convolution over the n rows of one block (never over the context):
/// out[t][c] = sum_o (base[side][o][c] + dyn[t][side][o][c / group]) * x[t - o][c], x[<0] = 0.
/// `dyn` is the conv_proj output [n][2*K*G]; accumulation order as the upstream code
/// (per tap: out += base*x, then out += dyn*x).
void DrafterReference::Conv(const float* x, std::size_t n, const TensorRef& base, std::size_t side,
                            const float* dyn, float* out) const {
  const auto& c = w_.config;
  const std::size_t E = c.embedding_length, K = c.conv_kernel_size, G = c.ConvGroups(),
                    gs = c.conv_group_size;
  const float* b = F32Data(base) + side * K * E;
  for (std::size_t t = 0; t < n; ++t) {
    float* o = out + t * E;
    std::fill(o, o + E, 0.0F);
    for (std::size_t tap = 0; tap < K; ++tap) {
      const std::size_t back = faults_.conv_taps_shifted ? tap + 1 : tap;
      if (t < back) continue;  // zero padding on the left
      const float* xs = x + (t - back) * E;
      const float* d = dyn + t * (2 * K * G) + side * K * G + tap * G;
      for (std::size_t ch = 0; ch < E; ++ch) {
        o[ch] += b[tap * E + ch] * xs[ch];
        o[ch] += d[ch / gs] * xs[ch];
      }
    }
  }
}

void DrafterReference::AppendContext(std::span<const float> features, std::size_t rows,
                                     const StageFn& stage) {
  const auto& c = w_.config;
  const std::size_t E = c.embedding_length, KV = c.KvWidth(), D = c.head_dim;
  if (features.size() != rows * c.FeatureWidth()) {
    throw std::invalid_argument("drafter reference: AppendContext expects rows x taps x hidden floats");
  }
  if (rows == 0) return;
  std::vector<float> proj(rows * E), ctx;
  MatMul(w_.fc, features.data(), rows, proj.data());
  RmsNormRows(proj, rows, E, w_.enc_output_norm, c.rms_eps, ctx);
  Emit(stage, "ctx_h", -1, ctx, rows, E);
  for (std::size_t il = 0; il < c.block_count; ++il) {
    const auto& L = w_.layers[il];
    std::vector<float> k(rows * KV), v(rows * KV);
    MatMul(L.attn_k, ctx.data(), rows, k.data());
    MatMul(L.attn_v, ctx.data(), rows, v.data());
    HeadNorm(k, rows, c.head_count_kv, D, L.attn_k_norm, c.rms_eps);
    if (LayerHasRope(il)) Rope(k.data(), rows, rows_, c.head_count_kv);
    Emit(stage, "ctx_k", static_cast<int>(il), k, rows, KV);
    Emit(stage, "ctx_v", static_cast<int>(il), v, rows, KV);
    k_[il].insert(k_[il].end(), k.begin(), k.end());
    v_[il].insert(v_[il].end(), v.begin(), v.end());
  }
  rows_ += rows;
}

DrafterReference::Draft DrafterReference::DraftStep(std::int32_t bonus_token,
                                                    std::size_t anchor_position,
                                                    const StageFn& stage) const {
  const auto& c = w_.config;
  const std::size_t E = c.embedding_length, D = c.head_dim, Q = c.QueryWidth(), KV = c.KvWidth(),
                    FF = c.feed_forward_length, B = c.block_size, NH = c.head_count,
                    NKV = c.head_count_kv, DYN = 2 * c.conv_kernel_size * c.ConvGroups();
  if (anchor_position > rows_) {
    throw std::invalid_argument("drafter reference: anchor position " +
                                std::to_string(anchor_position) + " is beyond the " +
                                std::to_string(rows_) + " context rows held");
  }
  const std::size_t s = anchor_position;
  // Rows below base_ were never given: they are not read, whatever the window says.
  const std::size_t hi = (faults_.context_one_row_short && s > base_) ? s - 1 : s;
  const std::size_t lo = std::min(
      std::max<std::size_t>((c.context_window != 0 && s > c.context_window) ? s - c.context_window : 0, base_), hi);
  const float eps = c.rms_eps;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));

  // Block input: the bonus token and MASK rows of the target's embedding table, unscaled.
  std::vector<std::int32_t> ids(B, static_cast<std::int32_t>(c.mask_token_id));
  ids[0] = bonus_token;
  std::vector<float> h;
  EmbedRows(ids, h);
  Emit(stage, "emb", -1, h, B, E);

  for (std::size_t il = 0; il < c.block_count; ++il) {
    const auto& L = w_.layers[il];
    const int ilayer = static_cast<int>(il);

    // ---- attention ---------------------------------------------------------------
    std::vector<float> a, dyn(B * DYN), x0(B * E);
    RmsNormRows(h, B, E, L.attn_norm, eps, a);
    Emit(stage, "attn_norm", ilayer, a, B, E);
    MatMul(L.attn_conv_proj, a.data(), B, dyn.data());
    Conv(a.data(), B, L.attn_conv_base, 0, dyn.data(), x0.data());
    Emit(stage, "attn_conv_in", ilayer, x0, B, E);

    std::vector<float> q(B * Q), kb(B * KV), vb(B * KV);
    MatMul(L.attn_q, x0.data(), B, q.data());
    MatMul(L.attn_k, x0.data(), B, kb.data());
    MatMul(L.attn_v, x0.data(), B, vb.data());
    HeadNorm(q, B, NH, D, L.attn_q_norm, eps);
    HeadNorm(kb, B, NKV, D, L.attn_k_norm, eps);
    if (LayerHasRope(il)) {
      Rope(q.data(), B, s, NH);
      Rope(kb.data(), B, s, NKV);
    }

    // Keys: context rows lo..s-1, then the block's own rows. Sliding layers see |q - k| < window.
    const bool sliding = c.sliding[il];
    const std::size_t nctx = hi - lo;
    std::vector<float> o(B * Q);
    for (std::size_t r = 0; r < B; ++r) {
      const std::size_t qpos = s + r;
      std::vector<std::size_t> keys;  // index into [context rows ; block rows]
      for (std::size_t j = 0; j < nctx + B; ++j) {
        const std::size_t kpos = lo + j;  // context row j sits at position lo + j; block row i at s + i
        const auto dist = qpos > kpos ? qpos - kpos : kpos - qpos;
        if (!sliding || dist < c.sliding_window) keys.push_back(j);
      }
      for (std::size_t hd = 0; hd < NH; ++hd) {
        const std::size_t kvh = hd / c.GqaGroup();
        const float* qh = &q[r * Q + hd * D];
        auto key = [&](std::size_t j) {
          return j < nctx ? &k_[il][(lo + j - base_) * KV + kvh * D] : &kb[(j - nctx) * KV + kvh * D];
        };
        auto val = [&](std::size_t j) {
          return j < nctx ? &v_[il][(lo + j - base_) * KV + kvh * D] : &vb[(j - nctx) * KV + kvh * D];
        };
        std::vector<float> sc(keys.size());
        float mx = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < keys.size(); ++i) {
          sc[i] = Dot(qh, key(keys[i]), D) * scale;
          mx = std::max(mx, sc[i]);
        }
        double sum = 0.0;
        std::vector<double> p(keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
          p[i] = std::exp(static_cast<double>(sc[i]) - mx);
          sum += p[i];
        }
        float* out = &o[r * Q + hd * D];
        std::fill(out, out + D, 0.0F);
        for (std::size_t i = 0; i < keys.size(); ++i) {
          const float pj = static_cast<float>(p[i] / sum);
          const float* vr = val(keys[i]);
          for (std::size_t d = 0; d < D; ++d) out[d] += pj * vr[d];
        }
      }
    }
    std::vector<float> attn(B * E), branch(B * E);
    MatMul(L.attn_output, o.data(), B, attn.data());
    Emit(stage, "attn_o", ilayer, attn, B, E);
    Conv(attn.data(), B, L.attn_conv_base, 1, dyn.data(), branch.data());
    for (std::size_t i = 0; i < h.size(); ++i) h[i] += branch[i];
    Emit(stage, "h_attn", ilayer, h, B, E);

    // ---- feed-forward ---------------------------------------------------------------
    std::vector<float> f, x1(B * E);
    RmsNormRows(h, B, E, L.ffn_norm, eps, f);
    Emit(stage, "ffn_norm", ilayer, f, B, E);
    MatMul(L.ffn_conv_proj, f.data(), B, dyn.data());
    Conv(f.data(), B, L.ffn_conv_base, 0, dyn.data(), x1.data());
    Emit(stage, "mlp_conv_in", ilayer, x1, B, E);
    std::vector<float> g(B * FF), u(B * FF);
    MatMul(L.ffn_gate, x1.data(), B, g.data());
    MatMul(L.ffn_up, x1.data(), B, u.data());
    for (std::size_t i = 0; i < g.size(); ++i) g[i] = SiluOf(g[i]) * u[i];  // no clamp
    std::vector<float> ff(B * E);
    MatMul(L.ffn_down, g.data(), B, ff.data());
    Emit(stage, "mlp_o", ilayer, ff, B, E);
    Conv(ff.data(), B, L.ffn_conv_base, 1, dyn.data(), branch.data());
    for (std::size_t i = 0; i < h.size(); ++i) h[i] += branch[i];
    Emit(stage, "h", ilayer, h, B, E);
  }

  std::vector<float> fin;
  RmsNormRows(h, B, E, w_.output_norm, eps, fin);
  Emit(stage, "final", -1, fin, B, E);

  // ---- logits of rows 1..B-1 with the target's output head (multiplier 1, no softcap) ------
  const std::size_t R = B - 1, V = target_.output.rows, KT = c.selector_top_k;
  std::vector<float> logits(R * V);
  MatMul(target_.output, fin.data() + E, R, logits.data());
  Emit(stage, "logits", -1, logits, R, V);
  std::vector<float> hproj(R * c.selector_rank);
  MatMul(w_.selector_hidden, fin.data() + E, R, hproj.data());
  Emit(stage, "hproj", -1, hproj, R, c.selector_rank);

  // ---- candidate selector: top-k per row, then the greedy left-to-right walk ---------------
  Draft out;
  out.context_first = lo;
  out.context_end = hi;
  out.candidates.resize(R * KT);
  out.scores.resize(R * KT);
  const std::size_t rank = c.selector_rank;
  std::vector<float> pred(rank), succ(rank);
  std::int32_t prev = bonus_token;
  for (std::size_t p = 0; p < R; ++p) {
    const float* lg = &logits[p * V];
    std::vector<std::int32_t> idx(V);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(KT), idx.end(),
                      [&](std::int32_t x, std::int32_t y) {
                        return lg[x] > lg[y] || (lg[x] == lg[y] && x < y);
                      });
    idx.resize(KT);
    std::sort(idx.begin(), idx.end());  // candidates in token-id order
    if (!faults_.selector_own_row) DequantRow(w_.selector_predecessor, static_cast<std::size_t>(prev), pred.data());
    std::size_t best = 0;
    for (std::size_t k = 0; k < KT; ++k) {
      if (faults_.selector_own_row) {
        DequantRow(w_.selector_predecessor, static_cast<std::size_t>(idx[k]), pred.data());
      }
      DequantRow(w_.selector_successor, static_cast<std::size_t>(idx[k]), succ.data());
      float dot = 0.0F;
      {
        float acc[8] = {};
        std::size_t i = 0;
        for (; i + 8 <= rank; i += 8) {
          for (int j = 0; j < 8; ++j) acc[j] += (pred[i + j] * hproj[p * rank + i + j]) * succ[i + j];
        }
        dot = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; i < rank; ++i) dot += (pred[i] * hproj[p * rank + i]) * succ[i];
      }
      const float score = lg[idx[k]] + dot;
      out.candidates[p * KT + k] = idx[k];
      out.scores[p * KT + k] = score;
      if (score > out.scores[p * KT + best]) best = k;
    }
    double mx = out.scores[p * KT + best], sum = 0.0;
    for (std::size_t k = 0; k < KT; ++k) sum += std::exp(static_cast<double>(out.scores[p * KT + k]) - mx);
    out.tokens.push_back(idx[best]);
    out.confidence.push_back(static_cast<float>(1.0 / sum));
    prev = idx[best];
  }
  return out;
}

}  // namespace gufo::models::ricursive

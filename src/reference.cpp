#include "reference.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <limits>
#include <stdexcept>
#include <string>

namespace gufo::models::ricursive {
namespace {

constexpr std::array<Variant, 13> kAllVariants = {
    Variant::kRopeFull,        Variant::kWinMinus1,      Variant::kWinPlus1,
    Variant::kVresOnVLayer0,   Variant::kVresOnVEquiv,   Variant::kVresSeedPrescale,
    Variant::kSymGateClamp,    Variant::kDropPostAttnNorm,
    Variant::kDropPostFfnNorm, Variant::kGateFromResid,  Variant::kSwapKV,
    Variant::kNonCausal,       Variant::kGqaModulo};

float Sigmoid(float x) {
  return static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(x))));
}

float Dot(const float* a, const float* b, std::size_t k) {
  float acc[8] = {};
  std::size_t i = 0;
  for (; i + 8 <= k; i += 8) {
    for (int j = 0; j < 8; ++j) acc[j] += a[i + j] * b[i + j];
  }
  float s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
            ((acc[4] + acc[5]) + (acc[6] + acc[7]));
  for (; i < k; ++i) s += a[i] * b[i];
  return s;
}

/// y = weight * x / sqrt(mean(x^2) + eps), one row of `n` elements.
void RmsNormRow(const float* x, const float* weight, std::size_t n, float eps,
                float* y) {
  double ss = 0.0;
  for (std::size_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const float scale =
      static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(n) + eps));
  for (std::size_t i = 0; i < n; ++i) y[i] = x[i] * scale * weight[i];
}

/// All F32 vectors (norm weights, lambda) are read straight from the mapping.
const float* F32Data(const TensorRef& t) {
  if (t.type != core::GgmlType::kF32) {
    throw std::runtime_error("ricursive reference: " + std::string(t.name) +
                             " must be F32");
  }
  return static_cast<const float*>(t.data);
}

void RmsNormRows(const std::vector<float>& x, std::size_t n, std::size_t width,
                 const TensorRef& weight, float eps, std::vector<float>& y) {
  const float* wt = F32Data(weight);
  y.resize(n * width);
  for (std::size_t t = 0; t < n; ++t) {
    RmsNormRow(&x[t * width], wt, width, eps, &y[t * width]);
  }
}

void Emit(const TapFn& tap, std::string_view name, int layer,
          const std::vector<float>& d, std::size_t rows, std::size_t width) {
  if (tap) tap(name, layer, d.data(), rows, width);
}

}  // namespace

std::string_view VariantName(Variant v) noexcept {
  switch (v) {
    case Variant::kNone: return "none";
    case Variant::kRopeFull: return "rope_full";
    case Variant::kWinMinus1: return "win_minus1";
    case Variant::kWinPlus1: return "win_plus1";
    case Variant::kVresOnVLayer0: return "vres_on_v_layer0";
    case Variant::kVresOnVEquiv: return "vres_on_v_equiv";
    case Variant::kVresSeedPrescale: return "vres_seed_prescale";
    case Variant::kSymGateClamp: return "sym_gate_clamp";
    case Variant::kDropPostAttnNorm: return "drop_post_attn_norm";
    case Variant::kDropPostFfnNorm: return "drop_post_ffn_norm";
    case Variant::kGateFromResid: return "gate_from_resid";
    case Variant::kSwapKV: return "swap_kv";
    case Variant::kNonCausal: return "non_causal";
    case Variant::kGqaModulo: return "gqa_modulo";
  }
  return "?";
}

std::optional<Variant> ParseVariant(std::string_view name) {
  if (name == "none") return Variant::kNone;
  for (const Variant v : kAllVariants) {
    if (VariantName(v) == name) return v;
  }
  return std::nullopt;
}

std::span<const Variant> AllVariants() noexcept { return kAllVariants; }

Reference::Reference(const ModelWeights& weights, Variant variant)
    : w_(weights), variant_(variant) {
  if (EffectiveWindow() == 0) {
    throw std::runtime_error("ricursive reference: effective window is 0");
  }
}

std::size_t Reference::EffectiveWindow() const noexcept {
  const std::size_t win = w_.config.sliding_window;
  if (variant_ == Variant::kWinMinus1) return win - 1;
  if (variant_ == Variant::kWinPlus1) return win + 1;
  return win;
}

/// y[t][r] = dot(W row r, x[t]); the row is dequantized once per call.
void Reference::MatMul(const TensorRef& w, const float* x, std::size_t n,
                       float* y) const {
  const std::size_t cols = w.cols;
  const std::size_t row_bytes = w.RowBytes();
  const auto* base = static_cast<const std::uint8_t*>(w.data);
  bool ok = true;
#pragma omp parallel
  {
    std::vector<float> row(cols);
#pragma omp for schedule(static)
    for (std::size_t r = 0; r < w.rows; ++r) {
      if (!quant::DequantizeRow(w.type, base + r * row_bytes, row.data(), cols)) {
#pragma omp atomic write
        ok = false;
        continue;
      }
      for (std::size_t t = 0; t < n; ++t) {
        y[t * w.rows + r] = Dot(row.data(), x + t * cols, cols);
      }
    }
  }
  if (!ok) {
    throw std::runtime_error("ricursive reference: cannot dequantize " +
                             std::string(w.name));
  }
}

std::vector<float> Reference::Embed(std::span<const std::int32_t> tokens,
                                    std::vector<float>* x0) const {
  const auto& c = w_.config;
  const std::size_t E = c.embedding_length;
  const auto& te = w_.token_embd;
  const std::size_t row_bytes = te.RowBytes();
  std::vector<float> raw(tokens.size() * E);
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    if (tokens[t] < 0 || static_cast<std::uint64_t>(tokens[t]) >= te.rows) {
      throw std::runtime_error("ricursive reference: token id " +
                               std::to_string(tokens[t]) + " out of range");
    }
    const auto* src = static_cast<const std::uint8_t*>(te.data) +
                      static_cast<std::size_t>(tokens[t]) * row_bytes;
    if (!quant::DequantizeRow(te.type, src, &raw[t * E], E)) {
      throw std::runtime_error("ricursive reference: cannot dequantize token_embd");
    }
  }
  std::vector<float> h(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) h[i] = raw[i] * c.embedding_scale;
  // x0 is the scaled embedding, the same for every layer.
  *x0 = variant_ == Variant::kVresSeedPrescale ? std::move(raw) : h;
  return h;
}

std::vector<float> Reference::Head(const std::vector<float>& h, std::size_t n,
                                   bool last_only, const TapFn& tap) const {
  const auto& c = w_.config;
  const std::size_t E = c.embedding_length;
  std::vector<float> hn;
  RmsNormRows(h, n, E, w_.output_norm, c.rms_eps, hn);
  Emit(tap, "result_norm", -1, hn, n, E);
  const std::size_t first = last_only ? n - 1 : 0;
  const std::size_t rows = n - first;
  std::vector<float> logits(rows * w_.output.rows);
  MatMul(w_.output, hn.data() + first * E, rows, logits.data());
  Emit(tap, "result_output", -1, logits, rows, w_.output.rows);
  return logits;
}

void Reference::AttendRow(
    const float* q, std::size_t lo, std::size_t hi,
    const std::function<const float*(std::size_t)>& key,
    const std::function<const float*(std::size_t)>& val, float* out) const {
  const auto& c = w_.config;
  const std::size_t D = c.head_dim;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(D)));
  const auto& score_rows = variant_ == Variant::kSwapKV ? val : key;
  const auto& mix_rows = variant_ == Variant::kSwapKV ? key : val;
  std::vector<float> s(hi - lo + 1);
  for (std::size_t h = 0; h < c.head_count; ++h) {
    const std::size_t kvh = variant_ == Variant::kGqaModulo
                                ? h % c.head_count_kv
                                : h / c.GqaGroup();
    const float* qh = q + h * D;
    float mx = -std::numeric_limits<float>::infinity();
    for (std::size_t j = lo; j <= hi; ++j) {
      s[j - lo] = Dot(qh, score_rows(j) + kvh * D, D) * scale;
      mx = std::max(mx, s[j - lo]);
    }
    double sum = 0.0;
    std::vector<double> p(s.size());
    for (std::size_t i = 0; i < s.size(); ++i) {
      p[i] = std::exp(static_cast<double>(s[i]) - mx);
      sum += p[i];
    }
    float* o = out + h * D;
    std::fill(o, o + D, 0.0F);
    for (std::size_t j = lo; j <= hi; ++j) {
      const float pj = static_cast<float>(p[j - lo] / sum);
      const float* vr = mix_rows(j) + kvh * D;
      for (std::size_t d = 0; d < D; ++d) o[d] += pj * vr[d];
    }
  }
}

void Reference::Layer(
    std::size_t il, std::size_t n, std::size_t p0, std::vector<float>& h,
    const std::vector<float>& x0, std::vector<float>& v_layer0,
    const std::function<void(const float*, const float*, const float*, float*)>&
        attend,
    const TapFn& tap) const {
  const auto& c = w_.config;
  const auto& L = w_.layers[il];
  const std::size_t E = c.embedding_length, D = c.head_dim,
                    Q = c.QueryWidth(), KV = c.KvWidth(),
                    FF = c.feed_forward_length;
  const int ilayer = static_cast<int>(il);
  const float eps = c.rms_eps;

  std::vector<float> a;
  RmsNormRows(h, n, E, L.attn_norm, eps, a);
  Emit(tap, "attn_norm", ilayer, a, n, E);

  std::vector<float> q(n * Q), k(n * KV), v(n * KV);
  MatMul(L.attn_q, a.data(), n, q.data());
  MatMul(L.attn_k, a.data(), n, k.data());
  const float wv = Sigmoid(F32Data(L.attn_vres_lambda)[0]);
  if (variant_ == Variant::kVresOnVLayer0 || variant_ == Variant::kVresOnVEquiv) {
    // Wrong (layer 0) or equivalent (same layer) mix, done on V outputs.
    MatMul(L.attn_v, a.data(), n, v.data());
    std::vector<float> ref(n * KV);
    if (variant_ == Variant::kVresOnVEquiv) {
      MatMul(L.attn_v, x0.data(), n, ref.data());
    } else {
      if (v_layer0.empty()) v_layer0 = v;
      ref = v_layer0;
    }
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = ref[i] + wv * (v[i] - ref[i]);
  } else {
    // Spec: lerp(x0, a, w) on the INPUT of attn_v.
    std::vector<float> vin(n * E);
    for (std::size_t i = 0; i < vin.size(); ++i) {
      vin[i] = x0[i] + wv * (a[i] - x0[i]);
    }
    Emit(tap, "vres_in", ilayer, vin, n, E);
    MatMul(L.attn_v, vin.data(), n, v.data());
  }

  // Per-head q/k RMSNorm; the [D] weights are shared by all heads.
  {
    const float* qn = F32Data(L.attn_q_norm);
    const float* kn = F32Data(L.attn_k_norm);
    std::vector<float> tmp(D);
    for (std::size_t t = 0; t < n; ++t) {
      for (std::size_t hh = 0; hh < c.head_count; ++hh) {
        float* x = &q[t * Q + hh * D];
        RmsNormRow(x, qn, D, eps, tmp.data());
        std::copy(tmp.begin(), tmp.end(), x);
      }
      for (std::size_t hh = 0; hh < c.head_count_kv; ++hh) {
        float* x = &k[t * KV + hh * D];
        RmsNormRow(x, kn, D, eps, tmp.data());
        std::copy(tmp.begin(), tmp.end(), x);
      }
    }
  }
  Emit(tap, "Qcur_normed", ilayer, q, n, Q);
  Emit(tap, "Kcur_normed", ilayer, k, n, KV);

  // NeoX RoPE (pairs i, i + D/2), sliding layers only.
  if (c.HasRope(il) || variant_ == Variant::kRopeFull) {
    const std::size_t half = D / 2;
    std::vector<float> cs(half), sn(half);
    for (std::size_t t = 0; t < n; ++t) {
      const double pos = static_cast<double>(p0 + t);
      for (std::size_t i = 0; i < half; ++i) {
        const double inv = std::pow(static_cast<double>(c.rope_freq_base),
                                    -2.0 * static_cast<double>(i) /
                                        static_cast<double>(D));
        cs[i] = static_cast<float>(std::cos(pos * inv));
        sn[i] = static_cast<float>(std::sin(pos * inv));
      }
      auto rope = [&](float* x) {
        for (std::size_t i = 0; i < half; ++i) {
          const float x1 = x[i], x2 = x[i + half];
          x[i] = x1 * cs[i] - x2 * sn[i];
          x[i + half] = x2 * cs[i] + x1 * sn[i];
        }
      };
      for (std::size_t hh = 0; hh < c.head_count; ++hh) rope(&q[t * Q + hh * D]);
      for (std::size_t hh = 0; hh < c.head_count_kv; ++hh) rope(&k[t * KV + hh * D]);
    }
  }
  Emit(tap, "Qcur", ilayer, q, n, Q);
  Emit(tap, "Kcur", ilayer, k, n, KV);
  Emit(tap, "Vcur", ilayer, v, n, KV);

  std::vector<float> o(n * Q);
  attend(q.data(), k.data(), v.data(), o.data());
  Emit(tap, "attn_out", ilayer, o, n, Q);

  // Gate: sigmoid(attn_gate(a)), elementwise over H*D, before attn_output.
  std::vector<float> gate(n * Q);
  MatMul(L.attn_gate, variant_ == Variant::kGateFromResid ? h.data() : a.data(),
         n, gate.data());
  for (std::size_t i = 0; i < gate.size(); ++i) {
    gate[i] = Sigmoid(gate[i]);
    o[i] *= gate[i];
  }
  Emit(tap, "attn_gate", ilayer, gate, n, Q);
  Emit(tap, "attn_gated", ilayer, o, n, Q);

  std::vector<float> attn(n * E);
  MatMul(L.attn_output, o.data(), n, attn.data());
  if (variant_ != Variant::kDropPostAttnNorm) {
    std::vector<float> normed;
    RmsNormRows(attn, n, E, L.post_attention_norm, eps, normed);
    attn = std::move(normed);
  }
  Emit(tap, "attn_post_norm", ilayer, attn, n, E);
  for (std::size_t i = 0; i < h.size(); ++i) h[i] += attn[i];
  Emit(tap, "ffn_inp", ilayer, h, n, E);

  std::vector<float> f;
  RmsNormRows(h, n, E, L.ffn_norm, eps, f);
  Emit(tap, "ffn_norm", ilayer, f, n, E);
  std::vector<float> g(n * FF), u(n * FF);
  MatMul(L.ffn_gate, f.data(), n, g.data());
  MatMul(L.ffn_up, f.data(), n, u.data());
  const float clamp = c.swiglu_clamp[il];
  for (std::size_t i = 0; i < g.size(); ++i) {
    float gi = g[i], ui = u[i];
    if (clamp > 0.0F) {
      gi = variant_ == Variant::kSymGateClamp ? std::clamp(gi, -clamp, clamp)
                                              : std::min(gi, clamp);
      ui = std::clamp(ui, -clamp, clamp);
    }
    g[i] = gi * Sigmoid(gi) * ui;  // silu(gate) * up
  }
  Emit(tap, "ffn_swiglu", ilayer, g, n, FF);
  std::vector<float> ff(n * E);
  MatMul(L.ffn_down, g.data(), n, ff.data());
  Emit(tap, "ffn_out", ilayer, ff, n, E);
  if (variant_ != Variant::kDropPostFfnNorm) {
    std::vector<float> normed;
    RmsNormRows(ff, n, E, L.post_ffw_norm, eps, normed);
    ff = std::move(normed);
  }
  Emit(tap, "ffn_post_norm", ilayer, ff, n, E);
  for (std::size_t i = 0; i < h.size(); ++i) h[i] += ff[i];
  Emit(tap, "l_out", ilayer, h, n, E);
}

std::vector<float> Reference::ForwardFull(std::span<const std::int32_t> tokens,
                                          const TapFn& tap) const {
  const auto& c = w_.config;
  const std::size_t T = tokens.size();
  if (T == 0) return {};
  const std::size_t Q = c.QueryWidth(), KV = c.KvWidth();
  std::vector<float> x0;
  std::vector<float> h = Embed(tokens, &x0);
  Emit(tap, "embd_scaled", -1, h, T, c.embedding_length);
  std::vector<float> v_layer0;
  const std::size_t win = EffectiveWindow();
  for (std::size_t il = 0; il < c.block_count; ++il) {
    const bool swa = c.IsSliding(il);
    auto attend = [&](const float* q, const float* k, const float* v, float* o) {
      for (std::size_t t = 0; t < T; ++t) {
        const std::size_t hi = variant_ == Variant::kNonCausal ? T - 1 : t;
        const std::size_t lo = swa && t + 1 > win ? t + 1 - win : 0;
        AttendRow(
            q + t * Q, lo, hi, [&](std::size_t j) { return k + j * KV; },
            [&](std::size_t j) { return v + j * KV; }, o + t * Q);
      }
    };
    Layer(il, T, 0, h, x0, v_layer0, attend, tap);
  }
  return Head(h, T, false, tap);
}

Reference::Session::Session(const Reference& reference, std::size_t max_rewind,
                            std::size_t ring_rows_for_test)
    : ref_(reference), max_rewind_(max_rewind) {
  if (reference.variant_ == Variant::kNonCausal) {
    throw std::runtime_error("ricursive reference: non_causal needs ForwardFull");
  }
  const auto& c = reference.w_.config;
  cache_.resize(c.block_count);
  for (std::size_t il = 0; il < c.block_count; ++il) {
    cache_[il].ring = c.IsSliding(il);
    cache_[il].capacity =
        !cache_[il].ring ? 0
        : ring_rows_for_test != 0 ? ring_rows_for_test
                                  : reference.EffectiveWindow() + max_rewind;
    if (cache_[il].ring) {
      cache_[il].k.assign(cache_[il].capacity * c.KvWidth(), 0.0F);
      cache_[il].v.assign(cache_[il].capacity * c.KvWidth(), 0.0F);
    }
  }
}

std::size_t Reference::Session::CacheRows(std::size_t il) const {
  const auto& l = cache_.at(il);
  return l.ring ? l.capacity : l.k.size() / ref_.w_.config.KvWidth();
}

void Reference::Session::Rewind(std::size_t pos) {
  if (pos > pos_) throw std::runtime_error("ricursive reference: rewind past the end");
  if (high_water_ - pos > max_rewind_) {
    throw std::runtime_error(
        "ricursive reference: rewind to " + std::to_string(pos) +
        " needs ring rows that were overwritten (high water " +
        std::to_string(high_water_) + ", max_rewind " +
        std::to_string(max_rewind_) + ")");
  }
  pos_ = pos;
}

std::vector<float> Reference::Session::Append(
    std::span<const std::int32_t> tokens, bool last_only, const TapFn& tap) {
  const auto& c = ref_.w_.config;
  const std::size_t n = tokens.size();
  if (n == 0) throw std::runtime_error("ricursive reference: empty chunk");
  const std::size_t Q = c.QueryWidth(), KV = c.KvWidth();
  const std::size_t p0 = pos_;
  std::vector<float> x0;
  std::vector<float> h = ref_.Embed(tokens, &x0);
  Emit(tap, "embd_scaled", -1, h, n, c.embedding_length);
  std::vector<float> v_layer0;
  const std::size_t win = ref_.EffectiveWindow();
  for (std::size_t il = 0; il < c.block_count; ++il) {
    LayerCache& lc = cache_[il];
    if (!lc.ring) {
      lc.k.resize(std::max(lc.k.size(), (p0 + n) * KV));
      lc.v.resize(std::max(lc.v.size(), (p0 + n) * KV));
    }
    auto slot = [&](std::size_t j) { return lc.ring ? j % lc.capacity : j; };
    auto attend = [&](const float* q, const float* k, const float* v, float* o) {
      for (std::size_t t = 0; t < n; ++t) {
        const std::size_t pos = p0 + t;
        // Write this token's row, then attend over the visible rows.
        std::copy(k + t * KV, k + (t + 1) * KV, &lc.k[slot(pos) * KV]);
        std::copy(v + t * KV, v + (t + 1) * KV, &lc.v[slot(pos) * KV]);
        const std::size_t lo = lc.ring && pos + 1 > win ? pos + 1 - win : 0;
        ref_.AttendRow(
            q + t * Q, lo, pos,
            [&](std::size_t j) { return &lc.k[slot(j) * KV]; },
            [&](std::size_t j) { return &lc.v[slot(j) * KV]; }, o + t * Q);
      }
    };
    ref_.Layer(il, n, p0, h, x0, v_layer0, attend, tap);
  }
  pos_ = p0 + n;
  high_water_ = std::max(high_water_, pos_);
  return ref_.Head(h, n, last_only, tap);
}

}  // namespace gufo::models::ricursive

// GPU checks of QkNormRopeAppend and Attention128 against FP64 references,
// with deliberately wrong twins that must be rejected by at least 100x.
//
// usage: attention_ops_test [small | real-sliding-f16 | real-sliding-f32 |
//                            real-full-f16 | real-full-f32]
// Each mode runs in well under 20 s so the GPU admission lock is held briefly.
// Exit 77 when there is no GPU.
//
// What one configuration (heads, kv_heads, window, T positions, cache type)
// does:
//  1. Random raw q/k/v projections (per-token scale varies), random norm weights.
//  2. One-shot run: a single QkNormRopeAppend + Attention128 launch pair over
//     all T rows into a cache big enough to keep every key (capacity
//     MinCapacity). Its stored K/V and post-norm q are read back and compared
//     with an FP64 pipeline (norm, RoPE via cos/sin tables computed in long
//     double, F16 rounding allowance); the attention output is compared with an
//     FP64 softmax over the keys the GPU actually stored (so an fp16 cache is
//     judged against the same F16-rounded K/V) on every row (small) or on
//     sampled rows around the window/capacity boundaries (real shape).
//  3. Twins on the same data (see RequireTwinFails).
//  4. Other launch plans, each compared BIT FOR BIT with the one-shot run:
//     ring caches of capacity window + chunk - 1 (so positions wrap many
//     times) with fixed chunks, random chunks, a chunk straddling the window,
//     prefill followed by single decode rows, and single rows into a ring of
//     exactly `window` rows.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "hip/kernels.hpp"
#include "tests/check.hpp"

namespace k = gufo::models::ricursive::rocm;
using namespace ricursive_test;

namespace {

constexpr std::uint32_t D = 128;
constexpr float kEps = 1e-6F;
constexpr float kTheta = 10000.0F;
constexpr double kAttnTol = 2e-5;   // brief: about 2e-5 against FP64
constexpr double kPostTol = 2e-6;   // q/K after norm+RoPE, relative to the head's largest element

hipStream_t g_stream = nullptr;

struct Cfg {
  std::string name;
  std::uint32_t heads, kv, window, T;
  k::KvType type;
  std::uint32_t rotary;      // 128 on sliding layers, 0 on full layers
  std::uint32_t ring_chunk;  // largest chunk of the ring plans
  bool sampled;              // check sampled rows instead of all
  std::uint64_t seed;
};

struct Data {
  std::vector<float> q, k, v, gq, gk;
};

Data Generate(const Cfg& c) {
  Rng rng(c.seed);
  Data d;
  d.q.resize(std::size_t{c.T} * c.heads * D);
  d.k.resize(std::size_t{c.T} * c.kv * D);
  d.v.resize(d.k.size());
  auto fill = [&](std::vector<float>& v, std::size_t per_token, double sigma_log) {
    for (std::size_t t = 0; t < c.T; ++t) {
      const double s = std::exp(sigma_log * rng.Normal());  // the norm must remove this scale
      for (std::size_t i = 0; i < per_token; ++i) v[t * per_token + i] = static_cast<float>(s * rng.Normal());
    }
  };
  fill(d.q, std::size_t{c.heads} * D, 0.5);
  fill(d.k, std::size_t{c.kv} * D, 0.5);
  fill(d.v, std::size_t{c.kv} * D, 0.3);
  d.gq.resize(D);
  d.gk.resize(D);
  for (auto& g : d.gq) g = static_cast<float>(0.5 + rng.Uniform());
  for (auto& g : d.gk) g = static_cast<float>(0.5 + rng.Uniform());
  return d;
}

// ---- FP64 references ----------------------------------------------------------------

/// cos/sin of pos * theta^(-2 pair / 128), in long double, [pos][pair][2].
std::vector<double> RopeTable(std::uint32_t T) {
  std::vector<double> t(std::size_t{T} * 64 * 2);
  for (std::uint32_t p = 0; p < T; ++p) {
    for (std::uint32_t i = 0; i < 64; ++i) {
      const long double ang = static_cast<long double>(p) * powl(10000.0L, -2.0L * i / 128.0L);
      t[(std::size_t{p} * 64 + i) * 2] = static_cast<double>(cosl(ang));
      t[(std::size_t{p} * 64 + i) * 2 + 1] = static_cast<double>(sinl(ang));
    }
  }
  return t;
}

/// Per-head RMSNorm with weight gamma, then NeoX RoPE when `rotary` (np_ref's
/// rotate_half form). Returns the largest |normed| element of the head.
double RefNormRope(const float* x, const std::vector<float>& gamma, std::uint32_t pos, bool rotary,
                   const std::vector<double>& table, double* out) {
  double ss = 0;
  for (std::uint32_t d = 0; d < D; ++d) ss += double{x[d]} * x[d];
  const double inv = 1.0 / std::sqrt(ss / D + double{kEps});
  double n[D];
  double headmax = 0;
  for (std::uint32_t d = 0; d < D; ++d) {
    n[d] = x[d] * inv * gamma[d];
    headmax = std::max(headmax, std::fabs(n[d]));
  }
  for (std::uint32_t d = 0; d < D; ++d) {
    if (!rotary) {
      out[d] = n[d];
      continue;
    }
    const std::uint32_t i = d % 64;
    const double c = table[(std::size_t{pos} * 64 + i) * 2];
    const double s = table[(std::size_t{pos} * 64 + i) * 2 + 1];
    out[d] = d < 64 ? n[d] * c - n[d + 64] * s : n[d - 64] * s + n[d] * c;
  }
  return headmax;
}

double Ulp16(double a) {
  a = std::fabs(a);
  if (a < std::ldexp(1.0, -14)) return std::ldexp(1.0, -24);
  int e;
  std::frexp(a, &e);
  return std::ldexp(1.0, e - 11);
}

float RoundHalf(float v) { return static_cast<float>(static_cast<_Float16>(v)); }
float HalfBitsToFloat(std::uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, 2);
  return static_cast<float>(h);
}

struct Var {
  int win_delta = 0;       // window twin: +-1
  int force_window = -1;   // >= 0: use this window whatever the layer's
  bool causal = true;
  bool swap_kv = false;
  bool gqa_mod = false;
  double scale = 1.0 / std::sqrt(128.0);
};

/// FP64 attention of one row at position p over the stored keys (slot == position).
void RefAttnRow(const float* qrow, const std::vector<float>& Kst, const std::vector<float>& Vst,
                std::uint32_t H, std::uint32_t KV, std::uint32_t p, std::uint32_t window, std::uint32_t T,
                const Var& var, std::vector<double>& out) {
  const std::uint32_t group = H / KV;
  std::int64_t win = var.force_window >= 0 ? var.force_window : (window != 0 ? window + var.win_delta : 0);
  const std::int64_t lo = win != 0 ? std::max<std::int64_t>(0, std::int64_t{p} - win + 1) : 0;
  const std::int64_t hi = var.causal ? p : T - 1;
  const auto& Kx = var.swap_kv ? Vst : Kst;
  const auto& Vx = var.swap_kv ? Kst : Vst;
  std::vector<double> s(static_cast<std::size_t>(hi - lo + 1));
  out.assign(std::size_t{H} * D, 0.0);
  for (std::uint32_t h = 0; h < H; ++h) {
    const std::uint32_t kvh = var.gqa_mod ? h % KV : h / group;
    double m = -1e300;
    for (std::int64_t j = lo; j <= hi; ++j) {
      const float* kr = &Kx[(std::size_t(j) * KV + kvh) * D];
      double a = 0;
      for (std::uint32_t d = 0; d < D; ++d) a += double{qrow[std::size_t{h} * D + d]} * kr[d];
      s[j - lo] = a * var.scale;
      m = std::max(m, s[j - lo]);
    }
    double z = 0;
    for (auto& v : s) {
      v = std::exp(v - m);
      z += v;
    }
    double* o = &out[std::size_t{h} * D];
    for (std::int64_t j = lo; j <= hi; ++j) {
      const float* vr = &Vx[(std::size_t(j) * KV + kvh) * D];
      const double w = s[j - lo] / z;
      for (std::uint32_t d = 0; d < D; ++d) o[d] += w * vr[d];
    }
  }
}

// ---- GPU runs --------------------------------------------------------------------------

struct Result {
  std::vector<float> out, q;
  std::vector<std::uint8_t> kc, vc;  // only when keep_cache
};

using Chunks = std::vector<std::pair<std::uint32_t, std::uint32_t>>;  // (start, rows)

std::size_t ElemBytes(k::KvType t) { return t == k::KvType::kF16 ? 2 : 4; }

Result RunPlan(const Cfg& c, const Data& d, std::uint32_t capacity, const Chunks& chunks, bool keep_cache,
               std::uint32_t rotary) {
  DevBuf<float> q(d.q.size()), kk(d.k.size()), vv(d.v.size()), gq(D), gk(D), out(d.q.size());
  q.Upload(d.q);
  kk.Upload(d.k);
  vv.Upload(d.v);
  gq.Upload(d.gq);
  gk.Upload(d.gk);
  const std::size_t cache_bytes = std::size_t{capacity} * c.kv * D * ElemBytes(c.type);
  DevBuf<std::uint8_t> kc(cache_bytes), vc(cache_bytes);
  kc.FillBytes(0xFF);  // NaN in both fp32 and fp16: an unwritten slot read poisons the output
  vc.FillBytes(0xFF);
  out.FillBytes(0xFF);
  const float scale = 1.0F / std::sqrt(128.0F);
  std::string err;
  for (auto [start, n] : chunks) {
    Require(k::QkNormRopeAppend(q.get() + std::size_t{start} * c.heads * D, kk.get() + std::size_t{start} * c.kv * D,
                                vv.get() + std::size_t{start} * c.kv * D, gq.get(), gk.get(), kc.get(), vc.get(),
                                c.type, n, c.heads, c.kv, kEps, start, capacity, rotary, kTheta, g_stream, &err),
            "QkNormRopeAppend: " + err);
    Require(k::Attention128(q.get() + std::size_t{start} * c.heads * D, kc.get(), vc.get(), c.type,
                            out.get() + std::size_t{start} * c.heads * D, n, c.heads, c.kv, c.window, start,
                            capacity, scale, g_stream, &err),
            "Attention128: " + err);
  }
  HipCheck(hipStreamSynchronize(g_stream), "plan sync");
  Result r;
  r.out = out.Download();
  r.q = q.Download();
  if (keep_cache) {
    r.kc = kc.Download();
    r.vc = vc.Download();
  }
  return r;
}

bool BitEqual(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

void Upd(double& m, double e) {
  if (std::isnan(e)) e = std::numeric_limits<double>::infinity();
  m = std::max(m, e);
}

std::vector<float> CacheToFloat(const std::vector<std::uint8_t>& bytes, k::KvType type, std::size_t elems) {
  std::vector<float> f(elems);
  if (type == k::KvType::kF16) {
    for (std::size_t i = 0; i < elems; ++i) {
      std::uint16_t b;
      std::memcpy(&b, &bytes[i * 2], 2);
      f[i] = HalfBitsToFloat(b);
    }
  } else {
    std::memcpy(f.data(), bytes.data(), elems * 4);
  }
  return f;
}

Chunks Fixed(std::uint32_t T, std::uint32_t n) {
  Chunks c;
  for (std::uint32_t s = 0; s < T; s += n) c.push_back({s, std::min(n, T - s)});
  return c;
}

Chunks Random(std::uint32_t T, std::uint32_t nmax, Rng& rng) {
  Chunks c;
  for (std::uint32_t s = 0; s < T;) {
    const std::uint32_t n = std::min(T - s, rng.Below(4) == 0 ? 1U : 1 + rng.Below(nmax));
    c.push_back({s, n});
    s += n;
  }
  return c;
}

std::vector<std::uint32_t> SampleRows(const Cfg& c, Rng& rng) {
  std::set<std::uint32_t> rows;
  if (!c.sampled) {
    for (std::uint32_t p = 0; p < c.T; ++p) rows.insert(p);
    return {rows.begin(), rows.end()};
  }
  const std::uint32_t W = c.window;
  std::vector<std::uint32_t> special = {0, 1, 2, 127, 128, 129, 1023, 1024, 1025, 2047, 2048, 2049, c.T - 2, c.T - 1};
  if (W != 0) {
    for (std::uint32_t base : {W, 2 * W, 3 * W, W + c.ring_chunk - 1, 2 * (W + c.ring_chunk - 1)}) {
      special.push_back(base - 1);
      special.push_back(base);
      special.push_back(base + 1);
    }
  }
  for (auto p : special) {
    if (p < c.T) rows.insert(p);
  }
  for (int i = 0; i < 24; ++i) rows.insert(rng.Below(c.T));
  return {rows.begin(), rows.end()};
}

struct Check {
  const Cfg& c;
  const Data& d;
  const Result& p0;
  std::vector<float> Kst, Vst;
};

/// Attention output of the one-shot run against the FP64 reference variant on `rows`.
Err CompareRows(const Check& ck, const std::vector<std::uint32_t>& rows, const Var& var) {
  Err e;
  std::vector<double> ref;
  for (auto p : rows) {
    RefAttnRow(&ck.p0.q[std::size_t{p} * ck.c.heads * D], ck.Kst, ck.Vst, ck.c.heads, ck.c.kv, p, ck.c.window, ck.c.T,
               var, ref);
    for (std::size_t i = 0; i < ref.size(); ++i) e.Add(ck.p0.out[std::size_t{p} * ck.c.heads * D + i], ref[i]);
  }
  return e;
}

void RunConfig(const Cfg& c) {
  const auto t0 = std::chrono::steady_clock::now();
  std::cout << c.name << ": heads " << c.heads << "/" << c.kv << " (group " << c.heads / c.kv << "), window "
            << c.window << ", T " << c.T << ", cache " << (c.type == k::KvType::kF16 ? "F16" : "F32") << ", rotary "
            << c.rotary << "\n";
  const Data d = Generate(c);
  const std::uint32_t W = c.window;
  const std::uint32_t cap0 = static_cast<std::uint32_t>(k::MinCapacity(0, c.T, W));
  Require(cap0 >= c.T, "one-shot capacity must keep every position");
  const Result p0 = RunPlan(c, d, cap0, {{0, c.T}}, true, c.rotary);
  const std::size_t kv_elems = std::size_t{cap0} * c.kv * D;
  Check ck{c, d, p0, CacheToFloat(p0.kc, c.type, kv_elems), CacheToFloat(p0.vc, c.type, kv_elems)};
  const auto table = RopeTable(c.T);
  Rng rng(c.seed ^ 0xABCDEF);
  const auto rows = SampleRows(c, rng);

  // Slots after the last position were never written.
  {
    const std::size_t from = std::size_t{c.T} * c.kv * D * ElemBytes(c.type);
    for (std::size_t i = from; i < p0.kc.size(); ++i) {
      Require(p0.kc[i] == 0xFF && p0.vc[i] == 0xFF, "a cache slot beyond the last position was written");
    }
  }

  // ---- K and V as stored, q as normed and rotated ---------------------------------
  {
    const bool f16 = c.type == k::KvType::kF16;
    double kq_good = 0, kq_twin = 0, k_raw = 0, q_good = 0, q_twin = 0, q_raw = 0;
    double ref[D], twin[D];
    for (std::uint32_t p = 0; p < c.T; ++p) {
      for (std::uint32_t h = 0; h < c.kv; ++h) {
        const float* x = &d.k[(std::size_t{p} * c.kv + h) * D];
        const double hm = RefNormRope(x, d.gk, p, c.rotary != 0, table, ref);
        RefNormRope(x, d.gk, p, c.rotary == 0, table, twin);
        for (std::uint32_t i = 0; i < D; ++i) {
          const double g = ck.Kst[(std::size_t{p} * c.kv + h) * D + i];
          const double allow = f16 ? 0.5 * std::max(Ulp16(ref[i]), Ulp16(g)) : 0.0;
          Upd(kq_good, std::max(0.0, std::fabs(g - ref[i]) - allow) / hm);
          Upd(kq_twin, std::max(0.0, std::fabs(g - twin[i]) - allow) / hm);
          Upd(k_raw, std::fabs(g - ref[i]));
        }
      }
    }
    for (auto p : rows) {
      for (std::uint32_t h = 0; h < c.heads; ++h) {
        const float* x = &d.q[(std::size_t{p} * c.heads + h) * D];
        const double hm = RefNormRope(x, d.gq, p, c.rotary != 0, table, ref);
        RefNormRope(x, d.gq, p, c.rotary == 0, table, twin);
        for (std::uint32_t i = 0; i < D; ++i) {
          const double g = p0.q[(std::size_t{p} * c.heads + h) * D + i];
          Upd(q_good, std::fabs(g - ref[i]) / hm);
          Upd(q_twin, std::fabs(g - twin[i]) / hm);
          Upd(q_raw, std::fabs(g - ref[i]));
        }
      }
    }
    std::cout << "  K cache after norm" << (c.rotary ? "+RoPE" : "") << ": max abs " << Sci(k_raw)
              << " (beyond the " << (f16 ? "half-ulp" : "zero") << " allowance, relative to the head's largest element: "
              << Sci(kq_good) << "), tolerance " << Sci(kPostTol) << "\n";
    std::cout << "  q after norm" << (c.rotary ? "+RoPE" : "") << ": max abs " << Sci(q_raw)
              << ", relative to the head's largest element " << Sci(q_good) << ", tolerance " << Sci(kPostTol) << "\n";
    Require(kq_good <= kPostTol, "K cache differs from the FP64 norm/RoPE");
    Require(q_good <= kPostTol, "q differs from the FP64 norm/RoPE");
    const std::string tw = c.rotary ? "RoPE not applied although rotary_dim=128 requested"
                                    : "RoPE applied although rotary_dim=0 requested";
    RequireTwinFails("K: " + tw, kq_good, kq_twin);
    RequireTwinFails("q: " + tw, q_good, q_twin);
    // V is stored raw: exactly the input (F32) or its binary16 rounding.
    Err ve, vt;
    for (std::uint32_t p = 0; p < c.T; ++p) {
      for (std::uint32_t h = 0; h < c.kv; ++h) {
        const float* x = &d.v[(std::size_t{p} * c.kv + h) * D];
        double ss = 0;
        for (std::uint32_t i = 0; i < D; ++i) ss += double{x[i]} * x[i];
        const double inv = 1.0 / std::sqrt(ss / D + double{kEps});
        for (std::uint32_t i = 0; i < D; ++i) {
          const double g = ck.Vst[(std::size_t{p} * c.kv + h) * D + i];
          ve.Add(g, f16 ? RoundHalf(x[i]) : x[i]);
          vt.Add(g, x[i] * inv);  // V normalized: wrong, V is stored raw
        }
      }
    }
    ReportAbs("V cache equals the (rounded) input exactly", ve, 0.0);
    RequireTwinFails("V cache normalized like K", ve.max_abs, vt.max_abs);
  }

  // ---- attention against FP64 over the stored keys ----------------------------------
  {
    const Err e = CompareRows(ck, rows, Var{});
    ReportAbs("attention vs FP64 on " + std::to_string(rows.size()) + " rows", e, kAttnTol);
    // Twins on a few rows each.
    std::vector<std::uint32_t> mid;
    for (auto p : rows) {
      if (p + 4 < c.T && mid.size() < 14 && (p % 3 == 0 || !c.sampled)) mid.push_back(p);
    }
    if (mid.size() > 14) mid.resize(14);
    if (W != 0) {
      for (int delta : {-1, 1}) {
        std::vector<std::uint32_t> b;  // rows where the window edge is sharp
        for (std::uint32_t p : {W - 1, W, W + 1}) {
          if (p < c.T && !(delta == 1 && p == W - 1)) b.push_back(p);  // +1 sees no key before 0
        }
        const Err good = CompareRows(ck, b, Var{});
        Var v;
        v.win_delta = delta;
        const Err twin = CompareRows(ck, b, v);
        RequireTwinFails(std::string("window ") + (delta < 0 ? "-1" : "+1") + " at positions window-1, window, window+1",
                         good.max_abs, twin.max_abs);
      }
    } else if (c.T > 1100) {
      std::vector<std::uint32_t> b = {1024, 1025, 1100, c.T - 1};
      const Err good = CompareRows(ck, b, Var{});
      Var v;
      v.force_window = 1024;
      RequireTwinFails("full layer treated as a 1024 window", good.max_abs, CompareRows(ck, b, v).max_abs);
    }
    const Err good = CompareRows(ck, mid, Var{});
    {
      Var v;
      v.causal = false;
      RequireTwinFails("non-causal mask", good.max_abs, CompareRows(ck, mid, v).max_abs);
    }
    {
      Var v;
      v.swap_kv = true;
      RequireTwinFails("K and V swapped", good.max_abs, CompareRows(ck, mid, v).max_abs);
    }
    if (c.heads / c.kv > 1) {
      Var v;
      v.gqa_mod = true;
      RequireTwinFails("GQA mapping h % kv_heads", good.max_abs, CompareRows(ck, mid, v).max_abs);
    }
    {
      Var v;
      v.scale = 1.0;
      RequireTwinFails("scale 1 instead of 1/sqrt(128)", good.max_abs, CompareRows(ck, mid, v).max_abs);
    }
  }

  // ---- other launch plans: bit-identical to the one-shot run -------------------------
  {
    std::vector<std::pair<std::string, std::pair<std::uint32_t, Chunks>>> plans;  // name, (capacity, chunks)
    Rng prng(c.seed ^ 0x5555);
    const std::uint32_t n = c.ring_chunk;
    if (W != 0) {
      const std::uint32_t cap = W + n - 1;
      plans.push_back({"ring chunks of " + std::to_string(n), {cap, Fixed(c.T, n)}});
      plans.push_back({"ring random chunks", {cap, Random(c.T, n, prng)}});
      Chunks st;
      for (std::uint32_t s = 0; s + 5 < W;) {
        const std::uint32_t m = std::min(n, W - 5 - s);
        st.push_back({s, m});
        s += m;
      }
      std::uint32_t pos = W - 5;
      st.push_back({pos, std::min(n, 24U)});  // chunk covering positions W-5 .. W+18: straddles the window
      pos += st.back().second;
      for (; pos < c.T; pos += std::min(n, c.T - pos)) st.push_back({pos, std::min(n, c.T - pos)});
      plans.push_back({"ring chunk straddling the window", {cap, st}});
      Chunks dec = Fixed(c.T - std::min<std::uint32_t>(c.T, 40), n);
      for (std::uint32_t p = c.T - std::min<std::uint32_t>(c.T, 40); p < c.T; ++p) dec.push_back({p, 1});
      plans.push_back({"ring prefill then decode rows", {cap, dec}});
      plans.push_back({"single rows, ring of exactly window rows", {W, Fixed(c.T, 1)}});
    } else {
      plans.push_back({"chunks of 128", {cap0, Fixed(c.T, 128)}});
      plans.push_back({"random chunks", {cap0, Random(c.T, 200, prng)}});
      Chunks dec = Fixed(c.T - std::min<std::uint32_t>(c.T, 40), 128);
      for (std::uint32_t p = c.T - std::min<std::uint32_t>(c.T, 40); p < c.T; ++p) dec.push_back({p, 1});
      plans.push_back({"prefill then decode rows", {cap0, dec}});
      plans.push_back({"single rows", {cap0, Fixed(c.T, 1)}});
    }
    for (auto& [name, plan] : plans) {
      const Result r = RunPlan(c, d, plan.first, plan.second, false, c.rotary);
      Require(BitEqual(r.q, p0.q), c.name + ": q differs with plan '" + name + "'");
      Require(BitEqual(r.out, p0.out), c.name + ": output differs with plan '" + name + "'");
      std::cout << "  plan '" << name << "' (capacity " << plan.first << ", " << plan.second.size()
                << " launches): bit-identical to the one-shot run\n";
    }
  }
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "  (" << ms << " ms)\n";
}

// ---- argument rejection ------------------------------------------------------------------

void TestRejections() {
  std::cout << "argument rejection (nothing is launched)\n";
  DevBuf<float> q(10 * D * 4), kk(D * 4), out(10 * D * 4), g(D);
  DevBuf<std::uint8_t> kc(64 * D * 4), vc(64 * D * 4);
  std::string err;
  auto attn = [&](std::uint32_t tokens, std::uint32_t heads, std::uint32_t kv, std::uint32_t window,
                  std::uint32_t start, std::uint32_t cap) {
    return k::Attention128(q.get(), kc.get(), vc.get(), k::KvType::kF32, out.get(), tokens, heads, kv, window, start,
                           cap, 0.0883883F, g_stream, &err);
  };
  auto post = [&](std::uint32_t tokens, std::uint32_t heads, std::uint32_t kv, std::uint32_t cap,
                  std::uint32_t rotary, float theta) {
    return k::QkNormRopeAppend(q.get(), kk.get(), kk.get(), g.get(), g.get(), kc.get(), vc.get(), k::KvType::kF32,
                               tokens, heads, kv, kEps, 0, cap, rotary, theta, g_stream, &err);
  };
  Require(attn(4, 10, 1, 8, 0, 11) && attn(4, 10, 1, 8, 0, 12), "valid attention launch rejected");
  Require(!attn(4, 10, 1, 8, 0, 10), "capacity < window + tokens - 1 accepted");
  Require(!attn(4, 10, 1, 0, 10, 13), "full attention with capacity < start + tokens accepted");
  Require(!attn(1, 5, 1, 8, 0, 8), "group 5 accepted");
  Require(!attn(1, 30, 10, 8, 0, 8), "group 3 accepted");
  Require(!attn(1, 10, 3, 8, 0, 8), "heads not a multiple of kv_heads accepted");
  Require(!attn(1, 10, 0, 8, 0, 8), "kv_heads 0 accepted");
  Require(!post(4, 10, 1, 3, 128, 10000.0F), "capacity < tokens accepted");
  Require(!post(4, 10, 1, 8, 64, 10000.0F), "rotary_dim 64 accepted");
  Require(!post(4, 10, 1, 8, 128, 0.0F), "theta 0 accepted");
  Require(post(4, 10, 1, 8, 0, 0.0F), "theta is irrelevant without RoPE");
  HipCheck(hipStreamSynchronize(g_stream), "sync");
  std::cout << "  capacity, group, head-count, rotary_dim and theta checks reject bad arguments\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "small";
  return RunGpu([&] {
    HipCheck(hipStreamCreate(&g_stream), "stream");
    std::vector<Cfg> cfgs;
    const auto F16 = k::KvType::kF16;
    const auto F32 = k::KvType::kF32;
    if (mode == "small") {
      TestRejections();
      std::uint64_t seed = 100;
      // Tiny-fixture shapes: hidden 256/512 -> 10 heads / 1 KV head and 20 / 2; windows 8 and 16.
      for (auto type : {F32, F16}) {
        for (auto [h, kv] : {std::pair<std::uint32_t, std::uint32_t>{10, 1}, {20, 2}}) {
          for (std::uint32_t w : {8U, 16U}) {
            cfgs.push_back({"tiny " + std::to_string(h) + "/" + std::to_string(kv) + " window " + std::to_string(w), h, kv,
                            w, 200, type, 128, 16, false, seed++});
          }
          cfgs.push_back({"tiny " + std::to_string(h) + "/" + std::to_string(kv) + " full", h, kv, 0, 150, type, 0, 16,
                          false, seed++});
        }
        // Groups 1 and 2.
        cfgs.push_back({"group 1 window 16", 4, 4, 16, 120, type, 128, 16, false, seed++});
        cfgs.push_back({"group 1 full", 4, 4, 0, 90, type, 0, 16, false, seed++});
        cfgs.push_back({"group 2 window 16", 8, 4, 16, 120, type, 128, 16, false, seed++});
        cfgs.push_back({"group 2 full", 8, 4, 0, 90, type, 0, 16, false, seed++});
      }
    } else if (mode.rfind("real-", 0) == 0) {
      const bool sliding = mode.find("sliding") != std::string::npos;
      const auto type = mode.find("f16") != std::string::npos ? F16 : F32;
      cfgs.push_back({std::string("real shape ") + (sliding ? "sliding" : "full"), 80, 8, sliding ? 1024U : 0U,
                      sliding ? 3300U : 3300U, type, sliding ? 128U : 0U, 32, true,
                      sliding ? 9001U : 9002U});
    } else {
      throw std::runtime_error("unknown mode " + mode);
    }
    for (const auto& c : cfgs) RunConfig(c);
    PrintTwinSummary();
    HipCheck(hipStreamDestroy(g_stream), "stream");
  });
}

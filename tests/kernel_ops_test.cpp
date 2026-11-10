// GPU checks of the ricursive elementwise operators against FP64 formulas on
// seeded random data of unequal sizes, each with deliberately wrong twins that
// must be rejected by at least 100x. Exit 77 when there is no GPU.
//
// Tolerances (relative to the largest reference value, which is O(1)..O(50)):
// the fp32 kernels do one expf, a few multiplies and (norms) a 5120-term
// fp32 sum, so 1e-7..3e-7 is expected; 2e-6 allows ~10x for the summation
// order and expf's few-ulp error without hiding a wrong formula (every twin
// is off by >= 1e-3).
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "hip/kernels.hpp"
#include "tests/check.hpp"

namespace k = gufo::models::ricursive::rocm;
using namespace ricursive_test;

namespace {

constexpr double kTol = 2e-6;
constexpr float kEps = 1e-6F;

hipStream_t g_stream = nullptr;

void Ok(bool ok, const std::string& error, const char* what) {
  Require(ok, std::string(what) + " launch failed: " + error);
  HipCheck(hipStreamSynchronize(g_stream), what);
}

std::vector<float> Normal(Rng& rng, std::size_t n, double sigma, double mean = 0) {
  std::vector<float> v(n);
  for (auto& x : v) x = static_cast<float>(mean + sigma * rng.Normal());
  return v;
}

// Bytes equal and every value finite (tests/check.hpp).
bool BitEqual(const std::vector<float>& a, const std::vector<float>& b) { return BitEqualFinite(a, b); }

// ---- RmsNorm -----------------------------------------------------------------------

std::vector<double> RefRmsNorm(const std::vector<float>& x, const std::vector<float>& w,
                               std::uint32_t rows, std::uint32_t dim, double eps,
                               double w_offset = 0) {
  std::vector<double> out(x.size());
  for (std::uint32_t r = 0; r < rows; ++r) {
    double ss = 0;
    for (std::uint32_t i = 0; i < dim; ++i) ss += double{x[r * dim + i]} * x[r * dim + i];
    const double inv = 1.0 / std::sqrt(ss / dim + eps);
    for (std::uint32_t i = 0; i < dim; ++i) {
      out[r * dim + i] = (w_offset + w[i]) * x[r * dim + i] * inv;
    }
  }
  return out;
}

Err Compare(const std::vector<float>& got, const std::vector<double>& ref) {
  Err e;
  for (std::size_t i = 0; i < got.size(); ++i) e.Add(got[i], ref[i]);
  return e;
}

void TestRmsNorm() {
  std::cout << "RmsNorm\n";
  struct Shape { std::uint32_t rows, dim; };
  bool first = true;
  for (Shape sh : {Shape{9, 5120}, Shape{4, 128}, Shape{1, 5120}, Shape{6, 100}, Shape{5, 513}, Shape{3, 1}}) {
    Rng rng(1000 + sh.rows * 7 + sh.dim);
    // Rows of very different scale; row r % 4 == 1 is tiny, so eps matters there.
    std::vector<float> x(std::size_t{sh.rows} * sh.dim);
    const double scales[4] = {1.0, 0.001, 30.0, 0.2};
    for (std::uint32_t r = 0; r < sh.rows; ++r) {
      for (std::uint32_t i = 0; i < sh.dim; ++i) {
        x[r * sh.dim + i] = static_cast<float>(scales[r % 4] * rng.Normal());
      }
    }
    std::vector<float> w(sh.dim);
    for (auto& v : w) v = static_cast<float>(0.5 + rng.Uniform());
    DevBuf<float> dx(x.size()), dw(w.size()), dout(x.size());
    dx.Upload(x);
    dw.Upload(w);
    std::string err;
    Ok(k::RmsNorm(dx.get(), dw.get(), dout.get(), sh.rows, sh.dim, kEps, g_stream, &err), err, "RmsNorm");
    const auto got = dout.Download();
    const auto ref = RefRmsNorm(x, w, sh.rows, sh.dim, kEps);
    const Err e = Compare(got, ref);
    ReportRel("rows=" + std::to_string(sh.rows) + " dim=" + std::to_string(sh.dim), e, kTol);
    // In place gives the same bits.
    DevBuf<float> inplace(x.size());
    inplace.Upload(x);
    Ok(k::RmsNorm(inplace.get(), dw.get(), inplace.get(), sh.rows, sh.dim, kEps, g_stream, &err), err, "RmsNorm");
    Require(BitEqual(inplace.Download(), got), "RmsNorm in place differs from out of place");
    // Every row alone gives the same bits as inside the batch.
    for (std::uint32_t r = 0; r < sh.rows; ++r) {
      Ok(k::RmsNorm(dx.get() + std::size_t{r} * sh.dim, dw.get(), inplace.get() + std::size_t{r} * sh.dim, 1,
                    sh.dim, kEps, g_stream, &err), err, "RmsNorm");
    }
    Require(BitEqual(inplace.Download(), got), "RmsNorm row alone differs from batch");
    if (first) {
      first = false;
      // Twins: Gemma-style (1 + w) weight; eps ignored (matters on the tiny rows).
      const Err t1 = Compare(got, RefRmsNorm(x, w, sh.rows, sh.dim, kEps, 1.0));
      RequireTwinFails("RmsNorm with (1+w) weight", e.Rel(), t1.Rel());
      const Err t2 = Compare(got, RefRmsNorm(x, w, sh.rows, sh.dim, 0.0));
      RequireTwinFails("RmsNorm without eps", e.Rel(), t2.Rel());
      const Err t3 = Compare(got, RefRmsNorm(x, std::vector<float>(sh.dim, 1.0F), sh.rows, sh.dim, kEps));
      RequireTwinFails("RmsNorm without weight", e.Rel(), t3.Rel());
    }
  }
}

// ---- ResidualPostNorm ----------------------------------------------------------------

void TestResidualPostNorm() {
  std::cout << "ResidualPostNorm\n";
  for (auto [rows, dim] : {std::pair<std::uint32_t, std::uint32_t>{7, 5120}, {3, 130}, {1, 5120}}) {
    Rng rng(2000 + rows + dim);
    auto x = Normal(rng, std::size_t{rows} * dim, 2.0);
    std::vector<float> b(x.size());
    const double scales[4] = {0.7, 0.001, 25.0, 0.2};
    for (std::uint32_t r = 0; r < rows; ++r)
      for (std::uint32_t i = 0; i < dim; ++i) b[r * dim + i] = static_cast<float>(scales[r % 4] * rng.Normal());
    std::vector<float> w(dim);
    for (auto& v : w) v = static_cast<float>(0.3 + 1.5 * rng.Uniform());
    DevBuf<float> dx(x.size()), db(b.size()), dw(dim);
    dx.Upload(x);
    db.Upload(b);
    dw.Upload(w);
    std::string err;
    Ok(k::ResidualPostNorm(dx.get(), db.get(), dw.get(), rows, dim, kEps, g_stream, &err), err, "ResidualPostNorm");
    const auto got = dx.Download();
    Require(BitEqual(db.Download(), b), "ResidualPostNorm modified the branch");
    const auto nb = RefRmsNorm(b, w, rows, dim, kEps);
    std::vector<double> ref(x.size()), twin_raw(x.size()), twin_nowt(x.size());
    const auto nb_nowt = RefRmsNorm(b, std::vector<float>(dim, 1.0F), rows, dim, kEps);
    for (std::size_t i = 0; i < x.size(); ++i) {
      ref[i] = double{x[i]} + nb[i];
      twin_raw[i] = double{x[i]} + double{w[i % dim]} * b[i];  // added without normalizing
      twin_nowt[i] = double{x[i]} + nb_nowt[i];                 // normalized, weight dropped
    }
    const Err e = Compare(got, ref);
    ReportRel("rows=" + std::to_string(rows) + " dim=" + std::to_string(dim), e, kTol);
    if (rows == 7) {
      RequireTwinFails("post-norm added without normalizing", e.Rel(), Compare(got, twin_raw).Rel());
      RequireTwinFails("post-norm without weight", e.Rel(), Compare(got, twin_nowt).Rel());
      // Rows alone are bit-identical to the batch.
      DevBuf<float> dx2(x.size());
      dx2.Upload(x);
      for (std::uint32_t r = 0; r < rows; ++r) {
        Ok(k::ResidualPostNorm(dx2.get() + std::size_t{r} * dim, db.get() + std::size_t{r} * dim, dw.get(), 1, dim,
                               kEps, g_stream, &err), err, "ResidualPostNorm");
      }
      Require(BitEqual(dx2.Download(), got), "ResidualPostNorm row alone differs from batch");
    }
  }
  // x aliasing branch is rejected.
  DevBuf<float> a(128), w(128);
  std::string err;
  Require(!k::ResidualPostNorm(a.get(), a.get(), w.get(), 1, 128, kEps, g_stream, &err), "alias accepted");
}

// ---- ScaleRows -----------------------------------------------------------------------

void TestScaleRows() {
  std::cout << "ScaleRows\n";
  Rng rng(3000);
  const std::uint32_t rows = 5, dim = 5120;
  const float scale = 71.5541763F;  // the kind of value ricursive.embedding_scale holds
  auto x = Normal(rng, std::size_t{rows} * dim, 0.02);
  DevBuf<float> dx(x.size()), dout(x.size());
  dx.Upload(x);
  std::string err;
  Ok(k::ScaleRows(dx.get(), dout.get(), rows, dim, scale, g_stream, &err), err, "ScaleRows");
  const auto got = dout.Download();
  std::vector<double> ref(x.size()), twin(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    ref[i] = double{x[i]} * double{scale};
    twin[i] = x[i];  // embedding left unscaled
  }
  const Err e = Compare(got, ref);
  ReportRel("rows=5 dim=5120", e, 1e-7);  // one fp32 multiply: correctly rounded
  RequireTwinFails("embedding scale skipped", e.Rel(), Compare(got, twin).Rel());
  Ok(k::ScaleRows(dx.get(), dx.get(), rows, dim, scale, g_stream, &err), err, "ScaleRows");
  Require(BitEqual(dx.Download(), got), "ScaleRows in place differs");
}

// ---- ValueMix ------------------------------------------------------------------------

void TestValueMix() {
  std::cout << "ValueMix\n";
  const std::uint32_t rows = 6, dim = 5120;
  std::uint64_t seed = 4000;
  for (float lambda : {0.37F, -3.2F, 1.9F, 6.0F, -100.0F, 100.0F, 0.0F}) {
    Rng rng(seed++);
    auto x0 = Normal(rng, std::size_t{rows} * dim, 1.5);
    auto a = Normal(rng, std::size_t{rows} * dim, 0.8, 0.3);
    DevBuf<float> dx0(x0.size()), da(a.size()), dout(a.size()), dl(1);
    dx0.Upload(x0);
    da.Upload(a);
    dl.Upload({lambda});
    std::string err;
    Ok(k::ValueMix(dx0.get(), da.get(), dl.get(), dout.get(), rows, dim, g_stream, &err), err, "ValueMix");
    const auto got = dout.Download();
    const double s = 1.0 / (1.0 + std::exp(-static_cast<double>(lambda)));
    std::vector<double> ref(a.size()), twin_nosig(a.size()), twin_swap(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      ref[i] = double{x0[i]} + s * (double{a[i]} - x0[i]);
      twin_nosig[i] = double{x0[i]} + double{lambda} * (double{a[i]} - x0[i]);
      twin_swap[i] = double{a[i]} + s * (double{x0[i]} - a[i]);
    }
    const Err e = Compare(got, ref);
    ReportRel("lambda=" + Sci(lambda) + " (sigmoid " + Sci(s) + ")", e, kTol);
    if (lambda == 0.37F || lambda == -3.2F || lambda == 1.9F) {
      RequireTwinFails("ValueMix without sigmoid (lambda=" + Sci(lambda) + ")", e.Rel(), Compare(got, twin_nosig).Rel());
      RequireTwinFails("ValueMix with x0 and a swapped (lambda=" + Sci(lambda) + ")", e.Rel(),
                       Compare(got, twin_swap).Rel());
    }
    if (lambda == 0.37F) {
      DevBuf<float> alias_a(a.size()), alias_x(x0.size());
      alias_a.Upload(a);
      alias_x.Upload(x0);
      Ok(k::ValueMix(dx0.get(), alias_a.get(), dl.get(), alias_a.get(), rows, dim, g_stream, &err), err, "ValueMix");
      Require(BitEqual(alias_a.Download(), got), "ValueMix in place on a differs");
      Ok(k::ValueMix(alias_x.get(), da.get(), dl.get(), alias_x.get(), rows, dim, g_stream, &err), err, "ValueMix");
      Require(BitEqual(alias_x.Download(), got), "ValueMix in place on x0 differs");
    }
  }
}

// ---- SigmoidGate -----------------------------------------------------------------------

void TestSigmoidGate() {
  std::cout << "SigmoidGate\n";
  const std::uint32_t rows = 3, dim = 10240;
  Rng rng(5000);
  auto attn = Normal(rng, std::size_t{rows} * dim, 1.5);
  auto gate = Normal(rng, std::size_t{rows} * dim, 3.0);
  DevBuf<float> da(attn.size()), dg(gate.size()), dout(attn.size());
  da.Upload(attn);
  dg.Upload(gate);
  std::string err;
  Ok(k::SigmoidGate(da.get(), dg.get(), dout.get(), rows, dim, g_stream, &err), err, "SigmoidGate");
  const auto got = dout.Download();
  auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
  std::vector<double> ref(attn.size()), t_wrong(attn.size()), t_nosig(attn.size());
  for (std::size_t i = 0; i < attn.size(); ++i) {
    ref[i] = double{attn[i]} * sig(gate[i]);
    t_wrong[i] = double{gate[i]} * sig(attn[i]);  // gate applied to the wrong tensor
    t_nosig[i] = double{attn[i]} * gate[i];
  }
  const Err e = Compare(got, ref);
  ReportRel("rows=3 dim=10240", e, kTol);
  RequireTwinFails("gate applied to the wrong tensor", e.Rel(), Compare(got, t_wrong).Rel());
  RequireTwinFails("gate without sigmoid", e.Rel(), Compare(got, t_nosig).Rel());
  Ok(k::SigmoidGate(da.get(), dg.get(), da.get(), rows, dim, g_stream, &err), err, "SigmoidGate");
  Require(BitEqual(da.Download(), got), "SigmoidGate in place on attn differs");
  da.Upload(attn);
  Ok(k::SigmoidGate(da.get(), dg.get(), dg.get(), rows, dim, g_stream, &err), err, "SigmoidGate");
  Require(BitEqual(dg.Download(), got), "SigmoidGate in place on gate differs");
}

// ---- ClampedSwiGlu ---------------------------------------------------------------------

void TestClampedSwiGlu() {
  std::cout << "ClampedSwiGlu\n";
  const std::uint32_t rows = 4, dim = 40960;
  auto silu = [](double g) { return g / (1.0 + std::exp(-g)); };
  for (float c : {7.0F, 2.5F, 0.8F}) {
    Rng rng(6000 + static_cast<std::uint64_t>(c * 10));
    auto g = Normal(rng, std::size_t{rows} * dim, 0.8 * c);
    auto u = Normal(rng, std::size_t{rows} * dim, 0.8 * c);
    std::size_t g_hi = 0, g_lo = 0, u_hi = 0, u_lo = 0;
    for (std::size_t i = 0; i < g.size(); ++i) {
      g_hi += g[i] > c;
      g_lo += g[i] < -c;
      u_hi += u[i] > c;
      u_lo += u[i] < -c;
    }
    Require(g_hi > 100 && g_lo > 100 && u_hi > 100 && u_lo > 100, "test data does not reach every clamp region");
    DevBuf<float> dg(g.size()), du(u.size()), dout(g.size());
    dg.Upload(g);
    du.Upload(u);
    std::string err;
    Ok(k::ClampedSwiGlu(dg.get(), du.get(), dout.get(), rows, dim, c, g_stream, &err), err, "ClampedSwiGlu");
    const auto got = dout.Download();
    std::vector<double> ref(g.size()), t_sym(g.size()), t_none(g.size()), t_uhalf(g.size()), t_gnone(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) {
      const double gg = std::min<double>(g[i], c);
      const double uu = std::max<double>(-c, std::min<double>(u[i], c));
      ref[i] = silu(gg) * uu;
      t_sym[i] = silu(std::max<double>(-c, gg)) * uu;                       // symmetric gate clamp
      t_none[i] = silu(g[i]) * u[i];                                        // no clamp
      t_uhalf[i] = silu(gg) * std::min<double>(u[i], c);                    // up clamped from above only
      t_gnone[i] = silu(g[i]) * uu;                                         // gate not clamped
    }
    const Err e = Compare(got, ref);
    ReportRel("clamp=" + Sci(c) + " (g>c " + std::to_string(g_hi) + ", g<-c " + std::to_string(g_lo) +
                  ", |u|>c " + std::to_string(u_hi + u_lo) + ")", e, kTol);
    const std::string tag = " (c=" + Sci(c) + ")";
    RequireTwinFails("symmetric gate clamp" + tag, e.Rel(), Compare(got, t_sym).Rel());
    RequireTwinFails("no clamp" + tag, e.Rel(), Compare(got, t_none).Rel());
    RequireTwinFails("up clamped one-sided" + tag, e.Rel(), Compare(got, t_uhalf).Rel());
    RequireTwinFails("gate not clamped" + tag, e.Rel(), Compare(got, t_gnone).Rel());
    if (c == 2.5F) {
      Ok(k::ClampedSwiGlu(dg.get(), du.get(), dg.get(), rows, dim, c, g_stream, &err), err, "ClampedSwiGlu");
      Require(BitEqual(dg.Download(), got), "ClampedSwiGlu in place on gate differs");
    }
  }
  // Invalid clamps are rejected and nothing is written.
  const std::vector<float> bad = {0.0F, -0.0F, -1.0F, std::nanf(""), INFINITY, -INFINITY};
  for (float c : bad) {
    DevBuf<float> dg(64), du(64), dout(64);
    dout.FillBytes(0x7B);
    std::string err;
    const bool ok = k::ClampedSwiGlu(dg.get(), du.get(), dout.get(), 1, 64, c, g_stream, &err);
    HipCheck(hipStreamSynchronize(g_stream), "sync");
    Require(!ok && !err.empty(), "clamp " + Sci(c) + " was accepted");
    const auto after = dout.Download();
    for (float v : after) {
      std::uint32_t bits;
      std::memcpy(&bits, &v, 4);
      Require(bits == 0x7B7B7B7BU, "rejected ClampedSwiGlu wrote output");
    }
  }
  std::cout << "  clamp 0, -0, -1, NaN, +inf, -inf rejected, output untouched\n";
  // NaN in, NaN out (a NaN is not replaced by the clamp value).
  std::vector<float> g = {1.0F, std::nanf(""), 2.0F, -1.0F}, u = {std::nanf(""), 1.0F, 0.5F, 0.25F};
  DevBuf<float> dg(4), du(4), dout(4);
  dg.Upload(g);
  du.Upload(u);
  std::string err;
  Ok(k::ClampedSwiGlu(dg.get(), du.get(), dout.get(), 1, 4, 3.0F, g_stream, &err), err, "ClampedSwiGlu");
  const auto o = dout.Download();
  Require(std::isnan(o[0]) && std::isnan(o[1]) && std::isfinite(o[2]) && std::isfinite(o[3]), "NaN propagation");
  std::cout << "  NaN inputs give NaN outputs\n";
}

// ---- GatherRows ------------------------------------------------------------------------

void TestGatherRows() {
  std::cout << "GatherRows\n";
  const std::uint32_t src_rows = 50, dim = 1000;
  Rng rng(7000);
  auto src = Normal(rng, std::size_t{src_rows} * dim, 1.0);
  const std::vector<std::int32_t> ids = {7, 0, 49, 7, 23, -1, 50, INT_MIN, INT_MAX, 12};
  DevBuf<float> dsrc(src.size()), dout(ids.size() * dim);
  DevBuf<std::int32_t> dids(ids.size());
  dsrc.Upload(src);
  dids.Upload(ids);
  std::string err;
  Ok(k::GatherRows(dsrc.get(), dids.get(), dout.get(), static_cast<std::uint32_t>(ids.size()), src_rows, dim, g_stream, &err),
     err, "GatherRows");
  const auto got = dout.Download();
  Err e, twin;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    const bool valid = ids[i] >= 0 && ids[i] < static_cast<int>(src_rows);
    for (std::uint32_t d = 0; d < dim; ++d) {
      const float v = got[i * dim + d];
      if (valid) {
        e.Add(v, src[std::size_t{static_cast<std::uint32_t>(ids[i])} * dim + d]);
        twin.Add(v, src[std::size_t{static_cast<std::uint32_t>((ids[i] + 1) % src_rows)} * dim + d]);
      } else {
        Require(std::isnan(v), "out-of-range id must give NaN");
      }
    }
  }
  ReportAbs("gathered rows exact; 4 out-of-range ids give NaN", e, 0.0);
  RequireTwinFails("gather of the neighbouring row", e.max_abs, twin.max_abs, 1e-7, 100.0);
}

}  // namespace

int main() {
  return RunGpu([] {
    HipCheck(hipStreamCreate(&g_stream), "stream");
    TestRmsNorm();
    TestResidualPostNorm();
    TestScaleRows();
    TestValueMix();
    TestSigmoidGate();
    TestClampedSwiGlu();
    TestGatherRows();
    PrintTwinSummary();
    HipCheck(hipStreamDestroy(g_stream), "stream");
  });
}

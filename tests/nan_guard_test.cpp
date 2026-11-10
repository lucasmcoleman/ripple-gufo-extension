// NaN / Inf guard: every comparison helper the tests use must report FAILURE when
// the "device result" holds a NaN or an Inf, and must still pass clean data.
// CPU only, no fixtures. The idioms this guards against:
//   std::max(0.0, NaN) == 0.0, NaN > tol == false, a[i] > a[best] never picks a NaN
//   (an all-NaN logit row "agrees" with argmax 0), and a running maximum that is
//   overwritten by a later finite value after a NaN was stored.
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "tests/oracle_common.hpp"

using namespace ricursive_test;

namespace {

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

/// `fn` must throw (the helper rejected the input); returning normally is a hole.
void MustReject(const std::string& what, const std::function<void()>& fn) {
  bool threw = false;
  try {
    fn();
  } catch (const std::exception&) {
    threw = true;
  }
  Require(threw, "HOLE: " + what + " was accepted");
}

void MustAccept(const std::string& what, const std::function<void()>& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    Require(false, "clean input rejected by " + what + ": " + e.what());
  }
}

std::vector<float> Ramp(std::size_t n) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<float>(i % 17) * 0.25F - 1.0F;
  return v;
}

/// Each variant puts the bad value at a different place (first, middle, last,
/// everywhere) in `got`; the reference stays finite, then the same in `ref`.
std::vector<std::pair<std::string, std::vector<float>>> Poisoned(const std::vector<float>& clean,
                                                                 float bad) {
  std::vector<std::pair<std::string, std::vector<float>>> out;
  for (const std::size_t at : {std::size_t{0}, clean.size() / 2, clean.size() - 1}) {
    auto v = clean;
    v[at] = bad;
    out.push_back({"at " + std::to_string(at), v});
  }
  out.push_back({"everywhere", std::vector<float>(clean.size(), bad)});
  return out;
}

void TestIdiomsAreHoles() {
  // Documents why the helpers exist: the plain idioms really do accept NaN.
  Require(std::max(0.0, static_cast<double>(kNaN)) == 0.0, "std::max(0, NaN) is no longer 0?");
  Require(!(static_cast<double>(kNaN) > 1.0), "NaN > tol is no longer false?");
  std::vector<float> all_nan(8, kNaN);
  std::size_t best = 0;
  for (std::size_t i = 1; i < all_nan.size(); ++i)
    if (all_nan[i] > all_nan[best]) best = i;
  Require(best == 0, "the plain argmax loop no longer returns 0 for NaN?");
}

void TestMaxAbsDiff() {
  const auto clean = Ramp(64);
  Require(MaxAbsDiff(clean, clean) == 0.0, "identical finite inputs must give 0");
  auto shifted = clean;
  shifted[10] += 0.5F;
  Require(MaxAbsDiff(clean, shifted) == 0.5, "finite difference must be reported");
  for (const float bad : {kNaN, kInf, -kInf}) {
    for (const auto& [where, v] : Poisoned(clean, bad)) {
      const std::string tag = std::string(std::isnan(bad) ? "NaN " : "Inf ") + where;
      Require(MaxAbsDiff(v, clean) == kInf, "MaxAbsDiff(bad, clean) " + tag);
      Require(MaxAbsDiff(clean, v) == kInf, "MaxAbsDiff(clean, bad) " + tag);
      // Both runs bad in the same place: equal bits, still a failure.
      Require(MaxAbsDiff(v, v) == kInf, "MaxAbsDiff(bad, bad) " + tag);
      Require(Exceeds(MaxAbsDiff(v, v), 1e30), "Exceeds(MaxAbsDiff(bad, bad)) " + tag);
    }
  }
  // The old loop kept the LAST difference after a NaN: NaN first, finite equal values after.
  auto first_nan = clean;
  first_nan[0] = kNaN;
  Require(MaxAbsDiff(first_nan, clean) == kInf, "NaN at index 0 followed by equal values was lost");
  MustReject("size mismatch", [&] { (void)MaxAbsDiff(clean, Ramp(63)); });
}

void TestMaxAndExceeds() {
  Require(std::isnan(MaxKeepNaN(0.0, static_cast<double>(kNaN))), "MaxKeepNaN(0, NaN)");
  Require(std::isnan(MaxKeepNaN(static_cast<double>(kNaN), 3.0)), "MaxKeepNaN(NaN, 3)");
  Require(MaxKeepNaN(1.0, 2.0) == 2.0 && MaxKeepNaN(2.0, 1.0) == 2.0, "MaxKeepNaN clean");
  Require(Exceeds(static_cast<double>(kNaN), 1.0), "Exceeds(NaN, tol)");
  Require(Exceeds(static_cast<double>(kInf), 1.0), "Exceeds(Inf, tol)");
  Require(Exceeds(2.0, 1.0) && !Exceeds(1.0, 1.0) && !Exceeds(0.0, 1.0), "Exceeds clean");
}

void TestStrictArgMax() {
  const auto clean = Ramp(40);
  Require(StrictArgMax(clean).has_value(), "clean row has an argmax");
  Require(*StrictArgMax(std::vector<float>{0.0F, 3.0F, 2.0F}) == 1, "argmax value");
  for (const float bad : {kNaN, kInf, -kInf})
    for (const auto& [where, v] : Poisoned(clean, bad))
      Require(!StrictArgMax(v).has_value(), "StrictArgMax accepted a row with a bad value " + where);
  Require(!StrictArgMax(std::vector<float>{}).has_value(), "empty row has no argmax");
}

void TestExcessOverAndBitEqual() {
  Require(ExcessOver(1.0, 1.0, 0.0) == 0.0 && ExcessOver(1.5, 1.0, 0.25) == 0.25, "ExcessOver clean");
  const double nan = kNaN, inf = kInf;
  for (const double bad : {nan, inf, -inf}) {
    Require(ExcessOver(bad, 1.0, 0.1) == inf, "ExcessOver(bad got)");
    Require(ExcessOver(1.0, bad, 0.1) == inf, "ExcessOver(bad ref)");
    Require(ExcessOver(1.0, 1.0, bad) == inf, "ExcessOver(bad allowance)");
  }
  const auto clean = Ramp(32);
  Require(BitEqualFinite(clean, clean), "BitEqualFinite clean");
  auto other = clean;
  other[3] += 1e-3F;
  Require(!BitEqualFinite(clean, other), "BitEqualFinite must see a difference");
  for (const float bad : {kNaN, kInf, -kInf})
    for (const auto& [where, v] : Poisoned(clean, bad))
      Require(!BitEqualFinite(v, v), "BitEqualFinite(bad, bad) accepted " + where);
}

void TestErrAndReports() {
  // Err::Add: any NaN/Inf on either side is an infinite error, even Inf against Inf.
  for (const double bad : {static_cast<double>(kNaN), static_cast<double>(kInf)}) {
    Err got_bad, ref_bad, both_bad;
    got_bad.Add(bad, 1.0);
    ref_bad.Add(1.0, bad);
    both_bad.Add(bad, bad);
    for (const Err* e : {&got_bad, &ref_bad, &both_bad}) {
      Require(e->max_abs == kInf, "Err::Add left a bad value out of max_abs");
      MustReject("ReportAbs of a bad value", [&] { ReportAbs("guard", *e, 1e30); });
      MustReject("ReportRel of a bad value", [&] { ReportRel("guard", *e, 1e30); });
    }
  }
  // A bad value after many good ones is not averaged or maxed away.
  Err late;
  for (int i = 0; i < 1000; ++i) late.Add(1.0, 1.0);
  late.Add(kNaN, 1.0);
  for (int i = 0; i < 1000; ++i) late.Add(1.0, 1.0);
  MustReject("ReportAbs with a NaN in the middle", [&] { ReportAbs("guard", late, 1e30); });
  Err merged;
  merged.Merge(late);
  MustReject("Err::Merge of a bad Err", [&] { ReportAbs("guard", merged, 1e30); });
  Err empty;
  MustReject("ReportAbs over nothing", [&] { ReportAbs("guard", empty, 1.0); });
  MustReject("ReportRel over nothing", [&] { ReportRel("guard", empty, 1.0); });
  Err clean;
  for (int i = 0; i < 10; ++i) clean.Add(1.0 + 1e-9 * i, 1.0);
  MustAccept("ReportAbs on clean data", [&] { ReportAbs("guard", clean, 1e-6); });
  MustAccept("ReportRel on clean data", [&] { ReportRel("guard", clean, 1e-6); });
  MustReject("ReportAbs over tolerance", [&] { ReportAbs("guard", clean, 1e-12); });
}

void TestTwinMargin() {
  // A twin must beat a FINITE correct error; infinity "beating" infinity proves nothing.
  const double nan = kNaN, inf = kInf;
  MustReject("RequireTwinFails with a NaN correct error", [&] { RequireTwinFails("guard", nan, 1.0); });
  MustReject("RequireTwinFails with an Inf correct error and an Inf twin",
             [&] { RequireTwinFails("guard", inf, inf); });
  MustReject("RequireTwinFails with a NaN twin", [&] { RequireTwinFails("guard", 1e-9, nan); });
  MustReject("RequireTwinFails with a twin as good as the kernel", [&] { RequireTwinFails("guard", 1e-3, 1e-3); });
  MustAccept("RequireTwinFails on a clearly wrong twin", [&] { RequireTwinFails("guard", 1e-9, 1.0); });
}

/// logits [T][vocab] with a clear argmax per row.
std::vector<float> Logits(std::size_t T, std::size_t vocab) {
  std::vector<float> v(T * vocab);
  for (std::size_t t = 0; t < T; ++t)
    for (std::size_t i = 0; i < vocab; ++i)
      v[t * vocab + i] = (i == (t * 7 + 3) % vocab) ? 5.0F : 0.01F * static_cast<float>((i * 13 + t) % 11);
  return v;
}

void TestCompare() {
  constexpr std::size_t T = 6, V = 32;
  const auto base = Logits(T, V);
  const Stats ok = Compare(base, base, V);
  Require(ok.finite && ok.max_abs == 0 && ok.top1 == 1.0 && ok.mean_kl == 0, "Compare of identical rows");

  auto off = base;
  off[2 * V + 5] += 1e-3F;
  const Stats small = Compare(off, base, V);
  Require(small.finite && small.max_abs > 0 && small.max_abs < 1e-2 && small.top1 == 1.0,
          "Compare of a small perturbation");

  auto flipped = base;  // top-1 really changes: must be reported
  flipped[1 * V + (1 * 7 + 3) % V] = -5.0F;
  Require(Compare(flipped, base, V).top1 < 1.0, "Compare must see a changed argmax");

  const auto check_bad = [&](const std::string& what, const std::vector<float>& ref,
                             const std::vector<float>& bas) {
    const Stats s = Compare(ref, bas, V);
    Require(!s.finite, "HOLE: Compare(" + what + ") says finite");
    Require(s.max_abs == kInf, "HOLE: Compare(" + what + ") max_abs " + std::to_string(s.max_abs));
    Require(s.mean_kl == kInf, "HOLE: Compare(" + what + ") mean_kl is finite");
    Require(s.top1 == 0.0, "HOLE: Compare(" + what + ") top1 " + std::to_string(s.top1));
    // How the tests consume it:
    Require(!(s.finite && s.max_abs <= 2e-3 && s.top1 >= 1.0), "HOLE: the usual pass condition holds on " + what);
  };
  for (const float bad : {kNaN, kInf, -kInf}) {
    const std::string name = std::isnan(bad) ? "NaN" : (bad > 0 ? "+Inf" : "-Inf");
    for (const std::size_t at : {std::size_t{0}, 3 * V + 4, T * V - 1}) {
      auto v = base;
      v[at] = bad;
      check_bad(name + " in ref at " + std::to_string(at), v, base);
      check_bad(name + " in base at " + std::to_string(at), base, v);
      check_bad(name + " in both at " + std::to_string(at), v, v);
    }
    // All logits bad, including the case where the base argmax is index 0 (a plain
    // argmax loop returns 0 for an all-NaN row and would agree).
    check_bad(name + " everywhere", std::vector<float>(T * V, bad), base);
    auto zero_top = base;
    for (std::size_t t = 0; t < T; ++t) zero_top[t * V] = 9.0F;  // argmax 0 in every row
    check_bad(name + " everywhere vs argmax-0 base", std::vector<float>(T * V, bad), zero_top);
  }
  // One bad row out of many must fail the whole comparison, not dilute the mean.
  auto one_row = base;
  for (std::size_t i = 0; i < V; ++i) one_row[4 * V + i] = kNaN;
  check_bad("one NaN row", one_row, base);
  MustReject("Compare of empty logits", [&] { (void)Compare({}, {}, V); });
}

void TestStrictOracleStyleChecks() {
  // The two idioms the tests combine: Exceeds(max_abs, limit) and top1 >= 1.
  const Stats bad = Compare(std::vector<float>(2 * 8, kNaN), Logits(2, 8), 8);
  Require(Exceeds(bad.max_abs, 1e30) && !(bad.top1 >= 1.0), "bad stats fail the usual conditions");
}

}  // namespace

int main() {
  return Run([] {
    TestIdiomsAreHoles();
    TestMaxAbsDiff();
    TestMaxAndExceeds();
    TestStrictArgMax();
    TestExcessOverAndBitEqual();
    TestErrAndReports();
    TestTwinMargin();
    TestCompare();
    TestStrictOracleStyleChecks();
    std::cout << "nan_guard: MaxAbsDiff, MaxKeepNaN, Exceeds, StrictArgMax, ExcessOver, BitEqualFinite, "
                 "Err/ReportAbs/ReportRel/RequireTwinFails, Compare reject NaN and Inf and pass clean data\n";
  });
}

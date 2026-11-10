#ifndef RICURSIVE_TESTS_CHECK_HPP_
#define RICURSIVE_TESTS_CHECK_HPP_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace ricursive_test {

/// Assertion that stays active in optimized builds.
inline void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

/// Requires `text` to contain `needle` (used for error-message checks).
inline void RequireContains(const std::string& text, const std::string& needle,
                            const std::string& what) {
  Require(text.find(needle) != std::string::npos,
          what + ": expected \"" + needle + "\" in \"" + text + "\"");
}

template<class Fn>
int Run(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}

// ---- seeded random data (the same sequence on every machine) -------------------

class Rng {
 public:
  explicit Rng(std::uint64_t seed) : s_(seed) {}
  std::uint64_t Next() {
    s_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = s_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double Uniform() { return static_cast<double>(Next() >> 11) * (1.0 / 9007199254740992.0); }
  std::uint32_t Below(std::uint32_t n) { return static_cast<std::uint32_t>(Next() % n); }
  double Normal() {
    if (have_) {
      have_ = false;
      return spare_;
    }
    double u1 = Uniform();
    if (u1 < 1e-300) u1 = 1e-300;
    const double u2 = Uniform();
    const double r = std::sqrt(-2.0 * std::log(u1));
    spare_ = r * std::sin(6.283185307179586 * u2);
    have_ = true;
    return r * std::cos(6.283185307179586 * u2);
  }

 private:
  std::uint64_t s_;
  bool have_ = false;
  double spare_ = 0;
};

// ---- NaN / Inf safe comparison helpers ------------------------------------------------
// A NaN compares false with everything: `d > tol` is false, std::max(0, NaN) is 0,
// `a[i] > a[best]` never selects a NaN row. A check written with those idioms accepts
// a device result that is NaN everywhere. Every comparison in the tests goes through
// the helpers below (or the `!(d <= tol)` idiom), and tests/nan_guard_test.cpp feeds
// them NaN and Inf.

/// True when every value is finite (not NaN, not +-Inf).
inline bool AllFinite(std::span<const float> v) {
  for (const float x : v)
    if (!std::isfinite(x)) return false;
  return true;
}

/// Max |a[i] - b[i]|, or +infinity when any value of either input is NaN or +-Inf (so
/// two runs that agree on a non-finite value still fail). Never NaN. Sizes must match.
inline double MaxAbsDiff(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size()) throw std::runtime_error("compared spans differ in size");
  double m = 0;
  bool bad = false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
      bad = true;
      continue;
    }
    const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > m) m = d;
  }
  return bad ? std::numeric_limits<double>::infinity() : m;
}

/// Maximum that keeps a NaN: NaN if either argument is NaN (std::max drops it).
inline double MaxKeepNaN(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::numeric_limits<double>::quiet_NaN();
  return a > b ? a : b;
}

/// True when the measure `d` is NOT within `tol`; NaN is not within any tolerance.
inline bool Exceeds(double d, double tol) { return !(d <= tol); }

/// Index of the largest logit, or nullopt when the span is empty or holds a NaN/Inf
/// (an all-NaN row must not agree with any reference argmax).
inline std::optional<std::size_t> StrictArgMax(std::span<const float> v) {
  if (v.empty() || !AllFinite(v)) return std::nullopt;
  std::size_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i)
    if (v[i] > v[best]) best = i;
  return best;
}

/// max(0, |got - ref| - allow) with +infinity for any non-finite input
/// (std::max(0.0, NaN) is 0.0).
inline double ExcessOver(double got, double ref, double allow) {
  if (!std::isfinite(got) || !std::isfinite(ref) || !std::isfinite(allow))
    return std::numeric_limits<double>::infinity();
  return std::max(0.0, std::fabs(got - ref) - allow);
}

/// Bytes equal AND every float finite (two runs that are NaN in the same place are
/// not "equal" in any sense a test cares about).
inline bool BitEqualFinite(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && AllFinite(a) && AllFinite(b) &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// ---- error statistics -------------------------------------------------------------

/// Max abs error of `got` against `ref`, and the largest |ref| seen. A NaN in
/// `got` or `ref` counts as infinite error (a NaN must never pass a check).
struct Err {
  double max_abs = 0;
  double max_ref = 0;
  std::uint64_t n = 0;
  void Add(double got, double ref) {
    ++n;
    double e = std::fabs(got - ref);
    if (std::isnan(e)) e = std::numeric_limits<double>::infinity();
    if (e > max_abs) max_abs = e;
    const double a = std::fabs(ref);
    if (a > max_ref && std::isfinite(a)) max_ref = a;
  }
  /// Max abs error relative to the largest reference value.
  double Rel() const { return max_ref > 0 ? max_abs / max_ref : max_abs; }
  void Merge(const Err& o) {
    n += o.n;
    max_abs = std::max(max_abs, o.max_abs);
    max_ref = std::max(max_ref, o.max_ref);
  }
};

inline std::string Sci(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.3g", v);
  return buf;
}

/// Prints one result line and requires max abs error <= tol_abs.
inline void ReportAbs(const std::string& what, const Err& e, double tol_abs) {
  Require(e.n > 0, what + ": nothing was compared");
  std::cout << "  " << what << ": max abs " << Sci(e.max_abs) << ", rel to max|ref| "
            << Sci(e.Rel()) << " (n=" << e.n << "), tolerance " << Sci(tol_abs) << "\n";
  Require(!Exceeds(e.max_abs, tol_abs), what + ": max abs error " + Sci(e.max_abs) +
                                    " exceeds " + Sci(tol_abs));
}

/// Prints one result line and requires the error relative to max|ref| <= tol.
inline void ReportRel(const std::string& what, const Err& e, double tol_rel) {
  Require(e.n > 0, what + ": nothing was compared");
  std::cout << "  " << what << ": max abs " << Sci(e.max_abs) << ", rel to max|ref| "
            << Sci(e.Rel()) << " (n=" << e.n << "), tolerance " << Sci(tol_rel) << " rel\n";
  Require(!Exceeds(e.Rel(), tol_rel), what + ": relative error " + Sci(e.Rel()) + " exceeds " +
                                  Sci(tol_rel));
}

// ---- broken twins ------------------------------------------------------------------

struct TwinRecord {
  std::string name;
  double good;
  double twin;
  double factor;
};
inline std::vector<TwinRecord>& Twins() {
  static std::vector<TwinRecord> t;
  return t;
}

/// A deliberately wrong variant must be far from the kernel: its error (same
/// measure as the correct reference's) has to be at least `factor` times the
/// correct error, where the correct error is floored at `floor` (the
/// resolution of fp32 data) so an exact kernel does not make the ratio
/// meaningless. Records the achieved factor.
inline void RequireTwinFails(const std::string& name, double good, double twin,
                             double floor = 1e-7, double factor = 100.0) {
  Require(std::isfinite(good), "twin '" + name + "': the correct kernel's error is not finite (" +
                                   Sci(good) + "), so the comparison proves nothing");
  const double ratio = twin / std::max(good, floor);
  std::cout << "  twin '" << name << "': error " << Sci(twin) << " vs correct "
            << Sci(good) << " (floor " << Sci(floor) << "), factor " << Sci(ratio)
            << " (need >= " << Sci(factor) << ")\n";
  Twins().push_back({name, good, twin, ratio});
  Require(twin >= factor * std::max(good, floor) && !std::isnan(twin),
          "twin '" + name + "' is not rejected by a factor of " + Sci(factor));
}

inline void PrintTwinSummary() {
  if (Twins().empty()) return;
  double lo = std::numeric_limits<double>::infinity();
  std::string who;
  for (const auto& t : Twins()) {
    if (t.factor < lo) {
      lo = t.factor;
      who = t.name;
    }
  }
  std::cout << Twins().size() << " broken twins, smallest margin " << Sci(lo)
            << "x (" << who << ")\n";
}

// ---- GPU helpers (HIP translation units only) ---------------------------------------

#if defined(__HIPCC__)

inline void HipCheck(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

/// Device buffer of n elements of T (hipMalloc), freed on destruction.
template<class T>
class DevBuf {
 public:
  explicit DevBuf(std::size_t n) : n_(n) {
    HipCheck(hipMalloc(reinterpret_cast<void**>(&p_), std::max<std::size_t>(n, 1) * sizeof(T)),
             "hipMalloc");
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  ~DevBuf() { (void)hipFree(p_); }
  T* get() const { return p_; }
  std::size_t size() const { return n_; }
  void Upload(const std::vector<T>& v) {
    Require(v.size() == n_, "DevBuf::Upload size");
    HipCheck(hipMemcpy(p_, v.data(), n_ * sizeof(T), hipMemcpyHostToDevice), "upload");
  }
  std::vector<T> Download() const {
    std::vector<T> v(n_);
    HipCheck(hipMemcpy(v.data(), p_, n_ * sizeof(T), hipMemcpyDeviceToHost), "download");
    return v;
  }
  /// Fills every byte (0xFF makes unwritten fp32 / fp16 elements NaN).
  void FillBytes(int byte) { HipCheck(hipMemset(p_, byte, n_ * sizeof(T)), "memset"); }

 private:
  T* p_ = nullptr;
  std::size_t n_;
};

/// Runs `fn` when a GPU device is present, otherwise returns 77 (skip).
template<class Fn>
int RunGpu(Fn&& fn) {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::cout << "SKIP: no GPU device\n";
    return 77;
  }
  return Run(fn);
}

#endif  // __HIPCC__

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_CHECK_HPP_

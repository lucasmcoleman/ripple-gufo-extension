#ifndef RICURSIVE_TESTS_ORACLE_COMMON_HPP_
#define RICURSIVE_TESTS_ORACLE_COMMON_HPP_

// Loading of fixtures, token files and llama.cpp logits, and the comparison
// statistics shared by the oracle, quantized and incremental tests.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "src/config.hpp"
#include "src/reference.hpp"
#include "src/weights.hpp"
#include "tests/check.hpp"
#include "tests/paths.hpp"

namespace ricursive_test {

template<class T>
std::vector<T> ReadBinary(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  Require(in.good(), "cannot open " + path);
  const auto bytes = static_cast<std::size_t>(in.tellg());
  Require(bytes % sizeof(T) == 0, "bad size: " + path);
  std::vector<T> v(bytes / sizeof(T));
  in.seekg(0);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
  return v;
}

/// A GGUF opened and resolved; keeps the reader alive beside the table.
struct Model {
  std::unique_ptr<gufo::core::GgufReader> reader;
  gufo::models::ricursive::ModelWeights weights;
};

inline Model OpenModel(const std::string& path) {
  Model m;
  std::string error;
  m.reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(m.reader != nullptr, "open " + path + ": " + error);
  const auto config = gufo::models::ricursive::ParseConfig(*m.reader, &error);
  Require(config.has_value(), path + ": " + error);
  auto w = gufo::models::ricursive::ResolveWeights(*m.reader, *config, &error);
  Require(w.has_value(), path + ": " + error);
  m.weights = std::move(*w);
  return m;
}

inline std::vector<std::int32_t> LoadTokens(const std::string& name, int seq) {
  return ReadBinary<std::int32_t>(OracleOutDir() + "/" + name + ".seq" +
                                  std::to_string(seq) + ".i32");
}
/// llama.cpp logits [T][vocab] for fixture `name` ("tinyB-Q8_0" etc.).
inline std::vector<float> LoadLlama(const std::string& name, int seq) {
  return ReadBinary<float>(OracleOutDir() + "/" + name + ".seq" +
                           std::to_string(seq) + ".llama.f32");
}

struct Stats {
  double max_abs{0}, mean_abs{0}, mean_kl{0}, top1{0};
  bool finite{true};
};

/// Compares `ref` against `base` (llama.cpp): |difference|, mean KL(base ||
/// ref) over positions, and the fraction of positions with equal argmax.
inline Stats Compare(const std::vector<float>& ref, const std::vector<float>& base,
                     std::size_t vocab) {
  Require(ref.size() == base.size() && ref.size() % vocab == 0,
          "logit size mismatch: " + std::to_string(ref.size()) + " vs " +
              std::to_string(base.size()));
  const std::size_t T = ref.size() / vocab;
  Stats s;
  double sum_abs = 0, sum_kl = 0;
  std::size_t agree = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const float* a = &ref[t * vocab];
    const float* b = &base[t * vocab];
    std::size_t am = 0, bm = 0;
    double amax = a[0], bmax = b[0];
    for (std::size_t i = 0; i < vocab; ++i) {
      if (!std::isfinite(a[i]) || !std::isfinite(b[i])) s.finite = false;
      const double d = std::abs(static_cast<double>(a[i]) - b[i]);
      s.max_abs = std::max(s.max_abs, d);
      sum_abs += d;
      if (a[i] > a[am]) am = i;
      if (b[i] > b[bm]) bm = i;
      amax = std::max<double>(amax, a[i]);
      bmax = std::max<double>(bmax, b[i]);
    }
    double za = 0, zb = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
      za += std::exp(a[i] - amax);
      zb += std::exp(b[i] - bmax);
    }
    double kl = 0;
    for (std::size_t i = 0; i < vocab; ++i) {
      const double lpb = (b[i] - bmax) - std::log(zb);
      const double lpa = (a[i] - amax) - std::log(za);
      kl += std::exp(lpb) * (lpb - lpa);
    }
    sum_kl += kl;
    if (am == bm) ++agree;
  }
  s.mean_abs = sum_abs / static_cast<double>(ref.size());
  s.mean_kl = sum_kl / static_cast<double>(T);
  s.top1 = static_cast<double>(agree) / static_cast<double>(T);
  if (s.max_abs != s.max_abs) s.finite = false;
  return s;
}

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_ORACLE_COMMON_HPP_

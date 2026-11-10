// Shared by the engine, continuation and chat end-to-end tests.
#ifndef RICURSIVE_TESTS_ENGINE_UTIL_HPP_
#define RICURSIVE_TESTS_ENGINE_UTIL_HPP_

#include <cstring>
#include <set>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gufo/ext/text.hpp>

#include "src/cpu_executor.hpp"
#include "src/engine.hpp"
#include "tests/check.hpp"
#include "tests/oracle_common.hpp"

namespace ricursive_test {

using namespace gufo::models::ricursive;
namespace cache = gufo::cache;
namespace ext = gufo::ext;
/// The engine's Model (oracle_common.hpp has an unrelated ricursive_test::Model).
using EngineModel = gufo::models::ricursive::Model;

struct TinyFixture {
  std::string name;  // "tinyA" / "tinyB"
  std::string path;
  std::uint32_t window{0};
};

inline std::vector<TinyFixture> TinyFixtures() {
  return {{"tinyA", FixtureDir() + "/tinyA.gguf", 8}, {"tinyB", FixtureDir() + "/tinyB.gguf", 16}};
}

/// A CPU-backend model (template check off: the tiny files have none).
/// `faults` builds every session's executor with deliberate defects.
inline std::shared_ptr<EngineModel> LoadCpu(const std::string& path, std::uint32_t context,
                                      std::uint32_t rows, std::uint32_t chunk = 4,
                                      const CpuExecutor::Faults* faults = nullptr) {
  ModelOptions options;
  options.max_context = context;
  options.prefill_rows = rows;
  options.window_chunk_rows = chunk;
  options.allow_template_mismatch = true;
  options.backend = CpuBackend();
  if (faults != nullptr) {
    const CpuExecutor::Faults f = *faults;
    options.executor_override = [f](const ModelWeights& w, const ExecutorConfig& c,
                                    std::string*) -> std::unique_ptr<Executor> {
      return std::make_unique<CpuExecutor>(w, c, f);
    };
  }
  std::string error;
  auto model = EngineModel::Load(path, options, &error);
  Require(model != nullptr, "load " + path + ": " + error);
  return model;
}

inline std::unique_ptr<Session> NewSession(const std::shared_ptr<EngineModel>& model) {
  std::string error;
  auto session = model->CreateSession(&error);
  Require(session != nullptr, "session: " + error);
  return session;
}

inline std::int32_t Argmax(std::span<const float> logits) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) best = i;
  }
  return static_cast<std::int32_t>(best);
}

inline double MaxAbsDiff(std::span<const float> a, std::span<const float> b) {
  Require(a.size() == b.size(), "compared spans differ in size");
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::abs(static_cast<double>(a[i]) - b[i]);
    if (!(d <= m)) m = d;  // NaN propagates
  }
  return m;
}

/// Greedy continuation after `prompt`, by running the scalar Reference's full
/// forward on the whole sequence for every token (no cache, no ring).
inline std::vector<std::int32_t> ReferenceChain(const ModelWeights& weights,
                                                std::vector<std::int32_t> sequence, std::size_t count) {
  const Reference ref(weights);
  std::vector<std::int32_t> out;
  const std::size_t vocab = weights.config.vocab_size;
  for (std::size_t i = 0; i < count; ++i) {
    const auto logits = ref.ForwardFull(sequence);
    const auto last = std::span<const float>(logits).subspan(logits.size() - vocab);
    out.push_back(Argmax(last));
    sequence.push_back(out.back());
  }
  return out;
}

/// Greedy continuation through a Session.
inline std::vector<std::int32_t> SessionChain(Session& session, std::span<const std::int32_t> prompt,
                                              std::size_t count) {
  std::string error;
  Require(session.Sync(prompt, &error), "sync: " + error);
  std::vector<std::int32_t> out;
  for (std::size_t i = 0; i < count; ++i) {
    out.push_back(Argmax(session.Logits()));
    if (i + 1 < count) Require(session.Evaluate(out.back(), &error), "evaluate: " + error);
  }
  return out;
}

inline std::vector<std::int32_t> Slice(const std::vector<std::int32_t>& v, std::size_t a, std::size_t b) {
  return {v.begin() + static_cast<std::ptrdiff_t>(a), v.begin() + static_cast<std::ptrdiff_t>(b)};
}

inline std::size_t Distinct(const std::vector<std::int32_t>& v) {
  return std::set<std::int32_t>(v.begin(), v.end()).size();
}

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_ENGINE_UTIL_HPP_

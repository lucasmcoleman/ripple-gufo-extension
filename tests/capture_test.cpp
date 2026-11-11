// Hidden-state capture (docs/DRAFTER-STAGE0.md) on the tiny fixtures, host ops
// and scalar CPU executor. CPU only.
//
//   1. taps against the scalar Reference: the captured layer outputs equal the
//      Reference's "l_out" for the same rows (existing host-ops tap tolerance);
//   2. layout and identity: the capture equals what a "l_out" tap callback sees
//      in the same Forward, bit for bit, in any layer order;
//   3. capture off changes nothing: logits are bit-identical without
//      capture_layers, with capture_layers but no buffer, and with a buffer;
//   4. passes of 16 rows reproduce passes of 1 row: logits row hashes and
//      captured rows are equal;
//   5. refusals: wrong buffer size, buffer without capture_layers, bad layer
//      lists. The executor is unchanged after a refusal.
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>

#include "src/cpu_executor.hpp"
#include "src/row_hash.hpp"
#include "tests/executor_test_backend.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;

namespace {

using Tokens = std::vector<std::int32_t>;
constexpr std::uint32_t kContext = 96;
constexpr std::uint32_t kSeq = 64;

struct NamedBackend {
  std::string name;
  std::function<std::unique_ptr<Executor>(const Model&, const ExecutorConfig&)> make;
};

ExecutorConfig Cfg(std::uint32_t max_rows, std::vector<std::uint32_t> capture) {
  ExecutorConfig c;
  c.max_context = kContext;
  c.max_rows = max_rows;
  c.max_logit_rows = max_rows;
  c.kv_type = KvType::kF32;
  c.exact = true;
  c.capture_layers = std::move(capture);
  return c;
}

struct Pass {
  std::vector<float> logits;    ///< [kSeq][vocab]
  std::vector<float> capture;   ///< [tap][kSeq][hidden], assembled from the passes
};

/// kSeq tokens in passes of `rows` rows, all-row logits. `give_buffer`: pass a
/// capture buffer (needs capture_layers). `tap_check`: also verify against an
/// "l_out" tap callback in the same Forward.
Pass Feed(Executor& e, const Tokens& t, std::uint32_t rows, bool give_buffer, bool tap_check) {
  const std::size_t V = e.Info().vocab, H = e.Info().hidden, K = e.Info().capture_layers.size();
  Pass r;
  r.logits.resize(kSeq * V);
  r.capture.assign(K * kSeq * H, 0.0F);
  for (std::uint32_t at = 0; at < kSeq; at += rows) {
    const std::uint32_t n = std::min(rows, kSeq - at);
    std::vector<float> cap(give_buffer ? K * n * H : 0);
    std::map<int, std::vector<float>> seen;
    Tap tap;
    if (tap_check) {
      tap = [&](std::string_view name, int layer, const float* d, std::size_t rws, std::size_t w) {
        if (name == "l_out") seen[layer].assign(d, d + rws * w);
      };
    }
    e.Forward(std::span(t).subspan(at, n), at, LogitRows::kAll,
              std::span(r.logits).subspan(std::size_t{at} * V, std::size_t{n} * V), tap,
              std::span<float>(cap));
    if (give_buffer) {
      for (std::size_t k = 0; k < K; ++k) {
        std::copy_n(cap.data() + k * n * H, n * H, r.capture.data() + (k * kSeq + at) * H);
        if (tap_check) {
          const auto it = seen.find(static_cast<int>(e.Info().capture_layers[k]));
          Require(it != seen.end(), "no l_out tap for a captured layer");
          Require(it->second.size() == std::size_t{n} * H &&
                      std::equal(cap.begin() + static_cast<std::ptrdiff_t>(k * n * H),
                                 cap.begin() + static_cast<std::ptrdiff_t>((k + 1) * n * H),
                                 it->second.begin()),
                  "captured rows differ from the l_out tap of the same Forward (layer " +
                      std::to_string(e.Info().capture_layers[k]) + ")");
        }
      }
    }
  }
  return r;
}

bool SameBits(std::span<const float> a, std::span<const float> b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

void Check(const NamedBackend& be, const std::string& name, const std::vector<std::uint32_t>& layers,
           const ExecBackend& host) {
  const Model m = OpenModel(FixtureDir() + "/" + name + ".gguf");
  const std::string base = name.substr(0, 5);
  const Tokens t = LoadTokens(base, 0);
  Require(t.size() >= kSeq, "token file too short");
  const std::size_t V = m.weights.config.vocab_size, H = m.weights.config.embedding_length;
  const std::string tag = be.name + "/" + name;

  // 1. against the Reference (full-sequence forward).
  std::map<int, std::vector<float>> want;
  const Reference ref(m.weights);
  (void)ref.ForwardFull(std::span(t).first(kSeq), [&](std::string_view n, int l, const float* d,
                                                      std::size_t r, std::size_t w) {
    if (n == "l_out") want[l].assign(d, d + r * w);
  });
  auto e16 = be.make(m, Cfg(16, layers));
  const Pass cap16 = Feed(*e16, t, 16, true, true);  // also check 2 (tap identity)
  double worst = 0;
  for (std::size_t k = 0; k < layers.size(); ++k) {
    const auto& w = want.at(static_cast<int>(layers[k]));
    Require(w.size() == kSeq * H, "reference l_out has the wrong size");
    double mx = 0, ab = 0;
    for (std::size_t i = 0; i < w.size(); ++i) {
      mx = std::max(mx, std::abs(static_cast<double>(w[i])));
      ab = std::max(ab, std::abs(static_cast<double>(w[i]) -
                                 static_cast<double>(cap16.capture[k * kSeq * H + i])));
    }
    worst = std::max(worst, ab / std::max(mx, 1e-6));
  }
  std::printf("  %s: capture vs Reference l_out: worst rel %.3g (tolerance %.3g)\n", tag.c_str(),
              worst, host.tap_rel_tol);
  Require(worst <= host.tap_rel_tol, tag + ": captured layer outputs differ from the Reference");

  // 3. capture off changes nothing.
  auto plain = be.make(m, Cfg(16, {}));
  auto armed = be.make(m, Cfg(16, layers));
  const Pass a = Feed(*plain, t, 16, false, false);
  const Pass b = Feed(*armed, t, 16, false, false);
  Require(SameBits(a.logits, b.logits), tag + ": capture_layers without a buffer changed the logits");
  Require(SameBits(a.logits, cap16.logits), tag + ": a capture buffer changed the logits");
  std::printf("  %s: logits identical without capture_layers, with them unused, and capturing\n",
              tag.c_str());

  // 4. 16-row passes equal 1-row passes (logit row hashes and captured rows).
  auto e1 = be.make(m, Cfg(16, layers));
  const Pass cap1 = Feed(*e1, t, 1, true, false);
  for (std::size_t p = 0; p < kSeq; ++p) {
    const auto h16 = RowHash(std::span<const float>(cap16.logits).subspan(p * V, V));
    const auto h1 = RowHash(std::span<const float>(cap1.logits).subspan(p * V, V));
    Require(h16 == h1, tag + ": logits row " + std::to_string(p) + " differs between 16-row and 1-row passes");
  }
  Require(SameBits(cap16.capture, cap1.capture), tag + ": captured rows differ between 16-row and 1-row passes");
  std::printf("  %s: %u logits row hashes and all captured rows equal between 16-row and 1-row passes\n",
              tag.c_str(), kSeq);

  // 5. refusals.
  const auto position = e16->Position();
  const std::size_t K = layers.size();
  std::vector<float> wrong(K * 2 * H + 1);
  std::vector<float> row(V * 2);
  const Tokens two(t.begin(), t.begin() + 2);
  bool refused = false;
  e16->Reset();
  try {
    e16->Forward(two, 0, LogitRows::kAll, row, {}, wrong);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  Require(refused && e16->Position() == 0 && e16->State() == ExecState::kReady,
          tag + ": a capture buffer of the wrong size was not refused cleanly");
  refused = false;
  try {
    std::vector<float> some(2 * H);
    plain->Reset();
    plain->Forward(two, 0, LogitRows::kAll, row, {}, some);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  Require(refused && plain->Position() == 0, tag + ": a capture buffer without capture_layers was not refused");
  (void)position;
  for (const auto& bad : std::vector<std::vector<std::uint32_t>>{
           {1, 1}, {m.weights.config.block_count}, {0, 1, 2, 3, 4, 5}}) {
    refused = false;
    try {
      (void)be.make(m, Cfg(16, bad));
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    Require(refused, tag + ": a bad capture_layers list was accepted");
  }
  std::printf("  %s: refusals ok\n", tag.c_str());
}

}  // namespace

int main() {
  return Run([] {
    const ExecBackend host = TestBackend();
    const std::vector<NamedBackend> backends = {
        {"hostops", [&](const Model& m, const ExecutorConfig& c) {
           return host.make(m, c, ExecFaults{}, false);
         }},
        {"cpu", [](const Model& m, const ExecutorConfig& c) -> std::unique_ptr<Executor> {
           return std::make_unique<CpuExecutor>(m.weights, c);
         }},
    };
    for (const auto& be : backends) {
      Check(be, "tinyB", {0, 1, 2, 3, 4}, host);
      Check(be, "tinyB", {4, 0, 2}, host);  // order is the slot order
      Check(be, "tinyA", {1, 3, 5, 7, 9}, host);
    }
  });
}

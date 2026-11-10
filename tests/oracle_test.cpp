// CPU reference vs llama.cpp on the tiny random F32 fixtures, plus negative
// controls: every deliberately wrong variant must fail the same comparison.
#include <cstdio>
#include <iostream>
#include <map>

#include "tests/oracle_common.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;

namespace {

// Pass thresholds. The weakest negative control (symmetric clamp on tinyB)
// moves logits by about 0.1, so 2e-3 leaves a margin of ~50x.
constexpr double kMaxAbs = 2e-3;
constexpr int kSeqs = 4;
const char* const kFixtures[] = {"tinyA", "tinyB"};

struct Case {
  std::string fixture;
  Model model;
  std::vector<std::vector<std::int32_t>> tokens;
  std::vector<std::vector<float>> llama;
};

bool Passes(const Stats& s) { return s.finite && s.max_abs <= kMaxAbs && s.top1 == 1.0; }

}  // namespace

int main() {
  return Run([] {
    std::vector<Case> cases;
    for (const char* f : kFixtures) {
      const std::string path = FixtureDir() + "/" + f + ".gguf";
      if (!std::filesystem::exists(path)) {
        std::cout << "SKIP: fixture " << path << " not found\n";
        std::exit(77);
      }
      Case c{f, OpenModel(path), {}, {}};
      for (int s = 0; s < kSeqs; ++s) {
        c.tokens.push_back(LoadTokens(f, s));
        c.llama.push_back(LoadLlama(f, s));
      }
      cases.push_back(std::move(c));
    }

    std::cout << "reference vs llama.cpp (F32 fixtures), threshold max abs <= "
              << kMaxAbs << " and top-1 100%\n";
    double worst = 0;
    for (const Case& c : cases) {
      const Reference ref(c.model.weights);
      for (int s = 0; s < kSeqs; ++s) {
        const auto logits = ref.ForwardFull(c.tokens[s]);
        const Stats st = Compare(logits, c.llama[s], c.model.weights.config.vocab_size);
        std::printf("  %s seq%d: T=%zu max_abs=%.3e mean_abs=%.3e mean_kl=%.3e top1=%.1f%%\n",
                    c.fixture.c_str(), s, c.tokens[s].size(), st.max_abs,
                    st.mean_abs, st.mean_kl, st.top1 * 100);
        Require(Passes(st), c.fixture + " seq" + std::to_string(s) + " exceeds the threshold");
        worst = std::max(worst, st.max_abs);
      }
    }
    std::printf("PASS main comparison: worst max_abs = %.3e (limit %.0e)\n", worst, kMaxAbs);

    std::cout << "negative controls (each must FAIL the same comparison; margin = max abs diff)\n";
    int cannot_fail = 0;
    for (const Variant v : AllVariants()) {
      std::map<std::string, std::pair<double, double>> per_fixture;  // max_abs, min top1
      std::map<std::string, bool> fails;
      for (const Case& c : cases) {
        const Reference ref(c.model.weights, v);
        double mx = 0, top1 = 1;
        bool any_fail = false;
        for (int s = 0; s < kSeqs; ++s) {
          const auto logits = ref.ForwardFull(c.tokens[s]);
          const Stats st = Compare(logits, c.llama[s], c.model.weights.config.vocab_size);
          mx = std::max(mx, st.max_abs);
          top1 = std::min(top1, st.top1);
          any_fail = any_fail || !Passes(st);
        }
        per_fixture[c.fixture] = {mx, top1};
        fails[c.fixture] = any_fail;
      }
      const std::string name(VariantName(v));
      std::printf("  %-20s", name.c_str());
      for (const Case& c : cases) {
        std::printf(" %s: max_abs=%.3e min_top1=%.1f%% %s |", c.fixture.c_str(),
                    per_fixture[c.fixture].first, per_fixture[c.fixture].second * 100,
                    fails[c.fixture] ? "FAILS" : "passes");
      }
      std::printf("\n");
      if (v == Variant::kVresOnVEquiv) {
        // Mathematically identical to the spec: must agree, not fail.
        for (const Case& c : cases) {
          Require(!fails[c.fixture], name + " is equivalent to the spec and must pass on " + c.fixture);
        }
        continue;
      }
      bool any = false;
      for (const Case& c : cases) any = any || fails[c.fixture];
      Require(any, name + " did not fail on any fixture: the comparison cannot see it");
      for (const Case& c : cases) {
        if (!fails[c.fixture]) {
          ++cannot_fail;
          std::printf("      note: %s passes on %s: head_count_kv=%u there, so h %% kv == h / group for that fixture\n",
                      name.c_str(), c.fixture.c_str(), c.model.weights.config.head_count_kv);
          Require(v == Variant::kGqaModulo && c.model.weights.config.head_count_kv == 1,
                  name + " passed on " + c.fixture + " without a structural reason");
        }
      }
    }
    std::printf("PASS negative controls (%d variant/fixture pairs cannot fail by construction)\n",
                cannot_fail);
  });
}

// Quantized tinyB fixtures: the reference runs on dequantized weights and is
// compared with llama.cpp on the same file. REPORT ONLY: llama.cpp quantizes
// activations in its CPU kernels, so equality is not expected. The test fails
// only if a result is not finite.
#include <cstdio>

#include "tests/oracle_common.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;

int main() {
  return Run([] {
    const char* const quants[] = {"tinyB-Q4_K_M", "tinyB-Q5_K_M", "tinyB-Q6_K", "tinyB-Q8_0"};
    for (const char* q : quants) {
      const std::string path = FixtureDir() + "/" + q + ".gguf";
      if (!std::filesystem::exists(path)) {
        std::cout << "SKIP: " << path << " not found\n";
        std::exit(77);
      }
      const Model m = OpenModel(path);
      const Reference ref(m.weights);
      double mx = 0, sum_abs = 0, sum_kl = 0, sum_top1 = 0;
      for (int s = 0; s < 4; ++s) {
        // Tokens are shared with the F32 tinyB fixture of the same name stem.
        const auto tokens = LoadTokens("tinyB", s);
        const auto logits = ref.ForwardFull(tokens);
        const Stats st = Compare(logits, LoadLlama(q, s), m.weights.config.vocab_size);
        Require(st.finite, std::string(q) + " seq" + std::to_string(s) + ": non-finite result");
        std::printf("  %-13s seq%d: max_abs=%.3e mean_abs=%.3e mean_kl=%.3e top1=%.1f%%\n", q, s,
                    st.max_abs, st.mean_abs, st.mean_kl, st.top1 * 100);
        mx = std::max(mx, st.max_abs);
        sum_abs += st.mean_abs;
        sum_kl += st.mean_kl;
        sum_top1 += st.top1;
      }
      std::printf("  %-13s all : max_abs=%.3e mean_abs=%.3e mean_kl=%.3e top1=%.1f%%\n", q, mx,
                  sum_abs / 4, sum_kl / 4, sum_top1 / 4 * 100);
    }
  });
}

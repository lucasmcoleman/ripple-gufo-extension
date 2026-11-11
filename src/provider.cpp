#include <gufo/ext/generated/ricursive_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "probe.hpp"
#include "runner.hpp"

// The provider registration named by extension.json (`providers[].entry`).
void ricursive_provider_register(gufo::ext::Registry& registry) {
  namespace r = gufo::models::ricursive;
  registry.AddProvider(gufo::ext::generated::extension_ricursive::kProvider_ricursive);
  registry.AddTextProvider({
      .id = "ricursive",
      .probe = &r::ProbeRicursive,
      .probe_gguf = &r::ProbeRicursiveGguf,
      .create = &r::CreateRicursiveTextRunner,
      .bench_driver = &r::RunRicursiveBench,
      .prompt_driver = &r::RunRicursivePrompt,
      .chat_driver = &r::RunRicursiveChat,
      // The model card names no sampling settings: gufo's defaults apply.
      .sampling_defaults = nullptr,
      // DFlash2: at most 4 drafted tokens per step (the cost model, DRAFTER-GUFO-DESIGN.md
      // section 5, puts the optimum at d = 3 to 6), a fixed length.
      .speculative_defaults = {{.backend = "dflash2",
                                .max_draft_tokens = 4,
                                .min_draft_tokens = 1,
                                .draft_policy = "fixed",
                                .draft_policies = {"fixed"}}},
  });
}

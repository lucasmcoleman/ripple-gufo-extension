#include <gufo/ext/generated/ricursive_constants.hpp>
#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>
#include <memory>
#include <string>

#include "probe.hpp"

namespace {

std::shared_ptr<gufo::ext::TextModelRunner> CreateRicursiveRunner(
    const gufo::ext::TextRunnerOptions&, std::string* error) {
  if (error != nullptr) {
    *error = "ricursive: runner not implemented (extension skeleton)";
  }
  return nullptr;
}

}  // namespace

void ricursive_provider_register(gufo::ext::Registry& registry) {
  namespace r = gufo::models::ricursive;
  registry.AddProvider(
      gufo::ext::generated::extension_ricursive::kProvider_ricursive);
  registry.AddTextProvider({
      .id = "ricursive",
      .probe = &r::ProbeRicursive,
      .probe_gguf = &r::ProbeRicursiveGguf,
      .create = &CreateRicursiveRunner,
  });
}

void ricursive_register(gufo::ext::Registry& registry) { (void)registry; }

#include <gufo/ext/registry.hpp>
#include <gufo/ext/text.hpp>

#include "output_parser.hpp"

// The provider (probe, runner factory, drivers) registers from provider.cpp.
// The tool dialect is host-side and must exist before any runner is created.
void ricursive_register(gufo::ext::Registry& registry) {
  (void)registry;
  gufo::ext::RegisterToolDialect(gufo::models::ricursive::MakeToolDialect());
}

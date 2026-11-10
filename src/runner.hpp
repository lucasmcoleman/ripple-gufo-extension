#ifndef RICURSIVE_RUNNER_HPP_
#define RICURSIVE_RUNNER_HPP_

#include <chrono>
#include <gufo/ext/options.hpp>
#include <gufo/ext/text.hpp>
#include <memory>
#include <stdexcept>
#include <string>

#include "engine.hpp"

namespace gufo::models::ricursive {

/// Applies the options `extension.json` declares to `options`; an option that
/// was not given keeps the default. Throws std::invalid_argument.
inline void ApplyProviderOptions(const ext::ParsedOptions& given, ModelOptions& options) {
  if (const auto* cap = ext::FindOption<std::int64_t>(given, "--context-cap")) {
    if (*cap <= 0 || *cap > static_cast<std::int64_t>(UINT32_MAX)) {
      throw std::invalid_argument("--context-cap must be between 1 and 4294967295");
    }
    options.max_context = static_cast<std::uint32_t>(*cap);
  }
  if (const auto* allow = ext::FindOption<bool>(given, "--allow-chat-template-mismatch")) {
    options.allow_template_mismatch = *allow;
  }
}

/// `TextProvider::create`: the serving runner over component-cache sessions.
std::shared_ptr<ext::TextModelRunner> CreateRicursiveTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

/// `gufo prompt`, `gufo chat` and `gufo bench` on one session, outside the
/// serving pool (drivers.cpp).
int RunRicursivePrompt(const ext::TextPromptOptions& options, const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point load_start);
int RunRicursiveChat(const ext::TextPromptOptions& options, const core::GgufReader& reader,
                     std::chrono::steady_clock::time_point load_start);
int RunRicursiveBench(const ext::TextBenchOptions& options,
                      const std::shared_ptr<const core::GgufReader>& reader,
                      std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_RUNNER_HPP_

#ifndef RICURSIVE_PROBE_HPP_
#define RICURSIVE_PROBE_HPP_

#include <gufo/ext/gguf_reader.hpp>
#include <string_view>

namespace gufo::models::ricursive {

/// True when `general.architecture == "ricursive"`.
[[nodiscard]] bool ProbeRicursiveGguf(const core::GgufReader& reader);
/// `model_root` is a GGUF file path.
[[nodiscard]] bool ProbeRicursive(std::string_view model_root);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_PROBE_HPP_

#include "hip/drafter_engine.hpp"

#include <stdexcept>
#include <string>

namespace gufo::models::ricursive {

void OpsDrafterEngine::Append(std::uint32_t first_position, std::span<const float> features) {
  if (first_position != End()) {
    throw std::invalid_argument("drafter: rows appended at position " + std::to_string(first_position) +
                                " but the context ends at " + std::to_string(End()));
  }
  const std::size_t width = FeatureWidth();
  if (width == 0 || features.size() % width != 0) {
    throw std::invalid_argument("drafter: feature rows of the wrong width");
  }
  exec_->AppendContext(features, features.size() / width);
}

DrafterEngine::Block OpsDrafterEngine::Draft(std::int32_t anchor_token, std::uint32_t anchor_position,
                                             std::uint32_t max_tokens) {
  auto draft = exec_->DraftStep(anchor_token, anchor_position, max_tokens);
  Block block;
  block.tokens = std::move(draft.tokens);
  block.context_first = draft.context_first;
  block.context_end = draft.context_end;
  return block;
}

}  // namespace gufo::models::ricursive

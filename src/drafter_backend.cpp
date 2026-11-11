#include "drafter_backend.hpp"

#include <algorithm>

namespace gufo::models::ricursive {

void DrafterIngest::Rows(std::uint32_t first_position, std::uint32_t rows,
                         std::span<const float> data) noexcept {
  if (rows == 0) return;
  try {
    if (first_position < engine_->End()) engine_->TruncateTo(first_position);
    if (first_position != engine_->End()) engine_->ResetTo(first_position);  // a hole
    engine_->Append(first_position, data);
    rows_given_ += rows;
  } catch (...) {
    ++failures_;
    try {
      engine_->ResetTo(first_position + rows);
    } catch (...) {
    }
  }
}

void DrafterIngest::Rewound(std::uint32_t position) noexcept {
  try {
    engine_->TruncateTo(position);
  } catch (...) {
    ++failures_;
  }
}

void DrafterIngest::Replaced(std::uint32_t position) noexcept {
  try {
    engine_->ResetTo(position);
  } catch (...) {
    ++failures_;
  }
}

DFlash2DraftBackend::DFlash2DraftBackend(std::unique_ptr<DrafterEngine> engine, DraftBackendFaults faults)
    : engine_(std::move(engine)), faults_(faults) {}

speculative::DraftProposal DFlash2DraftBackend::Propose(std::span<const sampling::TokenId> prompt_tokens,
                                                        std::uint32_t current_pos, std::uint32_t max_tokens) {
  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  last_block_ = {};
  if (max_tokens == 0) return proposal;
  if (prompt_tokens.size() != std::size_t{current_pos} + 1 || engine_->End() != current_pos) {
    ++desyncs_;
    engine_->ResetTo(current_pos);
    return proposal;
  }
  const std::size_t anchor_index = faults_.anchor_token_off_by_one && prompt_tokens.size() > 1
                                       ? prompt_tokens.size() - 2
                                       : prompt_tokens.size() - 1;
  const std::uint32_t anchor_position =
      faults_.anchor_position_off_by_one && current_pos > 0 ? current_pos - 1 : current_pos;
  last_anchor_position_ = current_pos;
  last_anchor_token_ = prompt_tokens.back();
  last_block_ = engine_->Draft(static_cast<std::int32_t>(prompt_tokens[anchor_index]), anchor_position);
  ++proposals_;
  const std::size_t count = std::min<std::size_t>(max_tokens, last_block_.tokens.size());
  for (std::size_t i = 0; i < count; ++i) {
    proposal.tokens.push_back(static_cast<sampling::TokenId>(last_block_.tokens[i]));
  }
  return proposal;
}

void DFlash2DraftBackend::AcceptFeedback(std::span<const sampling::TokenId>, sampling::TokenId) {}

}  // namespace gufo::models::ricursive

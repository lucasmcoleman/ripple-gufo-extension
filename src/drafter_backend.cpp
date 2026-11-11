#include "drafter_backend.hpp"

#include <algorithm>
#include <exception>
#include <iostream>

namespace gufo::models::ricursive {

namespace {

std::string Describe() noexcept {
  try {
    throw;
  } catch (const std::exception& e) {
    return e.what();
  } catch (...) {
    return "unknown exception";
  }
}

}  // namespace

void DrafterHealth::Failed(std::string_view where, std::string_view what, bool in_ingest) noexcept {
  try {
    ++failures_;
    ++consecutive_;
    if (in_ingest) failed_since_proposal_ = true;
    last_error_ = std::string(where) + ": " + std::string(what);
    if (consecutive_ >= kLimit && !disabled_) {
      disabled_ = true;
      std::cerr << "ricursive: drafting is switched off for this session after " << consecutive_
                << " consecutive drafter failures (last: " << last_error_
                << "); it generates without drafts from here on\n";
    }
  } catch (...) {
  }
}

void DrafterHealth::Clear() noexcept {
  try {
    before_clear_ = State{failures_, consecutive_, disabled_, failed_since_proposal_, last_error_};
  } catch (...) {
    before_clear_.reset();
  }
  failures_ = consecutive_ = 0;
  disabled_ = failed_since_proposal_ = false;
  last_error_.clear();
}

void DrafterHealth::Restored() noexcept {
  if (!before_clear_) return;
  try {
    last_error_ = before_clear_->last_error;
  } catch (...) {
  }
  failures_ = before_clear_->failures;
  consecutive_ = before_clear_->consecutive;
  disabled_ = before_clear_->disabled;
  failed_since_proposal_ = before_clear_->failed_since_proposal;
  before_clear_.reset();
}

void DrafterHealth::DraftSucceeded() noexcept {
  if (!failed_since_proposal_) consecutive_ = 0;
  failed_since_proposal_ = false;
}

void DrafterIngest::Rows(std::uint32_t first_position, std::uint32_t rows,
                         std::span<const float> data) noexcept {
  if (rows == 0 || (health_ != nullptr && health_->disabled())) return;
  try {
    if (first_position < engine_->End()) engine_->TruncateTo(first_position);
    if (first_position != engine_->End()) engine_->ResetTo(first_position);  // a hole
    engine_->Append(first_position, data);
    rows_given_ += rows;
  } catch (...) {
    ++failures_;
    if (health_ != nullptr) health_->Failed("ingest", Describe(), true);
    try {
      engine_->ResetTo(first_position + rows);
    } catch (...) {
    }
  }
}

void DrafterIngest::Rewound(std::uint32_t position) noexcept {
  if (health_ != nullptr && health_->disabled()) return;
  try {
    engine_->TruncateTo(position);
  } catch (...) {
    ++failures_;
    if (health_ != nullptr) health_->Failed("ingest", Describe(), true);
  }
}

void DrafterIngest::Replaced(std::uint32_t position) noexcept {
  if (health_ != nullptr) {
    if (position == 0) health_->Clear();  // a new conversation: try the drafter again
    else health_->Restored();             // a restore began with Replaced(0): it continues the conversation
  }
  if (health_ != nullptr && health_->disabled()) return;
  try {
    engine_->ResetTo(position);
  } catch (...) {
    ++failures_;
    if (health_ != nullptr) health_->Failed("ingest", Describe(), true);
  }
}

DFlash2DraftBackend::DFlash2DraftBackend(std::unique_ptr<DrafterEngine> engine, DraftBackendFaults faults,
                                         DrafterHealth* health)
    : engine_(std::move(engine)), faults_(faults), health_(health) {}

speculative::DraftProposal DFlash2DraftBackend::Propose(std::span<const sampling::TokenId> prompt_tokens,
                                                        std::uint32_t current_pos, std::uint32_t max_tokens) {
  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  last_block_ = {};
  if (max_tokens == 0) return proposal;
  if (health_ != nullptr && health_->disabled()) return proposal;
  // The engine is a device object that can fail (memory, a lost queue): that costs speed, never the request. A
  // failed call becomes an empty proposal and the context restarts at the current position.
  const auto reset = [&](std::string_view where) noexcept {
    try {
      engine_->ResetTo(current_pos);
    } catch (...) {
      if (health_ != nullptr) health_->Failed(where, Describe(), false);
    }
  };
  if (prompt_tokens.size() != std::size_t{current_pos} + 1 || engine_->End() != current_pos) {
    ++desyncs_;
    reset("reset after a desync");
    return proposal;
  }
  const std::size_t anchor_index = faults_.anchor_token_off_by_one && prompt_tokens.size() > 1
                                       ? prompt_tokens.size() - 2
                                       : prompt_tokens.size() - 1;
  const std::uint32_t anchor_position =
      faults_.anchor_position_off_by_one && current_pos > 0 ? current_pos - 1 : current_pos;
  last_anchor_position_ = current_pos;
  last_anchor_token_ = prompt_tokens.back();
  try {
    last_block_ = engine_->Draft(static_cast<std::int32_t>(prompt_tokens[anchor_index]), anchor_position, max_tokens);
  } catch (...) {
    last_block_ = {};
    if (health_ != nullptr) health_->Failed("draft", Describe(), false);
    reset("reset after a failed draft");
    return proposal;
  }
  if (health_ != nullptr) health_->DraftSucceeded();
  ++proposals_;
  const std::size_t count = std::min<std::size_t>(max_tokens, last_block_.tokens.size());
  for (std::size_t i = 0; i < count; ++i) {
    proposal.tokens.push_back(static_cast<sampling::TokenId>(last_block_.tokens[i]));
  }
  return proposal;
}

void DFlash2DraftBackend::AcceptFeedback(std::span<const sampling::TokenId>, sampling::TokenId) {}

}  // namespace gufo::models::ricursive

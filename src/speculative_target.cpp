#include "speculative_target.hpp"

#if defined(ENGINE_ENABLE_HIP)

#include <algorithm>
#include <stdexcept>

namespace gufo::models::ricursive {
namespace {

[[noreturn]] void Fail(const std::string& what, const std::string& error) {
  throw std::runtime_error("ricursive speculative target: " + what + ": " + error);
}

std::int32_t Int(sampling::TokenId token) {
  if (token > static_cast<sampling::TokenId>(INT32_MAX)) {
    throw std::invalid_argument("ricursive token ID exceeds engine range");
  }
  return static_cast<std::int32_t>(token);
}

sampling::TokenId Greedy(std::span<const float> logits) {
  sampling::SamplerState greedy;
  return greedy.Sample(logits);
}

}  // namespace

SessionSpeculativeTarget::SessionSpeculativeTarget(Session& session, Faults faults)
    : session_(&session), faults_(faults) {}

void SessionSpeculativeTarget::Settle() {
  if (!session_->VerificationOpen()) return;
  std::string error;
  if (!session_->AcceptVerified(open_rows_, &error)) Fail("settling a verification", error);
  open_rows_ = 0;
}

void SessionSpeculativeTarget::Reset() noexcept {
  session_->Reset();
  open_rows_ = 0;
}

sampling::TokenId SessionSpeculativeTarget::ForwardPromptBatch(std::span<const sampling::TokenId>) {
  throw std::logic_error("ricursive speculative target: prompts go through Session::Sync");
}

sampling::TokenId SessionSpeculativeTarget::ForwardToken(sampling::TokenId token, std::uint32_t position,
                                                         bool compute_logits) {
  Settle();
  if (position != session_->Position()) {
    throw std::logic_error("ricursive speculative target: ForwardToken at " + std::to_string(position) +
                           " but the session is at " + std::to_string(session_->Position()));
  }
  std::string error;
  if (!session_->Evaluate(Int(token), &error)) Fail("decode", error);
  if (observer_) observer_(position, session_->Logits());
  return compute_logits ? Greedy(session_->Logits()) : 0;
}

void SessionSpeculativeTarget::SaveState(std::uint32_t valid_context) { saved_position_ = valid_context; }

void SessionSpeculativeTarget::RestoreState() {
  // The rejected rows are dropped by CommitVerificationChunk (an exact rewind), not here.
}

std::span<const float> SessionSpeculativeTarget::CopyLastLogits() { return session_->Logits(); }

std::span<const float> SessionSpeculativeTarget::CopyVerificationLogits(std::size_t row) {
  const std::size_t vocab = session_->model().VocabSize();
  const auto all = session_->VerifyLogits();
  if ((row + 1) * vocab > all.size()) throw std::out_of_range("verification logits row");
  return all.subspan(row * vocab, vocab);
}

std::size_t SessionSpeculativeTarget::VocabularySize() const noexcept { return session_->model().VocabSize(); }

speculative::VerificationChunkResult SessionSpeculativeTarget::ForwardVerificationChunk(
    std::span<const sampling::TokenId> candidate_tokens, std::uint32_t start_position, bool capture_hidden,
    bool capture_logits) {
  Settle();
  if (capture_hidden) {
    throw std::logic_error("ricursive speculative target: features reach the drafter through the session's tap sink");
  }
  if (start_position != session_->Position()) {
    throw std::logic_error("ricursive speculative target: verification at " + std::to_string(start_position) +
                           " but the session is at " + std::to_string(session_->Position()));
  }
  std::vector<std::int32_t> tokens;
  tokens.reserve(candidate_tokens.size());
  for (const auto t : candidate_tokens) tokens.push_back(Int(t));
  std::string error;
  if (!session_->Verify(tokens, &error)) Fail("verification", error);
  open_rows_ = static_cast<std::uint32_t>(tokens.size());
  const std::size_t vocab = session_->model().VocabSize();
  const auto logits = session_->VerifyLogits();
  if (observer_) observer_(start_position, logits);
  speculative::VerificationChunkResult result;
  result.predictions.reserve(tokens.size());
  for (std::size_t row = 0; row < tokens.size(); ++row) {
    result.predictions.push_back(Greedy(logits.subspan(row * vocab, vocab)));
  }
  if (capture_logits) {
    result.logits.assign(logits.begin(), logits.end());
    result.vocab_size = vocab;
  }
  return result;
}

void SessionSpeculativeTarget::CommitVerificationChunk(std::span<const sampling::TokenId> committed_tokens,
                                                       std::uint32_t start_position) {
  if (!session_->VerificationOpen() || start_position != saved_position_) {
    throw std::logic_error("ricursive speculative target: commit without the verification it belongs to");
  }
  std::uint32_t keep = static_cast<std::uint32_t>(committed_tokens.size());
  if (faults_.accept_one_more && keep < open_rows_) ++keep;
  std::string error;
  if (!session_->AcceptVerified(keep, &error)) Fail("commit", error);
  open_rows_ = 0;
}

sampling::TokenId SessionSpeculativeTarget::GetEosTokenId() const noexcept {
  return static_cast<sampling::TokenId>(kStopTokens[0]);
}

bool SessionSpeculativeTarget::IsStopToken(sampling::TokenId token) const noexcept {
  return token <= static_cast<sampling::TokenId>(INT32_MAX) && session_->model().IsStopToken(static_cast<std::int32_t>(token));
}

std::string_view SessionSpeculativeTarget::DecodeToken(sampling::TokenId token) const noexcept {
  try {
    text_ = session_->model().TokenText(Int(token));
  } catch (...) {
    text_.clear();
  }
  return text_;
}

SpeculativeDecoder::SpeculativeDecoder(Session& session, std::unique_ptr<speculative::IDraftBackend> backend,
                                       std::uint32_t max_draft_tokens, SessionSpeculativeTarget::Faults faults)
    : session_(&session),
      backend_(backend.get()),
      max_draft_tokens_(std::clamp<std::uint32_t>(max_draft_tokens, 1,
                                                  std::max<std::uint32_t>(session.model().MaxVerifyRows(), 2) - 1)) {
  if (session.model().MaxVerifyRows() < 2) {
    throw std::invalid_argument("ricursive speculative decoding needs ModelOptions::verify_rows >= 2");
  }
  auto target = std::make_unique<SessionSpeculativeTarget>(session, faults);
  target_ = target.get();
  speculative::SpeculativeOptions options;
  options.max_draft_tokens = max_draft_tokens_;
  options.min_draft_tokens = 1;
  options.initial_draft_tokens = max_draft_tokens_;
  options.use_batched_verification = true;
  options.enable_adaptive_draft_length = false;
  verifier_ = std::make_unique<speculative::SpeculativeVerifier>(std::move(target), std::move(backend), options);
}

void SpeculativeDecoder::Restart() {
  target_->Settle();
  published_ = false;
  verifier_->BeginRequest();
}

DecodeResult SpeculativeDecoder::Step(std::size_t max_tokens, sampling::SamplerState& sampler, bool stop_at_eos) {
  DecodeResult result;
  if (max_tokens == 0) return result;
  target_->Settle();
  Session& s = *session_;
  Model& model = s.model();
  std::string error;
  // A step that used its whole budget ended the request (gufo's scheduler passes the tokens the request may still
  // emit, and finishes it by length when the step returns them all): the pending token is evaluated in a one-row
  // pass, as plain decoding evaluates every token it advances, so that the session, its frontier logits and the
  // checkpoint are where plain generation leaves them and the next request starts from rows computed the same way.
  const auto finish_budget = [&] {
    if (published_ && result.tokens.size() >= max_tokens && !result.stop && s.Position() < model.MaxContext()) {
      (void)target_->ForwardToken(static_cast<sampling::TokenId>(frontier_), s.Position(), false);
      published_ = false;
    }
  };
  if (!sampler.config().can_use_unmodified_argmax() || sampler.config().logprobs) {
    // The plain path, as TextModelRunner::DecodeStep does it.
    if (published_) throw std::logic_error("ricursive: a drafted token is pending on a request that does not draft");
    if (s.Logits().empty()) throw std::runtime_error("ricursive token selection has no logits");
    const auto token = static_cast<std::int32_t>(sampler.Sample(s.Logits()));
    if (stop_at_eos && model.IsStopToken(token)) {
      result.stop = true;
      return result;
    }
    result.tokens.push_back(token);
    if (!s.Evaluate(token, &error)) throw std::runtime_error("ricursive decode failed: " + error);
    return result;
  }
  if (!published_) {
    if (s.Logits().empty()) throw std::runtime_error("ricursive token selection has no logits");
    sampling::SamplerState greedy;
    frontier_ = static_cast<std::int32_t>(greedy.Sample(s.Logits()));
    if (stop_at_eos && model.IsStopToken(frontier_)) {
      result.stop = true;
      return result;
    }
    result.tokens.push_back(frontier_);
    published_ = true;
    if (result.tokens.size() >= max_tokens) {
      finish_budget();
      return result;
    }
  }
  const std::uint32_t position = s.Position();
  sequence_.assign(s.tokens().begin(), s.tokens().end());
  sequence_.push_back(static_cast<sampling::TokenId>(frontier_));
  sampling::SamplerState working = sampler;
  const auto verification = verifier_->VerifyStep(
      std::span<const sampling::TokenId>(sequence_), position, static_cast<sampling::TokenId>(frontier_),
      target_->GetEosTokenId(), static_cast<std::uint32_t>(max_tokens - result.tokens.size()), working);
  target_->Settle();
  if (s.Position() != position + verification.emitted_tokens.size()) {
    throw std::logic_error("ricursive: the session is at " + std::to_string(s.Position()) + " after a step from " +
                           std::to_string(position) + " that emitted " +
                           std::to_string(verification.emitted_tokens.size()) + " tokens");
  }
  result.draft_tokens = verification.draft_count;
  result.accepted_tokens = verification.accepted_count;
  for (const auto token : verification.emitted_tokens) {
    if (stop_at_eos && model.IsStopToken(static_cast<std::int32_t>(token))) {
      result.stop = true;
      break;
    }
    result.tokens.push_back(static_cast<std::int32_t>(token));
  }
  frontier_ = static_cast<std::int32_t>(verification.next_token);
  published_ = !result.stop;
  finish_budget();
  return result;
}

}  // namespace gufo::models::ricursive

#endif  // ENGINE_ENABLE_HIP

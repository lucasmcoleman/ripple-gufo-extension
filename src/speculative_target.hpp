#ifndef RICURSIVE_SPECULATIVE_TARGET_HPP_
#define RICURSIVE_SPECULATIVE_TARGET_HPP_

// The target side of gufo's speculative verifier over a ricursive Session
// (docs/DRAFTER-S4.md), and the loop that drives gufo's own SpeculativeVerifier
// with it. The declarations of gufo's verifier exist only in HIP builds
// (ENGINE_ENABLE_HIP), which is how extensions see them; without it this header
// is empty and the runner refuses --speculative.

#if defined(ENGINE_ENABLE_HIP)

#include <functional>
#include <gufo/ext/draft_backend.hpp>
#include <gufo/ext/speculative_verifier.hpp>
#include <memory>
#include <string>
#include <vector>

#include "engine.hpp"

namespace gufo::models::ricursive {

/// ISpeculativeTargetExecutor over a Session. A verification is one Session::Verify
/// (k rows, all logits); a partial accept is Session::AcceptVerified (the exact rewind: no
/// accepted token is forwarded again). Greedy argmax of every row is computed on the host
/// with the sampler's greedy rule, the same function plain decoding selects with.
class SessionSpeculativeTarget final : public speculative::ISpeculativeTargetExecutor {
 public:
  /// Deliberate defects for tests only.
  struct Faults {
    bool accept_one_more{false};  ///< a partial commit keeps one more row than the verifier decided
  };
  explicit SessionSpeculativeTarget(Session& session) : SessionSpeculativeTarget(session, Faults{}) {}
  SessionSpeculativeTarget(Session& session, Faults faults);

  /// Keeps what is left of an open verification (after a step that accepted every row the
  /// verifier never calls CommitVerificationChunk): the session is settled and the rows'
  /// taps reach the tap sink. Idempotent; called before every other call that changes the
  /// session, and by whoever drives the verifier when a step ends.
  void Settle();

  /// Test seam: called with the logits rows of every pass the verifier asked for (a
  /// verification or a one-row step): the first position, the rows [n][vocab].
  using Observer = std::function<void(std::uint32_t first_position, std::span<const float> rows)>;
  void SetObserver(Observer observer) { observer_ = std::move(observer); }

  void Reset() noexcept override;
  [[nodiscard]] sampling::TokenId ForwardPromptBatch(std::span<const sampling::TokenId> prompt_tokens) override;
  [[nodiscard]] sampling::TokenId ForwardToken(sampling::TokenId token, std::uint32_t position,
                                               bool compute_logits) override;
  void SaveState(std::uint32_t valid_context) override;
  void RestoreState() override;
  [[nodiscard]] std::span<const float> CopyLastLogits() override;
  [[nodiscard]] std::span<const float> CopyVerificationLogits(std::size_t row) override;
  [[nodiscard]] std::size_t VocabularySize() const noexcept override;
  [[nodiscard]] speculative::VerificationChunkResult ForwardVerificationChunk(
      std::span<const sampling::TokenId> candidate_tokens, std::uint32_t start_position,
      bool capture_hidden, bool capture_logits) override;
  void CommitVerificationChunk(std::span<const sampling::TokenId> committed_tokens,
                               std::uint32_t start_position) override;
  [[nodiscard]] sampling::TokenId GetEosTokenId() const noexcept override;
  [[nodiscard]] bool IsStopToken(sampling::TokenId token) const noexcept override;
  [[nodiscard]] std::string_view DecodeToken(sampling::TokenId token) const noexcept override;

 private:
  Session* session_;
  Faults faults_;
  std::uint32_t saved_position_{0};
  std::uint32_t open_rows_{0};
  mutable std::string text_;
  Observer observer_;
};

struct DecodeResult {
  std::vector<std::int32_t> tokens;  ///< emitted, in order; a stop token ends the request and is not in it
  bool stop{false};
  std::size_t draft_tokens{0};
  std::size_t accepted_tokens{0};
};

/// One conversation's drafted decoding over a Session. The Session holds the tokens whose rows
/// the target has; the last emitted token may be "pending": emitted, not yet in the target (it
/// is the next verification's first row). `published()` says whether the greedy token of the
/// session's frontier logits was emitted already.
///
/// Requests the verifier treats as sampled (temperature, penalties, grammar, reasoning budget)
/// and requests that ask for log-probabilities draft nothing and run the plain path
/// (select, evaluate), exactly as without speculation.
class SpeculativeDecoder {
 public:
  /// `max_draft_tokens` is clamped to Model::MaxVerifyRows() - 1 (a verification is the current
  /// token plus the drafts). The draft backend must obey speculative::IDraftBackend.
  SpeculativeDecoder(Session& session, std::unique_ptr<speculative::IDraftBackend> backend,
                     std::uint32_t max_draft_tokens)
      : SpeculativeDecoder(session, std::move(backend), max_draft_tokens, SessionSpeculativeTarget::Faults{}) {}
  SpeculativeDecoder(Session& session, std::unique_ptr<speculative::IDraftBackend> backend,
                     std::uint32_t max_draft_tokens, SessionSpeculativeTarget::Faults faults);

  /// Emits up to `max_tokens` (>= 1) tokens. Throws std::runtime_error when the session fails.
  [[nodiscard]] DecodeResult Step(std::size_t max_tokens, sampling::SamplerState& sampler, bool stop_at_eos);
  /// A new request on this session (prefix reuse, a restored checkpoint, a prefill): nothing is
  /// pending any more.
  void Restart();
  [[nodiscard]] bool published() const noexcept { return published_; }
  [[nodiscard]] Session& session() noexcept { return *session_; }
  [[nodiscard]] SessionSpeculativeTarget& target() noexcept { return *target_; }
  [[nodiscard]] speculative::SpeculativeVerifier& verifier() noexcept { return *verifier_; }
  [[nodiscard]] speculative::IDraftBackend& backend() noexcept { return *backend_; }
  [[nodiscard]] std::uint32_t max_draft_tokens() const noexcept { return max_draft_tokens_; }

 private:
  Session* session_;
  SessionSpeculativeTarget* target_;
  speculative::IDraftBackend* backend_;
  std::unique_ptr<speculative::SpeculativeVerifier> verifier_;
  std::uint32_t max_draft_tokens_;
  bool published_{false};
  std::int32_t frontier_{0};  ///< the pending token when published_
  std::vector<sampling::TokenId> sequence_;
};

}  // namespace gufo::models::ricursive

#else  // no gufo verifier in this build: stand-ins so that callers compile; drafting is refused earlier

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gufo/ext/sampling.hpp>

namespace gufo::models::ricursive {

struct DecodeResult {
  std::vector<std::int32_t> tokens;
  bool stop{false};
  std::size_t draft_tokens{0};
  std::size_t accepted_tokens{0};
};

class SpeculativeDecoder {
 public:
  [[nodiscard]] DecodeResult Step(std::size_t, sampling::SamplerState&, bool) {
    throw std::logic_error("ricursive: this build has no speculative verifier");
  }
  void Restart() {}
};

}  // namespace gufo::models::ricursive

#endif  // ENGINE_ENABLE_HIP
#endif  // RICURSIVE_SPECULATIVE_TARGET_HPP_

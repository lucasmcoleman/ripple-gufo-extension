#ifndef RICURSIVE_DRAFTER_BACKEND_HPP_
#define RICURSIVE_DRAFTER_BACKEND_HPP_

// gufo's IDraftBackend over a small drafter interface of our own (docs/DRAFTER-S4.md).
//
//   DrafterEngine   what a drafter must do: hold the target features of a contiguous run
//                   of positions [Base(), End()), forget rows (truncate, reset to a base)
//                   and draft the block that follows an anchor. One implementation now:
//                   the stage 1 CPU reference (drafter_engine_cpu.hpp); the GPU drafter of
//                   stage 2 plugs in behind the same interface. Nothing here assumes the
//                   host: rows arrive in passes, the context has a bounded window.
//   DrafterIngest   TapSink -> DrafterEngine: the session's accepted rows become context.
//   DFlash2DraftBackend   speculative::IDraftBackend: Propose = Draft from the anchor.
//
// The drafter's output never changes what is emitted (the verifier checks every drafted
// token against the target); only how many are accepted depends on it.

#include <cstddef>
#include <cstdint>
#include <gufo/ext/draft_backend.hpp>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "drafter_config.hpp"
#include "engine.hpp"

namespace gufo::models::ricursive {

class DrafterEngine {
 public:
  virtual ~DrafterEngine() = default;

  [[nodiscard]] virtual const DrafterConfig& config() const noexcept = 0;
  /// Zero-based target layers whose outputs are the features, in feature order.
  [[nodiscard]] virtual std::vector<std::uint32_t> TapLayers() const = 0;
  /// Floats in one row of features: taps x hidden.
  [[nodiscard]] std::size_t FeatureWidth() const noexcept { return config().FeatureWidth(); }

  /// The context: the features of positions [Base(), End()).
  [[nodiscard]] virtual std::uint32_t Base() const noexcept = 0;
  [[nodiscard]] virtual std::uint32_t End() const noexcept = 0;
  /// Empties the context; the next row appended is position `base`.
  virtual void ResetTo(std::uint32_t base) = 0;
  /// Drops the rows at positions >= `position` (a position below Base() empties the
  /// context at that base).
  virtual void TruncateTo(std::uint32_t position) = 0;
  /// Appends the rows of the positions starting at `first_position`, which must be End();
  /// `features` is [rows][FeatureWidth()] float32.
  virtual void Append(std::uint32_t first_position, std::span<const float> features) = 0;

  struct Block {
    std::vector<std::int32_t> tokens;  ///< the drafted tokens after the anchor
    /// The context rows the draft read: positions [context_first, context_end).
    std::uint32_t context_first{0};
    std::uint32_t context_end{0};
  };
  /// Drafts the block whose first row is `anchor_token` at `anchor_position` (<= End()).
  /// The context it reads is the rows below the anchor that the engine holds.
  [[nodiscard]] virtual Block Draft(std::int32_t anchor_token, std::uint32_t anchor_position) = 0;
  /// The same, when the caller needs at most `max_tokens` of the block: an engine may stop its
  /// work after that many tokens and return only them (the tokens it returns are the first ones of
  /// the whole block's, whatever the hint). The default ignores the hint, so an engine written
  /// before it keeps returning the whole block.
  [[nodiscard]] virtual Block Draft(std::int32_t anchor_token, std::uint32_t anchor_position,
                                    std::uint32_t max_tokens) {
    (void)max_tokens;
    return Draft(anchor_token, anchor_position);
  }
  /// Device memory this engine's context and scratch hold (0: none, or not tracked).
  [[nodiscard]] virtual std::uint64_t DeviceBytes() const noexcept { return 0; }
};

/// What went wrong inside one session's drafter, shared by its ingest and its backend (docs/DRAFTER-S4.md,
/// "Follow-ups", item C). A drafter that throws costs speed, never the request: the failed draft becomes an empty
/// proposal, the context is reset to the current position and the failure is counted here. After kLimit
/// consecutive failures drafting is switched off for the session, with one warning line.
class DrafterHealth {
 public:
  static constexpr std::size_t kLimit = 3;
  /// Records one failure; `where` names the call, `what` the exception's message; `in_ingest`: it happened
  /// while rows were being fed (between two proposals), not inside a proposal.
  void Failed(std::string_view where, std::string_view what, bool in_ingest) noexcept;
  /// A proposal whose draft succeeded: ends a run of failures unless rows failed to arrive since the previous one.
  void DraftSucceeded() noexcept;
  [[nodiscard]] std::size_t failures() const noexcept { return failures_; }
  [[nodiscard]] std::size_t consecutive() const noexcept { return consecutive_; }
  [[nodiscard]] bool disabled() const noexcept { return disabled_; }
  [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

 private:
  std::size_t failures_{0}, consecutive_{0};
  bool disabled_{false}, failed_since_proposal_{false};
  std::string last_error_;
};

/// Feeds the rows a Session keeps into a DrafterEngine. A row at a position the engine
/// does not continue is a rewind (rows from there are replaced) or a hole (the context
/// restarts at that position): either way the engine never holds rows that are not the
/// session's.
class DrafterIngest final : public TapSink {
 public:
  /// `health` (optional) counts the calls that threw; once drafting is switched off the rows are not fed any more.
  explicit DrafterIngest(DrafterEngine& engine, DrafterHealth* health = nullptr) : engine_(&engine), health_(health) {}
  void Rows(std::uint32_t first_position, std::uint32_t rows, std::span<const float> data) noexcept override;
  void Rewound(std::uint32_t position) noexcept override;
  void Replaced(std::uint32_t position) noexcept override;
  /// Calls that threw inside the engine (the context was reset past them).
  [[nodiscard]] std::size_t failures() const noexcept { return failures_; }
  [[nodiscard]] std::size_t rows_given() const noexcept { return rows_given_; }

 private:
  DrafterEngine* engine_;
  DrafterHealth* health_;
  std::size_t failures_{0};
  std::size_t rows_given_{0};
};

struct DraftBackendFaults {
  bool anchor_position_off_by_one{false};  ///< drafts from the row below the anchor
  bool anchor_token_off_by_one{false};     ///< the block starts with the token before the anchor
};

class DFlash2DraftBackend final : public speculative::IDraftBackend {
 public:
  /// `health` (optional, must outlive the backend): where failures of the engine are counted. Without it a
  /// failure is still absorbed (an empty proposal) but not counted and never switches drafting off.
  explicit DFlash2DraftBackend(std::unique_ptr<DrafterEngine> engine, DraftBackendFaults faults = {},
                               DrafterHealth* health = nullptr);

  [[nodiscard]] DrafterEngine& engine() noexcept { return *engine_; }
  [[nodiscard]] const DrafterEngine& engine() const noexcept { return *engine_; }

  [[nodiscard]] std::string_view Name() const noexcept override { return "dflash2-ricursive"; }
  /// `prompt_tokens` is the sequence including the anchor (the current token, not yet in
  /// the target): size current_pos + 1. The context must end exactly at current_pos; if it
  /// does not (a bookkeeping slip) the context is reset to current_pos, nothing is
  /// proposed and `desyncs()` counts it: a plain step costs speed, never correctness.
  [[nodiscard]] speculative::DraftProposal Propose(std::span<const sampling::TokenId> prompt_tokens,
                                                   std::uint32_t current_pos,
                                                   std::uint32_t max_tokens) override;
  /// Features reach the engine through DrafterIngest (the session's tap sink), not through
  /// the verifier: the session knows which rows were kept after a rejection or a rewind.
  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override { return false; }
  void AcceptFeedback(std::span<const sampling::TokenId> accepted, sampling::TokenId correction) override;
  void BeginRequest() noexcept override { /* the context survives requests (prefix reuse) */ }

  [[nodiscard]] std::size_t proposals() const noexcept { return proposals_; }
  [[nodiscard]] std::size_t desyncs() const noexcept { return desyncs_; }
  /// Last Propose: the engine's block (all of it) and the rows it read.
  [[nodiscard]] const DrafterEngine::Block& last_block() const noexcept { return last_block_; }
  /// The anchor of the last Propose that drafted (the true one, whatever the faults).
  [[nodiscard]] std::uint32_t last_anchor_position() const noexcept { return last_anchor_position_; }
  [[nodiscard]] sampling::TokenId last_anchor_token() const noexcept { return last_anchor_token_; }

 private:
  std::unique_ptr<DrafterEngine> engine_;
  DraftBackendFaults faults_;
  DrafterHealth* health_;
  std::size_t proposals_{0};
  std::size_t desyncs_{0};
  DrafterEngine::Block last_block_;
  std::uint32_t last_anchor_position_{0};
  sampling::TokenId last_anchor_token_{0};
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_DRAFTER_BACKEND_HPP_

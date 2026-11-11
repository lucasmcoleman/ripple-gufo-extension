#ifndef RICURSIVE_ENGINE_HPP_
#define RICURSIVE_ENGINE_HPP_

// Model (weights table, tokenizer, chat template, executor settings) and
// Session (one conversation's tokens, cache and frontier logits) for
// Ripple-65B-Preview. All compute goes through `Executor`; nothing here knows
// whether it runs on the GPU or on the CPU test backend.

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gufo/ext/gguf_reader.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "chat_prompt.hpp"
#include "config.hpp"
#include "executor_factory.hpp"
#include "executor_iface.hpp"
#include "text/json_value.hpp"
#include "text/tokenizer.hpp"
#include "weights.hpp"

namespace gufo::models::ricursive {

/// Context cap when none is given. The model's nominal 2,097,152 positions
/// would need 128 GiB of F16 cache for the full-attention layers alone.
inline constexpr std::uint32_t kDefaultContextCap = 32768;
/// Tokens per Forward (prefill pass).
inline constexpr std::uint32_t kDefaultPrefillRows = 2048;
/// `<|im_end|>` and `<|endoftext|>`.
inline constexpr std::array<std::int32_t, 2> kStopTokens{100265, 100257};
/// Most rows one Session::Verify may evaluate (docs/DRAFTER-S3.md): the widest
/// pass the GPU executor still computes bit-identically to one-row passes.
inline constexpr std::uint32_t kMaxVerifyRows = 16;

struct ModelOptions {
  std::uint32_t max_context{0};  ///< 0: min(kDefaultContextCap, the model's own)
  std::uint32_t prefill_rows{kDefaultPrefillRows};
  /// Rows per cache chunk of the sliding-layer components; the ring is a
  /// multiple of it. Tests use small values to make the ring wrap.
  std::uint32_t window_chunk_rows{64};
  /// Accept a GGUF whose chat template differs from the pinned one (the
  /// renderer implements only that template, so prompts may then be wrong).
  bool allow_template_mismatch{false};
  /// Default: RICURSIVE_BACKEND (BackendFromEnvironment).
  std::optional<BackendChoice> backend;
  /// Use this reader instead of opening `path`.
  std::shared_ptr<const core::GgufReader> reader;
  /// Rows a Session::Verify may evaluate in one pass, and so the all-row logits
  /// the executor can return (docs/DRAFTER-S3.md). 0 (the default) = drafting
  /// off: the executor holds the last row's logits only, exactly as before.
  /// At most kMaxVerifyRows; clamped to the prefill rows.
  std::uint32_t verify_rows{0};
  /// Layers whose output the sessions capture (ExecutorConfig::capture_layers,
  /// at most kMaxCaptureLayers). Empty (the default) = capture off.
  std::vector<std::uint32_t> capture_layers;
  /// Test seam: builds each session's executor instead of CreateExecutor
  /// (for executors with deliberate defects).
  std::function<std::unique_ptr<Executor>(const ModelWeights&, const ExecutorConfig&, std::string*)>
      executor_override;
};

struct ChatPrompt {
  std::vector<std::int32_t> tokens;
  /// Tokens before the generation prompt `<|im_start|>assistant\n`; what the
  /// next turn of the conversation shares. 0 when it cannot be told.
  std::size_t stable_prefix_tokens{0};
};

class Session;

class Model : public std::enable_shared_from_this<Model> {
 public:
  /// Returns nullptr and sets *error. Checks the chat template hash.
  [[nodiscard]] static std::shared_ptr<Model> Load(const std::string& path,
                                                   const ModelOptions& options,
                                                   std::string* error);

  [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }
  [[nodiscard]] const Config& config() const noexcept { return weights_.config; }
  [[nodiscard]] const ::ricursive::text::Tokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] const core::GgufReader& reader() const noexcept { return *reader_; }
  [[nodiscard]] const std::shared_ptr<const core::GgufReader>& reader_ptr() const noexcept {
    return reader_;
  }
  [[nodiscard]] const std::string& ModelName() const noexcept { return name_; }
  [[nodiscard]] const BackendChoice& backend() const noexcept { return backend_; }
  [[nodiscard]] const ExecutorConfig& executor_config() const noexcept { return executor_config_; }
  /// Sliding-ring slots per layer, chunk rows of the window components.
  [[nodiscard]] std::uint32_t RingRows() const noexcept { return executor_config_.ring_rows; }
  [[nodiscard]] std::uint32_t WindowChunkRows() const noexcept { return window_chunk_rows_; }
  /// The context cap actually in force.
  [[nodiscard]] std::uint32_t MaxContext() const noexcept { return executor_config_.max_context; }
  [[nodiscard]] std::uint32_t PrefillRows() const noexcept { return executor_config_.max_rows; }
  /// Rows one Session::Verify may evaluate: ModelOptions::verify_rows clamped
  /// to the prefill rows (1 when drafting is off).
  [[nodiscard]] std::uint32_t MaxVerifyRows() const noexcept {
    return executor_config_.max_logit_rows;
  }
  [[nodiscard]] std::uint32_t VocabSize() const noexcept { return config().vocab_size; }
  [[nodiscard]] std::string_view TemplateSha256() const noexcept { return template_sha256_; }

  /// Bytes of K and V per token in the full-attention layers, and of the
  /// sliding rings, at this backend's cache element size.
  [[nodiscard]] std::uint64_t FullLayerBytesPerToken() const;
  [[nodiscard]] std::uint64_t RingBytes() const;
  /// One session's cache: rings + full layers at `context` positions.
  [[nodiscard]] std::uint64_t SessionBytes(std::uint32_t context) const;

  [[nodiscard]] bool IsStopToken(std::int32_t id) const noexcept;
  /// Text of one token (control tokens in their literal spelling).
  [[nodiscard]] std::string TokenText(std::int32_t id) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> ids) const;
  /// Raw text with special-token parsing, as llama.cpp does for completions.
  /// Chat input goes through PrepareChat instead.
  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  /// Renders and tokenizes a conversation; message text cannot produce
  /// control tokens. nullopt and *error where the template raises.
  [[nodiscard]] std::optional<ChatPrompt> PrepareChat(
      const ::ricursive::text::JsonValue& messages,
      const ::ricursive::text::JsonValue* tools,
      const ::ricursive::text::ChatRenderOptions& options,
      std::string* error) const;

  /// A session with its own executor, or nullptr and *error.
  [[nodiscard]] std::unique_ptr<Session> CreateSession(std::string* error);

 private:
  Model(const ModelWeights& weights, std::shared_ptr<const core::GgufReader> reader)
      : weights_(weights), reader_(std::move(reader)) {}

  ModelWeights weights_;
  std::function<std::unique_ptr<Executor>(const ModelWeights&, const ExecutorConfig&, std::string*)>
      executor_override_;
  std::shared_ptr<const core::GgufReader> reader_;
  std::unique_ptr<::ricursive::text::Tokenizer> tokenizer_;
  std::string name_;
  std::string template_sha256_;
  BackendChoice backend_;
  ExecutorConfig executor_config_;
  std::uint32_t window_chunk_rows_{64};
  std::vector<bool> stop_;
};

/// Hidden-state rows a session captured for positions it has not handed out yet.
struct TapRows {
  std::uint32_t first_position{0};  ///< position of the first row
  std::uint32_t rows{0};
  /// [row][tap][hidden] float32: the row of position first_position + i is the
  /// concatenation of the captured layers' outputs, in capture_layers order
  /// (the drafter's feature input).
  std::vector<float> data;
};

/// Consumer of the captured rows, instead of the session's pending buffer (see
/// Session::SetTapSink). It hears of exactly the rows the session keeps, in
/// position order, one pass at a time, so a long prompt never accumulates more
/// than one pass of rows. Called with the session's mutex held: it must not call
/// the session. It must not throw.
class TapSink {
 public:
  virtual ~TapSink() = default;
  /// Rows of the positions [first_position, first_position + rows), [row][tap][hidden].
  /// Rows of a verification arrive when it is settled, without the rejected ones.
  virtual void Rows(std::uint32_t first_position, std::uint32_t rows,
                    std::span<const float> data) noexcept = 0;
  /// A rewind: rows at positions >= `position` that were delivered are no longer the session's.
  virtual void Rewound(std::uint32_t position) noexcept = 0;
  /// The session's content was replaced (reset, failed pass, full recompute, restored
  /// checkpoint): the consumer holds no row that is the session's below `position`
  /// (0 after a reset or a failure, the restored position after a restore).
  virtual void Replaced(std::uint32_t position) noexcept = 0;
};

class Session {
 public:
  /// Called before every Forward with what it will overwrite (the component
  /// cache's MutationGuard), and with [0, max_context) before a full reset.
  /// May throw to refuse; nothing has changed then.
  using WriteGuard = std::function<void(const WritePlan&)>;

  Session(std::shared_ptr<Model> model, std::unique_ptr<Executor> executor);

  [[nodiscard]] Model& model() const noexcept { return *model_; }
  [[nodiscard]] Executor& executor() noexcept { return *executor_; }
  [[nodiscard]] const Executor& executor() const noexcept { return *executor_; }
  void SetWriteGuard(WriteGuard guard) { guard_ = std::move(guard); }
  /// Test seam: passed to every Forward (an exception thrown by it fails the
  /// forward, which is how tests inject a device error).
  void SetTap(Tap tap) { tap_ = std::move(tap); }

  /// Tokens whose rows the executor holds (empty after a failure).
  [[nodiscard]] const std::vector<std::int32_t>& tokens() const noexcept { return tokens_; }
  [[nodiscard]] std::uint32_t Position() const noexcept {
    return static_cast<std::uint32_t>(tokens_.size());
  }
  /// Logits after the last token (every pass computes them, so they exist
  /// after any successful Sync/Evaluate), empty when none are held.
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return logits_valid_ ? std::span<const float>(logits_) : std::span<const float>();
  }
  /// False after a failed forward until Reset() or a restore.
  [[nodiscard]] bool IsValid() const noexcept {
    return executor_->State() == ExecState::kReady;
  }

  /// Makes the session hold exactly `target` and its last logits. Keeps the
  /// common prefix (less one token, whose logits are needed), rewinds when
  /// the executor can, otherwise recomputes from position 0. On failure
  /// returns false with *error; a failed forward also empties the session.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> target, std::string* error);
  /// Appends one token and its logits.
  [[nodiscard]] bool Evaluate(std::int32_t token, std::string* error);
  void Reset() noexcept;

  // ---- target side of speculative decoding (docs/DRAFTER-S3.md) ----

  /// Evaluates `tokens` (1 to Model::MaxVerifyRows(), appended at Position() = p) in ONE
  /// pass and keeps every row's logits: VerifyLogits() row i holds the logits
  /// after tokens[i], bit for bit what Evaluate would have produced for it
  /// (Verify refuses a pass wider than the executor's batch_invariant_rows).
  /// The session then holds p + k tokens and Logits() are those of the last row;
  /// the verification stays open until AcceptVerified, or until any call that
  /// changes the session (which keeps all k tokens). Needs frontier logits (a
  /// successful Sync/Evaluate/AcceptVerified before it). On failure returns
  /// false with *error; a failed forward also empties the session.
  [[nodiscard]] bool Verify(std::span<const std::int32_t> tokens, std::string* error);
  /// Rows of the last Verify, [k][vocab]; empty before one and after any call
  /// that starts another pass, Sync, Reset or a restore. Still readable after
  /// AcceptVerified.
  [[nodiscard]] std::span<const float> VerifyLogits() const noexcept {
    return std::span<const float>(verify_logits_);
  }
  /// Keeps the first `accepted` (0 to k) tokens of the open verification and
  /// puts the session exactly where plain decoding would be at p + accepted:
  /// the executor's caches rewound (no recomputation), tokens, frontier logits
  /// (row accepted - 1, or the logits held before Verify when accepted is 0)
  /// and the pending taps of the rejected rows. False with *error when no
  /// verification is open or accepted > k.
  [[nodiscard]] bool AcceptVerified(std::uint32_t accepted, std::string* error);
  /// True between a successful Verify and the call that settles it.
  [[nodiscard]] bool VerificationOpen() const noexcept { return verify_open_; }

  /// Captured rows (ModelOptions::capture_layers) for the positions evaluated,
  /// and not yet taken, since the last call: prompt passes, 1-row steps and
  /// verification passes alike, trimmed to what is still in the session after
  /// a rewind. Empty when capture is off. `first_position` below the end of
  /// what the caller took before means a rewind happened: the rows replace the
  /// caller's from there. The rows of an unsettled verification are not
  /// released (std::logic_error). Rows of a restored cache are never captured.
  /// Memory: hidden * taps * 4 bytes per untaken position (100 KiB with five
  /// taps on the real model), so take them after every Sync on long prompts.
  [[nodiscard]] TapRows TakeTaps();
  /// With a sink attached the captured rows go to it (after each pass, or when a
  /// verification settles) and TakeTaps() stays empty. nullptr detaches. The sink
  /// must outlive the session or be detached first.
  void SetTapSink(TapSink* sink) noexcept { sink_ = sink; }

  /// Deliberate defects for tests only; every one must make a drafting test fail.
  struct Faults {
    bool rewind_from_end{false};     ///< AcceptVerified rewinds to the logical end - accepted, not p + accepted
    bool stale_frontier{false};      ///< AcceptVerified keeps the last row's logits as the frontier
    bool keep_rejected_taps{false};  ///< AcceptVerified leaves the rejected rows' taps pending
  };
  void SetFaults(Faults faults) noexcept { faults_ = faults; }

  /// Component-cache restore: after the executor's CommitImport succeeded.
  void AdoptRestored(std::vector<std::int32_t> tokens, std::vector<float> logits);
  /// Cache-side view of the frontier logits (for the private component).
  [[nodiscard]] bool HasLogits() const noexcept { return logits_valid_; }
  [[nodiscard]] std::span<const float> RawLogits() const noexcept { return logits_; }
  /// Held by every call that changes the executor, the tokens or the logits,
  /// and by the cache's row transfers, so a capture on another thread never
  /// reads a half-written state. Never held while the write guard runs (the
  /// guard waits for those readers).
  [[nodiscard]] std::mutex& mutex() noexcept { return mutex_; }

  /// An object that lives and dies with this session: the runner keeps its drafting machinery (the drafter's
  /// context and the tap sink feeding it) here, because it belongs to the thing that owns the cached rows, not to
  /// the state a request binds over it (docs/DRAFTER-S4.md, "Follow-ups", item A). Destroyed with the session,
  /// before the rest of it.
  [[nodiscard]] const std::shared_ptr<void>& attachment() const noexcept { return attachment_; }
  void SetAttachment(std::shared_ptr<void> attachment) noexcept { attachment_ = std::move(attachment); }

 private:
  void SyncImpl(std::span<const std::int32_t> target);
  void ForwardChunks(std::span<const std::int32_t> tokens);
  void Fail() noexcept;
  /// A call that changes the session settles an open verification (keeps all k).
  void SettleVerification() noexcept;
  void AppendTaps(std::uint32_t first_position, std::uint32_t rows, std::span<const float> capture);
  void TrimTaps(std::uint32_t position) noexcept;
  void ClearTaps() noexcept;
  /// Hands the pending rows to the sink (none attached: they stay pending).
  void FlushTaps() noexcept;

  std::shared_ptr<Model> model_;
  std::unique_ptr<Executor> executor_;
  WriteGuard guard_;
  Tap tap_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  bool logits_valid_{false};
  Faults faults_;
  // Open verification: the position it started at, k, and the frontier logits
  // from before it (what a rewind to its start restores).
  bool verify_open_{false};
  std::uint32_t verify_base_{0};
  std::uint32_t verify_count_{0};
  std::vector<float> saved_logits_;
  std::vector<float> verify_logits_;
  // Captured rows not yet taken: positions [taps_first_, taps_first_ + rows),
  // always ending at Position() when not empty; [row][tap][hidden].
  std::uint32_t taps_first_{0};
  std::uint32_t taps_rows_{0};
  std::vector<float> taps_;
  TapSink* sink_{nullptr};
  std::mutex mutex_;
  std::shared_ptr<void> attachment_;  ///< last: destroyed first
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_ENGINE_HPP_

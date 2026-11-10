#ifndef RICURSIVE_EXECUTOR_IFACE_HPP_
#define RICURSIVE_EXECUTOR_IFACE_HPP_

// Contract between the host side (engine, component-cache adapter, runner)
// and a compute backend (CPU: src/cpu_executor.*, GPU: hip/executor.*).
// Prose, call sequences and worked ring examples: docs/EXECUTOR-CONTRACT.md.
//
// One Executor is one session's KV state plus the means to extend it. The
// public methods are non-virtual: this class owns the position bookkeeping
// (logical end, ring high water, import coverage) and validates every call,
// so all backends share one implementation of the rules that broke before
// (LESSONS.md "Rewinding caches and handling failures"). A backend implements
// only the three data paths (DoForward, DoExport, DoImport).
//
// Error model.
// - Refusal: std::invalid_argument (bad argument) or std::logic_error (call
//   not allowed in the current state), thrown before anything is changed.
//   State, positions and every row are as before the call.
// - Failure: any exception escaping DoForward or DoImport. The executor is
//   then kFailed: Position() and RetainedBegin() read 0, every call except
//   Reset() and BeginImport() is refused, row contents are undefined.
// - Rewind() returns false (state unchanged) when it would need recomputation.
// - Every call is synchronous: when it returns, its device work has finished,
//   so no error can surface later against a state already committed.
// Calls on one Executor, and on Executors that share device scratch, must be
// serialized by the caller (Gemma 4 engine.cpp:430 holds the model mutex).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "config.hpp"

namespace gufo::models::ricursive {

enum class KvType : std::uint8_t { kF16, kF32 };
enum class KvPart : std::uint8_t { kKey, kValue };
/// Logit rows a Forward returns: none (inner prefill chunks), the last row
/// (generation), or every row (tests, at most max_logit_rows).
enum class LogitRows : std::uint8_t { kNone, kLast, kAll };
enum class ExecState : std::uint8_t { kReady, kImporting, kFailed };

/// Test tap, same signature as Reference's TapFn: `data` is host float32
/// [rows][width], `layer` is -1 for global taps, names as in reference.hpp.
/// Backends emit at least "l_out" for every layer while a tap is set. An
/// exception thrown by the tap fails the Forward (used to inject failures).
using Tap = std::function<void(std::string_view name, int layer,
                               const float* data, std::size_t rows,
                               std::size_t width)>;

/// Smallest ring that lets a Forward of `max_rows` rows write all its rows
/// before attending: the first row reads `window - 1` earlier keys, the last
/// write evicts the position `ring` before it.
[[nodiscard]] constexpr std::uint32_t MinRingRows(
    std::uint32_t window, std::uint32_t max_rows) noexcept {
  return window + max_rows - 1;
}

struct ExecutorConfig {
  std::uint32_t max_context{0};     ///< positions; the runner's context cap
  std::uint32_t max_rows{0};        ///< tokens per Forward
  std::uint32_t max_logit_rows{1};  ///< rows a kAll Forward may return
  std::uint32_t ring_rows{0};       ///< sliding-layer slots; 0 = MinRingRows
  KvType kv_type{KvType::kF16};     ///< kF32: exact test mode
  bool exact{false};  ///< no activation rounding at any width (test mode)
};

struct ExecutorInfo {
  std::uint32_t layers{0};
  std::uint32_t window{0};    ///< keys a sliding query sees, itself included
  std::vector<bool> sliding;  ///< per layer: ring of ring_rows, else linear
  std::uint32_t kv_width{0};  ///< values per K or V row (kv_heads*head_dim)
  std::uint32_t vocab{0};
  std::uint32_t max_context{0};
  std::uint32_t max_rows{0};
  std::uint32_t max_logit_rows{0};
  std::uint32_t ring_rows{0};
  KvType kv_type{KvType::kF16};
  /// A Forward of at most this many rows computes every row exactly as a
  /// one-row Forward would, bit for bit; wider Forwards may round
  /// activations (GPU W8A8 prefill above 16 rows). Equals max_rows when
  /// config.exact is set or the backend never rounds.
  std::uint32_t batch_invariant_rows{0};

  /// Bytes of one exported K or V row: kv_width values of kv_type, layout
  /// [kv_head][head_dim], K after q/k-norm and RoPE, V as projected.
  [[nodiscard]] std::size_t RowBytes() const noexcept {
    return std::size_t{kv_width} * (kv_type == KvType::kF16 ? 2 : 4);
  }
  /// First position a continuation at `position` reads in a sliding layer.
  [[nodiscard]] std::uint32_t WindowFirst(std::uint32_t position) const {
    return position >= window ? position - window + 1 : 0;
  }
  /// Validated from the model and a config. Throws std::invalid_argument.
  [[nodiscard]] static ExecutorInfo Make(const Config& model,
                                         const ExecutorConfig& config,
                                         std::uint32_t batch_invariant_rows);
};

/// Absolute positions [first, end).
struct RowRange {
  std::uint32_t first{0};
  std::uint32_t end{0};
  [[nodiscard]] std::uint32_t size() const noexcept { return end - first; }
  [[nodiscard]] bool empty() const noexcept { return end <= first; }
  bool operator==(const RowRange&) const = default;
};

/// What a Forward of `rows` rows will overwrite. `written`: its own
/// positions, in every layer (rows there may still hold a rewound branch).
/// `evicted`: retained sliding-layer positions whose ring slots it reuses;
/// afterwards RetainedBegin() == evicted.end when evicted is not empty.
/// The adapter calls MutationGuard::BeforeOverwrite for both before Forward.
struct WritePlan {
  RowRange written;
  RowRange evicted;
};

class Executor {
public:
  virtual ~Executor() = default;
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] const ExecutorInfo& Info() const noexcept { return info_; }
  [[nodiscard]] ExecState State() const noexcept { return state_; }
  /// Tokens whose rows are valid: the next Forward starts here.
  [[nodiscard]] std::uint32_t Position() const noexcept;
  /// First position whose sliding-layer rows can still be exported or
  /// rewound to; every full-layer row below Position() is kept. Equals
  /// max(import floor, high water - ring_rows), where the high water is the
  /// largest position ever written, NOT Position().
  [[nodiscard]] std::uint32_t RetainedBegin() const noexcept;

  /// Ranges a Forward of `rows` would overwrite now. Pure.
  [[nodiscard]] WritePlan PlanForward(std::uint32_t rows) const;

  /// Evaluates `tokens` at positions [first_position, +size()), writing their
  /// K/V rows in every layer. first_position must equal Position(); 1 <= rows
  /// <= max_rows; Position() + rows <= max_context. `logits` must hold
  /// exactly 0 (kNone), vocab (kLast) or rows * vocab (kAll) floats.
  void Forward(std::span<const std::int32_t> tokens,
               std::uint32_t first_position, LogitRows which,
               std::span<float> logits, const Tap& tap = {});

  /// Moves Position() back to `position` (<= Position(), else
  /// invalid_argument). Returns false, changing nothing, when a sliding layer
  /// no longer holds [WindowFirst(position), position).
  [[nodiscard]] bool Rewind(std::uint32_t position);
  /// Empty and ready from any state. Rows are not cleared.
  void Reset() noexcept;

  /// Copies rows [rows.first, rows.end) of one layer's K or V in position
  /// order. Requires kReady, rows.end <= Position() and, in sliding layers,
  /// rows.first >= RetainedBegin(); out.size() == rows.size() * RowBytes().
  void ExportRows(std::uint32_t layer, KvPart part, RowRange rows,
                  std::span<std::byte> out) const;

  /// Restore: BeginImport(n) (any state; 1 <= n <= max_context), then
  /// ImportRows for every layer and part, then CommitImport(). In between
  /// only ImportRows, Reset and BeginImport are allowed.
  void BeginImport(std::uint32_t position);
  /// rows.end <= n; in sliding layers rows.first >= n - ring_rows (clamped
  /// at 0). Ranges may arrive in any order and overlap.
  void ImportRows(std::uint32_t layer, KvPart part, RowRange rows,
                  std::span<const std::byte> bytes);
  /// True and kReady when every full layer is covered from 0 and every
  /// sliding layer from at most WindowFirst(n), up to n; RetainedBegin()
  /// becomes the highest contiguous coverage start. Otherwise false and
  /// kFailed (latched like cache::Adapter::Validate).
  [[nodiscard]] bool CommitImport();

  /// Deliberately wrong bookkeeping, for contract-test twins only.
  enum class LedgerFault : std::uint8_t {
    kNone,
    kRetentionFromEnd,   ///< RetainedBegin from Position(), not high water
    kFailureKeepsReady,  ///< a failed Forward leaves kReady
    kNoImportCoverage,   ///< CommitImport accepts any coverage
  };

protected:
  /// Throws std::invalid_argument for a ring below MinRingRows.
  explicit Executor(ExecutorInfo info, LedgerFault fault = LedgerFault::kNone);

  /// Arguments are validated. Write the K/V rows of every position in every
  /// layer (sliding: slot position % ring_rows; full: slot position) before
  /// any query reads them; write nothing else. Sliding query q reads
  /// [WindowFirst(q), q], full query q reads [0, q].
  virtual void DoForward(std::span<const std::int32_t> tokens,
                         std::uint32_t first_position, LogitRows which,
                         std::span<float> logits, const Tap& tap) = 0;
  virtual void DoExport(std::uint32_t layer, KvPart part, RowRange rows,
                        std::span<std::byte> out) const = 0;
  virtual void DoImport(std::uint32_t layer, KvPart part, RowRange rows,
                        std::span<const std::byte> bytes) = 0;

private:
  void RequireReady(const char* what) const;
  void CheckRows(std::uint32_t layer, std::uint32_t first, std::uint32_t end,
                 std::uint32_t limit, std::size_t bytes) const;
  [[nodiscard]] std::uint32_t RetainedFrom(std::uint32_t high) const noexcept;

  ExecutorInfo info_;
  LedgerFault fault_;
  ExecState state_{ExecState::kReady};
  std::uint32_t end_{0};    ///< logical end
  std::uint32_t high_{0};   ///< high water: largest position written + 1
  std::uint32_t floor_{0};  ///< lowest retained row after an import
  std::vector<std::vector<RowRange>> imported_;  ///< [layer * 2 + part]
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_EXECUTOR_IFACE_HPP_

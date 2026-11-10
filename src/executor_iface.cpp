#include "executor_iface.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace gufo::models::ricursive {
namespace {

void Refuse(bool bad, const std::string& message) {
  if (bad) throw std::invalid_argument("ricursive executor: " + message);
}

const char* StateName(ExecState s) {
  switch (s) {
    case ExecState::kReady: return "ready";
    case ExecState::kImporting: return "importing";
    case ExecState::kFailed: return "failed";
  }
  return "?";
}

/// Start of the contiguous run of `ranges` that reaches `end` (end if none).
std::uint32_t CoveredFrom(const std::vector<RowRange>& ranges,
                          std::uint32_t end) {
  std::uint32_t from = end;
  for (bool grew = true; grew;) {
    grew = false;
    for (const RowRange& r : ranges) {
      if (r.first < from && r.end >= from) {
        from = r.first;
        grew = true;
      }
    }
  }
  return from;
}

}  // namespace

ExecutorInfo ExecutorInfo::Make(const Config& model,
                                const ExecutorConfig& config,
                                std::uint32_t batch_invariant_rows) {
  Refuse(model.block_count == 0 || model.sliding.size() != model.block_count,
         "model has no layers or no sliding pattern");
  Refuse(model.sliding_window == 0, "sliding window is 0");
  Refuse(config.max_context == 0 || config.max_context > model.context_length,
         "max_context must be in [1, " + std::to_string(model.context_length) +
             "]");
  Refuse(config.max_rows == 0 || config.max_rows > config.max_context,
         "max_rows must be in [1, max_context]");
  Refuse(config.max_logit_rows == 0 || config.max_logit_rows > config.max_rows,
         "max_logit_rows must be in [1, max_rows]");
  ExecutorInfo info;
  info.layers = model.block_count;
  info.window = model.sliding_window;
  info.sliding = model.sliding;
  info.kv_width = model.KvWidth();
  info.vocab = model.vocab_size;
  info.max_context = config.max_context;
  info.max_rows = config.max_rows;
  info.max_logit_rows = config.max_logit_rows;
  info.ring_rows = config.ring_rows != 0
                       ? config.ring_rows
                       : MinRingRows(model.sliding_window, config.max_rows);
  info.kv_type = config.kv_type;
  info.batch_invariant_rows =
      config.exact ? config.max_rows
                   : std::clamp<std::uint32_t>(batch_invariant_rows, 1,
                                               config.max_rows);
  return info;
}

Executor::Executor(ExecutorInfo info, LedgerFault fault)
    : info_(std::move(info)), fault_(fault) {
  Refuse(info_.layers == 0 || info_.sliding.size() != info_.layers,
         "info has no layers or no sliding pattern");
  const std::uint32_t min_ring = MinRingRows(info_.window, info_.max_rows);
  Refuse(info_.ring_rows < min_ring,
         "ring_rows " + std::to_string(info_.ring_rows) + " < window + " +
             "max_rows - 1 = " + std::to_string(min_ring));
}

void Executor::RequireReady(const char* what) const {
  if (state_ != ExecState::kReady) {
    throw std::logic_error(std::string("ricursive executor: ") + what +
                           " needs a ready executor, state is " +
                           StateName(state_));
  }
}

std::uint32_t Executor::Position() const noexcept {
  return state_ == ExecState::kFailed ? 0 : end_;
}

std::uint32_t Executor::RetainedFrom(std::uint32_t high) const noexcept {
  const bool any = std::find(info_.sliding.begin(), info_.sliding.end(),
                             true) != info_.sliding.end();
  if (!any) return 0;
  return std::max(floor_, high > info_.ring_rows ? high - info_.ring_rows : 0);
}

std::uint32_t Executor::RetainedBegin() const noexcept {
  if (state_ != ExecState::kReady) return Position();
  return RetainedFrom(fault_ == LedgerFault::kRetentionFromEnd ? end_ : high_);
}

WritePlan Executor::PlanForward(std::uint32_t rows) const {
  RequireReady("PlanForward");
  Refuse(rows == 0 || rows > info_.max_rows ||
             rows > info_.max_context - end_,
         "a forward of " + std::to_string(rows) + " rows at " +
             std::to_string(end_) + " exceeds max_rows or max_context");
  WritePlan plan;
  plan.written = {end_, end_ + rows};
  const std::uint32_t before = RetainedBegin();
  const std::uint32_t after = RetainedFrom(std::max(high_, end_ + rows));
  plan.evicted = {before, std::max(before, after)};
  return plan;
}

void Executor::Forward(std::span<const std::int32_t> tokens,
                       std::uint32_t first_position, LogitRows which,
                       std::span<float> logits, const Tap& tap) {
  RequireReady("Forward");
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  Refuse(first_position != end_,
         "forward at " + std::to_string(first_position) + " but position is " +
             std::to_string(end_) + " (rewind first)");
  Refuse(tokens.empty() || tokens.size() > info_.max_rows ||
             rows > info_.max_context - end_,
         "a forward of " + std::to_string(tokens.size()) + " rows at " +
             std::to_string(end_) + " exceeds max_rows or max_context");
  for (const std::int32_t t : tokens) {
    Refuse(t < 0 || static_cast<std::uint32_t>(t) >= info_.vocab,
           "token id " + std::to_string(t) + " out of range");
  }
  const std::size_t want = which == LogitRows::kNone   ? 0
                           : which == LogitRows::kLast ? info_.vocab
                                                       : std::size_t{rows} *
                                                             info_.vocab;
  Refuse(which == LogitRows::kAll && rows > info_.max_logit_rows,
         "kAll logits for more than max_logit_rows rows");
  Refuse(logits.size() != want, "logits buffer has " +
                                    std::to_string(logits.size()) +
                                    " floats, expected " +
                                    std::to_string(want));
  // Invariant (kept by Rewind and CommitImport): the keys the first row
  // reads are retained.
  if (info_.WindowFirst(end_) < end_ && info_.WindowFirst(end_) < RetainedBegin())
    throw std::logic_error("ricursive executor: window before the position "
                           "is not retained");
  try {
    DoForward(tokens, first_position, which, logits, tap);
  } catch (...) {
    if (fault_ != LedgerFault::kFailureKeepsReady) state_ = ExecState::kFailed;
    throw;
  }
  end_ += rows;
  high_ = std::max(high_, end_);
}

bool Executor::Rewind(std::uint32_t position) {
  RequireReady("Rewind");
  Refuse(position > end_, "rewind to " + std::to_string(position) +
                              " beyond position " + std::to_string(end_));
  const std::uint32_t needed = info_.WindowFirst(position);
  if (needed < position && needed < RetainedBegin()) return false;
  end_ = position;
  return true;
}

void Executor::Reset() noexcept {
  state_ = ExecState::kReady;
  end_ = high_ = floor_ = 0;
  imported_.clear();
}

void Executor::CheckRows(std::uint32_t layer, std::uint32_t first,
                         std::uint32_t end, std::uint32_t limit,
                         std::size_t bytes) const {
  Refuse(layer >= info_.layers, "layer " + std::to_string(layer) +
                                    " out of range");
  Refuse(first > end || end > limit,
         "rows [" + std::to_string(first) + ", " + std::to_string(end) +
             ") outside [0, " + std::to_string(limit) + ")");
  Refuse(bytes != std::size_t{end - first} * info_.RowBytes(),
         "row buffer has " + std::to_string(bytes) + " bytes, expected " +
             std::to_string(std::size_t{end - first} * info_.RowBytes()));
}

void Executor::ExportRows(std::uint32_t layer, KvPart part, RowRange rows,
                          std::span<std::byte> out) const {
  RequireReady("ExportRows");
  CheckRows(layer, rows.first, rows.end, end_, out.size());
  Refuse(info_.sliding[layer] && !rows.empty() && rows.first < RetainedBegin(),
         "row " + std::to_string(rows.first) + " of sliding layer " +
             std::to_string(layer) + " left the ring (retained from " +
             std::to_string(RetainedBegin()) + ")");
  if (!rows.empty()) DoExport(layer, part, rows, out);
}

void Executor::BeginImport(std::uint32_t position) {
  Refuse(position == 0 || position > info_.max_context,
         "import position " + std::to_string(position) + " outside [1, " +
             std::to_string(info_.max_context) + "]");
  state_ = ExecState::kImporting;
  end_ = high_ = position;
  floor_ = 0;
  imported_.assign(std::size_t{info_.layers} * 2, {});
}

void Executor::ImportRows(std::uint32_t layer, KvPart part, RowRange rows,
                          std::span<const std::byte> bytes) {
  if (state_ != ExecState::kImporting) {
    throw std::logic_error(
        std::string("ricursive executor: ImportRows outside an import, "
                    "state is ") +
        StateName(state_));
  }
  CheckRows(layer, rows.first, rows.end, end_, bytes.size());
  const std::uint32_t lowest =
      end_ > info_.ring_rows ? end_ - info_.ring_rows : 0;
  Refuse(info_.sliding[layer] && !rows.empty() && rows.first < lowest,
         "row " + std::to_string(rows.first) + " does not fit the ring of " +
             std::to_string(info_.ring_rows) + " below " +
             std::to_string(end_));
  if (rows.empty()) return;
  try {
    DoImport(layer, part, rows, bytes);
  } catch (...) {
    state_ = ExecState::kFailed;
    throw;
  }
  imported_[std::size_t{layer} * 2 + (part == KvPart::kValue ? 1 : 0)]
      .push_back(rows);
}

bool Executor::CommitImport() {
  if (state_ != ExecState::kImporting) {
    throw std::logic_error(
        std::string("ricursive executor: CommitImport outside an import, "
                    "state is ") +
        StateName(state_));
  }
  const std::uint32_t needed = info_.WindowFirst(end_);
  std::uint32_t floor = 0;
  bool complete = true;
  for (std::uint32_t l = 0; l < info_.layers; ++l) {
    for (std::size_t part = 0; part < 2; ++part) {
      const std::uint32_t from = CoveredFrom(imported_[l * 2 + part], end_);
      if (info_.sliding[l]) {
        complete = complete && from <= needed;
        floor = std::max(floor, from);
      } else {
        complete = complete && from == 0;
      }
    }
  }
  imported_.clear();
  if (fault_ == LedgerFault::kNoImportCoverage) {
    complete = true;
    floor = std::min(floor, needed);
  }
  if (!complete) {
    state_ = ExecState::kFailed;
    return false;
  }
  floor_ = floor;
  state_ = ExecState::kReady;
  return true;
}

}  // namespace gufo::models::ricursive

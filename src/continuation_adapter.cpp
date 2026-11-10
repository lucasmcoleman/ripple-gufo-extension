#include "continuation_adapter.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace gufo::models::ricursive {
namespace {

using cache::Rows;

struct Metadata {
  std::uint32_t version{1};
  std::uint32_t position{0};
  std::uint32_t kv_type{0};  ///< 0 = F16, 1 = F32
  std::uint32_t reserved{0};
};
static_assert(std::is_trivially_copyable_v<Metadata>);

using Coverage = std::vector<std::pair<std::size_t, std::size_t>>;

/// [begin, end) is covered by `ranges` without a gap.
bool Covered(Coverage ranges, std::size_t begin, std::size_t end) {
  std::ranges::sort(ranges);
  std::size_t frontier = begin;
  for (const auto& [first, last] : ranges) {
    if (first > frontier) return false;
    frontier = std::max(frontier, last);
  }
  return frontier >= end;
}

class DoneSignal final : public cache::CompletionSignal {
 public:
  [[nodiscard]] bool Ready() const noexcept override { return true; }
  [[nodiscard]] cache::TransferResult Wait() noexcept override {
    return cache::TransferResult::kSucceeded;
  }
};
cache::Completion Done() { return cache::Completion(std::make_unique<DoneSignal>()); }

}  // namespace

class ContinuationAdapter::State final : public cache::Slot {
 public:
  State(ContinuationAdapter& adapter, cache::MutationGuard& guard, std::unique_ptr<Session> session)
      : adapter(adapter), guard(guard), session(std::move(session)) {
    this->session->SetWriteGuard([this](const WritePlan& plan) { BeforeWrite(plan); });
  }
  ~State() override { BeforeRelease(); }

  [[nodiscard]] bool IsValid() const noexcept override {
    return !restoring && !failed && session->IsValid();
  }

  void BeforeWrite(const WritePlan& plan) {
    if (restoring || failed) throw std::logic_error("ricursive slot is not ready for execution");
    if (adapter.faults_.no_guard) return;
    for (const auto& component : adapter.components_) {
      const auto& d = component.descriptor;
      if (!cache::HasRows(d.kind)) continue;
      if (!plan.written.empty()) guard.BeforeOverwrite(d.id, plan.written.first, plan.written.end);
      if (d.kind == cache::ComponentKind::kWindowRows && !plan.evicted.empty()) {
        guard.BeforeOverwrite(d.id, plan.evicted.first, plan.evicted.end);
      }
    }
  }
  void GuardAll() {
    if (adapter.faults_.no_guard) return;
    for (const auto& component : adapter.components_) {
      if (cache::HasRows(component.descriptor.kind)) {
        guard.BeforeOverwrite(component.descriptor.id, 0, adapter.context_);
      }
    }
  }
  void BeforeRelease() noexcept {
    for (const auto& component : adapter.components_) {
      if (cache::HasRows(component.descriptor.kind)) {
        guard.BeforeRelease(component.descriptor.id, 0, adapter.context_);
      }
    }
  }

  ContinuationAdapter& adapter;
  cache::MutationGuard& guard;
  std::unique_ptr<Session> session;
  bool restoring{false};
  bool failed{false};
  std::vector<cache::ComponentPosition> expected;  ///< in component order
  std::vector<Coverage> coverage;                  ///< rows or bytes per component
  Metadata metadata{};
  std::vector<std::int32_t> restore_tokens;
  std::vector<float> restore_logits;
};

ContinuationAdapter::ContinuationAdapter(std::shared_ptr<Model> model, cache::Identity identity)
    : model_(std::move(model)),
      context_(model_ ? model_->MaxContext() : 0),
      identity_(std::move(identity)) {
  if (!model_ || identity_.empty() || context_ == 0) {
    throw std::invalid_argument("invalid ricursive adapter configuration");
  }
  const Config& c = model_->config();
  const ExecutorConfig& ec = model_->executor_config();
  const Rows window = c.sliding_window - 1;
  const Rows chunk = model_->WindowChunkRows();
  // A checkpoint keeps a window plus its partial leading chunk, and a pass
  // writes all its rows before it reads.
  if (ec.ring_rows % chunk != 0 || ec.ring_rows < window + chunk - 1 ||
      ec.ring_rows < MinRingRows(c.sliding_window, ec.max_rows)) {
    throw std::invalid_argument("ricursive ring cannot hold a window");
  }
  const auto add = [&](ContinuationPart part, std::uint32_t layer, cache::ComponentDescriptor d) {
    d.id = {static_cast<std::uint32_t>(components_.size() + 1)};
    d.layout_version = 1;
    components_.push_back({d, part, layer});
  };
  add(ContinuationPart::kTokens, 0,
      {.kind = cache::ComponentKind::kAppendRows,
       .row_bytes = sizeof(std::int32_t),
       .rows_per_chunk = kAppendChunkRows});
  const std::size_t row = std::size_t{c.KvWidth()} * (ec.kv_type == KvType::kF16 ? 2 : 4);
  for (std::uint32_t l = 0; l < c.block_count; ++l) {
    for (const auto part : {ContinuationPart::kKey, ContinuationPart::kValue}) {
      add(part, l,
          c.IsSliding(l) ? cache::ComponentDescriptor{.kind = cache::ComponentKind::kWindowRows,
                                                      .row_bytes = row,
                                                      .rows_per_chunk = chunk,
                                                      .window_rows = window}
                         : cache::ComponentDescriptor{.kind = cache::ComponentKind::kAppendRows,
                                                      .row_bytes = row,
                                                      .rows_per_chunk = kAppendChunkRows});
    }
  }
  add(ContinuationPart::kLogits, 0,
      {.kind = cache::ComponentKind::kPrivateState,
       .state_bytes = std::size_t{c.vocab_size} * sizeof(float)});
  add(ContinuationPart::kMetadata, 0,
      {.kind = cache::ComponentKind::kPrivateState, .state_bytes = sizeof(Metadata)});
  for (const auto& component : components_) descriptors_.push_back(component.descriptor);
  // Everything that makes kept bytes invalid elsewhere: layout, positions,
  // pass width (it changes rounding on the GPU), ring, cache precision.
  const auto put = [&](std::uint32_t value) {
    for (unsigned i = 0; i < 4; ++i) identity_.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
  };
  put(1);  // component layout version
  put(context_);
  put(ec.max_rows);
  put(ec.ring_rows);
  put(static_cast<std::uint32_t>(chunk));
  put(ec.kv_type == KvType::kF16 ? 0U : 1U);
  put(ec.exact ? 1U : 0U);
}

std::vector<cache::BackingClass> ContinuationAdapter::BackingPlan(std::size_t ram) const {
  // Rows, tails and private states share the same committed pages; a payload
  // takes a contiguous page group and charges all of its slack.
  constexpr std::size_t page = 65536, per_block = 1024, header = 4096;
  const auto budget = ram > (4ULL << 20) ? (ram - (4ULL << 20)) / 10 * 9 : 0;
  if (budget <= header) return {};
  const auto blocks = (budget - header) / (page + per_block);
  if (!blocks) return {};
  return {{page, blocks, header + blocks * per_block}};
}

cache::Capabilities ContinuationAdapter::GetCapabilities() const {
  return {.continuation = true, .persistent_encoding = true};
}
std::span<const cache::ComponentDescriptor> ContinuationAdapter::Components() const {
  return descriptors_;
}
cache::Identity ContinuationAdapter::CompatibilityIdentity() const { return identity_; }

ContinuationAdapter::State& ContinuationAdapter::As(cache::Slot& slot) const {
  auto& state = dynamic_cast<State&>(slot);
  if (&state.adapter != this) throw std::invalid_argument("slot belongs to another ricursive adapter");
  return state;
}
const ContinuationAdapter::State& ContinuationAdapter::As(const cache::Slot& slot) const {
  return As(const_cast<cache::Slot&>(slot));
}
const ContinuationComponent& ContinuationAdapter::Component(cache::ComponentId id) const {
  if (!id.value || id.value > components_.size()) throw std::invalid_argument("invalid ricursive component ID");
  return components_[id.value - 1];
}

Session& ContinuationAdapter::GetSession(cache::Slot& slot) {
  auto& state = As(slot);
  if (!state.IsValid()) throw std::logic_error("ricursive slot is not ready for execution");
  return *state.session;
}

std::unique_ptr<cache::Slot> ContinuationAdapter::CreateSlot(cache::MutationGuard& guard) {
  std::string error;
  auto session = model_->CreateSession(&error);
  if (!session) throw std::runtime_error(error);
  return std::make_unique<State>(*this, guard, std::move(session));
}

std::vector<cache::ComponentPosition> ContinuationAdapter::Positions(const cache::Slot& slot) const {
  const auto& state = As(slot);
  if (!state.IsValid()) throw std::logic_error("cannot query an invalid ricursive slot");
  std::vector<cache::ComponentPosition> positions;
  positions.reserve(descriptors_.size());
  for (const auto& d : descriptors_) positions.push_back({d.id, state.session->Position()});
  return positions;
}

std::vector<Rows> ContinuationAdapter::PlanPrefill(const cache::Slot& slot, Rows first, Rows end) const {
  const auto& state = As(slot);
  if (!state.IsValid() || first > end || end > context_) {
    throw std::invalid_argument("invalid ricursive prefill plan");
  }
  // The passes Session::ForwardChunks makes: full rows each, a shorter last.
  const Rows rows = model_->PrefillRows();
  std::vector<Rows> plan;
  while (first < end) {
    first += std::min<Rows>(rows, end - first);
    plan.push_back(first);
  }
  return plan;
}

void ContinuationAdapter::BeginRestore(cache::Slot& slot,
                                       std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (state.restoring || state.failed) {
    throw std::logic_error("reset failed restores before BeginRestore");
  }
  if (positions.size() != descriptors_.size()) {
    throw std::invalid_argument("incomplete ricursive restore positions");
  }
  std::vector<cache::ComponentPosition> ordered;
  ordered.reserve(descriptors_.size());
  for (const auto& d : descriptors_) {
    if (std::ranges::count(positions, d.id, &cache::ComponentPosition::id) != 1) {
      throw std::invalid_argument("missing or duplicate ricursive frontier");
    }
    ordered.push_back(*std::ranges::find(positions, d.id, &cache::ComponentPosition::id));
  }
  const Rows n = ordered.front().valid_rows;
  if (n == 0 || n > context_ ||
      std::ranges::any_of(ordered, [&](const auto& p) { return p.valid_rows != n; })) {
    throw std::invalid_argument("inconsistent ricursive component frontiers");
  }
  // Guard every range the restore replaces while the old rows are readable.
  state.GuardAll();
  auto& session = *state.session;
  session.Reset();
  {
    const std::lock_guard lock(session.mutex());
    session.executor().BeginImport(static_cast<std::uint32_t>(n));
  }
  state.restore_tokens.assign(n, 0);
  state.restore_logits.assign(model_->VocabSize(), 0.0F);
  state.metadata = {};
  state.expected = std::move(ordered);
  state.coverage.assign(descriptors_.size(), {});
  state.restoring = true;
}

cache::Completion ContinuationAdapter::CopyRowsOut(const cache::Slot& slot, cache::ComponentId id,
                                                   Rows first, Rows end, std::span<std::byte> bytes,
                                                   cache::Stream&) {
  auto& state = As(slot);
  const auto& component = Component(id);
  const auto& d = component.descriptor;
  if (!state.IsValid()) throw std::logic_error("cannot read an invalid ricursive slot");
  auto& session = *state.session;
  const std::lock_guard lock(session.mutex());
  const Rows position = session.Position();
  if (!cache::HasRows(d.kind) || first > end || end > position ||
      bytes.size() != (end - first) * d.row_bytes) {
    throw std::invalid_argument("invalid ricursive row capture");
  }
  if (bytes.empty()) return Done();
  if (component.part == ContinuationPart::kTokens) {
    std::memcpy(bytes.data(), session.tokens().data() + first, bytes.size());
    return Done();
  }
  // Throws std::invalid_argument for window rows that left the ring.
  session.executor().ExportRows(
      component.layer, component.part == ContinuationPart::kKey ? KvPart::kKey : KvPart::kValue,
      {static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(end)}, bytes);
  return Done();
}

cache::Completion ContinuationAdapter::CopyRowsIn(cache::Slot& slot, cache::ComponentId id,
                                                  Rows first, Rows end,
                                                  std::span<const std::byte> bytes, cache::Stream&) {
  auto& state = As(slot);
  if (!state.restoring || state.failed) {
    throw std::logic_error("ricursive load requires an unfailed restore");
  }
  try {
    const auto& component = Component(id);
    const auto& d = component.descriptor;
    const Rows n = state.expected[id.value - 1].valid_rows;
    if (!cache::HasRows(d.kind) || first > end || end > n || first < cache::FirstChunkRow(d, n) ||
        bytes.size() != (end - first) * d.row_bytes) {
      throw std::invalid_argument("invalid ricursive row restore");
    }
    state.coverage[id.value - 1].emplace_back(first, end);
    if (bytes.empty()) return Done();
    if (component.part == ContinuationPart::kTokens) {
      std::memcpy(state.restore_tokens.data() + first, bytes.data(), bytes.size());
      return Done();
    }
    if (faults_.skip_layer_on_load && component.layer == 1) return Done();  // twin
    auto& session = *state.session;
    const std::lock_guard lock(session.mutex());
    session.executor().ImportRows(
        component.layer, component.part == ContinuationPart::kKey ? KvPart::kKey : KvPart::kValue,
        {static_cast<std::uint32_t>(first), static_cast<std::uint32_t>(end)}, bytes);
    return Done();
  } catch (...) {
    state.failed = true;
    throw;
  }
}

cache::Completion ContinuationAdapter::CapturePrivate(const cache::Slot& slot, cache::ComponentId id,
                                                      std::span<std::byte> bytes, cache::Stream&) {
  if (bytes.size() != Component(id).descriptor.state_bytes) {
    throw std::invalid_argument("incorrect ricursive private buffer size");
  }
  return CapturePrivateImpl(slot, id, 0, bytes);
}
cache::Completion ContinuationAdapter::CapturePrivatePiece(const cache::Slot& slot,
                                                           cache::ComponentId id, std::size_t offset,
                                                           std::span<std::byte> bytes, cache::Stream&) {
  return CapturePrivateImpl(slot, id, offset, bytes);
}
cache::Completion ContinuationAdapter::CapturePrivateImpl(const cache::Slot& slot,
                                                          cache::ComponentId id, std::size_t offset,
                                                          std::span<std::byte> bytes) {
  auto& state = As(slot);
  if (!state.IsValid() || state.session->Position() == 0) {
    throw std::logic_error("cannot capture an empty or invalid ricursive slot");
  }
  const auto& component = Component(id);
  const std::size_t total = component.descriptor.state_bytes;
  if (component.descriptor.kind != cache::ComponentKind::kPrivateState || offset > total ||
      bytes.size() > total - offset) {
    throw std::invalid_argument("invalid ricursive private piece");
  }
  auto& session = *state.session;
  const std::lock_guard lock(session.mutex());
  if (component.part == ContinuationPart::kMetadata) {
    Metadata metadata;
    metadata.position = session.Position();
    metadata.kv_type = model_->executor_config().kv_type == KvType::kF16 ? 0U : 1U;
    std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(&metadata) + offset, bytes.size());
    return Done();
  }
  if (!session.HasLogits()) throw std::logic_error("ricursive session has no frontier logits");
  std::memcpy(bytes.data(), reinterpret_cast<const std::byte*>(session.RawLogits().data()) + offset,
              bytes.size());
  return Done();
}

cache::Completion ContinuationAdapter::LoadPrivate(cache::Slot& slot, cache::ComponentId id,
                                                   std::span<const std::byte> bytes, cache::Stream&) {
  auto& state = As(slot);
  if (!state.restoring || state.failed) {
    throw std::logic_error("ricursive load requires an unfailed restore");
  }
  if (bytes.size() != Component(id).descriptor.state_bytes) {
    state.failed = true;
    throw std::invalid_argument("incorrect ricursive private buffer size");
  }
  return LoadPrivateImpl(slot, id, 0, bytes);
}
cache::Completion ContinuationAdapter::LoadPrivatePiece(cache::Slot& slot, cache::ComponentId id,
                                                        std::size_t offset,
                                                        std::span<const std::byte> bytes,
                                                        cache::Stream&) {
  return LoadPrivateImpl(slot, id, offset, bytes);
}
cache::Completion ContinuationAdapter::LoadPrivateImpl(cache::Slot& slot, cache::ComponentId id,
                                                       std::size_t offset,
                                                       std::span<const std::byte> bytes) {
  auto& state = As(slot);
  if (!state.restoring || state.failed) {
    throw std::logic_error("ricursive load requires an unfailed restore");
  }
  try {
    const auto& component = Component(id);
    const std::size_t total = component.descriptor.state_bytes;
    if (component.descriptor.kind != cache::ComponentKind::kPrivateState || offset > total ||
        bytes.size() > total - offset) {
      throw std::invalid_argument("invalid ricursive private piece");
    }
    state.coverage[id.value - 1].emplace_back(offset, offset + bytes.size());
    std::byte* target = component.part == ContinuationPart::kLogits
                            ? reinterpret_cast<std::byte*>(state.restore_logits.data())
                            : reinterpret_cast<std::byte*>(&state.metadata);
    if (!bytes.empty()) std::memcpy(target + offset, bytes.data(), bytes.size());
    return Done();
  } catch (...) {
    state.failed = true;
    throw;
  }
}

bool ContinuationAdapter::Validate(cache::Slot& slot,
                                   std::span<const cache::ComponentPosition> positions) {
  auto& state = As(slot);
  if (!state.restoring) throw std::logic_error("ricursive validation requires an active restore");
  const auto refuse = [&] {
    state.failed = true;
    return false;
  };
  if (state.failed || positions.size() != state.expected.size()) return refuse();
  for (std::size_t i = 0; i < state.expected.size(); ++i) {
    const auto& expected = state.expected[i];
    if (std::ranges::count(positions, expected) != 1) return refuse();
    const auto& d = descriptors_[i];
    const Rows n = expected.valid_rows;
    const bool rows = cache::HasRows(d.kind);
    if (!Covered(state.coverage[i], rows ? cache::FirstChunkRow(d, n) : 0, rows ? n : d.state_bytes)) {
      return refuse();
    }
  }
  const Rows n = state.expected.front().valid_rows;
  const auto kv = model_->executor_config().kv_type == KvType::kF16 ? 0U : 1U;
  if (state.metadata.version != 1 || state.metadata.position != n || state.metadata.kv_type != kv) {
    return refuse();
  }
  auto& session = *state.session;
  bool committed = false;
  {
    const std::lock_guard lock(session.mutex());
    committed = session.executor().CommitImport();
  }
  if (!committed) return refuse();
  session.AdoptRestored(std::move(state.restore_tokens), std::move(state.restore_logits));
  state.restoring = false;
  state.expected.clear();
  state.coverage.clear();
  state.restore_tokens = {};
  state.restore_logits = {};
  return true;
}

bool ContinuationAdapter::Invalidate(cache::Slot& slot) noexcept {
  State* resetting = nullptr;
  try {
    auto& state = As(slot);
    resetting = &state;
    state.BeforeRelease();
    state.session->Reset();
    state.failed = false;
    state.restoring = false;
    state.expected.clear();
    state.coverage.clear();
    state.restore_tokens = {};
    state.restore_logits = {};
    state.guard.AfterReset();
    return true;
  } catch (...) {
    if (resetting != nullptr) resetting->failed = true;
    return false;
  }
}

}  // namespace gufo::models::ricursive

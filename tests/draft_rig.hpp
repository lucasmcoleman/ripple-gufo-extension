// Checkpoint capture and restore through the real ContinuationAdapter (copied from
// tests/draft_session_test.cpp, whose sources stay untouched): shared by tests/speculative_test.cpp.
#ifndef RICURSIVE_TESTS_DRAFT_RIG_HPP_
#define RICURSIVE_TESTS_DRAFT_RIG_HPP_

#include <memory>
#include <vector>

#include "src/continuation_adapter.hpp"
#include "tests/draft_util.hpp"

namespace ricursive_test {

using Bytes = std::vector<std::byte>;

class HostStream final : public cache::Stream {
 public:
  [[nodiscard]] cache::TransferResult Synchronize() noexcept override { return cache::TransferResult::kSucceeded; }
};

class NullGuard final : public cache::MutationGuard {
 public:
  void BeforeOverwrite(cache::ComponentId, cache::Rows, cache::Rows) override {}
  void BeforeRelease(cache::ComponentId, cache::Rows, cache::Rows) noexcept override {}
  void AfterReset() noexcept override {}
};

/// One slot of a ContinuationAdapter over `model` (its session is adapter->GetSession(*slot)).
struct Rig {
  std::shared_ptr<EngineModel> model;
  NullGuard guard;
  std::unique_ptr<ContinuationAdapter> adapter;
  std::unique_ptr<cache::Slot> slot;
  HostStream stream;
  explicit Rig(std::shared_ptr<EngineModel> m) : model(std::move(m)) {
    adapter = std::make_unique<ContinuationAdapter>(model, cache::Identity{1, 2, 3});
    slot = adapter->CreateSlot(guard);
  }
  Session& session() { return adapter->GetSession(*slot); }
};

struct Checkpoint {
  std::vector<cache::ComponentPosition> positions;
  std::vector<Bytes> data;
  std::vector<cache::Rows> first;
  cache::Rows n{0};
};

Checkpoint Capture(Rig& r) {
  Checkpoint ck;
  ck.positions = r.adapter->Positions(*r.slot);
  ck.n = ck.positions.front().valid_rows;
  for (const auto& d : r.adapter->Components()) {
    Bytes bytes;
    cache::Rows first = 0;
    if (cache::HasRows(d.kind)) {
      first = cache::FirstChunkRow(d, ck.n);
      bytes.resize((ck.n - first) * d.row_bytes);
      Require(r.adapter->CopyRowsOut(*r.slot, d.id, first, ck.n, bytes, r.stream).Wait() == cache::TransferResult::kSucceeded,
              "CopyRowsOut");
    } else {
      bytes.resize(d.state_bytes);
      Require(r.adapter->CapturePrivate(*r.slot, d.id, bytes, r.stream).Wait() == cache::TransferResult::kSucceeded,
              "CapturePrivate");
    }
    ck.data.push_back(std::move(bytes));
    ck.first.push_back(first);
  }
  return ck;
}

bool Restore(Rig& r, const Checkpoint& ck) {
  const auto comps = r.adapter->Components();
  r.adapter->BeginRestore(*r.slot, ck.positions);
  for (std::size_t k = 0; k < comps.size(); ++k) {
    const auto& d = comps[k];
    if (cache::HasRows(d.kind)) {
      Require(r.adapter->CopyRowsIn(*r.slot, d.id, ck.first[k], ck.n, ck.data[k], r.stream).Wait() == cache::TransferResult::kSucceeded,
              "CopyRowsIn");
    } else {
      Require(r.adapter->LoadPrivate(*r.slot, d.id, ck.data[k], r.stream).Wait() == cache::TransferResult::kSucceeded,
              "LoadPrivate");
    }
  }
  return r.adapter->Validate(*r.slot, ck.positions);
}

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_DRAFT_RIG_HPP_

#ifndef RICURSIVE_CONTINUATION_ADAPTER_HPP_
#define RICURSIVE_CONTINUATION_ADAPTER_HPP_

// Serves ricursive sessions to the component cache (extension API 0.2.25).
//
// Components, ID = index + 1:
//   tokens           append rows (4 bytes)
//   per layer K, V   sliding layers: window rows (window_rows = W - 1, chunks
//                    of Model::WindowChunkRows()); full layers: append rows
//   frontier logits  private, vocab floats
//   metadata         private, version / position / cache element type
// No hidden-state component (nothing reads one). All transfers are
// synchronous: the executor's calls are. Rows are copied, never lent
// (BorrowRowsView stays empty), so nothing outlives the slot's own rows.
//
// Mutation: before every forward the session reports what it will overwrite
// (Executor::PlanForward) and this adapter passes it to MutationGuard:
// `written` for every row component, `evicted` for the window ones.

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "engine.hpp"
#include "gufo/ext/cache.hpp"

namespace gufo::models::ricursive {

enum class ContinuationPart : std::uint8_t { kTokens, kKey, kValue, kLogits, kMetadata };

struct ContinuationComponent {
  cache::ComponentDescriptor descriptor;
  ContinuationPart part;
  std::uint32_t layer{0};
};

/// Rows per cache chunk of the token and full-layer components.
inline constexpr cache::Rows kAppendChunkRows = 256;

class ContinuationAdapter final : public cache::Adapter {
 public:
  /// `identity` is the caller's part (artifact, template, ABI); the adapter
  /// appends its layout, context, ring, pass width and cache precision.
  ContinuationAdapter(std::shared_ptr<Model> model, cache::Identity identity);
  ~ContinuationAdapter() override = default;

  [[nodiscard]] Session& GetSession(cache::Slot& slot);
  [[nodiscard]] std::span<const ContinuationComponent> Layout() const { return components_; }

  [[nodiscard]] std::vector<cache::BackingClass> BackingPlan(std::size_t ram_bytes) const override;
  [[nodiscard]] cache::Capabilities GetCapabilities() const override;
  [[nodiscard]] std::span<const cache::ComponentDescriptor> Components() const override;
  [[nodiscard]] cache::Identity CompatibilityIdentity() const override;
  [[nodiscard]] std::unique_ptr<cache::Slot> CreateSlot(cache::MutationGuard& guard) override;
  [[nodiscard]] std::vector<cache::ComponentPosition> Positions(const cache::Slot&) const override;
  [[nodiscard]] std::vector<cache::Rows> PlanPrefill(const cache::Slot&, cache::Rows first,
                                                     cache::Rows end) const override;
  void BeginRestore(cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] cache::Completion CapturePrivate(const cache::Slot&, cache::ComponentId,
                                                 std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion CapturePrivatePiece(const cache::Slot&, cache::ComponentId,
                                                      std::size_t, std::span<std::byte>,
                                                      cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsOut(const cache::Slot&, cache::ComponentId,
                                              cache::Rows first, cache::Rows end,
                                              std::span<std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion CopyRowsIn(cache::Slot&, cache::ComponentId, cache::Rows first,
                                             cache::Rows end, std::span<const std::byte>,
                                             cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivate(cache::Slot&, cache::ComponentId,
                                              std::span<const std::byte>, cache::Stream&) override;
  [[nodiscard]] cache::Completion LoadPrivatePiece(cache::Slot&, cache::ComponentId, std::size_t,
                                                   std::span<const std::byte>,
                                                   cache::Stream&) override;
  [[nodiscard]] bool Validate(cache::Slot&, std::span<const cache::ComponentPosition>) override;
  [[nodiscard]] bool Invalidate(cache::Slot&) noexcept override;

  /// Test twins: deliberate defects (all off in normal use).
  struct Faults {
    bool skip_layer_on_load{false};  ///< CopyRowsIn drops layer 1's rows silently
    bool no_guard{false};            ///< BeforeOverwrite is never called
  };
  void SetFaults(Faults faults) { faults_ = faults; }

 private:
  class State;
  [[nodiscard]] State& As(cache::Slot& slot) const;
  [[nodiscard]] const State& As(const cache::Slot& slot) const;
  [[nodiscard]] const ContinuationComponent& Component(cache::ComponentId id) const;
  [[nodiscard]] cache::Completion CapturePrivateImpl(const cache::Slot&, cache::ComponentId,
                                                     std::size_t, std::span<std::byte>);
  [[nodiscard]] cache::Completion LoadPrivateImpl(cache::Slot&, cache::ComponentId, std::size_t,
                                                  std::span<const std::byte>);

  std::shared_ptr<Model> model_;
  std::uint32_t context_;
  std::vector<ContinuationComponent> components_;
  std::vector<cache::ComponentDescriptor> descriptors_;
  cache::Identity identity_;
  Faults faults_;
};

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_CONTINUATION_ADAPTER_HPP_

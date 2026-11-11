#ifndef RICURSIVE_RUNNER_HPP_
#define RICURSIVE_RUNNER_HPP_

#include <chrono>
#include <functional>
#include <gufo/ext/options.hpp>
#include <gufo/ext/text.hpp>
#include <memory>
#include <stdexcept>
#include <string>

#include "drafter_backend.hpp"
#include "drafter_reference.hpp"
#include "drafter_weights.hpp"
#include "engine.hpp"
#include "speculative_target.hpp"
#include "hip/drafter_executor.hpp"

namespace gufo::models::ricursive {

/// Applies the options `extension.json` declares to `options`; an option that
/// was not given keeps the default. Throws std::invalid_argument.
inline void ApplyProviderOptions(const ext::ParsedOptions& given, ModelOptions& options) {
  if (const auto* cap = ext::FindOption<std::int64_t>(given, "--context-cap")) {
    if (*cap <= 0 || *cap > static_cast<std::int64_t>(UINT32_MAX)) {
      throw std::invalid_argument("--context-cap must be between 1 and 4294967295");
    }
    options.max_context = static_cast<std::uint32_t>(*cap);
  }
  if (const auto* allow = ext::FindOption<bool>(given, "--allow-chat-template-mismatch")) {
    options.allow_template_mismatch = *allow;
  }
}

/// The DFlash2 drafter of a runner or a driver (--speculative dflash2 --dflash-model F), loaded
/// once and shared by the sessions.
struct SpeculativeSetup {
  std::shared_ptr<const core::GgufReader> reader;  ///< the drafter GGUF (the weights view it)
  DrafterWeights weights;
  std::uint32_t max_draft_tokens{4};
  DraftBackendFaults faults;  ///< tests only
  DrafterFaults engine_faults;  ///< tests only (the CPU reference's)
  /// The drafter engine's options (context storage follows the model's cache type: F32 in exact test mode).
  DrafterOptions drafter_options;
  /// Tests only: builds the session's drafter engine instead of MakeDrafterEngine's choice.
  std::function<std::unique_ptr<DrafterEngine>(const SpeculativeSetup&, const Model&)> engine_factory;
  /// What the drafter shares between its engines (the GPU drafter's device weights): made by the first session
  /// that needs it and held here, so it lives as long as the setup (the runner) and is made once per model load.
  /// Tests: `shared_factory` replaces the device upload.
  mutable std::shared_ptr<void> shared;
  std::function<std::shared_ptr<void>(const SpeculativeSetup&, const Model&)> shared_factory;
};

/// Opens and validates the drafter GGUF against the target. Null and *error on failure.
/// `max_draft_tokens` is clamped to 1 .. kMaxVerifyRows - 1.
[[nodiscard]] std::shared_ptr<const SpeculativeSetup> LoadSpeculativeSetup(
    const std::string& draft_model_path, const core::GgufReader& target_reader,
    std::uint32_t max_draft_tokens, std::string* error);

/// The options the drafter engine of a session of a model on `backend` is built with: the setup's, with the context storage
/// and the exactness the engine follows from the backend (an F32 cache holds 296 MiB of rings, not 148), and the projection
/// routes of the environment variable RICURSIVE_DRAFTER_Q8 when the setup names none (docs/DRAFTER-S2.md, "Stage 2b"). The
/// resource claim and the engine use this one function, so the claim is the memory the engine allocates.
[[nodiscard]] DrafterOptions EffectiveDrafterOptions(const SpeculativeSetup& setup, const BackendChoice& backend);

/// Applies what drafting needs to the model options (verify rows, the drafter's tap layers).
void ArmForSpeculation(const SpeculativeSetup& setup, ModelOptions& options);

/// Whether this build has a drafter implementation: the GPU drafter on a HIP build with the GPU
/// executor (production), the scalar CPU reference in a test build (tiny models only).
[[nodiscard]] bool HaveDrafterEngine() noexcept;
/// One drafter engine for a session of `model`: `setup.engine_factory` when set, else the GPU
/// drafter when the model runs on the device, else the CPU reference; nullptr when the build has none.
/// A GPU engine that cannot be built throws std::runtime_error naming the cause.
[[nodiscard]] std::unique_ptr<DrafterEngine> MakeDrafterEngine(const SpeculativeSetup& setup,
                                                              const Model& model);
/// Makes (once per setup) and holds what the drafter's engines share; see SpeculativeSetup::shared.
void PinSharedDrafter(const SpeculativeSetup& setup, const Model& model);
/// MakeDrafterEngine without the factory: the GPU drafter for a model that runs on the device, else the CPU reference.
[[nodiscard]] std::unique_ptr<DrafterEngine> MakeDefaultDrafterEngine(const SpeculativeSetup& setup,
                                                                     const Model& model);

/// The serving runner over an already loaded model (CreateRicursiveTextRunner and tests).
/// With `speculative` the model must have been armed (ArmForSpeculation).
std::shared_ptr<ext::TextModelRunner> CreateRunnerForModel(std::shared_ptr<Model> model,
                                                           const std::string& artifact_fingerprint,
                                                           std::shared_ptr<const SpeculativeSetup> speculative = {});

#if defined(ENGINE_ENABLE_HIP)
/// The drafting machinery of one session: a drafter engine fed by the session's kept rows (tap
/// sink) and gufo's verifier driving the target. Detaches the sink when destroyed; the session
/// must outlive it or be destroyed with it.
class DraftedSession {
 public:
  /// Throws when the build has no drafter or the model does not capture the drafter's layers.
  DraftedSession(Session& session, const SpeculativeSetup& setup);
  /// The session's own drafting: the one attached to it (Session::SetAttachment) when there is one for this
  /// setup, else a new one, attached. gufo binds a new runner state over a cached slot on every acquire; the
  /// drafter's context belongs to the slot's session, so a conversation that continues on its live slot keeps
  /// it. The context is the session's when it ends at the session's position; a restore or a reset told it
  /// through the tap sink.
  [[nodiscard]] static std::shared_ptr<DraftedSession> For(Session& session, const SpeculativeSetup& setup);
  ~DraftedSession();
  DraftedSession(const DraftedSession&) = delete;
  DraftedSession& operator=(const DraftedSession&) = delete;
  [[nodiscard]] SpeculativeDecoder& decoder() noexcept { return *decoder_; }
  [[nodiscard]] DrafterEngine& engine() noexcept { return *engine_; }
  [[nodiscard]] DFlash2DraftBackend& backend() noexcept { return *backend_; }
  [[nodiscard]] DrafterIngest& ingest() noexcept { return *ingest_; }
  /// What went wrong inside this session's drafter (failures, and whether drafting was switched off).
  [[nodiscard]] const DrafterHealth& health() const noexcept { return *health_; }

 private:
  Session* session_;
  const SpeculativeSetup* setup_;
  DrafterEngine* engine_;
  DFlash2DraftBackend* backend_;
  std::unique_ptr<DrafterHealth> health_;
  std::unique_ptr<DrafterIngest> ingest_;
  std::unique_ptr<SpeculativeDecoder> decoder_;
};

/// The drafting of a runner state (nullptr: speculation off). Tests.
[[nodiscard]] DraftedSession* DraftingOf(ext::TextRunnerState& state);
#else
class DraftedSession {  // stand-in: drafting is refused before one is built (HaveDrafterEngine() is false)
 public:
  DraftedSession(Session&, const SpeculativeSetup&) {
    throw std::logic_error("ricursive: this build has no speculative verifier");
  }
  [[nodiscard]] SpeculativeDecoder& decoder() { throw std::logic_error("no speculative verifier"); }
};
#endif

/// `TextProvider::create`: the serving runner over component-cache sessions.
std::shared_ptr<ext::TextModelRunner> CreateRicursiveTextRunner(
    const ext::TextRunnerOptions& options, std::string* error);

/// `gufo prompt`, `gufo chat` and `gufo bench` on one session, outside the
/// serving pool (drivers.cpp).
int RunRicursivePrompt(const ext::TextPromptOptions& options, const core::GgufReader& reader,
                       std::chrono::steady_clock::time_point load_start);
int RunRicursiveChat(const ext::TextPromptOptions& options, const core::GgufReader& reader,
                     std::chrono::steady_clock::time_point load_start);
int RunRicursiveBench(const ext::TextBenchOptions& options,
                      const std::shared_ptr<const core::GgufReader>& reader,
                      std::chrono::steady_clock::time_point load_start);

}  // namespace gufo::models::ricursive

#endif  // RICURSIVE_RUNNER_HPP_

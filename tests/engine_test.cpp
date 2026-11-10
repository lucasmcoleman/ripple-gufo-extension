// Engine (src/engine.*) on the CPU backend and the tiny fixtures: greedy
// generation against the Reference's full forwards, chunked prefill, token-list
// sync around the window and ring boundaries, the context cap. Every check has
// a broken twin built from CpuExecutor::Faults that must fail it.
#include <iostream>

#include "tests/fixtures_text/engine_util.hpp"

using namespace ricursive_test;

namespace {

constexpr std::uint32_t kContext = 96;

// ---- 1. greedy generation == Reference argmax chain --------------------------------

void GreedyChain(const TinyFixture& f) {
  const std::uint32_t W = f.window;
  auto model = LoadCpu(f.path, kContext, W + 4);
  const auto tokens = LoadTokens(f.name, 0);
  const auto prompt = Slice(tokens, 0, 16);
  const auto want = ReferenceChain(model->weights(), prompt, 48);
  Require(Distinct(want) >= 4, f.name + ": the reference chain is degenerate (" + std::to_string(Distinct(want)) +
                                   " distinct tokens)");
  auto session = NewSession(model);
  const auto got = SessionChain(*session, prompt, 48);
  Require(got == want, f.name + ": session chain differs from the reference chain");
  // Logits too: every step equals the reference's last row.
  std::cout << "  " << f.name << ": 48 greedy tokens equal the Reference chain (" << Distinct(want)
            << " distinct); first " << want[0] << ' ' << want[1] << ' ' << want[2] << '\n';

  // Twin: attention window one key short.
  CpuExecutor::Faults faults;
  faults.variant = Variant::kWinMinus1;
  auto bad_model = LoadCpu(f.path, kContext, W + 4, 4, &faults);
  auto bad = NewSession(bad_model);
  const auto bad_chain = SessionChain(*bad, prompt, 48);
  std::size_t first_diff = 0;
  while (first_diff < 48 && bad_chain[first_diff] == want[first_diff]) ++first_diff;
  std::size_t differing = 0;
  for (std::size_t i = 0; i < 48; ++i) differing += bad_chain[i] != want[i];
  std::cout << "    twin window-1: " << differing << " of 48 tokens differ (first at " << first_diff << ")\n";
  Require(differing > 0, f.name + ": the window-1 twin must change the greedy chain");
}

// ---- 2. prefill chunking changes nothing ---------------------------------------------

void Chunking(const TinyFixture& f) {
  const std::uint32_t W = f.window;
  const auto tokens = LoadTokens(f.name, 1);
  const auto prompt = Slice(tokens, 0, 60);
  std::string error;
  auto whole_model = LoadCpu(f.path, kContext, 64);  // one pass
  auto whole = NewSession(whole_model);
  Require(whole->Sync(prompt, &error), error);
  const std::vector<float> want(whole->Logits().begin(), whole->Logits().end());
  for (const std::uint32_t rows : {1U, 3U, W - 3, W + 4, 59U}) {
    auto model = LoadCpu(f.path, kContext, rows);
    auto session = NewSession(model);
    Require(session->Sync(prompt, &error), error);
    const double diff = MaxAbsDiff(session->Logits(), want);
    Require(diff == 0 && session->tokens() == prompt, f.name + ": prefill with " + std::to_string(rows) +
                                                          " rows per pass changed the result (" + Sci(diff) + ")");
  }
  // Twin: RoPE positions restart in every pass.
  CpuExecutor::Faults faults;
  faults.chunk_local_rope = true;
  auto bad_model = LoadCpu(f.path, kContext, 3, 4, &faults);
  auto bad = NewSession(bad_model);
  Require(bad->Sync(prompt, &error), error);
  const double twin = MaxAbsDiff(bad->Logits(), want);
  std::cout << "  " << f.name << ": passes of 1, 3, " << W - 3 << ", " << W + 4 << ", 59 rows equal one pass of 60 "
            << "(max abs 0); twin RoPE-per-pass " << Sci(twin) << '\n';
  Require(twin > 1e-3, f.name + ": the chunk-local RoPE twin must change the logits");
}

// ---- 3. token-list sync == fresh session ------------------------------------------------

void TokenSync(const TinyFixture& f) {
  const std::uint32_t W = f.window;
  const auto t = LoadTokens(f.name, 2);
  std::string error;
  auto model = LoadCpu(f.path, kContext, W + 4);
  const std::uint32_t ring = model->RingRows();
  const std::uint32_t high = 60;
  const std::uint32_t retained = high - ring;  // first retained row after 60 tokens
  const auto prompt = Slice(t, 0, high);
  // Divergence points: well inside the first window, at and around W, around
  // the edge of what the ring retains, and near the end.
  std::vector<std::uint32_t> points{1, W - 1, W, W + 1, 2 * W + 1, retained + W - 2, retained + W - 1,
                                    retained + W, 55, 59};
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  std::size_t rewound = 0, recomputed = 0;
  for (const std::uint32_t c : points) {
    if (c >= high) continue;
    // New text equals the old up to c, then differs; it is shorter, equal and
    // longer than the old one for different c.
    std::vector<std::int32_t> target = Slice(t, 0, c);
    const std::size_t extra = 6 + (c % 5) * 4;
    for (std::size_t i = 0; i < extra; ++i) {
      target.push_back((t[(c + i) % high] + 1 + static_cast<std::int32_t>(i)) % 256);  // [c] differs from t[c]
    }
    auto session = NewSession(model);
    bool full = false;
    const std::uint32_t cap = model->MaxContext();
    session->SetWriteGuard([&](const WritePlan& plan) {
      full = full || (plan.written.first == 0 && plan.written.end == cap);
    });
    Require(session->Sync(prompt, &error), error);
    full = false;
    Require(session->Sync(target, &error), error);
    auto fresh = NewSession(LoadCpu(f.path, kContext, W + 4));
    Require(fresh->Sync(target, &error), error);
    const double diff = MaxAbsDiff(session->Logits(), fresh->Logits());
    Require(diff == 0 && session->tokens() == target,
            f.name + ": sync at divergence " + std::to_string(c) + " differs from a fresh session (" + Sci(diff) + ")");
    // The recompute path is taken exactly when the window before c left the ring.
    const std::uint32_t needed = c >= W ? c - W + 1 : 0;
    const bool legal = needed >= retained;
    Require(full == !legal, f.name + ": at " + std::to_string(c) + " the engine " +
                                (full ? "recomputed" : "rewound") + " but the ring " +
                                (legal ? "still holds" : "no longer holds") + " the window");
    (full ? recomputed : rewound) += 1;
  }
  std::cout << "  " << f.name << ": sync at " << points.size() << " divergence points (ring " << ring
            << ", retained from " << retained << "): " << rewound << " rewound, " << recomputed
            << " recomputed, all equal to a fresh session (max abs 0)\n";
  Require(rewound > 0 && recomputed > 0, f.name + ": both paths must be exercised");

  // The sequence that leaves end < high water: 60 tokens, back to 50 (a rewind
  // of one, end 50, high 60), then a text that agrees up to 44. A rewind rule
  // computed from the logical end accepts it; the true ring no longer holds
  // its first rows.
  const auto s1 = Slice(t, 0, 60);
  const auto s2 = Slice(t, 0, 50);
  const std::uint32_t d = retained + W - 2;  // the window before d just left the ring
  std::vector<std::int32_t> s3 = Slice(t, 0, d);
  for (std::size_t i = 0; i < 12; ++i) s3.push_back((t[d + i] + 1) % 256);
  const auto run = [&](const CpuExecutor::Faults* faults) {
    auto m = LoadCpu(f.path, kContext, W + 4, 4, faults);
    auto session = NewSession(m);
    Require(session->Sync(s1, &error), error);
    Require(session->Sync(s2, &error), error);
    Require(session->Sync(s3, &error), error);
    return std::vector<float>(session->Logits().begin(), session->Logits().end());
  };
  auto fresh = NewSession(model);
  Require(fresh->Sync(s3, &error), error);
  const std::vector<float> want(fresh->Logits().begin(), fresh->Logits().end());
  const double good = MaxAbsDiff(run(nullptr), want);
  CpuExecutor::Faults stale;
  stale.ledger = Executor::LedgerFault::kRetentionFromEnd;
  const double twin = MaxAbsDiff(run(&stale), want);
  std::cout << "    two syncs ending below the high water: max abs " << good << "; twin (retention from the logical end) "
            << Sci(twin) << '\n';
  Require(good == 0, f.name + ": sync after a shorter sync differs from a fresh session");
  Require(twin > 1e-3, f.name + ": the stale-prefix twin must produce wrong logits");
}

// ---- 4. context cap --------------------------------------------------------------------------

void ContextCap(const TinyFixture& f) {
  const auto t = LoadTokens(f.name, 0);
  std::string error;
  auto model = LoadCpu(f.path, 32, 8);
  Require(model->MaxContext() == 32, "cap is the configured one");
  auto session = NewSession(model);
  Require(session->Sync(Slice(t, 0, 32), &error), "a prompt of exactly the cap fits: " + error);
  Require(!session->Evaluate(5, &error), "evaluating past the cap must fail");
  RequireContains(error, "context cap of 32", f.name + ": evaluate error");
  Require(session->tokens().size() == 32, "a refused token changes nothing");
  auto other = NewSession(model);
  Require(!other->Sync(Slice(t, 0, 33), &error), "a prompt over the cap must fail");
  RequireContains(error, "33 tokens exceeds the context cap of 32 tokens", f.name + ": sync error");
  Require(other->tokens().empty(), "a refused prompt changes nothing");
  // The model's own context (4096 in the files) is not the default cap, and a
  // cap above it is refused.
  ModelOptions big;
  big.max_context = 1U << 21;
  big.allow_template_mismatch = true;
  big.backend = CpuBackend();
  Require(EngineModel::Load(f.path, big, &error) == nullptr, "a cap above the model's positions must be refused");
  RequireContains(error, "exceeds the model's", f.name + ": load error");
  // Twin: an executor configured with a larger cap accepts the long prompt.
  ModelOptions loose;
  loose.max_context = 64;
  loose.prefill_rows = 8;
  loose.window_chunk_rows = 4;
  loose.allow_template_mismatch = true;
  loose.backend = CpuBackend();
  auto loose_model = EngineModel::Load(f.path, loose, &error);
  Require(loose_model != nullptr, error);
  auto twin = NewSession(loose_model);
  const bool refused = !twin->Sync(Slice(t, 0, 33), &error);
  std::cout << "  " << f.name << ": 32-token cap refuses 33 prompt tokens and a 33rd evaluated token with a named error; "
            << "twin (cap 64) refuses: " << (refused ? "yes" : "no") << '\n';
  Require(!refused, f.name + ": the check must key off the configured cap (twin with cap 64 accepts 33 tokens)");
}

// ---- 5. memory arithmetic (documented in docs/S2-ENGINE.md) --------------------------------------

void Arithmetic() {
  // The real model's shape through ExecutorInfo/ModelOptions maths: 16 full
  // layers of 8 KV heads x 128 at F16 = 64 KiB per token; ring 3072 rows for
  // 2048-row passes and a 1024 window, 64 sliding layers.
  const std::uint64_t full_layers = 16, sliding_layers = 64, kv_width = 8 * 128, element = 2;
  const std::uint64_t per_token = full_layers * 2 * kv_width * element;
  const std::uint32_t window = 1024, rows = 2048, chunk = 64;
  std::uint32_t ring = std::max(MinRingRows(window, rows), window - 1 + chunk - 1);
  ring = (ring + chunk - 1) / chunk * chunk;
  const std::uint64_t ring_bytes = sliding_layers * 2 * ring * kv_width * element;
  std::cout << "  arithmetic: " << per_token << " B per token (full layers, F16); ring " << ring << " rows = "
            << ring_bytes / (1 << 20) << " MiB; 32768 tokens = " << per_token * 32768 / (1 << 20) << " MiB; nominal "
            << per_token * 2097152 / (1ULL << 30) << " GiB\n";
  Require(per_token == 65536 && ring == 3072 && ring_bytes == 3072ULL * 64 * 4096, "arithmetic");
}

}  // namespace

int main() {
  return Run([] {
    for (const auto& f : TinyFixtures()) {
      GreedyChain(f);
      Chunking(f);
      TokenSync(f);
      ContextCap(f);
    }
    Arithmetic();
  });
}

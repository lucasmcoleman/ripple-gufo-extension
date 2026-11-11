// The DFlash2 drafter on the operations layer (docs/DRAFTER-S2.md), host operations only: no GPU
// back end, no HIP call. The same executor (hip/drafter_executor.cpp) runs on the GPU operations; the
// kernels behind them are not run here, only their host twins, written with the same loops.
//
//   drafter_ops_test tiny TARGET.gguf STAGE_DIR
//   drafter_ops_test real DRAFTER.gguf TARGET.gguf DUMP_DIR STAGES.bin [max_seconds]
//
// RICURSIVE_TEST_BACKEND unset or cpu: the host operations (ctest ricursive.drafter_ops). gpu-exact: the
// same checks with the drafter on the GPU operations and F32 context storage; gpu: F16 context storage, the
// production route (only through the GPU lock, tests/run-gpu-gates.sh; the GPU kernels run for the first time there).
//
// tiny (ctest ricursive.drafter_ops), fixtures: the stage 1 random drafters (Q8_0, Q4_K; top-k 8) and
// the stage 2 one (top-k 32, rank 256, Q8_0 / Q4_K codebooks) paired with tinyB:
//   1. every named stage of the drafter on the host ops against DrafterReference (the ground truth),
//      worst relative RMSE over a walk of 80 anchors that goes past the context window (64 rows) and the
//      sliding window (8 rows) and wraps the rings: 1e-5 with F32 context storage, 1e-3 with F16; the
//      drafted tokens equal at every anchor (a token that F16 flips must be a near-tie, counted and shown);
//   2. the top-k 32 selector; twin: the selector limited to 16 candidates;
//   3. the context bookkeeping: rows appended in passes of different sizes, truncation (inside the ring
//      slack, and deeper), reset to a base above 0, more rows than the rings hold: drafts equal the
//      reference's given the same rows;
//   4. twins that must fail (1): attention window off by one row, rotary on the no-rotary layer,
//      convolution reading across the block start;
//   plus the draft-length hint, DequantAt against gufo::quant::DequantizeRow and DraftVisibleKeys by brute force.
// real: the real drafter weights on the host ops against the oracle's stage file (not part of ctest).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <map>
#include <optional>
#include <sstream>

#include "hip/drafter_engine.hpp"
#include "hip/drafter_gpu.hpp"
#include "hip/drafter_math.hpp"
#include "src/drafter_reference.hpp"
#include "tests/oracle_common.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;
namespace km = gufo::models::ricursive::rocm;

namespace {

using Tokens = std::vector<std::int32_t>;

// ---- the dump (ids, taps) of tests/drafter_cpu_test.cpp ----------------------------------------------

struct Dump {
  Tokens ids;
  std::vector<float> taps;  ///< [N][K][H]
  std::size_t n{0}, k{0}, hidden{0};
  std::vector<std::uint32_t> layers;
};

std::string Slurp(const std::string& path) {
  std::ifstream in(path);
  Require(in.good(), "cannot open " + path);
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}

std::size_t JsonUint(const std::string& text, const std::string& key) {
  const auto at = text.find("\"" + key + "\"");
  Require(at != std::string::npos, "meta.json has no " + key);
  return std::stoull(text.substr(text.find(':', at) + 1));
}

Dump LoadDump(const std::string& dir) {
  const std::string meta = Slurp(dir + "/meta.json");
  Dump d;
  d.n = JsonUint(meta, "prompt_len") + JsonUint(meta, "generated");
  d.hidden = JsonUint(meta, "hidden");
  const auto at = meta.find("\"tap_layers\"");
  Require(at != std::string::npos, "meta.json has no tap_layers");
  std::stringstream list(meta.substr(meta.find('[', at) + 1, meta.find(']', at) - meta.find('[', at) - 1));
  for (std::string item; std::getline(list, item, ',');) d.layers.push_back(std::stoul(item));
  d.k = d.layers.size();
  d.ids = ReadBinary<std::int32_t>(dir + "/ids.i32");
  d.taps = ReadBinary<float>(dir + "/taps.f32");
  Require(d.ids.size() == d.n && d.taps.size() == d.n * d.k * d.hidden, "dump sizes disagree with meta.json");
  return d;
}

double RelRmse(std::span<const float> got, std::span<const float> ref) {
  if (got.size() != ref.size()) return std::numeric_limits<double>::infinity();
  double num = 0, den = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (!std::isfinite(got[i]) || !std::isfinite(ref[i])) return std::numeric_limits<double>::infinity();
    const double d = static_cast<double>(got[i]) - ref[i];
    num += d * d;
    den += static_cast<double>(ref[i]) * ref[i];
  }
  if (den == 0.0) return std::sqrt(num);
  return std::sqrt(num / den);
}

struct Drafter {
  std::shared_ptr<const gufo::core::GgufReader> reader;
  DrafterWeights w;
};

Drafter OpenDrafter(const std::string& path, const Config& target) {
  Drafter d;
  std::string error;
  d.reader = std::shared_ptr<const gufo::core::GgufReader>(gufo::core::GgufReader::OpenFile(path, &error));
  Require(d.reader != nullptr, "open " + path + ": " + error);
  const auto cfg = ParseDrafterConfig(*d.reader, target, &error);
  Require(cfg.has_value(), path + ": " + error);
  auto w = ResolveDrafterWeights(*d.reader, *cfg, target, &error);
  Require(w.has_value(), path + ": " + error);
  d.w = std::move(*w);
  return d;
}

// ---- the engine under test: host operations, or the GPU operations (RICURSIVE_TEST_BACKEND) -------------------

struct Backend {
  bool gpu{false};
  bool f16{false};  ///< production route: F16 context storage
  std::string name{"host"};
};

Backend& TheBackend() {
  static Backend b = [] {
    const char* env = std::getenv("RICURSIVE_TEST_BACKEND");
    const std::string name = env != nullptr ? env : "cpu";
    Backend r;
    if (name == "gpu-exact") r = {true, false, "gpu-exact"};
    else if (name == "gpu") r = {true, true, "gpu"};
    else Require(name == "cpu", "RICURSIVE_TEST_BACKEND must be cpu, gpu-exact or gpu, not " + name);
    return r;
  }();
  return b;
}

std::unique_ptr<OpsDrafterEngine> MakeEngine(const Drafter& d, const ricursive_test::Model& target, const DrafterOptions& options,
                                             const DrafterExecFaults& faults = {}, const HostOpsOptions& host = {}) {
  if (!TheBackend().gpu) return MakeHostOpsDrafterEngine(d.w, target.weights, options, faults, host);
  Require(!host.conv_across_block_start, "the convolution twin exists on the host operations only");
  // The GPU executor of the drafter is built by the factory (weights uploaded once per drafter); the faults are
  // executor options, so build the executor here from the same pieces when a fault is asked for.
  std::string error;
  std::shared_ptr<const gufo::core::GgufReader> target_reader(target.reader.get(), [](const gufo::core::GgufReader*) {});
  DrafterOptions gpu_options = options;
  gpu_options.exact = !TheBackend().f16;  // gpu-exact: FP32 activations at every row count
  auto engine = MakeGpuDrafterEngineWithHead(d.w, d.reader, target.weights, target_reader, gpu_options, &error, faults);
  Require(engine != nullptr, "GPU drafter: " + error);
  auto* ops = dynamic_cast<OpsDrafterEngine*>(engine.get());
  Require(ops != nullptr, "the GPU drafter engine is an OpsDrafterEngine");
  engine.release();
  return std::unique_ptr<OpsDrafterEngine>(ops);
}

// ---- the walk: the executor against DrafterReference ------------------------------------------------------

struct Walk {
  std::string label;
  DrafterOptions opts;
  DrafterExecFaults faults;
  HostOpsOptions host;
  double tol{1e-5};
  std::uint32_t d{15};
  bool allow_ties{false};  ///< a token the executor chooses differently must be a near-tie (F16 storage)
};

struct WalkResult {
  std::map<std::string, double> worst;
  std::size_t anchors{0}, anchors_equal{0}, rows{0}, rows_equal{0}, sets_equal{0}, set_rows{0};
  std::size_t flips{0}, ties{0}, hard{0};  ///< anchors with a different token; of them near-ties; not near-ties
  double worst_gap{0};                     ///< the largest score gap of a flip counted as a near-tie
  std::string worst_name;
  double worst_rel{0};
  std::vector<std::string> failing;
  bool pass{false};
};

using StageMap = std::map<std::string, std::vector<float>>;
std::string Key(std::string_view name, int layer) {
  return layer >= 0 ? std::string(name) + "." + std::to_string(layer) : std::string(name);
}

WalkResult RunWalk(const Drafter& drafter, const ricursive_test::Model& target, const std::vector<float>& feats,
                   const Tokens& ids, const Walk& cfg, std::size_t first_anchor = 8) {
  const DrafterWeights& dw = drafter.w;
  DrafterReference ref(drafter.w, target.weights);
  auto engine = MakeEngine(drafter, target, cfg.opts, cfg.faults, cfg.host);
  DrafterExecutor& exec = engine->executor();
  const std::size_t F = dw.config.FeatureWidth(), N = ids.size();
  WalkResult r;
  auto note = [&](const std::string& name, double rel) {
    auto [it, fresh] = r.worst.try_emplace(name, rel);
    if (!fresh && !(rel <= it->second)) it->second = rel;  // keeps NaN / inf
  };
  const auto append = [&](std::size_t at, std::size_t n) {
    StageMap want, got;
    ref.AppendContext(std::span(feats).subspan(at * F, n * F), n,
                      [&](std::string_view name, int layer, const float* d, std::size_t rows, std::size_t w) {
                        want[Key(name, layer)].assign(d, d + rows * w);
                      });
    exec.AppendContext(std::span(feats).subspan(at * F, n * F), n,
                       [&](std::string_view name, int layer, const float* d, std::size_t rows, std::size_t w) {
                         got[Key(name, layer)].assign(d, d + rows * w);
                       });
    for (const auto& [name, g] : got) note(name, RelRmse(g, want.at(name)));
  };
  append(0, first_anchor);
  for (std::size_t s = first_anchor; s < N; ++s) {
    StageMap want, got;
    const auto rd = ref.DraftStep(ids[s], s, [&](std::string_view name, int layer, const float* d, std::size_t rows,
                                                 std::size_t w) { want[Key(name, layer)].assign(d, d + rows * w); });
    const auto md = exec.DraftStep(ids[s], static_cast<std::uint32_t>(s), cfg.d,
                                   [&](std::string_view name, int layer, const float* d, std::size_t rows,
                                       std::size_t w) { got[Key(name, layer)].assign(d, d + rows * w); });
    for (const auto& [name, g] : got) {
      const std::vector<float>& w = want.at(name);
      note(name, RelRmse(g, std::span<const float>(w).first(std::min(w.size(), g.size()))));
    }
    // Tokens: equal up to the first row that differs; the later rows follow another path and mean nothing.
    ++r.anchors;
    const std::size_t d = md.tokens.size(), KT = md.top_k, KR = dw.config.selector_top_k;
    std::size_t p = 0;
    while (p < d && md.tokens[p] == rd.tokens[p]) ++p;
    r.rows += d;
    r.rows_equal += p;
    const std::size_t upto = std::min(d, p + 1);  // rows whose predecessor chain is the reference's
    std::vector<float> gs, os;
    for (std::size_t row = 0; row < upto; ++row) {
      std::vector<std::int32_t> a(md.candidates.begin() + static_cast<std::ptrdiff_t>(row * KT),
                                  md.candidates.begin() + static_cast<std::ptrdiff_t>((row + 1) * KT));
      std::vector<std::int32_t> b(rd.candidates.begin() + static_cast<std::ptrdiff_t>(row * KR),
                                  rd.candidates.begin() + static_cast<std::ptrdiff_t>((row + 1) * KR));
      std::vector<std::int32_t> sa = a;
      std::sort(sa.begin(), sa.end());
      ++r.set_rows;
      r.sets_equal += sa == b ? 1 : 0;  // the reference lists its candidates sorted by token id
      for (std::size_t i = 0; i < KT; ++i) {
        const auto at = std::lower_bound(b.begin(), b.end(), a[i]);
        if (at != b.end() && *at == a[i]) {
          gs.push_back(md.scores[row * KT + i]);
          os.push_back(rd.scores[row * KR + static_cast<std::size_t>(at - b.begin())]);
        }
      }
    }
    note("score", RelRmse(gs, os));
    if (p == d) {
      ++r.anchors_equal;
    } else {
      ++r.flips;
      // Near-tie: in the executor's own scores the two choices are within the F16 error of each other.
      const std::int32_t mine = md.tokens[p], theirs = rd.tokens[p];
      const auto find = [&](std::int32_t id) -> std::optional<float> {
        for (std::size_t i = 0; i < KT; ++i)
          if (md.candidates[p * KT + i] == id) return md.scores[p * KT + i];
        return std::nullopt;
      };
      const auto a = find(mine), b = find(theirs);
      const double gap = a && b ? std::fabs(static_cast<double>(*a) - *b) : 1e30;
      if (cfg.allow_ties && gap <= 2e-3 * (1.0 + std::fabs(static_cast<double>(a.value_or(0))))) {
        ++r.ties;
        r.worst_gap = std::max(r.worst_gap, gap);
      } else {
        ++r.hard;
      }
    }
    append(s, 1);
  }
  bool ok = r.hard == 0 && (cfg.allow_ties || r.flips == 0);
  for (const auto& [name, rel] : r.worst) {
    if (r.worst_name.empty() || !(rel <= r.worst_rel)) {
      r.worst_name = name;
      r.worst_rel = rel;
    }
    if (Exceeds(rel, cfg.tol)) {
      ok = false;
      if (r.failing.size() < 6) r.failing.push_back(name);
    }
  }
  r.pass = ok && r.anchors > 0;
  return r;
}

void Print(const std::string& tag, const WalkResult& r, bool twin) {
  std::printf("  %s: %zu anchors, all %s tokens equal at %zu (rows equal before the first difference %zu of %zu), %zu differ (%zu near-ties, %zu not), "
              "candidate sets equal %zu of %zu rows, worst stage %s %s, %zu stages%s\n",
              tag.c_str(), r.anchors, "drafted", r.anchors_equal, r.rows_equal, r.rows, r.flips, r.ties, r.hard, r.sets_equal, r.set_rows,
              r.worst_name.c_str(), Sci(r.worst_rel).c_str(), r.worst.size(),
              twin ? (r.pass ? "  -> PASSES (the test is too weak)" : "") : (r.pass ? "" : "  -> FAILS"));
  if (r.ties > 0) std::printf("      near-ties: the largest gap between the two choices' scores is %s\n", Sci(r.worst_gap).c_str());
  if (!r.pass) {
    std::string which;
    for (const auto& s : r.failing) which += " " + s;
    std::printf("      first failing stages:%s\n", which.c_str());
  }
}

// ---- helpers: building blocks tested on their own -------------------------------------------------------

/// DequantAt (the function the GPU selector kernel decodes codebook elements with) against
/// gufo::quant::DequantizeRow, on random encoded rows of every type it supports.
void CheckDequantAt() {
  Rng rng(77);
  struct Case {
    gufo::core::GgmlType type;
    km::Enc enc;
    std::uint32_t cols;
    const char* name;
  };
  const Case cases[] = {{gufo::core::GgmlType::kF32, km::Enc::kF32, 96, "F32"},
                        {gufo::core::GgmlType::kBF16, km::Enc::kBF16, 96, "BF16"},
                        {gufo::core::GgmlType::kQ8_0, km::Enc::kQ8_0, 128, "Q8_0"},
                        {gufo::core::GgmlType::kQ4_K, km::Enc::kQ4_K, 512, "Q4_K"}};
  std::size_t elements = 0;
  for (const Case& c : cases) {
    Require(EncodingOf(c.type) == c.enc, std::string(c.name) + ": EncodingOf");
    const std::size_t row_bytes = km::DequantRowBytes(c.enc, c.cols);
    const bool dense = c.type == gufo::core::GgmlType::kF32 || c.type == gufo::core::GgmlType::kBF16;
    Require(row_bytes != 0 && (dense || row_bytes == gufo::quant::QuantizedRowBytes(c.type, c.cols)),
            std::string(c.name) + ": row bytes");
    for (int trial = 0; trial < 40; ++trial) {
      std::vector<std::uint8_t> row(row_bytes);
      for (auto& byte : row) byte = static_cast<std::uint8_t>(rng.Below(256));
      // Finite, moderate scales: the halves at the start of every block / the top bytes of the floats.
      auto set_half = [&](std::size_t at, float v) {
        const std::uint16_t h = FloatToHalf(v);
        row[at] = static_cast<std::uint8_t>(h & 0xFF);
        row[at + 1] = static_cast<std::uint8_t>(h >> 8);
      };
      if (c.enc == km::Enc::kQ8_0) {
        for (std::size_t b = 0; b < row_bytes; b += 34) set_half(b, 0.01F * static_cast<float>(1 + rng.Below(50)));
      } else if (c.enc == km::Enc::kQ4_K) {
        for (std::size_t b = 0; b < row_bytes; b += 144) {
          set_half(b, 0.001F * static_cast<float>(1 + rng.Below(50)));
          set_half(b + 2, 0.001F * static_cast<float>(1 + rng.Below(50)));
        }
      } else if (c.enc == km::Enc::kF32) {
        for (std::size_t i = 0; i < c.cols; ++i) {
          const float v = static_cast<float>(rng.Below(2000)) / 100.0F - 10.0F;
          std::memcpy(&row[4 * i], &v, 4);
        }
      } else {  // BF16: sign, small exponent, any mantissa
        for (std::size_t i = 0; i < c.cols; ++i) row[2 * i + 1] = static_cast<std::uint8_t>(rng.Below(0x40) | (rng.Below(2) << 7));
      }
      std::vector<float> want(c.cols);
      Require(gufo::quant::DequantizeRow(c.type, row.data(), want.data(), c.cols), std::string(c.name) + ": DequantizeRow refused");
      for (std::uint32_t i = 0; i < c.cols; ++i) {
        const float got = km::DequantAt(c.enc, row.data(), i);
        Require(std::memcmp(&got, &want[i], 4) == 0, std::string(c.name) + ": element " + std::to_string(i) + " is " + std::to_string(got) +
                                                       ", gufo::quant::DequantizeRow gives " + std::to_string(want[i]));
      }
      elements += c.cols;
    }
  }
  std::printf("  DequantAt (the GPU selector's element access) equals gufo::quant::DequantizeRow bit for bit: F32, BF16, Q8_0, Q4_K, %zu elements\n",
              elements);
}

/// DraftVisibleKeys against the definition: keys at positions anchor + i (block rows) and [first, anchor)
/// (context), visible iff window == 0 or |p - j| < window.
void CheckVisibleKeys() {
  std::size_t checked = 0;
  for (std::uint32_t window : {0U, 1U, 2U, 8U, 16U, 17U, 100U}) {
    for (std::uint32_t tokens : {1U, 4U, 16U}) {
      for (std::uint32_t anchor : {0U, 3U, 9U, 40U, 200U}) {
        for (std::uint32_t first : {0U, 2U, 30U, 150U}) {
          if (first > anchor) continue;
          for (std::uint32_t t = 0; t < tokens; ++t) {
            const km::DraftKeys k = km::DraftVisibleKeys(t, anchor, first, tokens, window);
            const std::uint32_t p = anchor + t;
            for (std::uint32_t j = first; j < anchor; ++j) {
              const bool vis = window == 0 || (p > j ? p - j : j - p) < window;
              Require((j >= k.ctx_begin && j < k.ctx_end) == vis, "DraftVisibleKeys: context row " + std::to_string(j));
            }
            for (std::uint32_t i = 0; i < tokens; ++i) {
              const std::uint32_t j = anchor + i;
              const bool vis = window == 0 || (p > j ? p - j : j - p) < window;
              Require((i >= k.blk_begin && i < k.blk_end) == vis, "DraftVisibleKeys: block row " + std::to_string(i));
            }
            ++checked;
          }
        }
      }
    }
  }
  std::printf("  DraftVisibleKeys against the definition: %zu (row, anchor, window) combinations\n", checked);
}

// ---- fixtures and cases --------------------------------------------------------------------------------

struct Tiny {
  const ricursive_test::Model& target;
  std::string dir;
  Dump dump;
};

Walk F32Walk(const std::string& label) {
  Walk w;
  w.label = label;
  w.opts.kv_type = KvType::kF32;
  w.opts.ring_slack = 8;  // the rings of the tiny drafter wrap during the walk
  w.opts.max_append_rows = 16;
  w.tol = 1e-5;
  return w;
}
Walk F16Walk(const std::string& label) {
  Walk w = F32Walk(label);
  w.opts.kv_type = KvType::kF16;
  w.tol = 1e-3;
  w.allow_ties = true;
  return w;
}

WalkResult MustPass(const Tiny& t, const Drafter& d, const Walk& w, const std::string& what) {
  const WalkResult r = RunWalk(d, t.target, t.dump.taps, t.dump.ids, w);
  Print(what, r, false);
  Require(r.pass, what + ": the executor does not reproduce the reference (worst " + r.worst_name + " " + Sci(r.worst_rel) +
                      ", tolerance " + Sci(w.tol) + ")");
  return r;
}

void MustFail(const Tiny& t, const Drafter& d, const Walk& w, const std::string& what) {
  const WalkResult r = RunWalk(d, t.target, t.dump.taps, t.dump.ids, w);
  Print("twin '" + what + "'", r, true);
  Require(!r.pass, "twin '" + what + "' PASSES the comparison: the test is too weak");
}

/// The draft-length hint: the first d tokens of a hinted draft are the first d of the whole block.
void CheckHint(const Tiny& t, const Drafter& d) {
  auto engine = MakeEngine(d, t.target, F32Walk("").opts);
  const std::size_t F = d.w.config.FeatureWidth();
  std::size_t checked = 0;
  for (std::size_t s = 4; s < t.dump.n; s += 7) {
    engine->TruncateTo(0);
    engine->ResetTo(0);
    engine->Append(0, std::span(t.dump.taps).subspan(0, s * F));
    const auto whole = engine->Draft(t.dump.ids[s], static_cast<std::uint32_t>(s));
    Require(whole.tokens.size() == 15, "the whole block has 15 tokens");
    for (const std::uint32_t want : {1U, 2U, 3U, 4U, 6U, 15U, 40U}) {
      const auto part = engine->Draft(t.dump.ids[s], static_cast<std::uint32_t>(s), want);
      const std::size_t n = std::min<std::size_t>(want, 15);
      Require(part.tokens.size() == n && std::equal(part.tokens.begin(), part.tokens.end(), whole.tokens.begin()),
              "a hinted draft of " + std::to_string(want) + " tokens differs from the whole block's first ones at anchor " + std::to_string(s));
      Require(part.context_first == whole.context_first && part.context_end == whole.context_end, "the rows read depend on the hint");
      ++checked;
    }
  }
  std::printf("  draft-length hint: %zu hinted drafts (1, 2, 3, 4, 6, 15 and more than the block) equal the first tokens of the whole block\n", checked);
}

/// 3. Bookkeeping: the engine and the reference are fed the same operations; every draft must equal.
struct Bookkeeping {
  const Tiny& t;
  const Drafter& d;
  DrafterReference ref;
  std::unique_ptr<OpsDrafterEngine> eng;
  std::size_t F, drafts{0}, deep{0};
  Bookkeeping(const Tiny& tiny, const Drafter& dr, const DrafterOptions& o)
      : t(tiny), d(dr), ref(dr.w, tiny.target.weights), eng(MakeEngine(dr, tiny.target, o)),
        F(dr.w.config.FeatureWidth()) {}
  void Append(std::size_t n) {
    const std::size_t at = eng->End();
    Require(at + n <= t.dump.n, "bookkeeping script runs past the dump");
    Require(ref.ContextRows() == at, "reference and engine ends differ before an append");
    ref.AppendContext(std::span(t.dump.taps).subspan(at * F, n * F), n);
    eng->Append(static_cast<std::uint32_t>(at), std::span(t.dump.taps).subspan(at * F, n * F));
    Same("append");
  }
  void Truncate(std::uint32_t p, bool expect_deep) {
    const std::uint32_t base_before = eng->Base();
    eng->TruncateTo(p);
    const bool deep_now = eng->Base() != base_before;
    Require(deep_now == expect_deep, "truncate to " + std::to_string(p) + ": expected a " + (expect_deep ? "deep (reset)" : "plain") +
                                         " truncation, the base went " + std::to_string(base_before) + " -> " + std::to_string(eng->Base()));
    deep += deep_now ? 1 : 0;
    if (deep_now) {
      Require(eng->Base() == p, "a deep truncation resets the context at the position");
      ref.ResetTo(p);
    } else {
      ref.TruncateTo(p);
    }
    Same("truncate");
  }
  void Reset(std::uint32_t base) {
    eng->ResetTo(base);
    ref.ResetTo(base);
    Same("reset");
  }
  void Same(const char* what) const {
    Require(eng->Base() == ref.ContextBase() && eng->End() == ref.ContextRows(),
            std::string(what) + ": the engine holds [" + std::to_string(eng->Base()) + ", " + std::to_string(eng->End()) + ") and the reference [" +
                std::to_string(ref.ContextBase()) + ", " + std::to_string(ref.ContextRows()) + ")");
  }
  /// Drafts at the context end (and, when it holds rows, one row below it) and compares with the reference.
  void Draft() {
    for (const std::uint32_t back : {0U, 1U}) {
      if (eng->End() < back + 1 && back != 0) continue;
      const std::uint32_t s = eng->End() - back;
      if (s < eng->Base() || s >= t.dump.ids.size()) continue;
      const auto want = ref.DraftStep(t.dump.ids[s], s);
      const auto got = eng->executor().DraftStep(t.dump.ids[s], s, 15);
      Require(got.tokens == want.tokens, "the draft at position " + std::to_string(s) + " (context [" + std::to_string(eng->Base()) + ", " +
                                             std::to_string(eng->End()) + ")) differs from the reference's");
      Require(got.context_first == want.context_first && got.context_end == want.context_end,
              "the rows read differ: [" + std::to_string(got.context_first) + ", " + std::to_string(got.context_end) + ") against [" +
                  std::to_string(want.context_first) + ", " + std::to_string(want.context_end) + ")");
      ++drafts;
    }
  }
};

void CheckBookkeeping(const Tiny& t, const Drafter& d) {
  DrafterOptions o;
  o.kv_type = KvType::kF32;
  o.ring_slack = 8;       // rings: sliding 8 + 8 = 16 rows, full 64 + 8 = 72 rows
  o.max_append_rows = 5;  // chunking inside one Append
  Bookkeeping b(t, d, o);
  // Rows in passes of different sizes (1, 3, 16, 17, 5, 2, 24, 20 = 88 rows: more than the full ring's 72), a draft after each.
  for (const std::size_t n : {1, 3, 16, 17, 5, 2, 24, 20}) {
    b.Append(n);
    b.Draft();
  }
  Require(b.eng->End() == 88, "the script appends the whole dump");
  b.Truncate(86, false);  // inside the slack (the highest row ever held is 88)
  b.Draft();
  b.Truncate(81, false);  // the edge: 88 - 81 = 7 rows dropped, one fewer than the slack of 8
  b.Draft();
  b.Append(6);            // rewritten rows over the old ones
  b.Draft();
  b.Truncate(80, true);   // 8 rows below the highest: the rings no longer hold what position 80's window needs
  b.Draft();
  b.Append(8);
  b.Draft();
  b.Reset(40);            // a base above 0 (a restored checkpoint): nothing below it is ever read
  b.Draft();
  b.Append(48);
  b.Draft();
  b.Truncate(60, true);   // deep again, onto a base that is not 0 (the highest row held since Reset(40) is 88)
  b.Draft();
  b.Append(28);
  b.Draft();
  b.Reset(0);
  b.Append(30);           // a context shorter than the windows
  b.Draft();
  b.Truncate(0, false);   // to the base itself: an empty context, no row lost
  b.Draft();              // the block attends to itself only
  b.Append(20);
  b.Truncate(20, false);  // truncating at the end changes nothing
  b.Draft();
  std::printf("  bookkeeping: appends in passes of 1..24 rows (5-row chunks inside), plain and deep truncations, resets to bases 40 and 0, more rows "
              "than the 72-row ring: %zu drafts equal the reference's (%zu deep truncations)\n",
              b.drafts, b.deep);
}

/// Item 2 of docs/DRAFTER-S2.md "Stage 2b": the route of every group of projections, by call counting on the host operations
/// (the host has no model of the Q8_1 activation rounding, so the arithmetic of a routed projection is not testable here; on
/// the GPU the bench and `drafter_ops_test real` against the oracle judge it). For each of the 8 subsets of
/// {block, head, context}, with and without `exact` (which overrides every route): a context append of 20 rows in two passes
/// (16 + 4) and one draft of 15 rows; the number of ProjectQ8 calls, their rows and the weights they got are what the routes
/// say, and the total of projection calls does not depend on the routes.
void CheckRoutes(const Tiny& t, const Drafter& d) {
  const DrafterConfig& c = d.w.config;
  const std::size_t F = c.FeatureWidth();
  const std::uint64_t layers = c.block_count, ctx = 1 + 2 * layers, block = 9 * layers, head = 2, B = c.block_size, D = B - 1;
  std::size_t configurations = 0;
  for (int mask = 0; mask < 8; ++mask) {
    for (const bool exact : {false, true}) {
      DrafterOptions o = F32Walk("").opts;
      o.q8 = {(mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0};
      o.exact = exact;
      HostOpsCounters counters;
      HostOpsOptions host;
      host.counters = &counters;
      auto engine = MakeEngine(d, t.target, o, {}, host);
      const std::string label = "routes {" + ToString(o.q8) + "}" + (exact ? ", exact" : "");
      engine->Append(0, std::span(t.dump.taps).subspan(0, 20 * F));  // passes of 16 and 4 rows
      const bool q8_ctx = o.q8.context && !exact;
      Require(counters.project_q8 == (q8_ctx ? 2 * ctx : 0) && counters.project + counters.project_q8 == 2 * ctx,
              label + ": the context pass made " + std::to_string(counters.project_q8) + " ProjectQ8 and " + std::to_string(counters.project) +
                  " Project calls, expected " + std::to_string(q8_ctx ? 2 * ctx : 0) + " and " + std::to_string(q8_ctx ? 0 : 2 * ctx));
      Require(counters.q8_rows == (q8_ctx ? ctx * 20 : 0), label + ": rows of the context pass on the 8-bit route");
      const std::uint64_t before_q8 = counters.project_q8, before_all = counters.project + counters.project_q8;
      counters.q8_weights.clear();
      (void)engine->Draft(t.dump.ids[20], 20, static_cast<std::uint32_t>(D));
      const bool q8_block = o.q8.block && !exact, q8_head = o.q8.head && !exact;
      const std::uint64_t want_q8 = (q8_block ? block : 0) + (q8_head ? head : 0);
      Require(counters.project_q8 - before_q8 == want_q8 && counters.project + counters.project_q8 - before_all == block + head,
              label + ": the draft step made " + std::to_string(counters.project_q8 - before_q8) + " ProjectQ8 calls of " +
                  std::to_string(block + head) + ", expected " + std::to_string(want_q8));
      if (q8_head) {  // the head's two matrices, in order, are the last ProjectQ8 calls
        const auto& w = counters.q8_weights;
        Require(w.size() >= 2 && w[w.size() - 2] == t.target.weights.output.data && w[w.size() - 1] == d.w.selector_hidden.data,
                label + ": the head route did not carry the output head and the selector's hidden projection");
      }
      ++configurations;
    }
  }
  std::printf("  projection routes: %zu configurations (8 subsets of block/head/context, with and without exact) route exactly the groups chosen "
              "(by call counting: the host has no model of the 8-bit activation rounding)\n", configurations);
}

/// Item 3 of docs/DRAFTER-S2.md "Stage 2b": what crosses between host and device and where the host waits, counted on the host
/// operations (every EmbedRows, CopyIn, CopyOut and Synchronize of the host operations is a point where the GPU's stream is
/// waited for). Per executor: the MASK rows are embedded once, at construction. Per draft step: one row embedded, the anchor
/// token copied in, one device-to-device copy of the 15 MASK rows, one copy-out of the tokens and candidates; no other wait.
void CheckTransfers(const Tiny& t, const Drafter& d) {
  const DrafterConfig& c = d.w.config;
  const std::uint64_t B = c.block_size, R = B - 1, K = c.selector_top_k, E = c.embedding_length, F = c.FeatureWidth();
  HostOpsCounters n;
  HostOpsOptions host;
  host.counters = &n;
  DrafterOptions o = F32Walk("").opts;
  auto engine = MakeEngine(d, t.target, o, {}, host);
  Require(n.embed_calls == 1 && n.embed_rows == R && n.copy_in_calls == 0 && n.copy_out_calls == 0,
          "building an executor embeds the " + std::to_string(R) + " MASK rows once (" + std::to_string(n.embed_calls) + " embed calls of " +
              std::to_string(n.embed_rows) + " rows, " + std::to_string(n.copy_in_calls) + " copies in)");
  engine->Append(0, std::span(t.dump.taps).subspan(0, 20 * F));  // passes of 16 and 4 rows
  Require(n.copy_in_calls == 2 && n.copy_in_bytes == 20 * F * sizeof(float) && n.copy_out_calls == 0 && n.embed_calls == 1,
          "a context pass is one copy in of its feature rows and no wait");
  for (const std::uint32_t want : {15U, 3U}) {
    const HostOpsCounters before = n;
    (void)engine->Draft(t.dump.ids[20], 20, want);
    const auto delta = [&](std::uint64_t now, std::uint64_t then) { return now - then; };
    Require(delta(n.embed_calls, before.embed_calls) == 1 && delta(n.embed_rows, before.embed_rows) == 1,
            "a draft step embeds one row (the anchor's)");
    Require(delta(n.copy_in_calls, before.copy_in_calls) == 1 && delta(n.copy_in_bytes, before.copy_in_bytes) == sizeof(std::int32_t),
            "a draft step copies in the anchor token only");
    Require(delta(n.copy_internal_calls, before.copy_internal_calls) == 1 &&
                delta(n.copy_internal_bytes, before.copy_internal_bytes) == R * E * sizeof(float),
            "a draft step copies the MASK rows on the device");
    Require(delta(n.copy_out_calls, before.copy_out_calls) == 1 &&
                delta(n.copy_out_bytes, before.copy_out_bytes) == (B + 2 * R * K) * sizeof(std::int32_t),
            "a draft step reads back the tokens and candidates in one copy");
    Require(n.synchronize_calls == before.synchronize_calls && n.SyncPoints() - before.SyncPoints() == 3,
            "a draft step waits for the device three times: the embedded row, the token copy, the read-back");
  }
  std::printf("  transfers: executor construction embeds the %llu MASK rows once; a draft step (any d) embeds 1 row, copies in 4 bytes, copies the MASK "
              "rows on the device (%llu bytes), copies out %llu bytes in one call and waits 3 times (embedded row, token copy, read-back)\n",
              static_cast<unsigned long long>(R), static_cast<unsigned long long>(R * E * sizeof(float)),
              static_cast<unsigned long long>((B + 2 * R * K) * sizeof(std::int32_t)));
}

void TinyCase(const ricursive_test::Model& target, const std::string& dir) {
  Tiny t{target, dir, LoadDump(dir + "/stage1/tiny/dump")};
  const Drafter q8 = OpenDrafter(dir + "/stage1/tiny/drafter-q8_0.gguf", target.weights.config);
  const Drafter q4 = OpenDrafter(dir + "/stage1/tiny/drafter-q4_k.gguf", target.weights.config);
  const Drafter k32 = OpenDrafter(dir + "/stage2/tiny/drafter-topk32.gguf", target.weights.config);
  Require(q8.w.config.selector_top_k == 8 && k32.w.config.selector_top_k == 32 && k32.w.config.selector_rank == 256 &&
              k32.w.selector_predecessor.type == gufo::core::GgmlType::kQ8_0 && k32.w.selector_successor.type == gufo::core::GgmlType::kQ4_K,
          "fixture configuration (top-k 8 / 32, rank 256, Q8_0 predecessor and Q4_K successor codebooks)");
  Require(t.dump.n >= 88 && q8.w.config.context_window == 64 && q8.w.config.sliding_window == 8,
          "the walk needs at least 88 positions past a 64-row context window and an 8-row sliding window");

  const bool host = !TheBackend().gpu;
  const bool f32 = host || !TheBackend().f16, f16 = host || TheBackend().f16;
  const auto base = [&](const std::string& label) { return f32 ? F32Walk(label) : F16Walk(label); };
  std::printf("  backend %s\n", TheBackend().name.c_str());
  if (host) {  // host code only: the functions the GPU kernels share with the host operations
    CheckDequantAt();
    CheckVisibleKeys();
  }

  // 1. stage by stage, F32 and F16 context storage, three fixtures.
  WalkResult f16_q8, f16_k32;
  if (f32) {
    MustPass(t, q8, F32Walk(""), "1. Q8_0 drafter, F32 context storage, tolerance 1e-5");
    MustPass(t, q4, F32Walk(""), "1. Q4_K drafter, F32 context storage, tolerance 1e-5");
  }
  if (f16) f16_q8 = MustPass(t, q8, F16Walk(""), "1. Q8_0 drafter, F16 context storage, tolerance 1e-3");
  // 2. top-k 32.
  if (f32) MustPass(t, k32, F32Walk(""), "2. top-k 32, rank 256, Q8_0 / Q4_K codebooks, F32 context storage");
  if (f16) f16_k32 = MustPass(t, k32, F16Walk(""), "2. top-k 32, F16 context storage");
  {
    Walk w = base("");
    w.faults.selector_top_k_limit = 16;
    MustFail(t, k32, w, "the selector limited to 16 candidates (top-k 32 fixture)");
  }
  CheckHint(t, q8);
  if (host) {
    CheckRoutes(t, q8);
    CheckTransfers(t, q8);
    Walk w = F16Walk("");
    w.opts.q8 = {true, true, true};
    MustPass(t, q8, w, "1. Q8_0 drafter, F16 context storage, every projection group on the 8-bit route (host: the same arithmetic)");
  }
  // 3. bookkeeping.
  CheckBookkeeping(t, q8);
  // 4. twins of 1.
  {
    Walk w = base("");
    w.faults.window_off_by_one = true;
    MustFail(t, q8, w, "attention window off by one row");
    w = base("");
    w.faults.rope_on_nope_layer = true;
    MustFail(t, q8, w, "rotary on the no-rotary layer");
    if (host) {
      w = base("");
      w.host.conv_across_block_start = true;
      MustFail(t, q8, w, "convolution reading across the block start");
    }
  }
  if (f16) {
    std::printf("  F16 context storage flipped %zu of %zu drafted blocks on the Q8_0 fixture and %zu of %zu on the top-k 32 fixture "
                "(all near-ties, counted above)\n",
                f16_q8.flips, f16_q8.anchors, f16_k32.flips, f16_k32.anchors);
  }
}

// ---- real weights (the oracle's stage file, tests/drafter_cpu_test.cpp) ----------------------------------

struct Rec {
  std::uint32_t rows{0}, width{0};
  bool is_int{false};
  std::vector<float> f;
  std::vector<std::int32_t> i;
};

std::map<std::string, Rec> LoadStages(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  Require(in.good(), "cannot open " + path);
  std::map<std::string, Rec> out;
  auto u32 = [&]() {
    std::uint32_t v = 0;
    in.read(reinterpret_cast<char*>(&v), 4);
    return v;
  };
  while (in.peek() != std::char_traits<char>::eof()) {
    std::string name(u32(), '\0');
    in.read(name.data(), static_cast<std::streamsize>(name.size()));
    Rec r;
    r.rows = u32();
    r.width = u32();
    r.is_int = u32() == 1;
    const std::size_t n = std::size_t{r.rows} * r.width;
    if (r.is_int) {
      r.i.resize(n);
      in.read(reinterpret_cast<char*>(r.i.data()), static_cast<std::streamsize>(n * 4));
    } else {
      r.f.resize(n);
      in.read(reinterpret_cast<char*>(r.f.data()), static_cast<std::streamsize>(n * 4));
    }
    Require(in.good(), "truncated stage file " + path);
    out.emplace(std::move(name), std::move(r));
  }
  return out;
}

int Real(int argc, char** argv) {
  return Run([&] {
    Require(argc >= 6, "usage: real DRAFTER.gguf TARGET.gguf DUMP_DIR STAGES.bin [max_seconds]");
    const double max_seconds = argc > 6 ? std::stod(argv[6]) : 1e30;
    const ricursive_test::Model target = OpenModel(argv[3]);
    const Drafter d = OpenDrafter(argv[2], target.weights.config);
    const Dump dump = LoadDump(argv[4]);
    Require(dump.layers == d.w.config.tap_layers, "the dump's tap layers differ from the drafter's configuration");
    const auto oracle = LoadStages(argv[5]);
    const Rec& anchors_rec = oracle.at("anchors");
    std::vector<std::size_t> anchors(anchors_rec.i.begin(), anchors_rec.i.end());
    const std::size_t max_anchor = *std::max_element(anchors.begin(), anchors.end());
    DrafterOptions o;  // production options: F16 context storage
    if (const char* routes = std::getenv("RICURSIVE_TEST_Q8")) {  // e.g. block,head,context (the host ignores the rounding)
      o.q8 = ParseDrafterRoutes(routes);
      std::printf("  projection routes on the 8-bit path: %s\n", ToString(o.q8).c_str());
    }
    auto engine = MakeEngine(d, target, o);
    const std::size_t F = d.w.config.FeatureWidth();
    const auto t0 = std::chrono::steady_clock::now();
    engine->Append(0, std::span(dump.taps).subspan(0, max_anchor * F));
    std::printf("  context of %zu rows appended in %.1f s\n", max_anchor,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::size_t done = 0, rows = 0, equal = 0, anchors_equal = 0;
    double draft_seconds = 0;
    for (const std::size_t s : anchors) {
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > max_seconds) break;
      const auto a = std::chrono::steady_clock::now();
      const auto block = engine->Draft(dump.ids[s], static_cast<std::uint32_t>(s));
      draft_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
      const auto& want = oracle.at("a" + std::to_string(s) + "/tokens").i;
      std::size_t same = 0, first = 0;
      while (first < want.size() && first < block.tokens.size() && block.tokens[first] == want[first]) ++first;
      for (std::size_t i = 0; i < want.size() && i < block.tokens.size(); ++i) same += block.tokens[i] == want[i] ? 1 : 0;
      if (first == want.size()) {
        std::printf("    anchor %zu: all %zu tokens equal\n", s, want.size());
      } else {
        std::printf("    anchor %zu: first difference at row %zu of %zu, %zu of %zu tokens equal\n", s, first, want.size(), same, want.size());
      }
      ++done;
      rows += want.size();
      equal += same;
      anchors_equal += first == want.size() ? 1 : 0;
    }
    Require(done > 0, "no anchor ran");
    std::printf("  real weights, host ops, F16 context: %zu anchors, %zu all 15 tokens equal, %zu of %zu token rows equal (%.1f%%), %.1f s per anchor\n", done,
                anchors_equal, equal, rows, 100.0 * static_cast<double>(equal) / static_cast<double>(rows),
                draft_seconds / static_cast<double>(done));
    Require(done >= 10 || max_seconds < 1e29, "fewer than 10 anchors ran");
    Require(equal * 100 >= rows * 95, "fewer than 95% of the token rows equal the stage 1 reference's");
  });
}

int Tiny(const std::string& target_path, const std::string& dir) {
  return Run([&] {
    const ricursive_test::Model target = OpenModel(target_path);
    TinyCase(target, dir);
  });
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 4 && std::string(argv[1]) == "tiny") return Tiny(argv[2], argv[3]);
  if (argc >= 2 && std::string(argv[1]) == "real") return Real(argc, argv);
  std::fprintf(stderr, "usage: drafter_ops_test tiny TARGET.gguf STAGE_DIR | real DRAFTER TARGET DUMP STAGES [s]\n");
  return 2;
}

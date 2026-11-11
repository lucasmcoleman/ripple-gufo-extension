// DFlash2 drafter on the CPU (docs/DRAFTER-S1.md), CPU only, no GPU.
//
//   drafter_cpu_test tiny TARGET.gguf FIXTURE_DIR
//   drafter_cpu_test real DRAFTER.gguf TARGET.gguf DUMP_DIR STAGES.bin [max_seconds]
//
// tiny (ctest ricursive.drafter_cpu): random tiny drafters (Q8_0 and Q4_K) paired with tinyB.
//   * config: the metadata is read, validated against the target, and refused when it disagrees;
//   * the C++ drafter (src/drafter_reference.*) against the Python oracle (tools/drafter_oracle.py,
//     variant R, stage files made by tools/drafter_stage1_fixtures.sh): every named stage within
//     relative RMSE 1e-5, identical candidates and draft tokens at every anchor of a walk over
//     79 positions (Q8_0) and 16 positions (Q4_K); the features come from the CPU executor's capture of the
//     layers the drafter's own configuration names;
//   * four deliberate defects (DrafterFaults) run through the SAME comparison and must each fail it.
// real: the real drafter weights against the oracle's stage file for the anchors in it (rule: only
//   after a free -g check; not part of ctest).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>

#include "src/cpu_executor.hpp"
#include "src/drafter_config.hpp"
#include "src/drafter_reference.hpp"
#include "src/drafter_weights.hpp"
#include "tests/oracle_common.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;

namespace {

using Tokens = std::vector<std::int32_t>;

// ---- the oracle's stage file (tools/drafter_oracle.py stages) -----------------------------------

struct Rec {
  std::uint32_t rows{0}, width{0};
  bool is_int{false};
  std::vector<float> f;
  std::vector<std::int32_t> i;
};
using StageFile = std::map<std::string, Rec>;

StageFile LoadStages(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  Require(in.good(), "cannot open " + path + " (regenerate: tools/drafter_stage1_fixtures.sh)");
  StageFile out;
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

const Rec& Need(const StageFile& s, const std::string& name) {
  const auto it = s.find(name);
  Require(it != s.end(), "the oracle's stage file has no record " + name);
  return it->second;
}

// ---- the dump (ids, taps) ------------------------------------------------------------------------

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

/// Features [N][K*H] (position major) of the CPU executor's capture of `layers` over `ids`.
std::vector<float> CaptureFeatures(const Model& target, const std::vector<std::uint32_t>& layers,
                                   const Tokens& ids) {
  const std::size_t H = target.weights.config.embedding_length, K = layers.size();
  ExecutorConfig c;
  c.max_context = static_cast<std::uint32_t>(ids.size() + 16);
  c.max_rows = 16;
  c.max_logit_rows = 1;
  c.kv_type = KvType::kF32;
  c.exact = true;
  c.capture_layers = layers;
  CpuExecutor e(target.weights, c);
  std::vector<float> feats(ids.size() * K * H);
  for (std::size_t at = 0; at < ids.size(); at += 16) {
    const std::size_t n = std::min<std::size_t>(16, ids.size() - at);
    std::vector<float> cap(K * n * H);
    e.Forward(std::span(ids).subspan(at, n), static_cast<std::uint32_t>(at), LogitRows::kNone, {}, {},
              std::span<float>(cap));
    for (std::size_t k = 0; k < K; ++k) {
      for (std::size_t t = 0; t < n; ++t) {
        std::copy_n(&cap[(k * n + t) * H], H, &feats[((at + t) * K + k) * H]);
      }
    }
  }
  return feats;
}

// ---- comparison ------------------------------------------------------------------------------------

/// RMS of (got - ref) over RMS of ref; +inf for any non-finite value.
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

struct Thresholds {
  double stage_rel{1e-5};
  double logits_abs{-1};  ///< < 0: not checked
};

struct CaseResult {
  std::map<std::string, double> worst;  ///< stage name -> worst relative RMSE over the anchors
  std::size_t anchors{0}, tokens_equal{0}, cands_equal{0}, cands_exact{0};
  double logits_max_abs{0};
  double seconds_per_anchor{0};
  bool pass{false};
  std::string worst_name;
  double worst_rel{0};
  std::vector<std::string> failing;  ///< stages above the threshold (first few)
};

std::string StageKey(std::string_view name, int layer) {
  return layer >= 0 ? std::string(name) + "." + std::to_string(layer) : std::string(name);
}

/// The one comparison: C++ drafter (with `faults`) against the oracle's stage file, for the anchors
/// the file holds. `feats` are the [N][K*H] target features; only rows below the largest anchor are used.
CaseResult RunCase(const DrafterWeights& dw, const ModelWeights& target, const DrafterFaults& faults,
                   const std::vector<float>& feats, const Tokens& ids, const StageFile& oracle,
                   const Thresholds& th, double max_seconds = 1e30) {
  DrafterReference ref(dw, target, faults);
  const std::size_t F = dw.config.FeatureWidth();
  const Rec& anchors_rec = Need(oracle, "anchors");
  std::vector<std::size_t> anchors(anchors_rec.i.begin(), anchors_rec.i.end());
  const std::size_t max_anchor = *std::max_element(anchors.begin(), anchors.end());
  CaseResult r;
  auto note = [&](const std::string& name, double rel) {
    auto [it, fresh] = r.worst.try_emplace(name, rel);
    if (!fresh && !(rel <= it->second)) it->second = rel;  // keeps NaN/inf
  };

  // Context in chunks (the stage callback gets them in position order).
  std::map<std::string, std::vector<float>> ctx;
  for (std::size_t at = 0; at < max_anchor; at += 128) {
    const std::size_t n = std::min<std::size_t>(128, max_anchor - at);
    ref.AppendContext(std::span(feats).subspan(at * F, n * F), n,
                      [&](std::string_view name, int layer, const float* d, std::size_t rows, std::size_t w) {
                        auto& v = ctx[StageKey(name, layer)];
                        v.insert(v.end(), d, d + rows * w);
                      });
  }
  for (const auto& [name, got] : ctx) {
    const Rec& want = Need(oracle, name);
    note(name, RelRmse(got, want.f));
  }

  const auto t0 = std::chrono::steady_clock::now();
  for (const std::size_t s : anchors) {
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > max_seconds) break;
    std::map<std::string, std::vector<float>> got;
    const auto draft = ref.DraftStep(
        ids[s], s, [&](std::string_view name, int layer, const float* d, std::size_t rows, std::size_t w) {
          got[StageKey(name, layer)].assign(d, d + rows * w);
        });
    const std::string p = "a" + std::to_string(s) + "/";
    for (const auto& [name, g] : got) {
      const Rec& want = Need(oracle, p + name);
      note(name, RelRmse(g, want.f));
      if (name == "logits" && th.logits_abs >= 0) {
        r.logits_max_abs = std::max(r.logits_max_abs, MaxAbsDiff(g, want.f));
        if (!std::isfinite(r.logits_max_abs)) r.logits_max_abs = std::numeric_limits<double>::infinity();
      }
    }
    // Candidates and scores. The 32nd and 33rd logits of a 100k vocabulary can be closer than the numeric
    // noise of the logits (real weights: 1e-4), so a candidate that only one side lists is accepted when its
    // logit lies within `th.logits_abs` of the oracle's 32nd largest logit; the scores of the candidates both
    // sides list are compared. On the tiny fixtures logits_abs is not set and the sets must be equal.
    const Rec& wc = Need(oracle, p + "cand");
    const Rec& ws = Need(oracle, p + "score");
    const Rec& wl = Need(oracle, p + "logits");
    const std::size_t KT = wc.width;
    const Rec& wconf = Need(oracle, p + "conf");
    std::vector<float> gs, os, gconf, oconf;
    bool sets_ok = true, sets_exact = draft.candidates == wc.i;
    for (std::size_t row = 0; row < wc.rows; ++row) {
      std::vector<float> lg(wl.f.begin() + static_cast<std::ptrdiff_t>(row * wl.width),
                            wl.f.begin() + static_cast<std::ptrdiff_t>((row + 1) * wl.width));
      std::vector<float> top = lg;
      std::nth_element(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(KT - 1), top.end(), std::greater<>());
      const float kth = top[KT - 1];
      const std::int32_t* g = &draft.candidates[row * KT];
      const std::int32_t* o = &wc.i[row * KT];
      if (std::equal(g, g + KT, o)) {  // the confidence is a softmax over all candidates: compare it on equal sets
        gconf.push_back(draft.confidence[row]);
        oconf.push_back(wconf.f[row]);
      }
      for (std::size_t a = 0; a < KT; ++a) {
        const auto in_g = std::find(g, g + KT, o[a]);
        if (in_g != g + KT) {
          gs.push_back(draft.scores[row * KT + static_cast<std::size_t>(in_g - g)]);
          os.push_back(ws.f[row * KT + a]);
        } else if (th.logits_abs < 0 || std::fabs(lg[static_cast<std::size_t>(o[a])] - kth) > th.logits_abs) {
          sets_ok = false;
        }
      }
      for (std::size_t a = 0; a < KT; ++a) {
        if (std::find(o, o + KT, g[a]) == o + KT &&
            (th.logits_abs < 0 || std::fabs(lg[static_cast<std::size_t>(g[a])] - kth) > th.logits_abs)) {
          sets_ok = false;
        }
      }
    }
    if (!sets_exact && sets_ok) {
      std::printf("    anchor %zu: candidate sets differ only at the 32nd logit (within %s); that row's confidence is not compared\n", s,
                  Sci(th.logits_abs).c_str());
    }
    note("score", RelRmse(gs, os));
    note("conf", RelRmse(gconf, oconf));
    ++r.anchors;
    r.cands_equal += sets_ok ? 1 : 0;
    r.cands_exact += sets_exact ? 1 : 0;
    r.tokens_equal += draft.tokens == Need(oracle, p + "tokens").i ? 1 : 0;
  }
  r.seconds_per_anchor =
      r.anchors ? std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / r.anchors : 0;

  bool stages_ok = true;
  for (const auto& [name, rel] : r.worst) {
    if (r.worst_name.empty() || !(rel <= r.worst_rel)) {  // also takes a NaN
      r.worst_name = name;
      r.worst_rel = rel;
    }
    if (Exceeds(rel, th.stage_rel)) {
      stages_ok = false;
      if (r.failing.size() < 6) r.failing.push_back(name);
    }
  }
  const bool logits_ok = th.logits_abs < 0 || !Exceeds(r.logits_max_abs, th.logits_abs);
  r.pass = stages_ok && logits_ok && r.anchors > 0 && r.tokens_equal == r.anchors &&
           r.cands_equal == r.anchors;
  return r;
}

void Print(const std::string& tag, const CaseResult& r) {
  std::printf("  %s: %zu anchors, tokens identical %zu, candidate sets identical %zu (exactly %zu), worst stage %s %s, "
              "%zu stages compared%s\n",
              tag.c_str(), r.anchors, r.tokens_equal, r.cands_equal, r.cands_exact, r.worst_name.c_str(),
              Sci(r.worst_rel).c_str(), r.worst.size(), r.pass ? "" : "  -> FAILS the comparison");
  if (r.pass) {  // the next largest stage errors, for the record
    std::vector<std::pair<double, std::string>> by;
    for (const auto& [name, rel] : r.worst) by.emplace_back(rel, name);
    std::sort(by.rbegin(), by.rend());
    std::printf("      largest stage errors:");
    for (std::size_t i = 0; i < std::min<std::size_t>(4, by.size()); ++i) {
      std::printf(" %s %s", by[i].second.c_str(), Sci(by[i].first).c_str());
    }
    std::printf("\n");
  }
}

// ---- config refusals -------------------------------------------------------------------------------

/// The drafter's configuration must be refused when it disagrees with the target: a target with another
/// hidden size, fewer layers than the tap layers need, or a smaller vocabulary than the mask token.
void CheckRefusals(const gufo::core::GgufReader& dreader, const Config& target) {
  std::string error;
  Config t = target;
  t.embedding_length += 16;
  Require(!ParseDrafterConfig(dreader, t, &error), "a target with another hidden size was accepted");
  RequireContains(error, "embedding_length", "hidden size refusal");
  t = target;
  t.block_count = 3;  // taps stored 1..4
  Require(!ParseDrafterConfig(dreader, t, &error), "a target with too few layers was accepted");
  RequireContains(error, "target_layers", "tap layer refusal");
  t = target;
  t.vocab_size = 100;
  Require(!ParseDrafterConfig(dreader, t, &error), "a target with a small vocabulary was accepted");
  RequireContains(error, "mask_token_id", "mask token refusal");
  std::printf("  config refusals: hidden size, tap layers beyond the target, mask token beyond the vocabulary\n");
}

void CheckConfig(const DrafterConfig& c, const std::string& tag) {
  Require(c.block_count == 5 && c.block_size == 16 && c.conv_kernel_size == 4 && c.conv_group_size == 16 &&
              c.selector_top_k == 8 && c.selector_rank == 16 && c.sliding_window == 8 &&
              c.context_window == 64 && c.mask_token_id == 511,
          tag + ": configuration values differ from the fixture generator's");
  Require(c.rope == std::vector<bool>({true, true, true, true, false}) &&
              c.sliding == std::vector<bool>({true, true, true, true, false}),
          tag + ": per-layer rotary or window pattern");
  Require(c.target_layers_stored == std::vector<std::uint32_t>({1, 2, 3, 4}) &&
              c.tap_layers == std::vector<std::uint32_t>({0, 1, 2, 3}),
          tag + ": tap layers must be the stored values minus one");
}

struct Drafter {
  std::unique_ptr<gufo::core::GgufReader> reader;
  DrafterWeights w;
};

Drafter OpenDrafter(const std::string& path, const Config& target) {
  Drafter d;
  std::string error;
  d.reader = gufo::core::GgufReader::OpenFile(path, &error);
  Require(d.reader != nullptr, "open " + path + ": " + error);
  const auto cfg = ParseDrafterConfig(*d.reader, target, &error);
  Require(cfg.has_value(), path + ": " + error);
  auto w = ResolveDrafterWeights(*d.reader, *cfg, target, &error);
  Require(w.has_value(), path + ": " + error);
  d.w = std::move(*w);
  return d;
}

void TinyCase(const Model& target, const std::string& dir, const std::string& quant, bool twins) {
  const std::string tag = "drafter-" + quant;
  const Drafter d = OpenDrafter(dir + "/drafter-" + quant + ".gguf", target.weights.config);
  CheckConfig(d.w.config, tag);
  const Dump dump = LoadDump(dir + "/dump");
  const StageFile oracle = LoadStages(dir + "/stages-" + quant + ".bin");
  Thresholds th;

  // Features from the CPU executor's capture of the layers the drafter's configuration names.
  const auto taps_of = [&](const DrafterFaults& f) {
    const DrafterReference probe(d.w, target.weights, f);
    return CaptureFeatures(target, probe.TapLayers(), dump.ids);
  };
  const std::vector<float> feats = taps_of({});
  {  // the dump the oracle read holds the same numbers (same code), a check on the harness
    Require(dump.layers == d.w.config.tap_layers, tag + ": the dump's layers differ from the configuration's");
    Require(feats.size() == dump.taps.size(), "feature sizes");
    std::printf("  %s: captured features vs the dump the oracle read: max abs %s\n", tag.c_str(),
                Sci(MaxAbsDiff(feats, dump.taps)).c_str());
  }
  const CaseResult good = RunCase(d.w, target.weights, {}, feats, dump.ids, oracle, th);
  Print(tag, good);
  Require(good.pass, tag + ": the C++ drafter does not reproduce the oracle (worst " + good.worst_name + " " +
                         Sci(good.worst_rel) + ", tolerance " + Sci(th.stage_rel) + ")");
  if (!twins) return;

  struct Twin {
    const char* name;
    DrafterFaults f;
  };
  std::vector<Twin> list(4);
  list[0] = {"rotary positions on the no-rotary layer", {}};
  list[0].f.rope_on_nope_layer = true;
  list[1] = {"convolution taps shifted by one", {}};
  list[1].f.conv_taps_shifted = true;
  list[2] = {"selector predecessor from the candidate's own row", {}};
  list[2].f.selector_own_row = true;
  list[3] = {"tap layer off by one", {}};
  list[3].f.tap_layer_plus_one = true;
  for (const auto& tw : list) {
    const CaseResult r = RunCase(d.w, target.weights, tw.f, taps_of(tw.f), dump.ids, oracle, th);
    Print(std::string("twin '") + tw.name + "'", r);
    Require(!r.pass, std::string("twin '") + tw.name + "' PASSES the comparison: the test is too weak");
    std::string which;
    for (const auto& s : r.failing) which += " " + s;
    std::printf("      first failing stages:%s\n", which.c_str());
  }
}

int Tiny(const std::string& target_path, const std::string& dir) {
  return Run([&] {
    const Model target = OpenModel(target_path);
    {
      std::string error;
      auto reader = gufo::core::GgufReader::OpenFile(dir + "/drafter-q8_0.gguf", &error);
      Require(reader != nullptr, error);
      CheckRefusals(*reader, target.weights.config);
    }
    TinyCase(target, dir, "q8_0", true);
    TinyCase(target, dir, "q4_k", false);
  });
}

int Real(int argc, char** argv) {
  return Run([&] {
    Require(argc >= 6, "usage: real DRAFTER.gguf TARGET.gguf DUMP_DIR STAGES.bin [max_seconds]");
    const double max_seconds = argc > 6 ? std::stod(argv[6]) : 1e30;
    const Model target = OpenModel(argv[3]);
    const Drafter d = OpenDrafter(argv[2], target.weights.config);
    const Dump dump = LoadDump(argv[4]);
    Require(dump.layers == d.w.config.tap_layers, "the dump's tap layers differ from the drafter's configuration");
    std::printf("  real drafter: %u layers, taps", d.w.config.block_count);
    for (const auto t : d.w.config.tap_layers) std::printf(" %u", t);
    std::printf(", top-k %u, context window %u\n", d.w.config.selector_top_k, d.w.config.context_window);
    const StageFile oracle = LoadStages(argv[5]);
    Thresholds th;
    th.stage_rel = 1e-4;
    th.logits_abs = 1e-3;
    const CaseResult r = RunCase(d.w, target.weights, {}, dump.taps, dump.ids, oracle, th, max_seconds);
    Print("real", r);
    for (const auto& [name, rel] : r.worst) std::printf("    %-14s %s\n", name.c_str(), Sci(rel).c_str());
    std::printf("  logits max abs difference %s (tolerance %s); scalar time per anchor %.1f s\n",
                Sci(r.logits_max_abs).c_str(), Sci(th.logits_abs).c_str(), r.seconds_per_anchor);
    Require(r.pass, "real weights: the comparison failed");
  });
}

}  // namespace

int main(int argc, char** argv) {
  if (argc >= 4 && std::string(argv[1]) == "tiny") return Tiny(argv[2], argv[3]);
  if (argc >= 2 && std::string(argv[1]) == "real") return Real(argc, argv);
  std::fprintf(stderr, "usage: drafter_cpu_test tiny TARGET.gguf FIXTURE_DIR | real DRAFTER TARGET DUMP STAGES [s]\n");
  return 2;
}

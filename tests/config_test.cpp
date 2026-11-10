// Config parser: one good case per shape of metadata and one test per error
// case. Every error case mutates a metadata set that parses cleanly (checked
// first), so a test cannot pass by failing for an unrelated reason.
#include <cmath>
#include <limits>
#include <vector>

#include "src/config.hpp"
#include "tests/check.hpp"
#include "tests/gguf_builder.hpp"

using namespace ricursive_test;
using gufo::models::ricursive::Config;
using gufo::models::ricursive::ParseConfig;

namespace {

// Deliberately unequal dimensions: 4 query heads over 2 KV heads, hidden 256
// != heads * head_dim (512), ffn 768, odd vocab.
Metadata Base() {
  return {
      {"general.architecture", std::string("ricursive")},
      {"ricursive.block_count", std::uint32_t{5}},
      {"ricursive.context_length", std::uint32_t{4096}},
      {"ricursive.embedding_length", std::uint32_t{256}},
      {"ricursive.feed_forward_length", std::uint32_t{768}},
      {"ricursive.attention.head_count", std::uint32_t{4}},
      {"ricursive.attention.head_count_kv", std::uint32_t{2}},
      {"ricursive.attention.key_length", std::uint32_t{128}},
      {"ricursive.attention.value_length", std::uint32_t{128}},
      {"ricursive.attention.layer_norm_rms_epsilon", 1e-6F},
      {"ricursive.rope.freq_base", 10000.0F},
      {"ricursive.rope.dimension_count", std::uint32_t{128}},
      {"ricursive.vocab_size", std::uint32_t{77}},
      {"ricursive.attention.sliding_window", std::uint32_t{8}},
      {"ricursive.attention.sliding_window_pattern",
       BoolArray{{true, true, true, true, false}}},
      {"ricursive.embedding_scale", 16.0F},
      {"ricursive.swiglu_clamp", 3.0F},
  };
}

struct Result {
  std::optional<Config> config;
  std::string error;
};

Result Parse(const Metadata& m) {
  GgufImage image(m);
  Require(image.reader() != nullptr, "builder: " + image.error());
  Result r;
  r.config = ParseConfig(*image.reader(), &r.error);
  return r;
}

Config Good(const Metadata& m) {
  const auto r = Parse(m);
  Require(r.config.has_value(), "expected a good config, got: " + r.error);
  return *r.config;
}

/// `mutate` turns a good set into a bad one; the error must mention `needle`.
template<class Fn>
void Bad(const std::string& what, const std::string& needle, Fn&& mutate) {
  Metadata m = Base();
  Good(m);  // the control: unmodified metadata parses
  mutate(m);
  const auto r = Parse(m);
  Require(!r.config.has_value(), what + ": expected an error, config parsed");
  RequireContains(r.error, needle, what);
  std::cout << "  ok  " << what << " -> " << r.error << '\n';
}

void GoodCases() {
  {  // scalar clamp, pattern given, no rope_pattern
    const Config c = Good(Base());
    Require(c.block_count == 5 && c.embedding_length == 256 &&
                c.feed_forward_length == 768 && c.head_count == 4 &&
                c.head_count_kv == 2 && c.head_dim == 128 && c.vocab_size == 77 &&
                c.sliding_window == 8 && c.context_length == 4096,
            "integer fields");
    Require(c.GqaGroup() == 2 && c.QueryWidth() == 512 && c.KvWidth() == 256,
            "derived widths");
    Require(c.rms_eps == 1e-6F && c.rope_freq_base == 10000.0F &&
                c.embedding_scale == 16.0F,
            "float fields");
    const std::vector<bool> want{true, true, true, true, false};
    Require(c.sliding == want, "pattern");
    for (std::uint32_t il = 0; il < 5; ++il) {
      Require(c.swiglu_clamp[il] == 3.0F, "scalar clamp broadcast");
      Require(c.HasRope(il) == want[il], "rope exactly on sliding layers");
    }
  }
  {  // per-layer clamp array, explicit equal rope_pattern, zero clamp entry
    Metadata m = Base();
    m["ricursive.swiglu_clamp"] = F32Array{{1.0F, 2.0F, 0.0F, 4.5F, 20.0F}};
    m["ricursive.attention.rope_pattern"] =
        BoolArray{{true, true, true, true, false}};
    const Config c = Good(m);
    const std::vector<float> want{1.0F, 2.0F, 0.0F, 4.5F, 20.0F};
    Require(c.swiglu_clamp == want, "clamp array");
  }
  {  // no pattern: every 5th layer is full (layers 4 and 9 of 10)
    Metadata m = Base();
    m["ricursive.block_count"] = std::uint32_t{10};
    m.erase("ricursive.attention.sliding_window_pattern");
    const Config c = Good(m);
    for (std::uint32_t il = 0; il < 10; ++il) {
      Require(c.sliding[il] == (il != 4 && il != 9), "default pattern");
    }
  }
  {  // a different pattern is honoured, not the default
    Metadata m = Base();
    m["ricursive.attention.sliding_window_pattern"] =
        BoolArray{{false, true, true, false, true}};
    const Config c = Good(m);
    const std::vector<bool> want{false, true, true, false, true};
    Require(c.sliding == want, "explicit pattern");
  }
}

void ErrorCases() {
  const char* const required[] = {
      "block_count", "context_length", "embedding_length",
      "feed_forward_length", "attention.head_count", "attention.head_count_kv",
      "attention.key_length", "attention.value_length",
      "attention.layer_norm_rms_epsilon", "rope.freq_base",
      "rope.dimension_count", "vocab_size", "attention.sliding_window",
      "embedding_scale", "swiglu_clamp"};
  for (const char* suffix : required) {
    const std::string key = std::string("ricursive.") + suffix;
    Bad("missing " + key, "missing metadata key " + key,
        [&](Metadata& m) { m.erase(key); });
  }
  Bad("wrong architecture", "general.architecture",
      [](Metadata& m) { m["general.architecture"] = std::string("llama"); });
  Bad("key_length != value_length", "key_length (128) != ricursive.attention.value_length (64)",
      [](Metadata& m) { m["ricursive.attention.value_length"] = std::uint32_t{64}; });
  Bad("heads not divisible by kv heads", "not divisible",
      [](Metadata& m) { m["ricursive.attention.head_count_kv"] = std::uint32_t{3}; });
  Bad("pattern too short", "sliding_window_pattern has 4 entries, expected block_count = 5",
      [](Metadata& m) {
        m["ricursive.attention.sliding_window_pattern"] =
            BoolArray{{true, true, true, false}};
      });
  Bad("pattern too long", "sliding_window_pattern has 6 entries",
      [](Metadata& m) {
        m["ricursive.attention.sliding_window_pattern"] =
            BoolArray{{true, true, true, true, false, true}};
      });
  Bad("pattern not boolean", "boolean",
      [](Metadata& m) {
        m["ricursive.attention.sliding_window_pattern"] =
            U32Array{{1, 1, 2, 1, 0}};
      });
  Bad("pattern is a scalar", "boolean array",
      [](Metadata& m) { m["ricursive.attention.sliding_window_pattern"] = true; });
  Bad("rope_pattern wrong length", "rope_pattern has 3 entries",
      [](Metadata& m) {
        m["ricursive.attention.rope_pattern"] = BoolArray{{true, true, false}};
      });
  Bad("rope_pattern differs from sliding pattern", "layer 4 has RoPE but no sliding window",
      [](Metadata& m) {
        m["ricursive.attention.rope_pattern"] =
            BoolArray{{true, true, true, true, true}};
      });
  Bad("rope on a full layer only", "layer 0 has no RoPE but a sliding window",
      [](Metadata& m) {
        m["ricursive.attention.rope_pattern"] =
            BoolArray{{false, true, true, true, false}};
      });
  Bad("all layers full", "no sliding-window layer",
      [](Metadata& m) {
        m["ricursive.attention.sliding_window_pattern"] =
            BoolArray{{false, false, false, false, false}};
      });
  Bad("rope dimension != key_length", "rope.dimension_count (64) != ricursive.attention.key_length (128)",
      [](Metadata& m) { m["ricursive.rope.dimension_count"] = std::uint32_t{64}; });
  Bad("scalar clamp NaN", "must be finite and non-negative",
      [](Metadata& m) {
        m["ricursive.swiglu_clamp"] = std::numeric_limits<float>::quiet_NaN();
      });
  Bad("scalar clamp infinite", "must be finite and non-negative",
      [](Metadata& m) {
        m["ricursive.swiglu_clamp"] = std::numeric_limits<float>::infinity();
      });
  Bad("scalar clamp negative", "must be finite and non-negative",
      [](Metadata& m) { m["ricursive.swiglu_clamp"] = -1.0F; });
  Bad("array clamp has a NaN entry", "clamp for layer 3",
      [](Metadata& m) {
        m["ricursive.swiglu_clamp"] = F32Array{
            {1.0F, 1.0F, 1.0F, std::numeric_limits<float>::quiet_NaN(), 1.0F}};
      });
  Bad("array clamp has a negative entry", "clamp for layer 1",
      [](Metadata& m) {
        m["ricursive.swiglu_clamp"] = F32Array{{1.0F, -0.5F, 1.0F, 1.0F, 1.0F}};
      });
  Bad("array clamp wrong length", "swiglu_clamp has 4 entries, expected block_count = 5",
      [](Metadata& m) {
        m["ricursive.swiglu_clamp"] = F32Array{{1.0F, 1.0F, 1.0F, 1.0F}};
      });
  Bad("zero block_count", "block_count must be > 0",
      [](Metadata& m) { m["ricursive.block_count"] = std::uint32_t{0}; });
  Bad("zero sliding_window", "attention.sliding_window must be > 0",
      [](Metadata& m) { m["ricursive.attention.sliding_window"] = std::uint32_t{0}; });
  Bad("integer key stored as float", "not an unsigned 32-bit integer",
      [](Metadata& m) { m["ricursive.embedding_length"] = 256.0F; });
  Bad("float key stored as string", "not a float",
      [](Metadata& m) { m["ricursive.embedding_scale"] = std::string("16"); });
  Bad("non-positive embedding scale", "embedding_scale must be finite and > 0",
      [](Metadata& m) { m["ricursive.embedding_scale"] = 0.0F; });
}

}  // namespace

int main() {
  return Run([] {
    GoodCases();
    ErrorCases();
  });
}

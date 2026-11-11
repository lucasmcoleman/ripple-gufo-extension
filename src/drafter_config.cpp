#include "drafter_config.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <variant>

#include "executor_iface.hpp"

namespace gufo::models::ricursive {
namespace {

using core::GgufMetadataValue;

struct Fail {
  std::string message;
};

std::string Key(std::string_view suffix) { return "dflash." + std::string(suffix); }

std::uint32_t U32Full(const core::GgufReader& r, const std::string& key) {
  if (r.FindMetadata(key) == nullptr) throw Fail{"missing metadata key " + key};
  const auto v = r.GetMetadataUint32(key);
  if (!v) throw Fail{"metadata key " + key + " is not an unsigned 32-bit integer"};
  return *v;
}
std::uint32_t U32(const core::GgufReader& r, std::string_view suffix) {
  return U32Full(r, Key(suffix));
}

float F32(const core::GgufReader& r, std::string_view suffix) {
  const auto key = Key(suffix);
  if (r.FindMetadata(key) == nullptr) throw Fail{"missing metadata key " + key};
  const auto v = r.GetMetadataFloat32(key);
  if (!v) throw Fail{"metadata key " + key + " is not a float"};
  return *v;
}

/// Array of booleans or integers. The reader widens unsigned types to uint64 and signed ones to int64
/// (gguf-py writes a Python list of ints as int32); a negative entry is refused.
std::vector<std::uint64_t> UintArray(const core::GgufReader& r, const std::string& key) {
  const GgufMetadataValue* m = r.FindMetadata(key);
  if (m == nullptr) throw Fail{"missing metadata key " + key};
  if (m->type != core::GgufValueType::kArray) throw Fail{"metadata key " + key + " is not an array"};
  if (const auto* a = std::get_if<std::vector<std::uint64_t>>(&m->value)) return *a;
  if (const auto* a = std::get_if<std::vector<std::int64_t>>(&m->value)) {
    std::vector<std::uint64_t> out;
    for (const auto v : *a) {
      if (v < 0) throw Fail{"metadata key " + key + " holds a negative entry"};
      out.push_back(static_cast<std::uint64_t>(v));
    }
    return out;
  }
  throw Fail{"metadata key " + key + " is not an array of booleans or integers"};
}

std::vector<bool> BoolArray(const core::GgufReader& r, const std::string& key,
                            std::uint32_t layers) {
  const auto a = UintArray(r, key);
  if (a.size() != layers) {
    std::ostringstream s;
    s << "metadata key " << key << " has " << a.size() << " entries, expected block_count = "
      << layers;
    throw Fail{s.str()};
  }
  std::vector<bool> out;
  for (const auto v : a) {
    if (v > 1) throw Fail{"metadata key " + key + " holds a non-boolean entry"};
    out.push_back(v == 1);
  }
  return out;
}

DrafterConfig Parse(const core::GgufReader& r, const Config& target) {
  const auto arch = r.GetMetadataString("general.architecture");
  if (!arch || *arch != "dflash") throw Fail{"general.architecture is not \"dflash\""};
  DrafterConfig c;
  c.block_count = U32(r, "block_count");
  c.context_length = U32(r, "context_length");
  c.embedding_length = U32(r, "embedding_length");
  c.feed_forward_length = U32(r, "feed_forward_length");
  c.head_count = U32(r, "attention.head_count");
  c.head_count_kv = U32(r, "attention.head_count_kv");
  const auto key_length = U32(r, "attention.key_length");
  const auto value_length = U32(r, "attention.value_length");
  c.block_size = U32(r, "block_size");
  c.conv_kernel_size = U32(r, "conv_kernel_size");
  c.conv_group_size = U32(r, "conv_group_size");
  c.selector_rank = U32(r, "selector_rank");
  c.selector_top_k = U32(r, "selector_top_k");
  c.sliding_window = U32(r, "attention.sliding_window");
  c.context_window = U32(r, "attention.context_window");
  c.mask_token_id = U32Full(r, "tokenizer.ggml.mask_token_id");
  c.rms_eps = F32(r, "attention.layer_norm_rms_epsilon");
  c.rope_freq_base = F32(r, "rope.freq_base");

  for (const auto& [name, v] : {std::pair{"block_count", c.block_count},
                                {"embedding_length", c.embedding_length},
                                {"feed_forward_length", c.feed_forward_length},
                                {"attention.head_count", c.head_count},
                                {"attention.head_count_kv", c.head_count_kv},
                                {"attention.key_length", key_length},
                                {"conv_kernel_size", c.conv_kernel_size},
                                {"conv_group_size", c.conv_group_size},
                                {"selector_rank", c.selector_rank},
                                {"selector_top_k", c.selector_top_k},
                                {"attention.sliding_window", c.sliding_window}}) {
    if (v == 0) throw Fail{Key(name) + " must be > 0"};
  }
  if (key_length != value_length) {
    std::ostringstream s;
    s << "dflash.attention.key_length (" << key_length << ") != dflash.attention.value_length ("
      << value_length << ")";
    throw Fail{s.str()};
  }
  c.head_dim = key_length;
  if (c.head_dim % 2 != 0) throw Fail{"head dimension must be even for RoPE"};
  if (c.head_count % c.head_count_kv != 0) {
    std::ostringstream s;
    s << "dflash.attention.head_count (" << c.head_count
      << ") is not divisible by dflash.attention.head_count_kv (" << c.head_count_kv << ")";
    throw Fail{s.str()};
  }
  if (c.embedding_length % c.conv_group_size != 0) {
    std::ostringstream s;
    s << "dflash.embedding_length (" << c.embedding_length
      << ") is not divisible by dflash.conv_group_size (" << c.conv_group_size << ")";
    throw Fail{s.str()};
  }
  if (c.block_size < 2) {
    throw Fail{"dflash.block_size must be >= 2 (the anchor and at least one draft row)"};
  }
  if (c.block_size < c.conv_kernel_size - 1) {
    // The pinned upstream convolution cannot run a shorter block (stage 0 notes, point 3).
    std::ostringstream s;
    s << "dflash.block_size (" << c.block_size << ") is below dflash.conv_kernel_size - 1 ("
      << c.conv_kernel_size - 1 << ")";
    throw Fail{s.str()};
  }
  if (!std::isfinite(c.rms_eps) || c.rms_eps <= 0.0F) {
    throw Fail{"dflash.attention.layer_norm_rms_epsilon must be finite and > 0"};
  }
  if (!std::isfinite(c.rope_freq_base) || c.rope_freq_base <= 0.0F) {
    throw Fail{"dflash.rope.freq_base must be finite and > 0"};
  }

  // The reference implements the non-causal block attention only. A missing key
  // is refused as well: upstream treats a missing flag as causal.
  const auto causal_key = Key("attention.causal");
  if (r.FindMetadata(causal_key) == nullptr) throw Fail{"missing metadata key " + causal_key};
  const auto causal = r.GetMetadataBool(causal_key);
  if (!causal) throw Fail{"metadata key " + causal_key + " is not a boolean"};
  if (*causal) {
    throw Fail{causal_key + " is true: only the non-causal drafter (is_causal false) is implemented"};
  }

  c.sliding = BoolArray(r, Key("attention.sliding_window_pattern"), c.block_count);
  c.rope = BoolArray(r, Key("attention.rope_pattern"), c.block_count);

  for (const auto v : UintArray(r, Key("target_layers"))) {
    if (v > 0xFFFFFFFFULL) throw Fail{"dflash.target_layers holds an entry beyond 32 bits"};
    c.target_layers_stored.push_back(static_cast<std::uint32_t>(v));
  }

  // ---- against the target ------------------------------------------------------------
  if (c.embedding_length != target.embedding_length) {
    std::ostringstream s;
    s << "dflash.embedding_length (" << c.embedding_length
      << ") differs from the target's ricursive.embedding_length (" << target.embedding_length
      << "): this drafter belongs to another model";
    throw Fail{s.str()};
  }
  if (c.mask_token_id >= target.vocab_size) {
    std::ostringstream s;
    s << "tokenizer.ggml.mask_token_id (" << c.mask_token_id
      << ") is outside the target's vocabulary (" << target.vocab_size << " tokens)";
    throw Fail{s.str()};
  }
  const auto& stored = c.target_layers_stored;
  if (stored.empty() || stored.size() > kMaxCaptureLayers) {
    std::ostringstream s;
    s << "dflash.target_layers has " << stored.size() << " entries, expected 1 to "
      << kMaxCaptureLayers;
    throw Fail{s.str()};
  }
  for (std::size_t i = 0; i < stored.size(); ++i) {
    // Stored numbering is the layer input, output of block i = i + 1; 0 would be the embedding.
    if (stored[i] < 1 || stored[i] > target.block_count) {
      std::ostringstream s;
      s << "dflash.target_layers[" << i << "] = " << stored[i]
        << " is outside 1.." << target.block_count
        << " (stored value = target block index + 1; the target has " << target.block_count
        << " blocks)";
      throw Fail{s.str()};
    }
    if (std::count(stored.begin(), stored.end(), stored[i]) != 1) {
      throw Fail{"dflash.target_layers lists a layer twice"};
    }
    c.tap_layers.push_back(stored[i] - 1);
  }
  return c;
}

}  // namespace

std::optional<DrafterConfig> ParseDrafterConfig(const core::GgufReader& reader,
                                                const Config& target, std::string* error) {
  try {
    return Parse(reader, target);
  } catch (const Fail& f) {
    if (error != nullptr) *error = "dflash drafter: " + f.message;
    return std::nullopt;
  }
}

}  // namespace gufo::models::ricursive

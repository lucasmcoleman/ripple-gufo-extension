#include "config.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <variant>

namespace gufo::models::ricursive {
namespace {

using core::GgufMetadataValue;

struct Fail {
  std::string message;
};

std::string Key(std::string_view suffix) {
  return "ricursive." + std::string(suffix);
}

std::uint32_t U32(const core::GgufReader& r, std::string_view suffix) {
  const auto key = Key(suffix);
  if (r.FindMetadata(key) == nullptr) {
    throw Fail{"missing metadata key " + key};
  }
  const auto v = r.GetMetadataUint32(key);
  if (!v) {
    throw Fail{"metadata key " + key + " is not an unsigned 32-bit integer"};
  }
  return *v;
}

float F32(const core::GgufReader& r, std::string_view suffix) {
  const auto key = Key(suffix);
  if (r.FindMetadata(key) == nullptr) {
    throw Fail{"missing metadata key " + key};
  }
  const auto v = r.GetMetadataFloat32(key);
  if (!v) {
    throw Fail{"metadata key " + key + " is not a float"};
  }
  return *v;
}

/// Boolean array. The reader widens bool/uint arrays to uint64; anything but
/// 0 or 1 is rejected. Returns nullopt when the key is absent.
std::optional<std::vector<bool>> BoolArray(const core::GgufReader& r,
                                           std::string_view suffix) {
  const auto key = Key(suffix);
  const GgufMetadataValue* m = r.FindMetadata(key);
  if (m == nullptr) return std::nullopt;
  const auto* a = std::get_if<std::vector<std::uint64_t>>(&m->value);
  if (a == nullptr || m->type != core::GgufValueType::kArray) {
    throw Fail{"metadata key " + key + " is not a boolean array"};
  }
  std::vector<bool> out;
  out.reserve(a->size());
  for (const auto v : *a) {
    if (v > 1) throw Fail{"metadata key " + key + " holds a non-boolean entry"};
    out.push_back(v == 1);
  }
  return out;
}

std::vector<float> Clamp(const core::GgufReader& r, std::uint32_t layers) {
  const auto key = Key("swiglu_clamp");
  const GgufMetadataValue* m = r.FindMetadata(key);
  if (m == nullptr) throw Fail{"missing metadata key " + key};
  std::vector<double> values;
  bool scalar = false;
  if (m->type == core::GgufValueType::kArray) {
    if (const auto* d = std::get_if<std::vector<double>>(&m->value)) {
      values = *d;
    } else if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&m->value)) {
      values.assign(u->begin(), u->end());
    } else if (const auto* i = std::get_if<std::vector<std::int64_t>>(&m->value)) {
      values.assign(i->begin(), i->end());
    } else {
      throw Fail{"metadata key " + key + " is not a number array"};
    }
  } else {
    scalar = true;
    if (const auto* d = std::get_if<double>(&m->value)) {
      values.push_back(*d);
    } else if (const auto* u = std::get_if<std::uint64_t>(&m->value);
               u != nullptr && m->type != core::GgufValueType::kBool) {
      values.push_back(static_cast<double>(*u));
    } else if (const auto* i = std::get_if<std::int64_t>(&m->value)) {
      values.push_back(static_cast<double>(*i));
    } else {
      throw Fail{"metadata key " + key + " is not a number or number array"};
    }
  }
  if (!scalar && values.size() != layers) {
    std::ostringstream s;
    s << "metadata key " << key << " has " << values.size()
      << " entries, expected block_count = " << layers;
    throw Fail{s.str()};
  }
  std::vector<float> out(layers);
  for (std::uint32_t il = 0; il < layers; ++il) {
    const double c = scalar ? values[0] : values[il];
    if (!std::isfinite(c) || c < 0.0 ||
        c > static_cast<double>(std::numeric_limits<float>::max())) {
      std::ostringstream s;
      s << "metadata key " << key << ": clamp for layer " << il
        << " must be finite and non-negative, got " << c;
      throw Fail{s.str()};
    }
    out[il] = static_cast<float>(c);
  }
  return out;
}

Config Parse(const core::GgufReader& r) {
  const auto arch = r.GetMetadataString("general.architecture");
  if (!arch || *arch != "ricursive") {
    throw Fail{"general.architecture is not \"ricursive\""};
  }
  Config c;
  c.block_count = U32(r, "block_count");
  c.context_length = U32(r, "context_length");
  c.embedding_length = U32(r, "embedding_length");
  c.feed_forward_length = U32(r, "feed_forward_length");
  c.head_count = U32(r, "attention.head_count");
  c.head_count_kv = U32(r, "attention.head_count_kv");
  const auto key_length = U32(r, "attention.key_length");
  const auto value_length = U32(r, "attention.value_length");
  c.rms_eps = F32(r, "attention.layer_norm_rms_epsilon");
  c.rope_freq_base = F32(r, "rope.freq_base");
  const auto rope_dim = U32(r, "rope.dimension_count");
  c.vocab_size = U32(r, "vocab_size");
  c.sliding_window = U32(r, "attention.sliding_window");
  c.embedding_scale = F32(r, "embedding_scale");

  for (const auto& [name, v] :
       {std::pair{"block_count", c.block_count},
        {"context_length", c.context_length},
        {"embedding_length", c.embedding_length},
        {"feed_forward_length", c.feed_forward_length},
        {"attention.head_count", c.head_count},
        {"attention.head_count_kv", c.head_count_kv},
        {"attention.key_length", key_length},
        {"vocab_size", c.vocab_size},
        {"attention.sliding_window", c.sliding_window}}) {
    if (v == 0) throw Fail{Key(name) + " must be > 0"};
  }
  if (key_length != value_length) {
    std::ostringstream s;
    s << "ricursive.attention.key_length (" << key_length
      << ") != ricursive.attention.value_length (" << value_length << ")";
    throw Fail{s.str()};
  }
  c.head_dim = key_length;
  if (c.head_dim % 2 != 0) throw Fail{"head dimension must be even for RoPE"};
  if (rope_dim != c.head_dim) {
    std::ostringstream s;
    s << "ricursive.rope.dimension_count (" << rope_dim
      << ") != ricursive.attention.key_length (" << c.head_dim
      << "); partial RoPE is not supported";
    throw Fail{s.str()};
  }
  if (c.head_count % c.head_count_kv != 0) {
    std::ostringstream s;
    s << "ricursive.attention.head_count (" << c.head_count
      << ") is not divisible by ricursive.attention.head_count_kv ("
      << c.head_count_kv << ")";
    throw Fail{s.str()};
  }
  if (!std::isfinite(c.rms_eps) || c.rms_eps <= 0.0F) {
    throw Fail{"ricursive.attention.layer_norm_rms_epsilon must be finite and > 0"};
  }
  if (!std::isfinite(c.rope_freq_base) || c.rope_freq_base <= 0.0F) {
    throw Fail{"ricursive.rope.freq_base must be finite and > 0"};
  }
  if (!std::isfinite(c.embedding_scale) || c.embedding_scale <= 0.0F) {
    throw Fail{"ricursive.embedding_scale must be finite and > 0"};
  }

  auto pattern = BoolArray(r, "attention.sliding_window_pattern");
  if (pattern) {
    if (pattern->size() != c.block_count) {
      std::ostringstream s;
      s << "ricursive.attention.sliding_window_pattern has " << pattern->size()
        << " entries, expected block_count = " << c.block_count;
      throw Fail{s.str()};
    }
    c.sliding = std::move(*pattern);
  } else {
    c.sliding.resize(c.block_count);  // every 5th layer (4, 9, ...) is full
    for (std::uint32_t il = 0; il < c.block_count; ++il) {
      c.sliding[il] = (il % 5) != 4;
    }
  }
  bool any_sliding = false;
  for (const bool s : c.sliding) any_sliding = any_sliding || s;
  if (!any_sliding) {
    throw Fail{
        "no sliding-window layer: the model would have no positional encoding"};
  }
  if (auto rope = BoolArray(r, "attention.rope_pattern")) {
    if (rope->size() != c.block_count) {
      std::ostringstream s;
      s << "ricursive.attention.rope_pattern has " << rope->size()
        << " entries, expected block_count = " << c.block_count;
      throw Fail{s.str()};
    }
    for (std::uint32_t il = 0; il < c.block_count; ++il) {
      if ((*rope)[il] != c.sliding[il]) {
        std::ostringstream s;
        s << "layer " << il << " has " << ((*rope)[il] ? "RoPE" : "no RoPE")
          << " but " << (c.sliding[il] ? "a sliding window" : "no sliding window")
          << " (ricursive.attention.rope_pattern must equal the sliding pattern)";
        throw Fail{s.str()};
      }
    }
  }
  c.swiglu_clamp = Clamp(r, c.block_count);
  return c;
}

}  // namespace

std::optional<Config> ParseConfig(const core::GgufReader& reader,
                                  std::string* error) {
  try {
    return Parse(reader);
  } catch (const Fail& f) {
    if (error != nullptr) *error = "ricursive: " + f.message;
    return std::nullopt;
  }
}

}  // namespace gufo::models::ricursive

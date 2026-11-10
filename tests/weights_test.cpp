// Weight table: synthetic files (each role, each accepted and rejected type,
// missing and mis-shaped tensors) and the real model's 1203-tensor table via
// sparse stand-ins built from the two real 16 MB headers. Tensor data is never
// read: stand-in data is zeros.
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gufo/ext/experimental/ggml_dequant.hpp>
#include <iostream>
#include <map>
#include <set>

#include "src/config.hpp"
#include "src/weights.hpp"
#include "tests/check.hpp"
#include "tests/gguf_builder.hpp"
#include "tests/paths.hpp"

using namespace ricursive_test;
using gufo::core::GgmlType;
using namespace gufo::models::ricursive;
namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kLayers = 5, kE = 256, kFF = 512, kH = 4, kKV = 2,
                        kD = 128, kV = 64;

Metadata BaseMetadata() {
  return {
      {"general.architecture", std::string("ricursive")},
      {"ricursive.block_count", kLayers},
      {"ricursive.context_length", std::uint32_t{4096}},
      {"ricursive.embedding_length", kE},
      {"ricursive.feed_forward_length", kFF},
      {"ricursive.attention.head_count", kH},
      {"ricursive.attention.head_count_kv", kKV},
      {"ricursive.attention.key_length", kD},
      {"ricursive.attention.value_length", kD},
      {"ricursive.attention.layer_norm_rms_epsilon", 1e-6F},
      {"ricursive.rope.freq_base", 10000.0F},
      {"ricursive.rope.dimension_count", kD},
      {"ricursive.vocab_size", kV},
      {"ricursive.attention.sliding_window", std::uint32_t{8}},
      {"ricursive.attention.sliding_window_pattern",
       BoolArray{{true, true, true, true, false}}},
      {"ricursive.embedding_scale", 16.0F},
      {"ricursive.swiglu_clamp", 3.0F},
  };
}

TensorSpec Zeros(std::vector<std::uint64_t> dims, GgmlType type) {
  std::uint64_t n = 1;
  for (auto d : dims) n *= d;
  const std::size_t bytes = gufo::quant::EncodedSizeBytes(type, n);
  Require(bytes != 0, "builder cannot size " + std::string(gufo::core::ToString(type)));
  return TensorSpec{std::move(dims), type, std::vector<std::uint8_t>(bytes, 0)};
}

struct Role {
  const char* name;  // "blk.N." prefix added for layer roles
  bool layer;
  bool matrix;
  std::uint64_t cols, rows;  // rows == 0: vector
};

const std::vector<Role>& Roles() {
  static const std::vector<Role> roles = {
      {"token_embd", false, true, kE, kV},
      {"output_norm", false, false, kE, 0},
      {"output", false, true, kE, kV},
      {"attn_norm", true, false, kE, 0},
      {"attn_q", true, true, kE, kH * kD},
      {"attn_k", true, true, kE, kKV * kD},
      {"attn_v", true, true, kE, kKV * kD},
      {"attn_output", true, true, kH * kD, kE},
      {"attn_q_norm", true, false, kD, 0},
      {"attn_k_norm", true, false, kD, 0},
      {"attn_gate", true, true, kE, kH * kD},
      {"attn_vres_lambda", true, false, 1, 0},
      {"post_attention_norm", true, false, kE, 0},
      {"ffn_norm", true, false, kE, 0},
      {"ffn_gate", true, true, kE, kFF},
      {"ffn_up", true, true, kE, kFF},
      {"ffn_down", true, true, kFF, kE},
      {"post_ffw_norm", true, false, kE, 0},
  };
  return roles;
}

std::string TensorName(const Role& r, std::uint32_t il) {
  return (r.layer ? "blk." + std::to_string(il) + "." : std::string()) + r.name +
         ".weight";
}

/// Complete tensor set. `matrix_type` applies to every matrix role.
Tensors FullSet(GgmlType matrix_type) {
  Tensors t;
  for (const Role& r : Roles()) {
    for (std::uint32_t il = 0; il < (r.layer ? kLayers : 1); ++il) {
      std::vector<std::uint64_t> dims{r.cols};
      if (r.rows != 0) dims.push_back(r.rows);
      t[TensorName(r, il)] =
          Zeros(dims, r.matrix ? matrix_type : GgmlType::kF32);
    }
  }
  return t;
}

struct Resolved {
  std::optional<ModelWeights> weights;
  std::string error;
};

Resolved Resolve(const Metadata& m, const Tensors& t) {
  GgufImage image(m, t);
  Require(image.reader() != nullptr, "builder: " + image.error());
  std::string error;
  const auto config = ParseConfig(*image.reader(), &error);
  Require(config.has_value(), "config: " + error);
  Resolved r;
  r.weights = ResolveWeights(*image.reader(), *config, &r.error);
  if (r.weights) {  // the reader dies with `image`; only inspect, never keep
    Require(r.weights->layers.size() == kLayers, "layer count");
  }
  return r;
}

void Good(const Resolved& r, const std::string& what) {
  Require(r.weights.has_value(), what + ": expected success, got " + r.error);
}
void Bad(const Resolved& r, const std::vector<std::string>& needles,
         const std::string& what) {
  Require(!r.weights.has_value(), what + ": expected failure");
  for (const auto& n : needles) RequireContains(r.error, n, what);
}

void SyntheticTests() {
  const GgmlType accepted[] = {GgmlType::kF32,  GgmlType::kBF16, GgmlType::kQ8_0,
                               GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                               GgmlType::kIQ4_XS};
  for (GgmlType type : accepted) {
    GgufImage image(BaseMetadata(), FullSet(type));
    Require(image.reader() != nullptr, "builder: " + image.error());
    std::string error;
    const auto config = ParseConfig(*image.reader(), &error);
    Require(config.has_value(), error);
    const auto w = ResolveWeights(*image.reader(), *config, &error);
    Require(w.has_value(), std::string("accepted type ") +
                               std::string(gufo::core::ToString(type)) + ": " + error);
    // Shapes and types land in the right slots (unequal dims make a swap visible).
    Require(w->layers[3].attn_q.cols == kE && w->layers[3].attn_q.rows == kH * kD &&
                w->layers[3].attn_q.type == type,
            "attn_q");
    Require(w->layers[4].attn_k.rows == kKV * kD && w->layers[4].attn_v.rows == kKV * kD,
            "attn_k/v");
    Require(w->layers[0].attn_output.cols == kH * kD && w->layers[0].attn_output.rows == kE,
            "attn_output");
    Require(w->layers[2].ffn_down.cols == kFF && w->layers[2].ffn_down.rows == kE &&
                w->layers[2].ffn_gate.rows == kFF,
            "ffn");
    Require(w->token_embd.rows == kV && w->output.rows == kV &&
                w->token_embd.data != w->output.data,
            "embd/output are separate tensors");
    Require(w->layers[1].attn_vres_lambda.cols == 1 &&
                w->layers[1].attn_vres_lambda.type == GgmlType::kF32,
            "vres lambda");
    Require(w->layers[0].attn_q.RowBytes() ==
                gufo::quant::EncodedSizeBytes(type, kE),
            "row bytes");
    Require(std::string(w->layers[3].ffn_up.name) == "blk.3.ffn_up.weight", "name");
  }
  std::cout << "  ok  all 7 accepted matrix types resolve with the right shapes\n";

  // Rejected matrix types, per role: name and type must be in the message.
  const std::pair<GgmlType, const char*> rejected[] = {
      {GgmlType::kF16, "F16"}, {GgmlType::kQ4_0, "Q4_0"}, {GgmlType::kQ2_K, "Q2_K"},
      {GgmlType::kIQ3_S, "IQ3_S"}, {GgmlType::kQ3_K, "Q3_K"}};
  for (const Role& r : Roles()) {
    if (!r.matrix) continue;
    for (const auto& [type, label] : rejected) {
      Tensors t = FullSet(GgmlType::kQ4_K);
      const std::uint32_t il = r.layer ? 3 : 0;
      t[TensorName(r, il)] = Zeros({r.cols, r.rows}, type);
      Bad(Resolve(BaseMetadata(), t), {TensorName(r, il), label},
          std::string("rejected ") + label + " on " + r.name);
    }
  }
  std::cout << "  ok  unsupported matrix types fail with tensor name and type (all roles)\n";

  // Norms, q/k norm and lambda: F32 only.
  for (const Role& r : Roles()) {
    if (r.matrix) continue;
    for (GgmlType type : {GgmlType::kBF16, GgmlType::kF16, GgmlType::kQ8_0}) {
      Tensors t = FullSet(GgmlType::kF32);
      const std::uint32_t il = r.layer ? 2 : 0;
      // Q8_0 needs a block-aligned length; lambda (1 element) cannot be built
      // in a block format, so use F16 only there.
      if (type == GgmlType::kQ8_0 && r.cols % 32 != 0) continue;
      t[TensorName(r, il)] = Zeros({r.cols}, type);
      Bad(Resolve(BaseMetadata(), t),
          {TensorName(r, il), std::string(gufo::core::ToString(type)), "F32"},
          std::string("non-F32 ") + r.name);
    }
  }
  std::cout << "  ok  norm / q_norm / k_norm / vres_lambda reject every non-F32 type\n";

  // Missing tensors.
  for (const Role& r : Roles()) {
    Tensors t = FullSet(GgmlType::kQ4_K);
    const std::uint32_t il = r.layer ? 4 : 0;
    t.erase(TensorName(r, il));
    Bad(Resolve(BaseMetadata(), t), {"missing tensor " + TensorName(r, il)},
        std::string("missing ") + r.name);
  }
  std::cout << "  ok  every missing tensor reports its name\n";

  // Wrong shapes: transposed matrices and wrong vector lengths.
  for (const Role& r : Roles()) {
    Tensors t = FullSet(GgmlType::kQ4_K);
    const std::uint32_t il = r.layer ? 1 : 0;
    if (r.matrix) {
      // Swap rows/cols keeping block alignment: use F32 so any shape sizes.
      t[TensorName(r, il)] = Zeros({r.rows, r.cols}, GgmlType::kF32);
    } else {
      t[TensorName(r, il)] = Zeros({r.cols + 1}, GgmlType::kF32);
    }
    Bad(Resolve(BaseMetadata(), t), {TensorName(r, il), "shape"},
        std::string("wrong shape ") + r.name);
  }
  // A 3-D tensor where a matrix belongs.
  {
    Tensors t = FullSet(GgmlType::kF32);
    t["blk.0.ffn_up.weight"] = Zeros({kE, kFF, 1}, GgmlType::kF32);
    Bad(Resolve(BaseMetadata(), t), {"blk.0.ffn_up.weight", "shape"}, "3-d tensor");
  }
  std::cout << "  ok  wrong shapes fail with the tensor name\n";

  // Config that disagrees with the file: same tensors, bigger vocab.
  {
    Metadata m = BaseMetadata();
    m["ricursive.vocab_size"] = kV + 1;
    Bad(Resolve(m, FullSet(GgmlType::kQ4_K)), {"token_embd.weight", "shape"},
        "vocab mismatch");
  }
}

// ---- real headers -----------------------------------------------------------

/// Creates (or reuses) a sparse stand-in: the real 16 MB head followed by a
/// hole up to the real file size, so the reader's extent checks pass without
/// any tensor data existing.
void EnsureStandin(const fs::path& head, const fs::path& out, std::uint64_t size) {
  std::error_code ec;
  if (fs::exists(out, ec) && fs::file_size(out, ec) == size) return;
  fs::copy_file(head, out, fs::copy_options::overwrite_existing);
  fs::resize_file(out, size);
}

void RealHeader(const std::string& label, const fs::path& head, std::uint64_t size,
                const std::string& dir) {
  const fs::path file = fs::path(dir) / (label + ".gguf");
  EnsureStandin(head, file, size);
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(file, &error);
  Require(reader != nullptr, label + ": cannot open stand-in: " + error);
  const auto config = ParseConfig(*reader, &error);
  Require(config.has_value(), label + ": config: " + error);
  const Config& c = *config;
  Require(c.block_count == 80 && c.embedding_length == 5120 &&
              c.feed_forward_length == 40960 && c.head_count == 80 &&
              c.head_count_kv == 8 && c.head_dim == 128 && c.vocab_size == 100352 &&
              c.sliding_window == 1024 && c.GqaGroup() == 10,
          label + ": config values");
  std::size_t full = 0;
  for (std::uint32_t il = 0; il < 80; ++il) {
    Require(c.sliding[il] == (il % 5 != 4), label + ": pattern");
    if (!c.sliding[il]) ++full;
    Require(c.swiglu_clamp[il] == 20.0F, label + ": clamp");
  }
  Require(full == 16, label + ": 16 full layers");
  Require(reader->GetTensorCount() == ExpectedTensorCount(80) &&
              reader->GetTensorCount() == 1203,
          label + ": tensor count " + std::to_string(reader->GetTensorCount()));

  const auto w = ResolveWeights(*reader, c, &error);
  Require(w.has_value(), label + ": " + error);
  // Every tensor of the file is claimed by exactly one slot of the table.
  std::set<std::string_view> claimed;
  std::map<std::string, int> types;
  auto claim = [&](const TensorRef& t) {
    Require(claimed.insert(t.name).second, "claimed twice: " + std::string(t.name));
    ++types[std::string(gufo::core::ToString(t.type))];
  };
  claim(w->token_embd);
  claim(w->output_norm);
  claim(w->output);
  for (const auto& l : w->layers) {
    for (const TensorRef* t :
         {&l.attn_norm, &l.attn_q, &l.attn_k, &l.attn_v, &l.attn_output,
          &l.attn_q_norm, &l.attn_k_norm, &l.attn_gate, &l.attn_vres_lambda,
          &l.post_attention_norm, &l.ffn_norm, &l.ffn_gate, &l.ffn_up,
          &l.ffn_down, &l.post_ffw_norm}) {
      claim(*t);
    }
  }
  Require(claimed.size() == 1203, label + ": claimed " + std::to_string(claimed.size()));
  for (const auto& t : reader->GetTensors()) {
    Require(claimed.contains(t.name), label + ": unclaimed tensor " + std::string(t.name));
  }
  Require(w->layers[79].attn_q.rows == 10240 && w->layers[79].attn_q.cols == 5120 &&
              w->layers[79].attn_k.rows == 1024 && w->layers[0].ffn_down.cols == 40960 &&
              w->layers[0].ffn_down.rows == 5120 && w->token_embd.rows == 100352,
          label + ": spot shapes");
  std::cout << "  ok  " << label << ": 1203/1203 tensors resolved; types";
  for (const auto& [k, v] : types) std::cout << ' ' << k << '=' << v;
  std::cout << '\n';

  // Broken twin: the same header with one tensor's type byte changed to Q2_K
  // must fail naming that tensor and type. The header is patched in a copy.
  {
    const fs::path bad = fs::path(dir) / (label + "-q2k-attn_q7.gguf");
    // Copy the 16 MB head, never the sparse stand-in (a copy could densify it).
    fs::copy_file(head, bad, fs::copy_options::overwrite_existing);
    fs::resize_file(bad, size);
    const std::string needle = "blk.7.attn_q.weight";
    std::ifstream in(head, std::ios::binary);
    std::string buf(4 << 20, '\0');
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const auto at = buf.find(needle);
    Require(at != std::string::npos, "tensor info not found in the first 4 MB");
    // name | u32 n_dims | u64 dims[n_dims] | u32 type | u64 offset
    std::uint32_t n_dims = 0;
    std::memcpy(&n_dims, buf.data() + at + needle.size(), 4);
    Require(n_dims == 2, "attn_q must be 2-d");
    const std::uint64_t type_pos = at + needle.size() + 4 + 8ULL * n_dims;
    std::uint32_t type = 0;
    std::memcpy(&type, buf.data() + type_pos, 4);
    Require(type == static_cast<std::uint32_t>(GgmlType::kQ4_K) ||
                type == static_cast<std::uint32_t>(GgmlType::kQ6_K),
            "unexpected original type of blk.7.attn_q");
    {
      std::fstream f(bad, std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(static_cast<std::streamoff>(type_pos));
      const std::uint32_t q2k = static_cast<std::uint32_t>(GgmlType::kQ2_K);
      f.write(reinterpret_cast<const char*>(&q2k), 4);
      Require(f.good(), "patching the stand-in header");
    }
    std::string e2;
    auto bad_reader = gufo::core::GgufReader::OpenFile(bad, &e2);
    Require(bad_reader != nullptr, "tampered stand-in must still open: " + e2);
    const auto bad_config = ParseConfig(*bad_reader, &e2);
    Require(bad_config.has_value(), e2);
    const auto bad_w = ResolveWeights(*bad_reader, *bad_config, &e2);
    Require(!bad_w.has_value(), "Q2_K attn_q must be rejected");
    RequireContains(e2, needle, "real-header twin");
    RequireContains(e2, "Q2_K", "real-header twin");
    std::cout << "  ok  tampered " << label << " -> " << e2 << '\n';
    fs::remove(bad);
  }
}

}  // namespace

int main() {
  return Run([] {
    SyntheticTests();
    const std::string dir = S1Dir();
    const fs::path head_a = HeadPlain(), head_b = HeadAd();
    if (!fs::exists(head_a) || !fs::exists(head_b)) {
      std::cout << "SKIP real headers (not found)\n";
      return;
    }
    fs::create_directories(dir);
    RealHeader("standin-plain-q4km", head_a, 38798820576ULL, dir);
    RealHeader("standin-ad-q4km", head_b, 40426079712ULL, dir);
  });
}

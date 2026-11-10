// ricursive_ref_logits MODEL.gguf TOKENS.i32 OUT.f32 [--variant NAME] [--taps DIR]
// Teacher-forced float32 logits [T][vocab] from the scalar CPU reference, in
// the file format of oracle/np_ref.py and oracle/bin/ripple_logits. --taps DIR
// writes <tap>-<layer>.npy (layer -1 for global taps), 2-D [rows][width].
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "src/reference.hpp"

using namespace gufo::models::ricursive;

namespace {

void WriteNpy(const std::string& path, const float* d, std::size_t rows, std::size_t width) {
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                       std::to_string(rows) + ", " + std::to_string(width) + "), }";
  while ((10 + header.size() + 1) % 64 != 0) header += ' ';
  header += '\n';
  std::ofstream out(path, std::ios::binary);
  out.write("\x93NUMPY\x01\x00", 8);
  const std::uint16_t len = static_cast<std::uint16_t>(header.size());
  out.write(reinterpret_cast<const char*>(&len), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(d), static_cast<std::streamsize>(rows * width * 4));
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> pos;
  std::string variant_name = "none", taps_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--variant" && i + 1 < argc) {
      variant_name = argv[++i];
    } else if (a == "--taps" && i + 1 < argc) {
      taps_dir = argv[++i];
    } else {
      pos.push_back(a);
    }
  }
  if (pos.size() != 3) {
    std::cerr << "usage: ricursive_ref_logits MODEL.gguf TOKENS.i32 OUT.f32 "
                 "[--variant NAME] [--taps DIR]\nvariants: none";
    for (const Variant v : AllVariants()) std::cerr << ' ' << VariantName(v);
    std::cerr << '\n';
    return 2;
  }
  const auto variant = ParseVariant(variant_name);
  if (!variant) {
    std::cerr << "unknown variant " << variant_name << '\n';
    return 2;
  }
  std::string error;
  const auto reader = gufo::core::GgufReader::OpenFile(pos[0], &error);
  if (!reader) {
    std::cerr << "open " << pos[0] << ": " << error << '\n';
    return 1;
  }
  const auto config = ParseConfig(*reader, &error);
  if (!config) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto weights = ResolveWeights(*reader, *config, &error);
  if (!weights) {
    std::cerr << error << '\n';
    return 1;
  }
  std::ifstream in(pos[1], std::ios::binary | std::ios::ate);
  if (!in) {
    std::cerr << "cannot open " << pos[1] << '\n';
    return 1;
  }
  std::vector<std::int32_t> tokens(static_cast<std::size_t>(in.tellg()) / 4);
  in.seekg(0);
  in.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * 4));

  TapFn tap;
  if (!taps_dir.empty()) {
    std::filesystem::create_directories(taps_dir);
    tap = [&](std::string_view name, int layer, const float* d, std::size_t rows,
              std::size_t width) {
      WriteNpy(taps_dir + "/" + std::string(name) + "-" + std::to_string(layer) + ".npy", d,
               rows, width);
    };
  }
  try {
    const Reference ref(*weights, *variant);
    const auto logits = ref.ForwardFull(tokens, tap);
    std::ofstream out(pos[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(logits.data()),
              static_cast<std::streamsize>(logits.size() * 4));
    if (!out) {
      std::cerr << "cannot write " << pos[2] << '\n';
      return 1;
    }
    std::printf("ricursive_ref_logits: T=%zu vocab=%u variant=%s -> %s\n", tokens.size(),
                weights->config.vocab_size, variant_name.c_str(), pos[2].c_str());
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

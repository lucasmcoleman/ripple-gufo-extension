// TEST-ONLY minimal GGUF metadata reader (v2/v3): enough to pull the tokenizer
// vocabulary and the chat template out of fixtures/vocab-only.gguf.
#ifndef RICURSIVE_TESTS_GGUF_VOCAB_HPP_
#define RICURSIVE_TESTS_GGUF_VOCAB_HPP_

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <vector>

struct GgufVocab {
  std::vector<std::string> tokens, merges;
  std::vector<std::int32_t> token_types;
  std::map<std::string, std::uint64_t> uints;
  std::map<std::string, std::string> strings;
};

inline GgufVocab LoadGgufVocab(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::string buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  if (buf.size() < 24 || buf.compare(0, 4, "GGUF") != 0) { std::fprintf(stderr, "not a GGUF: %s\n", path.c_str()); std::exit(2); }
  std::size_t p = 4;
  auto rd = [&](auto& v) { std::memcpy(&v, buf.data() + p, sizeof v); p += sizeof v; };
  std::uint32_t version; rd(version);
  std::uint64_t ntensors, nkv; rd(ntensors); rd(nkv);
  auto rstr = [&]() { std::uint64_t n; rd(n); std::string s = buf.substr(p, n); p += n; return s; };
  auto sizeof_type = [](std::uint32_t t) -> int {
    switch (t) { case 0: case 1: case 7: return 1; case 2: case 3: return 2; case 4: case 5: case 6: return 4; case 10: case 11: case 12: return 8; default: return 0; }
  };
  GgufVocab g;
  for (std::uint64_t k = 0; k < nkv; ++k) {
    std::string key = rstr();
    std::uint32_t type; rd(type);
    auto scalar = [&](std::uint32_t t) -> std::uint64_t {
      std::uint64_t v = 0; std::memcpy(&v, buf.data() + p, sizeof_type(t)); p += sizeof_type(t); return v;
    };
    if (type == 8) { g.strings[key] = rstr(); }
    else if (type == 9) {
      std::uint32_t et; std::uint64_t n; rd(et); rd(n);
      for (std::uint64_t i = 0; i < n; ++i) {
        if (et == 8) {
          std::string s = rstr();
          if (key == "tokenizer.ggml.tokens") g.tokens.push_back(std::move(s));
          else if (key == "tokenizer.ggml.merges") g.merges.push_back(std::move(s));
        } else {
          std::uint64_t v = scalar(et);
          if (key == "tokenizer.ggml.token_type") g.token_types.push_back(static_cast<std::int32_t>(v));
        }
      }
    } else {
      int sz = sizeof_type(type);
      if (!sz) { std::fprintf(stderr, "unsupported GGUF type %u\n", type); std::exit(2); }
      g.uints[key] = scalar(type);
    }
  }
  return g;
}

inline std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

inline std::string UnHex(const std::string& h) {
  std::string s;
  for (std::size_t i = 0; i + 1 < h.size(); i += 2) s.push_back(static_cast<char>(std::stoi(h.substr(i, 2), nullptr, 16)));
  return s;
}

#endif

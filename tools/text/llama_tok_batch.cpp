// Test oracle: tokenizes many strings with llama.cpp's own tokenizer in ONE
// process (llama-tokenize reloads the model per call).
//   llama_tok_batch VOCAB.gguf IN.hex OUT.txt
// IN.hex: one string per line, hex of its UTF-8 bytes (empty line = empty string).
// OUT.txt: per input line "<ids with parse_special=true> | <ids with parse_special=false>".
// Build (links the prebuilt CPU libllama, no llama.cpp build is started):
//   c++ -O2 -std=c++17 -I$L/include -I$L/ggml/include llama_tok_batch.cpp -o llama_tok_batch \
//       -L$L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "llama.h"

static std::string unhex(const std::string& h) {
  std::string s;
  for (size_t i = 0; i + 1 < h.size(); i += 2) s.push_back((char)std::stoi(h.substr(i, 2), nullptr, 16));
  return s;
}

int main(int argc, char** argv) {
  if (argc != 4) { std::fprintf(stderr, "usage: %s vocab.gguf in.hex out.txt\n", argv[0]); return 2; }
  llama_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
  llama_backend_init();
  llama_model_params mp = llama_model_default_params();
  mp.vocab_only = true;
  llama_model* m = llama_model_load_from_file(argv[1], mp);
  if (!m) { std::fprintf(stderr, "load failed\n"); return 1; }
  const llama_vocab* v = llama_model_get_vocab(m);
  std::ifstream in(argv[2]);
  std::ofstream out(argv[3]);
  std::string line;
  std::vector<llama_token> buf(1 << 16);
  auto run = [&](const std::string& s, bool parse_special) {
    int n = llama_tokenize(v, s.data(), (int32_t)s.size(), buf.data(), (int32_t)buf.size(), false, parse_special);
    if (n < 0) { buf.resize((size_t)-n + 16); n = llama_tokenize(v, s.data(), (int32_t)s.size(), buf.data(), (int32_t)buf.size(), false, parse_special); }
    std::string r;
    for (int i = 0; i < n; ++i) { if (i) r += ' '; r += std::to_string(buf[i]); }
    return r;
  };
  while (std::getline(in, line)) {
    std::string s = unhex(line);
    out << run(s, true) << " | " << run(s, false) << "\n";
  }
  llama_model_free(m);
  return 0;
}

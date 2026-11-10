// Tokenizer tests. Usage: test_tokenizer MODE VOCAB.gguf [args]   (see Makefile)
//   goldens  VOCAB GOLDENS.jsonl          221 goldens, both modes + round trip + streaming
//   prop     VOCAB IN.hex LLAMA.txt|- HF.txt|- [HFNS.txt|-]   compare with llama.cpp (2 columns) and/or HF (1 column)
//   pieces   IN.hex HFPIECES.txt         pre-tokenizer pieces vs Hugging Face (pieces-nocutoff: ICU data uncapped)
//   stream   VOCAB IN.hex                 streaming decoder invariants on the property strings
//   bench    VOCAB                        encode/decode speed on ~1 MB of mixed text
#include <chrono>
#include <cstdio>
#include <sstream>

#include <unicode/uchar.h>
#include <unicode/utf8.h>

#include "../../src/text/json_value.hpp"
#include "../../src/text/tokenizer.hpp"
#include "gguf_vocab.hpp"

using namespace ricursive::text;

static TokenizerOptions g_opts;
static GgufVocab g_vocab;
static std::unique_ptr<Tokenizer> Load(const std::string& path, GgufVocab* keep = nullptr) {
  GgufVocab& g = g_vocab;
  g = LoadGgufVocab(path);
  SpecialTokenIds ids;
  ids.bos = (TokenId)g.uints["tokenizer.ggml.bos_token_id"];
  ids.eos = (TokenId)g.uints["tokenizer.ggml.eos_token_id"];
  ids.eot = (TokenId)g.uints["tokenizer.ggml.eot_token_id"];
  ids.pad = (TokenId)g.uints["tokenizer.ggml.padding_token_id"];
  ids.unk = (TokenId)g.uints["tokenizer.ggml.unknown_token_id"];
  ids.stop = {100265, 100257};
  std::string err;
  auto t = Tokenizer::Create(g.tokens, g.merges, g.token_types, ids, g_opts, &err);
  if (!t) { std::fprintf(stderr, "tokenizer create failed: %s\n", err.c_str()); std::exit(2); }
  if (keep) *keep = g;
  return t;
}

static bool ValidUtf8(const std::string& s) {
  std::size_t i = 0, n = s.size();
  auto b = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
  while (i < n) {
    unsigned c = b(i);
    if (c < 0x80) { ++i; continue; }
    std::size_t need; unsigned lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) need = 1;
    else if (c >= 0xE0 && c <= 0xEF) { need = 2; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
    else if (c >= 0xF0 && c <= 0xF4) { need = 3; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
    else return false;
    for (std::size_t k = 1; k <= need; ++k) {
      if (i + k >= n) return false;
      unsigned cc = b(i + k);
      if (cc < (k == 1 ? lo : 0x80u) || cc > (k == 1 ? hi : 0xBFu)) return false;
    }
    i += need + 1;
  }
  return true;
}

static std::vector<TokenId> ParseIds(const std::string& s) {
  std::vector<TokenId> v; std::istringstream is(s); unsigned long x;
  while (is >> x) v.push_back((TokenId)x);
  return v;
}
static std::string Ids(const std::vector<TokenId>& v) {
  std::string r; for (std::size_t i = 0; i < v.size(); ++i) { if (i) r += ' '; r += std::to_string(v[i]); } return r;
}
static std::string Show(const std::string& s) {
  std::string r;
  for (unsigned char c : s) { if (c >= 0x20 && c < 0x7f) r.push_back((char)c); else { char b[8]; std::snprintf(b, sizeof b, "\\x%02x", c); r += b; } }
  return r;
}

// Round trip + streaming for one string/mode; returns failure text or "".
static std::string CheckDecode(const Tokenizer& t, const std::string& s, const std::vector<TokenId>& ids) {
  if (t.Decode(ids) != s) return "Decode(Encode(s)) != s";
  StreamDecoder sd(t);
  std::string acc;
  for (TokenId id : ids) {
    std::string piece = sd.Push(id);
    if (!ValidUtf8(piece)) return "stream chunk is not valid UTF-8";
    acc += piece;
  }
  acc += sd.Flush();
  if (acc != s) return "stream concatenation != s";
  return "";
}

static int Goldens(const std::string& vocab, const std::string& path) {
  auto t = Load(vocab);
  if (!UnicodeDataIsCurrent()) { std::printf("linked ICU does not know Unicode 16 (U+10D40 is not a digit): upgrade ICU to >= 76\n"); return 1; }
  std::istringstream in(ReadFile(path));
  std::string line; int n = 0, bad = 0, rt = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    JsonValue j; std::string err;
    if (!ParseJson(line, &j, &err)) { std::printf("bad golden json: %s\n", err.c_str()); return 2; }
    const std::string& text = j.Find("text")->text();
    auto ids_of = [&](const char* key) { std::vector<TokenId> v; for (auto& e : j.Find(key)->array()) v.push_back((TokenId)std::stoul(e.text())); return v; };
    ++n;
    auto want_d = ids_of("ids_default"), want_n = ids_of("ids_no_parse_special");
    auto got_d = t->Encode(text, true), got_n = t->Encode(text, false);
    if (got_d != want_d) { ++bad; std::printf("MISMATCH default id=%d text=%s\n  want %s\n  got  %s\n", n - 1, Show(text).c_str(), Ids(want_d).c_str(), Ids(got_d).c_str()); }
    if (got_n != want_n) { ++bad; std::printf("MISMATCH no_parse_special id=%d text=%s\n  want %s\n  got  %s\n", n - 1, Show(text).c_str(), Ids(want_n).c_str(), Ids(got_n).c_str()); }
    for (auto* ids : {&got_d, &got_n}) {
      // Decode round trip: with parse_special the special texts are rendered back literally.
      std::string why = CheckDecode(*t, text, *ids);
      if (!why.empty()) { ++rt; std::printf("ROUNDTRIP FAIL id=%d %s text=%s\n", n - 1, why.c_str(), Show(text).c_str()); }
    }
  }
  std::printf("goldens: %d strings x 2 modes: %d id mismatches, %d round-trip/stream failures\n", n, bad, rt);
  return (bad || rt || n != 221) ? 1 : 0;
}

// True when the text has a code point newer than llama.cpp b12252's Unicode tables.
static bool HasNewCodePoint(const std::string& s) {
  const auto* p = reinterpret_cast<const std::uint8_t*>(s.data());
  int32_t i = 0, n = (int32_t)s.size(); UChar32 c;
  while (i < n) {
    U8_NEXT_OR_FFFD(p, i, n, c);
    UVersionInfo age; u_charAge(c, age);
    if (age[0] >= 16) return true;
  }
  return false;
}

// True when two special-token occurrences in the text overlap partially (e.g. the shared
// "|||" of "|||PHONE_NUMBER|||||EMAIL_ADDRESS|||"): llama.cpp takes the longest token first,
// Hugging Face the leftmost, so they legitimately differ there.
static bool HasOverlappingSpecials(const std::string& s) {
  std::vector<std::pair<std::size_t, std::size_t>> occ;
  for (std::size_t i = 0; i < g_vocab.tokens.size(); ++i) {
    int ty = g_vocab.token_types[i];
    if (ty != 2 && ty != 3 && ty != 4) continue;
    const std::string& tk = g_vocab.tokens[i];
    for (std::size_t p = s.find(tk); p != std::string::npos; p = s.find(tk, p + 1)) occ.push_back({p, p + tk.size()});
  }
  for (std::size_t a = 0; a < occ.size(); ++a)
    for (std::size_t b = a + 1; b < occ.size(); ++b)
      if (occ[a] != occ[b] && occ[a].first < occ[b].second && occ[b].first < occ[a].second) return true;
  return false;
}

// llama_ref: lines "ids(parse_special) | ids(no parse_special)" or "" ; hf_ref: lines "ids" or "".
// A mismatch with llama.cpp is "explained" only if the string holds a code point of Unicode >= 16
// (newer than llama.cpp's tables) AND our ids equal Hugging Face's for that string.
static int Prop(const std::string& vocab, const std::string& hex_path, const std::string& llama_path,
                const std::string& hf_path, const std::string& hfns_path) {
  auto t = Load(vocab);
  std::istringstream hx(ReadFile(hex_path));
  std::istringstream lf(llama_path.empty() ? "" : ReadFile(llama_path)), hf(hf_path.empty() ? "" : ReadFile(hf_path)),
      hfns(hfns_path.empty() ? "" : ReadFile(hfns_path));
  std::string h, l, f, fn;
  int n = 0, llama_bad = 0, expl_unicode = 0, expl_overlap = 0, expl_longs = 0, hf_bad = 0, hfns_bad = 0, rt = 0, shown = 0;
  while (std::getline(hx, h)) {
    std::string s = UnHex(h); ++n;
    auto got_d = t->Encode(s, true), got_n = t->Encode(s, false);
    bool hf_ok = true, hfns_ok = true;
    if (!hf_path.empty()) {
      std::getline(hf, f);
      hf_ok = got_d == ParseIds(f);
      if (!hf_ok) { ++hf_bad; if (shown++ < 10) std::printf("MISMATCH vs HF #%d text=%s\n  hf   %s\n  got  %s\n", n - 1, Show(s).c_str(), f.c_str(), Ids(got_d).c_str()); }
    }
    if (!hfns_path.empty()) {
      std::getline(hfns, fn);
      hfns_ok = got_n == ParseIds(fn);
      if (!hfns_ok) { ++hfns_bad; if (shown++ < 10) std::printf("MISMATCH vs HF (no-parse variant) #%d text=%s\n  hf   %s\n  got  %s\n", n - 1, Show(s).c_str(), fn.c_str(), Ids(got_n).c_str()); }
    }
    if (!llama_path.empty()) {
      std::getline(lf, l);
      auto bar = l.find(" | ");
      bool ok = got_d == ParseIds(l.substr(0, bar)) && got_n == ParseIds(l.substr(bar + 3));
      if (!ok) {
        ++llama_bad;
        // Explained only where we equal Hugging Face AND the input has a known llama.cpp-vs-HF trigger.
        const bool matches_hf = !hf_path.empty() && !hfns_path.empty() && hf_ok && hfns_ok;
        if (matches_hf && HasNewCodePoint(s)) ++expl_unicode;
        else if (matches_hf && HasOverlappingSpecials(s)) ++expl_overlap;
        else if (matches_hf && s.find("'\xC5\xBF") != std::string::npos) ++expl_longs;  // 'ſ: Oniguruma folds it to 's, llama.cpp does not
        else if (shown++ < 10) std::printf("UNEXPLAINED MISMATCH vs llama.cpp #%d text=%s\n  ref  %s\n  got  %s\n  got(no-parse) %s\n", n - 1, Show(s).c_str(), l.c_str(), Ids(got_d).c_str(), Ids(got_n).c_str());
      }
    }
    for (bool ps : {true, false}) {
      std::string why = CheckDecode(*t, s, ps ? got_d : got_n);
      if (!why.empty()) { ++rt; if (rt < 5) std::printf("ROUNDTRIP FAIL #%d %s text=%s\n", n - 1, why.c_str(), Show(s).c_str()); }
    }
  }
  std::printf("prop: %d strings\n", n);
  if (!llama_path.empty()) std::printf("  vs llama.cpp (both modes): %d mismatching strings = %d with a Unicode>=16 code point + %d with overlapping special tokens + %d with U+017F after an apostrophe (in all of these we equal Hugging Face) + %d UNEXPLAINED\n",
                                       llama_bad, expl_unicode, expl_overlap, expl_longs, llama_bad - expl_unicode - expl_overlap - expl_longs);
  if (!hf_path.empty()) std::printf("  vs Hugging Face tokenizers (default mode): %d mismatching strings\n", hf_bad);
  if (!hfns_path.empty()) std::printf("  vs Hugging Face, control tokens removed (parse_special=false): %d mismatching strings\n", hfns_bad);
  std::printf("  decode round trip / stream failures: %d\n", rt);
  return (llama_bad - expl_unicode - expl_overlap - expl_longs || hf_bad || hfns_bad || rt) ? 1 : 0;
}

// Pre-tokenizer pieces (lengths in code points) against Hugging Face's regex engine.
static int Pieces(const std::string& hex_path, const std::string& ref_path) {
  std::istringstream hx(ReadFile(hex_path)), rf(ReadFile(ref_path));
  std::string h, r; long n = 0, bad = 0;
  while (std::getline(hx, h) && std::getline(rf, r)) {
    std::string s = UnHex(h); ++n;
    std::string got;
    std::size_t pos = 0;
    for (std::size_t len : PreTokenizeLengths(s, g_opts.unicode_max_major)) {
      std::size_t cps = 0;
      for (std::size_t k = pos; k < pos + len; ++k) cps += ((static_cast<unsigned char>(s[k]) & 0xC0) != 0x80);
      if (!got.empty()) got += ' ';
      got += std::to_string(cps); pos += len;
    }
    if (got != r) { ++bad; if (bad <= 10) std::printf("PIECES MISMATCH #%ld text=%s\n  hf   %s\n  got  %s\n", n - 1, Show(s).c_str(), r.c_str(), got.c_str()); }
  }
  std::printf("pieces: %ld strings, %ld differ from Hugging Face's pre-tokenizer (unicode_max_major=%d)\n", n, bad, g_opts.unicode_max_major);
  return bad ? 1 : 0;
}

// Streaming decoder invariants beyond the round trip.
static int Stream(const std::string& vocab, const std::string& hex_path) {
  auto t = Load(vocab);
  int fails = 0;
  // 1. Split-character check: emoji/CJK are byte-split across tokens; Push must hold back.
  const std::string probe = "ok 🫠 日本 👨‍👩‍👧‍👦 é";
  auto ids = t->Encode(probe, false);
  StreamDecoder sd(*t);
  int held = 0;
  for (TokenId id : ids) {
    std::size_t before = sd.pending_bytes();
    std::string piece = sd.Push(id);
    if (!ValidUtf8(piece)) { ++fails; std::printf("invalid chunk\n"); }
    if (sd.pending_bytes() > 0) ++held;
    (void)before;
  }
  sd.Flush();
  if (held == 0) { ++fails; std::printf("expected the decoder to hold back at least once on split characters\n"); }
  // 2. The naive per-token decoder does emit invalid UTF-8 on the same stream (the check can fail).
  int naive_bad = 0;
  for (TokenId id : ids) if (!ValidUtf8(t->TokenBytes(id))) ++naive_bad;
  std::printf("stream: held back on %d tokens; naive per-token decode gives %d invalid chunks (must be > 0)\n", held, naive_bad);
  if (naive_bad == 0) ++fails;
  // 3. Invalid byte soup never yields invalid UTF-8; truncated tail flushes to U+FFFD.
  auto byte_id = [&](unsigned b) { return t->Encode(std::string(1, (char)b), false); };
  (void)byte_id;
  const unsigned soup[] = {0xF0, 0x9F, 0x41, 0xC0, 0x80, 0xE2, 0x82, 0x80, 0xFF, 0xF4, 0x90, 0x80, 0x80, 0xED, 0xA0, 0x80, 0xF0, 0x9F, 0x98};
  StreamDecoder sd2(*t);
  std::string acc;
  for (unsigned b : soup) {
    // find the single-byte token for raw byte b through the GPT-2 byte alphabet
    TokenId id = kInvalidTokenId;
    for (TokenId k = 0; k < 512 && id == kInvalidTokenId; ++k) if (t->TokenBytes(k).size() == 1 && (unsigned char)t->TokenBytes(k)[0] == b) id = k;
    std::string piece = sd2.Push(id);
    if (!ValidUtf8(piece)) { ++fails; std::printf("soup: invalid chunk\n"); }
    acc += piece;
  }
  std::string tail = sd2.Flush();
  acc += tail;
  if (!ValidUtf8(acc) || tail != "\xEF\xBF\xBD") { ++fails; std::printf("soup: output invalid or tail not U+FFFD (%s)\n", Show(acc).c_str()); }
  // 4. special tokens and padding ids
  const TokenId im_end = 100265, pad0 = 100278;
  if (t->Decode(std::vector<TokenId>{im_end}, SpecialDecode::kRender) != "<|im_end|>" ||
      !t->Decode(std::vector<TokenId>{im_end}, SpecialDecode::kSkip).empty() ||
      !t->Decode(std::vector<TokenId>{pad0, 100351, 200000}).empty()) { ++fails; std::printf("special/padding decode policy violated\n"); }
  // 5. random property strings: encode (both modes) then stream, chunks always valid
  std::istringstream hx(ReadFile(hex_path)); std::string h; int n = 0;
  while (std::getline(hx, h)) {
    std::string s = UnHex(h); ++n;
    for (bool ps : {true, false}) { std::string why = CheckDecode(*t, s, t->Encode(s, ps)); if (!why.empty()) { ++fails; if (fails < 5) std::printf("%s: %s\n", why.c_str(), Show(s).c_str()); } }
  }
  std::printf("stream: %d property strings x 2 modes checked, failures total %d\n", n, fails);
  return fails ? 1 : 0;
}

static int Bench(const std::string& vocab, const std::string& hex_path) {
  auto t = Load(vocab);
  std::string corpus;
  std::istringstream hx(ReadFile(hex_path)); std::string h;
  std::vector<std::string> pool;
  while (std::getline(hx, h)) pool.push_back(UnHex(h));
  // ~1 MB of mixed text: property strings (multi-script, emoji, code, whitespace) round-robin, plus English prose.
  const std::string prose = "The quick brown fox jumps over the lazy dog. It's 2026, and 1234567 tokens don't count themselves. ";
  for (std::size_t i = 0; corpus.size() < 1000000; ++i) { corpus += pool[i % pool.size()]; if (i % 4 == 0) corpus += prose; }
  using clk = std::chrono::steady_clock;
  double best = 1e9; std::size_t ntok = 0;
  for (int rep = 0; rep < 5; ++rep) {
    auto a = clk::now();
    auto ids = t->Encode(corpus, true);
    double s = std::chrono::duration<double>(clk::now() - a).count();
    ntok = ids.size(); if (s < best) best = s;
  }
  auto ids = t->Encode(corpus, true);
  auto a = clk::now();
  std::string back = t->Decode(ids);
  double d = std::chrono::duration<double>(clk::now() - a).count();
  std::printf("bench: %zu bytes -> %zu tokens, encode best of 5 = %.1f ms (%.1f MB/s, %.0f ktok/s), decode %.2f ms, roundtrip %s\n",
              corpus.size(), ntok, best * 1e3, corpus.size() / best / 1e6, ntok / best / 1e3, d * 1e3, back == corpus ? "ok" : "FAIL");
  // one pathological pre-token: 200 KB of 'a' (no whitespace)
  std::string longword(200000, 'a');
  auto b0 = clk::now(); auto lw = t->Encode(longword, false);
  std::printf("bench: 200000-byte single word -> %zu tokens in %.1f ms\n", lw.size(), std::chrono::duration<double>(clk::now() - b0).count() * 1e3);
  return back == corpus ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s goldens|prop|stream|bench VOCAB ...\n", argv[0]); return 2; }
  std::string mode = argv[1];
  if (mode == "goldens" && argc >= 4) return Goldens(argv[2], argv[3]);
  // prop VOCAB IN.hex LLAMA.txt|- HF.txt|-
  auto opt = [&](int i) { return (i < argc && std::string(argv[i]) != "-") ? std::string(argv[i]) : std::string(); };
  if (mode == "prop" && argc >= 6) return Prop(argv[2], argv[3], opt(4), opt(5), opt(6));
  if (mode == "pieces-nocutoff") { g_opts.unicode_max_major = 0; return Pieces(argv[2], argv[3]); }
  if (mode == "pieces" && argc >= 4) return Pieces(argv[2], argv[3]);
  if (mode == "stream" && argc >= 4) return Stream(argv[2], argv[3]);
  if (mode == "bench" && argc >= 4) return Bench(argv[2], argv[3]);
  std::fprintf(stderr, "bad arguments\n");
  return 2;
}

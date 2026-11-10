#include "tokenizer.hpp"

#include <unicode/uchar.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <queue>

namespace ricursive::text {
namespace {

// ---------------------------------------------------------------- GPT-2 bytes
struct ByteMap {
  std::array<std::uint32_t, 256> to_cp{};
  std::unordered_map<std::uint32_t, std::uint8_t> to_byte;
  ByteMap() {
    std::uint32_t n = 0;
    for (std::uint32_t b = 0; b < 256; ++b) {
      bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      to_cp[b] = direct ? b : 256 + n++;
      to_byte[to_cp[b]] = static_cast<std::uint8_t>(b);
    }
  }
};
const ByteMap& Bytes() { static const ByteMap m; return m; }

void PutUtf8(std::string& o, std::uint32_t c) {
  if (c < 0x80) o.push_back(char(c));
  else if (c < 0x800) { o.push_back(char(0xC0 | (c >> 6))); o.push_back(char(0x80 | (c & 63))); }
  else if (c < 0x10000) { o.push_back(char(0xE0 | (c >> 12))); o.push_back(char(0x80 | ((c >> 6) & 63))); o.push_back(char(0x80 | (c & 63))); }
  else { o.push_back(char(0xF0 | (c >> 18))); o.push_back(char(0x80 | ((c >> 12) & 63))); o.push_back(char(0x80 | ((c >> 6) & 63))); o.push_back(char(0x80 | (c & 63))); }
}

// GPT-2 alphabet string -> raw bytes. A character outside the alphabet is kept literally.
std::string Unbytelevel(const std::string& tok) {
  std::string out;
  const auto* p = reinterpret_cast<const std::uint8_t*>(tok.data());
  const int32_t n = static_cast<int32_t>(tok.size());
  for (int32_t i = 0; i < n;) {
    int32_t start = i; UChar32 c;
    U8_NEXT_OR_FFFD(p, i, n, c);
    auto it = Bytes().to_byte.find(static_cast<std::uint32_t>(c));
    if (it != Bytes().to_byte.end()) out.push_back(char(it->second));
    else out.append(tok, static_cast<std::size_t>(start), static_cast<std::size_t>(i - start));
  }
  return out;
}

// ----------------------------------------------------------- pre-tokenizer
struct Cp {
  std::int32_t c;
  std::uint32_t len;
  bool letter, number, space;
};

inline Cp At(std::string_view t, std::size_t pos, int max_major) {
  Cp r{-1, 0, false, false, false};
  if (pos >= t.size()) return r;
  const auto* p = reinterpret_cast<const std::uint8_t*>(t.data());
  int32_t i = static_cast<int32_t>(pos); UChar32 c;
  U8_NEXT_OR_FFFD(p, i, static_cast<int32_t>(t.size()), c);
  r.c = c; r.len = static_cast<std::uint32_t>(i - static_cast<int32_t>(pos));
  if (c < 0x80) {  // ASCII fast path
    r.letter = (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
    r.number = c >= '0' && c <= '9';
    r.space = c == ' ' || (c >= 9 && c <= 13);
  } else {
    const std::uint32_t mask = U_GET_GC_MASK(c);
    r.letter = (mask & U_GC_L_MASK) != 0;
    r.number = (mask & U_GC_N_MASK) != 0;
    r.space = u_isUWhiteSpace(c) != 0;
    if (max_major > 0 && (r.letter || r.number)) {  // newer than the reference regex engine: unassigned
      UVersionInfo age;
      u_charAge(c, age);
      if (age[0] > max_major) r.letter = r.number = false;
    }
  }
  return r;
}
inline bool IsNl(const Cp& c) { return c.c == '\r' || c.c == '\n'; }
inline bool IsPunct(const Cp& c) { return c.len != 0 && !c.space && !c.letter && !c.number; }

// Byte length of the contraction after an apostrophe at `pos`, 0 if none:
// (?i:'s|'t|'re|'ve|'m|'ll|'d). Case-insensitive the way the Oniguruma engine
// is (Unicode simple case folding), so U+017F LATIN SMALL LETTER LONG S matches
// the 's of "'s". (llama.cpp only lower-cases ASCII and does not match it.)
std::size_t Contraction(std::string_view t, std::size_t pos) {
  if (pos >= t.size() || t[pos] != '\'') return 0;
  auto folded = [&](std::size_t k, std::size_t* len) -> std::int32_t {
    if (k >= t.size()) { *len = 0; return -1; }
    const auto* p = reinterpret_cast<const std::uint8_t*>(t.data());
    int32_t i = static_cast<int32_t>(k); UChar32 c;
    U8_NEXT_OR_FFFD(p, i, static_cast<int32_t>(t.size()), c);
    *len = static_cast<std::size_t>(i) - k;
    if (c < 0x80) return (c >= 'A' && c <= 'Z') ? c + 32 : c;
    return u_foldCase(c, U_FOLD_CASE_DEFAULT);
  };
  std::size_t l1, l2;
  const std::int32_t a = folded(pos + 1, &l1);
  if (a == 's' || a == 't' || a == 'm' || a == 'd') return 1 + l1;
  if (a == 'r' || a == 'v' || a == 'l') {
    const std::int32_t b = folded(pos + 1 + l1, &l2);
    if ((a == 'l' && b == 'l') || (a != 'l' && b == 'e')) return 1 + l1 + l2;
  }
  return 0;
}

std::size_t PieceEnd(std::string_view t, std::size_t pos, int mm) {
  if (std::size_t k = Contraction(t, pos)) return pos + k;
  const Cp c0 = At(t, pos, mm);

  // [^\r\n\p{L}\p{N}]?\p{L}+
  if (c0.letter || (!IsNl(c0) && !c0.number)) {
    std::size_t q = c0.letter ? pos : pos + c0.len;
    Cp cq = At(t, q, mm);
    if (cq.letter) {
      while (cq.letter) { q += cq.len; cq = At(t, q, mm); }
      return q;
    }
  }
  // \p{N}{1,3}
  if (c0.number) {
    std::size_t q = pos;
    for (int n = 0; n < 3; ++n) {
      Cp c = At(t, q, mm);
      if (!c.number) break;
      q += c.len;
    }
    return q;
  }
  // ' '?[^\s\p{L}\p{N}]+[\r\n]*
  {
    std::size_t q = pos;
    if (c0.c == ' ' && IsPunct(At(t, pos + 1, mm))) q = pos + 1;
    Cp c = At(t, q, mm);
    if (IsPunct(c)) {
      while (IsPunct(c)) { q += c.len; c = At(t, q, mm); }
      while (IsNl(c)) { q += c.len; c = At(t, q, mm); }
      return q;
    }
  }
  // whitespace run: \s*[\r\n]+ | \s+(?!\S) | \s+
  std::size_t q = pos, last_start = pos, last_nl_end = pos;
  std::size_t count = 0;
  for (Cp c = c0; c.space; c = At(t, q, mm)) {
    last_start = q;
    q += c.len;
    ++count;
    if (IsNl(c)) last_nl_end = q;
  }
  if (count == 0) return pos + std::max<std::uint32_t>(c0.len, 1);  // cannot happen; progress guard
  if (last_nl_end != pos) return last_nl_end;
  if (q >= t.size()) return q;
  return count >= 2 ? last_start : q;
}

}  // namespace

std::vector<std::size_t> PreTokenizeLengths(std::string_view text, int unicode_max_major) {
  std::vector<std::size_t> out;
  for (std::size_t pos = 0; pos < text.size();) {
    std::size_t end = PieceEnd(text, pos, unicode_max_major);
    out.push_back(end - pos);
    pos = end;
  }
  return out;
}

bool UnicodeDataIsCurrent() { return (U_GET_GC_MASK(0x10D40) & U_GC_ND_MASK) != 0; }

// ------------------------------------------------------------------ create
std::unique_ptr<Tokenizer> Tokenizer::Create(
    std::span<const std::string> tokens, std::span<const std::string> merges,
    std::span<const std::int32_t> types, const SpecialTokenIds& ids,
    const TokenizerOptions& options, std::string* error) {
  auto fail = [&](std::string m) -> std::unique_ptr<Tokenizer> {
    if (error) *error = std::move(m);
    return nullptr;
  };
  if (tokens.empty()) return fail("empty vocabulary");
  if (!types.empty() && types.size() != tokens.size()) return fail("token_type size != tokens size");

  std::unique_ptr<Tokenizer> t(new Tokenizer());
  t->ids_ = ids;
  t->unicode_max_major_ = options.unicode_max_major;
  if (t->ids_.stop.empty()) {
    for (TokenId id : {ids.eos, ids.eot})
      if (id != kInvalidTokenId && std::find(t->ids_.stop.begin(), t->ids_.stop.end(), id) == t->ids_.stop.end())
        t->ids_.stop.push_back(id);
  }
  const std::size_t n = tokens.size();
  t->bytes_.resize(n);
  t->kind_.assign(n, 0);
  t->token_to_id_.reserve(n * 2);
  for (std::size_t i = 0; i < n; ++i) {
    const auto type = types.empty() ? TokenType::kNormal : static_cast<TokenType>(types[i]);
    const auto id = static_cast<TokenId>(i);
    if (type == TokenType::kUnused) {
      t->kind_[i] = 2;
    } else if (type == TokenType::kControl || type == TokenType::kUnknown || type == TokenType::kUserDefined) {
      t->kind_[i] = 1;
      t->bytes_[i] = tokens[i];
      if (!tokens[i].empty())
        t->specials_.push_back({tokens[i], id, type == TokenType::kUserDefined});
    } else {
      t->bytes_[i] = Unbytelevel(tokens[i]);
    }
    t->token_to_id_.emplace(tokens[i], id);  // first id wins on duplicates
  }
  std::stable_sort(t->specials_.begin(), t->specials_.end(),
                   [](const SpecialEntry& a, const SpecialEntry& b) { return a.text.size() > b.text.size(); });

  for (std::uint32_t b = 0; b < 256; ++b) {
    std::string s; PutUtf8(s, Bytes().to_cp[b]);
    auto it = t->token_to_id_.find(s);
    if (it == t->token_to_id_.end()) return fail("vocabulary lacks the byte token for " + std::to_string(b));
    t->byte_token_[b] = it->second;
  }
  t->merges_.reserve(merges.size() * 2);
  for (std::uint32_t rank = 0; rank < merges.size(); ++rank) {
    const std::string& m = merges[rank];
    const std::size_t sp = m.find(' ', 1);
    if (sp == std::string::npos) continue;
    auto a = t->token_to_id_.find(m.substr(0, sp));
    auto b = t->token_to_id_.find(m.substr(sp + 1));
    if (a == t->token_to_id_.end() || b == t->token_to_id_.end()) continue;
    auto r = t->token_to_id_.find(m.substr(0, sp) + m.substr(sp + 1));
    if (r == t->token_to_id_.end()) continue;
    t->merges_.emplace((static_cast<std::uint64_t>(a->second) << 32) | b->second, Merge{rank, r->second});
  }
  return t;
}

// -------------------------------------------------------------------- BPE
void Tokenizer::BpeWord(std::string_view word, std::vector<TokenId>& out) const {
  const std::size_t n = word.size();
  if (n == 1) { out.push_back(byte_token_[static_cast<std::uint8_t>(word[0])]); return; }
  struct Sym { TokenId id; std::int32_t prev, next; };
  std::vector<Sym> sym(n);
  for (std::size_t i = 0; i < n; ++i)
    sym[i] = {byte_token_[static_cast<std::uint8_t>(word[i])], static_cast<std::int32_t>(i) - 1,
              i + 1 < n ? static_cast<std::int32_t>(i + 1) : -1};
  struct Big { std::uint32_t rank; std::int32_t left, right; TokenId lid, rid; TokenId result; };
  auto cmp = [](const Big& a, const Big& b) { return a.rank != b.rank ? a.rank > b.rank : a.left > b.left; };
  std::priority_queue<Big, std::vector<Big>, decltype(cmp)> q(cmp);
  auto push = [&](std::int32_t l, std::int32_t r) {
    if (l < 0 || r < 0) return;
    auto it = merges_.find((static_cast<std::uint64_t>(sym[l].id) << 32) | sym[r].id);
    if (it != merges_.end()) q.push({it->second.rank, l, r, sym[l].id, sym[r].id, it->second.token});
  };
  for (std::size_t i = 0; i + 1 < n; ++i) push(static_cast<std::int32_t>(i), static_cast<std::int32_t>(i + 1));
  std::vector<bool> dead(n, false);
  while (!q.empty()) {
    Big b = q.top(); q.pop();
    if (dead[b.left] || dead[b.right] || sym[b.left].next != b.right || sym[b.left].id != b.lid || sym[b.right].id != b.rid) continue;
    sym[b.left].id = b.result;
    dead[b.right] = true;
    sym[b.left].next = sym[b.right].next;
    if (sym[b.right].next >= 0) sym[sym[b.right].next].prev = b.left;
    push(sym[b.left].prev, b.left);
    push(b.left, sym[b.left].next);
  }
  for (std::int32_t i = 0; i >= 0; i = sym[i].next) out.push_back(sym[i].id);
}

void Tokenizer::EncodeRaw(std::string_view text, std::vector<TokenId>& out) const {
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t end = PieceEnd(text, pos, unicode_max_major_);
    BpeWord(text.substr(pos, end - pos), out);
    pos = end;
  }
}

std::vector<TokenId> Tokenizer::Encode(std::string_view text, bool parse_special) const {
  // One left-to-right pass: at each step take the special token that starts
  // first, longest on a tie (Hugging Face's added-token matching).
  std::vector<TokenId> out;
  out.reserve(text.size() / 3 + 8);
  std::vector<const SpecialEntry*> active;
  for (const SpecialEntry& sp : specials_)
    if (parse_special || sp.user_defined) active.push_back(&sp);
  constexpr std::size_t kUnknown = std::string_view::npos - 1;  // occurrence not searched yet
  std::vector<std::size_t> next(active.size(), kUnknown);
  std::size_t pos = 0;
  for (;;) {
    std::size_t best = std::string_view::npos, best_len = 0;
    TokenId best_id = 0;
    for (std::size_t i = 0; i < active.size(); ++i) {
      if (next[i] == std::string_view::npos) continue;
      if (next[i] == kUnknown || next[i] < pos) next[i] = text.find(active[i]->text, pos);
      if (next[i] == std::string_view::npos) continue;
      const std::size_t len = active[i]->text.size();
      if (next[i] < best || (next[i] == best && len > best_len)) { best = next[i]; best_len = len; best_id = active[i]->id; }
    }
    if (best == std::string_view::npos) break;
    if (best > pos) EncodeRaw(text.substr(pos, best - pos), out);
    out.push_back(best_id);
    pos = best + best_len;
  }
  if (pos < text.size()) EncodeRaw(text.substr(pos), out);
  return out;
}

// ----------------------------------------------------------------- decode
const std::string& Tokenizer::TokenBytes(TokenId id, SpecialDecode special) const noexcept {
  if (id >= bytes_.size() || kind_[id] == 2) return empty_;
  if (kind_[id] == 1 && special == SpecialDecode::kSkip) return empty_;
  return bytes_[id];
}

std::string Tokenizer::Decode(std::span<const TokenId> ids, SpecialDecode special) const {
  std::string out;
  for (TokenId id : ids) out += TokenBytes(id, special);
  return out;
}

bool Tokenizer::IsSpecial(TokenId id) const noexcept { return id < kind_.size() && kind_[id] == 1; }
bool Tokenizer::IsStopToken(TokenId id) const noexcept {
  return std::find(ids_.stop.begin(), ids_.stop.end(), id) != ids_.stop.end();
}
TokenId Tokenizer::FindToken(std::string_view raw) const noexcept {
  auto it = token_to_id_.find(std::string(raw));
  return it == token_to_id_.end() ? kInvalidTokenId : it->second;
}

std::string StreamDecoder::Push(TokenId id) {
  pending_ += tok_->TokenBytes(id, special_);
  std::string out;
  const auto* p = reinterpret_cast<const std::uint8_t*>(pending_.data());
  const std::size_t n = pending_.size();
  std::size_t pos = 0;
  while (pos < n) {
    const std::uint8_t b = p[pos];
    std::size_t need;
    std::uint8_t lo = 0x80, hi = 0xBF;  // allowed range of the first continuation byte
    if (b < 0x80) { out.push_back(char(b)); ++pos; continue; }
    else if (b >= 0xC2 && b <= 0xDF) need = 1;
    else if (b >= 0xE0 && b <= 0xEF) { need = 2; if (b == 0xE0) lo = 0xA0; if (b == 0xED) hi = 0x9F; }
    else if (b >= 0xF0 && b <= 0xF4) { need = 3; if (b == 0xF0) lo = 0x90; if (b == 0xF4) hi = 0x8F; }
    else { out += "\xEF\xBF\xBD"; ++pos; continue; }
    std::size_t k = 1;
    bool incomplete = false, bad = false;
    for (; k <= need; ++k) {
      if (pos + k >= n) { incomplete = true; break; }
      const std::uint8_t c = p[pos + k];
      const std::uint8_t l = k == 1 ? lo : 0x80, h = k == 1 ? hi : 0xBF;
      if (c < l || c > h) { bad = true; break; }
    }
    if (incomplete) break;  // hold back
    if (bad) { out += "\xEF\xBF\xBD"; pos += k; continue; }
    out.append(pending_, pos, need + 1);
    pos += need + 1;
  }
  pending_.erase(0, pos);
  return out;
}

std::string StreamDecoder::Flush() {
  std::string out;
  if (!pending_.empty()) out = "\xEF\xBF\xBD";
  pending_.clear();
  return out;
}

}  // namespace ricursive::text

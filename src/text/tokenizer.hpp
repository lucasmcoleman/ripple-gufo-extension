// Byte-level BPE tokenizer for Ripple-65B-Preview (GGUF tokenizer.ggml.model =
// gpt2, tokenizer.ggml.pre = dbrx). Standalone: the vocabulary is plain data.
//
// Pre-tokenizer regex (digits grouped by up to three):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// implemented by hand over ICU character properties (libicuuc only).
// No BOS and no EOS is ever added.
#ifndef RICURSIVE_TEXT_TOKENIZER_HPP_
#define RICURSIVE_TEXT_TOKENIZER_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ricursive::text {

using TokenId = std::uint32_t;
inline constexpr TokenId kInvalidTokenId = 0xFFFFFFFFU;

/// GGUF tokenizer.ggml.token_type values.
enum class TokenType : std::int32_t {
  kUndefined = 0, kNormal = 1, kUnknown = 2, kControl = 3,
  kUserDefined = 4, kUnused = 5, kByte = 6,
};

struct SpecialTokenIds {
  TokenId bos{kInvalidTokenId};
  TokenId eos{kInvalidTokenId};
  TokenId eot{kInvalidTokenId};
  TokenId pad{kInvalidTokenId};
  TokenId unk{kInvalidTokenId};
  /// Ids that end generation. When empty, {eos, eot} (those that are valid).
  std::vector<TokenId> stop;
};

/// What Decode does with CONTROL / UNKNOWN / USER_DEFINED tokens (their text is
/// literal, e.g. "<|im_end|>"). Ids of type UNUSED (the padding ids
/// 100278..100351 carry placeholder strings "[PAD100278]"...) and ids outside
/// the vocabulary always decode to the empty string.
enum class SpecialDecode : std::uint8_t { kRender, kSkip };

struct TokenizerOptions {
  /// Characters assigned only in a Unicode version newer than this are treated
  /// as unassigned (neither letter, number nor space), because that is what
  /// the Hugging Face tokenizers 0.23.3 used to train/serve the model does:
  /// its regex engine knows Unicode 16, so the ~4.6k letters/numbers that
  /// Unicode 17 added (and ICU 78 knows) are "punctuation" to it. 0 = trust the
  /// ICU data fully. Needs ICU >= 76 (Unicode 16) to be exact; see
  /// UnicodeDataIsCurrent().
  int unicode_max_major{16};
};

/// Byte lengths of the pieces the pre-tokenizer regex cuts `text` into
/// (no special-token handling; exposed for tests and debugging).
[[nodiscard]] std::vector<std::size_t> PreTokenizeLengths(std::string_view text, int unicode_max_major = 16);

/// True when the linked ICU knows Unicode 16 (U+10D40 is a decimal digit).
[[nodiscard]] bool UnicodeDataIsCurrent();

class Tokenizer {
public:
  /// `token_types` may be empty (all tokens kNormal). Returns nullptr and sets
  /// *error when the vocabulary is unusable (e.g. a byte token is missing).
  [[nodiscard]] static std::unique_ptr<Tokenizer> Create(
      std::span<const std::string> tokens, std::span<const std::string> merges,
      std::span<const std::int32_t> token_types, const SpecialTokenIds& ids,
      const TokenizerOptions& options = {}, std::string* error = nullptr);

  /// parse_special=true: special token texts (CONTROL/UNKNOWN/USER_DEFINED)
  /// become their ids. false: CONTROL and UNKNOWN texts are ordinary text,
  /// USER_DEFINED tokens (e.g. "|||PHONE_NUMBER|||") are still matched, which
  /// is llama.cpp's behaviour (verified against the goldens). Matching is one
  /// left-to-right pass taking the leftmost, then longest, token (Hugging Face
  /// semantics); llama.cpp instead takes the longest token first, which only
  /// differs for partially overlapping tokens such as
  /// "|||PHONE_NUMBER|||||EMAIL_ADDRESS|||".
  [[nodiscard]] std::vector<TokenId> Encode(std::string_view text, bool parse_special) const;

  /// Raw bytes of the concatenated tokens. Not sanitised: arbitrary ids may
  /// give invalid UTF-8; use StreamDecoder for user-facing text.
  [[nodiscard]] std::string Decode(std::span<const TokenId> ids,
                                   SpecialDecode special = SpecialDecode::kRender) const;
  /// Bytes of one token under the same rules (empty for skipped tokens).
  [[nodiscard]] const std::string& TokenBytes(TokenId id, SpecialDecode special = SpecialDecode::kRender) const noexcept;

  [[nodiscard]] std::size_t VocabSize() const noexcept { return bytes_.size(); }
  [[nodiscard]] bool IsSpecial(TokenId id) const noexcept;  ///< CONTROL/UNKNOWN/USER_DEFINED
  [[nodiscard]] bool IsStopToken(TokenId id) const noexcept;
  [[nodiscard]] const SpecialTokenIds& ids() const noexcept { return ids_; }
  [[nodiscard]] TokenId FindToken(std::string_view raw_text) const noexcept;

private:
  Tokenizer() = default;
  struct Merge { std::uint32_t rank; TokenId token; };
  struct SpecialEntry { std::string text; TokenId id; bool user_defined; };

  void EncodeRaw(std::string_view text, std::vector<TokenId>& out) const;
  void BpeWord(std::string_view word, std::vector<TokenId>& out) const;

  std::vector<std::string> bytes_;      // decoded bytes per id (literal text for specials)
  std::vector<std::uint8_t> kind_;      // 0 normal, 1 special, 2 unused
  std::unordered_map<std::string, TokenId> token_to_id_;  // raw GGUF strings
  std::unordered_map<std::uint64_t, Merge> merges_;
  std::array<TokenId, 256> byte_token_{};
  std::vector<SpecialEntry> specials_;  // longest text first
  SpecialTokenIds ids_;
  int unicode_max_major_{16};
  std::string empty_;
};

/// Incremental decoder: Push() returns only complete UTF-8 characters, holding
/// back an incomplete tail until later tokens finish it. Bytes that cannot be
/// part of valid UTF-8 are replaced by U+FFFD (one per maximal invalid
/// subpart), so the output is always valid UTF-8; for a token stream whose
/// concatenation is valid UTF-8 this never triggers. Flush() turns a truncated
/// tail into one U+FFFD.
class StreamDecoder {
public:
  explicit StreamDecoder(const Tokenizer& tok, SpecialDecode special = SpecialDecode::kRender)
      : tok_(&tok), special_(special) {}
  std::string Push(TokenId id);
  std::string Flush();
  [[nodiscard]] std::size_t pending_bytes() const noexcept { return pending_.size(); }

private:
  const Tokenizer* tok_;
  SpecialDecode special_;
  std::string pending_;
};

}  // namespace ricursive::text

#endif  // RICURSIVE_TEXT_TOKENIZER_HPP_

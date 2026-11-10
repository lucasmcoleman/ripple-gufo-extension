#!/bin/sh
# Negative controls: deliberately broken copies of the production sources must FAIL the tests.
# usage: negative.sh   (env: CXX CXXFLAGS ICU_LIBS VOCAB TOKGOLD CHATGOLD S1 BUILD)
set -u
SRC=../../src/text
N=$BUILD/neg
rm -rf "$N"; mkdir -p "$N"
status=0

# mutate NAME FILE 'sed expression'  -> $N/NAME/FILE (refuses a no-op mutation)
mutate() {
  mkdir -p "$N/$1"
  sed "$3" "$SRC/$2" > "$N/$1/$2"
  if cmp -s "$SRC/$2" "$N/$1/$2"; then echo "NEGATIVE $1: mutation did not change $2 (broken control)"; status=1; return 1; fi
}
expect_fail() {  # NAME COMMAND...
  name=$1; shift
  out=$("$@" 2>&1); rc=$?
  if [ $rc -eq 0 ]; then echo "NEGATIVE $name: test PASSED with a broken implementation (test cannot fail!)"; status=1
  elif [ $rc -ne 1 ]; then echo "NEGATIVE $name: broken control (exit $rc, not a test failure): $out" | head -3; status=1
  else echo "NEGATIVE $name: caught (exit $rc): $(printf '%s\n' "$out" | grep -E '^(goldens|chat cases|pieces|prop|  vs )' | head -2 | tr '\n' ' ')"; fi
}

tok() {  # NAME FILE sed  -> build mutated tokenizer test
  mutate "$1" tokenizer.cpp "$2" || return
  $CXX $CXXFLAGS -I$SRC test_tokenizer.cpp "$N/$1/tokenizer.cpp" $SRC/json_value.cpp $ICU_LIBS -o "$N/$1/t" 2>&1 | grep -E "error" ; true
}
chat() {  # NAME FILE sed
  mutate "$1" "$2" "$3" || return
  if [ "$2" = chat_template.cpp ]; then ct="$N/$1/chat_template.cpp"; jv="$SRC/json_value.cpp"; else ct="$SRC/chat_template.cpp"; jv="$N/$1/json_value.cpp"; fi
  $CXX $CXXFLAGS -I$SRC test_chat.cpp "$ct" "$jv" -o "$N/$1/t" 2>&1 | grep -E "error" ; true
}

# --- tokenizer ---
tok qwen_single_digit 's/for (int n = 0; n < 3; ++n)/for (int n = 0; n < 1; ++n)/'
expect_fail "tokenizer: Qwen-style single-digit \\p{N} vs goldens" "$N/qwen_single_digit/t" goldens "$VOCAB" "$TOKGOLD"
tok ascii_letters 's/r.letter = (mask \& U_GC_L_MASK) != 0;/r.letter = false;/'
expect_fail "tokenizer: ASCII-only letter class vs goldens" "$N/ascii_letters/t" goldens "$VOCAB" "$TOKGOLD"
tok no_prefix_char 's/if (c0.letter || (!IsNl(c0) \&\& !c0.number)) {/if (c0.letter) {/'
expect_fail "tokenizer: no optional prefix char before letters vs goldens" "$N/no_prefix_char/t" goldens "$VOCAB" "$TOKGOLD"
tok no_parse_special_ignored 's/if (parse_special || sp.user_defined) active.push_back(\&sp);/active.push_back(\&sp);/'
expect_fail "tokenizer: parse_special=false still parsing control tokens vs goldens" "$N/no_parse_special_ignored/t" goldens "$VOCAB" "$TOKGOLD"
tok no_cutoff 's/if (age\[0\] > max_major) r.letter = r.number = false;/(void)age;/'
expect_fail "tokenizer: no Unicode-16 cutoff vs Hugging Face (property strings)" "$N/no_cutoff/t" prop "$VOCAB" "$S1/prop.hex" - "$S1/prop-hf.txt"
expect_fail "tokenizer: unicode_max_major=0 pieces vs Hugging Face (all code points)" "$BUILD/test_tokenizer" pieces-nocutoff "$S1/cp8.hex" "$S1/cp8-hfpieces.txt"

# --- chat template ---
chat think_no_newline chat_template.cpp 's/c = "<think>\\n";/c = "<think>";/'
expect_fail "chat: missing newline after <think>" "$N/think_no_newline/t" cases "$CHATGOLD"
chat json_separators json_value.cpp 's/o += ", ";/o += ",";/'
expect_fail "chat: JSON item separator ',' instead of ', '" "$N/json_separators/t" cases "$CHATGOLD"
chat json_ensure_ascii json_value.cpp 's/else o.push_back(char(c));/else if (c >= 0x80) { char b[8]; std::snprintf(b, sizeof b, "\\\\u%04x", c); o += b; } else o.push_back(char(c));/'
expect_fail "chat: non-ASCII escaped (ensure_ascii=True)" "$N/json_ensure_ascii/t" cases "$CHATGOLD"
chat tool_response_nl chat_template.cpp 's|out += "\\n</tool_response>";|out += "</tool_response>";|'
expect_fail "chat: no newline before </tool_response>" "$N/tool_response_nl/t" cases "$CHATGOLD"
chat lstrip_ascii_only chat_template.cpp 's/c == 0xA0 || c == 0x1680 || (c >= 0x2000 \&\& c <= 0x200A) ||/c == 0xA0 ||/'
expect_fail "chat: lstrip without Unicode spaces vs property cases" "$N/lstrip_ascii_only/t" cases "$S1/chat-prop.jsonl"
chat float_repr json_value.cpp 's/if (e10 < -4 || e10 >= 16) {/if (e10 < -4 || e10 >= 17) {/'
expect_fail "chat: Python float repr threshold off by one vs property cases" "$N/float_repr/t" cases "$S1/chat-prop.jsonl"
exit $status

#!/usr/bin/env python3
"""Seeded random strings for the tokenizer property test.
usage: gen_tok_prop.py OUT.hex [N=6000] [SEED=20261110]
One string per line, hex of its UTF-8 bytes. Same seed -> same file."""
import random, sys

out = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
rng = random.Random(int(sys.argv[3]) if len(sys.argv) > 3 else 20261110)

words = "the of and to in is that for it as with was on be by this at from or an are not have but which you all we can there their what more when your said do if will each about how up out them then she many some so these would other into has".split()
contr = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'RE", "'Ve", "'M", "'LL", "'lL", "'D",
         "\u2019s", "\u02bcs", "'ſ", "'\u212a", "'S\u0301", "''s", "'", "'x", "'r", "'l", "'v", "s'", "'sit", "'dd", "'ll'"]
words_cased = ["I", "DON", "don", "O", "Neil", "can", "won", "they", "We", "YOU", "it", "he", "She", "ma", "o"]
scripts = ["你好世界", "日本語のテキスト", "ひらがな", "カタカナ", "한국어텍스트", "Привет мир", "مرحبا بالعالم", "שלום עולם",
           "Ελληνικά", "नमस्ते दुनिया", "ไทยภาษา", "ĀĒĪŌŪ", "ñandú", "Ǆ ǅ ǆ", "ᚠᚢᚦ", "ꯀꯁ", "𝒜𝓑𝓒", "𠀀𠀁", "ａｂｃ", "ｱｲｳ",
           "ʰʲˆ", "ªº", "ǅ", "ᾈ", "ϴ", "ª", "ⓐⒶ", "ℵℶ", "ǈ"]
digits_other = ["٣٤٥", "१२३", "１２３４５", "²³", "½¼", "Ⅻ", "①②③", "೧೨", "๑๒๓", "⁴", "𝟙𝟚𝟛", "ⅷ", "〇", "一二三", "㊿", "٠١٢٣٤٥٦٧٨٩"]
marks = ["\u0301", "\u0300\u0301", "\u0308", "\u200d", "\u200c", "\ufe0f", "\ufe0e", "\u20e3", "\u0e31", "\u093f", "\u3099", "\u0489", "\u064b", "\u1ab0"]
emoji = ["😀", "👍🏽", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🇺🇸", "🇯🇵🇩🇪", "❤️", "☺️", "1️⃣", "#️⃣", "🧑🏿‍🚀", "👩🏻‍❤️‍💋‍👨🏿", "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "✌🏼", "🤌", "🫠", "\U0001F3FB", "\U0001F1E6", "🧿", "⌚", "©️", "™"]
ws = [" ", "  ", "   ", "\t", "\n", "\n\n", "\r", "\r\n", "\r\n\r\n", "\v", "\f", "\u00a0", "\u2003", "\u2009", "\u3000", "\u0085", "\u2028", "\u2029",
      "\u1680", "\u202f", "\u205f", "\u200b", "\ufeff", "\u180e", " \n", "\n ", " \n ", "\t\n", "  \n\n  ", "\u00a0\u00a0", "\x1c", "\x1f", "\u200a"]
punct = ["!", "?", ".", ",", ";", ":", "...", "--", "->", "=>", "::", "//", "/*", "*/", "(", ")", "[]", "{}", "<>", "@#$%", "^&*", "~`", "\\", "\"", "'", "``",
         "«»", "…", "—", "–", "“”", "‘’", "¿¡", "§¶", "©®", "€£¥", "∑∫∂", "√∞", "≠≤≥", "→←↑", "★☆", "♠♣", "▀▄", "│┌", "、。", "「」", "؟،", "।", "·", "¦", "¨", "¬", "¯", "°±", "×÷"]
weird = ["\x00", "\x01", "\x07", "\x08", "\x1b", "\x7f", "\x80", "\x9f", "\u00ad", "\ue000", "\uf8ff", "\U000f0000", "\U0010ffff", "\ufffe", "\uffff", "\ufffd",
         "\u0378", "\u0379", "\U000e0001", "\U000e0020", "\U000e007f", "\U00030000", "\U0001fffe", "\ud7ff", "\u2065", "\u061c", "\u200e", "\u200f", "\u202a", "\u2066", "\U0002fa1d", "\U00031350", "\U0001e5d0", "\U00010d40", "\U0001bca0",
         "\u088f", "\U000323b0", "\U00011bc0", "\U0001e5f0", "\U00016d40", "\u0c5c", "\U00019718"]  # last row: assigned in Unicode 16/17 (HF's regex engine knows 16)
code = ["def f(x):\n    return x * 2\n", "for (int i = 0; i < n; ++i) { s += a[i]; }", "{\"a\": [1, 2, 3], \"b\": {\"c\": null}}", "<div class=\"x\">t</div>",
        "https://example.com/a/b?c=1&d=2#e", "user@example.com", "C:\\Users\\x\\f.txt", "x = 0xDEADBEEF + 1e-6", "SELECT * FROM t WHERE id = 42;", "#!/bin/sh\nset -e\n", "a_b_c camelCase SCREAMING_CASE",
        "```python\nprint(1)\n```", "$ ls -la | grep foo", "<?xml version=\"1.0\"?>", "if (a && b || !c) {}", "1.2.3-beta+4", "2026-11-10T12:00:00Z", "-----BEGIN-----", "\t\treturn 1;\n"]
special = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|endofprompt|>", "<|pad|>", "<|extra_id_0|>", "<|extra_id_7|>", "<|extra_id_10|>",
           "|||PHONE_NUMBER|||", "|||EMAIL_ADDRESS|||", "|||IP_ADDRESS|||",
           "<|im_start|", "<|im_start", "|im_start|>", "< |im_end|>", "<| im_end|>", "<|im_end |>", "<|im_end|>>", "<<|im_end|>", "<|IM_END|>", "<|Im_Start|>", "<|im_end|", "<|im_end", "im_end|>",
           "<|im_start|><|im_end|>", "<|im_start|im_end|>", "<|im_<|im_end|>start|>", "<|endoftext|><|endoftext|>", "<|im_start|>assistant\n", "<|im_start|>user\n", "<|im_start|>system\n",
           "<|im_end|>\n", "<|extra_id_11|>", "<|extra_id_|>", "<|extra_id_00|>", "|||PHONE_NUMBER||", "||PHONE_NUMBER|||", "|||phone_number|||", "|||EMAIL_ADDRESS||||", "<|fim_prefix|><|fim_suffix|>",
           "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>\n", "\n</think>\n", "<tools>", "<|", "|>", "<|>", "<||>"]

def rand_cp():
    r = rng.random()
    if r < 0.5: cp = rng.randrange(0x20, 0x3000)
    elif r < 0.8: cp = rng.randrange(0x3000, 0x20000)
    elif r < 0.95: cp = rng.randrange(0x20000, 0x110000)
    else: cp = rng.randrange(0, 0x110000)
    if 0xD800 <= cp < 0xE000: cp = 0xFFFD
    return chr(cp)

def seg():
    r = rng.random()
    if r < 0.18: return " ".join(rng.choice(words) for _ in range(rng.randint(1, 6)))
    if r < 0.26: return rng.choice(words_cased) + rng.choice(contr) + rng.choice(["", " ", " you", "t", "ve", "LL"])
    if r < 0.31: return rng.choice(contr)
    if r < 0.38: return "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 30)))
    if r < 0.42: return rng.choice(digits_other) + rng.choice(["", "1", " ", "a"])
    if r < 0.48: return rng.choice(scripts) + rng.choice(["", " ", "。", "!"])
    if r < 0.52: return rng.choice(["e", "a", "n", "x", "", "я", "ก", "क"]) + "".join(rng.choice(marks) for _ in range(rng.randint(1, 3)))
    if r < 0.58: return "".join(rng.choice(emoji) for _ in range(rng.randint(1, 4)))
    if r < 0.68: return "".join(rng.choice(ws) for _ in range(rng.randint(1, 4)))
    if r < 0.74: return "".join(rng.choice(punct) for _ in range(rng.randint(1, 4)))
    if r < 0.78: return "".join(rng.choice(weird) for _ in range(rng.randint(1, 3)))
    if r < 0.84: return rng.choice(code)
    if r < 0.92: return rng.choice(special)
    if r < 0.96: return "".join(rand_cp() for _ in range(rng.randint(1, 6)))
    return rng.choice(words).capitalize() + rng.choice(ws) + "".join(rng.choice("abcXYZ019_-") for _ in range(rng.randint(1, 5)))

strings = []
seen = set()
# a few directed families first (every special/whitespace/contraction pair)
for a in special[:14]:
    for b in ["", " ", "\n", "a", "1", "<", "|"]:
        for pre in ["", "x", " ", "\n"]:
            strings.append(pre + a + b)
for a in ws:
    for b in ws[:16]:
        strings.append("x" + a + b + "y")
        strings.append(a + b)
for c in contr:
    for p in ["", "a", " ", "1", "\n", "\u00e9"]:
        strings.append(p + c); strings.append(p + c + "x")
for a in weird:  # every odd code point in contexts where letter/number/punct class changes the pieces
    for b in ["'d", "123", "x", "1", "\n", " x"]:
        strings.append(a + b); strings.append("x" + a + b)
while len(strings) < N:
    strings.append("".join(seg() for _ in range(rng.randint(1, 12))))
rng.shuffle(strings)
with open(out, "w") as f:
    n = 0
    for s in strings:
        if s in seen: continue
        seen.add(s)
        f.write(s.encode("utf-8").hex() + "\n"); n += 1
print(n, "strings ->", out)

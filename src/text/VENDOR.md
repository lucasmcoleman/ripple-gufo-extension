# Vendored / adapted code

No upstream source file was copied or adapted line by line. Everything in `src/text/` was written for this
extension.

Looked at (read only) for interface shape and approach:

* gufo `src/families/qwen/text/{tokenizer.hpp,tokenizer.cpp,chat_template.hpp}` at upstream commit
  `0e2de43ec01540079fbec30492b06704796d725e` (`/server/programming/gufo-ripple/gufo`, MIT, "Copyright (c) 2026
  gufo contributors"). Kept close to it: `TokenId`/`kInvalidTokenId`, building the tokenizer from token and merge
  spans, `Encode`/`Decode`, ICU character properties (`u_charType` masks, `u_isUWhiteSpace`, `U8_NEXT`) for the
  hand-written pre-tokenizer, `Render(messages, tools, options)` returning an optional string with an error
  message. Not taken: its Qwen regex (letters-or-marks, single-digit numbers), its sequential special-token
  scan, its ChatML/Qwen templates.

Because nothing was copied, no upstream licence notice is carried in the sources. If a later change vendors
upstream code, add the file, the commit, the changes and the MIT notice here.

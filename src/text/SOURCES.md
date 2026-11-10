# Build inputs for src/text (for whoever edits extension.cmake)

Sources (all in `src/text/`, namespace `ricursive::text`):

    json_value.cpp      chat_template.cpp      tokenizer.cpp

Public headers (include relative to `src/text/`, or add `src/text` to the include path):

    json_value.hpp      order-preserving JsonValue, ParseJson (Python json.loads rules), DumpPythonJson
    chat_template.hpp   RenderChat(), ChatRenderOptions, kRippleChatTemplateSha256
    tokenizer.hpp       Tokenizer (Create/Encode/Decode/StreamDecoder), PreTokenizeLengths, UnicodeDataIsCurrent

* C++ standard: C++20 (std::span, string_view::starts_with); gufo's top-level CMake already sets C++20.
* Link: ICU common library only (`ICU::uc` / `-licuuc`; headers `unicode/uchar.h`, `unicode/utf8.h`).
  gufo core already does `find_package(ICU REQUIRED COMPONENTS uc i18n)`, so no new dependency.
  Needs ICU >= 76 (Unicode 16); the tests check this (`UnicodeDataIsCurrent()`). Host has ICU 78.2.
* No other libraries, no gufo headers, no exceptions escape the public API, no globals except a function-local
  static byte table.
* Verified flags: `/usr/bin/c++ -std=c++20 -O2 -Wall -Wextra` (g++ 15.2), zero warnings.
* Tests (not part of the extension build): `tests/text/Makefile` (`make -C tests/text test`), helpers in `tools/text/`.

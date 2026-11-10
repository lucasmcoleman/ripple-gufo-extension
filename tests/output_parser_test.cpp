// Output parser (src/output_parser.*): reasoning / content / tool-call split,
// the host-facing scanner, and invariance against where the stream is cut.
// Every group has a broken twin that must fail.
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "src/output_parser.hpp"
#include "tests/check.hpp"

using namespace ricursive_test;
using namespace gufo::models::ricursive;
namespace ext = gufo::ext;

namespace {

constexpr const char* kCallF =
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call>";
constexpr const char* kCallG =
    "<tool_call>\n{\"name\": \"g\", \"arguments\": {\"y\": \"two\"}}\n</tool_call>";
// Tags and brackets inside JSON strings, nested values, escapes, non-ASCII.
constexpr const char* kCallNested =
    "<tool_call>\n{\"name\": \"w\", \"arguments\": {\"s\": \"a</tool_call>b\", "
    "\"t\": \"</think>\", \"n\": {\"k\": [1, 2, {\"z\": \"}]\"}]}, "
    "\"u\": \"é\\u00e9 \\\"q\\\" \\\\\"}}\n</tool_call>";
constexpr const char* kNestedArgs =
    "{\"s\": \"a</tool_call>b\", \"t\": \"</think>\", \"n\": {\"k\": [1, 2, "
    "{\"z\": \"}]\"}]}, \"u\": \"éé \\\"q\\\" \\\\\"}";

struct Case {
  std::string name;
  std::string input;
  ParserEvents want;
};

ParserEvents Ev(std::string reasoning, std::string content,
                std::vector<ParsedToolCall> calls = {},
                std::string unterminated = {}, std::size_t malformed = 0) {
  ParserEvents e;
  e.reasoning = std::move(reasoning);
  e.content = std::move(content);
  e.calls = std::move(calls);
  e.unterminated = std::move(unterminated);
  e.malformed = malformed;
  return e;
}

std::vector<Case> Cases() {
  const ParsedToolCall f{"f", "{\"x\": 1}"};
  const ParsedToolCall g{"g", "{\"y\": \"two\"}"};
  const ParsedToolCall w{"w", kNestedArgs};
  std::vector<Case> c;
  c.push_back({"content only", "Hello, world.", Ev("", "Hello, world.")});
  c.push_back({"reasoning only", "<think>\nI think.\n</think>\n",
               Ev("I think.", "")});
  c.push_back({"reasoning and content", "<think>\nR\n</think>\nAnswer",
               Ev("R", "Answer")});
  c.push_back({"reasoning keeps inner and extra newlines",
               "<think>\na\n\nb\n\n</think>\nx", Ev("a\n\nb\n", "x")});
  c.push_back({"empty think block", "<think></think>answer",
               Ev("", "answer")});
  c.push_back({"think with only a newline", "<think>\n</think>\nanswer",
               Ev("", "answer")});
  c.push_back({"leading space: not reasoning", " <think>\nx\n</think>\ny",
               Ev("", " <think>\nx\n</think>\ny")});
  c.push_back({"think later in the text stays content", "Hi <think>\nx",
               Ev("", "Hi <think>\nx")});
  c.push_back({"one call", std::string(kCallF), Ev("", "", {f})});
  c.push_back({"text then one call",
               "Let me check.\n" + std::string(kCallF),
               Ev("", "Let me check.", {f})});
  c.push_back({"two newlines before a call keep one",
               "Hi\n\n" + std::string(kCallF), Ev("", "Hi\n", {f})});
  c.push_back({"several calls",
               "Two.\n" + std::string(kCallF) + "\n" + kCallG,
               Ev("", "Two.", {f, g})});
  c.push_back({"reasoning, empty text, a call",
               "<think>\nR\n</think>\n\n" + std::string(kCallF),
               Ev("R", "", {f})});
  c.push_back({"reasoning, text, calls",
               "<think>\nR\n</think>\nT\n" + std::string(kCallF) + "\n" + kCallG,
               Ev("R", "T", {f, g})});
  c.push_back({"nested JSON with tags inside strings",
               "<think>\nuse w\n</think>\n" + std::string(kCallNested),
               Ev("use w", "", {w})});
  c.push_back({"text after a call is kept",
               std::string(kCallF) + "\nTrailing words",
               Ev("", "\nTrailing words", {f})});
  c.push_back({"whitespace between and after calls is dropped",
               std::string(kCallF) + "\n  \n" + kCallG + "\n",
               Ev("", "", {f, g})});
  c.push_back({"call tag in reasoning stays reasoning",
               "<think>\nuse <tool_call> later\n</think>\nok",
               Ev("use <tool_call> later", "ok")});
  c.push_back({"unterminated call",
               "ok\n<tool_call>\n{\"name\": \"f\", \"argum",
               Ev("", "ok", {}, "<tool_call>\n{\"name\": \"f\", \"argum")});
  c.push_back({"unterminated call inside a JSON string",
               "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"s\": \"</tool_call>",
               Ev("", "", {},
                  "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"s\": "
                  "\"</tool_call>")});
  c.push_back({"unterminated reasoning", "<think>\npartial\n",
               Ev("partial\n", "")});
  c.push_back({"cut inside the think opener", "<thi", Ev("", "<thi")});
  c.push_back({"cut inside a call opener", "ab<tool_ca", Ev("", "ab<tool_ca")});
  c.push_back({"malformed call is released as content",
               "<tool_call>\nnot json\n</tool_call>",
               Ev("", "<tool_call>\nnot json\n</tool_call>", {}, "", 1)});
  c.push_back({"call without a name is malformed",
               "<tool_call>\n{\"arguments\": {}}\n</tool_call>",
               Ev("", "<tool_call>\n{\"arguments\": {}}\n</tool_call>", {}, "",
                  1)});
  c.push_back({"non-object arguments are a call",
               "<tool_call>\n{\"name\": \"f\", \"arguments\": 5}\n</tool_call>",
               Ev("", "", {{"f", "5"}})});
  c.push_back({"non-ASCII", "<think>\nhéllo ✓\n</think>\nwörld \U0001F600",
               Ev("héllo ✓", "wörld \U0001F600")});
  return c;
}

ParserEvents RunCuts(const std::string& input,
                     const std::vector<std::size_t>& cuts,
                     OutputParser::Faults faults = {}) {
  OutputParser parser(faults);
  ParserEvents all;
  std::size_t at = 0;
  for (const std::size_t cut : cuts) {
    all.Append(parser.Push(std::string_view(input).substr(at, cut - at)));
    at = cut;
  }
  all.Append(parser.Push(std::string_view(input).substr(at)));
  all.Append(parser.Finish());
  return all;
}

std::string Describe(const ParserEvents& e) {
  std::string s = "reasoning=[" + e.reasoning + "] content=[" + e.content +
                  "] calls=" + std::to_string(e.calls.size());
  for (const auto& c : e.calls) s += " {" + c.name + "|" + c.arguments_json + "}";
  s += " unterminated=[" + e.unterminated + "] malformed=" +
       std::to_string(e.malformed);
  return s;
}

/// Failures of one parser configuration over every case and every cut.
struct Tally {
  std::size_t runs{0};
  std::size_t wrong{0};  ///< runs that differ from the expected result
  std::string first;
};

Tally RunSuite(OutputParser::Faults faults, bool all_cuts) {
  Tally t;
  const auto note = [&](const Case& c, const char* how, const ParserEvents& got) {
    ++t.wrong;
    if (t.first.empty()) {
      t.first = c.name + " (" + how + "): got " + Describe(got) + "; want " +
                Describe(c.want);
    }
  };
  for (const Case& c : Cases()) {
    ++t.runs;
    if (auto got = RunCuts(c.input, {}, faults); !(got == c.want)) {
      note(c, "unsplit", got);
    }
    if (!all_cuts) continue;
    for (std::size_t i = 1; i < c.input.size(); ++i) {
      ++t.runs;
      if (auto got = RunCuts(c.input, {i}, faults); !(got == c.want)) {
        note(c, ("cut at " + std::to_string(i)).c_str(), got);
      }
    }
    std::vector<std::size_t> every;
    for (std::size_t i = 1; i < c.input.size(); ++i) every.push_back(i);
    ++t.runs;
    if (auto got = RunCuts(c.input, every, faults); !(got == c.want)) {
      note(c, "single bytes", got);
    }
  }
  return t;
}

// Random concatenations of fragments: whatever the result, cutting the stream
// must not change it.
std::size_t FuzzInvariance(OutputParser::Faults faults, std::size_t rounds) {
  static const char* kFragments[] = {
      "<think>\n", "\n</think>\n", "</think>", "<think>", "\n", " ", "text ",
      kCallF, kCallG, kCallNested, "<tool_call>", "</tool_call>", "{", "}",
      "\"", "<thi", "<tool_ca", "é", "\U0001F600", "\n<tool_call>\n{",
      "{\"name\": \"f\", \"arguments\": {}}", "\\"};
  Rng rng(20261110);
  std::size_t differing = 0;
  for (std::size_t round = 0; round < rounds; ++round) {
    std::string input;
    const std::uint32_t parts = 1 + rng.Below(7);
    for (std::uint32_t i = 0; i < parts; ++i) {
      input += kFragments[rng.Below(sizeof kFragments / sizeof *kFragments)];
    }
    const ParserEvents whole = RunCuts(input, {}, faults);
    std::vector<std::size_t> every;
    for (std::size_t i = 1; i < input.size(); ++i) {
      if (!(RunCuts(input, {i}, faults) == whole)) {
        ++differing;
        break;
      }
      every.push_back(i);
    }
    if (!(RunCuts(input, every, faults) == whole)) ++differing;
  }
  return differing;
}

// ---- host-facing scanner ----------------------------------------------------

std::vector<ext::ScannedToolCall> Scan(std::string_view text, bool partial,
                                       std::vector<std::pair<std::size_t, std::size_t>>* spans = nullptr) {
  std::vector<ext::ScannedToolCall> calls;
  const std::string_view markers[] = {kToolCallOpen};
  const auto first = text.find(kToolCallOpen);
  if (first == std::string_view::npos) return calls;
  ScanToolCalls(text.substr(first), markers, &calls, spans, partial);
  return calls;
}

void ScannerChecks() {
  std::vector<std::pair<std::size_t, std::size_t>> spans;
  const std::string text = std::string("x\n") + kCallF + "\n" + kCallNested + "\nend";
  auto calls = Scan(text, false, &spans);
  Require(calls.size() == 2, "scan: two calls");
  Require(calls[0].name == "f" && calls[0].complete && calls[0].valid &&
              calls[0].arguments.size() == 1 && calls[0].arguments[0].name == "x" &&
              calls[0].arguments[0].value == "1" && !calls[0].arguments[0].is_string,
          "scan: first call");
  Require(calls[1].name == "w" && calls[1].arguments.size() == 4 &&
              calls[1].arguments[0].value == "a</tool_call>b" &&
              calls[1].arguments[0].is_string &&
              calls[1].arguments[1].value == "</think>",
          "scan: nested call arguments");
  Require(spans.size() == 2, "scan: two spans");
  const std::string tail = text.substr(text.find(kToolCallOpen));
  Require(tail.substr(spans[0].first, spans[0].second - spans[0].first) == kCallF,
          "scan: span of the first call is exactly the block");
  // Non-object arguments are reported but not valid for the host.
  const auto five = Scan("<tool_call>\n{\"name\": \"f\", \"arguments\": 5}\n</tool_call>", false);
  Require(five.size() == 1 && !five[0].valid && five[0].complete, "scan: arguments 5");
  std::cout << "  scanner: 2 complete calls, spans exact, non-object arguments flagged\n";

  // Partial: for every prefix of a call, the closed arguments and the open
  // string only grow, and the final text agrees with the complete scan.
  const std::string full =
      "<tool_call>\n{\"name\": \"edit\", \"arguments\": {\"n\": 12, \"obj\": "
      "{\"a\": [1, 2]}, \"text\": \"h\\u00e9llo \\\"w\\\"\\n \\ud83d\\ude00 end\", "
      "\"flag\": true}}\n</tool_call>";
  std::vector<ext::ScannedToolArgument> previous;
  std::string previous_open;
  std::size_t saw_call = 0;
  std::size_t saw_open = 0;
  for (std::size_t k = 1; k <= full.size(); ++k) {
    const auto got = Scan(full.substr(0, k), true);
    if (got.empty()) continue;
    Require(got.size() == 1, "partial: at most the one unfinished call");
    const auto& call = got[0];
    if (k < full.size()) {
      Require(!call.complete, "partial: incomplete call flagged complete at " + std::to_string(k));
    }
    ++saw_call;
    Require(call.arguments.size() >= previous.size(), "partial: arguments shrank at " + std::to_string(k));
    for (std::size_t i = 0; i < previous.size(); ++i) {
      Require(call.arguments[i].name == previous[i].name &&
                  call.arguments[i].value == previous[i].value,
              "partial: a closed argument changed at " + std::to_string(k));
    }
    if (call.unfinished) {
      ++saw_open;
      // The open text only grows while the same argument is open.
      if (call.arguments.size() == previous.size() && !previous_open.empty()) {
        Require(call.unfinished->value.starts_with(previous_open),
                "partial: open string text shrank at " + std::to_string(k));
      }
      previous_open = call.unfinished->value;
    } else {
      previous_open.clear();
    }
    previous = call.arguments;
  }
  Require(saw_call > 20 && saw_open > 10, "partial: too few observations");
  const auto done = Scan(full, false);
  Require(done.size() == 1 && done[0].complete && done[0].arguments.size() == 4 &&
              done[0].arguments[2].value == "héllo \"w\"\n \U0001F600 end",
          "partial: complete scan decodes the string");
  std::cout << "  partial scan over " << full.size() << " prefixes: " << saw_call
            << " with a call, " << saw_open << " with an open string, all growth-only\n";
}

}  // namespace

int main() {
  return Run([] {
    // 1. Expected results, unsplit and with every cut.
    const Tally good = RunSuite({}, true);
    std::cout << "  cases: " << good.runs << " runs (unsplit, every two-way cut, single bytes)\n";
    Require(good.wrong == 0, "parser differs from expected: " + good.first);

    // 2. Random fragments: cutting must not change the aggregate.
    const std::size_t fuzz_diff = FuzzInvariance({}, 400);
    std::cout << "  fuzz: 400 random streams, " << fuzz_diff << " cut-dependent\n";
    Require(fuzz_diff == 0, "cut-dependent output on random streams");

    // 3. Reasoning emptiness guard: the reasoning cases really carry text.
    for (const Case& c : Cases()) {
      if (c.input.starts_with("<think>\n") && c.want.reasoning.empty() &&
          c.input.find("</think>") == std::string::npos) {
        Require(false, "case with unterminated empty reasoning: " + c.name);
      }
    }

    // 4. Twins.
    const Tally close_tag = RunSuite({.close_call_on_tag = true}, false);
    std::cout << "  twin close_call_on_tag: " << close_tag.wrong << " of "
              << close_tag.runs << " cases wrong (first: " << close_tag.first << ")\n";
    Require(close_tag.wrong >= 2, "twin close_call_on_tag must fail");
    const Tally leak = RunSuite({.no_think_holdback = true}, true);
    std::cout << "  twin no_think_holdback: " << leak.wrong << " of " << leak.runs
              << " runs wrong\n";
    Require(leak.wrong >= 10, "twin no_think_holdback must fail");
    const std::size_t leak_fuzz = FuzzInvariance({.no_think_holdback = true}, 400);
    std::cout << "  twin no_think_holdback fuzz: " << leak_fuzz << " cut-dependent\n";
    Require(leak_fuzz > 0, "twin no_think_holdback must be cut-dependent");
    // ReadCallBlock twin directly.
    const CallBlock right = ReadCallBlock(kCallNested);
    const CallBlock wrong = ReadCallBlock(kCallNested, true);
    Require(right.kind == CallBlock::Kind::kComplete && right.end == std::string(kCallNested).size(),
            "ReadCallBlock: nested call");
    Require(wrong.kind != CallBlock::Kind::kComplete || wrong.end < right.end,
            "ReadCallBlock twin must misread the nested call");

    ScannerChecks();
    PrintTwinSummary();
  });
}

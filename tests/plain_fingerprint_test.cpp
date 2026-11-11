// Plain generation fingerprint (docs/DRAFTER-S3.md): a hash of every logits
// row and every greedy token of a fixed plain schedule (prompt passes, 1-row
// decoding across the window and the ring wrap, a prefix-reuse sync), on the
// CPU executor and the host-ops executor, tinyA and tinyB. It uses only the
// engine API that existed before drafting (Sync, Evaluate, Logits), so the same
// file built on `drafter-s1` printed the same hashes: the golden values below
// were recorded there (commit 241e7bf). A change that alters plain generation,
// even by one bit, changes a hash and fails this test.
//
//   ricursive_plain_fingerprint_test          compare with the golden values
//   ricursive_plain_fingerprint_test print    print the current values
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>

#include "src/row_hash.hpp"
#include "tests/draft_util.hpp"

using namespace ricursive_test;

namespace {

constexpr std::uint32_t kContext = 96;

// key: "<backend>/<fixture>" -> hash, recorded on drafter-s1 (241e7bf).
const std::map<std::string, std::uint64_t> kGolden = {
    {"cpu/tinyA", 0x7d888f4b23f8c8b4ULL},
    {"cpu/tinyB", 0xda3faece90bdde5fULL},
    {"hostops/tinyA", 0x784ce93f47cb9cdeULL},
    {"hostops/tinyB", 0x91eda2a522732529ULL},
};

std::uint64_t Mix(std::uint64_t h, std::uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
  return h * 1099511628211ULL;
}

std::uint64_t Fingerprint(const TinyFixture& f, const DraftBackend& be) {
  const auto t = SyntheticTokens(f.window * 1000 + 7, 88);
  std::string error;
  std::uint64_t h = 14695981039346656037ULL;
  const auto note = [&](Session& s) {
    h = Mix(h, gufo::models::ricursive::RowHash(s.Logits()));
    h = Mix(h, static_cast<std::uint64_t>(Argmax(s.Logits())));
    h = Mix(h, s.Position());
  };
  const auto model = LoadDraftModel(f.path, be, kContext, f.window + 4, 4);
  {  // prompt passes, then 1-row decoding past the window and the ring wrap
    auto s = NewSession(model);
    Require(s->Sync(Slice(t, 0, 20), &error), error);
    note(*s);
    for (std::size_t i = 20; i < 80; ++i) {
      Require(s->Evaluate(t[i], &error), error);
      note(*s);
    }
    // prefix reuse: back inside the retained window, then a different tail
    auto target = Slice(t, 0, 70);
    for (std::size_t i = 0; i < 9; ++i) target.push_back((t[70 + i] + 3) % 256);
    Require(s->Sync(target, &error), error);
    note(*s);
    // back to a position the ring no longer holds: recompute from 0
    Require(s->Sync(Slice(t, 0, 12), &error), error);
    note(*s);
  }
  {  // one-row decoding from the very first token
    auto s = NewSession(model);
    Require(s->Sync(Slice(t, 0, 1), &error), error);
    note(*s);
    for (std::size_t i = 1; i < 50; ++i) {
      Require(s->Evaluate(t[i], &error), error);
      note(*s);
    }
  }
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  const bool print = argc > 1 && std::strcmp(argv[1], "print") == 0;
  return Run([&] {
    std::size_t checked = 0;
    for (const auto& be : DraftBackends()) {
      for (const auto& f : TinyFixtures()) {
        const std::string key = be.name + "/" + f.name;
        const std::uint64_t h = Fingerprint(f, be);
        if (print) {
          std::printf("    {\"%s\", 0x%016llxULL},\n", key.c_str(), static_cast<unsigned long long>(h));
          continue;
        }
        const auto it = kGolden.find(key);
        Require(it != kGolden.end(), key + ": no golden value recorded");
        Require(it->second == h, key + ": plain generation changed: fingerprint " + std::to_string(h) +
                                     " differs from the baseline " + std::to_string(it->second));
        std::cout << "  " << key << ": fingerprint " << std::hex << h << std::dec << " equals the drafter-s1 baseline\n";
        ++checked;
      }
    }
    if (!print) Require(checked == 4, "expected 4 fingerprints");
  });
}

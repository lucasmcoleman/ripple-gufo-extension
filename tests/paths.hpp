#ifndef RICURSIVE_TESTS_PATHS_HPP_
#define RICURSIVE_TESTS_PATHS_HPP_

#include <cstdlib>
#include <string>

namespace ricursive_test {

inline std::string EnvOr(const char* name, const char* fallback) {
  const char* v = std::getenv(name);
  return v != nullptr && *v != '\0' ? v : fallback;
}
/// Tiny GGUF fixtures (tinyA.gguf, tinyB*.gguf).
inline std::string FixtureDir() {
  return EnvOr("RICURSIVE_FIXTURES", "/server/staging/ripple-gufo/fixtures");
}
/// llama.cpp logits and token files (*.seqN.i32, *.seqN.llama.f32).
inline std::string OracleOutDir() {
  return EnvOr("RICURSIVE_ORACLE_OUT", "/server/staging/ripple-gufo/oracle/out");
}
/// Scratch directory for sparse stand-ins.
inline std::string S1Dir() {
  return EnvOr("RICURSIVE_S1_DIR", "/server/staging/ripple-gufo/s1-host");
}
inline std::string HeadPlain() {
  return EnvOr("RICURSIVE_HEAD_PLAIN",
               "/server/staging/ripple-gufo/ref/llamacpp/q4km-head-16MB.bin");
}
inline std::string HeadAd() {
  return EnvOr("RICURSIVE_HEAD_AD",
               "/server/staging/ripple-gufo/ref/gguf-heads/ad-q4km/head-16MB.bin");
}

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_PATHS_HPP_

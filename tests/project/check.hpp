// Minimal assertions for the standalone work-package-A tests (the Require/Run idea of the Gemma 4
// extension's tests/check.hpp, MIT, Copyright Francesco Moca; see hip/ggml/VENDOR.md).
#ifndef GUFO_TESTS_RICURSIVE_PROJECT_CHECK_HPP_
#define GUFO_TESTS_RICURSIVE_PROJECT_CHECK_HPP_

#include <iostream>
#include <stdexcept>
#include <string>

namespace ricursive_test {

/// Assertion that stays active in optimized builds.
inline void Require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

/// Runs `fn`; prints PASS and returns 0, or prints FAIL and returns 1.
template <class Fn>
int Run(Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}

}  // namespace ricursive_test

#endif  // GUFO_TESTS_RICURSIVE_PROJECT_CHECK_HPP_

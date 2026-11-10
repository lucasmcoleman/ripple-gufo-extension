#ifndef RICURSIVE_TESTS_CHECK_HPP_
#define RICURSIVE_TESTS_CHECK_HPP_

#include <iostream>
#include <stdexcept>
#include <string>

namespace ricursive_test {

/// Assertion that stays active in optimized builds.
inline void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

/// Requires `text` to contain `needle` (used for error-message checks).
inline void RequireContains(const std::string& text, const std::string& needle,
                            const std::string& what) {
  Require(text.find(needle) != std::string::npos,
          what + ": expected \"" + needle + "\" in \"" + text + "\"");
}

template<class Fn>
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

#endif  // RICURSIVE_TESTS_CHECK_HPP_

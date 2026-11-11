#ifndef RICURSIVE_TESTS_DRAFT_HOSTOPS_HPP_
#define RICURSIVE_TESTS_DRAFT_HOSTOPS_HPP_

#include <memory>

#include "src/executor_iface.hpp"
#include "src/weights.hpp"

namespace ricursive_test {

[[nodiscard]] std::unique_ptr<gufo::models::ricursive::Executor> MakeHostOpsExecutor(
    const gufo::models::ricursive::ModelWeights& weights,
    const gufo::models::ricursive::ExecutorConfig& config);

}  // namespace ricursive_test

#endif  // RICURSIVE_TESTS_DRAFT_HOSTOPS_HPP_

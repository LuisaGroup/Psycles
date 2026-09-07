#pragma once

#include <psycles/luisa/cycles_svm.h>

#include <stdexcept>
#include <string>
#include <unordered_set>

namespace psycles::test_support {

// Inspect every callable before backend optimization. The fixture deliberately
// records two different conservative bounds, so a hardcoded "small enough"
// array cannot satisfy this test. Runtime checks use the fixture's exact bound.
inline void require_svm_stack_extent(luisa::compute::Function function,
                                     std::size_t extent) {
  using namespace luisa::compute;
  const auto *full = Type::array(Type::of<float>(), SVM_STACK_SIZE);
  const auto *sized = extent > 4u ? Type::array(Type::of<float>(), extent) : nullptr;
  std::unordered_set<const luisa::compute::detail::FunctionBuilder *> visited;
  std::size_t full_count{}, sized_count{};
  const auto inspect = [&](auto &&self, Function f) -> void {
    if (!visited.emplace(f.builder()).second) { return; }
    for (const auto variable : f.local_variables()) {
      full_count += variable.type() == full;
      sized_count += variable.type() == sized;
    }
    for (const auto &callee : f.custom_callables()) { self(self, callee->function()); }
  };
  inspect(inspect, function);
  if ((extent != SVM_STACK_SIZE && full_count != 0u) ||
      (sized != nullptr && sized_count != 1u)) {
    throw std::runtime_error{"native SVM ignored scene stack extent " +
                             std::to_string(extent) + ": full arrays=" +
                             std::to_string(full_count) + ", sized arrays=" +
                             std::to_string(sized_count)};
  }
}
} // namespace psycles::test_support

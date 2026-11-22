#pragma once

#include <psycles/luisa/cycles_svm.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace psycles::test_support {

// Inspect every callable before backend optimization. The fixture deliberately
// records two different conservative bounds, so a hardcoded "small enough"
// array cannot satisfy this test. Runtime checks use the fixture's exact bound.
inline void require_svm_stack_extent(luisa::compute::Function function,
                                     std::size_t extent,
                                     bool allow_virtual_stack = false) {
  using namespace luisa::compute;
  const auto *full = Type::array(Type::of<float>(), SVM_STACK_SIZE);
  const auto *sized = extent > 4u ? Type::array(Type::of<float>(), extent) : nullptr;
  std::unordered_set<const luisa::compute::detail::FunctionBuilder *> visited;
  std::size_t full_count{}, sized_count{}, oversized_count{};
  std::string oversized_desc;
  const auto inspect = [&](auto &&self, Function f) -> void {
    if (!visited.emplace(f.builder()).second) { return; }
    for (const auto variable : f.local_variables()) {
      const auto *type = variable.type();
      full_count += type == full;
      sized_count += type == sized;
      if (type->is_array() && type->element() == Type::of<float>() &&
          type->dimension() > std::max<std::size_t>(4u, extent)) {
        ++oversized_count;
        if (!oversized_desc.empty()) { oversized_desc += ","; }
        oversized_desc += std::string{type->description()};
      }
    }
    for (const auto &callee : f.custom_callables()) { self(self, callee->function()); }
  };
  inspect(inspect, function);
  if ((extent != SVM_STACK_SIZE && full_count != 0u) ||
      oversized_count != 0u ||
      (!allow_virtual_stack && sized != nullptr && sized_count != 1u) ||
      (allow_virtual_stack && sized != nullptr && sized_count > 1u)) {
    throw std::runtime_error{"native SVM ignored scene stack extent " +
                             std::to_string(extent) + ": full arrays=" +
                             std::to_string(full_count) + ", sized arrays=" +
                             std::to_string(sized_count) +
                             ", oversized arrays=" +
                             std::to_string(oversized_count) +
                             " [" + oversized_desc + "]"};
  }
}
} // namespace psycles::test_support

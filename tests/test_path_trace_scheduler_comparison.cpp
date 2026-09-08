#include "path_trace_comparison_test_support.h"

#include <iostream>
#include <limits>

int main() {
  namespace schema = psycles::luisa_backend::path_trace_schema;
  using psycles::test::trace_component_matches;
  auto checks = std::size_t{};
  const auto check = [&](bool passed) {
    ++checks;
    if (!passed) {
      std::cerr << "scheduler trace comparison failed check " << checks << '\n';
    }
    return passed;
  };
  for (auto slot = std::size_t{}; slot < schema::slot_count; ++slot) {
    for (auto component = std::size_t{}; component < 4u; ++component) {
      const auto exact = (schema::scheduler_exact_component_masks[slot] >> component) & 1u;
      const auto value = component == 3u ? 1.0f : 0.25f;
      const auto next = std::nextafter(value, 2.0f);
      if (!check(trace_component_matches(slot, component, value, value, false)) ||
          !check(trace_component_matches(slot, component, value, next, false) == !exact) ||
          !check(!trace_component_matches(slot, component, value, next, true)) ||
          !check(!trace_component_matches(slot, component, value, value + 1.0e-3f, false))) {
        return 1;
      }
      for (const auto invalid : {std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::infinity()}) {
        if (!check(!trace_component_matches(slot, component, invalid, invalid, false)) ||
            !check(!trace_component_matches(slot, component, invalid, invalid, true))) {
          return 1;
        }
      }
    }
  }
  // Lossless capture from the original failing full fallback scheduler test.
  const auto expected = std::bit_cast<float>(0xbf1f8bfdu);
  const auto actual = std::bit_cast<float>(0xbf1f8a50u);
  const auto normal = schema::index(0u, schema::EventSlot::light_ng);
  const auto random = schema::index(0u, schema::EventSlot::random_light);
  const auto state = schema::index(0u, schema::EventSlot::state_depth);
  if (!check(trace_component_matches(normal, 2u, expected, actual, false)) ||
      !check(!trace_component_matches(normal, 2u, expected, actual, true)) ||
      !check(!trace_component_matches(random, 2u, expected, actual, false)) ||
      !check(!trace_component_matches(state, 2u, expected, actual, false)) ||
      !check(!trace_component_matches(normal, 3u, 1.0f, 0.0f, false)) ||
      !check(!trace_component_matches(normal, 3u, 0.5f, 0.5f, false)) ||
      !check(!trace_component_matches(schema::slot_count, 0u, 0.0f, 0.0f, false)) ||
      !check(!trace_component_matches(0u, 4u, 0.0f, 0.0f, false))) {
    return 1;
  }
  std::cout << checks << " scheduler trace comparison checks passed\n";
}

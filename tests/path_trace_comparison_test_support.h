#pragma once

#include <psycles/luisa/path_trace_scheduler_comparison.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace psycles::test {

// A cross-scheduler trace is an intermediate-state diagnostic, not a second
// film image. Legal fast-math normalization/intersection roundoff can be
// amplified (e.g. a small spherical emitter's Ng). Keep a separate 0.01%
// scale-relative / 1e-4 absolute continuous-state bound. Film's tighter bound
// and serial chunking's bit-exact gate are independent and unchanged.
inline constexpr auto scheduler_trace_tolerance = 1.0e-4f;

[[nodiscard]] inline bool trace_component_matches(
    std::size_t slot, std::size_t component, float expected, float actual,
    bool exact) noexcept {
  namespace schema = luisa_backend::path_trace_schema;
  if (slot >= schema::slot_count || component >= 4u ||
      !std::isfinite(expected) || !std::isfinite(actual)) {
    return false;
  }
  if (component == 3u &&
      ((expected != 0.0f && expected != 1.0f) ||
       (actual != 0.0f && actual != 1.0f))) {
    return false;
  }
  if (exact || ((schema::scheduler_exact_component_masks[slot] >> component) & 1u)) {
    return std::bit_cast<std::uint32_t>(expected) ==
           std::bit_cast<std::uint32_t>(actual);
  }
  return std::abs(expected - actual) <= scheduler_trace_tolerance *
             std::max({1.0f, std::abs(expected), std::abs(actual)});
}

} // namespace psycles::test

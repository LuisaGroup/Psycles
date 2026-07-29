#pragma once

#include <limits>

namespace psycles::luisa_backend::detail {

// Cycles scene synchronization converts the user-facing Filter Glossy value
// to a reciprocal device threshold. FLT_MAX is the exact disabled sentinel.
[[nodiscard]] constexpr float
cycles_filter_glossy_device_scale(float scene_filter_glossy) noexcept {
  return scene_filter_glossy > 0.0f ? 1.0f / scene_filter_glossy
                                    : std::numeric_limits<float>::max();
}

} // namespace psycles::luisa_backend::detail

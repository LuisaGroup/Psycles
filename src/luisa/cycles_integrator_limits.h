#pragma once

#include <cstdint>
#include <limits>

namespace psycles::luisa_backend::detail {

// Cycles' scene integrator sockets count global-illumination bounces after
// the first (direct-light) bounce. Device path state instead counts every
// non-transparent bounce, with zero denoting the camera segment. Scene sync
// therefore stores each opaque minimum/maximum as `socket + 1`.
//
// Saturation is only relevant to malformed external scene contracts: Blender
// constrains these settings to small values. This maps scene sockets only;
// path lifetime is owned by Cycles' independent device-state transitions.
[[nodiscard]] constexpr std::uint32_t
cycles_synced_bounce_limit(std::uint32_t scene_limit) noexcept {
  return scene_limit == std::numeric_limits<std::uint32_t>::max()
             ? scene_limit
             : scene_limit + 1u;
}

struct CyclesSceneBounceLimits {
  std::uint32_t maximum{};
  std::uint32_t minimum{};
  std::uint32_t maximum_diffuse{};
  std::uint32_t maximum_glossy{};
  std::uint32_t maximum_transmission{};
  std::uint32_t maximum_volume{};
  std::uint32_t transparent_minimum{};
  std::uint32_t transparent_maximum{};
};

struct CyclesKernelBounceLimits {
  std::uint32_t maximum{};
  std::uint32_t minimum{};
  std::uint32_t maximum_diffuse{};
  std::uint32_t maximum_glossy{};
  std::uint32_t maximum_transmission{};
  std::uint32_t maximum_volume{};
  std::uint32_t transparent_minimum{};
  std::uint32_t transparent_maximum{};
};

[[nodiscard]] constexpr CyclesKernelBounceLimits
cycles_kernel_bounce_limits(CyclesSceneBounceLimits scene) noexcept {
  const auto maximum = cycles_synced_bounce_limit(scene.maximum);
  return {.maximum = maximum,
          .minimum = cycles_synced_bounce_limit(scene.minimum),
          .maximum_diffuse = cycles_synced_bounce_limit(scene.maximum_diffuse),
          .maximum_glossy = cycles_synced_bounce_limit(scene.maximum_glossy),
          .maximum_transmission =
              cycles_synced_bounce_limit(scene.maximum_transmission),
          .maximum_volume =
              cycles_synced_bounce_limit(scene.maximum_volume),
          .transparent_minimum =
              cycles_synced_bounce_limit(scene.transparent_minimum),
          // Cycles deliberately does not add one to the transparent maximum.
          .transparent_maximum = scene.transparent_maximum};
}

} // namespace psycles::luisa_backend::detail

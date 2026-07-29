#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

namespace psycles::luisa_backend::detail {

// Cycles' scene integrator sockets count global-illumination bounces after
// the first (direct-light) bounce. Device path state instead counts every
// non-transparent bounce, with zero denoting the camera segment. Scene sync
// therefore stores each opaque minimum/maximum as `socket + 1`.
//
// Saturation is only relevant to malformed external scene contracts: Blender
// constrains these settings to small values, and Psycles bounds every device
// path to cycles_device_path_step_cap iterations.
[[nodiscard]] constexpr std::uint32_t
cycles_synced_bounce_limit(std::uint32_t scene_limit) noexcept {
  return scene_limit == std::numeric_limits<std::uint32_t>::max()
             ? scene_limit
             : scene_limit + 1u;
}

inline constexpr std::uint32_t cycles_device_path_step_cap = 1024u;

struct CyclesSceneBounceLimits {
  std::uint32_t maximum{};
  std::uint32_t minimum{};
  std::uint32_t maximum_diffuse{};
  std::uint32_t maximum_glossy{};
  std::uint32_t maximum_transmission{};
  std::uint32_t transparent_minimum{};
  std::uint32_t transparent_maximum{};
};

struct CyclesKernelBounceLimits {
  std::uint32_t maximum{};
  std::uint32_t minimum{};
  std::uint32_t maximum_diffuse{};
  std::uint32_t maximum_glossy{};
  std::uint32_t maximum_transmission{};
  std::uint32_t transparent_minimum{};
  std::uint32_t transparent_maximum{};
  std::uint32_t maximum_path_steps{};
};

// A path can scatter through at most `maximum` non-transparent surfaces and
// max(transparent_maximum, 1) transparent surfaces before Cycles marks it for
// termination. One final intersection is required to retain background or
// surface-emission contribution. The cap bounds malformed scene input and
// device work without changing any Blender-representable setting.
[[nodiscard]] constexpr std::uint32_t
cycles_path_step_limit(std::uint32_t synced_maximum,
                       std::uint32_t transparent_maximum) noexcept {
  const auto required =
      static_cast<std::uint64_t>(synced_maximum) +
      static_cast<std::uint64_t>(std::max(transparent_maximum, 1u)) + 1u;
  return static_cast<std::uint32_t>(
      std::min<std::uint64_t>(required, cycles_device_path_step_cap));
}

[[nodiscard]] constexpr CyclesKernelBounceLimits
cycles_kernel_bounce_limits(CyclesSceneBounceLimits scene) noexcept {
  const auto maximum = cycles_synced_bounce_limit(scene.maximum);
  return {.maximum = maximum,
          .minimum = cycles_synced_bounce_limit(scene.minimum),
          .maximum_diffuse = cycles_synced_bounce_limit(scene.maximum_diffuse),
          .maximum_glossy = cycles_synced_bounce_limit(scene.maximum_glossy),
          .maximum_transmission =
              cycles_synced_bounce_limit(scene.maximum_transmission),
          .transparent_minimum =
              cycles_synced_bounce_limit(scene.transparent_minimum),
          // Cycles deliberately does not add one to the transparent maximum.
          .transparent_maximum = scene.transparent_maximum,
          .maximum_path_steps =
              cycles_path_step_limit(maximum, scene.transparent_maximum)};
}

} // namespace psycles::luisa_backend::detail

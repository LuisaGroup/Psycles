#pragma once

#include "path_tracer_internal.h"

#include <span>
#include <string>

namespace psycles::luisa_backend::detail {

// Cycles partitions source lights before either the device table or the
// selection hierarchy is built. The roles are mutually exclusive:
//   regular: enabled direct-light emitter;
//   portal: environment proposal geometry, appended after regular lights;
//   background: accumulated environment contribution;
//   disabled: no host-proven contribution.
enum class AnalyticLightRole : std::uint8_t {
  disabled,
  regular,
  portal,
  background
};

[[nodiscard]] AnalyticLightRole
classify_analytic_light(const contract::LightDesc &light,
                        Vec3f shader_emission_estimate) noexcept;

struct AnalyticLightSceneUpload {
  // Cycles KernelLight ABI order: enabled regular lights, then portals.
  luisa::vector<LightGpu> device_lights;
  luisa::vector<Vec3f> regular_shader_emission_estimates;
  Vec3f background{};
  std::uint32_t regular_count{};
  std::uint32_t portal_count{};
  std::string diagnostic;

  [[nodiscard]] bool ok() const noexcept { return diagnostic.empty(); }

  [[nodiscard]] std::span<const LightGpu> regular_lights() const noexcept {
    return {device_lights.data(), regular_count};
  }

  [[nodiscard]] std::span<const LightGpu> portals() const noexcept {
    return {device_lights.data() + regular_count, portal_count};
  }
};

class AnalyticLightSceneComponent {

public:
  [[nodiscard]] AnalyticLightSceneUpload
  build(const contract::SceneSnapshot &snapshot,
        const LuisaSceneData &scene) const noexcept;
};

} // namespace psycles::luisa_backend::detail

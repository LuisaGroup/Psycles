#include "path_tracer_analytic_light_scene.h"

#include "cycles_shader_identity.h"

#include <psycles/compiler/surface_program.h>

#include <cmath>
#include <exception>
#include <limits>
#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] std::pair<Vec3f, float> normalized_axis(Vec3f axis) noexcept {
  const auto length =
      std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
  if (length <= 1.0e-20f) {
    return {Vec3f{}, 0.0f};
  }
  return {Vec3f{axis.x / length, axis.y / length, axis.z / length}, length};
}

[[nodiscard]] Vec3f shader_emission_estimate(const LuisaSceneData &scene,
                                             const contract::LightDesc &light) {
  if (!light.shader) {
    return {1.0f, 1.0f, 1.0f};
  }
  const auto *compiled = scene.materials.find(*light.shader);
  // A malformed reference must not silently remove a light. Scene
  // validation diagnoses it independently; Cycles also treats an unknown
  // linked output conservatively as contributing.
  return compiled == nullptr
             ? Vec3f{1.0f, 1.0f, 1.0f}
             : compiler::estimate_surface_emission(*compiled->surface_program(),
                                                   compiled->parameters());
}

[[nodiscard]] LightGpu make_device_light(const contract::LightDesc &light,
                                         const LuisaSceneData &scene) noexcept {
  const auto [axis_x, axis_x_length] =
      normalized_axis(matrix_axis(light.transform, 0u));
  const auto [axis_y, axis_y_length] =
      normalized_axis(matrix_axis(light.transform, 1u));
  auto [axis_z, axis_z_length] =
      normalized_axis(matrix_axis(light.transform, 2u));
  if (axis_z == Vec3f{}) {
    axis_z = {0.0f, 0.0f, 1.0f};
  }

  std::uint32_t flags = 0u;
  flags |= light.normalize ? light_flag_normalize : 0u;
  flags |= light.ellipse ? light_flag_ellipse : 0u;
  flags |= light.is_sphere ? light_flag_sphere : 0u;
  flags |=
      light.type == contract::LightType::area && light.spread >= pi - 1.0e-6f
          ? light_flag_full_spread
          : 0u;

  MaterialBinding binding{.surface_tag = ~std::uint32_t{0u},
                          .parameter_block = 0u,
                          .cycles_shader_index =
                              cycles_shader_identity::invalid_index};
  if (light.shader) {
    if (const auto iter = scene.material_bindings.find(*light.shader);
        iter != scene.material_bindings.end()) {
      binding = iter->second;
    }
  }

  const auto effective_mis =
      light.use_mis &&
      (light.type == contract::LightType::point ||
               light.type == contract::LightType::spot
           ? light.size > 0.0f
       : light.type == contract::LightType::area
           ? light.size * axis_x_length != 0.0f &&
                 (light.size_y > 0.0f ? light.size_y : light.size) *
                         axis_y_length !=
                     0.0f &&
                 light.spread > 0.0f
       : light.type == contract::LightType::distant ? light.angle > 0.0f
                                                    : true);
  const auto cycles_shader_id =
      light.cycles_shader_index
          ? cycles_shader_identity::analytic_light(
                *light.cycles_shader_index, light.cast_shadow,
                light.visibility_mask, light.is_shadow_catcher, effective_mis)
          : cycles_shader_identity::invalid_index;
  const auto cycles_shader_flags = cycles_shader_identity::analytic_light_flags(
      light.cast_shadow, light.visibility_mask, light.is_shadow_catcher,
      effective_mis);
  flags |= effective_mis ? light_flag_use_mis : 0u;
  flags |= effective_mis && (light.type == contract::LightType::area ||
                             light.type == contract::LightType::point ||
                             light.type == contract::LightType::spot)
               ? light_flag_forward_intersectable
               : 0u;
  flags |= (binding.surface_tag == ~std::uint32_t{0u} ||
            (binding.flags & material_flag_constant_emission) != 0u)
               ? light_flag_constant_emission
               : 0u;

  return LightGpu{.type = static_cast<std::uint32_t>(light.type),
                  .position = to_luisa(matrix_translation(light.transform)),
                  .axis_x = to_luisa(axis_x),
                  .axis_y = to_luisa(axis_y),
                  .axis_z = to_luisa(axis_z),
                  .axis_scale = luisa::make_float3(axis_x_length, axis_y_length,
                                                   axis_z_length),
                  .color = to_luisa(light.color),
                  .power = light.power,
                  .radius = light.size,
                  .size_u = light.size * axis_x_length,
                  .size_v = (light.size_y > 0.0f ? light.size_y : light.size) *
                            axis_y_length,
                  .spread = light.spread,
                  .spot_angle = light.spot_angle,
                  .spot_smooth = light.spot_smooth,
                  .angle = light.angle,
                  .flags = flags,
                  .surface_tag = binding.surface_tag,
                  .parameter_block = binding.parameter_block,
                  .cycles_object_index = light.cycles_object_index.value_or(
                      cycles_shader_identity::invalid_index),
                  .cycles_light_group = light.cycles_light_group,
                  .cycles_shader_id = cycles_shader_id,
                  .cycles_shader_flags = cycles_shader_flags,
                  .cycles_type = cycles_shader_identity::light_type(light.type),
                  .visibility_mask = light.visibility_mask,
                  .max_bounces = light.max_bounces};
}

} // namespace

AnalyticLightRole
classify_analytic_light(const contract::LightDesc &light,
                        Vec3f shader_emission_estimate) noexcept {
  if (light.type == contract::LightType::background) {
    return AnalyticLightRole::background;
  }
  // Portals are copied independently of Light::has_contribution(). Their
  // radiometric fields do not make them ordinary direct-light emitters.
  if (light.is_portal) {
    return AnalyticLightRole::portal;
  }
  const auto strength =
      Vec3f{light.color.x * light.power, light.color.y * light.power,
            light.color.z * light.power};
  if (strength == Vec3f{} || shader_emission_estimate == Vec3f{}) {
    return AnalyticLightRole::disabled;
  }
  if (light.type == contract::LightType::area) {
    const auto axis_x = matrix_axis(light.transform, 0u);
    const auto axis_y = matrix_axis(light.transform, 1u);
    const auto size_y = light.size_y > 0.0f ? light.size_y : light.size;
    if (light.size * size_y == 0.0f || axis_x == Vec3f{} || axis_y == Vec3f{}) {
      return AnalyticLightRole::disabled;
    }
  }
  return AnalyticLightRole::regular;
}

AnalyticLightSceneUpload
AnalyticLightSceneComponent::build(const contract::SceneSnapshot &snapshot,
                                   const LuisaSceneData &scene) const noexcept {
  AnalyticLightSceneUpload result;
  try {
    luisa::vector<LightGpu> regular;
    luisa::vector<LightGpu> portals;
    regular.reserve(snapshot.lights.size());
    portals.reserve(snapshot.lights.size());
    result.regular_shader_emission_estimates.reserve(snapshot.lights.size());

    for (const auto &[light_id, light] : snapshot.lights) {
      static_cast<void>(light_id);
      const auto estimate = shader_emission_estimate(scene, light);
      switch (classify_analytic_light(light, estimate)) {
      case AnalyticLightRole::disabled:
        break;
      case AnalyticLightRole::background:
        result.background.x += light.color.x * light.power;
        result.background.y += light.color.y * light.power;
        result.background.z += light.color.z * light.power;
        break;
      case AnalyticLightRole::portal:
        if (light.type != contract::LightType::area) {
          result.diagnostic = "Cycles portal light is not an area light";
          return result;
        }
        portals.emplace_back(make_device_light(light, scene));
        break;
      case AnalyticLightRole::regular:
        regular.emplace_back(make_device_light(light, scene));
        result.regular_shader_emission_estimates.emplace_back(estimate);
        break;
      }
    }

    constexpr auto maximum =
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
    if (regular.size() > maximum || portals.size() > maximum ||
        regular.size() + portals.size() > maximum) {
      result.diagnostic =
          "analytic light population exceeds the 32-bit device ABI";
      return result;
    }
    result.regular_count = static_cast<std::uint32_t>(regular.size());
    result.portal_count = static_cast<std::uint32_t>(portals.size());
    result.device_lights.reserve(regular.size() + portals.size());
    result.device_lights.insert(result.device_lights.end(), regular.begin(),
                                regular.end());
    result.device_lights.insert(result.device_lights.end(), portals.begin(),
                                portals.end());
  } catch (const std::exception &error) {
    result.diagnostic = error.what();
  } catch (...) {
    result.diagnostic = "unknown analytic-light scene construction error";
  }
  return result;
}

} // namespace psycles::luisa_backend::detail

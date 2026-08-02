#pragma once

#include <cstdint>

#include <psycles/contract/cycles_abi.h>
#include <psycles/contract/scene.h>

namespace psycles::luisa_backend::detail::cycles_shader_identity {

// Stable Cycles ShaderFlag ABI. The low bits identify Scene::shaders; the
// high bits are composed at the same semantic boundary as
// ShaderManager::get_shader_id and Light::copy_to_kernel.
inline constexpr auto smooth_normal =
    contract::cycles_abi::shader_smooth_normal;
inline constexpr auto cast_shadow =
    contract::cycles_abi::shader_cast_shadow;
inline constexpr auto use_mis =
    contract::cycles_abi::shader_use_mis;
inline constexpr auto exclude_diffuse =
    contract::cycles_abi::shader_exclude_diffuse;
inline constexpr auto exclude_glossy =
    contract::cycles_abi::shader_exclude_glossy;
inline constexpr auto exclude_transmit =
    contract::cycles_abi::shader_exclude_transmit;
inline constexpr auto exclude_camera =
    contract::cycles_abi::shader_exclude_camera;
inline constexpr auto exclude_scatter =
    contract::cycles_abi::shader_exclude_scatter;
inline constexpr auto exclude_shadow_catcher =
    contract::cycles_abi::shader_exclude_shadow_catcher;
inline constexpr auto exclude_any =
    contract::cycles_abi::shader_exclude_any;
inline constexpr auto shader_mask =
    contract::cycles_abi::shader_mask;
inline constexpr std::uint32_t invalid_index = ~std::uint32_t{0u};

// Cycles' KernelLightType order is a stable device ABI and differs from the
// renderer-neutral scene-contract order.
[[nodiscard]] constexpr std::uint32_t light_type(
    contract::LightType type) noexcept {
    using contract::LightType;
    switch (type) {
        case LightType::point:
            return 0u;
        case LightType::distant:
            return 1u;
        case LightType::background:
            return 2u;
        case LightType::area:
            return 3u;
        case LightType::spot:
            return 4u;
    }
    return invalid_index;
}

[[nodiscard]] constexpr std::uint32_t surface(
    std::uint32_t shader_index,
    bool smooth) noexcept {
    return shader_index |
           cast_shadow |
           (smooth ? smooth_normal : 0u);
}

[[nodiscard]] constexpr std::uint32_t object_visibility(
    std::uint32_t visibility,
    bool is_shadow_catcher) noexcept {
    using contract::RayVisibility;
    using contract::visibility_bit;
    std::uint32_t flags = 0u;
    flags |=
        (visibility & visibility_bit(RayVisibility::camera)) == 0u
            ? exclude_camera
            : 0u;
    flags |=
        (visibility & visibility_bit(RayVisibility::diffuse)) == 0u
            ? exclude_diffuse
            : 0u;
    flags |=
        (visibility & visibility_bit(RayVisibility::glossy)) == 0u
            ? exclude_glossy
            : 0u;
    flags |=
        (visibility & visibility_bit(RayVisibility::transmission)) == 0u
            ? exclude_transmit
            : 0u;
    flags |=
        (visibility &
         visibility_bit(RayVisibility::volume_scatter)) == 0u
            ? exclude_scatter
            : 0u;
    flags |=
        is_shadow_catcher
            ? 0u
            : exclude_shadow_catcher;
    return flags;
}

[[nodiscard]] constexpr std::uint32_t analytic_light_flags(
    bool casts_shadow,
    std::uint32_t visibility,
    bool is_shadow_catcher,
    bool has_competing_bsdf_technique) noexcept {
    return (casts_shadow ? cast_shadow : 0u) |
           object_visibility(
               visibility, is_shadow_catcher) |
           (has_competing_bsdf_technique ? use_mis : 0u);
}

[[nodiscard]] constexpr std::uint32_t analytic_light(
    std::uint32_t shader_index,
    bool casts_shadow,
    std::uint32_t visibility,
    bool is_shadow_catcher,
    bool has_competing_bsdf_technique) noexcept {
    return shader_index |
           analytic_light_flags(casts_shadow,
               visibility,
               is_shadow_catcher,
               has_competing_bsdf_technique);
}

[[nodiscard]] constexpr std::uint32_t emissive_triangle_flags(
    bool smooth,
    std::uint32_t visibility,
    bool is_shadow_catcher) noexcept {
    // Triangle lights retain the surface topology flags, then add the
    // sampled-emitter MIS and object-visibility flags at light upload time.
    return cast_shadow |
           (smooth ? smooth_normal : 0u) |
           use_mis |
           object_visibility(
               visibility, is_shadow_catcher);
}

[[nodiscard]] constexpr std::uint32_t emissive_triangle(
    std::uint32_t shader_index,
    bool smooth,
    std::uint32_t visibility,
    bool is_shadow_catcher) noexcept {
    return shader_index |
           emissive_triangle_flags(
               smooth, visibility, is_shadow_catcher);
}

[[nodiscard]] constexpr std::uint32_t background_light_flags(
    bool casts_shadow,
    std::uint32_t visibility) noexcept {
    using contract::RayVisibility;
    using contract::visibility_bit;
    // BackgroundLight::copy_to_kernel applies world visibility only to the
    // four scattering categories. Its synthetic Object is camera-visible and
    // is a shadow catcher by construction.
    const auto background_visibility =
        visibility |
        visibility_bit(RayVisibility::camera);
    return (casts_shadow ? cast_shadow : 0u) |
           use_mis |
           object_visibility(
               background_visibility,
               true);
}

[[nodiscard]] constexpr std::uint32_t background_light(
    std::uint32_t shader_index,
    bool casts_shadow,
    std::uint32_t visibility) noexcept {
    return shader_index |
           background_light_flags(casts_shadow, visibility);
}

}// namespace psycles::luisa_backend::detail::cycles_shader_identity

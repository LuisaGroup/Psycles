#pragma once

#include <cstdint>

#include <psycles/contract/cycles_abi.h>
#include <psycles/contract/scene.h>

namespace psycles::luisa_backend::detail::cycles_shader_identity {

// Stable Cycles ShaderFlag ABI. The low bits identify Scene::shaders; the
// high bits are composed at the same semantic boundary as
// ShaderManager::get_shader_id and Light::copy_to_kernel.
inline constexpr std::uint32_t smooth_normal = 1u << 31u;
inline constexpr std::uint32_t cast_shadow = 1u << 30u;
inline constexpr std::uint32_t use_mis = 1u << 28u;
inline constexpr std::uint32_t exclude_diffuse = 1u << 27u;
inline constexpr std::uint32_t exclude_glossy = 1u << 26u;
inline constexpr std::uint32_t exclude_transmit = 1u << 25u;
inline constexpr std::uint32_t exclude_camera = 1u << 24u;
inline constexpr std::uint32_t exclude_scatter = 1u << 23u;
inline constexpr std::uint32_t exclude_shadow_catcher = 1u << 22u;
inline constexpr std::uint32_t exclude_any =
    exclude_diffuse |
    exclude_glossy |
    exclude_transmit |
    exclude_camera |
    exclude_scatter |
    exclude_shadow_catcher;
inline constexpr std::uint32_t shader_mask =
    ~(smooth_normal | cast_shadow | use_mis | exclude_any);
static_assert(
    shader_mask ==
    contract::cycles_abi::shader_mask);
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

[[nodiscard]] constexpr std::uint32_t analytic_light(
    std::uint32_t shader_index,
    bool casts_shadow,
    std::uint32_t visibility,
    bool is_shadow_catcher,
    bool has_competing_bsdf_technique) noexcept {
    return shader_index |
           (casts_shadow ? cast_shadow : 0u) |
           object_visibility(
               visibility, is_shadow_catcher) |
           (has_competing_bsdf_technique ? use_mis : 0u);
}

}// namespace psycles::luisa_backend::detail::cycles_shader_identity

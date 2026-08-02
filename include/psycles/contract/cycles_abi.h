#pragma once

#include <cstdint>

namespace psycles::contract::cycles_abi {

// Current Cycles ShaderFlag ABI. Low bits identify Scene::shaders; the high
// bits are per-ray/per-hit flags and must not participate in shader lookup.
inline constexpr std::uint32_t shader_smooth_normal = 1u << 31u;
inline constexpr std::uint32_t shader_cast_shadow = 1u << 30u;
inline constexpr std::uint32_t shader_use_mis = 1u << 28u;
inline constexpr std::uint32_t shader_exclude_diffuse = 1u << 27u;
inline constexpr std::uint32_t shader_exclude_glossy = 1u << 26u;
inline constexpr std::uint32_t shader_exclude_transmit = 1u << 25u;
inline constexpr std::uint32_t shader_exclude_camera = 1u << 24u;
inline constexpr std::uint32_t shader_exclude_scatter = 1u << 23u;
inline constexpr std::uint32_t shader_exclude_shadow_catcher = 1u << 22u;
inline constexpr std::uint32_t shader_exclude_any =
    shader_exclude_diffuse |
    shader_exclude_glossy |
    shader_exclude_transmit |
    shader_exclude_camera |
    shader_exclude_scatter |
    shader_exclude_shadow_catcher;
inline constexpr std::uint32_t shader_mask =
    ~(shader_smooth_normal |
      shader_cast_shadow |
      shader_use_mis |
      shader_exclude_any);
static_assert(shader_mask == 0x203fffffu);

}// namespace psycles::contract::cycles_abi

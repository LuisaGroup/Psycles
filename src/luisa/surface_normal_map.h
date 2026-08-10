#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] Float3 normal_map_tangent_displaced_inline(
    const SurfaceNormalMapInput &input) noexcept;
[[nodiscard]] Float3 normal_map_tangent_original_inline(
    const SurfaceNormalMapInput &input) noexcept;
[[nodiscard]] Float3 normal_map_object_inline(
    const SurfaceNormalMapInput &input) noexcept;
[[nodiscard]] Float3 normal_map_world_inline(
    const SurfaceNormalMapInput &input) noexcept;

[[nodiscard]] Float3 normal_map_tangent_displaced(
    const ShaderServices &services,
    const SurfaceNormalMapInput &input) noexcept;
[[nodiscard]] Float3 normal_map_tangent_original(
    const ShaderServices &services,
    const SurfaceNormalMapInput &input) noexcept;
[[nodiscard]] Float3 normal_map_object(
    const ShaderServices &services,
    const SurfaceNormalMapInput &input) noexcept;
[[nodiscard]] Float3 normal_map_world(
    const ShaderServices &services,
    const SurfaceNormalMapInput &input) noexcept;

}// namespace psycles::luisa_backend::detail

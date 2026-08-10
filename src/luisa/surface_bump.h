#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] Float3 bump_world_inline(
    const SurfaceBumpInput &input) noexcept;
[[nodiscard]] Float3 bump_object_inline(
    const SurfaceBumpInput &input) noexcept;

[[nodiscard]] Float3 bump_world(
    const ShaderServices &services,
    const SurfaceBumpInput &input) noexcept;
[[nodiscard]] Float3 bump_object(
    const ShaderServices &services,
    const SurfaceBumpInput &input) noexcept;

}// namespace psycles::luisa_backend::detail

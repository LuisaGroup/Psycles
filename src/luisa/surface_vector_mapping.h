#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] Float3 rotate_euler(
    Float3 value, Float3 rotation) noexcept;
[[nodiscard]] Float3 rotate_euler_transposed(
    Float3 value, Float3 rotation) noexcept;
[[nodiscard]] Float3 safe_divide_components(
    Float3 numerator, Float3 denominator) noexcept;

[[nodiscard]] Float3 map_vector_point_inline(
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) noexcept;
[[nodiscard]] Float3 map_vector_texture_inline(
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) noexcept;
[[nodiscard]] Float3 map_vector_direction_inline(
    Float3 input,
    Float3 rotation,
    Float3 scale) noexcept;
[[nodiscard]] Float3 map_vector_normal_inline(
    Float3 input,
    Float3 rotation,
    Float3 scale) noexcept;

[[nodiscard]] Float3 map_vector_point(
    const ShaderServices &services,
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) noexcept;
[[nodiscard]] Float3 map_vector_texture(
    const ShaderServices &services,
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) noexcept;
[[nodiscard]] Float3 map_vector_direction(
    const ShaderServices &services,
    Float3 input,
    Float3 rotation,
    Float3 scale) noexcept;
[[nodiscard]] Float3 map_vector_normal(
    const ShaderServices &services,
    Float3 input,
    Float3 rotation,
    Float3 scale) noexcept;

}// namespace psycles::luisa_backend::detail

#pragma once

#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] luisa::compute::Float3 rotate_euler(
    luisa::compute::Float3 value, luisa::compute::Float3 rotation) noexcept;
[[nodiscard]] luisa::compute::Float3 rotate_euler_transposed(
    luisa::compute::Float3 value, luisa::compute::Float3 rotation) noexcept;
[[nodiscard]] luisa::compute::Float3 safe_divide_components(
    luisa::compute::Float3 numerator,
    luisa::compute::Float3 denominator) noexcept;

[[nodiscard]] luisa::compute::Float3 map_vector_point_inline(
    luisa::compute::Float3 input,
    luisa::compute::Float3 location,
    luisa::compute::Float3 rotation,
    luisa::compute::Float3 scale) noexcept;
[[nodiscard]] luisa::compute::Float3 map_vector_texture_inline(
    luisa::compute::Float3 input,
    luisa::compute::Float3 location,
    luisa::compute::Float3 rotation,
    luisa::compute::Float3 scale) noexcept;
[[nodiscard]] luisa::compute::Float3 map_vector_direction_inline(
    luisa::compute::Float3 input,
    luisa::compute::Float3 rotation,
    luisa::compute::Float3 scale) noexcept;
[[nodiscard]] luisa::compute::Float3 map_vector_normal_inline(
    luisa::compute::Float3 input,
    luisa::compute::Float3 rotation,
    luisa::compute::Float3 scale) noexcept;

} // namespace psycles::luisa_backend::detail

#pragma once

#include <cstdint>
#include <span>

#include <psycles/compiler/surface_execution_plan.h>
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

[[nodiscard]] Float3
evaluate_surface_mapping_svm(const ShaderServices &services, UInt immediate,
                             std::span<const std::uint16_t> immediate_domain,
                             Float3 input, Float3 location, Float3 rotation,
                             Float3 scale) noexcept;

}// namespace psycles::luisa_backend::detail

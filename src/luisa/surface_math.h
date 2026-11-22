#pragma once

#include <cstdint>

#include <luisa/dsl/syntax.h>

namespace psycles::compiler {

// Cycles ShaderNodeMath operations selected while tracing the Luisa AST.
enum class MathOperation : std::uint8_t {
  add,
  subtract,
  multiply,
  divide,
  multiply_add,
  power,
  logarithm,
  square_root,
  inverse_square_root,
  absolute,
  exponent,
  minimum,
  maximum,
  less_than,
  greater_than,
  sign,
  compare,
  smooth_minimum,
  smooth_maximum,
  round,
  floor,
  ceil,
  trunc,
  fraction,
  modulo,
  floored_modulo,
  wrap,
  snap,
  ping_pong,
  sine,
  cosine,
  tangent,
  arcsine,
  arccosine,
  arctangent,
  arctangent2,
  hyperbolic_sine,
  hyperbolic_cosine,
  hyperbolic_tangent,
  radians,
  degrees
};

inline constexpr std::uint32_t math_operation_count =
    static_cast<std::uint32_t>(MathOperation::degrees) + 1u;

} // namespace psycles::compiler

namespace psycles::luisa_backend::detail {

[[nodiscard]] luisa::compute::Float3 safe_normalize(
    luisa::compute::Float3 value,
    luisa::compute::Float3 fallback) noexcept;

// Cycles' safe_powf is total at a zero base: exponent zero returns one and
// every other exponent, including negative values, returns zero. Keeping this
// shared prevents scalar Math and component-wise VectorMath from drifting.
[[nodiscard]] luisa::compute::Float cycles_safe_power(
    luisa::compute::Float base,
    luisa::compute::Float exponent) noexcept;

[[nodiscard]] luisa::compute::Float evaluate_surface_math_operation(
    compiler::MathOperation operation,
    luisa::compute::Float a,
    luisa::compute::Float b,
    luisa::compute::Float c) noexcept;

} // namespace psycles::luisa_backend::detail

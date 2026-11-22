#pragma once

#include <cstdint>

#include <luisa/dsl/syntax.h>

namespace psycles::compiler {

enum class VectorMathOperation : std::uint8_t {
  add,
  subtract,
  multiply,
  divide,
  multiply_add,
  cross_product,
  project,
  reflect,
  refract,
  faceforward,
  dot_product,
  distance,
  length,
  scale,
  normalize,
  absolute,
  power,
  sign,
  minimum,
  maximum,
  floor,
  ceil,
  fraction,
  modulo,
  wrap,
  snap,
  sine,
  cosine,
  tangent,
  round,
  // Internal Cycles NormalNode primitive. Unlike the public Vector Math
  // NORMALIZE operation, Cycles deliberately performs a / len(a) without a
  // zero guard here; non-finite results are sanitized only at film commit.
  cycles_normalize
};

inline constexpr std::uint32_t vector_math_operation_count =
    static_cast<std::uint32_t>(VectorMathOperation::cycles_normalize) + 1u;

} // namespace psycles::compiler

namespace psycles::luisa_backend::detail {

struct SurfaceVectorMathResult {
    luisa::compute::Float value;
    luisa::compute::Float3 vector;
};

[[nodiscard]] SurfaceVectorMathResult evaluate_surface_vector_math_operation(
    compiler::VectorMathOperation operation,
    luisa::compute::Float3 a,
    luisa::compute::Float3 b,
    luisa::compute::Float3 c,
    luisa::compute::Float scale) noexcept;

} // namespace psycles::luisa_backend::detail

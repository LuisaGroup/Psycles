#pragma once

#include <cstdint>
#include <span>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Cycles' Vector Math node has one semantic evaluation with independently live
// scalar and vector outputs. Keep that typed result at the host/JIT component
// boundary even though the current compact bytecode has one result address per
// instruction. This makes a future multi-result record an ABI refinement, not
// a second formula implementation.
struct SurfaceVectorMathResult {
    Float value;
    Float3 vector;
};

[[nodiscard]] SurfaceVectorMathResult evaluate_surface_vector_math_operation(
    compiler::VectorMathOperation operation,
    Float3 a,
    Float3 b,
    Float3 c,
    Float scale) noexcept;

[[nodiscard]] Float evaluate_surface_vector_math_value_svm(
    UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    Float3 a,
    Float3 b,
    Float3 c,
    Float scale) noexcept;

[[nodiscard]] Float3 evaluate_surface_vector_math_vector_svm(
    UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    Float3 a,
    Float3 b,
    Float3 c,
    Float scale) noexcept;

} // namespace psycles::luisa_backend::detail

#pragma once

#include "path_tracer_surface_value_program.h"

namespace psycles::luisa_backend::detail {

// Transitional capability boundary for the Cycles-aligned direct SVM
// evaluator. A supported family reads its packed operands and writes the
// typed stack directly; it never constructs TracedValues, a
// SurfaceValueExpression, or a host ValueNode. Keeping this predicate total
// makes partial replacement explicit and testable while remaining a pure
// host/JIT decision.
[[nodiscard]] constexpr bool surface_value_family_has_direct_evaluator(
    compiler::SurfaceSvmValueOpcode opcode) noexcept {
    return opcode == compiler::SurfaceSvmValueOpcode::convert ||
           opcode == compiler::SurfaceSvmValueOpcode::math;
}

// Emits one statically selected family subtype. Returns false exactly when
// the family has not yet been migrated to the direct typed-stack evaluator.
// A true return means the result has already been written to instruction.y.
[[nodiscard]] bool emit_direct_surface_value_variant(
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot operand_slot, const ShaderServices &services,
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept;

}// namespace psycles::luisa_backend::detail

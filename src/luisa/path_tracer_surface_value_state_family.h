#pragma once

#include "path_tracer_surface_value_family.h"

namespace psycles::luisa_backend::detail {

// Direct typed-stack implementations of Cycles' ShaderData-reading SVM
// families. Derivative records receive explicit (dx, dy) operands produced by
// bump expansion; the evaluator itself has no hidden mutable sampling state.
void emit_direct_surface_state_family(
    compiler::SurfaceSvmValueOpcode family,
    const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept;

} // namespace psycles::luisa_backend::detail

#pragma once

#include "path_tracer_surface_value_family.h"

namespace psycles::luisa_backend::detail {

// Direct typed-stack implementations of Cycles' scalar Math, Vector Math and
// Clamp SVM families. The family subtype and operation immediate remain
// bytecode data; this function only records the exact cases reachable in the
// current scene.
void emit_direct_surface_numeric_family(
    compiler::SurfaceSvmValueOpcode family,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept;

} // namespace psycles::luisa_backend::detail

#pragma once

#include "path_tracer_surface_value_family.h"

namespace psycles::luisa_backend::detail {

// Direct typed-stack implementations of Cycles' procedural SVM families.
// Immutable finite shape domains are selected while recording the Luisa AST;
// authored per-record semantics remain ordinary device instruction data.
void emit_direct_surface_procedural_family(
    compiler::SurfaceSvmValueOpcode family,
    const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept;

} // namespace psycles::luisa_backend::detail

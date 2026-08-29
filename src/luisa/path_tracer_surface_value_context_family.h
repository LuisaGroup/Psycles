#pragma once

#include "path_tracer_surface_value_family.h"

namespace psycles::luisa_backend::detail {

// Direct typed-stack projection of Cycles' ShaderData/path context families.
// Every emitted operation is a pure function of the explicit SurfacePoint and
// typed operands; no ValueNode or weakly typed intermediate is constructed.
void emit_direct_surface_context_family(
    compiler::SurfaceSvmValueOpcode family, const SurfacePoint &point,
    const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept;

} // namespace psycles::luisa_backend::detail

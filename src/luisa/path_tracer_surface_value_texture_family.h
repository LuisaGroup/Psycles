#pragma once

#include "path_tracer_surface_value_family.h"

namespace psycles::luisa_backend::detail {

void emit_direct_surface_texture_family(
    compiler::SurfaceSvmValueOpcode family, const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots, const ShaderServices &services,
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept;

} // namespace psycles::luisa_backend::detail

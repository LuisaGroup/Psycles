#pragma once

#include "path_tracer_shader_services.h"
#include "path_tracer_surface_value_program.h"

#include "graph_surface_internal.h"

#include <cstdint>

namespace psycles::luisa_backend::detail {

// Decode one already-selected closure leaf. The static variant and exact
// Principled mask are host/JIT identities; authored operands remain device
// addresses in the unified stream. The single-PC interpreter is the sole
// caller, so operand provenance is explicit rather than ambient state.
[[nodiscard]] TracedClosure decode_surface_closure(
    std::uint32_t static_variant,
    compiler::PrincipledClosureFeatureMask principled_features,
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot closure_operand_slot,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    Float mix_weight) noexcept;

} // namespace psycles::luisa_backend::detail

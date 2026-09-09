#pragma once

#include "path_kernel_emissive_triangle.h"

namespace psycles::luisa_backend::detail {

// Host DSL recording boundary for Cycles integrate_surface_emission. It is
// consumed directly by the surface stage, not an outlined device callable.
void emit_surface_emission(
    SurfaceGeometryContext &surface, const SurfacePreparation &preparation,
    const EmissiveTriangleComponent &emissive_triangle) noexcept;

} // namespace psycles::luisa_backend::detail

#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Cycles primitive_tangent projection for the Geometry node. Curves and
// objectless ShaderData use the world-space parametric derivative; mesh
// surfaces construct the Generated-coordinate radial tangent and transform it
// as an object-space normal.
[[nodiscard]] Float3
surface_geometry_tangent(const SurfacePoint &point) noexcept;

} // namespace psycles::luisa_backend::detail

#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

// Cycles 5.2 standalone Metallic population. The graph opcode is the static
// Fresnel model tag; this component consumes authored values and produces the
// one canonical general-family physical closure used by eval/sample.
class MetallicClosureComponent final {

private:
    const ShaderServices &_services;
    const SurfaceClosurePoint &_point;

public:
    MetallicClosureComponent(
        const ShaderServices &services,
        const SurfaceClosurePoint &point) noexcept;

    [[nodiscard]] TracedClosure setup(
        const TracedClosure &graph_closure,
        Bool reflective_caustics) const noexcept;
};

} // namespace psycles::luisa_backend::detail

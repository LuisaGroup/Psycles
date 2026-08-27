#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

// Cycles legacy Hair setup is a typed projection from graph inputs to one
// physical Hair record. Tangent linkage is immutable shader topology; curve
// membership remains a device predicate. Keeping the two binding times
// separate prevents either a runtime component branch or a numerical
// "zero means unlinked" convention from entering scattering.
class HairClosureComponent {

private:
    const SurfacePoint &_point;
    SurfaceClosurePoint _closure_point;

public:
    explicit HairClosureComponent(
        const SurfacePoint &point) noexcept
        : _point{point}, _closure_point{point} {}

    [[nodiscard]] TracedClosure setup(
        const TracedClosure &graph_closure) const noexcept;
};

}// namespace psycles::luisa_backend::detail

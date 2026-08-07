#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

struct ThinSubsurfaceResult {
    TracedClosure reflection;
    TracedClosure smooth_transmission;
    TracedClosure rough_transmission;
};

// Expands Cycles' OpenPBR thin-subsurface approximation. The authored Thin
// Wall value and every material socket remain device expressions; the host
// component only records the fixed reflection/transmission closure topology.
class ThinSubsurfaceComponent final {

  public:
    [[nodiscard]] ThinSubsurfaceResult setup(
        const TracedClosure &prototype,
        Float3 weight,
        Bool enabled) const noexcept;
};

}// namespace psycles::luisa_backend::detail

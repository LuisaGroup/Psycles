#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

struct BssrdfSetupResult {
    TracedClosure bssrdf;
    TracedClosure diffuse_fallback;
};

// Cycles-compatible BSSRDF allocation and setup. This host-stage component
// records one algebra for both standalone and Principled nodes; all socket
// values remain Luisa expressions and no material value is evaluated or
// baked on the host.
class BssrdfClosureComponent final {

private:
    const SurfacePoint &_point;

public:
    explicit BssrdfClosureComponent(
        const SurfacePoint &point) noexcept;

    [[nodiscard]] BssrdfSetupResult
    setup(const TracedClosure &prototype) const noexcept;
};

}// namespace psycles::luisa_backend::detail

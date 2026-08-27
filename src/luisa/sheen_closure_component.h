#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

// Canonical result of Cycles 5.2's Sheen LTC setup. This relation is shared
// by Principled's ordered sheen layer and the standalone Microfiber closure;
// keeping one implementation prevents their table coordinates, validity
// boundary, or albedo rescaling from drifting apart.
struct SheenLtcState {
    Float roughness;
    Float transform_a;
    Float transform_b;
    Float albedo;
    Bool valid;
};

[[nodiscard]] SheenLtcState evaluate_sheen_ltc(
    const ShaderServices &services,
    Float3 normal,
    Float3 incoming,
    Float roughness) noexcept;

// Cycles 5.2 standalone Sheen population. The graph opcode is the static
// distribution tag. Microfiber occupies the existing general payload for its
// two LTC coefficients; Ashikhmin is common-only and stores its setup-time
// inverse sigma squared in the otherwise model-specific roughness scalar.
class SheenClosureComponent final {

private:
    const ShaderServices &_services;
    SurfaceClosurePoint _point;

public:
    SheenClosureComponent(
        const ShaderServices &services,
        const SurfacePoint &point) noexcept;

    [[nodiscard]] TracedClosure setup(
        const TracedClosure &graph_closure) const noexcept;
};

} // namespace psycles::luisa_backend::detail

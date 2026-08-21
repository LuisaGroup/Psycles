#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

struct ThinGlassSetupResult {
    TracedClosure reflection;
    TracedClosure transmission;
    TracedClosure transparency;
};

// Cycles' OpenPBR thin dielectric is a pair of constant-Fresnel GGX lobes,
// not an ordinary refractive Glass closure. This component owns that complete
// physical contract while retaining Thin Wall and all tints as device values.
class ThinGlassComponent final {

private:
    const ShaderServices &_services;
    SurfaceClosurePoint _point;

public:
    ThinGlassComponent(const ShaderServices &services,
                       const SurfaceClosurePoint &point) noexcept;

    [[nodiscard]] ThinGlassSetupResult setup(
        const TracedClosure &prototype,
        Float3 weight,
        Float3 normal,
        Float roughness,
        Float ior,
        Float3 reflection_tint,
        Float3 transmission_tint,
        Bool enabled,
        Bool reflective_caustics,
        Bool refractive_caustics) const noexcept;

    [[nodiscard]] MicrofacetEvaluation evaluate(
        const SurfaceClosurePhysicalRecord &closure,
        Float3 incoming,
        Float3 outgoing,
        Float glossy_filter_roughness) const noexcept;

    [[nodiscard]] MicrofacetReflectionSample sample(
        const SurfaceClosurePhysicalRecord &closure,
        Float3 incoming,
        Float2 random_direction,
        Float glossy_filter_roughness) const noexcept;
};

}// namespace psycles::luisa_backend::detail

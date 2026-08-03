#pragma once

#include "graph_surface_internal.h"
#include "microfacet_glass_component.h"

namespace psycles::luisa_backend::detail {

struct PrincipledBaseResult {
    TracedClosure metallic;
    TracedClosure transmission;
    TracedClosure dielectric;
    Float3 diffuse_weight;
};

// Records Cycles' ordered Principled base closure family. The component is a
// host-stage abstraction only: authored sockets, closure allocation, Fresnel
// tables, and layer attenuation remain Luisa device expressions.
class PrincipledBaseComponent final {

private:
    const ShaderServices &_services;
    const SurfacePoint &_point;
    MicrofacetGlassComponent _glass;

public:
    PrincipledBaseComponent(const ShaderServices &services,
                            const SurfacePoint &point) noexcept;

    [[nodiscard]] PrincipledBaseResult
    evaluate(const TracedClosure &closure, Bool reflective_caustics,
             Bool refractive_caustics) const noexcept;
};

}// namespace psycles::luisa_backend::detail

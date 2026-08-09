#pragma once

#include "graph_surface_internal.h"
#include "microfacet_glass_component.h"
#include "thin_glass_component.h"

#include <optional>

namespace psycles::luisa_backend::detail {

struct PrincipledBaseResult {
    std::optional<TracedClosure> metallic;
    std::optional<TracedClosure> transmission;
    std::optional<TracedClosure> thin_glass_reflection;
    std::optional<TracedClosure> thin_glass_transmission;
    std::optional<TracedClosure> thin_glass_transparency;
    std::optional<TracedClosure> dielectric;
    Float3 base_weight;
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
    ThinGlassComponent _thin_glass;

public:
    PrincipledBaseComponent(const ShaderServices &services,
                            const SurfacePoint &point) noexcept;

    [[nodiscard]] PrincipledBaseResult
    evaluate(const TracedClosure &closure,
             compiler::PrincipledClosureFeatureMask features,
             Bool reflective_caustics,
             Bool refractive_caustics) const noexcept;
};

}// namespace psycles::luisa_backend::detail

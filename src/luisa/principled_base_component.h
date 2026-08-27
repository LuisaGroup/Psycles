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
};

struct PrincipledDielectricSetupParameters {
    Float3 lower_weight;
    Float3 glossy_normal;
    Float incoming_cosine;
    Float roughness;
    Float ior;
    Float specular_ior_level;
    Float3 specular_tint;
    Float thin_film_thickness;
    Float thin_film_ior;
    bool thin_film_enabled{};
    bool preserve_ggx_energy{};
};

// Populate the physical dielectric inputs from authored socket expressions
// and shading-point state. Path-tracer callables keep this stage shared across
// material topologies; GraphSurface can reuse values already populated for
// the sibling Principled lobes.
[[nodiscard]] PrincipledDielectricSetupParameters
populate_principled_dielectric(
    const PrincipledDielectricSetupInput &input) noexcept;

// Canonical inline implementation behind both the standalone GraphSurface
// path and the path tracer's shared typed callable. Keeping one transfer
// function makes the two staging strategies semantically identical.
[[nodiscard]] PrincipledDielectricSetupResult
setup_principled_dielectric(
    const ShaderServices &services,
    const PrincipledDielectricSetupParameters &parameters,
    Bool reflective_caustics) noexcept;

// Records Cycles' ordered Principled base closure family. The component is a
// host-stage abstraction only: authored sockets, closure allocation, Fresnel
// tables, and layer attenuation remain Luisa device expressions.
class PrincipledBaseComponent final {

private:
    const ShaderServices &_services;
    SurfaceClosurePoint _point;
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

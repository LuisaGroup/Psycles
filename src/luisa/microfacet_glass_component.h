#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

// Host-stage input for one Cycles microfacet dielectric closure. Every field
// containing a Luisa DSL value remains device-side when setup() records the
// shader AST; the static fields select generalized-Schlick Glass or pure
// Refraction and the concrete distribution.
struct MicrofacetGlassSetup {
    TracedClosure prototype;
    Float3 weight;
    Float3 normal;
    Float roughness;
    Float ior;
    Float3 fresnel_f0;
    Float3 fresnel_f90;
    Float3 reflection_tint;
    Float3 transmission_tint;
    Bool enabled;
    PrincipledLobe principled_lobe{PrincipledLobe::none};
    bool refraction_only{};
    bool preserve_energy{};
    bool beckmann{};
};

// Builds and consumes physical microfacet Glass and Refraction closures while
// Luisa records kernels. Keeping setup, lobe selection/TIR, evaluation,
// density, and sampling behind a single component prevents those contracts
// from drifting independently.
class MicrofacetGlassComponent final {

private:
    const ShaderServices &_services;
    SurfaceClosurePoint _point;

public:
    MicrofacetGlassComponent(const ShaderServices &services,
                             const SurfaceClosurePoint &point) noexcept;

    [[nodiscard]] TracedClosure
    setup(const MicrofacetGlassSetup &parameters) const noexcept;

    [[nodiscard]] MicrofacetEvaluation evaluate(
        const SurfaceClosurePhysicalRecord &closure,
        Float3 incoming,
        Float3 outgoing,
        Float3 glossy_normal,
        Bool reflection_allowed,
        Bool transmission_allowed,
        Float glossy_filter_roughness) const noexcept;

    [[nodiscard]] GlassSample
    sample(const SurfaceClosurePhysicalRecord &closure,
        Float3 incoming,
        Float3 glossy_normal,
        Float2 random_direction,
        Float random_lobe,
        Bool reflection_allowed,
        Bool transmission_allowed,
        Float glossy_filter_roughness) const noexcept;
};

}// namespace psycles::luisa_backend::detail

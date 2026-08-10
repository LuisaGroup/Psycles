#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

struct PrincipledAlphaLayerResult {
    Float3 lower_weight;
    TransparentClosureState transparency;
};

[[nodiscard]] PrincipledAlphaLayerResult evaluate_principled_alpha_layer(
    const TracedClosure &closure) noexcept;

// Device-expression results of Cycles' ordered Principled layers. The
// host-stage component below records the corresponding Luisa AST while every
// socket, lookup, allocation predicate, and layer reduction stays device-side.
struct PrincipledEmissionLayerResult {
    Float3 lower_weight;
    Float3 radiance;
};

struct PrincipledSheenLayerResult {
    TracedClosure closure;
    Float3 lower_weight;
};

struct PrincipledCoatLayerResult {
    TracedClosure closure;
    Float3 lower_weight;
};

class PrincipledLayerComponent final {

private:
    const ShaderServices &_services;
    SurfaceClosurePoint _point;

public:
    PrincipledLayerComponent(
        const ShaderServices &services,
        const SurfacePoint &point) noexcept;

    [[nodiscard]] PrincipledSheenLayerResult evaluate_sheen(
        const TracedClosure &closure,
        Float3 lower_weight) const noexcept;

    [[nodiscard]] PrincipledCoatLayerResult evaluate_coat(
        const TracedClosure &closure,
        Float3 lower_weight,
        Bool reflective_caustics) const noexcept;

    [[nodiscard]] PrincipledEmissionLayerResult evaluate_emission(
        const TracedClosure &closure,
        compiler::PrincipledClosureFeatureMask features,
        Bool reflective_caustics) const noexcept;
};

}// namespace psycles::luisa_backend::detail

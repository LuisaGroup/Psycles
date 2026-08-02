#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

// Device-expression result of Cycles' ordered Principled layers. This is a
// host-stage component: ordinary C++ composition records the corresponding
// Luisa AST, while every socket and lookup remains a device-side value.
struct PrincipledEmissionLayerResult {
    Float3 lower_weight;
    Float3 radiance;
};

class PrincipledEmissionLayerComponent final {

private:
    const ShaderServices &_services;
    const SurfacePoint &_point;
    Bool _reflective_caustics;

public:
    PrincipledEmissionLayerComponent(
        const ShaderServices &services,
        const SurfacePoint &point,
        Bool reflective_caustics) noexcept;

    [[nodiscard]] PrincipledEmissionLayerResult evaluate(
        const TracedClosure &closure) const noexcept;
};

}// namespace psycles::luisa_backend::detail

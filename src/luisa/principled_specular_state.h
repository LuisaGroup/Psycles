#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

struct PrincipledSpecularStateInput {
    Float3 normal;
    Float3 incoming;
    Float3 surface_shading_normal;
    Float3 surface_geometric_normal;
    Float roughness;
    Float3 specular_tint;
    Bool use_bump_map_correction;
};

struct PrincipledSpecularState {
    Float3 glossy_normal;
    Float incoming_cosine;
    Float roughness;
    Float3 specular_tint;
};

// Canonical populate relation shared by Principled metallic and dielectric.
// It records expressions only; no authored value crosses to the host.
[[nodiscard]] PrincipledSpecularState
populate_principled_specular_state(
    const PrincipledSpecularStateInput &input) noexcept;

}// namespace psycles::luisa_backend::detail

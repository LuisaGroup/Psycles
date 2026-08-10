#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

struct PrincipledMetallicSetupParameters {
    Float3 lower_weight;
    Float3 glossy_normal;
    Float3 base_color;
    Float3 specular_tint;
    Float incoming_cosine;
    Float roughness;
    Float metallic;
    bool preserve_ggx_energy{};
};

[[nodiscard]] PrincipledMetallicSetupParameters
populate_principled_metallic(
    const PrincipledMetallicSetupInput &input) noexcept;

[[nodiscard]] PrincipledMetallicSetupResult
setup_principled_metallic(
    const ShaderServices &services,
    const PrincipledMetallicSetupParameters &parameters,
    Bool reflective_caustics) noexcept;

class PrincipledMetallicComponent final {

private:
    const ShaderServices &_services;

public:
    PrincipledMetallicComponent(
        const ShaderServices &services) noexcept;

    [[nodiscard]] PrincipledMetallicSetupResult
    setup(const PrincipledMetallicSetupInput &input,
          const PrincipledMetallicSetupParameters &direct_parameters,
          Bool reflective_caustics) const noexcept;
};

}// namespace psycles::luisa_backend::detail

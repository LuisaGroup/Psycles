#pragma once

#include <psycles/luisa/surface_closure_physical_blocks.h>

namespace psycles::luisa_backend::detail {

struct HairClosureEvaluation {
    Float intensity;
    Float pdf;
};

struct HairClosureSample {
    Float3 direction;
    Float2 roughness;
    Bool valid;
};

[[nodiscard]] HairClosureEvaluation evaluate_hair_reflection(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    Bool sampled_direction) noexcept;

[[nodiscard]] HairClosureEvaluation evaluate_hair_transmission(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    Bool sampled_direction) noexcept;

[[nodiscard]] HairClosureSample sample_hair_reflection(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float2 random) noexcept;

[[nodiscard]] HairClosureSample sample_hair_transmission(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float2 random) noexcept;

}// namespace psycles::luisa_backend::detail

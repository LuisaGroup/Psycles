#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/triangle_light_sampling.h> through the Psycles::luisa target."
#endif

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

struct TriangleLightSampleInput {
    luisa::compute::Float3 reference;
    luisa::compute::Float3 p0;
    luisa::compute::Float3 p1;
    luisa::compute::Float3 p2;
    luisa::compute::Float2 random;
};

struct TriangleLightSample {
    luisa::compute::Float3 position;
    // Luisa intersection convention:
    // p = (1 - u - v) p0 + u p1 + v p2.
    luisa::compute::Float2 barycentric;
    luisa::compute::Float3 direction;
    luisa::compute::Float distance;
    luisa::compute::Float conditional_pdf;
    luisa::compute::Bool uses_solid_angle;
    luisa::compute::Bool valid;
};

struct TriangleLightPdf {
    luisa::compute::Float value;
    luisa::compute::Bool uses_solid_angle;
    luisa::compute::Bool valid;
};

// Host-stage Luisa AST component for Cycles' triangle-light measures.
//
// Cycles intentionally uses two distinct sampling measures. A volume-segment
// proposal always samples triangle area so its sampled point can define the
// equiangular reference before the collision is known. Surface NEE and the
// volume collision-position resample switch to solid-angle sampling for a
// nearby triangle. Forward-hit MIS evaluates that same position-dependent
// measure without resampling.
class TriangleLightSampling {

  public:
    [[nodiscard]] TriangleLightSample from_segment(
        const TriangleLightSampleInput &input) const noexcept;

    [[nodiscard]] TriangleLightSample from_position(
        const TriangleLightSampleInput &input) const noexcept;

    [[nodiscard]] TriangleLightPdf from_intersection(
        const TriangleLightSampleInput &input,
        luisa::compute::Float3 light_position) const noexcept;
};

}// namespace psycles::luisa_backend

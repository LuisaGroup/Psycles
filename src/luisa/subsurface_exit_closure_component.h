#pragma once

#include <psycles/luisa/surface.h>

namespace psycles::luisa_backend::detail {

// Cycles replaces the material closures at a BSSRDF exit with one unit
// Lambert closure. Keeping that synthetic closure behind an integrator
// component avoids mutating or pre-baking the original material graph.
class SubsurfaceExitClosureComponent final {

public:
    [[nodiscard]] UInt runtime_flags(
        const SurfacePoint &point) const noexcept;

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query,
        Expr<std::uint32_t> light_shader_flags) const noexcept;

    [[nodiscard]] SurfaceSample sample(
        const SurfacePoint &point,
        Expr<luisa::float2> random,
        const SurfaceQuery &query) const noexcept;

    [[nodiscard]] SurfaceClosureTrace trace(
        const SurfacePoint &point,
        Expr<std::uint32_t> requested_index) const noexcept;

    [[nodiscard]] SurfaceSampleTrace sample_trace(
        const SurfacePoint &point,
        Expr<luisa::float2> random,
        const SurfaceQuery &query) const noexcept;
};

}// namespace psycles::luisa_backend::detail

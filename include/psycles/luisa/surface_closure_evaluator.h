#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_evaluator.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface_closure_set.h>

namespace psycles::luisa_backend {

// Shared device-stage consumer of the post-shader Cycles closure array. This
// object is constructed while Luisa records a callable, after runtime material
// dispatch has populated SurfaceClosureSet. Its methods therefore appear once
// in the shader AST instead of once per material implementation.
class SurfaceClosureEvaluator {

  private:
    const SurfacePoint &_point;
    const SurfaceClosureSet &_closures;
    Float3 _shading_normal;

  public:
    SurfaceClosureEvaluator(
        const SurfacePoint &point,
        const SurfaceClosureSet &closures,
        Float3 shading_normal) noexcept;

    [[nodiscard]] UInt runtime_flags(
        Float glossy_filter_roughness = 0.0f) const noexcept;

    [[nodiscard]] SurfaceClosureTrace closure_trace(
        UInt requested_index) const noexcept;

    [[nodiscard]] SurfaceAov aov() const noexcept;
};

}// namespace psycles::luisa_backend

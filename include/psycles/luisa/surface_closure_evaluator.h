#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_evaluator.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface_closure_reachability.h>
#include <psycles/luisa/surface_closure_population.h>
#include <psycles/luisa/surface_closure_set.h>

#include <optional>

namespace psycles::luisa_backend {

// Shared device-stage consumer of the post-shader Cycles closure array. This
// object is constructed while Luisa records a callable, after runtime material
// dispatch has populated SurfaceClosureSet. Its methods therefore appear once
// in the shader AST instead of once per material implementation.
class SurfaceClosureEvaluator {

  private:
    enum class EvaluationMode : std::uint8_t {
        regular,
        sampled_light,
        sampled_bsdf,
    };

    const SurfacePoint &_point;
    const SurfaceClosureSet &_closures;
    Float3 _shading_normal;
    SurfaceClosureReachability _reachability;
    std::optional<SurfaceClosurePopulationState>
        _populated_runtime_state;

    [[nodiscard]] SurfaceEvaluation evaluate_impl(
        const ShaderServices &services,
        Float3 outgoing,
        const SurfaceQuery &query,
        EvaluationMode mode,
        UInt light_shader_flags,
        UInt selected_closure_index) const noexcept;

    [[nodiscard]] SurfaceSampleTrace sample_impl(
        const ShaderServices &services,
        Float u_lobe,
        Float2 u_direction,
        const SurfaceQuery &query,
        bool trace_selection) const noexcept;

  public:
    SurfaceClosureEvaluator(
        const SurfacePoint &point,
        const SurfaceClosureSet &closures,
        Float3 shading_normal,
        SurfaceClosureReachability reachability =
            all_surface_closure_reachability) noexcept;

    // Production constructor: runtime flags were folded over the exact same
    // retained closure sequence while it was populated. Only the owning
    // collector can construct this provenance token.
    SurfaceClosureEvaluator(
        const SurfacePoint &point,
        const SurfaceClosureSet &closures,
        Float3 shading_normal,
        SurfaceClosurePopulationState populated_runtime_state,
        SurfaceClosureReachability reachability =
            all_surface_closure_reachability) noexcept;

    [[nodiscard]] UInt runtime_flags(
        Float glossy_filter_roughness = 0.0f) const noexcept;

    [[nodiscard]] SurfaceClosureTrace closure_trace(
        UInt requested_index) const noexcept;

    [[nodiscard]] SurfaceAov aov() const noexcept;

    [[nodiscard]] SurfaceEvaluation evaluate(
        const ShaderServices &services,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept;

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        const ShaderServices &services,
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept;

    [[nodiscard]] SurfaceSample sample(
        const ShaderServices &services,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept;

    [[nodiscard]] SurfaceSampleTrace sample_trace(
        const ShaderServices &services,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept;
};

}// namespace psycles::luisa_backend

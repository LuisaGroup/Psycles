#pragma once

#include "path_tracer_internal.h"

#include <psycles/luisa/surface_closure_evaluation.h>

namespace psycles::luisa_backend::detail {

// Compact, immutable state shared by every per-closure callable invocation
// of one surface evaluation. The SurfacePoint remains a separate callable
// parameter so existing pack/unpack helpers preserve its exact ABI.
struct SurfaceClosureEvaluationQueryCall {
    luisa::float3 incoming{};
    luisa::float3 outgoing{};
    luisa::uint lobe_mask{};
    luisa::uint transport_mode{};
    float glossy_filter_roughness{};
    luisa::uint reflective_caustics{};
    luisa::uint refractive_caustics{};
    luisa::uint diffuse_included{};
    luisa::uint glossy_included{};
    luisa::uint glass_included{};
};

}// namespace psycles::luisa_backend::detail

LUISA_STRUCT(
    psycles::luisa_backend::detail::
        SurfaceClosureEvaluationQueryCall,
    incoming,
    outgoing,
    lobe_mask,
    transport_mode,
    glossy_filter_roughness,
    reflective_caustics,
    refractive_caustics,
    diffuse_included,
    glossy_included,
    glass_included) {};

namespace psycles::luisa_backend::detail {

using SurfaceClosureEvaluationCallable =
    Callable<SurfaceClosureEvaluationContributionCall(
        Buffer<luisa::float4>,
        Buffer<float>,
        BindlessArray,
        BindlessArray,
        SurfacePointCall,
        SurfaceClosureEvaluationQueryCall,
        luisa::float3,
        bool,
        luisa::float4x4,
        luisa::float4x4,
        luisa::float4x4,
        luisa::float4x4)>;

[[nodiscard]] SurfaceClosureEvaluationCallable
make_surface_closure_evaluation_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

// Concrete host/JIT-stage component used by the path tracer. Virtual
// dispatch enters this object while recording each material branch; the
// generated device AST contains only calls to _callable and scalar folds.
class CallableSurfaceClosureEvaluationOperation final
    : public SurfaceClosureEvaluationOperation {

  private:
    const BufferFloat4 &_parameters;
    const BufferFloat &_cycles_bsdf_tables;
    const BindlessVar &_textures;
    const BindlessVar &_geometry_heap;
    const Var<SurfacePointCall> &_point;
    Var<SurfaceClosureEvaluationQueryCall> _query;
    const SurfaceClosureEvaluationCallable &_callable;

  public:
    CallableSurfaceClosureEvaluationOperation(
        const BufferFloat4 &parameters,
        const BufferFloat &cycles_bsdf_tables,
        const BindlessVar &textures,
        const BindlessVar &geometry_heap,
        const Var<SurfacePointCall> &packed_point,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query,
        const SurfaceClosureEvaluationPolicy &policy,
        const SurfaceClosureEvaluationCallable &callable) noexcept;

    [[nodiscard]] luisa::compute::Var<
        SurfaceClosureEvaluationContributionCall>
    evaluate(
        Expr<luisa::float3> shading_normal,
        const SurfaceClosureExpression &closure,
        Expr<bool> selected_sample) const noexcept override;
};

}// namespace psycles::luisa_backend::detail

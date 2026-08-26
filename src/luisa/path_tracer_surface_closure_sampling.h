#pragma once

#include "path_tracer_surface_closure_evaluation.h"

#include <psycles/luisa/surface_closure_sampling.h>

namespace psycles::luisa_backend::detail {

struct SurfaceClosureSamplingQueryCall {
    luisa::float3 incoming{};
    luisa::uint lobe_mask{};
    luisa::uint transport_mode{};
    float glossy_filter_roughness{};
    luisa::uint reflective_caustics{};
    luisa::uint refractive_caustics{};
};

}// namespace psycles::luisa_backend::detail

LUISA_STRUCT(psycles::luisa_backend::detail::SurfaceClosureSamplingQueryCall,
             incoming, lobe_mask, transport_mode, glossy_filter_roughness,
             reflective_caustics, refractive_caustics) {};

namespace psycles::luisa_backend::detail {

using SurfaceClosureSelectionCallable = Callable<SurfaceClosureSelectionCall(
    luisa::uint, float, luisa::uint, luisa::uint, luisa::uint, float, float,
    bool, luisa::float3, float, bool, bool)>;

using SurfaceClosureConditionalSampleCallable =
    Callable<SurfaceClosureConditionalSampleCall(
        Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
        BindlessArray, SurfaceClosurePointCall, SurfaceClosureSamplingQueryCall,
        luisa::float3, luisa::float3, luisa::float2, float, luisa::float4x4,
        luisa::float4x4)>;

struct SurfaceClosureSamplingCallables {
    SurfaceClosureSelectionCallable selection;
    SurfaceClosureConditionalSampleCallable conditional_sample;
};

[[nodiscard]] SurfaceClosureSamplingCallables
make_surface_closure_sampling_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

class CallableSurfaceClosureSamplingOperation final
    : public SurfaceClosureSamplingOperation {

private:
    const BufferFloat &_scalar_parameters;
    const BufferFloat3 &_vector_parameters;
    const BufferFloat &_cycles_bsdf_tables;
    const BindlessVar &_textures;
    const BindlessVar &_geometry_heap;
    const Var<SurfaceClosurePointCall> &_point;
    SurfaceClosureSelectionContext _selection_context;
    Var<SurfaceClosureSamplingQueryCall> _query;
    const SurfaceClosureSamplingCallables &_callables;

public:
    CallableSurfaceClosureSamplingOperation(
        const BufferFloat &scalar_parameters,
        const BufferFloat3 &vector_parameters,
        const BufferFloat &cycles_bsdf_tables, const BindlessVar &textures,
        const BindlessVar &geometry_heap,
        const Var<SurfaceClosurePointCall> &packed_point,
        const SurfaceClosurePoint &point, const SurfaceQuery &query,
        const SurfaceClosureSamplingCallables &callables) noexcept;

    [[nodiscard]] luisa::compute::Var<SurfaceClosureSelectionCall>
    selection(const SurfaceClosureExpression &closure) const noexcept override;

    [[nodiscard]] luisa::compute::Var<SurfaceClosureConditionalSampleCall>
    conditional_sample(Expr<luisa::float3> shading_normal,
                       const SurfaceClosureExpression &closure,
                       Expr<luisa::float3> glossy_normal,
                       Expr<luisa::float2> random_direction,
                       Expr<float> rescaled_lobe) const noexcept override;
};

[[nodiscard]] SurfaceSampleTrace sample_surface_closures(
    const LuisaSceneData &scene,
    const SurfaceClosureSamplingCallables &sampling_callables,
    const SurfaceClosureEvaluationCallable &evaluation_callable,
    const BufferFloat &scalar_parameters, const BufferFloat3 &vector_parameters,
    const BufferFloat &cycles_bsdf_tables, const BindlessVar &textures,
    const BindlessVar &geometry_heap, Expr<std::uint32_t> surface_tag,
    const ShaderServices &services, const SurfacePoint &point,
    Expr<float> random_lobe, Expr<luisa::float2> random_direction,
    const SurfaceQuery &query, bool trace_selection) noexcept;

// Topology-specialized entry point. The selected Surface is a host-stage
// object used only while recording this callable, so no runtime tag switch is
// emitted inside the material graph.
[[nodiscard]] SurfaceSampleTrace sample_surface_closures_for_surface(
    const LuisaSceneData &scene, const Surface &surface,
    const SurfaceClosureSamplingCallables &sampling_callables,
    const SurfaceClosureEvaluationCallable &evaluation_callable,
    const BufferFloat &scalar_parameters, const BufferFloat3 &vector_parameters,
    const BufferFloat &cycles_bsdf_tables, const BindlessVar &textures,
    const BindlessVar &geometry_heap, const ShaderServices &services,
    const SurfacePoint &point, Expr<float> random_lobe,
    Expr<luisa::float2> random_direction, const SurfaceQuery &query,
    bool trace_selection) noexcept;

}// namespace psycles::luisa_backend::detail

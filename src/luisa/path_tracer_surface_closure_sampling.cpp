#include "path_tracer_surface_closure_sampling.h"

#include "path_tracer_shader_services.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] SurfaceQuery unpack_sampling_query(
    const Var<SurfaceClosureSamplingQueryCall> &query) noexcept {
    return {.lobe_mask = query.lobe_mask,
            .transport_mode = query.transport_mode,
            .glossy_filter_roughness = query.glossy_filter_roughness,
            .reflective_caustics = query.reflective_caustics != 0u,
            .refractive_caustics = query.refractive_caustics != 0u};
}

}// namespace

SurfaceClosureSamplingCallables make_surface_closure_sampling_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    const auto reachability =
        scene->surface_values
            ? scene->surface_values->physical_closure_reachability
            : all_surface_closure_reachability;
    SurfaceClosureSelectionCallable selection =
        [reachability](UInt lobe_mask,
            Float glossy_filter_roughness,
            UInt closure_type,
            Float sample_weight,
            Float3 normal,
            Float roughness) noexcept {
            const auto context = SurfaceClosureSelectionContext{
                .lobe_mask = Expr<std::uint32_t>{lobe_mask.expression()},
                .glossy_filter_roughness =
                    Expr<float>{glossy_filter_roughness.expression()}};
            const auto closure = SurfaceClosurePhysicalCommonRecord{
                .closure_type = std::move(closure_type),
                .microfacet_fresnel = 0u,
                .weight = make_float3(0.0f),
                .sample_weight = std::move(sample_weight),
                .color_or_evaluation_scale = make_float3(0.0f),
                .normal = std::move(normal),
                .roughness = std::move(roughness)};
            return surface_closure_selection(
                context, closure, true, reachability);
        };
    selection.set_name("surface_closure_selection");

    SurfaceClosureConditionalSampleCallable conditional_sample =
        [scene, reachability](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, Var<SurfaceClosurePointCall> packed_point,
            Var<SurfaceClosureSamplingQueryCall> packed_query,
            Float3 shading_normal, Float3 glossy_normal, Float2 random_direction,
            Float rescaled_lobe, luisa::compute::Float4x4 block_0,
            luisa::compute::Float4x4 block_1) noexcept {
            BufferShaderServices services{scalar_parameters,
                                          vector_parameters,
                                          cycles_bsdf_tables,
                                          textures,
                                          geometry_heap,
                                          scene->attribute_binding_slot,
                                          scene->attribute_range_slot,
                                          scene->nishita_texture_bindings,
                                          scene->shader_color_space};
            const auto point = unpack_surface_closure_point(packed_point);
            return surface_closure_conditional_sample_from_physical_blocks(
                services,
                point,
                Expr<luisa::float3>{shading_normal.expression()},
                Expr<luisa::float4x4>{block_0.expression()},
                Expr<luisa::float4x4>{block_1.expression()},
                Expr<luisa::float3>{packed_query.incoming.expression()},
                Expr<luisa::float3>{glossy_normal.expression()},
                Expr<luisa::float2>{random_direction.expression()},
                Expr<float>{rescaled_lobe.expression()},
                unpack_sampling_query(packed_query), reachability);
        };
    conditional_sample.set_name("surface_closure_conditional_sample");
    return {.selection = std::move(selection),
            .conditional_sample = std::move(conditional_sample)};
}

CallableSurfaceClosureSamplingOperation::
    CallableSurfaceClosureSamplingOperation(
        const BufferFloat &scalar_parameters,
        const BufferFloat3 &vector_parameters,
        const BufferFloat &cycles_bsdf_tables, const BindlessVar &textures,
        const BindlessVar &geometry_heap,
        const Var<SurfaceClosurePointCall> &packed_point,
        const SurfaceClosurePoint &point, const SurfaceQuery &query,
        const SurfaceClosureSamplingCallables &callables) noexcept
    : _scalar_parameters{scalar_parameters},
      _vector_parameters{vector_parameters},
      _cycles_bsdf_tables{cycles_bsdf_tables}, _textures{textures},
      _geometry_heap{geometry_heap}, _point{packed_point},
      _selection_context{make_surface_closure_selection_context(query)},
      _callables{callables} {
    _query.incoming = make_surface_closure_sampling_incoming(point);
    _query.lobe_mask = query.lobe_mask;
    _query.transport_mode = query.transport_mode;
    _query.glossy_filter_roughness = query.glossy_filter_roughness;
    _query.reflective_caustics = select(0u, 1u, query.reflective_caustics);
    _query.refractive_caustics = select(0u, 1u, query.refractive_caustics);
}

luisa::compute::Var<SurfaceClosureSelectionCall>
CallableSurfaceClosureSamplingOperation::selection(
    const SurfaceClosureExpression &closure) const noexcept {
    return _callables.selection(
        _selection_context.lobe_mask, _selection_context.glossy_filter_roughness,
        closure.closure_type, closure.sample_weight,
        closure.normal, closure.roughness);
}

luisa::compute::Var<SurfaceClosureConditionalSampleCall>
CallableSurfaceClosureSamplingOperation::conditional_sample(
    Expr<luisa::float3> shading_normal, const SurfaceClosureExpression &closure,
    Expr<luisa::float3> glossy_normal, Expr<luisa::float2> random_direction,
    Expr<float> rescaled_lobe) const noexcept {
    const auto blocks = pack_surface_closure_physical(closure.reference());
    return conditional_sample_physical(shading_normal, blocks, glossy_normal,
                                       random_direction, rescaled_lobe);
}

luisa::compute::Var<SurfaceClosureConditionalSampleCall>
CallableSurfaceClosureSamplingOperation::conditional_sample_physical(
    Expr<luisa::float3> shading_normal,
    const SurfaceClosurePhysicalBlocks &closure,
    Expr<luisa::float3> glossy_normal, Expr<luisa::float2> random_direction,
    Expr<float> rescaled_lobe) const noexcept {
    return _callables.conditional_sample(
        _scalar_parameters, _vector_parameters, _cycles_bsdf_tables, _textures,
        _geometry_heap, _point, _query, shading_normal, glossy_normal,
        random_direction, rescaled_lobe, closure.block_0, closure.block_1);
}

SurfaceSampleTrace sample_surface_closures_for_surface(
    const LuisaSceneData &scene, const Surface &surface,
    const SurfaceClosureSamplingCallables &sampling_callables,
    const SurfaceClosureEvaluationCallable &evaluation_callable,
    const BufferFloat &scalar_parameters, const BufferFloat3 &vector_parameters,
    const BufferFloat &cycles_bsdf_tables, const BindlessVar &textures,
    const BindlessVar &geometry_heap, const ShaderServices &services,
    const SurfacePoint &point, Expr<float> random_lobe,
    Expr<luisa::float2> random_direction, const SurfaceQuery &query,
    bool trace_selection) noexcept {
    const SurfaceClosurePoint closure_point{point};
    const auto packed_closure_point = pack_surface_closure_point(closure_point);
    const auto policy =
        make_surface_closure_evaluation_policy(false, Expr<std::uint32_t>{0u});
    CallableSurfaceClosureEvaluationOperation evaluation{scalar_parameters,
                                                         vector_parameters,
                                                         cycles_bsdf_tables,
                                                         textures,
                                                         geometry_heap,
                                                         packed_closure_point,
                                                         closure_point,
                                                         query,
                                                         policy,
                                                         evaluation_callable};
    CallableSurfaceClosureSamplingOperation sampling{
        scalar_parameters, vector_parameters, cycles_bsdf_tables, textures,
        geometry_heap, packed_closure_point, closure_point, query,
        sampling_callables};
    SurfaceClosureSamplingVisitor visitor{
        scene.volume_metadata.closure_allocation_budget,
        closure_point,
        sampling,
        evaluation,
        random_lobe,
        random_direction,
        trace_selection};
    static_cast<void>(
        surface.collect_closures(services, point, query.reflective_caustics,
                                 query.refractive_caustics, visitor));
    return visitor.result();
}

SurfaceSampleTrace sample_surface_closures(
    const LuisaSceneData &scene,
    const SurfaceClosureSamplingCallables &sampling_callables,
    const SurfaceClosureEvaluationCallable &evaluation_callable,
    const BufferFloat &scalar_parameters, const BufferFloat3 &vector_parameters,
    const BufferFloat &cycles_bsdf_tables, const BindlessVar &textures,
    const BindlessVar &geometry_heap, Expr<std::uint32_t> surface_tag,
    const ShaderServices &services, const SurfacePoint &point,
    Expr<float> random_lobe, Expr<luisa::float2> random_direction,
    const SurfaceQuery &query, bool trace_selection) noexcept {
    auto result = SurfaceSampleTrace::zero();
    scene.surfaces.dispatch(surface_tag, [&](const Surface *surface) noexcept {
        result = sample_surface_closures_for_surface(
            scene, *surface, sampling_callables, evaluation_callable,
            scalar_parameters, vector_parameters, cycles_bsdf_tables, textures,
            geometry_heap, services, point, random_lobe, random_direction, query,
            trace_selection);
    });
    return result;
}

}// namespace psycles::luisa_backend::detail

#include "path_tracer_surface_closure_evaluation.h"

#include "path_tracer_shader_services.h"

namespace psycles::luisa_backend::detail {

SurfaceClosureEvaluationCallable
make_surface_closure_evaluation_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    const auto reachability =
        scene->surface_values
            ? scene->surface_values->physical_closure_reachability
            : all_surface_closure_reachability;
    SurfaceClosureEvaluationCallable callable = [scene, reachability](
               BufferFloat scalar_parameters,
               BufferFloat3 vector_parameters,
               BufferFloat cycles_bsdf_tables,
               BindlessVar textures,
               BindlessVar geometry_heap,
               Var<SurfaceClosurePointCall> packed_point,
               Var<SurfaceClosureEvaluationQueryCall> packed_query,
               Float3 shading_normal,
               Bool selected_sample,
               luisa::compute::Float4x4 block_0,
               luisa::compute::Float4x4 block_1) noexcept {
        BufferShaderServices services{
            scalar_parameters,
            vector_parameters,
            cycles_bsdf_tables,
            textures,
            geometry_heap,
            scene->attribute_binding_slot,
            scene->attribute_range_slot,
            scene->nishita_texture_bindings,
            scene->shader_color_space};
        const auto point = unpack_surface_closure_point(packed_point);
        const auto query = SurfaceQuery{
            .lobe_mask = packed_query.lobe_mask,
            .transport_mode = packed_query.transport_mode,
            .glossy_filter_roughness =
                packed_query.glossy_filter_roughness,
            .reflective_caustics =
                packed_query.reflective_caustics != 0u,
            .refractive_caustics =
                packed_query.refractive_caustics != 0u};
        const auto policy = SurfaceClosureEvaluationPolicy{
            .diffuse_included =
                packed_query.diffuse_included != 0u,
            .glossy_included =
                packed_query.glossy_included != 0u,
            .glass_included =
                packed_query.glass_included != 0u,
            .transmission_included =
                packed_query.transmission_included != 0u,
            // PDF gating is applied once by the outer accumulator.
            .preserve_pdf = true};
        return surface_closure_evaluation_contribution_from_physical_blocks(
            services,
            point,
            Expr<luisa::float3>{shading_normal.expression()},
            Expr<luisa::float4x4>{block_0.expression()},
            Expr<luisa::float4x4>{block_1.expression()},
            Expr<luisa::float3>{packed_query.incoming.expression()},
            Expr<luisa::float3>{packed_query.outgoing.expression()},
            query,
            policy,
            Expr<bool>{selected_sample.expression()}, reachability);
    };
    callable.set_name("surface_closure_evaluation");
    return callable;
}

CallableSurfaceClosureEvaluationOperation::
    CallableSurfaceClosureEvaluationOperation(
        const BufferFloat &scalar_parameters,
        const BufferFloat3 &vector_parameters,
        const BufferFloat &cycles_bsdf_tables,
        const BindlessVar &textures,
        const BindlessVar &geometry_heap,
        const Var<SurfaceClosurePointCall> &packed_point,
        const SurfaceClosurePoint &point,
        const SurfaceQuery &query,
        const SurfaceClosureEvaluationPolicy &policy,
        const SurfaceClosureEvaluationCallable &callable) noexcept
    : _scalar_parameters{scalar_parameters},
      _vector_parameters{vector_parameters},
      _cycles_bsdf_tables{cycles_bsdf_tables},
      _textures{textures},
      _geometry_heap{geometry_heap},
      _point{packed_point},
      _surface_point{point},
      _callable{callable} {
    _query.lobe_mask = query.lobe_mask;
    _query.transport_mode = query.transport_mode;
    _query.glossy_filter_roughness =
        query.glossy_filter_roughness;
    _query.reflective_caustics = select(
        0u, 1u, query.reflective_caustics);
    _query.refractive_caustics = select(
        0u, 1u, query.refractive_caustics);
    _query.diffuse_included = select(
        0u, 1u, policy.diffuse_included);
    _query.glossy_included = select(
        0u, 1u, policy.glossy_included);
    _query.glass_included = select(
        0u, 1u, policy.glass_included);
    _query.transmission_included = select(
        0u, 1u, policy.transmission_included);
}

void CallableSurfaceClosureEvaluationOperation::set_outgoing(
    Expr<luisa::float3> outgoing) noexcept {
    const auto directions =
        make_surface_closure_evaluation_directions(
            _surface_point, outgoing);
    _query.incoming = directions.incoming;
    _query.outgoing = directions.outgoing;
}

luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
CallableSurfaceClosureEvaluationOperation::evaluate(
    Expr<luisa::float3> shading_normal,
    const SurfaceClosureExpression &closure,
    Expr<bool> selected_sample) const noexcept {
    const auto blocks = pack_surface_closure_physical(
        closure.reference());
    return evaluate_physical(shading_normal, blocks, selected_sample);
}

luisa::compute::Var<SurfaceClosureEvaluationContributionCall>
CallableSurfaceClosureEvaluationOperation::evaluate_physical(
    Expr<luisa::float3> shading_normal,
    const SurfaceClosurePhysicalBlocks &closure,
    Expr<bool> selected_sample) const noexcept {
    return _callable(
        _scalar_parameters,
        _vector_parameters,
        _cycles_bsdf_tables,
        _textures,
        _geometry_heap,
        _point,
        _query,
        shading_normal,
        selected_sample,
        closure.block_0,
        closure.block_1);
}

}// namespace psycles::luisa_backend::detail

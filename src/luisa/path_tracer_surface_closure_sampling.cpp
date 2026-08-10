#include "path_tracer_surface_closure_sampling.h"

#include "path_tracer_shader_services.h"

#include <psycles/luisa/surface_closure_blocks.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] SurfaceQuery unpack_sampling_query(
    const Var<SurfaceClosureSamplingQueryCall> &query) noexcept {
    return {
        .lobe_mask = query.lobe_mask,
        .transport_mode = query.transport_mode,
        .glossy_filter_roughness =
            query.glossy_filter_roughness,
        .reflective_caustics =
            query.reflective_caustics != 0u,
        .refractive_caustics =
            query.refractive_caustics != 0u};
}

}// namespace

SurfaceClosureSamplingCallables
make_surface_closure_sampling_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    SurfaceClosureSelectionCallable selection =
        [](Float3 geometric_normal,
            Float3 incoming,
            UInt lobe_mask,
            Float glossy_filter_roughness,
            Bool use_bump_map_correction,
            UInt kind,
            UInt lobe,
            UInt bssrdf_method,
            Float allocation_weight,
            Float sample_weight,
            Bool setup_valid,
            Float3 normal,
            Float roughness,
            Bool preserve_ggx_energy,
            Bool beckmann) noexcept {
            const auto context = SurfaceClosureSelectionContext{
                .geometric_normal = Expr<luisa::float3>{
                    geometric_normal.expression()},
                .incoming = Expr<luisa::float3>{
                    incoming.expression()},
                .lobe_mask = Expr<std::uint32_t>{
                    lobe_mask.expression()},
                .glossy_filter_roughness = Expr<float>{
                    glossy_filter_roughness.expression()},
                .use_bump_map_correction = Expr<bool>{
                    use_bump_map_correction.expression()}};
            const auto closure = SurfaceClosureSelectionInput{
                .kind = Expr<std::uint32_t>{
                    kind.expression()},
                .lobe = Expr<std::uint32_t>{
                    lobe.expression()},
                .bssrdf_method = Expr<std::uint32_t>{
                    bssrdf_method.expression()},
                .allocation_weight = Expr<float>{
                    allocation_weight.expression()},
                .sample_weight = Expr<float>{
                    sample_weight.expression()},
                .setup_valid = Expr<bool>{
                    setup_valid.expression()},
                .normal = Expr<luisa::float3>{
                    normal.expression()},
                .roughness = Expr<float>{
                    roughness.expression()},
                .preserve_ggx_energy = Expr<bool>{
                    preserve_ggx_energy.expression()},
                .beckmann = Expr<bool>{
                    beckmann.expression()}};
            return surface_closure_selection(
                context, closure);
        };
    selection.set_name("surface_closure_selection");

    SurfaceClosureConditionalSampleCallable conditional_sample =
        [scene](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            Var<SurfacePointCall> packed_point,
            Var<SurfaceClosureSamplingQueryCall> packed_query,
            Float3 shading_normal,
            Float3 glossy_normal,
            Float2 random_direction,
            Float rescaled_lobe,
            luisa::compute::Float4x4 block_0,
            luisa::compute::Float4x4 block_1,
            luisa::compute::Float4x4 block_2,
            luisa::compute::Float4x4 block_3) noexcept {
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
            const auto point = unpack_surface_point(packed_point);
            const auto closure = unpack_surface_closure(
                Expr<luisa::float4x4>{block_0.expression()},
                Expr<luisa::float4x4>{block_1.expression()},
                Expr<luisa::float4x4>{block_2.expression()},
                Expr<luisa::float4x4>{block_3.expression()});
            return surface_closure_conditional_sample(
                services,
                point,
                Expr<luisa::float3>{
                    shading_normal.expression()},
                closure,
                Expr<luisa::float3>{
                    packed_query.incoming.expression()},
                Expr<luisa::float3>{
                    glossy_normal.expression()},
                Expr<luisa::float2>{
                    random_direction.expression()},
                Expr<float>{rescaled_lobe.expression()},
                unpack_sampling_query(packed_query));
        };
    conditional_sample.set_name(
        "surface_closure_conditional_sample");
    return {
        .selection = std::move(selection),
        .conditional_sample = std::move(conditional_sample)};
}

CallableSurfaceClosureSamplingOperation::
    CallableSurfaceClosureSamplingOperation(
        const BufferFloat &scalar_parameters,
        const BufferFloat3 &vector_parameters,
        const BufferFloat &cycles_bsdf_tables,
        const BindlessVar &textures,
        const BindlessVar &geometry_heap,
        const Var<SurfacePointCall> &packed_point,
        const SurfacePoint &point,
        const SurfaceQuery &query,
        const SurfaceClosureSamplingCallables &callables) noexcept
    : _scalar_parameters{scalar_parameters},
      _vector_parameters{vector_parameters},
      _cycles_bsdf_tables{cycles_bsdf_tables},
      _textures{textures},
      _geometry_heap{geometry_heap},
      _point{packed_point},
      _selection_context{make_surface_closure_selection_context(
          point,
          Expr<luisa::float3>{
              make_surface_closure_sampling_incoming(point).expression()},
          query)},
      _callables{callables} {
    _query.incoming = Float3{_selection_context.incoming};
    _query.lobe_mask = query.lobe_mask;
    _query.transport_mode = query.transport_mode;
    _query.glossy_filter_roughness =
        query.glossy_filter_roughness;
    _query.reflective_caustics = select(
        0u, 1u, query.reflective_caustics);
    _query.refractive_caustics = select(
        0u, 1u, query.refractive_caustics);
}

luisa::compute::Var<SurfaceClosureSelectionCall>
CallableSurfaceClosureSamplingOperation::selection(
    const SurfaceClosureExpression &closure) const noexcept {
    return _callables.selection(
        _selection_context.geometric_normal,
        _selection_context.incoming,
        _selection_context.lobe_mask,
        _selection_context.glossy_filter_roughness,
        _selection_context.use_bump_map_correction,
        closure.kind,
        closure.lobe,
        closure.bssrdf_method,
        closure.allocation_weight,
        closure.sample_weight,
        closure.setup_valid,
        closure.normal,
        closure.roughness,
        closure.preserve_ggx_energy,
        closure.beckmann);
}

luisa::compute::Var<SurfaceClosureConditionalSampleCall>
CallableSurfaceClosureSamplingOperation::conditional_sample(
    Expr<luisa::float3> shading_normal,
    const SurfaceClosureExpression &closure,
    Expr<luisa::float3> glossy_normal,
    Expr<luisa::float2> random_direction,
    Expr<float> rescaled_lobe) const noexcept {
    const auto blocks = pack_surface_closure(closure.reference());
    return _callables.conditional_sample(
        _scalar_parameters,
        _vector_parameters,
        _cycles_bsdf_tables,
        _textures,
        _geometry_heap,
        _point,
        _query,
        shading_normal,
        glossy_normal,
        random_direction,
        rescaled_lobe,
        blocks.block_0,
        blocks.block_1,
        blocks.block_2,
        blocks.block_3);
}

SurfaceSampleTrace sample_surface_closures(
    const LuisaSceneData &scene,
    const SurfaceClosureSamplingCallables &sampling_callables,
    const SurfaceClosureEvaluationCallable &evaluation_callable,
    const BufferFloat &scalar_parameters,
    const BufferFloat3 &vector_parameters,
    const BufferFloat &cycles_bsdf_tables,
    const BindlessVar &textures,
    const BindlessVar &geometry_heap,
    Expr<std::uint32_t> surface_tag,
    const Var<SurfacePointCall> &packed_point,
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<float> random_lobe,
    Expr<luisa::float2> random_direction,
    const SurfaceQuery &query,
    bool trace_selection) noexcept {
    const auto policy = make_surface_closure_evaluation_policy(
        false, Expr<std::uint32_t>{0u});
    CallableSurfaceClosureEvaluationOperation evaluation{
        scalar_parameters,
        vector_parameters,
        cycles_bsdf_tables,
        textures,
        geometry_heap,
        packed_point,
        point,
        query,
        policy,
        evaluation_callable};
    CallableSurfaceClosureSamplingOperation sampling{
        scalar_parameters,
        vector_parameters,
        cycles_bsdf_tables,
        textures,
        geometry_heap,
        packed_point,
        point,
        query,
        sampling_callables};
    SurfaceClosureSamplingVisitor visitor{
        scene.volume_metadata.closure_allocation_budget,
        point,
        sampling,
        evaluation,
        random_lobe,
        random_direction,
        trace_selection};
    static_cast<void>(scene.surfaces.collect_closures(
        UInt{surface_tag},
        services,
        point,
        query.reflective_caustics,
        query.refractive_caustics,
        visitor));
    return visitor.result();
}

}// namespace psycles::luisa_backend::detail

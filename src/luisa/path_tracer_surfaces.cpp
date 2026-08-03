#include "path_tracer_surfaces.h"

#include "path_tracer_shader_services.h"
#include "path_tracer_surface_closure_evaluation.h"

#include <psycles/luisa/surface_closure_evaluator.h>
#include <psycles/luisa/surface_closure_operations.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

template<typename Consumer>
[[nodiscard]] decltype(auto) evaluate_surface_closures(
    const LuisaSceneData &scene,
    SurfaceClosureStorageProfile profile,
    const ShaderServices &services,
    UInt surface_tag,
    const SurfacePoint &point,
    Bool reflective_caustics,
    Bool refractive_caustics,
    Consumer &&consumer) noexcept {
    SurfaceClosureSet closures{
        scene.volume_metadata.closure_allocation_budget,
        profile};
    const auto collection = scene.surfaces.collect_closures(
        surface_tag,
        services,
        point,
        reflective_caustics,
        refractive_caustics,
        closures);
    const SurfaceClosureEvaluator evaluator{
        point, closures, collection.shading_normal};
    return std::forward<Consumer>(consumer)(evaluator);
}

}// namespace

SurfaceCallables make_surface_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    const auto closure_identity =
        make_surface_closure_identity_callable();
    const auto closure_aov =
        make_surface_closure_aov_callable();
    const auto closure_evaluation =
        make_surface_closure_evaluation_callable(scene);
    SurfaceEvaluateLightCallable evaluate_light =
        [scene, closure_evaluation](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Float3 outgoing,
            UInt lobe_mask,
            UInt transport_mode,
            Float glossy_filter_roughness,
            Bool reflective_caustics,
            Bool refractive_caustics,
            UInt shader_flags) noexcept {
            BufferShaderServices services{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            auto query = SurfaceLightQuery{
                .surface = {
                    .lobe_mask = lobe_mask,
                    .transport_mode = transport_mode,
                    .glossy_filter_roughness =
                        glossy_filter_roughness,
                    .reflective_caustics = reflective_caustics,
                    .refractive_caustics = refractive_caustics},
                .shader_flags = shader_flags};
            const auto point =
                unpack_surface_point(packed_point);
            const auto policy =
                make_surface_closure_evaluation_policy(
                    true,
                    Expr<std::uint32_t>{
                        shader_flags.expression()});
            CallableSurfaceClosureEvaluationOperation operation{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                packed_point,
                point,
                query.surface,
                policy,
                closure_evaluation};
            operation.set_outgoing(
                Expr<luisa::float3>{outgoing.expression()});
            SurfaceClosureEvaluationVisitor visitor{
                scene->volume_metadata.closure_allocation_budget,
                operation,
                Expr<bool>{policy.preserve_pdf.expression()}};
            static_cast<void>(scene->surfaces.collect_closures(
                surface_tag,
                services,
                point,
                reflective_caustics,
                refractive_caustics,
                visitor));
            return pack_surface_evaluation(visitor.result());
        };
    SurfaceRuntimeFlagsCallable runtime_flags =
        [scene, closure_identity](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Float glossy_filter_roughness,
            Bool reflective_caustics,
            Bool refractive_caustics) noexcept {
            BufferShaderServices services{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            const auto point =
                unpack_surface_point(packed_point);
            SurfaceRuntimeFlagsVisitor visitor{
                point,
                glossy_filter_roughness,
                scene->volume_metadata.closure_allocation_budget,
                closure_identity};
            static_cast<void>(scene->surfaces.collect_closures(
                surface_tag,
                services,
                point,
                reflective_caustics,
                refractive_caustics,
                visitor));
            return visitor.result();
        };
    SurfaceEmissionCallable emission =
        [scene](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Float3 outgoing,
            Bool reflective_caustics) noexcept {
            BufferShaderServices services{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            return scene->surfaces.emission(
                surface_tag,
                services,
                unpack_surface_point(packed_point),
                outgoing,
                reflective_caustics);
        };
    SurfaceConstantEmissionCallable constant_emission =
        [scene](
            BufferFloat4 parameters,
            UInt surface_tag,
            UInt parameter_block) noexcept {
            BufferSurfaceParameterServices services{
                parameters};
            return scene->surfaces.constant_emission(
                surface_tag,
                services,
                parameter_block);
        };
    SurfaceSampleCallable sample =
        [scene](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Float u_lobe,
            Float2 u_direction,
            UInt lobe_mask,
            UInt transport_mode,
            Float glossy_filter_roughness,
            Bool reflective_caustics,
            Bool refractive_caustics) noexcept {
            BufferShaderServices services{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            auto query = SurfaceQuery{
                .lobe_mask = lobe_mask,
                .transport_mode = transport_mode,
                .glossy_filter_roughness =
                    glossy_filter_roughness,
                .reflective_caustics = reflective_caustics,
                .refractive_caustics = refractive_caustics};
            const auto point =
                unpack_surface_point(packed_point);
            return pack_surface_sample(
                evaluate_surface_closures(
                    *scene,
                    SurfaceClosureStorageProfile::complete,
                    services,
                    surface_tag,
                    point,
                    reflective_caustics,
                    refractive_caustics,
                    [&](const SurfaceClosureEvaluator
                            &evaluator) noexcept {
                        return evaluator.sample(
                            services,
                            u_lobe,
                            u_direction,
                            query);
                    }));
        };
    SurfaceClosureTraceCallable closure_trace =
        [scene, closure_identity](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            UInt requested_index,
            Bool reflective_caustics,
            Bool refractive_caustics) noexcept {
            BufferShaderServices services{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            const auto point =
                unpack_surface_point(packed_point);
            SurfaceClosureTraceVisitor visitor{
                point,
                requested_index,
                scene->volume_metadata.closure_allocation_budget,
                closure_identity};
            static_cast<void>(scene->surfaces.collect_closures(
                surface_tag,
                services,
                point,
                reflective_caustics,
                refractive_caustics,
                visitor));
            return pack_surface_closure_trace(visitor.result());
        };
    SurfaceSampleTraceCallable sample_trace =
        [scene](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Float u_lobe,
            Float2 u_direction,
            UInt lobe_mask,
            UInt transport_mode,
            Float glossy_filter_roughness,
            Bool reflective_caustics,
            Bool refractive_caustics) noexcept {
            BufferShaderServices services{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            auto query = SurfaceQuery{
                .lobe_mask = lobe_mask,
                .transport_mode = transport_mode,
                .glossy_filter_roughness =
                    glossy_filter_roughness,
                .reflective_caustics = reflective_caustics,
                .refractive_caustics = refractive_caustics};
            const auto point =
                unpack_surface_point(packed_point);
            return pack_surface_sample_trace(
                evaluate_surface_closures(
                    *scene,
                    SurfaceClosureStorageProfile::complete,
                    services,
                    surface_tag,
                    point,
                    reflective_caustics,
                    refractive_caustics,
                    [&](const SurfaceClosureEvaluator
                            &evaluator) noexcept {
                        return evaluator.sample_trace(
                            services,
                            u_lobe,
                            u_direction,
                            query);
                    }));
        };
    SurfaceAovCallable aov =
        [scene, closure_aov](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point) noexcept {
            BufferShaderServices services{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            const auto point =
                unpack_surface_point(packed_point);
            SurfaceAovVisitor visitor{
                point,
                scene->volume_metadata.closure_allocation_budget,
                closure_aov};
            static_cast<void>(scene->surfaces.collect_closures(
                surface_tag,
                services,
                point,
                true,
                true,
                visitor));
            return pack_surface_aov(visitor.result());
        };
    SurfaceShadingNormalCallable shading_normal =
        [scene](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point) noexcept {
            BufferShaderServices services{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            return scene->surfaces.shading_normal(
                surface_tag,
                services,
                unpack_surface_point(packed_point));
        };
    return {
        std::move(evaluate_light),
        std::move(runtime_flags),
        std::move(constant_emission),
        std::move(emission),
        std::move(sample),
        std::move(closure_trace),
        std::move(sample_trace),
        std::move(aov),
        std::move(shading_normal)};
}

}// namespace psycles::luisa_backend::detail

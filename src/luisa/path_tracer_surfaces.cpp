#include "path_tracer_surfaces.h"

#include "path_tracer_shader_services.h"

namespace psycles::luisa_backend::detail {

SurfaceCallables make_surface_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    SurfaceEvaluateLightCallable evaluate_light =
        [scene](
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
            return pack_surface_evaluation(
                scene->surfaces.evaluate_light(
                    surface_tag,
                    services,
                    unpack_surface_point(packed_point),
                    outgoing,
                    query));
        };
    SurfaceRuntimeFlagsCallable runtime_flags =
        [scene](
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
            return scene->surfaces.runtime_flags(
                surface_tag,
                services,
                unpack_surface_point(packed_point),
                glossy_filter_roughness,
                reflective_caustics,
                refractive_caustics);
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
            return pack_surface_sample(
                scene->surfaces.sample(
                    surface_tag,
                    services,
                    unpack_surface_point(packed_point),
                    u_lobe,
                    u_direction,
                    query));
        };
    SurfaceClosureTraceCallable closure_trace =
        [scene](
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
            return pack_surface_closure_trace(
                scene->surfaces.closure_trace(
                    surface_tag,
                    services,
                    unpack_surface_point(packed_point),
                    requested_index,
                    reflective_caustics,
                    refractive_caustics));
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
            return pack_surface_sample_trace(
                scene->surfaces.sample_trace(
                    surface_tag,
                    services,
                    unpack_surface_point(packed_point),
                    u_lobe,
                    u_direction,
                    query));
        };
    SurfaceAovCallable aov =
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
            return pack_surface_aov(
                scene->surfaces.aov(
                    surface_tag,
                    services,
                    unpack_surface_point(packed_point)));
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

#include "path_tracer_surfaces.h"

#include "path_tracer_shader_services.h"

namespace psycles::luisa_backend::detail {

SurfaceCallables make_surface_callables(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    SurfaceEvaluateCallable evaluate =
        [scene](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Float3 outgoing,
            UInt lobe_mask,
            UInt transport_mode) noexcept {
            BufferShaderServices services{
                parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            auto point =
                unpack_surface_point(packed_point);
            auto query = SurfaceQuery{
                .lobe_mask = lobe_mask,
                .transport_mode = transport_mode};
            return pack_surface_evaluation(
                scene->surfaces.evaluate(
                    surface_tag,
                    services,
                    point,
                    outgoing,
                    query));
        };
    SurfaceEmissionCallable emission =
        [scene](
            BufferFloat4 parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Float3 outgoing) noexcept {
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
                outgoing);
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
            UInt transport_mode) noexcept {
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
                .transport_mode = transport_mode};
            return pack_surface_sample(
                scene->surfaces.sample(
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
    return {
        std::move(evaluate),
        std::move(emission),
        std::move(sample),
        std::move(aov)};
}

}// namespace psycles::luisa_backend::detail

#include "path_tracer_environment.h"

#include "path_tracer_shader_services.h"

#include <psycles/luisa/cycles_nishita.h>

namespace psycles::luisa_backend::detail {

EnvironmentCallables make_environment_callables(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize,
    const SurfaceEmissionCallable &surface_emission) {
    EnvironmentBaseCallable base =
        [scene, surface_emission](
            Float3 direction,
            Float3 background) noexcept {
            BufferShaderServices services{
                scene->parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_bindings,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            auto evaluate_world_graph =
                [&](Float3 world_direction) noexcept {
                    Float3 world = background;
                    if (scene->world_surface) {
                        SurfacePoint world_point{
                            .position = make_float3(0.0f),
                            .object_position =
                                make_float3(0.0f),
                            .object_location =
                                make_float3(0.0f),
                            .generated = world_direction,
                            .geometric_normal =
                                -world_direction,
                            .shading_normal =
                                -world_direction,
                            .dpdu = make_float3(
                                1.0f, 0.0f, 0.0f),
                            .dpdv = make_float3(
                                0.0f, 1.0f, 0.0f),
                            .dPdx = make_float3(0.0f),
                            .dPdy = make_float3(0.0f),
                            .object_dPdx =
                                make_float3(0.0f),
                            .object_dPdy =
                                make_float3(0.0f),
                            .generated_dx =
                                make_float3(0.0f),
                            .generated_dy =
                                make_float3(0.0f),
                            .incoming = -world_direction,
                            .uv = make_float2(0.0f),
                            .uv_dx = make_float2(0.0f),
                            .uv_dy = make_float2(0.0f),
                            .geometry_index = ~0u,
                            .barycentric =
                                make_float2(0.0f),
                            .barycentric_dx =
                                make_float2(0.0f),
                            .barycentric_dy =
                                make_float2(0.0f),
                            .instance_id = 0u,
                            .primitive_id = 0u,
                            .parameter_block =
                                scene->world_surface
                                    ->parameter_block,
                            .object_random = 0.0f,
                            .particle_index = 0u,
                            .random_per_island = 0.0f,
                            .ray_visibility =
                                camera_visibility,
                            .ray_events = 0u,
                            .ray_depth = 0u,
                            .diffuse_depth = 0u,
                            .glossy_depth = 0u,
                            .transparent_depth = 0u,
                            .transmission_depth = 0u,
                            .ray_length = 0.0f,
                            .time = 0.0f,
                            .back_facing = false};
                        world += surface_emission(
                            scene->parameter_buffer,
                            scene->cycles_bsdf_table_buffer,
                            scene->texture_heap,
                            scene->heap,
                            UInt{
                                scene->world_surface
                                    ->surface_tag},
                            pack_surface_point(world_point),
                            -world_direction);
                    }
                    return world;
                };
            if (scene->environment_texture_slot) {
                if (scene->nishita_environment) {
                    const auto &sky =
                        scene->nishita_environment
                            ->parameters;
                    return max(
                               services.xyz_to_rgb(
                                   cycles_nishita::
                                       sky_radiance_xyz(
                                           scene->texture_heap
                                               ->tex2d(
                                                   *scene
                                                        ->environment_texture_slot),
                                           direction,
                                           sky.sun_rotation)),
                               make_float3(0.0f)) *
                           sky.background_strength;
                }
                auto u = fract(
                    (pi - atan2(direction.y, direction.x)) /
                    (2.0f * pi));
                auto half_texel_y =
                    0.5f /
                    static_cast<float>(std::max(
                        scene->environment_height, 1u));
                auto v = clamp(
                    acos(clamp(direction.z, -1.0f, 1.0f)) /
                        pi,
                    half_texel_y,
                    1.0f - half_texel_y);
                return scene->texture_heap
                    ->tex2d(*scene->environment_texture_slot)
                    .sample(make_float2(u, v))
                    .xyz();
            }
            return evaluate_world_graph(direction);
        };

    std::vector<EnvironmentSunCallable> suns;
    suns.reserve(scene->environment_suns.size());
    for (const auto &sun : scene->environment_suns) {
        EnvironmentSunCallable evaluate =
            [safe_normalize, sun](
                Float3 direction) noexcept {
                Float3 axis = safe_normalize(
                    to_luisa(sun.direction),
                    make_float3(0.0f, 0.0f, 1.0f));
                auto cosine = clamp(
                    dot(direction, axis), -1.0f, 1.0f);
                auto radius = std::max(
                    sun.angular_radius, 1.0e-7f);
                auto radial_distance =
                    acos(cosine) / radius;
                auto limb =
                    0.4f +
                    0.6f *
                        sqrt(max(
                            1.0f -
                                radial_distance *
                                    radial_distance,
                            0.0f));
                auto inside =
                    cosine >= std::cos(sun.angular_radius);
                return select(
                    make_float3(0.0f),
                    to_luisa(sun.radiance) *
                        (limb / 0.8f),
                    inside);
            };
        suns.emplace_back(std::move(evaluate));
    }

    EnvironmentSunCallable nishita_sun =
        [scene](Float3 direction) noexcept {
            if (!scene->nishita_environment ||
                scene->nishita_environment
                        ->angular_radius <=
                    0.0f) {
                return Float3{make_float3(0.0f)};
            }
            BufferShaderServices services{
                scene->parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_bindings,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            const auto &sun =
                *scene->nishita_environment;
            const auto &sky = sun.parameters;
            return max(
                       services.xyz_to_rgb(
                           cycles_nishita::
                               sun_disc_radiance_xyz(
                                   direction,
                                   make_float3(
                                       sun.sun_direction),
                                   make_float3(
                                       sun.pixel_bottom_xyz),
                                   make_float3(
                                       sun.pixel_top_xyz),
                                   sky.sun_elevation,
                                   sky.angular_diameter,
                                   sky.sun_intensity)),
                       make_float3(0.0f)) *
                   sky.background_strength;
        };

    return {
        std::move(base),
        std::move(suns),
        std::move(nishita_sun)};
}

}// namespace psycles::luisa_backend::detail

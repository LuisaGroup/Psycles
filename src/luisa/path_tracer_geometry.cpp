#include "path_tracer_geometry.h"

#include "path_tracer_shader_services.h"

namespace psycles::luisa_backend::detail {

TraceShadowCallable make_trace_shadow_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize) noexcept {
    TraceShadowCallable trace_shadow =
        [scene, safe_normalize](
            Var<luisa::compute::Ray> shadow_ray) noexcept {
            BufferShaderServices services{
                scene->parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            Float3 transmittance = make_float3(1.0f);
            auto committed =
                scene->accel
                    ->traverse(
                        shadow_ray,
                        {.visibility_mask =
                             shadow_visibility})
                    .on_surface_candidate(
                        [&](luisa::compute::SurfaceCandidate
                                &candidate) noexcept {
                            auto hit = candidate.hit();
                            Var<InstanceGpu> instance =
                                scene->instance_buffer->read(
                                    hit->inst);
                            Var<GeometryGpu> geometry =
                                scene->geometry_buffer->read(
                                    instance.geometry_index);
                            Var<Triangle> triangle =
                                scene->heap
                                    ->buffer<Triangle>(
                                        geometry.bindless_base)
                                    .read(hit->prim);
                            Float3 p0 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        geometry.bindless_base +
                                        1u)
                                    .read(triangle.i0);
                            Float3 p1 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        geometry.bindless_base +
                                        1u)
                                    .read(triangle.i1);
                            Float3 p2 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        geometry.bindless_base +
                                        1u)
                                    .read(triangle.i2);
                            Float3 n0 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        geometry.bindless_base +
                                        2u)
                                    .read(triangle.i0);
                            Float3 n1 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        geometry.bindless_base +
                                        2u)
                                    .read(triangle.i1);
                            Float3 n2 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        geometry.bindless_base +
                                        2u)
                                    .read(triangle.i2);
                            Float2 uv0 =
                                scene->heap
                                    ->buffer<luisa::float2>(
                                        geometry.bindless_base +
                                        3u)
                                    .read(triangle.i0);
                            Float2 uv1 =
                                scene->heap
                                    ->buffer<luisa::float2>(
                                        geometry.bindless_base +
                                        3u)
                                    .read(triangle.i1);
                            Float2 uv2 =
                                scene->heap
                                    ->buffer<luisa::float2>(
                                        geometry.bindless_base +
                                        3u)
                                    .read(triangle.i2);
                            Float4 tangent0 =
                                scene->heap
                                    ->buffer<luisa::float4>(
                                        geometry.bindless_base +
                                        7u)
                                    .read(triangle.i0);
                            Float4 tangent1 =
                                scene->heap
                                    ->buffer<luisa::float4>(
                                        geometry.bindless_base +
                                        7u)
                                    .read(triangle.i1);
                            Float4 tangent2 =
                                scene->heap
                                    ->buffer<luisa::float4>(
                                        geometry.bindless_base +
                                        7u)
                                    .read(triangle.i2);
                            Float3 generated0 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        geometry.bindless_base +
                                        5u)
                                    .read(triangle.i0);
                            Float3 generated1 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        geometry.bindless_base +
                                        5u)
                                    .read(triangle.i1);
                            Float3 generated2 =
                                scene->heap
                                    ->buffer<luisa::float3>(
                                        geometry.bindless_base +
                                        5u)
                                    .read(triangle.i2);
                            Float random_per_island =
                                scene->heap
                                    ->buffer<float>(
                                        geometry.bindless_base +
                                        6u)
                                    .read(hit->prim);
                            UInt material_slot =
                                scene->heap
                                    ->buffer<luisa::uint>(
                                        geometry.bindless_base +
                                        4u)
                                    .read(hit->prim);

                            auto object_to_world =
                                scene->accel
                                    ->instance_transform(
                                        hit->inst);
                            auto normal_to_world =
                                transpose(inverse(
                                    object_to_world));
                            Float3 wp0 =
                                (object_to_world *
                                 make_float4(p0, 1.0f))
                                    .xyz();
                            Float3 wp1 =
                                (object_to_world *
                                 make_float4(p1, 1.0f))
                                    .xyz();
                            Float3 wp2 =
                                (object_to_world *
                                 make_float4(p2, 1.0f))
                                    .xyz();
                            static_cast<void>(wp2);
                            auto candidate_ray =
                                candidate.ray();
                            Float3 object_geometric_normal =
                                safe_normalize(
                                    cross(
                                        p1 - p0,
                                        p2 - p0),
                                    make_float3(
                                        0.0f,
                                        0.0f,
                                        1.0f));
                            Float3 geometric_normal =
                                safe_normalize(
                                    (normal_to_world *
                                     make_float4(
                                         object_geometric_normal,
                                         0.0f))
                                        .xyz(),
                                    -candidate_ray
                                         ->direction());
                            Float3 object_shading_normal =
                                triangle_interpolate(
                                    hit->bary,
                                    n0,
                                    n1,
                                    n2);
                            Float4 object_tangent =
                                triangle_interpolate(
                                    hit->bary,
                                    tangent0,
                                    tangent1,
                                    tangent2);
                            Float3 shading_normal =
                                safe_normalize(
                                    (normal_to_world *
                                     make_float4(
                                         object_shading_normal,
                                         0.0f))
                                        .xyz(),
                                    geometric_normal);
                            Bool back_facing =
                                dot(
                                    geometric_normal,
                                    -candidate_ray
                                         ->direction()) <
                                0.0f;
                            geometric_normal = select(
                                geometric_normal,
                                -geometric_normal,
                                back_facing);
                            shading_normal = select(
                                shading_normal,
                                -shading_normal,
                                back_facing);
                            shading_normal = select(
                                shading_normal,
                                -shading_normal,
                                dot(
                                    shading_normal,
                                    geometric_normal) <
                                    0.0f);
                            Float3 position =
                                candidate_ray->origin() +
                                candidate_ray->direction() *
                                    hit->committed_ray_t;
                            Float3 tangent =
                                safe_normalize(
                                    (object_to_world *
                                     make_float4(
                                         object_tangent.xyz(),
                                         0.0f))
                                        .xyz(),
                                    safe_normalize(
                                        (wp1 - wp0) -
                                            geometric_normal *
                                                dot(
                                                    wp1 - wp0,
                                                    geometric_normal),
                                        make_float3(
                                            1.0f,
                                            0.0f,
                                            0.0f)));
                            UInt2 material_binding =
                                scene
                                    ->geometry_material_buffer
                                    ->read(
                                        geometry.material_offset +
                                        min(
                                            material_slot,
                                            max(
                                                geometry
                                                    .material_count,
                                                1u) -
                                                1u));
                            $if (material_slot <
                                 instance.override_count) {
                                material_binding =
                                    scene
                                        ->override_material_buffer
                                        ->read(
                                            instance
                                                .override_offset +
                                            material_slot);
                            };
                            UInt surface_tag =
                                material_binding.x;
                            SurfacePoint point{
                                .position = position,
                                .object_position =
                                    triangle_interpolate(
                                        hit->bary,
                                        p0,
                                        p1,
                                        p2),
                                .object_location =
                                    (object_to_world *
                                     make_float4(
                                         0.0f,
                                         0.0f,
                                         0.0f,
                                         1.0f))
                                        .xyz(),
                                .generated =
                                    triangle_interpolate(
                                        hit->bary,
                                        generated0,
                                        generated1,
                                        generated2),
                                .geometric_normal =
                                    geometric_normal,
                                .shading_normal =
                                    shading_normal,
                                .object_shading_normal =
                                    object_shading_normal,
                                .object_tangent =
                                    object_tangent.xyz(),
                                .tangent_sign =
                                    object_tangent.w,
                                .normal_to_world_x =
                                    (normal_to_world *
                                     make_float4(
                                         1.0f,
                                         0.0f,
                                         0.0f,
                                         0.0f))
                                        .xyz(),
                                .normal_to_world_y =
                                    (normal_to_world *
                                     make_float4(
                                         0.0f,
                                         1.0f,
                                         0.0f,
                                         0.0f))
                                        .xyz(),
                                .normal_to_world_z =
                                    (normal_to_world *
                                     make_float4(
                                         0.0f,
                                         0.0f,
                                         1.0f,
                                         0.0f))
                                        .xyz(),
                                .dpdu = tangent,
                                .dpdv = cross(
                                    shading_normal, tangent),
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
                                .incoming =
                                    -candidate_ray
                                         ->direction(),
                                .uv = triangle_interpolate(
                                    hit->bary,
                                    uv0,
                                    uv1,
                                    uv2),
                                .uv_dx = make_float2(0.0f),
                                .uv_dy = make_float2(0.0f),
                                .geometry_index =
                                    instance.geometry_index,
                                .barycentric = hit->bary,
                                .barycentric_dx =
                                    make_float2(0.0f),
                                .barycentric_dy =
                                    make_float2(0.0f),
                                .instance_id = hit->inst,
                                .primitive_id = hit->prim,
                                .parameter_block =
                                    material_binding.y,
                                .object_random =
                                    instance.object_random,
                                .particle_index =
                                    instance.particle_index,
                                .random_per_island =
                                    random_per_island,
                                .ray_visibility =
                                    shadow_visibility,
                                .ray_events = 0u,
                                .ray_depth = 0u,
                                .diffuse_depth = 0u,
                                .glossy_depth = 0u,
                                .transparent_depth = 0u,
                                .transmission_depth = 0u,
                                .ray_length =
                                    hit->committed_ray_t,
                                .time = 0.0f,
                                .back_facing = back_facing};
                            auto transparent =
                                clamp(
                                    scene->surfaces
                                        .transparent_extinction(
                                            surface_tag,
                                            services,
                                            point),
                                    make_float3(0.0f),
                                    make_float3(1.0f));
                            auto carries_light =
                                max(
                                    transparent.x,
                                    max(
                                        transparent.y,
                                        transparent.z)) >
                                0.0f;
                            transmittance *= select(
                                make_float3(1.0f),
                                transparent,
                                carries_light);
                            $if (!carries_light) {
                                candidate.commit();
                            };
                        })
                    .on_procedural_candidate(
                        [](luisa::compute::
                               ProceduralCandidate &) noexcept {})
                    .trace();
            return select(
                make_float3(0.0f),
                transmittance,
                committed->miss());
        };
    return trace_shadow;
}

}// namespace psycles::luisa_backend::detail

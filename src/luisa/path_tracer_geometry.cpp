#include "path_tracer_geometry.h"

#include "path_kernel_triangle_geometry.h"
#include "path_tracer_shader_services.h"

#include <psycles/luisa/surface_ray.h>

namespace psycles::luisa_backend::detail {

using EvaluateShadowSurfaceCallable =
    Callable<luisa::float3(
        luisa::compute::Ray,
        luisa::compute::CommittedHit,
        ShaderEvaluationStateCall)>;

[[nodiscard]] static EvaluateShadowSurfaceCallable
make_evaluate_shadow_surface_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize) noexcept {
    const auto triangle_geometry =
        make_triangle_geometry_component();
    EvaluateShadowSurfaceCallable evaluate_shadow_surface =
        [scene, safe_normalize, triangle_geometry](
            Var<luisa::compute::Ray> candidate_ray,
            Var<luisa::compute::CommittedHit> hit,
            Var<ShaderEvaluationStateCall>
                shader_state_call) noexcept {
            const auto shader_state =
                unpack_shader_evaluation_state(
                    shader_state_call);
            BufferShaderServices services{
                scene->parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
                                auto attributes =
                                    triangle_geometry->emit(
                                        scene,
                                        hit->inst,
                                        hit->prim);
                                auto &primitive =
                                    attributes.primitive;
                                auto &instance =
                                    primitive.instance;
                                auto &p0 = attributes.p0;
                                auto &p1 = attributes.p1;
                                auto &p2 = attributes.p2;
                                auto &n0 = attributes.n0;
                                auto &n1 = attributes.n1;
                                auto &n2 = attributes.n2;
                                auto &uv0 = attributes.uv0;
                                auto &uv1 = attributes.uv1;
                                auto &uv2 = attributes.uv2;
                                auto &tangent0 =
                                    attributes.tangent0;
                                auto &tangent1 =
                                    attributes.tangent1;
                                auto &tangent2 =
                                    attributes.tangent2;
                                auto &generated0 =
                                    attributes.generated0;
                                auto &generated1 =
                                    attributes.generated1;
                                auto &generated2 =
                                    attributes.generated2;
                                auto &random_per_island =
                                    attributes.random_per_island;

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
                                    select(
                                        object_geometric_normal,
                                        triangle_interpolate(
                                            hit->bary,
                                            n0,
                                            n1,
                                            n2),
                                        primitive.smooth);
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
                                auto &material_binding =
                                    primitive
                                        .material_binding;
                                UInt surface_tag =
                                    material_binding
                                        .surface_tag;
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
                                        material_binding
                                            .parameter_block,
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
                                    .use_bump_map_correction =
                                        (material_binding.flags &
                                            material_flag_use_bump_map_correction) !=
                                        0u,
                                    .back_facing = back_facing};
                                cycles_path_state::
                                    apply_shader_state(
                                        point,
                                        shader_state);
            return clamp(
                scene->surfaces.transparent_extinction(
                    surface_tag,
                    services,
                    point),
                make_float3(0.0f),
                make_float3(1.0f));
        };
    return evaluate_shadow_surface;
}

TraceShadowCallable make_trace_shadow_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize) noexcept {
    const auto evaluate_shadow_surface =
        make_evaluate_shadow_surface_callable(
            scene, safe_normalize);
    TraceShadowCallable trace_shadow =
        [scene, evaluate_shadow_surface](
            Var<luisa::compute::Ray> shadow_ray,
            UInt source_instance,
            UInt source_primitive,
            UInt light_instance,
            UInt light_primitive,
            UInt transparent_maximum,
            Var<ShaderEvaluationStateCall>
                shader_state_call) noexcept {
            const auto initial_shader_state =
                unpack_shader_evaluation_state(
                    shader_state_call);
            Float3 transmittance = make_float3(1.0f);
            UInt transparent_depth =
                initial_shader_state.transparent_depth;
            Bool active = true;

            // Candidate callbacks have no traversal-order contract. Cycles
            // shades transparent shadow hits after sorting by t, so iterate
            // the order-independent closest-hit reduction and advance tmin
            // by the same one-ULP offset as transparent continuation.
            $while (active) {
                const auto committed =
                    surface_ray::
                        closest_shadow_intersection(
                            scene->accel,
                            shadow_ray,
                            source_instance,
                            source_primitive,
                            light_instance,
                            light_primitive,
                            shadow_visibility);
                $if (committed->miss()) {
                    active = false;
                }
                $else {
                    // Cycles makes the next transparent intersection
                    // opaque once transparent_max_bounce is exhausted.
                    $if (transparent_depth >=
                         transparent_maximum) {
                        transmittance = make_float3(0.0f);
                        active = false;
                    }
                    $else {
                        auto shader_state =
                            initial_shader_state;
                        shader_state.transparent_depth =
                            transparent_depth;
                        const auto transparent =
                            evaluate_shadow_surface(
                                shadow_ray,
                                committed,
                                pack_shader_evaluation_state(
                                    shader_state));
                        const auto carries_light =
                            max(
                                transparent.x,
                                max(
                                    transparent.y,
                                    transparent.z)) >
                            0.0f;
                        $if (carries_light) {
                            transmittance *= transparent;
                            transparent_depth += 1u;
                            shadow_ray->set_t_min(
                                surface_ray::
                                    intersection_t_offset(
                                        committed
                                            ->committed_ray_t));
                        }
                        $else {
                            transmittance =
                                make_float3(0.0f);
                            active = false;
                        };
                    };
                };
            };
            return transmittance;
        };
    return trace_shadow;
}

}// namespace psycles::luisa_backend::detail

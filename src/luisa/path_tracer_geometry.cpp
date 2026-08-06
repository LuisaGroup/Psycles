#include "path_tracer_geometry.h"

#include "path_kernel_scene_traversal.h"
#include "path_kernel_surface_primitive.h"
#include "path_tracer_shader_services.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

using EvaluateShadowSurfaceCallable =
    Callable<ShadowSurfaceEvaluationCall(
        luisa::compute::Ray,
        luisa::compute::CommittedHit,
        float,
        float,
        ShaderEvaluationStateCall)>;

[[nodiscard]] EvaluateShadowSurfaceCallable
make_evaluate_shadow_surface_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize) noexcept {
    const auto geometry = make_surface_primitive_geometry_component();
    EvaluateShadowSurfaceCallable evaluate_shadow_surface =
        [scene, safe_normalize, geometry](
            Var<luisa::compute::Ray> candidate_ray,
            Var<luisa::compute::CommittedHit> hit,
            Float ray_dP,
            Float ray_dD,
            Var<ShaderEvaluationStateCall> shader_state_call) noexcept {
            const auto shader_state =
                unpack_shader_evaluation_state(shader_state_call);
            BufferShaderServices services{
                scene->scalar_parameter_buffer,
                scene->vector_parameter_buffer,
                scene->cycles_bsdf_table_buffer,
                scene->texture_heap,
                scene->heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space};
            auto primitive = geometry->emit(
                scene,
                hit,
                candidate_ray,
                ray_dP,
                ray_dD,
                safe_normalize);
            auto point = std::move(primitive.point);
            point.ray_visibility = shadow_visibility;
            cycles_path_state::apply_shader_state(point, shader_state);
            Var<ShadowSurfaceEvaluationCall> result;
            result->transmittance = clamp(
                scene->surfaces.transparent_extinction(
                    primitive.surface_tag, services, point),
                make_float3(0.0f),
                make_float3(1.0f));
            result->object = primitive.cycles_object_index;
            result->primitive = primitive.cycles_primitive_index;
            result->kind = select(
                geometry_kind_triangle,
                geometry_kind_curve,
                primitive.is_curve);
            return result;
        };
    return evaluate_shadow_surface;
}

} // namespace

TraceShadowCallable make_trace_shadow_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize) noexcept {
    const auto evaluate_shadow_surface =
        make_evaluate_shadow_surface_callable(scene, safe_normalize);
    const auto traversal = make_scene_traversal_component();
    TraceShadowCallable trace_shadow =
        [scene, evaluate_shadow_surface, traversal](
            Var<luisa::compute::Ray> shadow_ray,
            Float ray_dP,
            Float ray_dD,
            UInt source_object,
            UInt source_primitive,
            UInt light_object,
            UInt light_primitive,
            UInt transparent_maximum,
            Var<ShaderEvaluationStateCall> shader_state_call) noexcept {
            const auto initial_shader_state =
                unpack_shader_evaluation_state(shader_state_call);
            Float3 transmittance = make_float3(1.0f);
            UInt transparent_depth = initial_shader_state.transparent_depth;
            Bool active = true;
            Bool first_hit = false;
            UInt first_object = surface_ray::invalid_primitive;
            UInt first_primitive = surface_ray::invalid_primitive;
            UInt first_kind = surface_ray::invalid_primitive;
            Float first_distance = 0.0f;
            Float2 first_barycentric = make_float2(0.0f);

            // Candidate callbacks have no traversal-order contract. Cycles
            // shades transparent shadow hits after sorting by t, so iterate
            // the order-independent closest-hit reduction and advance tmin
            // by the same one-ULP offset as transparent continuation.
            $while(active) {
                const auto committed = traversal->closest_shadow(
                    scene,
                    shadow_ray,
                    shadow_visibility,
                    {.object = source_object,
                     .primitive = source_primitive},
                    {.object = light_object,
                     .primitive = light_primitive});
                $if(committed->miss()) {
                    active = false;
                }
                $else {
                    // Cycles makes the next transparent intersection opaque
                    // once transparent_max_bounce is exhausted.
                    $if(transparent_depth >= transparent_maximum) {
                        transmittance = make_float3(0.0f);
                        active = false;
                    }
                    $else {
                        auto shader_state = initial_shader_state;
                        shader_state.transparent_depth = transparent_depth;
                        const auto surface = evaluate_shadow_surface(
                            shadow_ray,
                            committed,
                            ray_dP,
                            ray_dD,
                            pack_shader_evaluation_state(shader_state));
                        $if(!first_hit) {
                            first_hit = true;
                            first_object = surface->object;
                            first_primitive = surface->primitive;
                            first_kind = surface->kind;
                            first_distance = committed->committed_ray_t;
                            first_barycentric = committed->bary;
                        };
                        const auto transparent = surface->transmittance;
                        const auto carries_light =
                            max(transparent.x,
                                max(transparent.y, transparent.z)) >
                            0.0f;
                        $if(carries_light) {
                            transmittance *= transparent;
                            transparent_depth += 1u;
                            shadow_ray->set_t_min(
                                surface_ray::intersection_t_offset(
                                    committed->committed_ray_t));
                        }
                        $else {
                            transmittance = make_float3(0.0f);
                            active = false;
                        };
                    };
                };
            };
            Var<ShadowTraceResultCall> result;
            result->transmittance = transmittance;
            result->first_hit = cast<uint>(first_hit);
            result->first_object = first_object;
            result->first_primitive = first_primitive;
            result->first_kind = first_kind;
            result->first_distance = first_distance;
            result->first_barycentric = first_barycentric;
            return result;
        };
    return trace_shadow;
}

} // namespace psycles::luisa_backend::detail

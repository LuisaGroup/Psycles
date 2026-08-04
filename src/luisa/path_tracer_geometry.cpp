#include "path_tracer_geometry.h"

#include "path_kernel_scene_traversal.h"
#include "path_kernel_surface_primitive.h"
#include "path_tracer_shader_services.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

using EvaluateShadowSurfaceCallable =
    Callable<luisa::float3(
        luisa::compute::Ray,
        luisa::compute::CommittedHit,
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
                scene, hit, candidate_ray, 0.0f, 0.0f, safe_normalize);
            auto point = std::move(primitive.point);
            point.ray_visibility = shadow_visibility;
            cycles_path_state::apply_shader_state(point, shader_state);
            return clamp(
                scene->surfaces.transparent_extinction(
                    primitive.surface_tag, services, point),
                make_float3(0.0f),
                make_float3(1.0f));
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
                        const auto transparent = evaluate_shadow_surface(
                            shadow_ray,
                            committed,
                            pack_shader_evaluation_state(shader_state));
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
            return transmittance;
        };
    return trace_shadow;
}

} // namespace psycles::luisa_backend::detail

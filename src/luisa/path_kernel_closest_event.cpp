#include "path_kernel_builder.h"
#include "path_kernel_curve_primitive.h"
#include "path_kernel_triangle_primitive.h"

#include <psycles/luisa/analytic_light_intersection.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class ClosestEventStageImpl final
    : public ClosestEventStage {

  private:
    bool _analytic_light_endpoints;
    bool _ambient_occlusion_bounce_approximation;
    ScenePrimitiveStagePlan _primitive_plan;
    std::shared_ptr<const TrianglePrimitiveComponent>
        _triangles;
    std::shared_ptr<const CurvePrimitiveComponent>
        _curves;

  public:
    explicit ClosestEventStageImpl(
        bool analytic_light_endpoints,
        ScenePrimitiveStagePlan primitive_plan,
        bool ambient_occlusion_bounce_approximation) noexcept
        : _analytic_light_endpoints{
              analytic_light_endpoints},
          _ambient_occlusion_bounce_approximation{
              ambient_occlusion_bounce_approximation},
          _primitive_plan{primitive_plan},
          _triangles{
              primitive_plan.triangles
                  ? make_triangle_primitive_component()
                  : nullptr},
          _curves{
              primitive_plan.curves
                  ? make_curve_primitive_component()
                  : nullptr} {}

    ClosestPathEvent
    emit(
        PathBounceContext &bounce,
        Expr<std::uint32_t>
            excluded_analytic_light)
        const noexcept override {
        auto &sample = bounce.sample;
        const auto &config =
            sample.invocation.config;
        const auto &scene = config.scene;
        auto &ray = sample.ray;
        const auto ray_visibility =
            sample.traversal_ray_visibility();
        auto &previous_mis_origin_normal =
            sample.previous_mis_origin_normal;
        auto &path_flags = sample.path_flags;
        auto &continuation_probability =
            sample.continuation_probability;
        auto &throughput = sample.throughput;
        auto &path_depth = sample.path_depth;
        auto &transparent_depth = sample.transparent_depth;
        const auto &parameters =
            sample.invocation.parameters;

        Bool light_hit = false;
        // Unified traversal defines committed_ray_t for every result: the
        // exact selected distance for a hit and ray.t_max() for a miss.
        // Keeping a second distance in PathBounceContext would duplicate the
        // same value across the surface suspension.
        Float event_distance =
            bounce.hit->committed_ray_t;
        UInt light_hit_index =
            surface_ray::invalid_primitive;
        Float3 light_hit_position =
            make_float3(0.0f);
        Float3 light_hit_normal =
            make_float3(0.0f);
        Float2 light_hit_uv =
            make_float2(0.0f);
        Float light_hit_pdf = 0.0f;
        Float light_hit_evaluation_factor =
            0.0f;

        if (_analytic_light_endpoints &&
            scene != nullptr &&
            scene->intersectable_light_count > 0u) {
            const bool outline_lights =
                static_cast<bool>(scene->device) &&
                scene->device.backend_name() == "simd" &&
                scene->device.compute_warp_size() > 1u;
            const Float3 ray_origin = ray->origin();
            const Float3 ray_direction = ray->direction();
            const Float ray_t_min = ray->t_min();
            const Bool camera_path =
                (ray_visibility & camera_visibility) != 0u;
            const Bool had_transmission =
                (path_flags &
                 cycles_path_state::flag_mis_had_transmission) != 0u;
            const UInt excluded_light = excluded_analytic_light;
            const std::uint32_t type_mask =
                scene->intersectable_light_type_mask;
            const bool has_area =
                (type_mask & (1u << static_cast<std::uint32_t>(
                                  LightType::area))) != 0u;
            const bool has_point =
                (type_mask & (1u << static_cast<std::uint32_t>(
                                  LightType::point))) != 0u;
            const bool has_spot =
                (type_mask & (1u << static_cast<std::uint32_t>(
                                  LightType::spot))) != 0u;
            const bool has_point_or_spot = has_point | has_spot;
            const bool need_loop_guard =
                !scene->any_intersectable_light_camera_visible ||
                !scene->any_intersectable_light_indirect_mis;
            Bool run_light_loop = true;
            if (!scene->any_intersectable_light_camera_visible &&
                !scene->any_intersectable_light_indirect_mis) {
                run_light_loop =
                    camera_path & (ray_visibility != camera_visibility);
            } else if (!scene->any_intersectable_light_camera_visible) {
                run_light_loop = ray_visibility != camera_visibility;
            } else if (!scene->any_intersectable_light_indirect_mis) {
                run_light_loop = camera_path;
            }
            auto run_loop = [&]() {
                const Float inv_dir_len_sq =
                    1.0f / max(dot(ray_direction, ray_direction), 1.0e-30f);
                $for(compact_index, scene->intersectable_light_count) {
                    const Float4 bound =
                        scene->intersectable_light_bounds->read(compact_index);
                    const Float3 oc = bound.xyz() - ray_origin;
                    const Float t_proj =
                        dot(oc, ray_direction) * inv_dir_len_sq;
                    const Float t_clamped =
                        min(max(t_proj, ray_t_min),
                            max(event_distance, ray_t_min));
                    const Float3 diff = oc - ray_direction * t_clamped;
                    $if(dot(diff, diff) <= bound.w) {
                        const UInt light_index =
                            scene->intersectable_light_indices->read(
                                compact_index);
                        $if(light_index != excluded_light) {
                            Var<LightGpu> light =
                                scene->light_buffer->read(light_index);
                            const auto use_mis =
                                (light.flags & light_flag_use_mis) != 0u;
                            const auto visible =
                                (light.visibility_mask & ray_visibility) != 0u;
                            const auto eligible =
                                // Cycles lights_intersect_impl checks camera
                                // exclusion here, but indirect shader
                                // visibility in shade_light. Even an excluded
                                // diffuse/glossy lamp is an endpoint.
                                select(use_mis, visible, camera_path);
                            auto emit_area = [&]() {
                                const auto ellipse =
                                    (light.flags & light_flag_ellipse) != 0u;
                                const auto normalize_power =
                                    (light.flags & light_flag_normalize) != 0u;
                                const auto candidate =
                                    analytic_light_intersection::intersect_area(
                                        ray_origin,
                                        ray_direction,
                                        ray_t_min,
                                        event_distance,
                                        light.position,
                                        light.axis_x,
                                        light.size_u,
                                        light.axis_y,
                                        light.size_v,
                                        light.axis_z,
                                        ellipse,
                                        light.area,
                                        normalize_power);
                                $if(candidate.valid) {
                                    light_hit = true;
                                    event_distance = candidate.distance;
                                    light_hit_index = light_index;
                                    light_hit_position = candidate.position;
                                    light_hit_normal = candidate.normal;
                                    light_hit_uv = candidate.uv;
                                    light_hit_pdf = candidate.conditional_pdf;
                                    light_hit_evaluation_factor =
                                        candidate.evaluation_factor;
                                };
                            };
                            auto emit_point_spot = [&]() {
                                const auto sphere =
                                    (light.flags & light_flag_sphere) != 0u;
                                const auto normalize_power =
                                    (light.flags & light_flag_normalize) != 0u;
                                analytic_light_intersection::PointIntersection
                                    candidate{
                                        .valid = false,
                                        .distance = 0.0f,
                                        .position = make_float3(0.0f),
                                        .normal = make_float3(0.0f),
                                        .uv = make_float2(0.0f),
                                        .conditional_pdf = 0.0f,
                                        .evaluation_factor = 0.0f};
                                if (has_spot && has_point) {
                                    const auto is_spot =
                                        light.type ==
                                        static_cast<std::uint32_t>(
                                            LightType::spot);
                                    $if(is_spot) {
                                        candidate =
                                            analytic_light_intersection::
                                                intersect_spot(
                                                    ray_origin,
                                                    ray_direction,
                                                    ray_t_min,
                                                    event_distance,
                                                    light.position,
                                                    light.radius,
                                                    sphere,
                                                    light.axis_x,
                                                    light.axis_y,
                                                    light.axis_z,
                                                    light.axis_scale,
                                                    light.spot,
                                                    normalize_power,
                                                    previous_mis_origin_normal,
                                                    had_transmission);
                                    }
                                    $else {
                                        candidate =
                                            analytic_light_intersection::
                                                intersect_point(
                                                    ray_origin,
                                                    ray_direction,
                                                    ray_t_min,
                                                    event_distance,
                                                    light.position,
                                                    light.radius,
                                                    sphere,
                                                    light.axis_x,
                                                    light.axis_y,
                                                    light.axis_z,
                                                    light.axis_scale,
                                                    normalize_power,
                                                    previous_mis_origin_normal,
                                                    had_transmission);
                                    };
                                } else if (has_spot) {
                                    candidate =
                                        analytic_light_intersection::
                                            intersect_spot(
                                                ray_origin,
                                                ray_direction,
                                                ray_t_min,
                                                event_distance,
                                                light.position,
                                                light.radius,
                                                sphere,
                                                light.axis_x,
                                                light.axis_y,
                                                light.axis_z,
                                                light.axis_scale,
                                                light.spot,
                                                normalize_power,
                                                previous_mis_origin_normal,
                                                had_transmission);
                                } else if (has_point) {
                                    candidate =
                                        analytic_light_intersection::
                                            intersect_point(
                                                ray_origin,
                                                ray_direction,
                                                ray_t_min,
                                                event_distance,
                                                light.position,
                                                light.radius,
                                                sphere,
                                                light.axis_x,
                                                light.axis_y,
                                                light.axis_z,
                                                light.axis_scale,
                                                normalize_power,
                                                previous_mis_origin_normal,
                                                had_transmission);
                                }
                                $if(candidate.valid) {
                                    light_hit = true;
                                    event_distance = candidate.distance;
                                    light_hit_index = light_index;
                                    light_hit_position = candidate.position;
                                    light_hit_normal = candidate.normal;
                                    light_hit_uv = candidate.uv;
                                    light_hit_pdf = candidate.conditional_pdf;
                                    light_hit_evaluation_factor =
                                        candidate.evaluation_factor;
                                };
                            };
                            $if(eligible) {
                                if (has_area && has_point_or_spot) {
                                    $if(light.type ==
                                        static_cast<std::uint32_t>(
                                            LightType::area)) {
                                        emit_area();
                                    }
                                    $elif((light.type ==
                                           static_cast<std::uint32_t>(
                                               LightType::point)) |
                                          (light.type ==
                                           static_cast<std::uint32_t>(
                                               LightType::spot))) {
                                        emit_point_spot();
                                    };
                                } else if (has_area) {
                                    emit_area();
                                } else if (has_point_or_spot) {
                                    emit_point_spot();
                                }
                            };
                        };
                    };
                };
            };
            $outline_when_with_name(outline_lights,
                                    "simd_intersect_analytic_lights") {
                if (need_loop_guard) {
                    $if(run_light_loop) {
                        run_loop();
                    };
                } else {
                    run_loop();
                }
            };
        }

        const auto surface =
            (!light_hit) &
            (!bounce.hit->miss());
        const auto background =
            (!light_hit) &
            bounce.hit->miss();
        Bool surface_may_emit = false;
        Bool surface_has_transparent_shadow = false;
        if (!_primitive_plan.empty()) {
            const auto resolve_curve = [&] {
                const auto primitive =
                    _curves->emit(
                        scene,
                        bounce.hit->inst,
                        bounce.hit->prim);
                surface_may_emit =
                    primitive.may_emit;
                surface_has_transparent_shadow =
                    primitive.has_transparent_shadow;
            };
            const auto resolve_triangle = [&] {
                const auto primitive =
                    _triangles->emit(
                        scene,
                        bounce.hit->inst,
                        bounce.hit->prim);
                surface_may_emit =
                    primitive.may_emit;
                surface_has_transparent_shadow =
                    primitive.has_transparent_shadow;
            };
            $if(surface) {
                if (_primitive_plan.mixed()) {
                    $if(bounce.hit->is_procedural()) {
                        resolve_curve();
                    }
                    $else {
                        resolve_triangle();
                    };
                } else if (_primitive_plan.curves) {
                    resolve_curve();
                } else {
                    resolve_triangle();
                }
            };
        }
        Bool inside_volume = false;
        if (config.volume_state) {
            inside_volume = !sample.volume.stack->empty();
        }
        Bool terminated = false;
        if (_ambient_occlusion_bounce_approximation) {
            const auto ambient_occlusion_bounce =
                cycles_path_state::ambient_occlusion_bounce(
                    path_depth,
                    sample.transmission_depth,
                    sample.glossy_depth,
                    parameters.ambient_occlusion_bounces);
            const auto ambient_occlusion_decision =
                cycles_path_state::decide_ambient_occlusion_continuation(
                    ambient_occlusion_bounce,
                    surface,
                    surface_may_emit | surface_has_transparent_shadow,
                    inside_volume);
            path_flags |= ambient_occlusion_decision.deferred_flags;
            terminated =
                ambient_occlusion_decision.terminate_immediately;
        }
        // This is Cycles' integrator_intersect_terminate transfer. A surface
        // needs the decision even in vacuum; a lamp or background endpoint
        // needs it only while an active volume segment still contributes.
        $if((surface | inside_volume) & !terminated) {
            continuation_probability =
                cycles_path_state::continuation_probability(
                    path_flags,
                    path_depth,
                    transparent_depth,
                    parameters.min_bounces,
                    parameters.transparent_min_bounces,
                    throughput,
                    sample.spectral_secondary_terminated());
            const auto decision =
                cycles_path_state::decide_closest_continuation(
                    continuation_probability,
                    sample.continuation_terminate_sample(),
                    surface_may_emit,
                    inside_volume);
            path_flags |= decision.deferred_flags;
            terminated = decision.terminate_immediately;
        };
        return {
            bounce,
            std::move(light_hit),
            std::move(surface),
            std::move(background),
            std::move(event_distance),
            std::move(light_hit_index),
            std::move(light_hit_position),
            std::move(light_hit_normal),
            std::move(light_hit_uv),
            std::move(light_hit_pdf),
            std::move(
                light_hit_evaluation_factor),
            std::move(terminated),
            std::move(surface_may_emit)};
    }
};

}// namespace

std::unique_ptr<ClosestEventStage>
make_closest_event_stage(
    bool analytic_light_endpoints,
    ScenePrimitiveStagePlan primitive_plan,
    bool ambient_occlusion_bounce_approximation) {
    return std::make_unique<
        ClosestEventStageImpl>(
        analytic_light_endpoints,
        primitive_plan,
        ambient_occlusion_bounce_approximation);
}

}// namespace psycles::luisa_backend::detail

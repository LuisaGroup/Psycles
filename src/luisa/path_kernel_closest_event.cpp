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
    ScenePrimitiveStagePlan _primitive_plan;
    std::shared_ptr<const TrianglePrimitiveComponent>
        _triangles;
    std::shared_ptr<const CurvePrimitiveComponent>
        _curves;

  public:
    explicit ClosestEventStageImpl(
        bool analytic_light_endpoints,
        ScenePrimitiveStagePlan primitive_plan) noexcept
        : _analytic_light_endpoints{
              analytic_light_endpoints},
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
        auto &ray_visibility =
            sample.ray_visibility;
        auto &previous_mis_origin_normal =
            sample.previous_mis_origin_normal;
        auto &path_flags = sample.path_flags;

        Bool light_hit = false;
        Float event_distance =
            bounce.closest_surface_distance;
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

        if (_analytic_light_endpoints) {
            $for(light_index, scene->light_count) {
                Var<LightGpu> light =
                    scene->light_buffer->read(
                        light_index);
                const auto camera_path =
                    (ray_visibility &
                     camera_visibility) != 0u;
                const auto use_mis =
                    (light.flags &
                     light_flag_use_mis) != 0u;
                const auto visible =
                    (light.visibility_mask &
                     ray_visibility) != 0u;
                const auto eligible =
                    visible &
                    (camera_path | use_mis) &
                    (light_index !=
                     excluded_analytic_light);
                $if(eligible &
                    (light.type ==
                     static_cast<std::uint32_t>(
                         LightType::area))) {
                    const auto ellipse =
                        (light.flags &
                         light_flag_ellipse) != 0u;
                    const auto full_spread =
                        (light.flags &
                         light_flag_full_spread) != 0u;
                    const auto normalize_power =
                        (light.flags &
                         light_flag_normalize) != 0u;
                    const auto candidate =
                        analytic_light_intersection::
                            intersect_area(
                                ray->origin(),
                                ray->direction(),
                                ray->t_min(),
                                event_distance,
                                light.position,
                                light.axis_x,
                                light.size_u,
                                light.axis_y,
                                light.size_v,
                                light.axis_z,
                                ellipse,
                                full_spread,
                                light.spread,
                                normalize_power);
                    $if(candidate.valid) {
                        light_hit = true;
                        event_distance =
                            candidate.distance;
                        light_hit_index =
                            light_index;
                        light_hit_position =
                            candidate.position;
                        light_hit_normal =
                            candidate.normal;
                        light_hit_uv =
                            candidate.uv;
                        light_hit_pdf =
                            candidate.conditional_pdf;
                        light_hit_evaluation_factor =
                            candidate.evaluation_factor;
                    };
                };
                const auto point_type =
                    light.type ==
                    static_cast<std::uint32_t>(
                        LightType::point);
                const auto spot_type =
                    light.type ==
                    static_cast<std::uint32_t>(
                        LightType::spot);
                $if(eligible &
                    (point_type | spot_type)) {
                    const auto sphere =
                        (light.flags &
                         light_flag_sphere) != 0u;
                    const auto normalize_power =
                        (light.flags &
                         light_flag_normalize) != 0u;
                    const auto had_transmission =
                        (path_flags &
                         cycles_path_state::
                             flag_mis_had_transmission) !=
                        0u;
                    analytic_light_intersection::
                        PointIntersection candidate{
                            .valid = false,
                            .distance = 0.0f,
                            .position =
                                make_float3(0.0f),
                            .normal =
                                make_float3(0.0f),
                            .uv = make_float2(0.0f),
                            .conditional_pdf = 0.0f,
                            .evaluation_factor = 0.0f};
                    $if(spot_type) {
                        candidate =
                            analytic_light_intersection::
                                intersect_spot(
                                    ray->origin(),
                                    ray->direction(),
                                    ray->t_min(),
                                    event_distance,
                                    light.position,
                                    light.radius,
                                    sphere,
                                    light.axis_x,
                                    light.axis_y,
                                    light.axis_z,
                                    light.axis_scale,
                                    light.spot_angle,
                                    light.spot_smooth,
                                    normalize_power,
                                    previous_mis_origin_normal,
                                    had_transmission);
                    }
                    $else {
                        candidate =
                            analytic_light_intersection::
                                intersect_point(
                                    ray->origin(),
                                    ray->direction(),
                                    ray->t_min(),
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
                    $if(candidate.valid) {
                        light_hit = true;
                        event_distance =
                            candidate.distance;
                        light_hit_index =
                            light_index;
                        light_hit_position =
                            candidate.position;
                        light_hit_normal =
                            candidate.normal;
                        light_hit_uv =
                            candidate.uv;
                        light_hit_pdf =
                            candidate.conditional_pdf;
                        light_hit_evaluation_factor =
                            candidate.evaluation_factor;
                    };
                };
            };
        }

        const auto surface =
            (!light_hit) &
            (!bounce.hit->miss());
        const auto background =
            (!light_hit) &
            bounce.hit->miss();
        Bool surface_may_emit = false;
        UInt surface_emission_sampling =
            static_cast<std::uint32_t>(
                contract::EmissionSampling::none);
        if (!_primitive_plan.empty()) {
            const auto resolve_curve = [&] {
                const auto primitive =
                    _curves->emit(
                        scene,
                        bounce.hit->inst,
                        bounce.hit->prim);
                surface_may_emit =
                    primitive.may_emit;
                surface_emission_sampling =
                    primitive
                        .triangle_emission_sampling;
            };
            const auto resolve_triangle = [&] {
                const auto primitive =
                    _triangles->emit(
                        scene,
                        bounce.hit->inst,
                        bounce.hit->prim);
                surface_may_emit =
                    primitive.may_emit;
                surface_emission_sampling =
                    primitive
                        .triangle_emission_sampling;
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
            std::move(surface_may_emit),
            std::move(
                surface_emission_sampling)};
    }
};

}// namespace

std::unique_ptr<ClosestEventStage>
make_closest_event_stage(
    bool analytic_light_endpoints,
    ScenePrimitiveStagePlan primitive_plan) {
    return std::make_unique<
        ClosestEventStageImpl>(
        analytic_light_endpoints,
        primitive_plan);
}

}// namespace psycles::luisa_backend::detail

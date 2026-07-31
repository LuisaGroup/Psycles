#include "path_kernel_builder.h"

#include <psycles/luisa/analytic_light_intersection.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class ClosestEventStageImpl final
    : public ClosestEventStage {

  public:
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

        const auto surface =
            !light_hit &
            !bounce.hit->miss();
        const auto background =
            !light_hit &
            bounce.hit->miss();
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
                light_hit_evaluation_factor)};
    }
};

}// namespace

std::unique_ptr<ClosestEventStage>
make_closest_event_stage() {
    return std::make_unique<
        ClosestEventStageImpl>();
}

}// namespace psycles::luisa_backend::detail

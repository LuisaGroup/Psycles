#include "path_kernel_builder.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class ForwardLightStageImpl final
    : public ForwardLightStage {

  public:
    Bool emit(ClosestPathEvent &event)
        const noexcept override {
        auto &bounce = event.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config =
            invocation.config;
        const auto &scene = config.scene;
        const auto &kernel_parameters =
            invocation.parameters;
        auto &ray = sample.ray;
        auto &previous_delta =
            sample.previous_delta;
        auto &previous_bsdf_pdf =
            sample.previous_bsdf_pdf;
        auto &previous_mis_origin_normal =
            sample.previous_mis_origin_normal;
        auto &previous_light_tree_dt =
            sample.previous_light_tree_dt;
        auto &throughput = sample.throughput;
        auto &path_depth = sample.path_depth;
        auto &path_flags = sample.path_flags;
        auto &transparent_depth =
            sample.transparent_depth;
        const auto &forward_light_weight =
            config.light_transport
                .forward_light_weight;

        Var<LightGpu> light =
            scene->light_buffer->read(
                event.light_index);
        Float selection_pdf = scene->light_selection_pdf;
        if (config.use_light_tree) {
            const auto emitter_id =
                scene->emissive_triangle_count + event.light_index;
            selection_pdf = config.light_tree.forward_pdf(
                emitter_id,
                ray->origin(),
                previous_mis_origin_normal,
                previous_light_tree_dt,
                sample.cycles_path_visibility,
                path_flags);
        }
        const auto light_pdf =
            event.light_pdf * selection_pdf;
        const auto competing =
            (path_depth > 0u) &
            (!previous_delta);
        const auto mis_weight =
            forward_light_weight(
                previous_bsdf_pdf,
                light_pdf,
                competing,
                light_pdf > 0.0f);
        auto light_radiance =
            light.color *
            (light.power *
             event.light_evaluation_factor);
        light_radiance *=
            sample.analytic_light_shader(
                light,
                event.light_index,
                event.light_position,
                event.light_normal,
                event.light_uv,
                -ray->direction(),
                event.distance);
        const auto contribution =
            invocation
                .clamp_emission_contribution(
                throughput *
                    light_radiance *
                    mis_weight,
                path_depth);
        sample.accumulate_radiance(
            contribution);
        const auto directly_visible =
            (path_flags &
             cycles_path_state::flag_any_pass) ==
            0u;
        sample.accumulate_light_pass(
            LightPassBuffer::emission,
            select(
                make_float3(0.0f),
                contribution,
                directly_visible));
        sample.accumulate_scattered_light(
            select(
                contribution,
                make_float3(0.0f),
                directly_visible));

        transparent_depth += 1u;
        ray = make_ray(
            ray->origin(),
            ray->direction(),
            surface_ray::
                intersection_t_offset(
                    event.distance),
            ray->t_max());
        return transparent_depth >=
               kernel_parameters
                   .transparent_max_bounces;
    }
};

}// namespace

std::unique_ptr<ForwardLightStage>
make_forward_light_stage() {
    return std::make_unique<
        ForwardLightStageImpl>();
}

}// namespace psycles::luisa_backend::detail

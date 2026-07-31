#include "path_kernel_builder.h"

#include <psycles/luisa/background_sampling.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class BackgroundEventStageImpl final
    : public BackgroundEventStage {

  public:
    void emit(ClosestPathEvent &event)
        const noexcept override {
        auto &sample = event.bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config =
            invocation.config;
        const auto &scene = config.scene;
        auto &ray = sample.ray;
        auto &ray_visibility =
            sample.ray_visibility;
        auto &previous_delta =
            sample.previous_delta;
        auto &previous_bsdf_pdf =
            sample.previous_bsdf_pdf;
        auto &throughput = sample.throughput;
        auto &radiance = sample.radiance;
        auto &path_depth = sample.path_depth;
        auto &path_flags = sample.path_flags;
        auto &transparent_depth =
            sample.transparent_depth;
        auto &diffuse_depth =
            sample.diffuse_depth;
        auto &glossy_depth =
            sample.glossy_depth;
        auto &transmission_depth =
            sample.transmission_depth;
        auto &sample_environment =
            sample.sample_environment;
        auto &ray_events = sample.ray_events;
        const auto &forward_light_weight =
            config.light_transport
                .forward_light_weight;

        Bool competing =
            (path_depth > 0u) &
            (!previous_delta);
        const auto environment_selection_pdf =
            scene->environment_in_light_distribution
                ? scene->light_selection_pdf
                : 0.0f;
        Float background_pdf =
            background_sampling::pdf(
                scene->background_conditional_cdf,
                scene->background_marginal_cdf,
                scene->background_map_width,
                scene->background_map_height,
                scene->background_map_weight,
                scene->background_guided_sun_weight,
                make_float3(
                    scene->
                        background_guided_sun_axis),
                scene->background_guided_sun_radius,
                ray->direction());
        Float environment_pdf =
            environment_selection_pdf *
            background_pdf;
        Float environment_weight =
            forward_light_weight(
                previous_bsdf_pdf,
                environment_pdf,
                competing,
                environment_pdf > 0.0f);
        Float3 environment_contribution =
            invocation
                .clamp_emission_contribution(
                throughput *
                    invocation
                        .evaluate_environment(
                            ray->direction(),
                            cycles_path_state::
                                background_emission_shader_state(
                                    ray_visibility,
                                    ray_events,
                                    path_depth,
                                    diffuse_depth,
                                    glossy_depth,
                                    transparent_depth,
                                    transmission_depth)) *
                    environment_weight,
                path_depth);
        radiance += environment_contribution;
        const auto directly_visible =
            (path_flags &
             cycles_path_state::flag_any_pass) ==
            0u;
        sample_environment +=
            select(
                make_float3(0.0f),
                environment_contribution,
                directly_visible);
        sample.accumulate_scattered_light(
            select(
                environment_contribution,
                make_float3(0.0f),
                directly_visible));
    }
};

}// namespace

std::unique_ptr<BackgroundEventStage>
make_background_event_stage() {
    return std::make_unique<
        BackgroundEventStageImpl>();
}

}// namespace psycles::luisa_backend::detail

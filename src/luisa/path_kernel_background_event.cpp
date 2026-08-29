#include "path_kernel_builder.h"
#include "path_kernel_environment_light.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class BackgroundEventStageImpl final
    : public BackgroundEventStage {

  private:
    std::shared_ptr<
        const EnvironmentLightComponent>
        _environment_light{
            make_environment_light_component()};

  public:
    void emit(ClosestPathEvent &event)
        const noexcept override {
        auto &sample = event.bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config =
            invocation.config;
        const auto &scene = config.scene;
        auto &ray = sample.ray;
        const auto traversal_ray_visibility =
            sample.traversal_ray_visibility();
        const auto shader_ray_visibility =
            sample.shader_ray_visibility();
        const auto mis_competition_skipped =
            sample.mis_competition_skipped();
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
        auto &diffuse_depth =
            sample.diffuse_depth;
        auto &glossy_depth =
            sample.glossy_depth;
        auto &transmission_depth =
            sample.transmission_depth;
        auto &ray_events = sample.ray_events;
        const auto &forward_light_weight =
            config.light_transport
                .forward_light_weight;

        Bool competing =
            (path_depth > 0u) &
            (!mis_competition_skipped);
        Float environment_selection_pdf =
            scene->environment_in_light_distribution
                ? scene->light_selection_pdf
                : 0.0f;
        if (config.use_light_tree) {
            const auto environment_emitter_id =
                scene->emissive_triangle_count + scene->light_count;
            environment_selection_pdf = select(
                0.0f,
                config.light_tree.forward_pdf(
                    environment_emitter_id,
                    ray->origin(),
                    previous_mis_origin_normal,
                    previous_light_tree_dt,
                    sample.cycles_path_visibility,
                    path_flags),
                scene->environment_in_light_distribution);
        }
        Float environment_pdf =
            _environment_light
                ->from_direction(
                    scene,
                    ray->origin(),
                    ray->direction(),
                    environment_selection_pdf,
                    invocation.parameters
                        .analytic_light_count,
                    invocation.parameters
                        .portal_count);
        Float environment_weight =
            forward_light_weight(
                previous_bsdf_pdf,
                environment_pdf,
                competing,
                environment_pdf > 0.0f);
        const auto world_visible =
            (scene->world_visibility_mask &
             traversal_ray_visibility) != 0u;
        Float3 environment_radiance;
        if (scene->environment_emission_is_constant) {
            environment_radiance =
                _environment_light
                    ->evaluate_constant_emission(
                        sample);
        } else {
            environment_radiance =
                _environment_light
                    ->evaluate_emission(
                        sample,
                        ray->direction(),
                        cycles_path_state::
                            background_emission_shader_state(
                                shader_ray_visibility,
                                ray_events,
                                path_depth,
                                diffuse_depth,
                                glossy_depth,
                                transparent_depth,
                                transmission_depth));
        }
        Float3 environment_contribution =
            invocation
                .clamp_emission_contribution(
                    throughput *
                        environment_radiance *
                        environment_weight,
                    path_depth);
        environment_contribution =
            select(
                make_float3(0.0f),
                environment_contribution,
                world_visible);
        const auto transparent_background_ray =
            (invocation.parameters
                 .transparent_background !=
             0u) &
            ((path_flags &
              cycles_path_state::
                  flag_transparent_background) !=
             0u);
        $if(transparent_background_ray) {
            sample.accumulate_transparency(
                (throughput.x +
                 throughput.y +
                 throughput.z) *
                (1.0f / 3.0f));
        }
        $else {
            sample.accumulate_radiance(
                environment_contribution);
        };
        const auto directly_visible =
            (path_flags &
             cycles_path_state::flag_any_pass) ==
            0u;
        sample.accumulate_light_pass(
            LightPassBuffer::environment,
            select(
                make_float3(0.0f),
                environment_contribution,
                directly_visible));
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

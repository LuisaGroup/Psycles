#include "path_kernel_builder.h"
#include "path_kernel_environment_light.h"
#include "virtual_texture_cache.h"

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
    bool _ambient_occlusion_bounce_approximation;

  public:
    explicit BackgroundEventStageImpl(
        bool ambient_occlusion_bounce_approximation) noexcept
        : _ambient_occlusion_bounce_approximation{
              ambient_occlusion_bounce_approximation} {}

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
        auto &glossy_depth = sample.glossy_depth;
        auto &transmission_depth =
            sample.transmission_depth;
        const auto &forward_light_weight =
            config.light_transport
                .forward_light_weight;

        const auto world_visible =
            (scene->world_visibility_mask &
             traversal_ray_visibility) != 0u;
        Bool competing =
            (path_depth > 0u) &
            (!mis_competition_skipped);
        const auto spectral = sample.is_spectral();
        Float environment_weight = 1.0f;
        Float3 environment_radiance = make_float3(0.0f);
        std::optional<Float4> environment_radiance_spec;
        if (spectral) {
            environment_radiance_spec.emplace(make_float4(0.0f));
        }
        // Cycles rejects invisible background shaders before evaluating SVM,
        // and non-competing rays (primary / delta) always receive unit MIS
        // weight so environment PDF evaluation is not needed.
        $if(world_visible) {
            $if(competing) {
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
                            *config.background_sampling,
                            ray->origin(),
                            ray->direction(),
                            environment_selection_pdf,
                            invocation.parameters
                                .analytic_light_count,
                            invocation.parameters
                                .portal_count);
                environment_weight =
                    forward_light_weight(
                        previous_bsdf_pdf,
                        environment_pdf,
                        competing,
                        environment_pdf > 0.0f);
            };
            if (spectral) {
                if (scene->environment_emission_is_constant) {
                    *environment_radiance_spec =
                        _environment_light->evaluate_constant_emission_spectral(
                            sample);
                } else {
                    *environment_radiance_spec =
                        _environment_light->evaluate_emission_spectral(
                            sample, ray->origin(), ray->direction(), sample.ray_dD,
                            CyclesSvmBackgroundEvaluation::forward);
                }
            } else {
                if (scene->environment_emission_is_constant) {
                    environment_radiance =
                        _environment_light->evaluate_constant_emission(sample);
                } else {
                    environment_radiance = _environment_light->evaluate_emission(
                        sample, ray->origin(), ray->direction(), sample.ray_dD,
                        CyclesSvmBackgroundEvaluation::forward);
                }
            }
        };
        const auto commit_background = [&]() noexcept {
            if (_ambient_occlusion_bounce_approximation) {
                const auto ambient_occlusion_bounce =
                    cycles_path_state::ambient_occlusion_bounce(
                        path_depth,
                        transmission_depth,
                        glossy_depth,
                        invocation.parameters.ambient_occlusion_bounces);
                const auto ao_scale = select(
                    1.0f,
                    invocation.parameters.ambient_occlusion_factor,
                    ambient_occlusion_bounce);
                if (spectral) {
                    *environment_radiance_spec *= ao_scale;
                } else {
                    environment_radiance *= ao_scale;
                }
            }
            Float3 environment_contribution = make_float3(0.0f);
            if (spectral) {
                const auto spectral_contrib =
                    throughput *
                    cycles_spectrum::Spectrum::from_lanes(
                        *environment_radiance_spec) *
                    environment_weight;
                environment_contribution =
                    invocation.clamp_emission_contribution(
                        sample.spectrum_to_rgb(spectral_contrib), path_depth);
            } else {
                environment_contribution =
                    invocation
                        .clamp_emission_contribution(
                            throughput.rgb() *
                                environment_radiance *
                                environment_weight,
                            path_depth);
            }
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
                sample.accumulate_transparency(sample.throughput_average());
            }
            $else {
                sample.accumulate_radiance(
                    environment_contribution);
            };
            sample.accumulate_emission_or_background(
                environment_contribution,
                LightPassBuffer::environment);
        };
        if (sample.vt_miss_page != nullptr) {
            $if(*sample.vt_miss_page == vt_invalid_page) {
                commit_background();
            };
        } else {
            commit_background();
        }
    }
};

}// namespace

std::unique_ptr<BackgroundEventStage>
make_background_event_stage(
    bool ambient_occlusion_bounce_approximation) {
    return std::make_unique<
        BackgroundEventStageImpl>(
            ambient_occlusion_bounce_approximation);
}

}// namespace psycles::luisa_backend::detail

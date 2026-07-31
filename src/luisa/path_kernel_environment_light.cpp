#include "path_kernel_environment_light.h"

#include <psycles/luisa/background_sampling.h>
#include <psycles/luisa/cycles_path_state.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class PathEnvironmentLightComponent final
    : public EnvironmentLightComponent {

  private:
    [[nodiscard]] static
    background_sampling::BackgroundSample
    _sample_direction(
        const std::shared_ptr<
            LuisaSceneData> &scene,
        Float2 random) noexcept {
        return background_sampling::sample(
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
            std::move(random));
    }

    [[nodiscard]] static Float
    _direction_pdf(
        const std::shared_ptr<
            LuisaSceneData> &scene,
        Float3 direction) noexcept {
        return background_sampling::pdf(
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
            std::move(direction));
    }

  public:
    EnvironmentLightSample
    from_position(
        PathSampleContext &sample,
        Float3 reference,
        Float2 random,
        Float selection_pdf)
        const noexcept override {
        // Retain P in the semantic boundary so adding Cycles portal sampling
        // cannot silently create a second environment-light API.
        static_cast<void>(reference);
        const auto direction_sample =
            _sample_direction(
                sample.invocation
                    .config.scene,
                std::move(random));
        const auto pdf =
            direction_sample.pdf *
            selection_pdf;
        const auto radiance =
            sample.invocation
                .evaluate_environment(
                    direction_sample
                        .direction,
                    cycles_path_state::
                        light_emission_shader_state(
                            sample.path_depth,
                            sample.diffuse_depth,
                            sample.glossy_depth,
                            sample
                                .transparent_depth,
                            sample
                                .transmission_depth));
        return {
            .direction =
                direction_sample.direction,
            .radiance = radiance,
            .pdf = pdf,
            .valid = pdf > 0.0f};
    }

    Float from_direction(
        const std::shared_ptr<
            LuisaSceneData> &scene,
        Float3 reference,
        Float3 direction,
        Float selection_pdf)
        const noexcept override {
        static_cast<void>(reference);
        return _direction_pdf(
                   scene,
                   std::move(direction)) *
               selection_pdf;
    }
};

}// namespace

std::shared_ptr<
    const EnvironmentLightComponent>
make_environment_light_component() {
    return std::make_shared<
        PathEnvironmentLightComponent>();
}

}// namespace psycles::luisa_backend::detail

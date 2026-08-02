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
    EnvironmentLightProposal
    from_position(
        const std::shared_ptr<
            LuisaSceneData> &scene,
        Float3 reference,
        Float2 random,
        Float selection_pdf)
        const noexcept override {
        // Retain P in the semantic boundary so adding Cycles portal sampling
        // cannot silently create a second environment-light API.
        static_cast<void>(reference);
        const auto direction_sample =
            _sample_direction(
                scene,
                std::move(random));
        const auto pdf =
            direction_sample.pdf *
            selection_pdf;
        return {
            .direction =
                direction_sample.direction,
            .pdf = pdf,
            .valid = pdf > 0.0f};
    }

    Float3 evaluate_emission(
        PathSampleContext &sample,
        Float3 direction,
        const cycles_path_state::
            ShaderEvaluationState
                &shader_state)
        const noexcept override {
        return sample.invocation
            .evaluate_environment(
                std::move(direction),
                shader_state);
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

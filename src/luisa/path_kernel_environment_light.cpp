#include "path_kernel_environment_light.h"
#include "path_kernel_background_portal.h"

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
        Float3 reference,
        Float2 random,
        UInt portal_offset,
        UInt portal_count) noexcept {
        if (scene->background_portal_weight <= 0.0f) {
            return background_sampling::sample(
                scene->background_conditional_cdf,
                scene->background_marginal_cdf,
                scene->background_map_width,
                scene->background_map_height,
                scene->background_map_weight,
                scene->background_guided_sun_weight,
                make_float3(scene->background_guided_sun_axis),
                scene->background_guided_sun_radius,
                std::move(random));
        }

        const BackgroundPortalSampling portal_sampling;
        const auto possible_portals = portal_sampling.count_possible(
            scene->light_buffer,
            portal_offset,
            portal_count,
            reference);
        const auto portal_weight = select(
            0.0f,
            scene->background_portal_weight,
            possible_portals > 0u);
        const auto total_weight =
            portal_weight + scene->background_map_weight +
            scene->background_guided_sun_weight;
        const auto inverse_total = 1.0f / max(total_weight, 1.0e-20f);
        const auto portal_probability = portal_weight * inverse_total;
        const auto sun_probability =
            scene->background_guided_sun_weight * inverse_total;
        const auto map_probability = scene->background_map_weight * inverse_total;
        const auto sun_cdf = portal_probability + sun_probability;

        Float3 direction = make_float3(0.0f);
        Float result_pdf = 0.0f;
        $if(total_weight <= 0.0f) {
            direction = background_sampling::sample_uniform_sphere(random);
            result_pdf = background_sampling::uniform_sphere_pdf;
        }
        $elif(random.x < portal_probability) {
            const auto portal_random = make_float2(
                random.x / max(portal_probability, 1.0e-20f), random.y);
            const auto portal = portal_sampling.sample(
                scene->light_buffer,
                portal_offset,
                portal_count,
                reference,
                portal_random);
            direction = portal.direction;
            result_pdf = portal_probability * portal.pdf;
            if (scene->background_guided_sun_weight > 0.0f) {
                result_pdf += sun_probability * background_sampling::sun_pdf(
                    make_float3(scene->background_guided_sun_axis),
                    scene->background_guided_sun_radius,
                    direction);
            }
            if (scene->background_map_weight > 0.0f) {
                result_pdf += map_probability * background_sampling::map_pdf(
                    scene->background_conditional_cdf,
                    scene->background_marginal_cdf,
                    scene->background_map_width,
                    scene->background_map_height,
                    direction);
            }
            result_pdf = select(0.0f, result_pdf, portal.valid);
        }
        $elif(random.x < sun_cdf) {
            const auto sun_random = make_float2(
                (random.x - portal_probability) /
                    max(sun_probability, 1.0e-20f),
                random.y);
            const auto sun = background_sampling::sample_sun(
                make_float3(scene->background_guided_sun_axis),
                scene->background_guided_sun_radius,
                sun_random);
            direction = sun.direction;
            result_pdf = sun_probability * sun.pdf;
            result_pdf += portal_probability * portal_sampling.pdf(
                scene->light_buffer,
                portal_offset,
                portal_count,
                reference,
                direction);
            if (scene->background_map_weight > 0.0f) {
                result_pdf += map_probability * background_sampling::map_pdf(
                    scene->background_conditional_cdf,
                    scene->background_marginal_cdf,
                    scene->background_map_width,
                    scene->background_map_height,
                    direction);
            }
        }
        $else {
            const auto map_random = make_float2(
                (random.x - sun_cdf) / max(map_probability, 1.0e-20f),
                random.y);
            const auto map = background_sampling::sample_map(
                scene->background_conditional_cdf,
                scene->background_marginal_cdf,
                scene->background_map_width,
                scene->background_map_height,
                map_random);
            direction = map.direction;
            result_pdf = map_probability * map.pdf;
            result_pdf += portal_probability * portal_sampling.pdf(
                scene->light_buffer,
                portal_offset,
                portal_count,
                reference,
                direction);
            if (scene->background_guided_sun_weight > 0.0f) {
                result_pdf += sun_probability * background_sampling::sun_pdf(
                    make_float3(scene->background_guided_sun_axis),
                    scene->background_guided_sun_radius,
                    direction);
            }
        };
        return {.direction = direction, .pdf = result_pdf};
    }

    [[nodiscard]] static Float
    _direction_pdf(
        const std::shared_ptr<
            LuisaSceneData> &scene,
        Float3 reference,
        Float3 direction,
        UInt portal_offset,
        UInt portal_count) noexcept {
        if (scene->background_portal_weight <= 0.0f) {
            return background_sampling::pdf(
                scene->background_conditional_cdf,
                scene->background_marginal_cdf,
                scene->background_map_width,
                scene->background_map_height,
                scene->background_map_weight,
                scene->background_guided_sun_weight,
                make_float3(scene->background_guided_sun_axis),
                scene->background_guided_sun_radius,
                std::move(direction));
        }
        const BackgroundPortalSampling portal_sampling;
        const auto portal = portal_sampling.evaluate_pdf(
            scene->light_buffer,
            portal_offset,
            portal_count,
            reference,
            direction);
        const auto portal_weight = select(
            0.0f,
            scene->background_portal_weight,
            portal.possible);
        const auto total_weight =
            portal_weight + scene->background_map_weight +
            scene->background_guided_sun_weight;
        Float result = background_sampling::uniform_sphere_pdf;
        $if(total_weight > 0.0f) {
            const auto inverse_total = 1.0f / total_weight;
            result = portal_weight * inverse_total * portal.pdf;
            if (scene->background_guided_sun_weight > 0.0f) {
                result += scene->background_guided_sun_weight * inverse_total *
                          background_sampling::sun_pdf(
                              make_float3(scene->background_guided_sun_axis),
                              scene->background_guided_sun_radius,
                              direction);
            }
            if (scene->background_map_weight > 0.0f) {
                result += scene->background_map_weight * inverse_total *
                          background_sampling::map_pdf(
                              scene->background_conditional_cdf,
                              scene->background_marginal_cdf,
                              scene->background_map_width,
                              scene->background_map_height,
                              direction);
            }
        };
        return result;
    }

  public:
    EnvironmentLightProposal
    from_position(
        const std::shared_ptr<
            LuisaSceneData> &scene,
        Float3 reference,
        Float2 random,
        Float selection_pdf,
        UInt portal_offset,
        UInt portal_count)
        const noexcept override {
        const auto direction_sample =
            _sample_direction(
                scene,
                std::move(reference),
                std::move(random),
                std::move(portal_offset),
                std::move(portal_count));
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

    Float3 evaluate_constant_emission(
        PathSampleContext &sample)
        const noexcept override {
        return sample.invocation
            .constant_environment();
    }

    Float from_direction(
        const std::shared_ptr<
            LuisaSceneData> &scene,
        Float3 reference,
        Float3 direction,
        Float selection_pdf,
        UInt portal_offset,
        UInt portal_count)
        const noexcept override {
        return _direction_pdf(
                   scene,
                   std::move(reference),
                   std::move(direction),
                   std::move(portal_offset),
                   std::move(portal_count)) *
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

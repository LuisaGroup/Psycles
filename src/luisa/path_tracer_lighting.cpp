#include "path_tracer_lighting.h"

namespace psycles::luisa_backend::detail {

LightTransportCallables make_light_transport_callables(
    contract::DirectLightSampling mode) noexcept {
    SafeNormalizeCallable safe_normalize = [](
                                                  Float3 value,
                                                  Float3 fallback) noexcept {
        auto valid = dot(value, value) > 1.0e-20f;
        auto selected = select(fallback, value, valid);
        return normalize(select(
            make_float3(0.0f, 0.0f, 1.0f),
            selected,
            dot(selected, selected) > 1.0e-20f));
    };
    Callable<float(float, float)> power_heuristic = [](
                                                            Float sampled_pdf,
                                                            Float other_pdf) noexcept {
        auto sampled_squared = sampled_pdf * sampled_pdf;
        auto other_squared = other_pdf * other_pdf;
        return sampled_squared /
               max(
                   sampled_squared + other_squared,
                   1.0e-20f);
    };
    ForwardLightWeightCallable forward_light_weight =
        [=](Float forward_pdf,
            Float nee_pdf,
            Bool competing,
            Bool nee_available) noexcept {
            if (mode ==
                contract::DirectLightSampling::
                    forward_path_tracing) {
                return Float{1.0f};
            }
            if (mode ==
                contract::DirectLightSampling::
                    next_event_estimation) {
                return select(
                    1.0f,
                    0.0f,
                    competing & nee_available);
            }
            return select(
                1.0f,
                power_heuristic(
                    forward_pdf, nee_pdf),
                competing & nee_available);
        };
    NeeLightWeightCallable nee_light_weight =
        [=](Float nee_pdf,
            Float forward_pdf) noexcept {
            if (mode ==
                contract::DirectLightSampling::
                    next_event_estimation) {
                return Float{1.0f};
            }
            if (mode ==
                contract::DirectLightSampling::
                    forward_path_tracing) {
                return select(
                    0.0f,
                    1.0f,
                    forward_pdf <= 0.0f);
            }
            return power_heuristic(
                nee_pdf, forward_pdf);
        };
    ClampLightContributionCallable clamp_light_contribution =
        [](Float3 contribution,
           UInt depth,
           Float direct_limit,
           Float indirect_limit) noexcept {
            Float limit = select(
                direct_limit,
                indirect_limit,
                depth > 0u);
            Float magnitude =
                abs(contribution.x) +
                abs(contribution.y) +
                abs(contribution.z);
            Bool should_clamp =
                (limit > 0.0f) & (magnitude > limit);
            return select(
                contribution,
                contribution *
                    (limit / max(magnitude, 1.0e-20f)),
                should_clamp);
        };
    LightSampleRouletteCallable light_sample_roulette_weight =
        [](Float3 unshadowed_contribution,
           Float random,
           Float inverse_threshold) noexcept {
            Float maximum = max(
                abs(unshadowed_contribution.x),
                max(
                    abs(unshadowed_contribution.y),
                    abs(unshadowed_contribution.z)));
            Float probability =
                maximum * inverse_threshold;
            Bool roulette =
                (inverse_threshold > 0.0f) &
                (probability < 1.0f);
            Bool survives =
                (!roulette) | (random < probability);
            Float inverse_probability = select(
                1.0f,
                1.0f / max(probability, 1.0e-20f),
                roulette);
            return select(
                0.0f,
                inverse_probability,
                survives);
        };
    LightComponentRatioCallable light_component_ratio =
        [](Float3 numerator,
           Float3 denominator) noexcept {
            return make_float3(
                select(
                    0.0f,
                    numerator.x / denominator.x,
                    abs(denominator.x) > 1.0e-20f),
                select(
                    0.0f,
                    numerator.y / denominator.y,
                    abs(denominator.y) > 1.0e-20f),
                select(
                    0.0f,
                    numerator.z / denominator.z,
                    abs(denominator.z) > 1.0e-20f));
        };
    SplitScatteredLightCallable split_scattered_light =
        [](Float3 contribution,
           Float3 diffuse_weight,
           Float3 glossy_weight,
           Bool direct) noexcept {
            auto diffuse_contribution =
                contribution * diffuse_weight;
            auto glossy_contribution =
                contribution * glossy_weight;
            auto transmission_contribution =
                contribution -
                diffuse_contribution -
                glossy_contribution;
            Var<LightPassContributionCall> result;
            result.diffuse_direct = select(
                make_float3(0.0f),
                diffuse_contribution,
                direct);
            result.diffuse_indirect = select(
                diffuse_contribution,
                make_float3(0.0f),
                direct);
            result.glossy_direct = select(
                make_float3(0.0f),
                glossy_contribution,
                direct);
            result.glossy_indirect = select(
                glossy_contribution,
                make_float3(0.0f),
                direct);
            result.transmission_direct = select(
                make_float3(0.0f),
                transmission_contribution,
                direct);
            result.transmission_indirect = select(
                transmission_contribution,
                make_float3(0.0f),
                direct);
            return result;
        };
    SplitNeeLightCallable split_nee_light =
        [&light_component_ratio,
         &split_scattered_light](
            Float3 contribution,
            Float3 f,
            Float3 diffuse_f,
            Float3 path_diffuse_weight,
            Float3 path_glossy_weight,
            UInt depth) noexcept {
            auto local_diffuse_weight =
                light_component_ratio(diffuse_f, f);
            auto local_glossy_weight =
                light_component_ratio(f - diffuse_f, f);
            auto direct = depth == 0u;
            return split_scattered_light(
                contribution,
                select(
                    path_diffuse_weight,
                    local_diffuse_weight,
                    direct),
                select(
                    path_glossy_weight,
                    local_glossy_weight,
                    direct),
                direct);
        };
    return {
        std::move(safe_normalize),
        std::move(forward_light_weight),
        std::move(nee_light_weight),
        std::move(clamp_light_contribution),
        std::move(light_sample_roulette_weight),
        std::move(light_component_ratio),
        std::move(split_scattered_light),
        std::move(split_nee_light)};
}

}// namespace psycles::luisa_backend::detail

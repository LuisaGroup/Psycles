#include "path_tracer_lighting.h"

#include <psycles/luisa/cycles_film_light.h>

namespace psycles::luisa_backend::detail {

LightTransportCallables
make_light_transport_callables(contract::DirectLightSampling mode) noexcept {
  SafeNormalizeCallable safe_normalize = [](Float3 value,
                                                  Float3 fallback) noexcept {
        auto valid = dot(value, value) > 1.0e-20f;
        auto selected = select(fallback, value, valid);
    return normalize(select(make_float3(0.0f, 0.0f, 1.0f), selected,
            dot(selected, selected) > 1.0e-20f));
    };
  Callable<float(float, float)> power_heuristic = [](Float sampled_pdf,
                                                            Float other_pdf) noexcept {
        auto sampled_squared = sampled_pdf * sampled_pdf;
        auto other_squared = other_pdf * other_pdf;
    return sampled_squared / max(sampled_squared + other_squared, 1.0e-20f);
    };
    ForwardLightWeightCallable forward_light_weight =
      [=](Float forward_pdf, Float nee_pdf, Bool competing,
            Bool nee_available) noexcept {
        if (mode == contract::DirectLightSampling::forward_path_tracing) {
                return Float{1.0f};
            }
        if (mode == contract::DirectLightSampling::next_event_estimation) {
          return select(1.0f, 0.0f, competing & nee_available);
            }
        return select(1.0f, power_heuristic(forward_pdf, nee_pdf),
                competing & nee_available);
        };
  NeeLightWeightCallable nee_light_weight = [=](Float nee_pdf,
            Float forward_pdf) noexcept {
    if (mode == contract::DirectLightSampling::next_event_estimation) {
                return Float{1.0f};
            }
    if (mode == contract::DirectLightSampling::forward_path_tracing) {
      return select(0.0f, 1.0f, forward_pdf <= 0.0f);
            }
    return power_heuristic(nee_pdf, forward_pdf);
        };
    ClampLightContributionCallable clamp_light_contribution =
      [](Float3 contribution, UInt depth, Float direct_limit,
           Float indirect_limit) noexcept {
        return cycles_film_light::clamp(contribution, depth, direct_limit,
                indirect_limit);
        };
    LightSampleRouletteCallable light_sample_roulette_weight =
      [](Float3 unshadowed_contribution, Float random,
           Float inverse_threshold) noexcept {
        Float maximum = max(abs(unshadowed_contribution.x),
                            max(abs(unshadowed_contribution.y),
                    abs(unshadowed_contribution.z)));
        Float probability = maximum * inverse_threshold;
        Bool roulette = (inverse_threshold > 0.0f) & (probability < 1.0f);
        Bool survives = (!roulette) | (random < probability);
        Float inverse_probability =
            select(1.0f, 1.0f / max(probability, 1.0e-20f), roulette);
        return select(0.0f, inverse_probability, survives);
        };
    LightComponentRatioCallable light_component_ratio =
      [](Float3 numerator, Float3 denominator) noexcept {
        // Original bsdf_eval_pass_*_weight uses safe_divide: zero only.
        // A small but nonzero BSDF still owns its full lobe proportion.
        return make_float3(select(0.0f, numerator.x / denominator.x,
                    denominator.x != 0.0f),
                           select(0.0f, numerator.y / denominator.y,
                    denominator.y != 0.0f),
                           select(0.0f, numerator.z / denominator.z,
                    denominator.z != 0.0f));
        };
  return {std::move(safe_normalize),
        std::move(forward_light_weight),
        std::move(nee_light_weight),
        std::move(clamp_light_contribution),
        std::move(light_sample_roulette_weight),
        std::move(light_component_ratio)};
}

}// namespace psycles::luisa_backend::detail

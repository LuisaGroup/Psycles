#include <psycles/luisa/heterogeneous_volume_transmittance.h>

#include <type_traits>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

namespace {

constexpr auto log_one_tenth = -2.302585092994046f;

template <bool Spectral>
[[nodiscard]] auto evaluate_impl(
    const VolumeMajorantSegment &segment,
    Float minimum,
    Float maximum,
    const HeterogeneousVolumeTrackingRandomSource &random,
    const HeterogeneousVolumeCollisionProvider &collisions,
    UInt rng_offset) noexcept {
    using ColorT = std::conditional_t<Spectral, Float4, Float3>;
    const auto zero_color = []() noexcept -> ColorT {
        if constexpr (Spectral) { return make_float4(0.0f); } else { return make_float3(0.0f); }
    };
    const auto one_color = []() noexcept -> ColorT {
        if constexpr (Spectral) { return make_float4(1.0f); } else { return make_float3(1.0f); }
    };
    const auto read_sigma_t = [](const VolumeCoefficients &c) noexcept -> ColorT {
        if constexpr (Spectral) {
            return c.spectral_sigma_t.value_or(make_float4(c.sigma_t, 0.0f));
        } else {
            return c.sigma_t;
        }
    };
    const auto safe_exp = [](const ColorT &od) noexcept -> ColorT {
        if constexpr (Spectral) {
            return exp(-min(max(od, make_float4(0.0f)), make_float4(105.0f)));
        } else {
            return exp(-od);
        }
    };

    ColorT transmittance = one_color();
    Float probability_mass = 1.0f;
    UInt base_samples = 1u;
    UInt independent_estimators = 1u;
    UInt evaluations = 0u;

    const auto length = maximum - minimum;
    const auto coefficient_range =
        segment.sigma_maximum - segment.sigma_minimum;
    const auto active = segment.valid & (length > 0.0f);
    $if(active) {
        const auto expected_steps = coefficient_range * length;
        base_samples = clamp(
            cast<luisa::uint>(floor(expected_steps + 0.5f)),
            1u, heterogeneous_volume_maximum_steps);

        $if(coefficient_range != 0.0f) {
            const auto geometric_random = random.expansion_order(rng_offset);
            const auto order = min(
                cast<luisa::uint>(floor(log(geometric_random) / log_one_tenth)),
                4u);
            independent_estimators = 1u << order;
            const auto tail = pow(0.1f, cast<float>(order));
            probability_mass = select(0.9f * tail, tail, order == 4u);
        };

        const auto total_samples = base_samples * independent_estimators;
        const auto step_size = length / cast<float>(total_samples);
        const auto shade_offset = random.transmittance_shade_offset(rng_offset);

        $if(independent_estimators == 1u) {
            ColorT optical_depth = zero_color();
            UInt index = 0u;
            $while(index < base_samples) {
                const auto distance = min(
                    maximum,
                    minimum + (shade_offset + cast<float>(index)) * step_size);
                const auto coefficients =
                    collisions.evaluate(distance, false, nullptr);
                optical_depth += read_sigma_t(coefficients);
                evaluations += 1u;
                index += 1u;
            };
            $if(all(optical_depth == zero_color())) {
                transmittance = one_color();
            }
            $else {
                transmittance = safe_exp(optical_depth * step_size);
            };
        }
        $else {
            ColorT alternating_even = zero_color();
            ColorT alternating_odd = zero_color();
            ColorT full_optical_depth = zero_color();
            ColorT biased_sum = zero_color();
            UInt estimator = 0u;
            $while(estimator < independent_estimators) {
                ColorT estimator_optical_depth = zero_color();
                UInt index = 0u;
                $while(index < base_samples) {
                    const auto step = index * independent_estimators + estimator;
                    const auto distance = min(
                        maximum,
                        minimum + (shade_offset + cast<float>(step)) * step_size);
                    const auto coefficients =
                        collisions.evaluate(distance, false, nullptr);
                    const auto optical_depth =
                        read_sigma_t(coefficients) * step_size;
                    estimator_optical_depth +=
                        optical_depth * cast<float>(independent_estimators);
                    $if((step & 1u) == 0u) {
                        alternating_even += optical_depth * 2.0f;
                    }
                    $else {
                        alternating_odd += optical_depth * 2.0f;
                    };
                    full_optical_depth += optical_depth;
                    evaluations += 1u;
                    index += 1u;
                };
                biased_sum += safe_exp(estimator_optical_depth);
                estimator += 1u;
            };

            const auto biased =
                biased_sum / cast<float>(independent_estimators);
            const auto previous_order =
                0.5f * (safe_exp(alternating_even) + safe_exp(alternating_odd));
            const auto current_order = safe_exp(full_optical_depth);
            transmittance =
                biased + (current_order - previous_order) / probability_mass;
        };
    };

    if constexpr (Spectral) {
        return HeterogeneousVolumeTransmittanceResultSpectral{
            .transmittance = transmittance,
            .probability_mass = probability_mass,
            .base_samples = base_samples,
            .independent_estimators = independent_estimators,
            .evaluations = evaluations};
    } else {
        return HeterogeneousVolumeTransmittanceResult{
            .transmittance = transmittance,
            .probability_mass = probability_mass,
            .base_samples = base_samples,
            .independent_estimators = independent_estimators,
            .evaluations = evaluations};
    }
}

}// namespace

HeterogeneousVolumeTransmittanceResult
HeterogeneousVolumeTransmittance::evaluate(
    const VolumeMajorantSegment &segment,
    Float minimum,
    Float maximum,
    const HeterogeneousVolumeTrackingRandomSource
        &random,
    const HeterogeneousVolumeCollisionProvider
        &collisions,
    UInt rng_offset) const noexcept {
    return evaluate_impl<false>(
        segment, minimum, maximum, random, collisions, rng_offset);
}

HeterogeneousVolumeTransmittanceResultSpectral
HeterogeneousVolumeTransmittance::evaluate_spectral(
    const VolumeMajorantSegment &segment,
    Float minimum,
    Float maximum,
    const HeterogeneousVolumeTrackingRandomSource
        &random,
    const HeterogeneousVolumeCollisionProvider
        &collisions,
    UInt rng_offset) const noexcept {
    return evaluate_impl<true>(
        segment, minimum, maximum, random, collisions, rng_offset);
}

}// namespace psycles::luisa_backend

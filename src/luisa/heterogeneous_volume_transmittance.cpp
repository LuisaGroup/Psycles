#include <psycles/luisa/heterogeneous_volume_transmittance.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

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
    Float3 transmittance =
        make_float3(1.0f);
    Float probability_mass = 1.0f;
    UInt base_samples = 1u;
    UInt independent_estimators = 1u;
    UInt evaluations = 0u;

    const auto length =
        maximum - minimum;
    const auto coefficient_range =
        segment.sigma_maximum -
        segment.sigma_minimum;
    const auto active =
        segment.valid &
        (length > 0.0f);
    $if(active) {
        const auto expected_steps =
            coefficient_range * length;
        // expected_steps is non-negative. floor(x + 0.5) therefore matches
        // C roundf before Cycles clamps the biased order to [1, 1024].
        base_samples =
            clamp(
                cast<luisa::uint>(
                    floor(
                        expected_steps +
                        0.5f)),
                1u,
                heterogeneous_volume_maximum_steps);

        $if(coefficient_range != 0.0f) {
            const auto geometric_random =
                random.expansion_order(
                    rng_offset);
            const auto order =
                min(
                    cast<luisa::uint>(
                        floor(
                            log(geometric_random) /
                            log(0.1f))),
                    4u);
            independent_estimators =
                1u << order;
            const auto tail =
                pow(
                    0.1f,
                    cast<float>(order));
            probability_mass =
                select(
                    0.9f * tail,
                    tail,
                    order == 4u);
        };

        const auto total_samples =
            base_samples *
            independent_estimators;
        const auto step_size =
            length /
            cast<float>(
                total_samples);
        const auto shade_offset =
            random
                .transmittance_shade_offset(
                    rng_offset);

        $if(independent_estimators == 1u) {
            Float3 optical_depth =
                make_float3(0.0f);
            UInt index = 0u;
            $while(index < base_samples) {
                const auto distance =
                    min(
                        maximum,
                        minimum +
                            (shade_offset +
                             cast<float>(index)) *
                                step_size);
                const auto coefficients =
                    collisions.evaluate(
                        distance,
                        false,
                        nullptr);
                optical_depth +=
                    coefficients.sigma_t;
                evaluations += 1u;
                index += 1u;
            };
            $if(all(
                optical_depth ==
                make_float3(0.0f))) {
                transmittance =
                    make_float3(1.0f);
            }
            $else {
                transmittance =
                    exp(
                        -optical_depth *
                        step_size);
            };
        }
        $else {
            Float3 alternating_even =
                make_float3(0.0f);
            Float3 alternating_odd =
                make_float3(0.0f);
            Float3 full_optical_depth =
                make_float3(0.0f);
            Float3 biased_sum =
                make_float3(0.0f);
            UInt estimator = 0u;
            $while(estimator <
                   independent_estimators) {
                Float3 estimator_optical_depth =
                    make_float3(0.0f);
                UInt index = 0u;
                $while(index < base_samples) {
                    const auto step =
                        index *
                            independent_estimators +
                        estimator;
                    const auto distance =
                        min(
                            maximum,
                            minimum +
                                (shade_offset +
                                 cast<float>(step)) *
                                    step_size);
                    const auto coefficients =
                        collisions.evaluate(
                            distance,
                            false,
                            nullptr);
                    const auto optical_depth =
                        coefficients.sigma_t *
                        step_size;
                    estimator_optical_depth +=
                        optical_depth *
                        cast<float>(
                            independent_estimators);
                    $if((step & 1u) == 0u) {
                        alternating_even +=
                            optical_depth *
                            2.0f;
                    }
                    $else {
                        alternating_odd +=
                            optical_depth *
                            2.0f;
                    };
                    full_optical_depth +=
                        optical_depth;
                    evaluations += 1u;
                    index += 1u;
                };
                biased_sum +=
                    exp(
                        -estimator_optical_depth);
                estimator += 1u;
            };

            const auto biased =
                biased_sum /
                cast<float>(
                    independent_estimators);
            const auto previous_order =
                0.5f *
                (exp(-alternating_even) +
                 exp(-alternating_odd));
            const auto current_order =
                exp(
                    -full_optical_depth);
            transmittance =
                biased +
                (current_order -
                 previous_order) /
                    probability_mass;
        };
    };

    return {
        .transmittance =
            transmittance,
        .probability_mass =
            probability_mass,
        .base_samples =
            base_samples,
        .independent_estimators =
            independent_estimators,
        .evaluations =
            evaluations};
}

}// namespace psycles::luisa_backend

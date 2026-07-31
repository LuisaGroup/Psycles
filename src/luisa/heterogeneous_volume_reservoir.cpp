#include "heterogeneous_volume_reservoir.h"

#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

HeterogeneousVolumeReservoir::
    HeterogeneousVolumeReservoir(
        Float random) noexcept
    : _total_weight{0.0f},
      _random{std::move(random)},
      _candidate{
          .emission = make_float3(0.0f),
          .distance = 0.0f,
          .throughput = make_float3(0.0f),
          .distance_pdf = 0.0f,
          .valid = false} {}

Bool HeterogeneousVolumeReservoir::empty()
    const noexcept {
    return _total_weight == 0.0f;
}

Float HeterogeneousVolumeReservoir::total_weight()
    const noexcept {
    return _total_weight;
}

Float HeterogeneousVolumeReservoir::random()
    const noexcept {
    return _random;
}

HeterogeneousVolumeReservoirCandidate
HeterogeneousVolumeReservoir::candidate()
    const noexcept {
    return _candidate;
}

void HeterogeneousVolumeReservoir::set_random(
    Float random) noexcept {
    _random = std::move(random);
}

void HeterogeneousVolumeReservoir::add(
    Float weight,
    const HeterogeneousVolumeReservoirCandidate
        &candidate) noexcept {
    $if(weight > 0.0f) {
        const auto previous_weight =
            _total_weight;
        _total_weight += weight;
        const auto threshold =
            weight / _total_weight;
        const auto select_candidate =
            (_random <= threshold) |
            (previous_weight == 0.0f);
        $if(select_candidate) {
            _candidate = candidate;
            _candidate.valid = true;
            _random /= threshold;
        }
        $else {
            _random =
                (_random - threshold) /
                (1.0f - threshold);
        };
        _random =
            clamp(
                _random,
                0.0f,
                1.0f);
    };
}

HeterogeneousVolumeReservoirResult
HeterogeneousVolumeReservoir::finalize(
    Float3 transmitted_throughput,
    Float3 transmitted_emission,
    const HeterogeneousVolumeGuidingSample
        &guiding,
    Bool sample_direct_distance) noexcept {
    Float3 indirect_throughput =
        transmitted_throughput;
    Float3 emission =
        transmitted_emission;
    Float indirect_distance = 0.0f;
    Float3 direct_throughput =
        make_float3(0.0f);
    Float direct_distance = 0.0f;
    Float direct_distance_pdf = 0.0f;
    Float unguided_scatter_probability =
        0.0f;
    Float guided_scatter_probability =
        0.0f;
    Bool indirect_scatter = false;
    Bool direct_scatter = false;
    const auto was_empty = empty();

    $if(!was_empty) {
        const auto scatter_candidate =
            _candidate;
        unguided_scatter_probability =
            _total_weight;
        guided_scatter_probability =
            _total_weight;

        $if(sample_direct_distance) {
            direct_scatter = true;
            direct_distance =
                scatter_candidate.distance;
            direct_distance_pdf =
                scatter_candidate.distance_pdf;
            direct_throughput =
                scatter_candidate.throughput *
                select(
                    1.0f,
                    unguided_scatter_probability,
                    guiding.enabled);
        };

        $if(!guiding.enabled) {
            indirect_throughput =
                scatter_candidate.throughput;
            emission =
                scatter_candidate.emission;
            indirect_distance =
                scatter_candidate.distance;
            indirect_scatter = true;
        }
        $else {
            // VSPG guides only the event decision. Emission remains the
            // unguided mixture of transmitted and selected-scatter
            // estimators.
            emission =
                lerp(
                    transmitted_emission,
                    scatter_candidate.emission,
                    unguided_scatter_probability);

            const auto transmitted_zero =
                all(
                    transmitted_throughput ==
                    make_float3(0.0f));
            $if(transmitted_zero) {
                guided_scatter_probability =
                    1.0f;
            }
            $else {
                guided_scatter_probability =
                    lerp(
                        unguided_scatter_probability,
                        guiding
                            .scatter_probability,
                        0.75f);
                // Cycles repurposes total_weight as the guided scatter mass
                // before streaming the transmitted candidate. The two masses
                // then sum to one.
                _total_weight =
                    guided_scatter_probability;
                add(
                    1.0f -
                        guided_scatter_probability,
                    {.emission = emission,
                     .distance =
                         scatter_candidate
                             .distance,
                     .throughput =
                         transmitted_throughput,
                     .distance_pdf = 0.0f,
                     .valid = true});
            };

            indirect_scatter =
                _candidate.distance_pdf >
                0.0f;
            Float scale = 0.0f;
            $if(indirect_scatter) {
                scale =
                    unguided_scatter_probability /
                    guided_scatter_probability;
            }
            $else {
                scale =
                    (1.0f -
                     unguided_scatter_probability) /
                    (1.0f -
                     guided_scatter_probability);
            };
            indirect_throughput =
                _candidate.throughput *
                scale;
            indirect_distance =
                _candidate.distance;
        };
    };

    return {
        .indirect_throughput =
            indirect_throughput,
        .emission = emission,
        .indirect_distance =
            indirect_distance,
        .direct_throughput =
            direct_throughput,
        .direct_distance =
            direct_distance,
        .direct_distance_pdf =
            direct_distance_pdf,
        .unguided_scatter_probability =
            unguided_scatter_probability,
        .guided_scatter_probability =
            guided_scatter_probability,
        .random = _random,
        .indirect_scatter =
            indirect_scatter,
        .direct_scatter =
            direct_scatter,
        .empty = was_empty};
}

}// namespace psycles::luisa_backend::detail

#include <psycles/luisa/heterogeneous_volume_segment.h>
#include <psycles/luisa/heterogeneous_volume_transmittance.h>

#include <algorithm>
#include <limits>

#include "heterogeneous_volume_reservoir.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

[[nodiscard]] Float3 safe_divide(
    Float3 numerator,
    Float denominator) noexcept {
    const auto nonzero =
        denominator != 0.0f;
    return select(
        make_float3(0.0f),
        numerator /
            select(
                1.0f,
                denominator,
                nonzero),
        nonzero);
}

class HeterogeneousVolumeDirectObserver final
    : public HeterogeneousVolumeCandidateObserver {

  private:
    const HeterogeneousVolumeTrackingRandomSource
        &_random;
    const HeterogeneousVolumeCollisionProvider
        &_collisions;
    Float3 &_indirect_throughput;
    Float &_null_transmittance;
    VolumeDirectSamplingState _sampling;
    VolumeEquiangularCoefficients
        _equiangular;
    VolumeEquiangularSample
        _equiangular_sample;
    Float _absolute_distance;
    Float3 _throughput;
    Float _distance_pdf;
    Float _sampled_majorant;
    Float _sample_dt;
    Float _direct_rr_scale;
    Bool _scattered;
    HeterogeneousVolumeTransmittance
        _transmittance;

    [[nodiscard]] Bool _contains(
        const VolumeMajorantSegment &segment)
        const noexcept {
        return segment.valid &
               (_absolute_distance >=
                segment.minimum) &
               (_absolute_distance <=
                segment.maximum);
    }

  public:
    HeterogeneousVolumeDirectObserver(
        const HeterogeneousVolumeSegmentInput
            &input,
        Float3 &indirect_throughput,
        Float &null_transmittance) noexcept
        : _random{input.random},
          _collisions{input.collisions},
          _indirect_throughput{
              indirect_throughput},
          _null_transmittance{
              null_transmittance},
          _sampling{
              VolumeDirectSampling{}
                  .prepare(
                      input.direct
                          .requested_method,
                      input.direct_random,
                      input.direct.enabled &
                          !input.terminate)},
          _equiangular{
              .light_position =
                  input.direct
                      .light_position,
              .interval =
                  input.direct.interval},
          _equiangular_sample{
              VolumeDirectSampling{}
                  .sample_equiangular(
                      input.segment_origin,
                      input.phase_axis,
                      _equiangular,
                      _sampling.random)},
          _absolute_distance{
              input.ray_minimum +
              _equiangular_sample
                  .distance},
          _throughput{
              select(
                  make_float3(0.0f),
                  input.throughput,
                  _sampling.enabled)},
          _distance_pdf{0.0f},
          _sampled_majorant{0.0f},
          _sample_dt{
              std::numeric_limits<
                  float>::max()},
          _direct_rr_scale{1.0f},
          _scattered{false} {}

    void enter_segment(
        const VolumeMajorantSegment &segment,
        UInt rng_offset)
        noexcept override {
        const auto contains =
            _contains(segment);
        const auto eligible =
            (_sampling.method ==
             volume_sample_equiangular) &
            !_sampling.use_mis &
            !_scattered &
            segment.valid;
        $if(eligible) {
            const auto maximum =
                select(
                    segment.maximum,
                    _absolute_distance,
                    contains);
            const auto estimate =
                _transmittance.evaluate(
                    segment,
                    segment.minimum,
                    maximum,
                    _random,
                    _collisions,
                    rng_offset);
            _throughput *=
                estimate.transmittance;
            _scattered |= contains;
        };
    }

    void advance_candidate(
        const VolumeMajorantSegment &segment,
        Float previous_distance,
        Float proposed_distance,
        Float sampled_majorant,
        UInt rng_offset)
        noexcept override {
        static_cast<void>(rng_offset);
        const auto eligible =
            (_sampling.method ==
             volume_sample_equiangular) &
            _sampling.use_mis &
            !_scattered &
            _contains(segment) &
            (proposed_distance >
             _absolute_distance);
        $if(eligible) {
            _scattered = true;
            _throughput =
                _indirect_throughput *
                _null_transmittance *
                _direct_rr_scale;
            _sample_dt =
                _absolute_distance -
                previous_distance;
            _distance_pdf =
                _null_transmittance *
                sampled_majorant;
            _sampled_majorant =
                sampled_majorant;
        };
    }

    [[nodiscard]] VolumeDirectSamplingState
    sampling() const noexcept {
        return _sampling;
    }

    [[nodiscard]] Float relative_distance()
        const noexcept {
        return _equiangular_sample.distance;
    }

    [[nodiscard]] Float absolute_distance()
        const noexcept {
        return _absolute_distance;
    }

    [[nodiscard]] Float equiangular_pdf()
        const noexcept {
        return _equiangular_sample.pdf;
    }

    [[nodiscard]] Float3 throughput()
        const noexcept {
        return _throughput;
    }

    [[nodiscard]] Float distance_pdf()
        const noexcept {
        return _distance_pdf;
    }

    [[nodiscard]] Float sampled_majorant()
        const noexcept {
        return _sampled_majorant;
    }

    [[nodiscard]] Float sample_dt()
        const noexcept {
        return _sample_dt;
    }

    [[nodiscard]] Bool scattered()
        const noexcept {
        return _scattered;
    }

    [[nodiscard]] Bool coupled_pending()
        const noexcept {
        return
            (_sampling.method ==
             volume_sample_equiangular) &
            _sampling.use_mis &
            !_scattered;
    }

    void mark_distance_scatter() noexcept {
        _scattered |=
            _sampling.method ==
            volume_sample_distance;
    }

    void terminate_throughput(
        Bool clear,
        Bool complete) noexcept {
        _throughput =
            select(
                _throughput,
                make_float3(0.0f),
                clear);
        _scattered |= complete;
    }

    void scale_direct_roulette(
        Float probability) noexcept {
        _direct_rr_scale /=
            probability;
    }
};

class HeterogeneousVolumeSegmentComponentImpl final
    : public HeterogeneousVolumeSegmentComponent {

  private:
    std::size_t _closure_allocation_budget;
    HeterogeneousVolumeTracking _tracking;

  public:
    explicit HeterogeneousVolumeSegmentComponentImpl(
        std::size_t closure_allocation_budget) noexcept
        : _closure_allocation_budget{
              std::max(
                  closure_allocation_budget,
                  std::size_t{1u})} {}

    HeterogeneousVolumeSegmentResult
    emit(
        const HeterogeneousVolumeSegmentInput
            &input)
        const noexcept override {
        Float3 current_throughput =
            input.throughput;
        Float3 accumulated_emission =
            make_float3(0.0f);
        Float null_transmittance = 1.0f;
        detail::HeterogeneousVolumeReservoir
            reservoir{
                input.reservoir_random};
        HeterogeneousVolumeDirectObserver
            direct{
                input,
                current_throughput,
                null_transmittance};
        HeterogeneousVolumeCandidateWalk walk{
            input.segments,
            input.random,
            input.guiding.majorant_scale,
            input.ray_minimum,
            input.tracking_rng_offset,
            &direct};
        Bool indirect_selected = false;
        Bool traversal_exhausted = false;
        Bool step_limit_exceeded = false;
        Bool majorant_exceeded = false;
        Bool active = false;
        Float optical_depth = 0.0f;
        UInt next_tracking_rng_offset =
            input.tracking_rng_offset;
        UInt steps = 0u;
        Bool integrating = true;

        $while(integrating) {
            const auto candidate =
                walk.advance();
            optical_depth =
                candidate.optical_depth;
            next_tracking_rng_offset =
                candidate
                    .next_rng_offset;
            steps = candidate.step;
            traversal_exhausted |=
                candidate
                    .traversal_exhausted;
            step_limit_exceeded |=
                candidate
                    .step_limit_exceeded;
            active |= candidate.candidate;

            $if(candidate.candidate) {
                Bool roulette_returned =
                    direct.scattered() &
                    indirect_selected;
                const auto threshold =
                    max(
                        abs(current_throughput.x),
                        max(
                            abs(current_throughput.y),
                            abs(current_throughput.z)));
                $if(!roulette_returned &
                    (threshold <= 0.05f)) {
                    const auto equiangular =
                        direct.coupled_pending();
                    const auto has_scatter_samples =
                        !reservoir.empty() &
                        !equiangular;
                    $if(input.terminate |
                        has_scatter_samples) {
                        $if(reservoir.random() >
                            threshold) {
                            current_throughput =
                                make_float3(0.0f);
                            const auto direct_depends =
                                equiangular |
                                (direct.sampling()
                                     .method ==
                                 volume_sample_distance);
                            direct
                                .terminate_throughput(
                                    direct_depends,
                                    false);
                            roulette_returned = true;
                        }
                        $else {
                            reservoir.set_random(
                                clamp(
                                    reservoir.random() /
                                        threshold,
                                    0.0f,
                                    1.0f));
                            current_throughput /=
                                threshold;
                        };
                    };

                    $if(!roulette_returned &
                        equiangular) {
                        $if(reservoir.random() >
                            threshold) {
                            direct
                                .terminate_throughput(
                                    true,
                                    true);
                            reservoir.set_random(
                                (reservoir.random() -
                                 threshold) /
                                (1.0f -
                                 threshold));
                        }
                        $else {
                            reservoir.set_random(
                                reservoir.random() /
                                threshold);
                            direct
                                .scale_direct_roulette(
                                    threshold);
                        };
                        reservoir.set_random(
                            clamp(
                                reservoir.random(),
                                0.0f,
                                1.0f));
                    };
                };

                $if(!roulette_returned) {
                    auto coefficients =
                        input.collisions.evaluate(
                            candidate.distance,
                            true,
                            nullptr);
                    // PATH_RAY_TERMINATE causes Cycles closure allocation to
                    // omit scattering while retaining absorption/emission.
                    coefficients.sigma_s =
                        select(
                            coefficients.sigma_s,
                            make_float3(0.0f),
                            input.terminate);
                    coefficients.has_scatter =
                        coefficients
                            .has_scatter &
                        !input.terminate;
                    const auto collision =
                        _tracking
                            .evaluate_collision(
                                coefficients,
                                candidate
                                    .sampled_majorant,
                                candidate
                                    .step_distance,
                                current_throughput,
                                null_transmittance);
                    majorant_exceeded |=
                        collision.null_event
                            .majorant_exceeded;
                    accumulated_emission +=
                        collision.emission;

                    $if(!collision
                             .absorption_only) {
                        const auto weight =
                            null_transmittance *
                            collision
                                .scatter_probability;
                        const detail::
                            HeterogeneousVolumeReservoirCandidate
                            sample{
                                .emission =
                                    accumulated_emission,
                                .distance =
                                    candidate.distance,
                                .throughput =
                                    collision
                                        .scatter_throughput,
                                .distance_pdf =
                                    weight *
                                    candidate
                                        .sampled_majorant,
                                .valid = true};
                        $if(input.guiding.enabled) {
                            reservoir.add(
                                weight,
                                sample);
                        }
                        $elif(!indirect_selected) {
                            const auto selection =
                                _tracking
                                    .select_event(
                                        reservoir.random(),
                                        collision
                                            .scatter_probability);
                            reservoir.set_random(
                                selection.random);
                            $if(selection.scattered) {
                                reservoir.add(
                                    weight,
                                    sample);
                                indirect_selected =
                                    true;
                                direct
                                    .mark_distance_scatter();
                            };
                        };
                    };
                    current_throughput =
                        collision.null_throughput;
                    null_transmittance =
                        collision
                            .null_transmittance;
                };

                const auto both_zero =
                    all(
                        current_throughput ==
                        make_float3(0.0f)) &
                    all(
                        direct.throughput() ==
                        make_float3(0.0f));
                const auto both_scattered =
                    direct.scattered() &
                    indirect_selected;
                integrating =
                    !(both_zero |
                      both_scattered);
            }
            $else {
                integrating = false;
            };
        };

        const auto sampling =
            direct.sampling();
        const auto distance_direct =
            !input.terminate &
            (sampling.method ==
             volume_sample_distance);
        const auto reservoir_result =
            reservoir.finalize(
                current_throughput,
                accumulated_emission,
                input.guiding,
                distance_direct);
        Float3 final_throughput =
            reservoir_result
                .indirect_throughput;
        Float3 final_emission =
            reservoir_result.emission;
        Float final_distance =
            select(
                input.ray_maximum,
                reservoir_result
                    .indirect_distance,
                reservoir_result
                    .indirect_scatter);
        const auto selected_scatter =
            reservoir_result
                .indirect_scatter;
        VolumeCoefficients
            selected_coefficients =
                VolumeCoefficients::zero();
        VolumePhaseSetSample phase{
            .direction =
                input.phase_axis,
            .pdf = 0.0f,
            .sampled_roughness = 1.0f,
            .selection_rescaled =
                input.phase_random.x,
            .closure_index = 0u,
            .closure_type = 0u,
            .valid = false};
        $if(selected_scatter) {
            VolumePhaseSet phases{
                _closure_allocation_budget};
            selected_coefficients =
                input.collisions.evaluate(
                    final_distance,
                    true,
                    &phases);
            phase =
                phases.sample(
                    input.phase_axis,
                    input.phase_random);
        };
        const auto scattered =
            selected_scatter &
            phase.valid;

        Float3 direct_throughput =
            select(
                direct.throughput(),
                reservoir_result
                    .direct_throughput,
                distance_direct);
        Float direct_distance =
            select(
                direct.absolute_distance(),
                reservoir_result
                    .direct_distance,
                distance_direct);
        Float direct_distance_pdf =
            select(
                direct.distance_pdf(),
                reservoir_result
                    .direct_distance_pdf,
                distance_direct);
        Float direct_equiangular_pdf =
            direct.equiangular_pdf();
        Bool direct_scatter =
            select(
                direct.scattered(),
                reservoir_result
                    .direct_scatter,
                distance_direct);
        VolumePhaseSetEvaluation direct_phase{
            .value = 0.0f,
            .pdf = 0.0f,
            .sample_weight = 0.0f,
            .valid = false};
        Float direct_mis_weight = 0.0f;

        $if(direct_scatter) {
            VolumePhaseSet direct_phases{
                _closure_allocation_budget};
            auto direct_coefficients =
                input.collisions.evaluate(
                    direct_distance,
                    true,
                    &direct_phases);
            const auto equiangular =
                sampling.method ==
                volume_sample_equiangular;
            $if(equiangular) {
                const auto has_scatter =
                    direct_coefficients
                        .has_scatter &
                    any(
                        direct_coefficients
                                .sigma_s !=
                            make_float3(0.0f));
                $if(has_scatter) {
                    $if(sampling.use_mis) {
                        const auto null_event =
                            _tracking
                                .null_event_coefficients(
                                    direct_coefficients
                                        .sigma_t,
                                    direct
                                        .sampled_majorant());
                        $if((direct.sample_dt() !=
                             std::numeric_limits<
                                 float>::max()) &
                            (null_event.majorant !=
                             direct
                                 .sampled_majorant())) {
                            direct_throughput *=
                                exp(
                                    (direct
                                         .sampled_majorant() -
                                     null_event.majorant) *
                                    direct.sample_dt());
                        };
                        const auto scatter_measure =
                            _tracking
                                .scatter_measure(
                                    direct_coefficients,
                                    null_event.sigma_n,
                                    direct_throughput);
                        direct_distance_pdf *=
                            scatter_measure
                                .probability;
                    };
                    direct_throughput *=
                        safe_divide(
                            direct_coefficients
                                .sigma_s,
                            direct_equiangular_pdf);
                }
                $else {
                    direct_scatter = false;
                };
            };

            const VolumeDirectSampling
                direct_sampling;
            $if(distance_direct) {
                direct_equiangular_pdf =
                    direct_sampling
                        .equiangular_pdf(
                            input.segment_origin,
                            input.phase_axis,
                            {.light_position =
                                 input.direct
                                     .light_position,
                             .interval =
                                 input.direct
                                     .interval},
                            direct_distance -
                                input
                                    .ray_minimum);
            };
            direct_mis_weight = 1.0f;
            $if(sampling.use_mis) {
                const auto selected_pdf =
                    select(
                        direct_distance_pdf,
                        direct_equiangular_pdf,
                        equiangular);
                const auto competing_pdf =
                    select(
                        direct_equiangular_pdf,
                        direct_distance_pdf,
                        equiangular);
                direct_mis_weight =
                    2.0f *
                    direct_sampling
                        .power_heuristic(
                            selected_pdf,
                            competing_pdf);
                direct_throughput *=
                    direct_mis_weight;
            };

            VolumeDirectDirectionSample
                direction{
                    .direction =
                        make_float3(0.0f),
                    .valid = false};
            if (input.direct_light !=
                nullptr) {
                direction =
                    input.direct_light
                        ->sample_direction(
                            direct_distance -
                            input.ray_minimum);
                input.direct_light
                    ->evaluate_constant_emission();
            }
            const auto phase_evaluation =
                direct_phases.evaluate(
                    input.phase_axis,
                    direction.direction);
            direct_phase = {
                .value = select(
                    0.0f,
                    phase_evaluation.value,
                    direction.valid),
                .pdf = select(
                    0.0f,
                    phase_evaluation.pdf,
                    direction.valid),
                .sample_weight = select(
                    0.0f,
                    phase_evaluation
                        .sample_weight,
                    direction.valid),
                .valid =
                    phase_evaluation.valid &
                    direction.valid};
            if (input.direct_light !=
                nullptr) {
                input.direct_light
                    ->evaluate_deferred_emission(
                        direct_phase.valid &
                        (direct_phase.value !=
                         0.0f));
            }
        };

        return {
            .coefficients =
                selected_coefficients,
            .transport =
                {.throughput =
                     final_throughput,
                 .emission =
                     final_emission,
                 .distance =
                     final_distance,
                 .null_transmittance =
                     null_transmittance,
                 .reservoir_random =
                     reservoir_result.random,
                 .optical_depth =
                     optical_depth,
                 .unguided_scatter_probability =
                     reservoir_result
                         .unguided_scatter_probability,
                 .guided_scatter_probability =
                     reservoir_result
                         .guided_scatter_probability,
                 .majorant_scale =
                     input.guiding
                         .majorant_scale,
                 .next_tracking_rng_offset =
                     next_tracking_rng_offset,
                 .steps = steps,
                 .selected_scatter =
                     selected_scatter,
                 .traversal_exhausted =
                     traversal_exhausted,
                 .step_limit_exceeded =
                     step_limit_exceeded,
                 .majorant_exceeded =
                     majorant_exceeded,
                 .active = active},
            .phase = phase,
            .direct_transport =
                {.throughput =
                     direct_throughput,
                 .distance =
                     direct_distance,
                 .distance_pdf =
                     direct_distance_pdf,
                 .equiangular_pdf =
                     direct_equiangular_pdf,
                 .mis_weight =
                     direct_mis_weight,
                 .sample_method =
                     sampling.method,
                 .use_mis =
                     sampling.use_mis,
                 .scattered =
                     direct_scatter},
            .direct_phase =
                direct_phase,
            .scattered = scattered,
            .phase_failed =
                selected_scatter &
                !phase.valid};
    }
};

}// namespace

HeterogeneousVolumeSegmentResult
HeterogeneousVolumeSegmentComponent::emit(
    VolumeMajorantSegmentSequence &segments,
    const HeterogeneousVolumeTrackingRandomSource
        &random,
    const HeterogeneousVolumeCollisionProvider
        &collisions,
    Float majorant_scale,
    Float ray_minimum,
    Float ray_maximum,
    Float3 phase_axis,
    Float3 throughput,
    Float reservoir_random,
    Float2 phase_random,
    UInt tracking_rng_offset,
    Bool terminate) const noexcept {
    return emit(
        {.segments = segments,
         .random = random,
         .collisions = collisions,
         .guiding =
             {.scatter_probability = 1.0f,
              .majorant_scale =
                  majorant_scale,
              .enabled = false},
         .direct =
             {.requested_method =
                  volume_sample_none,
              .light_position =
                  make_float3(0.0f),
              .interval =
                  {.minimum = 0.0f,
                   .maximum = 0.0f},
              .enabled = false},
         .direct_light = nullptr,
         .ray_minimum = ray_minimum,
         .ray_maximum = ray_maximum,
         .segment_origin =
             make_float3(0.0f),
         .phase_axis = phase_axis,
         .throughput = throughput,
         .direct_random = 0.0f,
         .reservoir_random =
             reservoir_random,
         .phase_random = phase_random,
         .tracking_rng_offset =
             tracking_rng_offset,
         .terminate = terminate});
}

std::unique_ptr<HeterogeneousVolumeSegmentComponent>
make_heterogeneous_volume_segment_component(
    std::size_t closure_allocation_budget) {
    return std::make_unique<
        HeterogeneousVolumeSegmentComponentImpl>(
        closure_allocation_budget);
}

}// namespace psycles::luisa_backend

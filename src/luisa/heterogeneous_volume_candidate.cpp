#include <psycles/luisa/heterogeneous_volume_candidate.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

using namespace luisa::compute;

HeterogeneousVolumeCandidateWalk::
    HeterogeneousVolumeCandidateWalk(
        VolumeMajorantSegmentSequence &segments,
        const HeterogeneousVolumeTrackingRandomSource
            &random,
        Float majorant_scale,
        Float ray_minimum,
        UInt tracking_rng_offset,
        HeterogeneousVolumeCandidateObserver
            *observer) noexcept
    : _segments{segments},
      _random{random},
      _majorant_scale{majorant_scale},
      _distance{ray_minimum},
      _optical_depth{0.0f},
      _rng_offset{tracking_rng_offset},
      _step{0u},
      _finished{false},
      _observer{observer} {
    const auto initial = _segments.current();
    _optical_depth =
        select(
            0.0f,
            initial.sigma_maximum *
                (initial.maximum -
                 initial.minimum),
            initial.valid);
    _finished = !initial.valid;
    if (_observer != nullptr) {
        _observer->enter_segment(
            initial,
            _rng_offset);
    }
}

HeterogeneousVolumeCandidate
HeterogeneousVolumeCandidateWalk::advance()
    noexcept {
    Float candidate_minimum = 0.0f;
    Float candidate_maximum = 0.0f;
    Float candidate_sigma_minimum = 0.0f;
    Float candidate_sigma_maximum = 0.0f;
    UInt candidate_object = 0u;
    UInt candidate_shader = 0u;
    UInt candidate_node = 0u;
    Bool candidate_no_overlap = false;
    Bool candidate_lookup_complete = false;
    Float step_distance = 0.0f;
    Float sampled_majorant = 0.0f;
    Float scatter_random = 0.0f;
    const auto sample_rng_offset =
        _rng_offset;
    Bool candidate = false;
    Bool traversal_exhausted =
        _finished;
    Bool step_limit_exceeded = false;

    $if(!_finished) {
        const auto exceeds_step_limit =
            _step >
            heterogeneous_volume_maximum_steps;
        // Cycles uses `if (step++ > VOLUME_MAX_STEPS)`: the failed call is
        // observable as step 1026 after 1025 accepted candidate attempts.
        _step += 1u;
        $if(exceeds_step_limit) {
            _finished = true;
            step_limit_exceeded = true;
        }
        $else {
            scatter_random =
                _random.scatter_distance(
                    _rng_offset);
            Bool searching = true;
            $while(searching) {
                const auto segment =
                    _segments.current();
                sampled_majorant =
                    segment.sigma_maximum *
                    _majorant_scale;
                const auto residual_optical_depth =
                    (segment.maximum -
                     _distance) *
                    sampled_majorant;
                Bool beyond_segment =
                    sampled_majorant == 0.0f;
                $if(sampled_majorant != 0.0f) {
                    step_distance =
                        _tracking
                            .candidate_distance(
                                scatter_random,
                                sampled_majorant);
                    if (_observer != nullptr) {
                        _observer
                            ->advance_candidate(
                                segment,
                                _distance,
                                _distance +
                                    step_distance,
                                sampled_majorant,
                                _rng_offset);
                    }
                    _distance +=
                        step_distance;
                    beyond_segment =
                        _distance >
                        segment.maximum;
                };

                $if(beyond_segment) {
                    const auto advanced =
                        _segments.advance(
                            _random.shade_offset(
                                _rng_offset));
                    $if(advanced) {
                        const auto next =
                            _segments.current();
                        if (_observer != nullptr) {
                            _observer
                                ->enter_segment(
                                    next,
                                    _rng_offset);
                        }
                        _optical_depth +=
                            next.sigma_maximum *
                            (next.maximum -
                             next.minimum);
                        _distance =
                            next.minimum;
                        // Reuse the same uniform sample after subtracting
                        // the optical depth already consumed in this voxel.
                        scatter_random =
                            clamp(
                                1.0f -
                                    (1.0f -
                                     scatter_random) *
                                        exp(
                                            residual_optical_depth),
                                0.0f,
                                1.0f);
                    }
                    $else {
                        _finished = true;
                        traversal_exhausted = true;
                        searching = false;
                    };
                }
                $else {
                    candidate_minimum =
                        segment.minimum;
                    candidate_maximum =
                        segment.maximum;
                    candidate_sigma_minimum =
                        segment.sigma_minimum;
                    candidate_sigma_maximum =
                        segment.sigma_maximum;
                    candidate_object =
                        segment.object;
                    candidate_shader =
                        segment.shader;
                    candidate_node =
                        segment.node;
                    candidate_no_overlap =
                        segment.no_overlap;
                    candidate_lookup_complete =
                        segment.lookup_complete;
                    candidate = true;
                    searching = false;
                    _rng_offset +=
                        heterogeneous_volume_tracking_rng_stride;
                };
            };
        };
    };

    return {
        .segment =
            {.minimum = candidate_minimum,
             .maximum = candidate_maximum,
             .sigma_minimum =
                 candidate_sigma_minimum,
             .sigma_maximum =
                 candidate_sigma_maximum,
             .object = candidate_object,
             .shader = candidate_shader,
             .node = candidate_node,
             .valid = candidate,
             .no_overlap =
                 candidate_no_overlap,
             .lookup_complete =
                 candidate_lookup_complete},
        .distance = _distance,
        .step_distance = step_distance,
        .sampled_majorant =
            sampled_majorant,
        .scatter_random =
            scatter_random,
        .optical_depth =
            _optical_depth,
        .sample_rng_offset =
            sample_rng_offset,
        .next_rng_offset =
            _rng_offset,
        .step = _step,
        .candidate = candidate,
        .traversal_exhausted =
            traversal_exhausted,
        .step_limit_exceeded =
            step_limit_exceeded};
}

}// namespace psycles::luisa_backend

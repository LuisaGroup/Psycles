#include <psycles/luisa/heterogeneous_volume_segment.h>

#include <algorithm>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

class StackedHeterogeneousVolumeCollisionProvider final
    : public HeterogeneousVolumeCollisionProvider {

  private:
    const SurfaceDispatch &_surfaces;
    std::shared_ptr<
        const VolumeStackEntryPointProvider>
        _points;
    const VolumeStack &_stack;
    const ShaderServices &_services;
    const VolumeShadingState &_base_state;
    Float3 _ray_origin;
    Float3 _ray_direction;

  public:
    StackedHeterogeneousVolumeCollisionProvider(
        const SurfaceDispatch &surfaces,
        std::shared_ptr<
            const VolumeStackEntryPointProvider>
            points,
        const VolumeStack &stack,
        const ShaderServices &services,
        const VolumeShadingState &base_state,
        Float3 ray_origin,
        Float3 ray_direction) noexcept
        : _surfaces{surfaces},
          _points{std::move(points)},
          _stack{stack},
          _services{services},
          _base_state{base_state},
          _ray_origin{std::move(ray_origin)},
          _ray_direction{
              std::move(ray_direction)} {}

    VolumeCoefficients evaluate(
        Float distance,
        Bool evaluate_emission,
        VolumePhaseSet *phases)
        const noexcept override {
        const VolumeShadingState state{
            .position =
                _ray_origin +
                _ray_direction * distance,
            .incoming =
                _base_state.incoming,
            .ray_visibility =
                _base_state.ray_visibility,
            .ray_events =
                _base_state.ray_events,
            .ray_depth =
                _base_state.ray_depth,
            .diffuse_depth =
                _base_state.diffuse_depth,
            .glossy_depth =
                _base_state.glossy_depth,
            .transparent_depth =
                _base_state
                    .transparent_depth,
            .transmission_depth =
                _base_state
                    .transmission_depth,
            // shader_setup_from_volume() deliberately initializes this to
            // zero in current Cycles.
            .ray_length =
                _base_state.ray_length,
            .time = _base_state.time};
        const StackedVolumeEvaluator evaluator{
            _surfaces, *_points};
        return evaluator.evaluate(
            _stack,
            _services,
            state,
            evaluate_emission,
            phases);
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
        Bool terminate)
        const noexcept override {
        HeterogeneousVolumeCandidateWalk walk{
            segments,
            random,
            majorant_scale,
            ray_minimum,
            tracking_rng_offset};
        Float3 current_throughput =
            throughput;
        Float3 accumulated_emission =
            make_float3(0.0f);
        Float null_transmittance = 1.0f;
        Float selected_distance =
            ray_maximum;
        Float3 selected_throughput =
            make_float3(0.0f);
        Float3 selected_emission =
            make_float3(0.0f);
        Bool selected_scatter = false;
        Bool traversal_exhausted = false;
        Bool step_limit_exceeded = false;
        Bool majorant_exceeded = false;
        Bool active = false;
        Float optical_depth = 0.0f;
        UInt next_tracking_rng_offset =
            tracking_rng_offset;
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
                // Cycles performs this roulette after free flight has
                // consumed a candidate and before evaluating its closure.
                const auto threshold =
                    max(
                        abs(current_throughput.x),
                        max(
                            abs(current_throughput.y),
                            abs(current_throughput.z)));
                const auto roulette_domain =
                    terminate |
                    selected_scatter;
                Bool roulette_stopped = false;
                $if(roulette_domain &
                    (threshold <= 0.05f)) {
                    $if(reservoir_random >
                        threshold) {
                        current_throughput =
                            make_float3(0.0f);
                        roulette_stopped = true;
                    }
                    $else {
                        reservoir_random =
                            clamp(
                                reservoir_random /
                                    threshold,
                                0.0f,
                                1.0f);
                        current_throughput /=
                            threshold;
                    };
                };

                $if(roulette_stopped) {
                    integrating = false;
                }
                $else {
                    auto coefficients =
                        collisions.evaluate(
                            candidate.distance,
                            true,
                            nullptr);
                    // PATH_RAY_TERMINATE causes Cycles closure allocation to
                    // omit scattering while retaining absorption/emission.
                    coefficients.sigma_s =
                        select(
                            coefficients.sigma_s,
                            make_float3(0.0f),
                            terminate);
                    coefficients.has_scatter =
                        coefficients
                            .has_scatter &
                        !terminate;
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
                        $if(!selected_scatter) {
                            const auto selection =
                                _tracking
                                    .select_event(
                                        reservoir_random,
                                        collision
                                            .scatter_probability);
                            reservoir_random =
                                selection.random;
                            $if(selection.scattered) {
                                selected_scatter = true;
                                selected_distance =
                                    candidate.distance;
                                selected_throughput =
                                    collision
                                        .scatter_throughput;
                                selected_emission =
                                    accumulated_emission;
                            };
                        };
                    };
                    current_throughput =
                        collision.null_throughput;
                    null_transmittance =
                        collision
                            .null_transmittance;
                    const auto zero_throughput =
                        all(
                            current_throughput ==
                            make_float3(0.0f));
                    integrating =
                        !zero_throughput;
                };
            }
            $else {
                integrating = false;
            };
        };

        Float3 final_throughput =
            current_throughput;
        Float3 final_emission =
            accumulated_emission;
        Float final_distance =
            ray_maximum;
        VolumeCoefficients
            selected_coefficients =
                VolumeCoefficients::zero();
        VolumePhaseSetSample phase{
            .direction = phase_axis,
            .pdf = 0.0f,
            .sampled_roughness = 1.0f,
            .selection_rescaled =
                phase_random.x,
            .closure_index = 0u,
            .closure_type = 0u,
            .valid = false};
        $if(selected_scatter) {
            final_throughput =
                selected_throughput;
            final_emission =
                selected_emission;
            final_distance =
                selected_distance;
            VolumePhaseSet phases{
                _closure_allocation_budget};
            selected_coefficients =
                collisions.evaluate(
                    selected_distance,
                    true,
                    &phases);
            phase =
                phases.sample(
                    phase_axis,
                    phase_random);
        };
        const auto scattered =
            selected_scatter &
            phase.valid;
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
                     reservoir_random,
                 .optical_depth =
                     optical_depth,
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
            .scattered = scattered,
            .phase_failed =
                selected_scatter &
                !phase.valid};
    }
};

}// namespace

std::unique_ptr<HeterogeneousVolumeSegmentComponent>
make_heterogeneous_volume_segment_component(
    std::size_t closure_allocation_budget) {
    return std::make_unique<
        HeterogeneousVolumeSegmentComponentImpl>(
        closure_allocation_budget);
}

std::unique_ptr<HeterogeneousVolumeCollisionProvider>
make_stacked_heterogeneous_volume_collision_provider(
    const SurfaceDispatch &surfaces,
    std::shared_ptr<
        const VolumeStackEntryPointProvider> points,
    const VolumeStack &stack,
    const ShaderServices &services,
    const VolumeShadingState &base_state,
    Float3 ray_origin,
    Float3 ray_direction) {
    return std::make_unique<
        StackedHeterogeneousVolumeCollisionProvider>(
        surfaces,
        std::move(points),
        stack,
        services,
        base_state,
        std::move(ray_origin),
        std::move(ray_direction));
}

}// namespace psycles::luisa_backend

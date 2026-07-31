#include <psycles/luisa/homogeneous_volume_segment.h>

#include <algorithm>
#include <utility>

namespace psycles::luisa_backend {
namespace {

class HomogeneousVolumeSegmentComponentImpl final
    : public HomogeneousVolumeSegmentComponent {

  private:
    const SurfaceDispatch &_surfaces;
    std::shared_ptr<const VolumeStackEntryPointProvider> _points;
    std::size_t _closure_allocation_budget;
    HomogeneousVolumeTransport _transport;
    HomogeneousVolumeScatterProbability
        _scatter_probability;

  public:
    HomogeneousVolumeSegmentComponentImpl(
        const SurfaceDispatch &surfaces,
        std::shared_ptr<const VolumeStackEntryPointProvider> points,
        std::size_t closure_allocation_budget) noexcept
        : _surfaces{surfaces},
          _points{std::move(points)},
          _closure_allocation_budget{
              std::max(
                  closure_allocation_budget,
                  std::size_t{1u})} {}

    HomogeneousVolumeSegmentResult
    emit(const VolumeStack &stack,
         const ShaderServices &services,
         const VolumeShadingState &state,
         Float distance,
         Float3 throughput,
         Float scatter_random,
         Float channel_random,
         Float2 phase_random,
         Bool terminate,
         const VolumeScatterProbabilityGuidingState &guiding,
         const HomogeneousVolumeDirectInput &direct,
         const VolumeDirectDirectionProvider
             *direct_direction) const noexcept override {
        VolumePhaseSet phases{
            _closure_allocation_budget};
        const StackedVolumeEvaluator evaluator{
            _surfaces, *_points};
        const auto coefficients =
            evaluator.evaluate(
                stack,
                services,
                state,
                true,
                &phases);
        const auto scatter_probability =
            _scatter_probability.evaluate(
                coefficients,
                distance,
                terminate,
                guiding);
        const VolumeDirectSampling
            direct_sampling;
        const auto direct_state =
            direct_sampling.prepare(
                direct.requested_method,
                scatter_random,
                direct.enabled &
                    !terminate);
        const auto transport =
            _transport.sample_with_probability(
                coefficients,
                distance,
                throughput,
                direct_state.random,
                channel_random,
                scatter_probability,
                terminate);
        const auto direct_transport =
            _transport.sample_direct(
                coefficients,
                distance,
                throughput,
                transport.scatter_random,
                transport.reservoir_random,
                direct_state,
                state.position,
                -state.incoming,
                {.light_position =
                     direct.light_position,
                 .interval =
                     direct.interval});
        VolumeDirectDirectionSample
            direction_sample{
                .direction =
                    make_float3(0.0f),
                .valid = false};
        if (direct_direction != nullptr) {
            direction_sample =
                direct_direction->emit(
                    direct_transport
                        .distance);
        }
        const auto direct_phase_raw =
            phases.evaluate(
                -state.incoming,
                direction_sample.direction);
        const VolumePhaseSetEvaluation
            direct_phase{
                .value = select(
                    0.0f,
                    direct_phase_raw.value,
                    direction_sample.valid),
                .pdf = select(
                    0.0f,
                    direct_phase_raw.pdf,
                    direction_sample.valid),
                .sample_weight = select(
                    0.0f,
                    direct_phase_raw
                        .sample_weight,
                    direction_sample.valid),
                .valid =
                    direct_phase_raw.valid &
                    direction_sample.valid};

        // Cycles phase functions use -sd->wi as their axis. Volume ShaderData
        // stores sd->wi = -ray.D, so the sampling axis here is the propagation
        // direction rather than the viewer-facing incoming vector.
        const auto phase =
            phases.sample(
                -state.incoming,
                phase_random);
        const auto scattered =
            transport.scattered &
            phase.valid;
        return {
            .coefficients =
                coefficients,
            .transport =
                transport,
            .phase = phase,
            .direct_transport =
                direct_transport,
            .direct_phase =
                direct_phase,
            .scattered =
                scattered,
            .phase_failed =
                transport.scattered &
                !phase.valid};
    }
};

}// namespace

std::unique_ptr<HomogeneousVolumeSegmentComponent>
make_homogeneous_volume_segment_component(
    const SurfaceDispatch &surfaces,
    std::shared_ptr<const VolumeStackEntryPointProvider> points,
    std::size_t closure_allocation_budget) {
    return std::make_unique<
        HomogeneousVolumeSegmentComponentImpl>(
        surfaces,
        std::move(points),
        closure_allocation_budget);
}

}// namespace psycles::luisa_backend

#include <psycles/luisa/homogeneous_volume_segment.h>

#include <algorithm>
#include <utility>

namespace psycles::luisa_backend {
namespace {

class HomogeneousVolumeSegmentComponentImpl final
    : public HomogeneousVolumeSegmentComponent {

  private:
    std::size_t _closure_allocation_budget;
    HomogeneousVolumeTransport _transport;
    HomogeneousVolumeScatterProbability
        _scatter_probability;

  public:
    HomogeneousVolumeSegmentComponentImpl(
        std::size_t closure_allocation_budget) noexcept
        : _closure_allocation_budget{
              std::max(
                  closure_allocation_budget,
                  std::size_t{1u})} {}

    HomogeneousVolumeSegmentResult
    emit(const VolumeShaderEvaluator &shader,
         const VolumeStack &stack,
         Float3 position,
         Float3 incoming,
         Float distance,
         Float3 throughput,
         Float scatter_random,
         Float channel_random,
         Float2 phase_random,
         Bool terminate,
         const VolumeScatterProbabilityGuidingState &guiding,
         const HomogeneousVolumeDirectInput &direct,
         const VolumeDirectLightProvider
             *direct_light) const noexcept override {
        const VolumeDirectSampling
            direct_sampling;
        const auto direct_state =
            direct_sampling.prepare(
                direct.requested_method,
                scatter_random,
                direct.enabled &
                    !terminate);
        const VolumeEquiangularCoefficients equiangular{
            .light_position = direct.light_position, .interval = direct.interval};
        VolumeEquiangularSample equiangular_sample{
            .distance = 0.0f, .pdf = 0.0f, .valid = false};
        // Original volume_integrate_result_init precedes coefficient shading,
        // including on an emission-only segment. Do not prune this based on
        // sigma_s; only the chosen direct-sampling technique controls it.
        $if(direct_state.method == volume_sample_equiangular) {
            equiangular_sample = direct_sampling.sample_equiangular(
                position, -incoming, equiangular, direct_state.random);
        };
        VolumePhaseSet phases{_closure_allocation_budget};
        const auto coefficients = shader.evaluate(stack, position, &phases);
        Float3 scatter_probability = make_float3(0.0f);
        $if(!terminate & coefficients.has_scatter &
            any(coefficients.sigma_s != make_float3(0.0f)) & (distance > 0.0f)) {
            scatter_probability = _scatter_probability.evaluate(
                coefficients, distance, terminate, guiding);
        };
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
                position,
                -incoming,
                equiangular,
                equiangular_sample);
        VolumePhaseSetEvaluation direct_phase{
            .value = 0.0f, .pdf = 0.0f, .sample_weight = 0.0f, .valid = false};
        if (direct_light != nullptr) {
            // Cycles volume_integrate_event calls direct lighting only for
            // result.direct_scatter. Its light resample may itself fail before
            // constant emission or phase evaluation is reached.
            $if(direct_transport.scattered) {
                const auto direction_sample =
                    direct_light->sample_direction(direct_transport.distance);
                $if(direction_sample.valid) {
                    direct_light->evaluate_constant_emission();
                    direct_phase = phases.evaluate(-incoming, direction_sample.direction);
                    direct_light->evaluate_deferred_emission(
                        direct_phase.valid & (direct_phase.value != 0.0f));
                };
            };
        }

        // Cycles phase functions use -sd->wi as their axis. Volume ShaderData
        // stores sd->wi = -ray.D, so the sampling axis here is the propagation
        // direction rather than the viewer-facing incoming vector.
        VolumePhaseSetSample phase{
            .direction = -incoming, .pdf = 0.0f, .sampled_roughness = 1.0f,
            .selection_rescaled = phase_random.x, .closure_index = 0u,
            .closure_type = 0u, .valid = false};
        $if(transport.scattered) {
            phase = phases.sample(-incoming, phase_random);
        };
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
    std::size_t closure_allocation_budget) {
    return std::make_unique<
        HomogeneousVolumeSegmentComponentImpl>(
        closure_allocation_budget);
}

}// namespace psycles::luisa_backend

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
         Bool terminate) const noexcept override {
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
        const auto transport =
            _transport.sample(
                coefficients,
                distance,
                throughput,
                scatter_random,
                channel_random,
                terminate);

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

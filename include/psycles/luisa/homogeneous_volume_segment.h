#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/homogeneous_volume_segment.h> through the Psycles::luisa target."
#endif

#include <cstddef>
#include <memory>

#include <psycles/luisa/homogeneous_volume_transport.h>
#include <psycles/luisa/stacked_volume.h>

namespace psycles::luisa_backend {

struct HomogeneousVolumeSegmentResult {
    VolumeCoefficients coefficients;
    HomogeneousVolumeSample transport;
    VolumePhaseSetSample phase;
    Bool scattered;
    Bool phase_failed;
};

// Host-stage composition of the three independent Cycles volume contracts:
// raw stacked-graph evaluation, analytic homogeneous distance sampling, and
// phase continuation. The component records one fused Luisa AST. It does not
// own path state, closest-event ordering, film routing, or Sobol dimensions;
// those remain responsibilities of the path-kernel stage.
class HomogeneousVolumeSegmentComponent {

  public:
    virtual ~HomogeneousVolumeSegmentComponent() noexcept = default;

    [[nodiscard]] virtual HomogeneousVolumeSegmentResult
    emit(const VolumeStack &stack,
         const ShaderServices &services,
         const VolumeShadingState &state,
         Float distance,
         Float3 throughput,
         Float scatter_random,
         Float channel_random,
         Float2 phase_random,
         Bool terminate) const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<HomogeneousVolumeSegmentComponent>
make_homogeneous_volume_segment_component(
    const SurfaceDispatch &surfaces,
    std::shared_ptr<const VolumeStackEntryPointProvider> points,
    std::size_t closure_allocation_budget);

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(
    psycles::luisa_backend::HomogeneousVolumeSegmentResult)

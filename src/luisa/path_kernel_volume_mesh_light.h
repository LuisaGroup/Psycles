#pragma once

#include "path_kernel_volume_direct_light.h"

namespace psycles::luisa_backend::detail {

// Mesh-emitter half of Cycles' two-stage volume direct-light protocol. The
// proposal is sampled against the full ray segment; the direction provider
// resamples the same emitter from the actual collision point.
class VolumeMeshLightComponent {

  public:
    virtual ~VolumeMeshLightComponent() noexcept =
        default;

    virtual void propose(
        const ClosestPathEvent &event,
        const VolumeStack &path_stack,
        Float3 segment_position,
        Float3 segment_direction,
        Float segment_length,
        VolumeDirectLightProposal &result)
        const noexcept = 0;

    [[nodiscard]] virtual std::unique_ptr<
        VolumeDirectDirectionProvider>
    make_direction_provider(
        ClosestPathEvent &event,
        const VolumeDirectLightProposal &proposal,
        Float3 segment_position,
        Float3 segment_direction,
        VolumeDirectLightSample &result)
        const = 0;
};

[[nodiscard]]
std::unique_ptr<VolumeMeshLightComponent>
make_volume_mesh_light_component(
    const PathKernelConfig &config);

}// namespace psycles::luisa_backend::detail

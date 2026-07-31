#pragma once

#include "path_kernel_volume_direct_light.h"

namespace psycles::luisa_backend::detail {

// Background-emitter half of Cycles' two-stage volume direct-light protocol.
// The segment proposal deliberately does not sample a direction: infinite
// emitters force distance sampling, and PRNG_LIGHT.xy is consumed only after
// the volume collision position is known.
class VolumeEnvironmentLightComponent {

  public:
    virtual ~VolumeEnvironmentLightComponent() noexcept =
        default;

    virtual void propose(
        const ClosestPathEvent &event,
        const VolumeStack &path_stack,
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
std::unique_ptr<
    VolumeEnvironmentLightComponent>
make_volume_environment_light_component(
    const PathKernelConfig &config);

}// namespace psycles::luisa_backend::detail

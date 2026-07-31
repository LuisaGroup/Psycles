#pragma once

#include "path_kernel_builder.h"

#include <psycles/luisa/homogeneous_volume_segment.h>

#include <memory>

namespace psycles::luisa_backend::detail {

struct VolumeDirectLightSample {
    Float3 direction;
    Float3 radiance;
    Float pdf;
    Float maximum_distance;
    UInt light_instance;
    UInt light_primitive;
    Bool use_mis;
    Bool valid;
};

struct VolumeDirectLightProposal {
    UInt emitter_kind;
    UInt emitter_index;
    UInt requested_method;
    Float3 light_position;
    VolumeDirectSampleInterval interval;
    Bool valid;
};

// The light is sampled before volume-distance integration because Cycles
// chooses distance/equiangular transport from the selected emitter. The
// contribution is completed afterwards, once the direct collision point and
// phase closures are known.
class VolumeDirectLightingComponent {

  public:
    virtual ~VolumeDirectLightingComponent() noexcept =
        default;

    [[nodiscard]] virtual VolumeDirectLightProposal
    propose(
        const ClosestPathEvent &event,
        const VolumeStack &path_stack,
        Float3 segment_position,
        Float3 segment_direction,
        Float segment_length) const noexcept = 0;

    [[nodiscard]] virtual std::unique_ptr<
        VolumeDirectDirectionProvider>
    make_direction_provider(
        ClosestPathEvent &event,
        const VolumeDirectLightProposal &proposal,
        Float3 segment_position,
        Float3 segment_direction,
        VolumeDirectLightSample &result)
        const = 0;

    virtual void accumulate(
        ClosestPathEvent &event,
        const VolumeDirectLightSample &light,
        const HomogeneousVolumeSegmentResult &volume,
        const VolumeStack &path_stack,
        Float3 segment_position,
        Bool inside_volume) const noexcept = 0;
};

[[nodiscard]]
std::unique_ptr<VolumeDirectLightingComponent>
make_volume_direct_lighting_component(
    const PathKernelConfig &config);

}// namespace psycles::luisa_backend::detail

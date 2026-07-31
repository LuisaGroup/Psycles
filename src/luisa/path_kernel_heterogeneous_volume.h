#pragma once

#include "path_kernel_volume_point.h"

#include <psycles/luisa/heterogeneous_volume_segment.h>

#include <memory>

namespace psycles::luisa_backend::detail {

// All path-state inputs needed to record one heterogeneous volume segment.
// References identify host-stage components/resources; scalar DSL expressions
// remain values so the implementation can assemble a fused device AST without
// depending on the enclosing path-kernel context type.
struct PathHeterogeneousVolumeInput {
    const VolumeStack &stack;
    const ShaderServices &services;
    const VolumeShadingState &state;
    const BufferFloat4 &sobol_table;
    UInt sobol_sequence_size;
    UInt sample_index;
    UInt rng_hash;
    UInt path_rng_offset;
    Float3 ray_origin;
    Float3 ray_direction;
    Float ray_minimum;
    Float ray_maximum;
    Float3 throughput;
    Float reservoir_random;
    Float2 phase_random;
    Float majorant_scale;
    Bool terminate;
};

// Production host-stage adapter between path state, scene majorant resources,
// Cycles random dimensions, raw stacked GraphSurface closures, and the generic
// heterogeneous transport component.
class PathHeterogeneousVolumeComponent {

  public:
    virtual ~PathHeterogeneousVolumeComponent()
        noexcept = default;

    [[nodiscard]] virtual Bool stack_is_heterogeneous(
        const VolumeStack &stack) const noexcept = 0;

    [[nodiscard]] virtual HeterogeneousVolumeSegmentResult
    emit(
        const PathHeterogeneousVolumeInput &input)
        const noexcept = 0;
};

[[nodiscard]]
std::unique_ptr<PathHeterogeneousVolumeComponent>
make_path_heterogeneous_volume_component(
    std::shared_ptr<LuisaSceneData> scene,
    std::shared_ptr<
        const VolumeStackEntryPointProvider> points,
    std::size_t closure_allocation_budget);

}// namespace psycles::luisa_backend::detail

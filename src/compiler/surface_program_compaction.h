#pragma once

#include <vector>

#include <psycles/compiler/surface_program.h>

namespace psycles::compiler::detail {

// The graph validator schedules reachable nodes, but a Blender node may have
// several outputs with different dependencies. Lowering those outputs eagerly
// can therefore leave dead value/closure instructions in the typed program.
// This storage-level pass computes reachability from the surface, volume,
// automatic surface-normal, and displacement roots, then preserves
// topological order while assigning dense IDs. Automatic bump is an explicit
// surface-normal side effect; geometric displacement remains a pure vector.
struct SurfaceProgramStorage {
    std::vector<ParameterDesc> parameters;
    std::vector<ValueInstruction> values;
    std::vector<ClosureInstruction> closures;
    ClosureExpressionId root;
    std::vector<VolumeInstruction> volumes;
    VolumeExpressionId volume_root;
    ValueExpressionId surface_normal_root;
    ValueExpressionId displacement_root;
};

[[nodiscard]] SurfaceProgramStorage compact_surface_program(
    SurfaceProgramStorage storage);

}// namespace psycles::compiler::detail

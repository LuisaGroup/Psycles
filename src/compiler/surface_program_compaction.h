#pragma once

#include <vector>

#include <psycles/compiler/surface_program.h>

namespace psycles::compiler::detail {

// The graph validator schedules reachable nodes, but a Blender node may have
// several outputs with different dependencies. Lowering those outputs eagerly
// can therefore leave dead value/closure instructions in the typed program.
// This storage-level pass computes reachability from the surface, volume, and
// displacement roots, then preserves topological order while assigning dense
// IDs. The displacement root is semantically observable even though its value
// is not consumed by a closure: bump evaluation updates the shading normal.
struct SurfaceProgramStorage {
    std::vector<ParameterDesc> parameters;
    std::vector<ValueInstruction> values;
    std::vector<ClosureInstruction> closures;
    ClosureExpressionId root;
    std::vector<VolumeInstruction> volumes;
    VolumeExpressionId volume_root;
    ValueExpressionId displacement_root;
};

[[nodiscard]] SurfaceProgramStorage compact_surface_program(
    SurfaceProgramStorage storage);

}// namespace psycles::compiler::detail

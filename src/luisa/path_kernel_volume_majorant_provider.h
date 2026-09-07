#pragma once

#include "path_tracer_cycles_svm_volume.h"

#include <psycles/luisa/volume_majorant_overlap.h>

#include <memory>

namespace psycles::luisa_backend::detail {

// Builds the production object/world transform and runtime-extrema policy for
// one overlap traversal. The returned object exists only while the host
// records the enclosing kernel AST; device execution remains fully fused.
[[nodiscard]] std::unique_ptr<VolumeMajorantEntryProvider>
make_scene_volume_majorant_entry_provider(
    const PathCyclesSvmVolumeShader &shader);

}// namespace psycles::luisa_backend::detail

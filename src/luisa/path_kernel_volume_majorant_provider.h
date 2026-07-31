#pragma once

#include "path_kernel_volume_point.h"
#include "path_tracer_shader_services.h"

#include <psycles/luisa/volume_majorant_overlap.h>

#include <memory>

namespace psycles::luisa_backend::detail {

// Builds the production object/world transform and runtime-extrema policy for
// one overlap traversal. The returned object exists only while the host
// records the enclosing kernel AST; device execution remains fully fused.
[[nodiscard]] std::unique_ptr<VolumeMajorantEntryProvider>
make_scene_volume_majorant_entry_provider(
    std::shared_ptr<LuisaSceneData> scene,
    std::shared_ptr<
        const VolumeStackEntryPointProvider> points,
    const ShaderServices &services,
    const VolumeShadingState &state,
    Float shade_offset,
    bool evaluate_emission);

}// namespace psycles::luisa_backend::detail

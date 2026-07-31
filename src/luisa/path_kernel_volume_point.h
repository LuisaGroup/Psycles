#pragma once

#include "path_tracer_internal.h"

#include <psycles/luisa/stacked_volume.h>

#include <memory>

namespace psycles::luisa_backend::detail {

// Builds Cycles ShaderData semantics for a point inside each active medium.
// The implementation is host-polymorphic and records only the object/world
// branch needed at device runtime; raw volume graphs remain responsible for
// evaluating their own coordinates and closures.
[[nodiscard]]
std::shared_ptr<const VolumeStackEntryPointProvider>
make_scene_volume_stack_entry_point_provider(
    std::shared_ptr<LuisaSceneData> scene);

}// namespace psycles::luisa_backend::detail

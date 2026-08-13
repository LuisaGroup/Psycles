#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

// Maps a candidate from the compact local-BSSRDF TLAS back to the canonical
// primary-TLAS/InstanceGpu identity. The local TLAS user-id relation is built
// as a host-side injection, so no lookup table or scene-specific shader switch
// is required.
[[nodiscard]] UInt subsurface_primary_instance(
    const std::shared_ptr<LuisaSceneData> &scene,
    Expr<std::uint32_t> local_instance) noexcept;

}// namespace psycles::luisa_backend::detail

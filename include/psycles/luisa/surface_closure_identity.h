#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_identity.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface.h>
#include <psycles/luisa/surface_closure_reachability.h>

namespace psycles::luisa_backend::detail {

[[nodiscard]] Bool closure_allocated(
    const SurfaceClosureRecord &closure) noexcept;

[[nodiscard]] UInt cycles_runtime_flags(
    const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness = 0.0f,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;
// Runtime flags for an already-retained ShaderClosure. Prefix membership
// proves allocation and type_none is the complete setup-failure state, so no
// authoring identity or compatibility flags participate in this overload.
[[nodiscard]] UInt cycles_runtime_flags(
    UInt closure_type,
    Float roughness,
    Float glossy_filter_roughness,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;

} // namespace psycles::luisa_backend::detail

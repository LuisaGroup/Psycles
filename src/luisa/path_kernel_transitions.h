#pragma once

namespace psycles::luisa_backend::detail::path_transition {

// Semantic queue identities for host/JIT coroutine construction. Keeping
// these names in one place makes scheduler policy refer to the exact cuts
// emitted by PathKernelPipeline without introducing a device-side enum.
inline constexpr char path_bounce[] = "path_bounce";
inline constexpr char intersect_closest[] = "intersect_closest";
inline constexpr char shade_volume[] = "shade_volume";
inline constexpr char shade_light_forward[] = "shade_light_forward";
inline constexpr char shade_background[] = "shade_background";
inline constexpr char surface_shading[] = "surface_shading";
inline constexpr char shade_surface[] = "shade_surface";
// Cycles-compatible side shadow-path state machine. SHADE_SURFACE forks one
// prepared shadow state into SHADE_LIGHT_NEE (only for non-constant
// emitters) or directly into INTERSECT_SHADOW. SHADE_SHADOW may loop back to
// INTERSECT_SHADOW for further transparent hits.
inline constexpr char shade_light_nee[] = "shade_light_nee";
inline constexpr char intersect_shadow[] = "intersect_shadow";
inline constexpr char shade_shadow[] = "shade_shadow";
inline constexpr char intersect_subsurface[] = "intersect_subsurface";

// Scheduler-only frame ABI. The value exported under this name is not an
// integrator input: it only orders frames already queued at shade_surface.
inline constexpr char scheduler_hint[] = "coro_hint";

}// namespace psycles::luisa_backend::detail::path_transition

#pragma once

#include "path_kernel_builder.h"

namespace psycles::luisa_backend::detail {

// Direct consumer of the one retained native ShaderData/closure population.
// Native labels stay native through continuation; SurfaceSample is populated
// only for the existing BSSRDF entry-parameter handoff.
[[nodiscard]] SurfaceScatterStage::Result emit_cycles_svm_surface_scatter(
    DirectLightingContext &context, const CyclesSvmSurfaceState &native) noexcept;

} // namespace psycles::luisa_backend::detail

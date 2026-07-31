#pragma once

#include "path_tracer_internal.h"

#include <psycles/luisa/area_light_sampling.h>

namespace psycles::luisa_backend::detail {

// Adapt the shared GPU light record to the host-stage Cycles area-sampling
// component. Keeping this map in one translation unit prevents surface,
// volume, and forward paths from assigning different meanings to light flags
// or axes while they construct their Luisa AST.
[[nodiscard]] AreaLightSampleInput
area_light_sample_input(
    luisa::compute::Var<LightGpu> light,
    luisa::compute::Float3 reference,
    luisa::compute::Float2 random) noexcept;

}// namespace psycles::luisa_backend::detail

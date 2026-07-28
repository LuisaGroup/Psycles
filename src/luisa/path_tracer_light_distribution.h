#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

using LightDistributionSampleCallable =
    Callable<LightDistributionGpu(float)>;

// Device implementation of Cycles' flat-distribution upper-bound lookup.
// This is intentionally a Luisa callable: there is no host-side mirror of
// the selection algorithm.
[[nodiscard]] LightDistributionSampleCallable
make_light_distribution_sample_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept;

}// namespace psycles::luisa_backend::detail

#pragma once

#include "path_tracer_lighting.h"
#include "path_tracer_surfaces.h"

namespace psycles::luisa_backend::detail {

using EnvironmentBaseCallable =
    Callable<luisa::float3(
        luisa::float3,
        luisa::float3,
        luisa::uint)>;
using EnvironmentSunCallable =
    Callable<luisa::float3(luisa::float3)>;

struct EnvironmentCallables {
    EnvironmentBaseCallable base;
    std::vector<EnvironmentSunCallable> suns;
    EnvironmentSunCallable nishita_sun;
};

[[nodiscard]] EnvironmentCallables
make_environment_callables(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize,
    const SurfaceEmissionCallable &surface_emission);

}// namespace psycles::luisa_backend::detail

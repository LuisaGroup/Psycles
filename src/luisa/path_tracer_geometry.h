#pragma once

#include "path_tracer_lighting.h"

namespace psycles::luisa_backend::detail {

using TraceShadowCallable =
    Callable<luisa::float3(luisa::compute::Ray)>;

[[nodiscard]] TraceShadowCallable make_trace_shadow_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize) noexcept;

}// namespace psycles::luisa_backend::detail

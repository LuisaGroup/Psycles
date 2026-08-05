#pragma once

#include "path_tracer_lighting.h"

namespace psycles::luisa_backend::detail {

using TraceShadowCallable =
    Callable<ShadowTraceResultCall(
        luisa::compute::Ray,
        // Exact Cycles source (object, primitive) identity.
        luisa::uint,
        luisa::uint,
        // Exact Cycles light (object, primitive) identity.
        luisa::uint,
        luisa::uint,
        luisa::uint,
        ShaderEvaluationStateCall)>;

[[nodiscard]] TraceShadowCallable make_trace_shadow_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize) noexcept;

}// namespace psycles::luisa_backend::detail

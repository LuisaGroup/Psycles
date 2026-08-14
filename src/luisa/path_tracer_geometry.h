#pragma once

#include "path_tracer_lighting.h"

namespace psycles::luisa_backend::detail {

using IntersectShadowCallable =
    Callable<ShadowIntersectionCall(
        luisa::compute::Ray,
        // Exact Cycles source (object, primitive) identity.
        luisa::uint,
        luisa::uint,
        // Exact Cycles light (object, primitive) identity.
        luisa::uint,
        luisa::uint)>;

using EvaluateShadowSurfaceCallable =
    Callable<ShadowSurfaceEvaluationCall(
        luisa::compute::Ray,
        ShadowIntersectionCall,
        float,
        float,
        ShaderEvaluationStateCall)>;

using TraceShadowCallable =
    Callable<ShadowTraceResultCall(
        luisa::compute::Ray,
        // Compact positional and angular ray differentials.
        float,
        float,
        // Exact Cycles source (object, primitive) identity.
        luisa::uint,
        luisa::uint,
        // Exact Cycles light (object, primitive) identity.
        luisa::uint,
        luisa::uint,
        luisa::uint,
        ShaderEvaluationStateCall)>;

struct ShadowTraceCallables {
    IntersectShadowCallable intersect;
    EvaluateShadowSurfaceCallable shade_surface;
    TraceShadowCallable trace;
};

[[nodiscard]] ShadowTraceCallables make_shadow_trace_callables(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize) noexcept;

}// namespace psycles::luisa_backend::detail

#pragma once

#include "path_tracer_lighting.h"

namespace psycles::luisa_backend::detail {

class ShadowIntersectionBatchStorage;

using IntersectShadowCallable = Callable<ShadowIntersectionBatchCall(
    luisa::compute::Ray,
    // Exact Cycles source (object, primitive) identity.
    luisa::uint, luisa::uint,
    // Exact Cycles light (object, primitive) identity.
    luisa::uint, luisa::uint,
    // Remaining transparent-bounce budget.
    luisa::uint,
    // Runtime SoA capacity; a value, never a shader specialization constant.
    luisa::uint,
    // Enclosing kernel's physical x block size. Callables do not own launch
    // dimensions, so this must remain an explicit invocation-context value.
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
        luisa::uint,
        luisa::uint,
        ShaderEvaluationStateCall)>;

struct ShadowTraceCallables {
    IntersectShadowCallable intersect;
    EvaluateShadowSurfaceCallable shade_surface;
    TraceShadowCallable trace;
};

// Orders the bounded nearest-hit set at its first order-sensitive consumer.
// Traversal deliberately keeps the set unordered so its candidate loop never
// shifts complete intersection records.
void sort_shadow_intersection_batch(
    Var<ShadowIntersectionBatchCall> &batch) noexcept;

[[nodiscard]] ShadowTraceCallables make_shadow_trace_callables(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SafeNormalizeCallable &safe_normalize,
    std::shared_ptr<const ShadowIntersectionBatchStorage> storage = {}) noexcept;

}// namespace psycles::luisa_backend::detail

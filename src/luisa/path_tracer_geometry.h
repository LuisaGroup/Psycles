#pragma once

#include "path_tracer_lighting.h"

#include <variant>

namespace psycles::luisa_backend::detail {

class ShadowIntersectionBatchStorage;

using IntersectShadowCallable = Callable<ShadowIntersectionSummaryCall(
    luisa::compute::Ray,
    // Exact Cycles source (object, primitive) identity.
    luisa::uint, luisa::uint,
    // Exact Cycles light (object, primitive) identity.
    luisa::uint, luisa::uint,
    // Remaining transparent-bounce budget.
    luisa::uint,
    // Invocation-owned SoA slot computed by the enclosing kernel.
    luisa::uint,
    // Runtime SoA capacity; a value, never a shader specialization constant.
    luisa::uint)>;

using LocalShadowIntersectionCallable =
    Callable<ShadowIntersectionBatchCall(
        luisa::compute::Ray, luisa::uint, luisa::uint, luisa::uint,
        luisa::uint, luisa::uint)>;

// Host-side selection of the bounded traversal storage. Both paths return a
// complete value before the caller may suspend; external invocation identity
// never crosses a coroutine edge. No storage-policy dispatch survives in DSL.
class ShadowIntersectionComponent {

private:
    std::shared_ptr<const ShadowIntersectionBatchStorage> _storage;
    std::variant<LocalShadowIntersectionCallable, IntersectShadowCallable>
        _intersect;

public:
    explicit ShadowIntersectionComponent(
        LocalShadowIntersectionCallable intersect) noexcept;

    ShadowIntersectionComponent(
        std::shared_ptr<const ShadowIntersectionBatchStorage> storage,
        IntersectShadowCallable intersect) noexcept;

    [[nodiscard]] Var<ShadowIntersectionBatchCall> collect(
        Var<luisa::compute::Ray> shadow_ray,
        Expr<std::uint32_t> source_object,
        Expr<std::uint32_t> source_primitive,
        Expr<std::uint32_t> light_object,
        Expr<std::uint32_t> light_primitive,
        Expr<std::uint32_t> transparent_maximum,
        Expr<std::uint32_t> storage_capacity,
        Expr<std::uint32_t> storage_block_size) const noexcept;

    [[nodiscard]] const IntersectShadowCallable &
    summary_callable() const noexcept;
};

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
    std::shared_ptr<const ShadowIntersectionComponent> intersect;
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

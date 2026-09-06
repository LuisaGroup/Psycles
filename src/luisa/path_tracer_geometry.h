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
        ShadowShaderContextCall,
        RenderKernelParameters)>;

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
        // Initial Cycles shadow_path.throughput (not just transparency).
        luisa::float3,
        ShadowShaderContextCall,
        RenderKernelParameters)>;

[[nodiscard]] Var<ShadowShaderContextCall> make_shadow_shader_context(
    const Var<ShaderEvaluationStateCall> &path, Expr<float> time,
    Expr<luisa::uint> sample, Expr<luisa::uint> rng_hash,
    Expr<luisa::uint> rng_offset,
    Expr<luisa::uint> volume_bounds_bounce = 0u) noexcept;

// Cycles integrate_transparent_shadow's per-surface state transition. A zero
// cumulative throughput terminates before committing throughput/RNG/bounce;
// the volume-boundary counter is updated by surface evaluation itself.
[[nodiscard]] Bool advance_shadow_surface_state(
    Var<ShadowShaderContextCall> &context, Float3 &throughput,
    const Var<ShadowSurfaceEvaluationCall> &surface) noexcept;

inline constexpr luisa::uint shadow_volume_bounds_max = 1024u;

[[nodiscard]] TraceShadowCallable make_fused_shadow_trace_callable(
    std::shared_ptr<const ShadowIntersectionComponent> intersection,
    EvaluateShadowSurfaceCallable evaluate_shadow_surface) noexcept;

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

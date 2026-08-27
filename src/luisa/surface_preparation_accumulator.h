#pragma once

#include <cstddef>
#include <cstdint>

#include <psycles/luisa/surface_closure_operations.h>

namespace psycles::luisa_backend::detail {

enum class RuntimeFlagReductionMode : std::uint8_t {
    projected_output,
    retained_state,
};

// Device-stage left fold over the retained source-order closure sequence.
// Unlike SurfaceClosureExpressionVisitor, this remains valid inside a runtime
// bytecode loop: every add() records one dynamic transaction instead of
// appending a host-side expression handle.
class SurfacePreparationAccumulator {

  private:
    SurfacePoint _point;
    std::size_t _capacity;
    Float _glossy_filter_roughness;
    Bool _include_runtime_flags;
    Bool _include_aov;
    RuntimeFlagReductionMode _runtime_flag_mode;
    SurfaceClosureIdentityCallable _identity;
    SurfaceClosureAovCallable _aov_operation;
    UInt _retained_count;
    UInt _runtime_flags;
    SurfaceAov _aov;
    Float _aov_total_weight;
    Float _aov_roughness_weight;
    Float _aov_roughness;
    Float3 _aov_normal;

    void fold_runtime_identity(
        const SurfaceClosureRecord &closure) noexcept;
    void fold_retained(
        const SurfaceClosureRecord &closure) noexcept;

  public:
    SurfacePreparationAccumulator(
        const SurfacePoint &point,
        std::size_t capacity,
        Expr<float> glossy_filter_roughness,
        Expr<bool> include_runtime_flags,
        Expr<bool> include_aov,
        const SurfaceClosureIdentityCallable &identity,
        const SurfaceClosureAovCallable &aov_operation,
        RuntimeFlagReductionMode runtime_flag_mode =
            RuntimeFlagReductionMode::projected_output) noexcept;

    // SurfaceClosureCollector::begin supplies the final shader normal after
    // automatic bump evaluation. Updating the fold state here makes that
    // lifecycle contract observable by every collector consumer without
    // re-running or baking any material node.
    void set_shading_normal(
        Expr<luisa::float3> shading_normal) noexcept;

    // Applies Cycles' allocation cutoff and source-order capacity transaction.
    void add(const SurfaceClosureRecord &closure) noexcept;

    // Used only from a collector transaction which has already proved and
    // retained this exact closure. The caller and this fold start at count zero
    // and advance together, preserving their count-equality invariant.
    void add_retained(
        const SurfaceClosureRecord &closure) noexcept;

    // Cycles observes an above-cutoff transparent setup before closure_alloc:
    // runtime identity and extinction therefore survive exhausted closure
    // capacity, while retained count advances only if the first slot is
    // actually allocated. The caller emits exactly one begin/finalize pair and
    // conditionally commits the slot between them.
    void begin_transparent_setup(
        const SurfaceClosureRecord &closure) noexcept;
    void retain_transparent_slot() noexcept;
    void finalize_transparent_setup(
        Expr<luisa::float3> weight) noexcept;

    void finish() noexcept;

    // Raw ShaderData::flag-equivalent state over the retained sequence. A
    // population transaction may keep this state even when preparation()
    // masks the public pass output, because later closure consumers still
    // observe the same populated ShaderData.
    [[nodiscard]] Expr<std::uint32_t>
    runtime_flags() const noexcept;

    [[nodiscard]] SurfacePreparation preparation(
        Float3 emission) const noexcept;
};

} // namespace psycles::luisa_backend::detail

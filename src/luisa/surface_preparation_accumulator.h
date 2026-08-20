#pragma once

#include <cstddef>

#include <psycles/luisa/surface_closure_operations.h>

namespace psycles::luisa_backend::detail {

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
    SurfaceClosureIdentityCallable _identity;
    SurfaceClosureAovCallable _aov_operation;
    UInt _retained_count;
    UInt _runtime_flags;
    SurfaceAov _aov;
    Float _aov_total_weight;
    Float _aov_roughness_weight;
    Float _aov_roughness;
    Float3 _aov_normal;

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
        const SurfaceClosureAovCallable &aov_operation) noexcept;

    // Applies Cycles' allocation cutoff and source-order capacity transaction.
    void add(const SurfaceClosureRecord &closure) noexcept;

    // Used only from a collector transaction which has already proved and
    // retained this exact closure. The caller and this fold start at count zero
    // and advance together, preserving their count-equality invariant.
    void add_retained(
        const SurfaceClosureRecord &closure) noexcept;

    void finish() noexcept;

    [[nodiscard]] SurfacePreparation preparation(
        Float3 emission) const noexcept;
};

} // namespace psycles::luisa_backend::detail

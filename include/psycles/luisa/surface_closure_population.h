#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_population.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface_closure_operations.h>

#include <cstddef>
#include <memory>

namespace psycles::luisa_backend {

class SurfaceClosureSet;

// ShaderData-equivalent state produced by one retained-closure transaction.
// Its private constructor makes provenance part of the type boundary: a
// consumer can copy this token, but cannot substitute an unrelated UInt.
class SurfaceClosurePopulationState {

  private:
    friend class SurfaceClosurePopulationCollector;
    friend class SurfaceClosureEvaluator;

    Expr<std::uint32_t> _runtime_flags;

    explicit SurfaceClosurePopulationState(
        Expr<std::uint32_t> runtime_flags) noexcept
        : _runtime_flags{runtime_flags} {}

  public:
    SurfaceClosurePopulationState(
        const SurfaceClosurePopulationState &) noexcept = default;
    SurfaceClosurePopulationState(
        SurfaceClosurePopulationState &&) noexcept = default;
};

// One-pass product of Cycles-compatible closure allocation. For the retained
// source-order subsequence S, this collector stores exactly physical(S), while
// runtime flags and camera AOVs are folded over that same S before setup-only
// expressions leave the material branch. No directional response is baked.
class SurfaceClosurePopulationCollector final
    : public SurfaceClosureCollector {

  private:
    struct Impl;
    std::unique_ptr<Impl> _impl;

  public:
    SurfaceClosurePopulationCollector(
        const SurfacePoint &point,
        std::size_t capacity,
        const SurfacePopulationQuery &query,
        const SurfaceClosureIdentityCallable &identity,
        const SurfaceClosureAovCallable &aov_operation) noexcept;
    ~SurfaceClosurePopulationCollector() noexcept override;

    SurfaceClosurePopulationCollector(
        const SurfaceClosurePopulationCollector &) = delete;
    SurfaceClosurePopulationCollector &operator=(
        const SurfaceClosurePopulationCollector &) = delete;

    void begin(
        Expr<luisa::float3> shading_normal) noexcept override;
    void add(
        const SurfaceClosureRecord &closure) noexcept override;
    [[nodiscard]] bool
    supports_transparent_closure_finalization()
        const noexcept override;
    void begin_transparent_closure(
        const SurfaceClosureRecord &closure) noexcept override;
    void finalize_transparent_closure(
        Expr<luisa::float3> weight,
        Expr<float> sample_weight) noexcept override;
    void finish() noexcept override;

    [[nodiscard]] const SurfaceClosureSet &closures() const noexcept;
    [[nodiscard]] SurfaceClosurePopulationState
    runtime_state() const noexcept;
    [[nodiscard]] SurfacePreparation preparation(
        Float3 emission) const noexcept;
};

}// namespace psycles::luisa_backend

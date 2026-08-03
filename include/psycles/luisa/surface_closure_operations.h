#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_operations.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface_closure_visitor.h>

namespace psycles::luisa_backend {

// Shared device callable for the closure identity algebra. The body is
// recorded once; material-specific visitors emit only calls and reductions.
using SurfaceClosureIdentityCallable =
    luisa::compute::Callable<luisa::uint2(
        luisa::uint,
        luisa::uint,
        float,
        bool,
        float,
        bool,
        bool,
        float)>;

[[nodiscard]] SurfaceClosureIdentityCallable
make_surface_closure_identity_callable() noexcept;

class SurfaceRuntimeFlagsVisitor final
    : public SurfaceClosureExpressionVisitor {

  private:
    const SurfacePoint &_point;
    Expr<float> _glossy_filter_roughness;
    const SurfaceClosureIdentityCallable &_identity;
    UInt _result{0u};

  protected:
    void visit(
        Expr<luisa::float3> shading_normal,
        const luisa::vector<SurfaceClosureExpression> &closures) noexcept override;

  public:
    SurfaceRuntimeFlagsVisitor(
        const SurfacePoint &point,
        Expr<float> glossy_filter_roughness,
        std::size_t capacity,
        const SurfaceClosureIdentityCallable &identity) noexcept;

    [[nodiscard]] Expr<std::uint32_t> result() const noexcept;
};

class SurfaceClosureTraceVisitor final
    : public SurfaceClosureExpressionVisitor {

  private:
    const SurfacePoint &_point;
    Expr<std::uint32_t> _requested_index;
    const SurfaceClosureIdentityCallable &_identity;
    SurfaceClosureTrace _result;

  protected:
    void visit(
        Expr<luisa::float3> shading_normal,
        const luisa::vector<SurfaceClosureExpression> &closures) noexcept override;

  public:
    SurfaceClosureTraceVisitor(
        const SurfacePoint &point,
        Expr<std::uint32_t> requested_index,
        std::size_t capacity,
        const SurfaceClosureIdentityCallable &identity) noexcept;

    [[nodiscard]] const SurfaceClosureTrace &result() const noexcept;
};

}// namespace psycles::luisa_backend

#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_operations.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface_closure_visitor.h>

#include <luisa/dsl/struct.h>

namespace psycles::luisa_backend {

// Reflected contribution returned by the shared per-closure AOV callable.
// Aggregation and normalization remain in the branch-local visitor.
struct SurfaceAovContributionCall {
    luisa::float3 albedo{};
    luisa::float3 glossy_albedo{};
    luisa::float3 transmission_albedo{};
    luisa::float3 transparency{};
    luisa::float3 normal{};
    float total_weight{};
    float roughness_weight{};
    float roughness{};
};

}// namespace psycles::luisa_backend

LUISA_STRUCT(
    psycles::luisa_backend::SurfaceAovContributionCall,
    albedo,
    glossy_albedo,
    transmission_albedo,
    transparency,
    normal,
    total_weight,
    roughness_weight,
    roughness) {};

namespace psycles::luisa_backend {

// Shared device callable for the closure identity algebra. The body is
// recorded once; material-specific visitors emit only calls and reductions.
using SurfaceClosureIdentityCallable =
    luisa::compute::Callable<luisa::uint2(
        luisa::uint,
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

using SurfaceClosureAovCallable =
    luisa::compute::Callable<SurfaceAovContributionCall(
        luisa::float3,
        luisa::float3,
        luisa::float3,
        bool,
        luisa::uint,
        luisa::uint,
        luisa::float3,
        bool,
        luisa::float3,
        luisa::float3,
        luisa::float3,
        luisa::float3,
        float)>;

[[nodiscard]] SurfaceClosureAovCallable
make_surface_closure_aov_callable() noexcept;

[[nodiscard]] luisa::compute::Var<SurfaceAovContributionCall>
surface_closure_aov_contribution(
    const SurfacePoint &point,
    const SurfaceClosureRecord &closure) noexcept;

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

// Cycles' surface_shader_bssrdf_normal() reduction. This intentionally sees
// the original, retained closure array at the exit shading point; it neither
// substitutes the entry closure nor derives the result from a combined AOV.
class SurfaceBssrdfNormalVisitor final
    : public SurfaceClosureExpressionVisitor {

  private:
    Float3 _result{make_float3(0.0f, 0.0f, 1.0f)};

  protected:
    void visit(
        Expr<luisa::float3> shading_normal,
        const luisa::vector<SurfaceClosureExpression> &closures) noexcept override;

  public:
    explicit SurfaceBssrdfNormalVisitor(
        std::size_t capacity) noexcept;

    [[nodiscard]] Expr<luisa::float3> result() const noexcept;
};

class SurfaceAovVisitor final
    : public SurfaceClosureExpressionVisitor {

  private:
    const SurfacePoint &_point;
    const SurfaceClosureAovCallable &_aov;
    SurfaceAov _result;

  protected:
    void visit(
        Expr<luisa::float3> shading_normal,
        const luisa::vector<SurfaceClosureExpression> &closures) noexcept override;

  public:
    SurfaceAovVisitor(
        const SurfacePoint &point,
        std::size_t capacity,
        const SurfaceClosureAovCallable &aov) noexcept;

    [[nodiscard]] const SurfaceAov &result() const noexcept;
};

// Combined production reduction over one raw physical-closure sequence.
// Runtime flags and the optional camera AOV are deliberately reduced in the
// same host-scheduled pass, after GraphSurface has evaluated its typed value
// graph exactly once.
class SurfacePreparationVisitor final
    : public SurfaceClosureExpressionVisitor {

  private:
    const SurfacePoint &_point;
    Expr<float> _glossy_filter_roughness;
    Expr<bool> _include_runtime_flags;
    Expr<bool> _include_aov;
    const SurfaceClosureIdentityCallable &_identity;
    const SurfaceClosureAovCallable &_aov_operation;
    UInt _runtime_flags{0u};
    Float3 _shading_normal;
    SurfaceAov _aov;

  protected:
    void visit(
        Expr<luisa::float3> shading_normal,
        const luisa::vector<SurfaceClosureExpression>
            &closures) noexcept override;

  public:
    SurfacePreparationVisitor(
        const SurfacePoint &point,
        Expr<float> glossy_filter_roughness,
        Expr<bool> include_runtime_flags,
        Expr<bool> include_aov,
        std::size_t capacity,
        const SurfaceClosureIdentityCallable &identity,
        const SurfaceClosureAovCallable &aov_operation) noexcept;

    [[nodiscard]] Expr<std::uint32_t>
    runtime_flags() const noexcept;
    [[nodiscard]] Expr<luisa::float3>
    shading_normal() const noexcept;
    [[nodiscard]] const SurfaceAov &aov() const noexcept;
};

}// namespace psycles::luisa_backend

#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_evaluation.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface_closure_visitor.h>

#include <luisa/dsl/struct.h>

namespace psycles::luisa_backend {

// Additive image-space and probability contribution of one canonical
// closure. The full surface evaluation is the ordered fold of these values;
// the callable ABI therefore contains no device pointer, dynamic closure
// index, or hidden mutable state.
struct SurfaceClosureEvaluationContributionCall {
    luisa::float3 f{};
    luisa::float3 diffuse_f{};
    luisa::float3 glossy_f{};
    float total_sample_weight{};
    float weighted_pdf{};
    float weighted_roughness_squared{};
    luisa::uint events{};
};

}// namespace psycles::luisa_backend

LUISA_STRUCT(
    psycles::luisa_backend::SurfaceClosureEvaluationContributionCall,
    f,
    diffuse_f,
    glossy_f,
    total_sample_weight,
    weighted_pdf,
    weighted_roughness_squared,
    events) {};

namespace psycles::luisa_backend {

// Inclusion and MIS policy derived once from the evaluation mode. Regular
// and sampled-BSDF evaluations include every enabled closure and preserve the
// mixture PDF; sampled-light evaluation applies Cycles' shader exclusions and
// shader_use_mis gate.
struct SurfaceClosureEvaluationPolicy {
    Bool diffuse_included;
    Bool glossy_included;
    Bool glass_included;
    Bool transmission_included;
    Bool preserve_pdf;
};

struct SurfaceClosureEvaluationDirections {
    Float3 incoming;
    Float3 outgoing;
};

[[nodiscard]] SurfaceClosureEvaluationDirections
make_surface_closure_evaluation_directions(
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> outgoing) noexcept;

[[nodiscard]] SurfaceClosureEvaluationPolicy
make_surface_closure_evaluation_policy(
    bool sampled_light,
    Expr<std::uint32_t> light_shader_flags) noexcept;

// Canonical per-closure algebra shared by the legacy Local evaluator and the
// branch-local callable visitor. incoming and outgoing must be normalized by
// the caller so normalization is performed once per surface evaluation.
[[nodiscard]] luisa::compute::Var<
    SurfaceClosureEvaluationContributionCall>
surface_closure_evaluation_contribution(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> shading_normal,
    const SurfaceClosureRecord &closure,
    Expr<luisa::float3> incoming,
    Expr<luisa::float3> outgoing,
    const SurfaceQuery &query,
    const SurfaceClosureEvaluationPolicy &policy,
    Expr<bool> selected_sample) noexcept;

// Ordered additive fold over per-closure contributions followed by the one
// non-linear normalization step. The visitor preserves Cycles allocation
// order, including floating-point accumulation order. Keeping this reduction
// shared prevents the Local oracle and callable visitor from drifting.
class SurfaceClosureEvaluationAccumulator {

  private:
    SurfaceEvaluation _result;
    Float _total_sample_weight{0.0f};
    Float _weighted_pdf{0.0f};
    Float _weighted_roughness_squared{0.0f};
    UInt _events{0u};

  public:
    SurfaceClosureEvaluationAccumulator() noexcept;

    void add(
        const luisa::compute::Var<
            SurfaceClosureEvaluationContributionCall>
            &contribution) noexcept;

    [[nodiscard]] SurfaceEvaluation finish(
        Expr<bool> preserve_pdf) const noexcept;
};

// Host/JIT-stage operation boundary. Implementations may call a nested Luisa
// Callable, while the visitor remains independent of scene resource types.
// Dynamic C++ dispatch happens only while recording the shader AST.
class SurfaceClosureEvaluationOperation {

  public:
    virtual ~SurfaceClosureEvaluationOperation() noexcept = default;

    // Bind the one outgoing direction shared by the ordered contribution
    // fold. Sampling operations call this after categorical inversion; light
    // evaluation calls it before material dispatch. Implementations must emit
    // an ordinary DSL assignment, not retain a host-side material value.
    virtual void set_outgoing(
        Expr<luisa::float3> outgoing) noexcept = 0;

    [[nodiscard]] virtual luisa::compute::Var<
        SurfaceClosureEvaluationContributionCall>
    evaluate(
        Expr<luisa::float3> shading_normal,
        const SurfaceClosureExpression &closure,
        Expr<bool> selected_sample) const noexcept = 0;
};

// Direct expression implementation used when a Surface owns its shader
// services in the current kernel. Keeping this behind the same operation
// interface as the resource-packed path makes both routes execute the exact
// same per-closure algebra while Luisa records the AST.
class DirectSurfaceClosureEvaluationOperation final
    : public SurfaceClosureEvaluationOperation {

  private:
    const ShaderServices &_services;
    SurfaceClosurePoint _point;
    const SurfaceQuery &_query;
    const SurfaceClosureEvaluationPolicy &_policy;
    Float3 _incoming{make_float3(0.0f)};
    Float3 _outgoing{make_float3(0.0f)};

  public:
    DirectSurfaceClosureEvaluationOperation(
        const ShaderServices &services,
        const SurfaceClosurePoint &point,
        const SurfaceQuery &query,
        const SurfaceClosureEvaluationPolicy &policy) noexcept;

    void set_outgoing(
        Expr<luisa::float3> outgoing) noexcept override;

    [[nodiscard]] luisa::compute::Var<
        SurfaceClosureEvaluationContributionCall>
    evaluate(
        Expr<luisa::float3> shading_normal,
        const SurfaceClosureExpression &closure,
        Expr<bool> selected_sample) const noexcept override;
};

class SurfaceClosureEvaluationVisitor final
    : public SurfaceClosureExpressionVisitor {

  private:
    const SurfaceClosureEvaluationOperation &_operation;
    Expr<bool> _preserve_pdf;
    Expr<std::uint32_t> _selected_closure_index;
    SurfaceEvaluation _result;

  protected:
    void visit(
        Expr<luisa::float3> shading_normal,
        const luisa::vector<SurfaceClosureExpression>
            &closures) noexcept override;

  public:
    SurfaceClosureEvaluationVisitor(
        std::size_t capacity,
        const SurfaceClosureEvaluationOperation &operation,
        Expr<bool> preserve_pdf,
        Expr<std::uint32_t> selected_closure_index =
            ~std::uint32_t{0u}) noexcept;

    [[nodiscard]] const SurfaceEvaluation &result() const noexcept;
};

}// namespace psycles::luisa_backend

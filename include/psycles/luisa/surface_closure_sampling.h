#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface_closure_sampling.h> through the Psycles::luisa target."
#endif

#include <psycles/luisa/surface_closure_evaluation.h>

#include <luisa/dsl/struct.h>

namespace psycles::luisa_backend {

struct SurfaceClosurePhysicalCommonRecord;

// Strongly typed, expression-only projection of the closure fields used by
// categorical selection. This is intentionally smaller than the conditional
// BSDF record: graph expansion retains the original socket types and only
// projects the values consumed by p(i) across a shared callable boundary.
struct SurfaceClosureSelectionInput {
    Expr<std::uint32_t> kind;
    Expr<std::uint32_t> lobe;
    Expr<std::uint32_t> bssrdf_method;
    Expr<float> allocation_weight;
    Expr<float> sample_weight;
    Expr<bool> setup_valid;
    Expr<luisa::float3> normal;
    Expr<float> roughness;
    Expr<bool> preserve_ggx_energy;
    Expr<bool> beckmann;
};

// The exact query projection needed by closure selection. Surface closure
// setup has already converted the authored normal into the final Cycles
// ShaderClosure normal before this boundary, so categorical selection must
// not observe geometry or repeat normal correction.
struct SurfaceClosureSelectionContext {
    Expr<std::uint32_t> lobe_mask;
    Expr<float> glossy_filter_roughness;
};

// The complete device-stage projection needed before categorical inversion.
// It is deliberately independent of storage: both a Local-array evaluator and
// a branch-local visitor consume this exact result.
struct SurfaceClosureSelectionCall {
    float weight{};
    luisa::float3 glossy_normal{};
    luisa::uint runtime_flags{};
    luisa::uint closure_type{};
    float closure_sample_weight{};
};

// Result of the conditional p(w_i | i) sampler for one already selected
// closure. The categorical choice and closure metadata remain outside this
// payload, making it impossible for the conditional sampler to change p(i).
struct SurfaceClosureConditionalSampleCall {
    luisa::float3 direction{};
    luisa::float2 roughness{};
    luisa::float3 singular_evaluation{};
    float singular_pdf{};
    float eta{};
    luisa::uint properties{};
    luisa::uint bssrdf_method{};
    luisa::float3 bssrdf_radius{};
    luisa::float3 bssrdf_albedo{};
    luisa::float3 bssrdf_normal{};
    float bssrdf_ior{};
    float bssrdf_roughness{};
    float bssrdf_anisotropy{};
    bool valid{};
};

}// namespace psycles::luisa_backend

LUISA_STRUCT(
    psycles::luisa_backend::SurfaceClosureSelectionCall,
    weight,
    glossy_normal,
    runtime_flags,
    closure_type,
    closure_sample_weight) {};

LUISA_STRUCT(
    psycles::luisa_backend::SurfaceClosureConditionalSampleCall,
    direction,
    roughness,
    singular_evaluation,
    singular_pdf,
    eta,
    properties,
    bssrdf_method,
    bssrdf_radius,
    bssrdf_albedo,
    bssrdf_normal,
    bssrdf_ior,
    bssrdf_roughness,
    bssrdf_anisotropy,
    valid) {};

namespace psycles::luisa_backend {

namespace surface_closure_sample_property {
inline constexpr std::uint32_t transparent = 1u << 0u;
inline constexpr std::uint32_t translucent = 1u << 1u;
inline constexpr std::uint32_t glossy = 1u << 2u;
inline constexpr std::uint32_t glass = 1u << 3u;
inline constexpr std::uint32_t transmission = 1u << 4u;
inline constexpr std::uint32_t singular = 1u << 5u;
inline constexpr std::uint32_t bssrdf = 1u << 6u;
}// namespace surface_closure_sample_property

[[nodiscard]] Float3 make_surface_closure_sampling_incoming(
    const SurfaceClosurePoint &point) noexcept;

[[nodiscard]] SurfaceClosureSelectionInput
make_surface_closure_selection_input(
    const SurfaceClosurePhysicalRecord &closure) noexcept;

// The categorical p(i) projection is a function of the tagged-union header
// alone. Keeping this overload independent of the physical payload prevents
// selection from extending mutually exclusive family values across both
// inverse-CDF passes.
[[nodiscard]] SurfaceClosureSelectionInput
make_surface_closure_selection_input(
    const SurfaceClosurePhysicalCommonRecord &closure) noexcept;

[[nodiscard]] SurfaceClosureSelectionInput
make_surface_closure_selection_input(
    const SurfaceClosureExpression &closure) noexcept;

[[nodiscard]] SurfaceClosureSelectionContext
make_surface_closure_selection_context(
    const SurfaceQuery &query) noexcept;

// Canonical p(i) state over an already populated physical closure.
[[nodiscard]] luisa::compute::Var<SurfaceClosureSelectionCall>
surface_closure_selection(
    const SurfaceClosureSelectionContext &context,
    const SurfaceClosureSelectionInput &closure) noexcept;

// Canonical conditional sampler p(w_i | i). It must only be invoked under the
// categorical `choose` predicate. In particular, this function never decides
// which closure was selected and never renormalizes the lobe dimension.
[[nodiscard]] luisa::compute::Var<
    SurfaceClosureConditionalSampleCall>
surface_closure_conditional_sample(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> shading_normal,
    const SurfaceClosurePhysicalRecord &closure,
    Expr<luisa::float3> incoming,
    Expr<luisa::float3> glossy_normal,
    Expr<luisa::float2> random_direction,
    Expr<float> rescaled_lobe,
    const SurfaceQuery &query,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;

// Conditional-sampling eliminator for the encoded physical tagged union.
// Like the evaluation counterpart, this merges only the compact sample call;
// inactive family payloads never enter the merged SSA state.
[[nodiscard]] luisa::compute::Var<SurfaceClosureConditionalSampleCall>
surface_closure_conditional_sample_from_physical_blocks(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> shading_normal,
    Expr<luisa::float4x4> block_0,
    Expr<luisa::float4x4> block_1,
    Expr<luisa::float3> incoming,
    Expr<luisa::float3> glossy_normal,
    Expr<luisa::float2> random_direction,
    Expr<float> rescaled_lobe,
    const SurfaceQuery &query,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;

// Storage-aware conditional-sampling eliminator. The loader is evaluated in
// exactly one payload family branch and never for a common-only closure.
[[nodiscard]] luisa::compute::Var<SurfaceClosureConditionalSampleCall>
surface_closure_conditional_sample_from_physical_common(
    const ShaderServices &services,
    const SurfaceClosurePoint &point,
    Expr<luisa::float3> shading_normal,
    const SurfaceClosurePhysicalCommonRecord &common,
    const SurfaceClosurePhysicalPayloadLoader &load_payload,
    Expr<luisa::float3> incoming,
    Expr<luisa::float3> glossy_normal,
    Expr<luisa::float2> random_direction,
    Expr<float> rescaled_lobe,
    const SurfaceQuery &query,
    SurfaceClosureReachability reachability =
        all_surface_closure_reachability) noexcept;

// First pass: construct the finite categorical measure over the retained
// Cycles allocation sequence. Retained count includes setup-invalid entries,
// exactly like ShaderData::num_closure; their selection weight remains zero.
class SurfaceClosureSelectionMeasure {

  private:
    Float _total_weight{0.0f};
    UInt _runtime_flags{0u};
    UInt _retained_count{0u};

  public:
    explicit SurfaceClosureSelectionMeasure(
        Expr<bool> back_facing) noexcept;

    void add(
        const luisa::compute::Var<
            SurfaceClosureSelectionCall> &selection) noexcept;

    // Predicated form for statically scheduled closure lists. Keeping the
    // retention predicate in SSA avoids one control-flow merge per closure
    // while preserving Cycles' exact retained-prefix measure.
    void add(
        const luisa::compute::Var<
            SurfaceClosureSelectionCall> &selection,
        Expr<bool> retained) noexcept;

    [[nodiscard]] Expr<float> total_weight() const noexcept;
    [[nodiscard]] Expr<std::uint32_t> runtime_flags() const noexcept;
    [[nodiscard]] Expr<std::uint32_t> retained_count() const noexcept;
};

struct SurfaceClosureCategoricalChoice {
    Bool choose;
    Float rescaled;
};

// Second pass: exact inverse-CDF state machine. For retained closure i with
// mass s_i and total W, consider() returns the unique interval predicate and
// (u W - sum_{j<i} s_j) / s_i. No BSDF sampling code is part of this state.
class SurfaceClosureCategoricalInversion {

  private:
    Float _random_lobe{0.0f};
    Float _target{0.0f};
    Float _accumulated{0.0f};
    UInt _retained_count{0u};
    Bool _selected{false};

  public:
    SurfaceClosureCategoricalInversion(
        Expr<float> random_lobe,
        const SurfaceClosureSelectionMeasure &measure) noexcept;

    [[nodiscard]] SurfaceClosureCategoricalChoice consider(
        const luisa::compute::Var<
            SurfaceClosureSelectionCall> &selection) noexcept;

    [[nodiscard]] SurfaceClosureCategoricalChoice consider(
        const luisa::compute::Var<
            SurfaceClosureSelectionCall> &selection,
        Expr<bool> retained) noexcept;

    [[nodiscard]] Expr<bool> selected() const noexcept;
};

// Scalar state retained after the selected conditional sampler executes. It
// contains no closure array and therefore has no runtime-indexed aggregate
// load. finish() performs the one common Cycles delta/MIS composition.
class SurfaceClosureSelectedSample {

  private:
    Bool _selected{false};
    UInt _closure_index{~std::uint32_t{0u}};
    UInt _closure_type{0u};
    Float _closure_sample_weight{0.0f};
    Float _selection_rescaled{0.0f};
    Float3 _closure_weight{make_float3(0.0f)};
    Float3 _closure_normal{make_float3(0.0f, 0.0f, 1.0f)};
    Float _selected_weight{0.0f};
    Float3 _direction{make_float3(0.0f, 0.0f, 1.0f)};
    Float2 _roughness{make_float2(0.0f)};
    Float3 _singular_evaluation{make_float3(0.0f)};
    Float _singular_pdf{0.0f};
    Float _eta{1.0f};
    UInt _properties{0u};
    UInt _bssrdf_method{0u};
    Float3 _bssrdf_radius{make_float3(0.0f)};
    Float3 _bssrdf_albedo{make_float3(0.0f)};
    Float3 _bssrdf_normal{make_float3(0.0f, 0.0f, 1.0f)};
    Float _bssrdf_ior{1.4f};
    Float _bssrdf_roughness{1.0f};
    Float _bssrdf_anisotropy{0.0f};
    Bool _candidate_valid{true};

  public:
    SurfaceClosureSelectedSample() noexcept = default;

    // Called only inside the categorical choice branch.
    void accept(
        Expr<std::uint32_t> closure_index,
        Expr<luisa::float3> closure_weight,
        Expr<luisa::float3> closure_normal,
        Expr<float> selection_rescaled,
        const luisa::compute::Var<
            SurfaceClosureSelectionCall> &selection,
        const luisa::compute::Var<
            SurfaceClosureConditionalSampleCall> &sample) noexcept;

    [[nodiscard]] Expr<bool> selected() const noexcept;
    [[nodiscard]] Expr<std::uint32_t>
    closure_index() const noexcept;
    [[nodiscard]] Expr<luisa::float3> direction() const noexcept;

    [[nodiscard]] SurfaceSampleTrace finish(
        const SurfaceClosurePoint &point,
        const SurfaceClosureSelectionMeasure &measure,
        const SurfaceEvaluation &mixture_evaluation,
        bool trace_selection) const noexcept;
};

// Host/JIT-stage sampling component. Dynamic C++ dispatch selects the
// resource-bearing callable implementation while the shader AST is recorded.
class SurfaceClosureSamplingOperation {

  public:
    virtual ~SurfaceClosureSamplingOperation() noexcept = default;

    // Pure device expression: recording a selection must not mutate
    // device-visible state. The sampling visitor schedules this expression
    // once per reachable closure and reuses it across measure construction
    // and categorical inversion.
    [[nodiscard]] virtual luisa::compute::Var<
        SurfaceClosureSelectionCall>
    selection(
        const SurfaceClosureExpression &closure) const noexcept = 0;

    [[nodiscard]] virtual luisa::compute::Var<
        SurfaceClosureConditionalSampleCall>
    conditional_sample(
        Expr<luisa::float3> shading_normal,
        const SurfaceClosureExpression &closure,
        Expr<luisa::float3> glossy_normal,
        Expr<luisa::float2> random_direction,
        Expr<float> rescaled_lobe) const noexcept = 0;
};

// Direct expression implementation paired with
// DirectSurfaceClosureEvaluationOperation. Runtime material values remain
// typed Luisa expressions; C++ virtual dispatch only chooses how the common
// selection and conditional-sampling AST is emitted.
class DirectSurfaceClosureSamplingOperation final
    : public SurfaceClosureSamplingOperation {

  private:
    const ShaderServices &_services;
    SurfaceClosurePoint _point;
    const SurfaceQuery &_query;
    SurfaceClosureSelectionContext _selection_context;
    Float3 _incoming{make_float3(0.0f)};
    SurfaceClosureReachability _reachability;

  public:
    DirectSurfaceClosureSamplingOperation(
        const ShaderServices &services,
        const SurfaceClosurePoint &point,
        const SurfaceQuery &query,
        SurfaceClosureReachability reachability =
            all_surface_closure_reachability) noexcept;

    [[nodiscard]] luisa::compute::Var<
        SurfaceClosureSelectionCall>
    selection(
        const SurfaceClosureExpression &closure) const noexcept override;

    [[nodiscard]] luisa::compute::Var<
        SurfaceClosureConditionalSampleCall>
    conditional_sample(
        Expr<luisa::float3> shading_normal,
        const SurfaceClosureExpression &closure,
        Expr<luisa::float3> glossy_normal,
        Expr<luisa::float2> random_direction,
        Expr<float> rescaled_lobe) const noexcept override;
};

// Three-pass branch-local implementation of the formal product measure:
// schedule p(i) once, construct and invert the measure, execute one
// p(w_i | i), then evaluate the complete retained mixture at the chosen
// direction.
class SurfaceClosureSamplingVisitor final
    : public SurfaceClosureExpressionVisitor {

  private:
    SurfaceClosurePoint _point;
    const SurfaceClosureSamplingOperation &_sampling;
    SurfaceClosureEvaluationOperation &_evaluation;
    Expr<float> _random_lobe;
    Expr<luisa::float2> _random_direction;
    bool _trace_selection;
    SurfaceSampleTrace _result;

  protected:
    void visit(
        Expr<luisa::float3> shading_normal,
        const luisa::vector<SurfaceClosureExpression>
            &closures) noexcept override;

  public:
    SurfaceClosureSamplingVisitor(
        std::size_t capacity,
        const SurfaceClosurePoint &point,
        const SurfaceClosureSamplingOperation &sampling,
        SurfaceClosureEvaluationOperation &evaluation,
        Expr<float> random_lobe,
        Expr<luisa::float2> random_direction,
        bool trace_selection) noexcept;

    [[nodiscard]] const SurfaceSampleTrace &result() const noexcept;
};

}// namespace psycles::luisa_backend

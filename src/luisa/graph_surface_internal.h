#pragma once

#include "surface_color_encoding.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface.h>
#include <psycles/luisa/surface_closure_identity.h>
#include <psycles/luisa/surface_closure_physical_blocks.h>
#include <psycles/luisa/surface_closure_reachability.h>

#include "graph_surface_value_expression.h"
#include "surface_color_nodes.h"
#include "surface_fresnel.h"
#include "surface_math.h"
#include "surface_math_constants.h"
#include "surface_vector_mapping.h"

#include <luisa/core/stl/vector.h>

namespace psycles::luisa_backend::detail {

inline constexpr std::uint32_t camera_ray_visibility = 1u << 0u;
inline constexpr std::uint32_t diffuse_ray_visibility = 1u << 1u;
inline constexpr std::uint32_t glossy_ray_visibility = 1u << 2u;
inline constexpr std::uint32_t transmission_ray_visibility = 1u << 3u;
inline constexpr std::uint32_t shadow_ray_visibility = 1u << 4u;
inline constexpr std::uint32_t volume_ray_visibility = 1u << 5u;

struct TracedValues {
    luisa::vector<SurfaceValueExpression> values;
    Float3 shading_normal;
};

// A Principled graph leaf is expanded into the same physical closure
// order as Cycles before any trace, AOV, evaluation, or sampling query
// consumes it. The tag is host-stage metadata used while Luisa records
// the shader AST.
enum class PrincipledLobe : std::uint8_t {
    none,
    sheen,
    coat,
    metallic,
    transmission,
    dielectric
};

// Exact device identity produced by one successful Cycles-compatible setup
// routine. This is host-stage ownership of Luisa expressions, not an
// additional device aggregate. Keeping it optional distinguishes a raw graph
// closure from a setup closure without manufacturing default device values.
struct TracedCyclesClosureIdentity {
    UInt closure_type;
    UInt microfacet_fresnel;
};

struct TracedClosure {
    compiler::ClosureOperation operation{
        compiler::ClosureOperation::diffuse};
    // A graph operation may expand into a more specific Cycles physical
    // closure. `none` delegates to `operation`; every other value is an
    // immutable host-stage type tag while all authored values remain Luisa
    // device expressions.
    SurfaceClosureKind physical_kind{SurfaceClosureKind::none};
    PrincipledLobe principled_lobe{PrincipledLobe::none};
    compiler::PrincipledClosureFeatureMask principled_features{};
    // Empty for raw graph closures. Exactly one setup component must populate
    // this before the closure crosses the physical retention boundary.
    std::optional<TracedCyclesClosureIdentity> cycles_identity;
    Float3 weight;
    // Allocation and sampling are distinct in Cycles: Fresnel setup may
    // reduce sample_weight after a closure has already been allocated.
    Float allocation_weight;
    Float sample_weight;
    // Setup validity is independent of allocation. Some Cycles setup
    // routines retain an occupied closure slot while changing its type to
    // CLOSURE_NONE and clearing sample_weight (for example an invalid Sheen
    // LTC). Keeping the states orthogonal preserves closure indices and the
    // random-dimension rescaling contract.
    Bool setup_valid;
    // Cycles' best-effort closure albedo, including
    // ShaderClosure::weight. This drives closure selection and the
    // Diff/Gloss/Trans color passes.
    Float3 albedo;
    // Generalized-Schlick glass keeps its two physical lobes separate.
    // This is required both for spectral lobe selection and for Cycles'
    // Glossy/Transmission Color passes; re-splitting a combined albedo with
    // a scalar dielectric Fresnel loses authored tint information.
    Float3 reflection_albedo;
    Float3 transmission_albedo;
    Float3 color;
    Float3 normal;
    Float roughness;
    Float diffuse_roughness;
    Float subsurface_weight;
    Float3 subsurface_radius;
    Float subsurface_scale;
    compiler::BssrdfMethod subsurface_method{
        compiler::BssrdfMethod::random_walk};
    Float subsurface_ior;
    Float subsurface_anisotropy;
    Float transmission_weight;
    Float metallic;
    Float ior;
    Float specular_ior_level;
    Float3 specular_tint;
    // Authored anisotropy inputs. `anisotropy_enabled` is host/JIT topology
    // metadata proven by SurfaceClosurePlan; when false none of these device
    // expressions are read or recorded.
    bool anisotropy_enabled{};
    Float anisotropy;
    Float anisotropic_rotation;
    Float3 tangent;
    bool hair_tangent_linked{};
    // Cycles legacy Hair's signed longitudinal shift after setup. Tangent
    // linkage stays host/JIT metadata; this field is a device expression.
    Float hair_offset;
    // Physical microfacet state. Setup initializes this for every emitted
    // closure; scattering observes only this common representation and never
    // reinterprets the authored node parameterization.
    Float3 microfacet_tangent;
    Float microfacet_alpha_x;
    Float microfacet_alpha_y;
    bool microfacet_state_configured{};
    Float alpha;
    Bool thin_wall{false};
    Float sheen_weight;
    Float sheen_roughness;
    Float3 sheen_tint;
    Float coat_weight;
    Float coat_roughness;
    Float coat_ior;
    Float3 coat_tint;
    Float3 coat_normal;
    bool coat_normal_linked{};
    // Linearly transformed cosine parameters for a physical Principled
    // Sheen closure. They are device values loaded from Cycles' versioned
    // table by the ordered host-stage layer component.
    Float sheen_transform_a;
    Float sheen_transform_b;
    // Raw authored Principled emission. Closure-tree Mix/Add and
    // Principled layer weights are applied by the emission component.
    Float3 emission;
    // Host/JIT capability and device values are intentionally separate.
    // When the capability is false, no thin-film table lookup or arithmetic
    // is recorded in the material variant at all.
    bool thin_film_enabled{};
    Float thin_film_thickness;
    Float thin_film_ior;
    // Microfacet multiple-scattering scale after any weight darkening
    // has already been applied to `weight`.
    Float3 evaluation_scale;
    Float3 fresnel_f0;
    Float3 fresnel_f90;
    Float3 reflection_tint;
    Float3 transmission_tint;
    bool preserve_ggx_energy{};
    bool beckmann{};
};

// Complete the setup transition once. `successful_type` and
// `successful_fresnel` are chosen by the setup component which owns the
// corresponding Cycles algorithm; a failed setup has the single canonical
// identity (NONE, NONE). The single-assignment assertion prevents a later
// layer from silently reclassifying an already-setup closure.
inline void set_cycles_closure_identity_after_setup(
    TracedClosure &closure,
    UInt successful_type,
    UInt successful_fresnel = static_cast<std::uint32_t>(
        cycles_closure::MicrofacetFresnel::none)) noexcept {
    LUISA_ASSERT(
        !closure.cycles_identity.has_value(),
        "Cycles closure identity may only be produced once per setup.");
    closure.cycles_identity.emplace(TracedCyclesClosureIdentity{
        .closure_type = luisa::compute::select(
            UInt{cycles_closure::type_none},
            successful_type,
            closure.setup_valid),
        .microfacet_fresnel = luisa::compute::select(
            UInt{static_cast<std::uint32_t>(
                cycles_closure::MicrofacetFresnel::none)},
            successful_fresnel,
            closure.setup_valid)});
}

using ClosureVisitor = std::function<void(const TracedClosure &)>;

struct SurfaceClosureReachabilityIdentity {
    SurfaceClosureKind kind{SurfaceClosureKind::none};
    SurfaceClosureLobe lobe{SurfaceClosureLobe::none};
};

// Resolve only the immutable host-stage reachability tags. These tags decide
// which family branches Luisa records, but never reconstruct the retained
// ClosureType/Fresnel pair produced by setup.
[[nodiscard]] SurfaceClosureReachabilityIdentity
surface_closure_reachability_identity(
    const TracedClosure &closure) noexcept;

// Project a host-tagged setup closure into the canonical device-tagged
// physical record consumed by every directional scattering component.
[[nodiscard]] SurfaceClosureRecord canonical_surface_closure(
    const TracedClosure &closure) noexcept;

// Fixed semantic expansion of one raw graph leaf into its Cycles-compatible
// physical closure sequence. Closure-tree traversal and transparent merging
// are deliberately outside this component so expanded graphs and the compact
// bytecode interpreter share exactly one setup implementation.
void expand_physical_surface_closure(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedClosure &graph_closure,
    Bool reflective_caustics,
    Bool refractive_caustics,
    const ClosureVisitor &emit) noexcept;

struct GlassSample {
    Float3 direction;
    Float3 singular_evaluation;
    Float singular_pdf;
    Float eta;
    Float alpha;
    Bool transmission;
    Bool singular;
    Bool valid;
};

// A directional microfacet query has one shared geometric measure. Returning
// the BSDF value and PDF together keeps the half-vector, distribution, and
// masking-shadowing terms in one expression DAG, matching Cycles' eval ABI.
struct MicrofacetEvaluation {
    Float3 intensity;
    Float pdf;
    Float roughness_squared;
};

struct MicrofacetDistributionTerms {
    Float distribution;
    Float lambda_incoming;
    Float lambda_outgoing;
};

// Result of sampling a reflection-only microfacet closure. Regular samples
// are evaluated by the aggregate evaluator so competing closures share one
// balance-heuristic denominator. Delta samples carry the selected closure's
// singular numerator explicitly because their directional eval is zero.
struct MicrofacetReflectionSample {
    Float3 direction;
    Float3 singular_evaluation;
    Float singular_pdf;
    Float2 roughness;
    Bool singular;
    Bool valid;
};

struct AdjustedIor {
    Float eta;
    Float f0;
};

struct GgxEnergy {
    Float3 darkening;
    Float3 energy_scale;
};

struct TransparentClosureState {
    Float3 weight;
    Float sample_weight;
};

template <typename Id, typename Values>
[[nodiscard]] const auto &get(Id id, const Values &values) noexcept {
    return values[id.value];
}

[[nodiscard]] Float scalar(compiler::ValueExpressionId id,
    const TracedValues &values) noexcept;
[[nodiscard]] Float3 vector(compiler::ValueExpressionId id,
    const TracedValues &values) noexcept;
[[nodiscard]] ULong unsigned_integer(compiler::ValueExpressionId id,
    const TracedValues &values) noexcept;
[[nodiscard]] Float sample_weight(Float3 value) noexcept;
[[nodiscard]] TransparentClosureState transparent_closure_state(
    Float3 weight) noexcept;
[[nodiscard]] Float3 bsdf_allocated_weight(Float3 value) noexcept;
[[nodiscard]] Float pass_weight(Float3 value) noexcept;
[[nodiscard]] Float max_component(Float3 value) noexcept;
[[nodiscard]] Float cycles_table_1d(const ShaderServices &services,
    Float x,
    Expr<std::uint32_t> offset,
    std::uint32_t size) noexcept;
[[nodiscard]] Float cycles_table_2d(const ShaderServices &services,
    Float x,
    Float y,
    Expr<std::uint32_t> offset,
    std::uint32_t x_size,
    std::uint32_t y_size) noexcept;
[[nodiscard]] Float cycles_table_3d(const ShaderServices &services,
    Float x,
    Float y,
    Float z,
    Expr<std::uint32_t> offset,
    std::uint32_t x_size,
    std::uint32_t y_size,
    std::uint32_t z_size) noexcept;
[[nodiscard]] Float f0_from_ior(Float ior) noexcept;
[[nodiscard]] Float ior_from_f0(Float f0) noexcept;
[[nodiscard]] Float fresnel_dielectric_fss(Float eta) noexcept;
[[nodiscard]] AdjustedIor adjusted_ior(
    const TracedClosure &closure) noexcept;
[[nodiscard]] AdjustedIor adjusted_ior(
    Float ior, Float specular_ior_level) noexcept;
[[nodiscard]] Float3 generalized_dielectric_fresnel(
    Float cosine, Float eta, Float3 f0) noexcept;
[[nodiscard]] Float3 fresnel_f82_b(Float3 f0, Float3 tint) noexcept;
[[nodiscard]] Float3 fresnel_f82(
    Float cosine, Float3 f0, Float3 b) noexcept;
[[nodiscard]] Float3 fresnel_conductor(
    Float cosine, Float3 ior, Float3 extinction) noexcept;
[[nodiscard]] Float3 fresnel_conductor_fss(
    Float3 ior, Float3 extinction) noexcept;
[[nodiscard]] Float3 ensure_valid_specular_reflection(
    Float3 geometric_normal,
    Float3 incoming,
    Float3 shading_normal) noexcept;
[[nodiscard]] Float3 maybe_ensure_valid_specular_reflection(
    const SurfaceClosurePoint &point,
    Float3 incoming,
    Float3 shading_normal) noexcept;
[[nodiscard]] GgxEnergy ggx_energy(const ShaderServices &services,
    const TracedClosure &closure,
    Float incoming_cosine,
    Float3 fss) noexcept;
[[nodiscard]] GgxEnergy ggx_energy(const ShaderServices &services,
    Float roughness,
    bool preserve_ggx_energy,
    Float incoming_cosine,
    Float3 fss) noexcept;
[[nodiscard]] Float closure_sample_weight(
    const SurfaceClosurePhysicalCommonRecord &closure) noexcept;
// Exact Cycles bump_shadowing_term contract. `smooth_normal` is the final
// shader-wide sd->N; closure.normal may be an independently linked socket.
// A nonzero factor changes closure energy but never density. A zero factor
// rejects bsdf_eval's competing PDF, while the closure selected through
// bsdf_sample retains its original sampling PDF; EvaluationMode expresses
// that distinction at the aggregate evaluator.
[[nodiscard]] Float bump_shadowing_term(
    const SurfaceClosurePoint &point,
    Float3 smooth_normal,
    Float3 closure_normal,
    Bool diffuse_closure,
    Float3 direction,
    Bool is_evaluation) noexcept;
[[nodiscard]] Float oren_nayar_g(Float cosine) noexcept;
[[nodiscard]] Float3 diffuse_intensity(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float3 color,
    Float3 incoming,
    Float3 outgoing) noexcept;
// Evaluate the distribution-specific D and masking-shadowing terms under one
// device branch. This is the dynamic counterpart of Cycles' static
// MicrofacetType specialization: a lane executes exactly one model.
[[nodiscard]] MicrofacetDistributionTerms microfacet_distribution_terms(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float n_dot_h,
    Float n_dot_incoming,
    Float n_dot_outgoing,
    Float alpha) noexcept;
[[nodiscard]] Float microfacet_alpha(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float2 microfacet_alpha(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Bool microfacet_is_singular(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Bool microfacet_is_singular(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float glossy_filter_roughness) noexcept;
// Exact microfacet branch of Cycles' bsdf_get_specular_roughness_squared.
// The family eliminator owns the remaining classification, so this helper
// cannot re-introduce runtime kind tests after the tagged-union dispatch.
[[nodiscard]] Float microfacet_specular_roughness_squared(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float3 microfacet_reflection_fresnel(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float cosine,
    const ShaderServices *services,
    bool may_have_f82,
    bool may_have_f82_thin_film,
    bool may_have_dielectric,
    bool may_have_generalized_schlick,
    bool may_have_generalized_schlick_thin_film,
    bool may_have_conductor,
    bool may_have_conductor_thin_film) noexcept;
[[nodiscard]] MicrofacetEvaluation microfacet_evaluate(
    const ShaderServices &services,
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness,
    bool may_be_anisotropic,
    bool may_have_f82,
    bool may_have_f82_thin_film,
    bool may_have_dielectric,
    bool may_have_generalized_schlick,
    bool may_have_generalized_schlick_thin_film,
    bool may_have_conductor,
    bool may_have_conductor_thin_film) noexcept;
[[nodiscard]] MicrofacetReflectionSample sample_microfacet_reflection(
    const SurfaceClosurePoint &point,
    Float3 smooth_normal,
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float2 random,
    Float3 glossy_normal,
    Float glossy_filter_roughness,
    bool may_be_anisotropic,
    const ShaderServices *services,
    bool may_have_f82,
    bool may_have_f82_thin_film,
    bool may_have_dielectric,
    bool may_have_generalized_schlick,
    bool may_have_generalized_schlick_thin_film,
    bool may_have_conductor,
    bool may_have_conductor_thin_film) noexcept;
[[nodiscard]] Float sheen_intensity(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float3 outgoing) noexcept;
[[nodiscard]] Float3 sample_sheen(
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    Float2 random) noexcept;
struct AshikhminVelvetEvaluation {
    Float intensity;
    Float pdf;
    Bool valid;
};
[[nodiscard]] AshikhminVelvetEvaluation evaluate_ashikhmin_velvet(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    bool sampling_domain = false) noexcept;
[[nodiscard]] Float3 sample_uniform_hemisphere(
    Float3 normal,
    Float2 random) noexcept;
[[nodiscard]] Float3 sample_cosine_hemisphere(
    Float3 normal, Float2 random) noexcept;
class GraphSurfaceImplementation;

struct ValueStaticTableView {
    Expr<luisa::compute::BindlessArray> resources;
    std::uint32_t buffer_slot{};
    Expr<std::uint32_t> begin;
};

struct ValueEvaluationContext {
    const ShaderServices &services;
    const SurfacePoint &point;
    TracedValues &result;
    const GraphSurfaceImplementation *surface{};
    // Bytecode execution keeps table ParameterId as instruction data. The
    // ordinary topology-expanded path leaves this null and uses the host IR
    // binding; both paths call the same node implementation.
    const Expr<std::uint32_t> *parameter_override{};
    // Compact bytecode stores authored fixed-layout node data in a scene-wide
    // immutable stream. The host variant retains the exact table length, so
    // every constant index below is proven in range by lowering and semantic
    // variant interning. Expanded evaluation leaves this null and reads the
    // original instruction payload.
    const ValueStaticTableView *static_table_override{};
    // Rare unbounded opcode data which cannot fit the 14-bit hot immediate.
    // The unified SVM currently uses this only for Nishita's scene texture
    // binding index. It remains instruction data, never handler identity.
    const UInt *static_u0_override{};
    // Compact SVM execution owns selected immutable node fields as a validated
    // opcode-local instruction immediate. The exact host domain is supplied
    // so a dynamic handler records only modes reachable by its equivalence
    // class. Topology-expanded evaluation leaves both fields empty and keeps
    // the original statically specialized path.
    const UInt *svm_immediate_override{};
    std::span<const std::uint16_t> svm_immediate_domain{};
};

[[nodiscard]] inline Float value_static_table_entry(
    const ValueEvaluationContext &context,
    const compiler::ValueInstruction &instruction,
    std::size_t index) noexcept {
    if (context.static_table_override != nullptr) {
        return context.static_table_override->resources
            ->buffer<float>(
                context.static_table_override->buffer_slot,
                false,
                true)
            .read(context.static_table_override->begin +
                  static_cast<std::uint32_t>(index));
    }
    return instruction.static_table[index];
}

// Host-stage node interface. Implementations emit Luisa expressions
// when evaluate() is called; no instance of this hierarchy reaches
// device code.
class ValueNode {

private:
    const compiler::ValueInstruction *_instruction;

protected:
    [[nodiscard]] const compiler::ValueInstruction &
    instruction() const noexcept {
        return *_instruction;
    }

public:
    explicit ValueNode(
        const compiler::ValueInstruction &instruction) noexcept
        : _instruction{&instruction} {
    }
    virtual ~ValueNode() noexcept = default;

    [[nodiscard]] virtual SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept = 0;
};

[[nodiscard]] std::unique_ptr<ValueNode> make_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_math_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_context_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_image_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_normal_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode>
try_make_ambient_occlusion_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_bump_expanded_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_wave_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_magic_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_voronoi_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_procedural_value_node(
    const compiler::ValueInstruction &instruction) noexcept;

using VolumeVisitor =
    std::function<void(const compiler::VolumeInstruction &, Float)>;

class GraphSurfaceImplementation {

private:
    enum class EvaluationMode : std::uint8_t {
        regular,
        sampled_light,
        sampled_bsdf
    };

    struct EvaluationContext {
        EvaluationMode mode;
        Expr<std::uint32_t> light_shader_flags;
        Expr<std::uint32_t> selected_closure_index;
    };

    std::shared_ptr<const compiler::SurfaceProgram> _program;
    compiler::SurfaceClosurePlan _closure_plan;
    compiler::SurfaceValueDependencyPlan _value_dependency_plan;
    // Host/JIT abstract interpretation of this program's physical closure
    // image. It is computed from the validated closure plan once and threaded
    // into every generic evaluator/sampler, so absent capabilities cannot
    // materialize dead shader AST.
    SurfaceClosureReachability _physical_closure_reachability;
    SurfaceCapabilities _capabilities;
    std::vector<std::unique_ptr<ValueNode>> _value_nodes;
    std::vector<bool> _displacement_dependency_mask;
    std::vector<bool> _surface_normal_dependency_mask;
    bool _automatic_bump_uses_undisplaced_geometry{false};

    // Evaluate one compiler domain without applying the automatic
    // SetNormal stage. Cycles compiles bump and surface as consecutive SVM
    // regions; this primitive keeps authored Bump nodes pure.
    [[nodiscard]] TracedValues trace_value_stage(
        const ShaderServices &services,
        const SurfacePoint &point,
        const std::vector<bool> *active_mask) const noexcept;

    // Evaluate one consumer-specific surface domain while retaining Cycles'
    // automatic bump ordering. The dependency plan is topology-closed, so
    // every selected value is still emitted exactly once in program order.
    [[nodiscard]] TracedValues trace_surface_values(
        const ShaderServices &services,
        const SurfacePoint &point,
        const std::vector<bool> *active_mask) const noexcept;

    // Cycles' BOTH program wraps only its automatic bump region in
    // NODE_ENTER/LEAVE_BUMP_EVAL. The surface region must keep the truly
    // displaced position while inheriting the normal produced by this
    // temporary undisplaced state.
    [[nodiscard]] SurfacePoint automatic_bump_point(
        const SurfacePoint &point) const noexcept;

    [[nodiscard]] SurfaceEvaluation evaluate_traced(
        const ShaderServices &services,
        const TracedValues &values,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query,
        const EvaluationContext &context) const noexcept;

    [[nodiscard]] SurfaceSampleTrace sample_with_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query,
        bool trace_selection) const noexcept;

    [[nodiscard]] Float3 emission_traced(
        const ShaderServices &services,
        const SurfacePoint &point,
        const TracedValues &values,
        Expr<bool> reflective_caustics) const noexcept;

    [[nodiscard]] SurfaceClosureCollection collect_traced_closures(
        const ShaderServices &services,
        const SurfacePoint &point,
        const TracedValues &values,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics,
        SurfaceClosureCollector &collector) const noexcept;

public:
    [[nodiscard]] std::vector<bool> value_dependency_mask(
        compiler::ValueExpressionId root) const;

    [[nodiscard]] TracedValues trace_values(
        const ShaderServices &services,
        const SurfacePoint &point,
        const std::vector<bool> *active_mask = nullptr) const noexcept;

private:
    void for_each_closure(const TracedValues &values,
        const std::vector<bool> &closure_mask,
        const std::vector<bool> &endpoint_mask,
        const ClosureVisitor &visitor) const noexcept;
    void for_each_physical_closure(const ShaderServices &services,
        const SurfacePoint &point,
        const TracedValues &values,
        Bool reflective_caustics,
        Bool refractive_caustics,
        const ClosureVisitor &visitor) const noexcept;
    void for_each_volume(const TracedValues &values,
        const VolumeVisitor &visitor) const noexcept;

public:
    explicit GraphSurfaceImplementation(
        std::shared_ptr<const compiler::SurfaceProgram>
            program) noexcept;
    GraphSurfaceImplementation(
        std::shared_ptr<const compiler::SurfaceProgram> program,
        compiler::SurfaceClosurePlan closure_plan) noexcept;
    ~GraphSurfaceImplementation() noexcept;

    [[nodiscard]] SurfaceCapabilities capabilities() const noexcept;
    [[nodiscard]] SurfaceClosureCollection collect_closures(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics,
        SurfaceClosureCollector &collector) const noexcept;
    [[nodiscard]] SurfacePopulation populate(
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector) const noexcept;
    [[nodiscard]] SurfaceEvaluation evaluate(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept;
    [[nodiscard]] SurfaceEvaluation evaluate_light(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept;
    [[nodiscard]] SurfaceSample sample(const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept;
    [[nodiscard]] SurfacePreparation prepare(
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePreparationQuery &query) const noexcept;
    [[nodiscard]] SurfacePreparation prepare_traced_values(
        const ShaderServices &services,
        const SurfacePoint &point,
        const TracedValues &values,
        const SurfacePreparationQuery &query) const noexcept;
    [[nodiscard]] SurfaceClosureTrace closure_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<std::uint32_t> requested_index,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics) const noexcept;
    [[nodiscard]] SurfaceSampleTrace sample_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept;
    [[nodiscard]] Float3 emission(const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        Expr<bool> reflective_caustics) const noexcept;
    [[nodiscard]] Float3 constant_emission(
        const SurfaceParameterServices &services,
        Expr<std::uint32_t> parameter_block) const noexcept;
    [[nodiscard]] Float3 transparent_extinction(
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept;
    [[nodiscard]] VolumeCoefficients evaluate_volume(
        const ShaderServices &services,
        const SurfacePoint &point,
        const VolumeQuery &query,
        VolumePhaseCollector *collector) const noexcept;
    [[nodiscard]] Float3 displacement(
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept;
    [[nodiscard]] Float3 shading_normal(const ShaderServices &services,
        const SurfacePoint &point) const noexcept;
};

} // namespace psycles::luisa_backend::detail

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/surface.h>

#include "graph_surface_value_expression.h"

#include <luisa/core/stl/vector.h>

namespace psycles::luisa_backend::detail {

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float inverse_pi = 0.31830988618379067154f;
inline constexpr float two_pi = 6.28318530717958647692f;
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

struct TracedClosure {
    compiler::ClosureOperation operation{
        compiler::ClosureOperation::diffuse};
    PrincipledLobe principled_lobe{PrincipledLobe::none};
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
    Float alpha;
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

// Project a host-tagged setup closure into the canonical device-tagged
// physical record consumed by every directional scattering component.
[[nodiscard]] SurfaceClosureRecord canonical_surface_closure(
    const TracedClosure &closure) noexcept;

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

// Result of sampling a reflection-only microfacet closure. Regular samples
// are evaluated by the aggregate evaluator so competing closures share one
// balance-heuristic denominator. Delta samples carry the selected closure's
// singular numerator explicitly because their directional eval is zero.
struct MicrofacetReflectionSample {
    Float3 direction;
    Float3 singular_evaluation;
    Float singular_pdf;
    Float alpha;
    Bool singular;
    Bool valid;
};

// Minimal expression-only projection shared by closure classification
// callables. It preserves the exact allocation/setup identity needed by
// Cycles without materializing the complete scattering record.
struct SurfaceClosureIdentityExpression {
    Expr<std::uint32_t> kind;
    Expr<std::uint32_t> lobe;
    Expr<std::uint32_t> bssrdf_method;
    Expr<float> allocation_weight;
    Expr<bool> setup_valid;
    Expr<float> roughness;
    Expr<bool> preserve_ggx_energy;
    Expr<bool> beckmann;
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
[[nodiscard]] Float sample_weight(Float3 value) noexcept;
[[nodiscard]] TransparentClosureState transparent_closure_state(
    Float3 weight) noexcept;
[[nodiscard]] Float3 bsdf_allocated_weight(Float3 value) noexcept;
[[nodiscard]] Float pass_weight(Float3 value) noexcept;
[[nodiscard]] Float max_component(Float3 value) noexcept;
[[nodiscard]] Float srgb_to_linear(Float value) noexcept;
[[nodiscard]] Float3 srgb_to_linear(Float3 value) noexcept;
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
[[nodiscard]] Float3 safe_normalize(
    Float3 value, Float3 fallback) noexcept;
[[nodiscard]] Float3 rgb_to_hsv(Float3 rgb) noexcept;
[[nodiscard]] Float3 hsv_to_rgb(Float3 hsv) noexcept;
[[nodiscard]] Float3 rgb_to_hsl(Float3 rgb) noexcept;
[[nodiscard]] Float3 hsl_to_rgb(Float3 hsl) noexcept;
[[nodiscard]] Float3 separate_color(
    Float3 color, std::uint64_t mode) noexcept;
[[nodiscard]] Float3 combine_color(
    Float3 channels, std::uint64_t mode) noexcept;

[[nodiscard]] Float fresnel_dielectric_cos(
    Float cosine, Float eta) noexcept;
[[nodiscard]] Float f0_from_ior(Float ior) noexcept;
[[nodiscard]] Float ior_from_f0(Float f0) noexcept;
[[nodiscard]] Float fresnel_dielectric_fss(Float eta) noexcept;
[[nodiscard]] AdjustedIor adjusted_ior(
    const TracedClosure &closure) noexcept;
[[nodiscard]] Float3 generalized_dielectric_fresnel(
    Float cosine, Float eta, Float3 f0) noexcept;
[[nodiscard]] Float3 fresnel_f82_b(Float3 f0, Float3 tint) noexcept;
[[nodiscard]] Float3 fresnel_f82(
    Float cosine, Float3 f0, Float3 b) noexcept;
[[nodiscard]] Float3 ensure_valid_specular_reflection(
    Float3 geometric_normal,
    Float3 incoming,
    Float3 shading_normal) noexcept;
[[nodiscard]] Float3 maybe_ensure_valid_specular_reflection(
    const SurfacePoint &point,
    Float3 incoming,
    Float3 shading_normal) noexcept;
[[nodiscard]] GgxEnergy ggx_energy(const ShaderServices &services,
    const TracedClosure &closure,
    Float incoming_cosine,
    Float3 fss) noexcept;
[[nodiscard]] Float closure_sample_weight(
    const SurfaceClosureRecord &closure) noexcept;
[[nodiscard]] Bool closure_allocated(
    const SurfaceClosureRecord &closure) noexcept;
[[nodiscard]] Bool closure_allocated(
    const SurfaceClosureIdentityExpression &closure) noexcept;
// Exact Cycles bump_shadowing_term contract. `smooth_normal` is the final
// shader-wide sd->N; closure.normal may be an independently linked socket.
// A nonzero factor changes closure energy but never density. A zero factor
// rejects bsdf_eval's competing PDF, while the closure selected through
// bsdf_sample retains its original sampling PDF; EvaluationMode expresses
// that distinction at the aggregate evaluator.
[[nodiscard]] Float bump_shadowing_term(const SurfacePoint &point,
    Float3 smooth_normal,
    const SurfaceClosureRecord &closure,
    Float3 direction,
    Bool is_evaluation) noexcept;
[[nodiscard]] UInt cycles_runtime_flags(const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness = 0.0f) noexcept;
[[nodiscard]] UInt cycles_runtime_flags(
    const SurfaceClosureIdentityExpression &closure,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] UInt cycles_closure_type(
    const SurfaceClosureRecord &closure) noexcept;
[[nodiscard]] UInt cycles_closure_type(
    const SurfaceClosureIdentityExpression &closure) noexcept;
[[nodiscard]] Float oren_nayar_g(Float cosine) noexcept;
[[nodiscard]] Float3 diffuse_intensity(const SurfaceClosureRecord &closure,
    Float3 incoming,
    Float3 outgoing) noexcept;
[[nodiscard]] Float ggx_distribution(
    Float n_dot_h, Float alpha) noexcept;
[[nodiscard]] Float microfacet_alpha(const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Bool microfacet_is_singular(
    const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float smith_g1(Float n_dot_v, Float alpha) noexcept;
[[nodiscard]] Float3 specular_f0(
    const SurfaceClosureRecord &closure) noexcept;
[[nodiscard]] Float3 microfacet_reflection_fresnel(
    const SurfaceClosureRecord &closure,
    Float cosine) noexcept;
[[nodiscard]] Float3 microfacet_intensity(
    const ShaderServices &services,
    const SurfaceClosureRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float microfacet_pdf(const SurfaceClosureRecord &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] MicrofacetReflectionSample sample_microfacet_reflection(
    const SurfacePoint &point,
    Float3 smooth_normal,
    const SurfaceClosureRecord &closure,
    Float3 incoming,
    Float2 random,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float sheen_intensity(const SurfaceClosureRecord &closure,
    Float3 incoming,
    Float3 outgoing) noexcept;
[[nodiscard]] Float3 sample_sheen(const SurfaceClosureRecord &closure,
    Float3 incoming,
    Float2 random) noexcept;
[[nodiscard]] Float3 sample_cosine_hemisphere(
    Float3 normal, Float2 random) noexcept;
[[nodiscard]] Float3 rotate_euler(
    Float3 value, Float3 rotation) noexcept;
[[nodiscard]] Float3 rotate_euler_transposed(
    Float3 value, Float3 rotation) noexcept;
[[nodiscard]] Float3 safe_divide_components(
    Float3 numerator, Float3 denominator) noexcept;

class GraphSurfaceImplementation;

struct ValueEvaluationContext {
    const ShaderServices &services;
    const SurfacePoint &point;
    TracedValues &result;
    const GraphSurfaceImplementation &surface;
};

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
[[nodiscard]] std::unique_ptr<ValueNode> try_make_wave_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_voronoi_value_node(
    const compiler::ValueInstruction &instruction) noexcept;
[[nodiscard]] std::unique_ptr<ValueNode> try_make_procedural_value_node(
    const compiler::ValueInstruction &instruction) noexcept;

using ClosureVisitor = std::function<void(const TracedClosure &)>;
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
    SurfaceCapabilities _capabilities;
    std::vector<std::unique_ptr<ValueNode>> _value_nodes;

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

public:
    [[nodiscard]] std::vector<bool> value_dependency_mask(
        compiler::ValueExpressionId root) const;

    [[nodiscard]] TracedValues trace_values(
        const ShaderServices &services,
        const SurfacePoint &point,
        const std::vector<bool> *active_mask = nullptr) const noexcept;

private:
    void for_each_closure(const TracedValues &values,
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
    ~GraphSurfaceImplementation() noexcept;

    [[nodiscard]] SurfaceCapabilities capabilities() const noexcept;
    [[nodiscard]] SurfaceClosureCollection collect_closures(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics,
        SurfaceClosureCollector &collector) const noexcept;
    [[nodiscard]] UInt runtime_flags(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> glossy_filter_roughness,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics) const noexcept;
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
    [[nodiscard]] Float3 shading_normal(const ShaderServices &services,
        const SurfacePoint &point) const noexcept;
    [[nodiscard]] SurfaceAov aov(const ShaderServices &services,
        const SurfacePoint &point) const noexcept;
};

} // namespace psycles::luisa_backend::detail

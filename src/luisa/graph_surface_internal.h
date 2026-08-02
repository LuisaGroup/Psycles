#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/surface.h>

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
    luisa::vector<Float4> values;
    Float3 shading_normal;
};

// A Principled graph leaf is expanded into the same physical closure
// order as Cycles before any trace, AOV, evaluation, or sampling query
// consumes it. The tag is host-stage metadata used while Luisa records
// the shader AST.
enum class PrincipledLobe : std::uint8_t { none, metallic, dielectric };

struct TracedClosure {
    compiler::ClosureOperation operation{
        compiler::ClosureOperation::diffuse};
    PrincipledLobe principled_lobe{PrincipledLobe::none};
    Float3 weight;
    // Allocation and sampling are distinct in Cycles: Fresnel setup may
    // reduce sample_weight after a closure has already been allocated.
    Float allocation_weight;
    Float sample_weight;
    // Cycles' best-effort closure albedo, including
    // ShaderClosure::weight. This drives closure selection and the
    // Diff/Gloss/Trans color passes.
    Float3 albedo;
    Float3 color;
    Float3 normal;
    Float roughness;
    Float diffuse_roughness;
    Float subsurface_weight;
    Float3 subsurface_radius;
    Float subsurface_scale;
    Float metallic;
    Float ior;
    Float specular_ior_level;
    Float3 specular_tint;
    // Microfacet multiple-scattering scale after any weight darkening
    // has already been applied to `weight`.
    Float3 evaluation_scale;
    bool preserve_ggx_energy{};
    bool beckmann{};
};

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

struct AdjustedIor {
    Float eta;
    Float f0;
};

struct GgxEnergy {
    Float3 darkening;
    Float3 energy_scale;
};

struct PrincipledState {
    Float eta;
    Float3 dielectric_f0;
    Float3 metallic_f0;
    Float3 metallic_b;
    Float3 dielectric_energy_scale;
    Float3 metallic_energy_scale;
    Float3 dielectric_weight;
    Float dielectric_allocation_weight;
    Float dielectric_sample_weight;
    Float3 dielectric_albedo;
    Float3 metallic_weight;
    Float metallic_allocation_weight;
    Float metallic_sample_weight;
    Float3 metallic_albedo;
    Float3 diffuse_weight;
};

struct ClosureSelectionState {
    Bool eligible;
    Float weight;
    Float3 glossy_normal;
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
[[nodiscard]] GgxEnergy ggx_energy(const ShaderServices &services,
    const TracedClosure &closure,
    Float incoming_cosine,
    Float3 fss) noexcept;
[[nodiscard]] PrincipledState principled_state(
    const ShaderServices &services,
    const TracedClosure &closure,
    Float3 incoming,
    Float3 glossy_normal) noexcept;
[[nodiscard]] bool is_scattering_operation(
    compiler::ClosureOperation operation) noexcept;
[[nodiscard]] Float closure_sample_weight(
    const TracedClosure &closure) noexcept;
[[nodiscard]] Bool closure_allocated(
    const TracedClosure &closure) noexcept;
[[nodiscard]] UInt cycles_runtime_flags(const TracedClosure &closure,
    Float glossy_filter_roughness = 0.0f) noexcept;
[[nodiscard]] UInt cycles_closure_type(
    const TracedClosure &closure) noexcept;
[[nodiscard]] ClosureSelectionState closure_selection_state(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedClosure &closure,
    Float3 incoming,
    const SurfaceQuery &query) noexcept;
[[nodiscard]] Float oren_nayar_g(Float cosine) noexcept;
[[nodiscard]] Float3 diffuse_intensity(const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing) noexcept;
[[nodiscard]] Float ggx_distribution(
    Float n_dot_h, Float alpha) noexcept;
[[nodiscard]] Float microfacet_alpha(const TracedClosure &closure,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float smith_g1(Float n_dot_v, Float alpha) noexcept;
[[nodiscard]] Float3 specular_f0(const TracedClosure &closure) noexcept;
[[nodiscard]] Float3 microfacet_intensity(
    const ShaderServices &services,
    const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float microfacet_pdf(const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float3 sample_ggx(const TracedClosure &closure,
    Float3 incoming,
    Float2 random,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float3 glass_microfacet_intensity(
    const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] Float glass_microfacet_pdf(
    const TracedClosure &closure,
    Float3 incoming,
    Float3 outgoing,
    Float3 glossy_normal,
    Bool reflection_allowed,
    Bool transmission_allowed,
    Float glossy_filter_roughness) noexcept;
[[nodiscard]] GlassSample sample_glass(
    const TracedClosure &closure,
    Float3 incoming,
    Float3 geometric_normal,
    Float3 glossy_normal,
    Float2 random_direction,
    Float random_lobe,
    Bool reflection_allowed,
    Bool transmission_allowed,
    Float glossy_filter_roughness) noexcept;
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

    [[nodiscard]] virtual Float4 evaluate(
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
[[nodiscard]] std::unique_ptr<ValueNode> try_make_procedural_value_node(
    const compiler::ValueInstruction &instruction) noexcept;

using ClosureVisitor = std::function<void(const TracedClosure &)>;
using VolumeVisitor =
    std::function<void(const compiler::VolumeInstruction &, Float)>;

class GraphSurfaceImplementation {

private:
    std::shared_ptr<const compiler::SurfaceProgram> _program;
    SurfaceCapabilities _capabilities;
    std::vector<std::unique_ptr<ValueNode>> _value_nodes;

    [[nodiscard]] SurfaceEvaluation evaluate_traced(
        const ShaderServices &services,
        const TracedValues &values,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept;

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
        const ClosureVisitor &visitor) const noexcept;
    void for_each_volume(const TracedValues &values,
        const VolumeVisitor &visitor) const noexcept;

public:
    explicit GraphSurfaceImplementation(
        std::shared_ptr<const compiler::SurfaceProgram>
            program) noexcept;
    ~GraphSurfaceImplementation() noexcept;

    [[nodiscard]] SurfaceCapabilities capabilities() const noexcept;
    [[nodiscard]] SurfaceEvaluation evaluate(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept;
    [[nodiscard]] SurfaceSample sample(const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept;
    [[nodiscard]] SurfaceClosureTrace closure_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<std::uint32_t> requested_index) const noexcept;
    [[nodiscard]] SurfaceSampleTrace sample_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept;
    [[nodiscard]] Float3 emission(const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing) const noexcept;
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

#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/surface.h> through the Psycles::luisa target."
#endif

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <psycles/contract/cycles_abi.h>
#include <psycles/contract/surface.h>

#include <luisa/core/stl/vector.h>
#include <luisa/dsl/polymorphic.h>
#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend {

namespace cycles_volume_phase {
struct Closure;
}// namespace cycles_volume_phase

using luisa::compute::Bool;
using luisa::compute::Expr;
using luisa::compute::Float;
using luisa::compute::Float2;
using luisa::compute::Float3;
using luisa::compute::Float4;
using luisa::compute::Polymorphic;
using luisa::compute::UInt;
using luisa::compute::ULong;
using luisa::compute::make_float2;
using luisa::compute::make_float3;
using luisa::compute::make_float4;

using contract::SurfaceEvent;
using contract::TransportMode;
using contract::event_diffuse;
using contract::event_glossy;
using contract::event_none;
using contract::event_reflection;
using contract::event_singular;
using contract::event_subsurface;
using contract::event_transmission;
using contract::event_transparent;

struct SurfaceCapabilities {
    std::uint64_t features{};
    bool may_emit{false};
    // Cycles' SD_HAS_CONSTANT_EMISSION scheduling property. This remains
    // true for a non-emitting graph (whose constant value is zero) and false
    // only when full shading-point evaluation must be deferred.
    bool emission_is_constant{true};
    bool may_be_transparent{false};
    bool may_have_subsurface{false};
    bool may_have_volume{false};
};

struct SurfacePoint {
    Float3 position;
    Float3 object_position;
    Float3 object_location;
    Float3 generated;
    Float3 geometric_normal;
    Float3 shading_normal;
    // Cycles tangent-space Normal Map is constructed in object space from
    // Blender's evaluated MikkTSpace attributes and only then transformed as
    // a normal. The interpolated shading normal remains raw here: the Normal
    // Map node owns Cycles' normalization-before-frame contract. Keeping
    // these values explicit avoids an invalid world-space shortcut under
    // non-uniform instance transforms.
    Float3 object_shading_normal;
    Float3 object_tangent;
    Float tangent_sign;
    // Cycles' BOTH bump program temporarily evaluates from the geometry
    // state saved immediately before true displacement. Normal Map's
    // ORIGINAL base consumes the same normal/tangent attributes. These are
    // first-class geometry values, never reconstructed from a baked shader.
    Float3 undisplaced_position;
    Float3 undisplaced_object_position;
    Float3 undisplaced_shading_normal;
    Float3 undisplaced_object_shading_normal;
    Float3 undisplaced_object_tangent;
    Float undisplaced_tangent_sign;
    Float3 normal_to_world_x;
    Float3 normal_to_world_y;
    Float3 normal_to_world_z;
    Float3 dpdu;
    Float3 dpdv;
    Float3 dPdx;
    Float3 dPdy;
    Float3 object_dPdx;
    Float3 object_dPdy;
    Float3 undisplaced_dPdx;
    Float3 undisplaced_dPdy;
    Float3 undisplaced_object_dPdx;
    Float3 undisplaced_object_dPdy;
    Float3 generated_dx;
    Float3 generated_dy;
    Float3 incoming;
    Float2 uv;
    Float2 uv_dx;
    Float2 uv_dy;
    UInt geometry_index;
    Float2 barycentric;
    Float2 barycentric_dx;
    Float2 barycentric_dy;
    UInt instance_id;
    UInt primitive_id;
    // Runtime parameter storage base for the material bound to this hit.
    // Materials with an identical Cycles graph structure share one
    // GraphSurface AST and differ only by this block.
    UInt parameter_block;
    Float object_random;
    UInt particle_index;
    Float random_per_island;
    Bool triangle_smooth;
    // Native Cycles curve state. These are zero for non-curve primitives.
    // Keeping the values on the typed shading point preserves the original
    // Hair Info graph and avoids translating curve attributes into material
    // parameters or baked textures.
    Bool is_curve;
    Float curve_intercept;
    Float curve_length;
    Float curve_thickness;
    Float3 curve_tangent_normal;
    Float curve_random;
    UInt ray_visibility;
    UInt ray_events;
    UInt ray_depth;
    UInt diffuse_depth;
    UInt glossy_depth;
    UInt transparent_depth;
    UInt transmission_depth;
    Float ray_length;
    Float time;
    // Cycles SD_USE_BUMP_MAP_CORRECTION is material-owned runtime state.
    // Keep it on the shading point because one GraphSurface AST may be
    // shared by materials with different policies.
    Bool use_bump_map_correction;
    Bool back_facing;
};

struct SurfaceQuery {
    UInt lobe_mask;
    UInt transport_mode;
    // Cycles Filter Glossy widens microfacet alpha after closure setup. Zero
    // leaves the material closure unchanged.
    Float glossy_filter_roughness;
    // Cycles allocates reflective closures only when reflective caustics are
    // enabled for the current path. This is an allocation predicate, not
    // merely a lobe-selection mask: standalone Glossy must disappear from
    // the closure mixture, while Coat/specular layering of lower Principled
    // closures also depends on it.
    Bool reflective_caustics{true};
    // Transmission closures have an independent Cycles caustics gate. Glass
    // remains one allocated closure when either reflective or refractive
    // transport is enabled, with the disabled Fresnel branch carrying zero
    // tint.
    Bool refractive_caustics{true};
    // The Cycles subsurface intersection kernel shades the selected exit
    // point with one synthetic unit Lambert closure. This is integrator
    // state, not a material rewrite: the original graph remains untouched
    // and is still evaluated at ordinary surface entries.
    Bool subsurface_exit{false};
    // Cycles re-evaluates a bump-capable material at a BSSRDF exit, averages
    // only the retained BSSRDF closure normals by fabs(average(weight)), and
    // uses that normal for the synthetic unit Lambert. It remains distinct
    // from SurfacePoint::shading_normal, which still owns ShaderData and ray
    // offset semantics at the exit intersection.
    Float3 subsurface_normal{make_float3(0.0f, 0.0f, 1.0f)};
};

// Cycles sampled-light visibility is deliberately separate from the path
// lobe mask. Shader exclude flags remove closure contributions, but every
// otherwise eligible closure remains in the one-sample-model PDF.
struct SurfaceLightQuery {
    SurfaceQuery surface;
    UInt shader_flags;
};

struct ShaderAttribute {
    Float4 value;
    Bool found;

    [[nodiscard]] static ShaderAttribute missing() noexcept {
        return {
            .value = make_float4(0.0f),
            .found = false};
    }
};

// Cycles evaluates a volume graph in two distinct contexts. Shadow and
// extinction-only queries suppress emission, while full volume shading keeps
// it. Object-space grids additionally scale coefficients by the instance's
// volume-density correction.
struct VolumeQuery {
    Float object_density;
    Bool evaluate_emission;
};

struct VolumeCoefficients {
    Float3 sigma_t;
    Float3 sigma_s;
    Float3 emission;
    Bool has_extinction;
    Bool has_scatter;
    Bool has_emission;

    [[nodiscard]] static VolumeCoefficients zero() noexcept {
        return {
            .sigma_t = make_float3(0.0f),
            .sigma_s = make_float3(0.0f),
            .emission = make_float3(0.0f),
            .has_extinction = false,
            .has_scatter = false,
            .has_emission = false};
    }
};

// Host-stage sink for raw phase closures. A GraphSurface invokes this while
// Luisa records the shader AST; implementations may retain closures in device
// local storage, merge them across stacked media, or expose them to a focused
// diagnostic kernel. The closure parameters are never averaged or pre-baked.
class VolumePhaseCollector {

  public:
    virtual ~VolumePhaseCollector() noexcept = default;
    virtual void add(
        const cycles_volume_phase::Closure &phase,
        Float3 weight) noexcept = 0;
};

// Device-side identity of one post-shader Cycles surface closure. Principled
// is intentionally retained as a family tag: its independently allocated
// physical lobes remain distinguishable without baking them into a combined
// BSDF.
enum class SurfaceClosureKind : std::uint32_t {
    none,
    diffuse,
    translucent,
    principled,
    glossy,
    glass,
    transparent,
    refraction,
    bssrdf
};

enum class SurfaceClosureLobe : std::uint32_t {
    none,
    sheen,
    coat,
    metallic,
    transmission,
    dielectric
};

enum class SurfaceBssrdfMethod : std::uint32_t {
    burley,
    random_walk,
    random_walk_legacy,
    random_walk_skin
};

// Cycles kernel/types.h::MAX_CLOSURE. Scene analysis may specialize to a
// smaller graph-derived value, but no surface closure allocation may exceed
// this ABI limit.
inline constexpr std::uint32_t
    maximum_surface_closure_capacity = 64u;

// Canonical device-expression record emitted after Cycles-compatible closure
// allocation and setup. Fields which do not belong to a closure family are
// explicitly zeroed by the producer. This makes the record safe to retain in
// backend-independent Local storage and later consume through a runtime tag.
// No radiance, directional response, or material input is evaluated on the
// host: the record consists entirely of Luisa expressions recorded into the
// shader AST.
struct SurfaceClosureRecord {
    UInt kind;
    UInt lobe;
    Float3 weight;
    Float allocation_weight;
    Float sample_weight;
    Bool setup_valid;
    Float3 albedo;
    Float3 reflection_albedo;
    Float3 transmission_albedo;
    Float3 color;
    Float3 normal;
    Float roughness;
    Float diffuse_roughness;
    Float metallic;
    Float ior;
    Float specular_ior_level;
    Float3 specular_tint;
    Float sheen_transform_a;
    Float sheen_transform_b;
    Float3 evaluation_scale;
    Float3 fresnel_f0;
    Float3 fresnel_f90;
    Float3 reflection_tint;
    Float3 transmission_tint;
    Bool preserve_ggx_energy;
    Bool beckmann;
    UInt bssrdf_method;
    Float3 bssrdf_radius;
    Float3 bssrdf_albedo;
    Float bssrdf_ior;
    Float bssrdf_roughness;
    Float bssrdf_anisotropy;

    [[nodiscard]] static SurfaceClosureRecord zero() noexcept {
        return {
            .kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::none),
            .lobe = static_cast<std::uint32_t>(
                SurfaceClosureLobe::none),
            .weight = make_float3(0.0f),
            .allocation_weight = 0.0f,
            .sample_weight = 0.0f,
            .setup_valid = false,
            .albedo = make_float3(0.0f),
            .reflection_albedo = make_float3(0.0f),
            .transmission_albedo = make_float3(0.0f),
            .color = make_float3(0.0f),
            .normal = make_float3(0.0f, 0.0f, 1.0f),
            .roughness = 0.0f,
            .diffuse_roughness = 0.0f,
            .metallic = 0.0f,
            .ior = 1.0f,
            .specular_ior_level = 0.0f,
            .specular_tint = make_float3(0.0f),
            .sheen_transform_a = 0.0f,
            .sheen_transform_b = 0.0f,
            .evaluation_scale = make_float3(1.0f),
            .fresnel_f0 = make_float3(0.0f),
            .fresnel_f90 = make_float3(0.0f),
            .reflection_tint = make_float3(0.0f),
            .transmission_tint = make_float3(0.0f),
            .preserve_ggx_energy = false,
            .beckmann = false,
            .bssrdf_method = static_cast<std::uint32_t>(
                SurfaceBssrdfMethod::random_walk),
            .bssrdf_radius = make_float3(0.0f),
            .bssrdf_albedo = make_float3(0.0f),
            .bssrdf_ior = 1.4f,
            .bssrdf_roughness = 1.0f,
            .bssrdf_anisotropy = 0.0f};
    }
};

// Non-owning host-stage view of a canonical closure's Luisa expressions.
// Copying this type only copies AST expression handles: it neither declares
// device variables nor evaluates, serializes, or bakes material data. This is
// the representation retained by multistage visitors while a material branch
// is being recorded.
struct SurfaceClosureExpression {
    Expr<std::uint32_t> kind;
    Expr<std::uint32_t> lobe;
    Expr<luisa::float3> weight;
    Expr<float> allocation_weight;
    Expr<float> sample_weight;
    Expr<bool> setup_valid;
    Expr<luisa::float3> albedo;
    Expr<luisa::float3> reflection_albedo;
    Expr<luisa::float3> transmission_albedo;
    Expr<luisa::float3> color;
    Expr<luisa::float3> normal;
    Expr<float> roughness;
    Expr<float> diffuse_roughness;
    Expr<float> metallic;
    Expr<float> ior;
    Expr<float> specular_ior_level;
    Expr<luisa::float3> specular_tint;
    Expr<float> sheen_transform_a;
    Expr<float> sheen_transform_b;
    Expr<luisa::float3> evaluation_scale;
    Expr<luisa::float3> fresnel_f0;
    Expr<luisa::float3> fresnel_f90;
    Expr<luisa::float3> reflection_tint;
    Expr<luisa::float3> transmission_tint;
    Expr<bool> preserve_ggx_energy;
    Expr<bool> beckmann;
    Expr<std::uint32_t> bssrdf_method;
    Expr<luisa::float3> bssrdf_radius;
    Expr<luisa::float3> bssrdf_albedo;
    Expr<float> bssrdf_ior;
    Expr<float> bssrdf_roughness;
    Expr<float> bssrdf_anisotropy;

    explicit SurfaceClosureExpression(
        const SurfaceClosureRecord &closure) noexcept;

    // Re-wrap the retained handles in the mutable DSL facade expected by
    // existing scattering components. This creates no AST declarations or
    // assignments; callers must treat the returned facade as read-only.
    [[nodiscard]] SurfaceClosureRecord reference() const noexcept;
};

// Host-stage sink matching VolumePhaseCollector. Surface implementations call
// begin(), add(), and finish() while Luisa records each material branch. A
// multistage consumer may retain SurfaceClosureExpression handles until
// finish(), which remains inside that branch. The default lifecycle hooks keep
// existing streaming collectors source compatible.
class SurfaceClosureCollector {

  public:
    virtual ~SurfaceClosureCollector() noexcept = default;
    virtual void begin(
        Expr<luisa::float3>) noexcept {}
    virtual void add(
        const SurfaceClosureRecord &closure) noexcept = 0;
    virtual void finish() noexcept {}
};

struct SurfaceClosureCollection {
    Float3 shading_normal;
};

struct SurfaceEvaluation {
    Float3 f;
    Float pdf;
    Float3 diffuse_f;
    Float3 glossy_f;
    Float diffuse_pdf;
    // Cycles' PDF-weighted average of bsdf_get_specular_roughness_squared
    // over every closure that participates in the directional mixture. This
    // is a transport quantity: surface and NEE rays use it to widen their
    // compact angular differential after a non-transparent scatter.
    Float average_roughness_squared;
    UInt events;

    [[nodiscard]] static SurfaceEvaluation zero() noexcept {
        return {
            .f = make_float3(0.0f),
            .pdf = 0.0f,
            .diffuse_f = make_float3(0.0f),
            .glossy_f = make_float3(0.0f),
            .diffuse_pdf = 0.0f,
            .average_roughness_squared = 0.0f,
            .events = static_cast<std::uint32_t>(event_none)};
    }
};

struct SurfaceSample {
    SurfaceEvaluation evaluation;
    Float3 wi;
    Float eta;
    Float2 roughness;
    UInt runtime_flags;
    UInt bssrdf_method;
    Float3 bssrdf_radius;
    Float3 bssrdf_albedo;
    Float3 bssrdf_normal;
    Float bssrdf_ior;
    Float bssrdf_roughness;
    Float bssrdf_anisotropy;
    Bool valid;

    [[nodiscard]] static SurfaceSample zero() noexcept {
        return {
            .evaluation = SurfaceEvaluation::zero(),
            .wi = make_float3(0.0f, 0.0f, 1.0f),
            .eta = 1.0f,
            .roughness = make_float2(0.0f),
            .runtime_flags = 0u,
            .bssrdf_method = static_cast<std::uint32_t>(
                SurfaceBssrdfMethod::random_walk),
            .bssrdf_radius = make_float3(0.0f),
            .bssrdf_albedo = make_float3(0.0f),
            .bssrdf_normal = make_float3(0.0f, 0.0f, 1.0f),
            .bssrdf_ior = 1.4f,
            .bssrdf_roughness = 1.0f,
            .bssrdf_anisotropy = 0.0f,
            .valid = false};
    }
};

// Diagnostic view of one entry in the post-shader closure array. `count` is
// repeated so a single runtime-indexed callable can expose the fixed-capacity
// trace without returning a backend-specific aggregate array.
struct SurfaceClosureTrace {
    UInt count;
    UInt runtime_flags;
    UInt index;
    UInt type;
    Float sample_weight;
    Float3 weight;
    Float3 normal;
    Bool valid;

    [[nodiscard]] static SurfaceClosureTrace zero(
        Expr<std::uint32_t> requested_index = 0u) noexcept {
        return {
            .count = 0u,
            .runtime_flags = 0u,
            .index = requested_index,
            .type = 0u,
            .sample_weight = 0.0f,
            .weight = make_float3(0.0f),
            .normal = make_float3(0.0f, 0.0f, 1.0f),
            .valid = false};
    }
};

// Trace-only extension of SurfaceSample. Normal render kernels call `sample`
// and retain the compact production ABI; differential kernels call
// `sample_trace` to observe the exact closure selection without reimplementing
// it in the integrator.
struct SurfaceSampleTrace {
    SurfaceSample sample;
    UInt closure_index;
    UInt closure_type;
    Float closure_sample_weight;
    Float selection_rescaled;
    Float3 closure_weight;
    Float3 closure_normal;
    Bool closure_valid;

    [[nodiscard]] static SurfaceSampleTrace zero() noexcept {
        return {
            .sample = SurfaceSample::zero(),
            .closure_index = 0u,
            .closure_type = 0u,
            .closure_sample_weight = 0.0f,
            .selection_rescaled = 0.0f,
            .closure_weight = make_float3(0.0f),
            .closure_normal = make_float3(0.0f, 0.0f, 1.0f),
            .closure_valid = false};
    }
};

struct SurfaceAov {
    Float3 albedo;
    Float3 glossy_albedo;
    Float3 transmission_albedo;
    Float2 roughness;
    Float3 normal;
    Float3 transparency;
};

class SurfaceParameterServices {

public:
    virtual ~SurfaceParameterServices() noexcept = default;

    [[nodiscard]] virtual Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept = 0;

    [[nodiscard]] virtual Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept = 0;
};

class ShaderServices : public SurfaceParameterServices {

public:
    ~ShaderServices() noexcept override = default;

    [[nodiscard]] virtual Float4 texture_2d(
        Expr<std::uint32_t> handle,
        Expr<luisa::float2> uv,
        Expr<luisa::float2> d_uv_dx,
        Expr<luisa::float2> d_uv_dy,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept = 0;

    [[nodiscard]] virtual ShaderAttribute attribute(
        Expr<std::uint64_t> attribute_id,
        const SurfacePoint &point) const noexcept = 0;

    // Versioned Cycles compatibility data. The index addresses the
    // contiguous Blender BSDF table buffer; interpolation and table shape
    // remain explicit in the Luisa shader so all backends execute the same
    // semantics.
    [[nodiscard]] virtual Float cycles_bsdf_data(
        Expr<std::uint32_t> index) const noexcept = 0;

    [[nodiscard]] virtual Float3 xyz_to_rgb(
        Expr<luisa::float3> xyz) const noexcept = 0;

    [[nodiscard]] virtual Float3 rec709_to_rgb(
        Expr<luisa::float3> rec709) const noexcept = 0;

    // Cycles Nishita is a precomputed 512x128 spectral LUT, not an analytic
    // color approximation. The host/JIT-stage sky index identifies the LUT
    // associated with this material parameter block; direction-dependent
    // sampling and the solar disc remain Luisa expressions.
    [[nodiscard]] virtual Float3 nishita_sky(
        Expr<std::uint32_t> block,
        std::uint32_t sky_index,
        Expr<luisa::float3> direction,
        Expr<float> sun_elevation,
        Expr<float> sun_rotation,
        Expr<float> angular_diameter,
        Expr<float> sun_intensity) const noexcept = 0;
};

class Surface {

public:
    virtual ~Surface() noexcept = default;

    [[nodiscard]] virtual SurfaceCapabilities capabilities() const noexcept = 0;

    // Record the material's Cycles-ordered physical closure array once. The
    // default keeps custom diagnostic Surface implementations source
    // compatible; production GraphSurface overrides this boundary.
    [[nodiscard]] virtual SurfaceClosureCollection collect_closures(
        const ShaderServices &,
        const SurfacePoint &point,
        Expr<bool>,
        Expr<bool>,
        SurfaceClosureCollector &collector) const noexcept {
        collector.begin(point.shading_normal);
        collector.finish();
        return {.shading_normal = point.shading_normal};
    }

    // Runtime ShaderData flags produced by closure setup. This is separate
    // from sampling so integrators can apply Cycles' pre-NEE capability gate
    // without selecting a BSDF closure early.
    [[nodiscard]] virtual UInt runtime_flags(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<float>,
        Expr<bool> = true,
        Expr<bool> = true) const noexcept {
        return 0u;
    }

    [[nodiscard]] virtual SurfaceEvaluation evaluate(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept = 0;

    [[nodiscard]] virtual SurfaceEvaluation evaluate_light(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept = 0;

    [[nodiscard]] virtual SurfaceSample sample(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept = 0;

    [[nodiscard]] virtual SurfaceClosureTrace closure_trace(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<std::uint32_t> requested_index,
        Expr<bool> = true,
        Expr<bool> = true) const noexcept {
        return SurfaceClosureTrace::zero(requested_index);
    }

    [[nodiscard]] virtual SurfaceSampleTrace sample_trace(
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept {
        auto result = SurfaceSampleTrace::zero();
        result.sample = sample(
            services,
            point,
            u_lobe,
            u_direction,
            query);
        return result;
    }

    // Device-side evaluation of Cycles' constant-emission proof. The narrow
    // services type makes shading-point, texture, attribute and BSDF-table
    // access unavailable while Luisa records this AST.
    [[nodiscard]] virtual Float3 constant_emission(
        const SurfaceParameterServices &,
        Expr<std::uint32_t>) const noexcept {
        return make_float3(0.0f);
    }

    [[nodiscard]] virtual Float3 emission(
        const ShaderServices &,
        const SurfacePoint &,
        Expr<luisa::float3>,
        Expr<bool>) const noexcept {
        return make_float3(0.0f);
    }

    [[nodiscard]] virtual Float3 transparent_extinction(
        const ShaderServices &,
        const SurfacePoint &) const noexcept {
        return make_float3(0.0f);
    }

    // Combined volume evaluation is the host-stage semantic boundary.
    // Passing a collector emits raw phase closures from the same traced graph
    // that produced the coefficients, so the path kernel never evaluates a
    // volume shader twice.
    [[nodiscard]] virtual VolumeCoefficients evaluate_volume(
        const ShaderServices &,
        const SurfacePoint &,
        const VolumeQuery &,
        VolumePhaseCollector *) const noexcept {
        return VolumeCoefficients::zero();
    }

    // Raw world-space displacement produced by the material displacement
    // root. This is evaluated during mesh compilation, before BLAS build;
    // it is never a bump approximation and does not mutate shading state.
    [[nodiscard]] virtual Float3 displacement(
        const ShaderServices &,
        const SurfacePoint &) const noexcept {
        return make_float3(0.0f);
    }

    // Final Cycles sd->N after shader bump evaluation. This is distinct from
    // closure-specific Normal inputs and is consumed by shadow-terminator
    // geometry offset.
    [[nodiscard]] virtual Float3 shading_normal(
        const ShaderServices &,
        const SurfacePoint &point) const noexcept {
        return point.shading_normal;
    }

    [[nodiscard]] virtual SurfaceAov aov(
        const ShaderServices &,
        const SurfacePoint &point) const noexcept {
        return {
            .albedo = make_float3(0.0f),
            .glossy_albedo = make_float3(0.0f),
            .transmission_albedo = make_float3(0.0f),
            .roughness = make_float2(0.0f),
            .normal = point.shading_normal,
            .transparency = make_float3(0.0f)};
    }
};

class SurfaceDispatch {

private:
    // This runtime-tag dispatch is a semantic boundary, not a temporary
    // fallback. Scene-specific fusion belongs to a later compiler/IR pass.
    Polymorphic<Surface> _surfaces;

public:
    [[nodiscard]] bool empty() const noexcept { return _surfaces.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return _surfaces.size(); }
    [[nodiscard]] const Surface *implementation(std::size_t index) const noexcept {
        return _surfaces.impl(index);
    }
    [[nodiscard]] SurfaceCapabilities capabilities(std::size_t index) const noexcept {
        return implementation(index)->capabilities();
    }

    [[nodiscard]] SurfaceClosureCollection collect_closures(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics,
        SurfaceClosureCollector &collector) const noexcept {
        auto result = SurfaceClosureCollection{
            .shading_normal = point.shading_normal};
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->collect_closures(
                services,
                point,
                reflective_caustics,
                refractive_caustics,
                collector);
        });
        return result;
    }

    [[nodiscard]] SurfaceClosureCollection collect_bssrdf_bump_closures(
        Expr<std::uint32_t> tag,
        const luisa::vector<luisa::uint> &bssrdf_bump_tags,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<bool> reflective_caustics,
        Expr<bool> refractive_caustics,
        SurfaceClosureCollector &collector) const noexcept {
        auto result = SurfaceClosureCollection{
            .shading_normal = point.shading_normal};
        // The group is the scene-level Cycles has_bssrdf_bump result mapped
        // onto deduplicated program tags. Cycles skips shader evaluation at a
        // BSSRDF exit unless this exact flag is present. Restrict this JIT
        // switch to that semantic set; every program remains registered for
        // all other surface operations.
        auto empty_collection = [&]() noexcept {
            collector.begin(point.shading_normal);
            collector.finish();
        };
        if (bssrdf_bump_tags.empty()) {
            empty_collection();
        } else {
            _surfaces.dispatch_group_with_default(
                tag,
                bssrdf_bump_tags,
                [&](const Surface *surface) noexcept {
                    result = surface->collect_closures(
                        services,
                        point,
                        reflective_caustics,
                        refractive_caustics,
                        collector);
                },
                empty_collection);
        }
        return result;
    }

    template<typename Implementation, typename... Args>
        requires std::derived_from<Implementation, Surface>
    [[nodiscard]] std::uint32_t create(Args &&...args) noexcept {
        return _surfaces.template create<Implementation>(
            std::forward<Args>(args)...);
    }

    template<typename Tag, typename Function>
    void dispatch(Tag &&tag, Function &&function) const noexcept {
        _surfaces.dispatch(
            std::forward<Tag>(tag),
            std::forward<Function>(function));
    }

    [[nodiscard]] SurfaceEvaluation evaluate(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceQuery &query) const noexcept {
        auto result = SurfaceEvaluation::zero();
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->evaluate(
                services, point, outgoing, query);
        });
        return result;
    }

    [[nodiscard]] UInt runtime_flags(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> glossy_filter_roughness,
        Expr<bool> reflective_caustics = true,
        Expr<bool> refractive_caustics = true) const noexcept {
        UInt result = 0u;
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->runtime_flags(
                services,
                point,
                glossy_filter_roughness,
                reflective_caustics,
                refractive_caustics);
        });
        return result;
    }

    [[nodiscard]] SurfaceEvaluation evaluate_light(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> outgoing,
        const SurfaceLightQuery &query) const noexcept {
        auto result = SurfaceEvaluation::zero();
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->evaluate_light(
                services, point, outgoing, query);
        });
        return result;
    }

    [[nodiscard]] SurfaceSample sample(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept {
        auto result = SurfaceSample::zero();
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->sample(
                services, point, u_lobe, u_direction, query);
        });
        return result;
    }

    [[nodiscard]] SurfaceClosureTrace closure_trace(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<std::uint32_t> requested_index,
        Expr<bool> reflective_caustics = true,
        Expr<bool> refractive_caustics = true) const noexcept {
        auto result =
            SurfaceClosureTrace::zero(requested_index);
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->closure_trace(
                services,
                point,
                requested_index,
                reflective_caustics,
                refractive_caustics);
        });
        return result;
    }

    [[nodiscard]] SurfaceSampleTrace sample_trace(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<float> u_lobe,
        Expr<luisa::float2> u_direction,
        const SurfaceQuery &query) const noexcept {
        auto result = SurfaceSampleTrace::zero();
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->sample_trace(
                services,
                point,
                u_lobe,
                u_direction,
                query);
        });
        return result;
    }

    [[nodiscard]] Float3 emission(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        Expr<luisa::float3> wo,
        Expr<bool> reflective_caustics) const noexcept {
        Float3 result = make_float3(0.0f);
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->emission(
                services,
                point,
                wo,
                reflective_caustics);
        });
        return result;
    }

    [[nodiscard]] Float3 constant_emission(
        Expr<std::uint32_t> tag,
        const SurfaceParameterServices &services,
        Expr<std::uint32_t> parameter_block) const noexcept {
        Float3 result = make_float3(0.0f);
        luisa::vector<luisa::uint> constant_tags;
        constant_tags.reserve(_surfaces.size());
        for (auto index = std::size_t{0u};
             index < _surfaces.size();
             ++index) {
            if (_surfaces.impl(index)
                    ->capabilities()
                    .emission_is_constant) {
                constant_tags.emplace_back(
                    static_cast<luisa::uint>(index));
            }
        }
        if (!constant_tags.empty()) {
            _surfaces.dispatch_group_with_default(
                tag,
                constant_tags,
                [&](const Surface *surface) noexcept {
                    result = surface->constant_emission(
                        services,
                        parameter_block);
                },
                []() noexcept {});
        }
        return result;
    }

    [[nodiscard]] Float3 transparent_extinction(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept {
        Float3 result = make_float3(0.0f);
        // Transparency is a structural property of the original closure
        // program. Restrict the JIT dispatch to programs which can actually
        // produce a Transparent closure: opaque programs retain their raw
        // closures for every other surface operation, but do not inflate the
        // ray-query callback with unreachable shader code.
        luisa::vector<luisa::uint> transparent_tags;
        transparent_tags.reserve(_surfaces.size());
        for (auto index = std::size_t{0u};
             index < _surfaces.size();
             ++index) {
            if (_surfaces.impl(index)
                    ->capabilities()
                    .may_be_transparent) {
                transparent_tags.emplace_back(
                    static_cast<luisa::uint>(index));
            }
        }
        if (!transparent_tags.empty()) {
            _surfaces.dispatch_group_with_default(
                tag,
                transparent_tags,
                [&](const Surface *surface) noexcept {
                    result =
                        surface->transparent_extinction(
                            services, point);
                },
                []() noexcept {});
        }
        return result;
    }

    [[nodiscard]] VolumeCoefficients evaluate_volume(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const VolumeQuery &query,
        VolumePhaseCollector *collector) const noexcept {
        auto result = VolumeCoefficients::zero();
        luisa::vector<luisa::uint> volume_tags;
        volume_tags.reserve(_surfaces.size());
        for (auto index = std::size_t{0u};
             index < _surfaces.size();
             ++index) {
            if (_surfaces.impl(index)
                    ->capabilities()
                    .may_have_volume) {
                volume_tags.emplace_back(
                    static_cast<luisa::uint>(index));
            }
        }
        if (!volume_tags.empty()) {
            _surfaces.dispatch_group_with_default(
                tag,
                volume_tags,
                [&](const Surface *surface) noexcept {
                    result = surface->evaluate_volume(
                        services,
                        point,
                        query,
                        collector);
                },
                []() noexcept {});
        }
        return result;
    }

    [[nodiscard]] VolumeCoefficients volume_coefficients(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const VolumeQuery &query) const noexcept {
        return evaluate_volume(
            tag,
            services,
            point,
            query,
            nullptr);
    }

    void volume_phases(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const VolumeQuery &query,
        VolumePhaseCollector &collector) const noexcept {
        static_cast<void>(
            evaluate_volume(
                tag,
                services,
                point,
                query,
                &collector));
    }

    [[nodiscard]] Float3 shading_normal(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept {
        Float3 result = point.shading_normal;
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->shading_normal(services, point);
        });
        return result;
    }

    [[nodiscard]] Float3 displacement(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept {
        Float3 result = make_float3(0.0f);
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->displacement(services, point);
        });
        return result;
    }

    [[nodiscard]] SurfaceAov aov(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point) const noexcept {
        auto result = SurfaceAov{
            .albedo = make_float3(0.0f),
            .glossy_albedo = make_float3(0.0f),
            .transmission_albedo = make_float3(0.0f),
            .roughness = make_float2(0.0f),
            .normal = point.shading_normal,
            .transparency = make_float3(0.0f)};
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->aov(services, point);
        });
        return result;
    }
};

}// namespace psycles::luisa_backend

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfacePoint)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceQuery)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceLightQuery)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::ShaderAttribute)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::VolumeQuery)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::VolumeCoefficients)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceEvaluation)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceSample)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceClosureTrace)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceSampleTrace)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfaceAov)

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

// Exact shading-point projection consumed after a material graph has
// populated its physical closures. Closure selection, directional sampling,
// and evaluation cannot observe texture coordinates, attributes, geometry
// derivatives, or material parameter storage. Keeping that boundary in the
// type system prevents nested callable ABIs from accidentally carrying the
// complete SurfacePoint and makes future dependency growth explicit.
struct SurfaceClosurePoint {
    Float3 geometric_normal;
    Float3 shading_normal;
    Float3 incoming;
    UInt ray_visibility;
    // Cycles changes two closure laws for curve primitives: bsdf_sample uses
    // the selected ShaderClosure::N as Ng, and all bump-normal correction is
    // disabled. Keep the primitive predicate in this exact post-population
    // projection; reconstructing it from either normal is not possible.
    Bool is_curve;
    Bool use_bump_map_correction;
    Bool back_facing;

    explicit SurfaceClosurePoint(
        const SurfacePoint &point) noexcept
        : geometric_normal{point.geometric_normal},
          shading_normal{point.shading_normal},
          incoming{point.incoming},
          ray_visibility{point.ray_visibility},
          is_curve{point.is_curve},
          use_bump_map_correction{
              point.use_bump_map_correction},
          back_facing{point.back_facing} {}

    SurfaceClosurePoint(
        Float3 geometric_normal_value,
        Float3 shading_normal_value,
        Float3 incoming_value,
        UInt ray_visibility_value,
        Bool is_curve_value,
        Bool use_bump_map_correction_value,
        Bool back_facing_value) noexcept
        : geometric_normal{std::move(geometric_normal_value)},
          shading_normal{std::move(shading_normal_value)},
          incoming{std::move(incoming_value)},
          ray_visibility{std::move(ray_visibility_value)},
          is_curve{std::move(is_curve_value)},
          use_bump_map_correction{
              std::move(use_bump_map_correction_value)},
          back_facing{std::move(back_facing_value)} {}
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
    // Standalone Metallic retains its authored Fresnel algebra as a physical
    // tag. Both variants share the general payload layout; the tag is the
    // eliminator which assigns (color, specular_tint) to either (F0, B) or
    // (n, k) without a runtime mode field.
    metallic_f82,
    metallic_conductor,
    // Standalone Sheen keeps the authored distribution in the physical tag.
    // This is required because Microfiber and Ashikhmin differ not only in
    // payload, but also in sampling measure and Cycles pass classification.
    sheen_microfiber,
    sheen_ashikhmin,
    hair_reflection,
    hair_transmission,
    glass,
    transparent,
    refraction,
    bssrdf,
    // Cycles gives these physical closures distinct identities because
    // their directional transforms differ from ordinary Translucent and
    // Refraction even though they occupy the same transport lobes.
    rough_translucent,
    thin_glass_transmission
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

// Exact post-setup dependency cut for physical closure operations. The first
// two fields are the Cycles ShaderClosure ABI produced by setup; authoring
// kind/lobe, allocation bookkeeping, and compatibility setup flags cannot
// cross this boundary. `closure_type == NONE` is the sole failed-setup state.
// Surface preparation owns the three AOV albedos, while adjusted Principled
// IOR is resolved during setup. Keeping this as a distinct type makes those
// non-observability claims C++ invariants.
struct SurfaceClosurePhysicalRecord {
    UInt closure_type;
    UInt microfacet_fresnel;
    Float3 weight;
    Float sample_weight;
    Float3 color;
    Float3 normal;
    Float roughness;
    // Canonical reflection-microfacet state produced by closure setup.
    // alpha_x/alpha_y are the post-setup distribution roughnesses; tangent
    // is observable only when they differ. Keeping this state physical avoids
    // reinterpreting node-specific anisotropy conventions during sampling.
    Float3 microfacet_tangent;
    Float microfacet_alpha_x;
    Float microfacet_alpha_y;
    Float ior;
    Float thin_film_thickness;
    Float thin_film_ior;
    Float3 specular_tint;
    Float sheen_transform_a;
    Float sheen_transform_b;
    Float3 evaluation_scale;
    Float3 fresnel_f0;
    Float3 fresnel_f90;
    Float3 reflection_tint;
    Float3 transmission_tint;
    Float3 bssrdf_radius;
    Float3 bssrdf_albedo;
    Float bssrdf_ior;
    Float bssrdf_roughness;
    Float bssrdf_anisotropy;
};

// Canonical device-expression record emitted after Cycles-compatible closure
// allocation and setup. Fields which do not belong to a closure family are
// explicitly zeroed by the producer. This makes the record safe to retain in
// backend-independent Local storage and later consume through a runtime tag.
// No radiance, directional response, or material input is evaluated on the
// host: the record consists entirely of Luisa expressions recorded into the
// shader AST.
struct SurfaceClosureRecord {
    // Final Cycles identities written by closure setup. Authoring kind/lobe
    // live only in TracedClosure and host/JIT metadata; they cannot cross this
    // post-setup record boundary.
    UInt closure_type;
    UInt microfacet_fresnel;
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
    Float3 microfacet_tangent;
    Float microfacet_alpha_x;
    Float microfacet_alpha_y;
    Float diffuse_roughness;
    Float metallic;
    Float ior;
    Float thin_film_thickness;
    Float thin_film_ior;
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

    // Intentional implicit narrowing for physical consumers. Their signatures
    // accept SurfaceClosurePhysicalRecord, so setup/AOV fields are absent from
    // the implementation type rather than merely ignored by convention.
    [[nodiscard]] operator SurfaceClosurePhysicalRecord() const noexcept {
        return {
            .closure_type = closure_type,
            .microfacet_fresnel = microfacet_fresnel,
            .weight = weight,
            .sample_weight = sample_weight,
            .color = color,
            .normal = normal,
            .roughness = roughness,
            .microfacet_tangent = microfacet_tangent,
            .microfacet_alpha_x = microfacet_alpha_x,
            .microfacet_alpha_y = microfacet_alpha_y,
            .ior = ior,
            .thin_film_thickness = thin_film_thickness,
            .thin_film_ior = thin_film_ior,
            .specular_tint = specular_tint,
            .sheen_transform_a = sheen_transform_a,
            .sheen_transform_b = sheen_transform_b,
            .evaluation_scale = evaluation_scale,
            .fresnel_f0 = fresnel_f0,
            .fresnel_f90 = fresnel_f90,
            .reflection_tint = reflection_tint,
            .transmission_tint = transmission_tint,
            .bssrdf_radius = bssrdf_radius,
            .bssrdf_albedo = bssrdf_albedo,
            .bssrdf_ior = bssrdf_ior,
            .bssrdf_roughness = bssrdf_roughness,
            .bssrdf_anisotropy = bssrdf_anisotropy};
    }

    [[nodiscard]] static SurfaceClosureRecord zero() noexcept {
        return {
            .closure_type = 0u,
            .microfacet_fresnel = 0u,
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
            .microfacet_tangent = make_float3(0.0f),
            .microfacet_alpha_x = 0.0f,
            .microfacet_alpha_y = 0.0f,
            .diffuse_roughness = 0.0f,
            .metallic = 0.0f,
            .ior = 1.0f,
            .thin_film_thickness = 0.0f,
            .thin_film_ior = 0.0f,
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

// Typed multistage boundaries for physical closure setup shared across
// material topologies. Inputs remain Luisa expressions produced by the
// authored shader graph; the host Boolean is immutable topology metadata.
// No material value is evaluated or serialized on the host.
struct PrincipledMetallicSetupInput {
    Float3 lower_weight;
    Float3 color;
    Float3 normal;
    Float3 incoming;
    Float3 surface_shading_normal;
    Float3 surface_geometric_normal;
    Float3 specular_tint;
    Float roughness;
    Float metallic;
    Float thin_film_thickness;
    Float thin_film_ior;
    Bool use_bump_map_correction;
    bool thin_film_enabled{};
    bool preserve_ggx_energy{};
};

struct PrincipledMetallicSetupResult {
    Float3 weight;
    Float allocation_weight;
    Float sample_weight;
    Float3 albedo;
    Float3 normal;
    Float3 color;
    Float3 specular_tint;
    Float3 evaluation_scale;
    Float3 lower_weight;
};

struct PrincipledDiffuseSetupInput {
    Float3 lower_weight;
    Float3 color;
    Float subsurface_weight;
};

struct PrincipledDiffuseSetupResult {
    Float3 weight;
    Float allocation_weight;
    Float sample_weight;
};

struct PrincipledDielectricSetupInput {
    Float3 lower_weight;
    Float3 normal;
    Float3 incoming;
    Float3 surface_shading_normal;
    Float3 surface_geometric_normal;
    Float roughness;
    Float ior;
    Float specular_ior_level;
    Float3 specular_tint;
    Float thin_film_thickness;
    Float thin_film_ior;
    Bool use_bump_map_correction;
    bool thin_film_enabled{};
    bool preserve_ggx_energy{};
};

struct PrincipledDielectricSetupResult {
    Float3 weight;
    Float allocation_weight;
    Float sample_weight;
    Float3 albedo;
    Float3 normal;
    Float3 color;
    Float ior;
    Float3 evaluation_scale;
    Float3 lower_weight;
};

class SurfaceClosureSetupProvider {

public:
    virtual ~SurfaceClosureSetupProvider() noexcept = default;

    [[nodiscard]] virtual PrincipledMetallicSetupResult
    principled_metallic(
        const PrincipledMetallicSetupInput &input,
        Expr<bool> reflective_caustics) const noexcept = 0;

    [[nodiscard]] virtual PrincipledDiffuseSetupResult
    principled_diffuse(
        const PrincipledDiffuseSetupInput &input) const noexcept = 0;

    [[nodiscard]] virtual PrincipledDielectricSetupResult
    principled_dielectric(
        const PrincipledDielectricSetupInput &input,
        Expr<bool> reflective_caustics) const noexcept = 0;
};

// Host-stage semantic boundary for pure color-space transforms shared across
// independently lowered shader graphs. Implementations receive and return
// typed Luisa expressions; no device-side opcode, weak register protocol, or
// host-evaluated material value crosses this interface.
class SurfaceColorTransformProvider {

public:
    virtual ~SurfaceColorTransformProvider() noexcept = default;

    [[nodiscard]] virtual Float3 rgb_to_hsv(
        Float3 rgb) const noexcept = 0;

    [[nodiscard]] virtual Float3 hsv_to_rgb(
        Float3 hsv) const noexcept = 0;

    [[nodiscard]] virtual Float3 rgb_to_hsl(
        Float3 rgb) const noexcept = 0;

    [[nodiscard]] virtual Float3 hsl_to_rgb(
        Float3 hsl) const noexcept = 0;
};

// The Mapping node's vector type is immutable graph metadata, so production
// integrations can select one strongly typed transform while recording the
// shader instead of materializing a device-side mode switch.
class SurfaceVectorMappingProvider {

public:
    virtual ~SurfaceVectorMappingProvider() noexcept = default;

    [[nodiscard]] virtual Float3 map_point(
        Float3 input,
        Float3 location,
        Float3 rotation,
        Float3 scale) const noexcept = 0;

    [[nodiscard]] virtual Float3 map_texture(
        Float3 input,
        Float3 location,
        Float3 rotation,
        Float3 scale) const noexcept = 0;

    [[nodiscard]] virtual Float3 map_vector(
        Float3 input,
        Float3 rotation,
        Float3 scale) const noexcept = 0;

    [[nodiscard]] virtual Float3 map_normal(
        Float3 input,
        Float3 rotation,
        Float3 scale) const noexcept = 0;
};

// Complete typed input of one Cycles Image Texture BOX projection. The
// interpolation and extension modes remain immutable host-stage metadata on
// SurfaceImageBoxProvider::evaluate; all fields below are true device values.
// In particular, no SVM stack, program counter, or weak float4 register bank
// crosses this operation boundary.
struct SurfaceImageBoxInput {
    Float3 coordinate;
    Float3 signed_normal;
    Float blend;
    UInt texture_handle;
    Bool unassociate_alpha;
    Bool encoded_as_srgb;
};

// Host-stage semantic boundary for the pure Image Texture BOX operation.
// Implementations may share one typed callable per canonical sampling mode.
// The operation consumes one node's inputs and returns its one RGBA result;
// loading graph operands and storing the result remain in the owning graph
// transaction.
class SurfaceImageBoxProvider {

public:
    virtual ~SurfaceImageBoxProvider() noexcept = default;

    [[nodiscard]] virtual Float4 evaluate(
        const SurfaceImageBoxInput &input,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept = 0;
};

// Host-stage expression view for the finite Normal Map evaluation family.
// Immutable graph metadata selects one strongly typed endpoint before shader
// recording. This record gives those values semantic names; production
// providers project only the fields required by the selected endpoint into its
// narrow callable ABI, never a weak device-side register protocol.
struct SurfaceNormalMapInput {
    Float3 mapped;
    Float strength;
    Float3 object_tangent;
    Float tangent_sign;
    Bool tangent_attribute_found;
    Float3 object_shading_normal;
    Float3 undisplaced_object_shading_normal;
    Bool triangle_smooth;
    Float3 normal_to_world_x;
    Float3 normal_to_world_y;
    Float3 normal_to_world_z;
    Float3 shading_normal;
    Bool back_facing;
    UInt geometry_index;
    Bool is_curve;
};

class SurfaceNormalMapProvider {

public:
    virtual ~SurfaceNormalMapProvider() noexcept = default;

    [[nodiscard]] virtual Float3 evaluate_tangent_displaced(
        const SurfaceNormalMapInput &input) const noexcept = 0;

    [[nodiscard]] virtual Float3 evaluate_tangent_original(
        const SurfaceNormalMapInput &input) const noexcept = 0;

    [[nodiscard]] virtual Float3 evaluate_object(
        const SurfaceNormalMapInput &input) const noexcept = 0;

    [[nodiscard]] virtual Float3 evaluate_world(
        const SurfaceNormalMapInput &input) const noexcept = 0;
};

// Bump height dependencies are evaluated by the owning graph at the center
// and two differential points. This expression view begins only after those
// topology-specific evaluations and describes the finite geometric perturbation
// stage shared by all Bump nodes.
struct SurfaceBumpInput {
    Float3 normal;
    Float filter_width;
    Float3 dPdx;
    Float3 dPdy;
    Float height_center;
    Float height_x;
    Float height_y;
    Float distance;
    Float strength;
    Float3 normal_to_world_x;
    Float3 normal_to_world_y;
    Float3 normal_to_world_z;
    Float3 object_shading_normal;
    Float3 shading_normal;
};

class SurfaceBumpProvider {

public:
    virtual ~SurfaceBumpProvider() noexcept = default;

    [[nodiscard]] virtual Float3 evaluate_world(
        const SurfaceBumpInput &input) const noexcept = 0;

    [[nodiscard]] virtual Float3 evaluate_object(
        const SurfaceBumpInput &input) const noexcept = 0;
};

struct SurfaceShaderTableView {
    UInt offset;
    UInt count;
    UInt width;
};

// Color Ramp and RGB Curves use runtime table descriptors, so authored table
// cardinality remains data rather than shader structure. Their representation
// and interpolation policies are immutable graph metadata selected here at the
// host stage; implementations never receive a weak device opcode.
class SurfaceShaderTableProvider {

public:
    virtual ~SurfaceShaderTableProvider() noexcept = default;

    [[nodiscard]] virtual Float4 color_ramp_sampled_linear(
        const SurfaceShaderTableView &table,
        Float factor) const noexcept = 0;

    [[nodiscard]] virtual Float4 color_ramp_sampled_constant(
        const SurfaceShaderTableView &table,
        Float factor) const noexcept = 0;

    [[nodiscard]] virtual Float4 color_ramp_control_linear(
        const SurfaceShaderTableView &table,
        Float factor) const noexcept = 0;

    [[nodiscard]] virtual Float4 color_ramp_control_constant(
        const SurfaceShaderTableView &table,
        Float factor) const noexcept = 0;

    [[nodiscard]] virtual Float3 rgb_curve_sampled(
        const SurfaceShaderTableView &table,
        Float3 input,
        Float factor,
        Float min_x,
        Float max_x,
        Float extrapolate) const noexcept = 0;

    [[nodiscard]] virtual Float3 rgb_curve_control(
        const SurfaceShaderTableView &table,
        Float3 input,
        Float factor) const noexcept = 0;
};

// Non-owning host-stage view of a canonical closure's Luisa expressions.
// Copying this type only copies AST expression handles: it neither declares
// device variables nor evaluates, serializes, or bakes material data. This is
// the representation retained by multistage visitors while a material branch
// is being recorded.
struct SurfaceClosureExpression {
    Expr<std::uint32_t> closure_type;
    Expr<std::uint32_t> microfacet_fresnel;
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
    Expr<luisa::float3> microfacet_tangent;
    Expr<float> microfacet_alpha_x;
    Expr<float> microfacet_alpha_y;
    Expr<float> diffuse_roughness;
    Expr<float> metallic;
    Expr<float> ior;
    Expr<float> thin_film_thickness;
    Expr<float> thin_film_ior;
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

    // Cycles allocates the first retained transparent closure at its source
    // position and merges every later transparent contribution into that same
    // record. A collector with write-only finalization may opt into receiving
    // the first record at its source position and the final additive state at
    // the end of traversal. Other collectors keep the canonical pre-merged
    // sequence produced by the surface implementation, so this protocol is an
    // implementation capability rather than a change to add()'s contract.
    [[nodiscard]] virtual bool
    supports_transparent_closure_finalization() const noexcept {
        return false;
    }
    virtual void begin_transparent_closure(
        const SurfaceClosureRecord &closure) noexcept {
        add(closure);
    }
    virtual void finalize_transparent_closure(
        Expr<luisa::float3>,
        Expr<float>) noexcept {}
    virtual void finish() noexcept {}
};

struct SurfaceClosureCollection {
    Float3 shading_normal;
};

// Result of the single path-hit shader execution. Emission and the canonical
// physical closure sequence are produced from one shared value schedule; the
// latter is streamed into the caller-provided SurfaceClosureCollector. This is
// the Luisa host/JIT analogue of Cycles populating one local ShaderData before
// any BSDF consumer runs.
struct SurfacePopulation {
    Float3 emission;
    Float3 shading_normal;
};

struct SurfacePopulationQuery {
    Expr<bool> emission_reflective_caustics;
    Expr<bool> reflective_caustics;
    Expr<bool> refractive_caustics;
    Expr<float> glossy_filter_roughness;
    Expr<bool> include_runtime_flags;
    Expr<bool> include_aov;
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

struct SurfacePreparationQuery {
    Float3 outgoing;
    Float glossy_filter_roughness;
    Bool emission_reflective_caustics;
    Bool reflective_caustics;
    Bool refractive_caustics;
    Bool include_runtime_flags;
    Bool include_aov;
};

// Compact output of one production surface-program traversal. This is not a
// baked material representation: GraphSurface still evaluates the original
// typed value graph and raw physical closures at the current SurfacePoint.
// Only the reductions consumed before BSDF sampling cross this boundary.
struct SurfacePreparation {
    Float3 emission;
    // Final ShaderData::N-equivalent after the graph's Bump/Set Normal
    // domain. This is deliberately distinct from the closure-weighted Normal
    // AOV so downstream geometry and BSSRDF consumers never replay the graph.
    Float3 shading_normal;
    UInt runtime_flags;
    SurfaceAov aov;

    [[nodiscard]] static SurfacePreparation zero(
        const SurfacePoint &point) noexcept {
        return {
            .emission = make_float3(0.0f),
            .shading_normal = point.shading_normal,
            .runtime_flags = 0u,
            .aov = {
                .albedo = make_float3(0.0f),
                .glossy_albedo = make_float3(0.0f),
                .transmission_albedo = make_float3(0.0f),
                .roughness = make_float2(0.0f),
                .normal = point.shading_normal,
                .transparency = make_float3(0.0f)}};
    }
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

    [[nodiscard]] virtual ULong parameter_uint64(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept = 0;
};

class ShaderServices : public SurfaceParameterServices {

public:
    ~ShaderServices() noexcept override = default;

    // Path-tracer integrations may provide shared typed setup callables.
    // Standalone GraphSurface clients keep the exact inline implementation.
    [[nodiscard]] virtual const SurfaceClosureSetupProvider *
    surface_closure_setup_provider() const noexcept {
        return nullptr;
    }

    // Production integrations may share the finite family of pure color
    // transforms across material topologies. A null provider preserves the
    // canonical inline GraphSurface path for standalone clients.
    [[nodiscard]] virtual const SurfaceColorTransformProvider *
    surface_color_transform_provider() const noexcept {
        return nullptr;
    }

    [[nodiscard]] virtual const SurfaceVectorMappingProvider *
    surface_vector_mapping_provider() const noexcept {
        return nullptr;
    }

    [[nodiscard]] virtual const SurfaceImageBoxProvider *
    surface_image_box_provider() const noexcept {
        return nullptr;
    }

    [[nodiscard]] virtual const SurfaceNormalMapProvider *
    surface_normal_map_provider() const noexcept {
        return nullptr;
    }

    [[nodiscard]] virtual const SurfaceBumpProvider *
    surface_bump_provider() const noexcept {
        return nullptr;
    }

    [[nodiscard]] virtual const SurfaceShaderTableProvider *
    surface_shader_table_provider() const noexcept {
        return nullptr;
    }

    [[nodiscard]] virtual Float4 texture_2d(
        Expr<std::uint32_t> handle,
        Expr<luisa::float2> uv,
        Expr<luisa::float2> d_uv_dx,
        Expr<luisa::float2> d_uv_dy,
        std::uint32_t interpolation,
        std::uint32_t extension) const noexcept = 0;

    [[nodiscard]] virtual ShaderAttribute attribute(
        Expr<luisa::ulong> attribute_id,
        const SurfacePoint &point) const noexcept = 0;

    // Attribute hashes are stored and serialized as std::uint64_t on the
    // host, while device expressions use Luisa's canonical ulong type.
    // Keep the only fundamental-type conversion at this staging boundary;
    // the generated shader receives the exact 64-bit bit pattern.
    [[nodiscard]] ShaderAttribute attribute(
        std::uint64_t attribute_id,
        const SurfacePoint &point) const noexcept {
        static_assert(sizeof(std::uint64_t) == sizeof(luisa::ulong));
        return attribute(
            Expr<luisa::ulong>{
                static_cast<luisa::ulong>(attribute_id)},
            point);
    }

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

    // Production graph surfaces override this boundary to share one value
    // trace between emission and closure population. The conservative default
    // keeps custom/test surfaces source-compatible; it preserves semantics but
    // does not claim the single-evaluation property.
    [[nodiscard]] virtual SurfacePopulation populate(
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector) const noexcept {
        const auto collection = collect_closures(
            services,
            point,
            query.reflective_caustics,
            query.refractive_caustics,
            collector);
        return {
            .emission = emission(
                services,
                point,
                point.incoming,
                query.emission_reflective_caustics),
            .shading_normal = collection.shading_normal};
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

    // The only path-hit preparation boundary. Implementations must derive
    // emission, runtime flags, and optional camera AOVs from one material
    // evaluation; keeping split virtual entry points would permit accidental
    // replay of the shader graph.
    [[nodiscard]] virtual SurfacePreparation prepare(
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePreparationQuery &query) const noexcept = 0;

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

    [[nodiscard]] SurfacePopulation populate(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector) const noexcept {
        auto result = SurfacePopulation{
            .emission = make_float3(0.0f),
            .shading_normal = point.shading_normal};
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->populate(
                services, point, query, collector);
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

    [[nodiscard]] SurfacePreparation prepare(
        Expr<std::uint32_t> tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePreparationQuery &query) const noexcept {
        auto result = SurfacePreparation::zero(point);
        _surfaces.dispatch(tag, [&](const Surface *surface) noexcept {
            result = surface->prepare(services, point, query);
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
        luisa::vector<luisa::uint> emissive_tags;
        emissive_tags.reserve(_surfaces.size());
        for (auto index = std::size_t{0u};
             index < _surfaces.size();
             ++index) {
            if (_surfaces.impl(index)
                    ->capabilities()
                    .may_emit) {
                emissive_tags.emplace_back(
                    static_cast<luisa::uint>(index));
            }
        }
        if (!emissive_tags.empty()) {
            _surfaces.dispatch_group_with_default(
                tag,
                emissive_tags,
                [&](const Surface *surface) noexcept {
                    result = surface->emission(
                        services,
                        point,
                        wo,
                        reflective_caustics);
                },
                []() noexcept {});
        }
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
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfacePreparationQuery)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(psycles::luisa_backend::SurfacePreparation)

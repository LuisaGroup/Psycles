#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <luisa/core/basic_types.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

struct GeometryGpu {
    luisa::uint bindless_base{};
    luisa::uint material_offset{};
    luisa::uint material_count{};
    luisa::uint attribute_domains{};
    luisa::uint cycles_primitive_offset{};
    luisa::uint cycles_segment_offset{};
    luisa::uint primitive_kind{};
    luisa::uint curve_shape{};
    luisa::uint curve_subdivision_level{};
    // Cycles ATTR_STD_GENERATED_TRANSFORM. Surface points retain explicit
    // per-vertex Generated values; volume points apply this transform to
    // object-space P because they have no primitive to interpolate.
    luisa::float4x4 generated_transform{};
};

inline constexpr std::uint32_t geometry_kind_triangle = 0u;
inline constexpr std::uint32_t geometry_kind_curve = 1u;

struct CurveSegmentGpu {
    luisa::uint key_before{};
    luisa::uint key_begin{};
    luisa::uint key_end{};
    luisa::uint key_after{};
    luisa::uint curve_index{};
    luisa::uint cycles_curve_index{};
    luisa::uint cycles_segment_index{};
    luisa::uint padding{};
};

struct AttributeBindingGpu {
    luisa::ulong id{};
    luisa::uint value_slot{};
    // Low 8 bits are the geometry domain; upper bits encode the element
    // storage format. Keeping both in one word preserves the 16-byte ABI.
    luisa::uint domain{};
};

struct AttributeRangeGpu {
    luisa::uint offset{};
    luisa::uint count{};
    // Triangle geometry stores Triangle records here; curve geometry stores
    // CurveSegmentGpu records. Each binding domain selects the typed view.
    luisa::uint primitive_slot{};
    luisa::uint padding{};
};

struct InstanceGpu {
    luisa::uint geometry_index{};
    luisa::uint override_offset{};
    luisa::uint override_count{};
    // Contract ray-visibility bits normalized during scene upload. The
    // permissive default keeps hand-authored device fixtures shadow-visible.
    luisa::uint visibility_mask{0xffu};
    float object_random{};
    luisa::uint particle_index{};
    float shadow_terminator_geometry_offset{};
    luisa::uint cycles_object_index{};
    // Immutable denormalization of
    // GeometryGpu::cycles_primitive_offset for this instance. Closest-hit
    // candidate filtering reads object and primitive identity together from
    // one compact table location instead of chasing a second 112-byte-stride
    // geometry record. Scene construction owns the equality invariant.
    luisa::uint cycles_primitive_offset{};
    std::int32_t cycles_light_group{};
    luisa::uint is_shadow_catcher{};
    luisa::uint cycles_transform_applied{};
    luisa::float4x4 cycles_world_to_object{};
};

static_assert(sizeof(InstanceGpu) == 112u);
static_assert(offsetof(InstanceGpu, cycles_object_index) == 28u);
static_assert(offsetof(InstanceGpu, cycles_primitive_offset) == 32u);
static_assert(offsetof(InstanceGpu, cycles_world_to_object) == 48u);

// Immutable quotient of InstanceGpu + GeometryGpu for traversal callbacks.
// The callback observes only identity, primitive addressing, the material
// selection relation, and curve intersection metadata. Keeping exactly those
// fields in one cache-line-sized record avoids loading either full shading
// record for every any-hit candidate. Material flags live in a separate dense
// uint table; no closure value or sampled opacity is stored here.
struct SceneTraversalInstanceGpu {
    luisa::uint bindless_base{};
    luisa::uint geometry_material_offset{};
    luisa::uint geometry_material_count{};
    luisa::uint override_material_offset{};
    luisa::uint override_material_count{};
    luisa::uint cycles_object_index{};
    luisa::uint cycles_primitive_offset{};
    // bit 0: primitive kind; bits [31:1]: curve subdivision level.
    luisa::uint primitive_kind_and_curve_subdivision{};
};

inline constexpr std::uint32_t
scene_traversal_primitive_kind_mask = 1u;
inline constexpr std::uint32_t
scene_traversal_curve_subdivision_shift = 1u;
inline constexpr std::uint32_t
scene_traversal_curve_subdivision_maximum =
    std::numeric_limits<std::uint32_t>::max() >>
    scene_traversal_curve_subdivision_shift;

[[nodiscard]] constexpr std::uint32_t
pack_scene_traversal_primitive(std::uint32_t primitive_kind,
                               std::uint32_t curve_subdivision) noexcept {
    return (curve_subdivision <<
            scene_traversal_curve_subdivision_shift) |
           (primitive_kind & scene_traversal_primitive_kind_mask);
}

static_assert(sizeof(SceneTraversalInstanceGpu) == 32u);
static_assert(alignof(SceneTraversalInstanceGpu) == alignof(luisa::uint));
static_assert(offsetof(SceneTraversalInstanceGpu, cycles_object_index) == 20u);

inline constexpr std::uint32_t material_flag_has_volume =
    1u << 0u;
inline constexpr std::uint32_t material_flag_may_emit =
    1u << 1u;
inline constexpr std::uint32_t material_flag_constant_emission =
    1u << 2u;
inline constexpr std::uint32_t material_flag_use_bump_map_correction =
    1u << 3u;
// EmissionSampling has five states and therefore occupies three bits. Packing
// it into the existing flag word keeps MaterialBindingGpu at its original
// 24-byte stride instead of adding another 32-bit lane to every lookup.
inline constexpr std::uint32_t material_emission_sampling_shift = 4u;
inline constexpr std::uint32_t material_emission_sampling_mask =
    0x7u << material_emission_sampling_shift;

// Exact Cycles SD_HAS_TRANSPARENT_SHADOW projection. A clear bit proves that
// the material is opaque to shadow rays; a set bit requests deferred raw
// closure evaluation and also includes volume boundaries. It does not encode
// sampled or precomputed opacity.
inline constexpr std::uint32_t material_flag_has_transparent_shadow =
    1u << 7u;
// Exact Cycles Shader::has_bssrdf_bump metadata for this parameter block.
// Unlike surface_tag, this must remain material-specific: two materials can
// share graph topology while direct parameters prove the BSSRDF unreachable
// for only one of them.
inline constexpr std::uint32_t material_flag_has_bssrdf_bump = 1u << 8u;

[[nodiscard]] constexpr std::uint32_t
material_emission_sampling_bits(
    std::uint32_t sampling) noexcept {
    return (sampling <<
            material_emission_sampling_shift) &
           material_emission_sampling_mask;
}

[[nodiscard]] constexpr std::uint32_t
material_emission_sampling_value(
    std::uint32_t flags) noexcept {
    return (flags &
            material_emission_sampling_mask) >>
           material_emission_sampling_shift;
}

static_assert(
    material_emission_sampling_value(
        material_emission_sampling_bits(4u)) == 4u);
static_assert((material_emission_sampling_mask &
               (material_flag_has_volume | material_flag_may_emit |
                material_flag_constant_emission |
                material_flag_use_bump_map_correction |
                material_flag_has_bssrdf_bump |
                material_flag_has_transparent_shadow)) == 0u);

struct MaterialBindingGpu {
    luisa::uint surface_tag{};
    luisa::uint parameter_block{};
    luisa::uint cycles_shader_index{};
    luisa::uint material_identity{};
    luisa::uint flags{};
    luisa::uint volume_sampling{};
};

// Preserve authored slot-table indices without compiling unreachable shader
// graphs. A hole can only be emitted after host reachability proved that no
// runtime primitive resolves to it; all identity-bearing fields are poisoned
// so the record cannot accidentally masquerade as material/program zero.
[[nodiscard]] constexpr MaterialBindingGpu
inert_material_binding() noexcept {
    constexpr auto invalid = ~luisa::uint{0u};
    return {
        .surface_tag = invalid,
        .parameter_block = invalid,
        .cycles_shader_index = invalid,
        .material_identity = invalid};
}

static_assert(sizeof(MaterialBindingGpu) == 24u);
static_assert(alignof(MaterialBindingGpu) == alignof(luisa::uint));
static_assert(offsetof(MaterialBindingGpu, flags) == 16u);
static_assert(offsetof(MaterialBindingGpu, volume_sampling) == 20u);

struct LightGpu {
    luisa::uint type{};
    luisa::float3 position{};
    luisa::float3 axis_x{};
    luisa::float3 axis_y{};
    luisa::float3 axis_z{};
    luisa::float3 axis_scale{1.0f, 1.0f, 1.0f};
    luisa::float3 color{};
    float power{};
    float radius{};
    float size_u{};
    float size_v{};
    float spread{};
    float spot_angle{};
    float spot_smooth{};
    float angle{};
    luisa::uint flags{};
    luisa::uint surface_tag{};
    luisa::uint parameter_block{};
    luisa::uint cycles_object_index{};
    std::int32_t cycles_light_group{};
    luisa::uint cycles_shader_id{};
    luisa::uint cycles_shader_flags{};
    luisa::uint cycles_type{};
    luisa::uint visibility_mask{};
    luisa::uint max_bounces{1024u};
};

struct EmissiveTriangleGpu {
    luisa::uint instance_index{};
    luisa::uint geometry_index{};
    luisa::uint primitive_index{};
    luisa::uint surface_tag{};
    luisa::uint parameter_block{};
    luisa::uint emission_sampling{};
    luisa::uint emission_is_constant{};
    luisa::uint visibility_mask{};
    luisa::uint cycles_primitive_index{};
    luisa::uint cycles_object_index{};
    luisa::uint cycles_shader_id{};
    luisa::uint cycles_shader_flags{};
    std::int32_t cycles_light_group{};
};

struct LightDistributionGpu {
    float cumulative{};
    float selection_pdf{};
    luisa::uint kind{};
    luisa::uint index{};
    // Position in Cycles' flattened emitter array. This is distinct from the
    // per-kind index above and is part of the sampled-light identity.
    luisa::uint emitter_id{};
    luisa::uint padding_0{};
    luisa::uint padding_1{};
    luisa::uint padding_2{};
};

// Light-tree measures are kept in 16-byte lanes. Besides making the ABI
// explicit across fallback/HIP/Vulkan, this avoids the backend-dependent
// padding of interleaved float3/scalar fields.
struct LightTreeNodeGpu {
    luisa::float4 bounds_min_energy{};
    luisa::float4 bounds_max_theta_o{};
    luisa::float4 cone_axis_theta_e{};
    // parent, left child, right child, LightTreeNodeKind
    luisa::uint4 topology{};
    // first emitter, emitter count, measure flags, reserved
    luisa::uint4 emitters{};
};

struct LightTreeEmitterGpu {
    luisa::float4 bounds_min_energy{};
    luisa::float4 bounds_max_theta_o{};
    luisa::float4 cone_axis_theta_e{};
    // payload 0, measure flags | LightTreeEmitterKind, payload 1, payload 2
    luisa::uint4 identity{};
};

inline constexpr std::uint32_t light_tree_measure_has_bounds = 1u << 0u;
inline constexpr std::uint32_t light_tree_measure_has_orientation = 1u << 1u;
inline constexpr std::uint32_t light_tree_measure_is_distant = 1u << 2u;

enum class LightTreeEmitterKind : std::uint32_t {
    direct,
    mesh_instance,
    mesh_triangle
};

inline constexpr std::uint32_t light_tree_emitter_kind_shift = 3u;
inline constexpr std::uint32_t light_tree_emitter_kind_mask =
    0x3u << light_tree_emitter_kind_shift;

[[nodiscard]] constexpr std::uint32_t light_tree_emitter_kind_bits(
    LightTreeEmitterKind kind) noexcept {
    return static_cast<std::uint32_t>(kind) <<
           light_tree_emitter_kind_shift;
}

// A tree emitter has two independent identities. LightTreeEmitterKind
// describes how traversal resolves it (direct distribution entry, mesh
// instance proxy, or mesh-local triangle); LightTreeEmitterSource describes
// which exact Cycles leaf-importance equation supplies its radiometric
// measure. Keeping these axes orthogonal prevents hierarchy metadata from
// being overloaded with an implicit light subtype.
enum class LightTreeEmitterSource : std::uint32_t {
    measure,
    triangle,
    analytic_light,
    environment
};

inline constexpr std::uint32_t light_tree_emitter_source_shift = 5u;
inline constexpr std::uint32_t light_tree_emitter_source_mask =
    0x3u << light_tree_emitter_source_shift;

[[nodiscard]] constexpr std::uint32_t light_tree_emitter_source_bits(
    LightTreeEmitterSource source) noexcept {
    return static_cast<std::uint32_t>(source) <<
           light_tree_emitter_source_shift;
}

static_assert(sizeof(LightTreeNodeGpu) == 80u);
static_assert(sizeof(LightTreeEmitterGpu) == 64u);
static_assert((light_tree_emitter_kind_mask &
               (light_tree_measure_has_bounds |
                light_tree_measure_has_orientation |
                light_tree_measure_is_distant)) == 0u);
static_assert((light_tree_emitter_source_mask &
               (light_tree_emitter_kind_mask |
                light_tree_measure_has_bounds |
                light_tree_measure_has_orientation |
                light_tree_measure_is_distant)) == 0u);

struct ShaderEvaluationStateCall {
    luisa::uint ray_visibility{};
    luisa::uint ray_events{};
    luisa::uint ray_depth{};
    luisa::uint diffuse_depth{};
    luisa::uint glossy_depth{};
    luisa::uint transparent_depth{};
    luisa::uint transmission_depth{};
};

// IntegratorShadowState inputs read by surface_shader_eval. RNG offset is
// advanced after every surviving surface hit, not captured once per batch.
struct ShadowShaderContextCall {
    ShaderEvaluationStateCall path{};
    float ray_time{};
    luisa::uint sample_index{};
    luisa::uint rng_hash{};
    luisa::uint rng_offset{};
    luisa::uint volume_bounds_bounce{};
};

// Backend-neutral result of the geometry-only shadow traversal stage. Keeping
// this record free of material data makes the INTERSECT_SHADOW ->
// SHADE_SHADOW boundary explicit: traversal identifies the closest candidate,
// while the following stage alone evaluates transparency/volume closures.
struct ShadowIntersectionCall {
    luisa::uint instance{};
    luisa::uint primitive{};
    luisa::uint hit_type{};
    float distance{};
    luisa::float2 barycentric{};
};

// Cycles' GPU shadow state retains four closest intersections from one BVH
// traversal. Keep the same bounded state transition here: `count` entries form
// the unordered nearest-hit set, `total` counts accepted transparent
// candidates, and a nonzero `blocked` proves that an opaque candidate or
// bounce-budget overflow terminated traversal. Consumers order the bounded set
// immediately before shading. Cycles continues traversal whenever all four
// slots were filled, including an exactly-full final batch.
inline constexpr std::size_t shadow_intersection_batch_capacity = 4u;

// Private reduction state returned by an externally stored shadow traversal.
// The retained intersections are deliberately absent: candidate callbacks
// write them directly to the invocation-owned SoA, exactly as Cycles writes
// shadow intersections to IntegratorShadowState. This type boundary prevents
// a RayQuery callable from silently rematerializing the four-hit array in its
// return ABI and extending that private live range across traversal.
struct ShadowIntersectionSummaryCall {
  luisa::uint count{};
  luisa::uint total{};
  luisa::uint blocked{};
};

static_assert(sizeof(ShadowIntersectionSummaryCall) == 12u);

struct ShadowIntersectionBatchCall {
  std::array<ShadowIntersectionCall, shadow_intersection_batch_capacity> hits{};
  luisa::uint count{};
  luisa::uint total{};
  luisa::uint blocked{};
};

struct ShadowSurfaceEvaluationCall {
    luisa::float3 transmittance{};
    luisa::uint object{};
    luisa::uint primitive{};
    luisa::uint kind{};
    luisa::uint volume_boundary{};
};

struct ShadowTraceResultCall {
    // The factor is retained for legacy volume and optional path tracing
    // diagnostics. Native direct lighting consumes the canonical throughput.
    luisa::float3 transmittance{};
    luisa::float3 throughput{};
    luisa::uint first_hit{};
    luisa::uint first_object{};
    luisa::uint first_primitive{};
    luisa::uint first_kind{};
    float first_distance{};
    luisa::float2 first_barycentric{};
};

struct SurfacePointCall {
    luisa::float3 position{};
    luisa::float3 object_position{};
    luisa::float3 object_location{};
    luisa::float3 generated{};
    luisa::float3 geometric_normal{};
    luisa::float3 shading_normal{};
    luisa::float3 object_shading_normal{};
    luisa::float3 object_tangent{};
    float tangent_sign{};
    luisa::float3 undisplaced_position{};
    luisa::float3 undisplaced_object_position{};
    luisa::float3 undisplaced_shading_normal{};
    luisa::float3 undisplaced_object_shading_normal{};
    luisa::float3 undisplaced_object_tangent{};
    float undisplaced_tangent_sign{};
    luisa::float3 normal_to_world_x{};
    luisa::float3 normal_to_world_y{};
    luisa::float3 normal_to_world_z{};
    luisa::float3 dpdu{};
    luisa::float3 dpdv{};
    luisa::float3 dPdx{};
    luisa::float3 dPdy{};
    luisa::float3 object_dPdx{};
    luisa::float3 object_dPdy{};
    luisa::float3 undisplaced_dPdx{};
    luisa::float3 undisplaced_dPdy{};
    luisa::float3 undisplaced_object_dPdx{};
    luisa::float3 undisplaced_object_dPdy{};
    luisa::float3 generated_dx{};
    luisa::float3 generated_dy{};
    luisa::float3 incoming{};
    luisa::float2 uv{};
    luisa::float2 uv_dx{};
    luisa::float2 uv_dy{};
    luisa::uint geometry_index{};
    luisa::float2 barycentric{};
    luisa::float2 barycentric_dx{};
    luisa::float2 barycentric_dy{};
    luisa::uint instance_id{};
    luisa::uint primitive_id{};
    luisa::uint parameter_block{};
    float object_random{};
    luisa::uint particle_index{};
    float random_per_island{};
    luisa::uint triangle_smooth{};
    luisa::uint is_curve{};
    float curve_intercept{};
    float curve_length{};
    float curve_thickness{};
    luisa::float3 curve_tangent_normal{};
    float curve_random{};
    luisa::uint ray_visibility{};
    luisa::uint ray_events{};
    luisa::uint ray_depth{};
    luisa::uint diffuse_depth{};
    luisa::uint glossy_depth{};
    luisa::uint transparent_depth{};
    luisa::uint transmission_depth{};
    float ray_length{};
    float time{};
    luisa::uint use_bump_map_correction{};
    luisa::uint back_facing{};
};

struct SurfaceEvaluationCall {
    luisa::float3 f{};
    float pdf{};
    luisa::float3 diffuse_f{};
    luisa::float3 glossy_f{};
    float diffuse_pdf{};
    float average_roughness_squared{};
    luisa::uint events{};
};

struct SurfaceSampleCall {
    luisa::float3 f{};
    float pdf{};
    luisa::float3 diffuse_f{};
    luisa::float3 glossy_f{};
    float diffuse_pdf{};
    float average_roughness_squared{};
    luisa::uint events{};
    luisa::float3 wi{};
    float eta{};
    luisa::float2 roughness{};
    luisa::uint runtime_flags{};
    luisa::uint bssrdf_method{};
    luisa::float3 bssrdf_radius{};
    luisa::float3 bssrdf_albedo{};
    luisa::float3 bssrdf_normal{};
    float bssrdf_ior{};
    float bssrdf_roughness{};
    float bssrdf_anisotropy{};
    luisa::uint valid{};
};

struct SurfaceClosureTraceCall {
    luisa::uint count{};
    luisa::uint runtime_flags{};
    luisa::uint index{};
    luisa::uint type{};
    float sample_weight{};
    luisa::float3 weight{};
    luisa::float3 normal{};
    luisa::uint valid{};
};

struct SurfaceSampleTraceCall {
    luisa::float3 f{};
    float pdf{};
    luisa::float3 diffuse_f{};
    luisa::float3 glossy_f{};
    float diffuse_pdf{};
    float average_roughness_squared{};
    luisa::uint events{};
    luisa::float3 wi{};
    float eta{};
    luisa::float2 roughness{};
    luisa::uint runtime_flags{};
    luisa::uint bssrdf_method{};
    luisa::float3 bssrdf_radius{};
    luisa::float3 bssrdf_albedo{};
    luisa::float3 bssrdf_normal{};
    float bssrdf_ior{};
    float bssrdf_roughness{};
    float bssrdf_anisotropy{};
    luisa::uint valid{};
    luisa::uint closure_index{};
    luisa::uint closure_type{};
    float closure_sample_weight{};
    float selection_rescaled{};
    luisa::float3 closure_weight{};
    luisa::float3 closure_normal{};
    luisa::uint closure_valid{};
};

struct SurfacePreparationQueryCall {
    luisa::float3 outgoing{};
    float glossy_filter_roughness{};
    luisa::uint flags{};
};

namespace surface_preparation_query_flag {
inline constexpr luisa::uint emission_reflective_caustics = 1u << 0u;
inline constexpr luisa::uint reflective_caustics = 1u << 1u;
inline constexpr luisa::uint refractive_caustics = 1u << 2u;
inline constexpr luisa::uint include_runtime_flags = 1u << 3u;
inline constexpr luisa::uint include_aov = 1u << 4u;
}// namespace surface_preparation_query_flag

// Largest-alignment fields first: this callable ABI is crossed for every
// surface hit, so avoid padding holes between the vector reductions.
struct SurfacePreparationCall {
    luisa::float3 emission{};
    // ShaderData::N-equivalent. `normal` below is the closure-weighted AOV.
    luisa::float3 shading_normal{};
    luisa::float3 albedo{};
    luisa::float3 glossy_albedo{};
    luisa::float3 transmission_albedo{};
    luisa::float3 normal{};
    luisa::float3 transparency{};
    luisa::float2 roughness{};
    luisa::uint runtime_flags{};
};

struct LightPassContributionCall {
    luisa::float3 diffuse_direct{};
    luisa::float3 diffuse_indirect{};
    luisa::float3 glossy_direct{};
    luisa::float3 glossy_indirect{};
    luisa::float3 transmission_direct{};
    luisa::float3 transmission_indirect{};
};

struct RenderKernelParameters {
    luisa::uint window_x{};
    luisa::uint window_y{};
    luisa::uint window_width{};
    luisa::uint full_width{};
    luisa::uint full_height{};
    luisa::uint seed{};
    luisa::uint sobol_sequence_size{};
    luisa::uint max_bounces{};
    luisa::uint min_bounces{};
    luisa::uint max_diffuse_bounces{};
    luisa::uint max_glossy_bounces{};
    luisa::uint max_transmission_bounces{};
    luisa::uint max_volume_bounces{};
    luisa::uint transparent_min_bounces{};
    luisa::uint transparent_max_bounces{};
    luisa::uint max_path_steps{};
    luisa::uint transparent_background{};
    luisa::uint path_trace_enabled{};
    luisa::uint path_trace_pixel_x{};
    luisa::uint path_trace_pixel_y{};
    luisa::uint path_trace_sample{};
    // Runtime Cycles KernelLight partition. These bounds must not specialize
    // shader AST/cache identity or invite portal-loop unrolling.
    luisa::uint analytic_light_count{};
    luisa::uint portal_count{};
    // Dense table indexed by Cycles object identity. A zero authored entry
    // inherits ambient_occlusion_distance exactly as Cycles does.
    luisa::uint ambient_occlusion_object_distance_count{};
    // Runtime allocation bound for coroutine-owned auxiliary queues. This is
    // deliberately a kernel argument: resolution and frame-pool capacity must
    // not specialize the shader AST or its cache identity.
    luisa::uint wavefront_frame_capacity{};
    // Runtime extent for an explicitly selected transient shadow-hit SoA.
    // Production local-batch traversal does not read either storage argument;
    // its cross-stage hit lifetime is owned by the coroutine frame alone.
    luisa::uint shadow_storage_capacity{};
    // Physical launch stride used to derive an injective transient-storage
    // owner from (block_id.x, thread_id.x). A callable has no block size of
    // its own, so the enclosing kernel must provide this value explicitly.
    luisa::uint shadow_storage_block_size{};
    luisa::uint ambient_occlusion_bounces{};
    float ambient_occlusion_factor{};
    float ambient_occlusion_distance{};
    float sample_clamp_direct{};
    float sample_clamp_indirect{};
    float filter_glossy{};
    float light_inv_rr_threshold{};
    float camera_horizontal_tangent{};
    float camera_vertical_tangent{};
    float camera_ortho_vertical_span{};
    float camera_shift_x{};
    float camera_shift_y{};
    float camera_near{};
    float camera_far{};
    float camera_aperture_radius{};
    float camera_focal_distance{};
    float camera_aperture_ratio{};
    float pass_alpha_threshold{};
    luisa::float3 background{};
    luisa::float4x4 camera_transform{};
    // Host-prepared inverse of the renderer camera pose; never invert per path.
    luisa::float4x4 camera_inverse_transform{};
};

}// namespace psycles::luisa_backend::detail

LUISA_STRUCT(
    psycles::luisa_backend::detail::GeometryGpu,
    bindless_base,
    material_offset,
    material_count,
    attribute_domains,
    cycles_primitive_offset,
    cycles_segment_offset,
    primitive_kind,
    curve_shape,
    curve_subdivision_level,
    generated_transform) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::CurveSegmentGpu,
    key_before,
    key_begin,
    key_end,
    key_after,
    curve_index,
    cycles_curve_index,
    cycles_segment_index,
    padding) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::AttributeBindingGpu,
    id,
    value_slot,
    domain) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::AttributeRangeGpu,
    offset,
    count,
    primitive_slot,
    padding) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::InstanceGpu,
    geometry_index,
    override_offset,
    override_count,
    visibility_mask,
    object_random,
    particle_index,
    shadow_terminator_geometry_offset,
    cycles_object_index,
    cycles_primitive_offset,
    cycles_light_group,
    is_shadow_catcher,
    cycles_transform_applied,
    cycles_world_to_object) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SceneTraversalInstanceGpu,
    bindless_base,
    geometry_material_offset,
    geometry_material_count,
    override_material_offset,
    override_material_count,
    cycles_object_index,
    cycles_primitive_offset,
    primitive_kind_and_curve_subdivision) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::MaterialBindingGpu,
    surface_tag,
    parameter_block,
    cycles_shader_index,
    material_identity,
    flags,
    volume_sampling) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::LightGpu,
    type,
    position,
    axis_x,
    axis_y,
    axis_z,
    axis_scale,
    color,
    power,
    radius,
    size_u,
    size_v,
    spread,
    spot_angle,
    spot_smooth,
    angle,
    flags,
    surface_tag,
    parameter_block,
    cycles_object_index,
    cycles_light_group,
    cycles_shader_id,
    cycles_shader_flags,
    cycles_type,
    visibility_mask,
    max_bounces) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::EmissiveTriangleGpu,
    instance_index,
    geometry_index,
    primitive_index,
    surface_tag,
    parameter_block,
    emission_sampling,
    emission_is_constant,
    visibility_mask,
    cycles_primitive_index,
    cycles_object_index,
    cycles_shader_id,
    cycles_shader_flags,
    cycles_light_group) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::LightDistributionGpu,
    cumulative,
    selection_pdf,
    kind,
    index,
    emitter_id,
    padding_0,
    padding_1,
    padding_2) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::LightTreeNodeGpu,
    bounds_min_energy,
    bounds_max_theta_o,
    cone_axis_theta_e,
    topology,
    emitters) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::LightTreeEmitterGpu,
    bounds_min_energy,
    bounds_max_theta_o,
    cone_axis_theta_e,
    identity) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::ShaderEvaluationStateCall,
    ray_visibility,
    ray_events,
    ray_depth,
    diffuse_depth,
    glossy_depth,
    transparent_depth,
    transmission_depth) {};
LUISA_STRUCT(psycles::luisa_backend::detail::ShadowShaderContextCall,
             path, ray_time, sample_index, rng_hash, rng_offset,
             volume_bounds_bounce){};
LUISA_STRUCT(
    psycles::luisa_backend::detail::ShadowIntersectionCall,
    instance,
    primitive,
    hit_type,
    distance,
    barycentric) {};
LUISA_STRUCT(psycles::luisa_backend::detail::ShadowIntersectionSummaryCall,
             count, total, blocked){};
LUISA_STRUCT(psycles::luisa_backend::detail::ShadowIntersectionBatchCall, hits,
             count, total, blocked){};
LUISA_STRUCT(psycles::luisa_backend::detail::ShadowSurfaceEvaluationCall,
             transmittance, object, primitive, kind, volume_boundary){};
LUISA_STRUCT(
    psycles::luisa_backend::detail::ShadowTraceResultCall,
    transmittance,
    throughput,
    first_hit,
    first_object,
    first_primitive,
    first_kind,
    first_distance,
    first_barycentric) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfacePointCall,
    position,
    object_position,
    object_location,
    generated,
    geometric_normal,
    shading_normal,
    object_shading_normal,
    object_tangent,
    tangent_sign,
    undisplaced_position,
    undisplaced_object_position,
    undisplaced_shading_normal,
    undisplaced_object_shading_normal,
    undisplaced_object_tangent,
    undisplaced_tangent_sign,
    normal_to_world_x,
    normal_to_world_y,
    normal_to_world_z,
    dpdu,
    dpdv,
    dPdx,
    dPdy,
    object_dPdx,
    object_dPdy,
    undisplaced_dPdx,
    undisplaced_dPdy,
    undisplaced_object_dPdx,
    undisplaced_object_dPdy,
    generated_dx,
    generated_dy,
    incoming,
    uv,
    uv_dx,
    uv_dy,
    geometry_index,
    barycentric,
    barycentric_dx,
    barycentric_dy,
    instance_id,
    primitive_id,
    parameter_block,
    object_random,
    particle_index,
    random_per_island,
    triangle_smooth,
    is_curve,
    curve_intercept,
    curve_length,
    curve_thickness,
    curve_tangent_normal,
    curve_random,
    ray_visibility,
    ray_events,
    ray_depth,
    diffuse_depth,
    glossy_depth,
    transparent_depth,
    transmission_depth,
    ray_length,
    time,
    use_bump_map_correction,
    back_facing) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceEvaluationCall,
    f,
    pdf,
    diffuse_f,
    glossy_f,
    diffuse_pdf,
    average_roughness_squared,
    events) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceSampleCall,
    f,
    pdf,
    diffuse_f,
    glossy_f,
    diffuse_pdf,
    average_roughness_squared,
    events,
    wi,
    eta,
    roughness,
    runtime_flags,
    bssrdf_method,
    bssrdf_radius,
    bssrdf_albedo,
    bssrdf_normal,
    bssrdf_ior,
    bssrdf_roughness,
    bssrdf_anisotropy,
    valid) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceClosureTraceCall,
    count,
    runtime_flags,
    index,
    type,
    sample_weight,
    weight,
    normal,
    valid) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceSampleTraceCall,
    f,
    pdf,
    diffuse_f,
    glossy_f,
    diffuse_pdf,
    average_roughness_squared,
    events,
    wi,
    eta,
    roughness,
    runtime_flags,
    bssrdf_method,
    bssrdf_radius,
    bssrdf_albedo,
    bssrdf_normal,
    bssrdf_ior,
    bssrdf_roughness,
    bssrdf_anisotropy,
    valid,
    closure_index,
    closure_type,
    closure_sample_weight,
    selection_rescaled,
    closure_weight,
    closure_normal,
    closure_valid) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfacePreparationQueryCall,
    outgoing,
    glossy_filter_roughness,
    flags) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfacePreparationCall,
    emission,
    shading_normal,
    albedo,
    glossy_albedo,
    transmission_albedo,
    normal,
    transparency,
    roughness,
    runtime_flags) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::LightPassContributionCall,
    diffuse_direct,
    diffuse_indirect,
    glossy_direct,
    glossy_indirect,
    transmission_direct,
    transmission_indirect) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::RenderKernelParameters,
    window_x,
    window_y,
    window_width,
    full_width,
    full_height,
    seed,
    sobol_sequence_size,
    max_bounces,
    min_bounces,
    max_diffuse_bounces,
    max_glossy_bounces,
    max_transmission_bounces,
    max_volume_bounces,
    transparent_min_bounces,
    transparent_max_bounces,
    max_path_steps,
    transparent_background,
    path_trace_enabled,
    path_trace_pixel_x,
    path_trace_pixel_y,
    path_trace_sample,
    analytic_light_count,
    portal_count,
    ambient_occlusion_object_distance_count,
    wavefront_frame_capacity,
    shadow_storage_capacity,
    shadow_storage_block_size,
    ambient_occlusion_bounces,
    ambient_occlusion_factor,
    ambient_occlusion_distance,
    sample_clamp_direct,
    sample_clamp_indirect,
    filter_glossy,
    light_inv_rr_threshold,
    camera_horizontal_tangent,
    camera_vertical_tangent,
    camera_ortho_vertical_span,
    camera_shift_x,
    camera_shift_y,
    camera_near,
    camera_far,
    camera_aperture_radius,
    camera_focal_distance,
    camera_aperture_ratio,
    pass_alpha_threshold,
    background,
    camera_transform,
    camera_inverse_transform) {};

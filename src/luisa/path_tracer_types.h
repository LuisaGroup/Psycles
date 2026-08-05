#pragma once

#include <cstdint>

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
    std::uint64_t id{};
    luisa::uint value_slot{};
    luisa::uint domain{};
};

struct AttributeRangeGpu {
    luisa::uint offset{};
    luisa::uint count{};
    luisa::uint triangle_slot{};
    luisa::uint padding{};
};

struct PrimitiveCompletionGpu {
    luisa::uint local_primitive{};
    luisa::uint instance_offset{};
    luisa::uint instance_count{};
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
    std::int32_t cycles_light_group{};
    luisa::uint is_shadow_catcher{};
    // Circular exact-support equivalence class. Singleton hand-authored test
    // fixtures may leave count at zero; traversal normalizes that to one.
    luisa::uint coincident_next{};
    luisa::uint coincident_count{1u};
    // Sorted sparse records for corresponding local primitives whose closed
    // finite world bounds overlap across whole-instance classes.
    luisa::uint primitive_completion_offset{};
    luisa::uint primitive_completion_count{};
    luisa::uint cycles_transform_applied{};
    luisa::uint intersection_padding{};
    luisa::float4x4 cycles_world_to_object{};
};

inline constexpr std::uint32_t material_flag_has_volume =
    1u << 0u;
inline constexpr std::uint32_t material_flag_may_emit =
    1u << 1u;
inline constexpr std::uint32_t material_flag_constant_emission =
    1u << 2u;
inline constexpr std::uint32_t material_flag_use_bump_map_correction =
    1u << 3u;

struct MaterialBindingGpu {
    luisa::uint surface_tag{};
    luisa::uint parameter_block{};
    luisa::uint cycles_shader_index{};
    luisa::uint material_identity{};
    luisa::uint flags{};
    luisa::uint volume_sampling{};
};

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

struct ShaderEvaluationStateCall {
    luisa::uint ray_visibility{};
    luisa::uint ray_events{};
    luisa::uint ray_depth{};
    luisa::uint diffuse_depth{};
    luisa::uint glossy_depth{};
    luisa::uint transparent_depth{};
    luisa::uint transmission_depth{};
};

struct ShadowSurfaceEvaluationCall {
    luisa::float3 transmittance{};
    luisa::uint object{};
    luisa::uint primitive{};
    luisa::uint kind{};
};

struct ShadowTraceResultCall {
    luisa::float3 transmittance{};
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
    luisa::uint events{};
};

struct SurfaceSampleCall {
    luisa::float3 f{};
    float pdf{};
    luisa::float3 diffuse_f{};
    luisa::float3 glossy_f{};
    float diffuse_pdf{};
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

struct SurfaceAovCall {
    luisa::float3 albedo{};
    luisa::float3 glossy_albedo{};
    luisa::float3 transmission_albedo{};
    luisa::float2 roughness{};
    luisa::float3 normal{};
    luisa::float3 transparency{};
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
    triangle_slot,
    padding) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::PrimitiveCompletionGpu,
    local_primitive,
    instance_offset,
    instance_count,
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
    cycles_light_group,
    is_shadow_catcher,
    coincident_next,
    coincident_count,
    primitive_completion_offset,
    primitive_completion_count,
    cycles_transform_applied,
    intersection_padding,
    cycles_world_to_object) {};
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
    visibility_mask) {};
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
    psycles::luisa_backend::detail::ShaderEvaluationStateCall,
    ray_visibility,
    ray_events,
    ray_depth,
    diffuse_depth,
    glossy_depth,
    transparent_depth,
    transmission_depth) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::ShadowSurfaceEvaluationCall,
    transmittance,
    object,
    primitive,
    kind) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::ShadowTraceResultCall,
    transmittance,
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
    events) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceSampleCall,
    f,
    pdf,
    diffuse_f,
    glossy_f,
    diffuse_pdf,
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
    psycles::luisa_backend::detail::SurfaceAovCall,
    albedo,
    glossy_albedo,
    transmission_albedo,
    roughness,
    normal,
    transparency) {};
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
    camera_transform) {};

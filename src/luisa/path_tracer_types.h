#pragma once

#include <cstdint>

#include <luisa/core/basic_types.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

struct GeometryGpu {
    luisa::uint bindless_base{};
    luisa::uint material_offset{};
    luisa::uint material_count{};
    luisa::uint padding{};
};

struct AttributeBindingGpu {
    std::uint64_t id{};
    luisa::uint value_slot{};
    luisa::uint padding{};
};

struct AttributeRangeGpu {
    luisa::uint offset{};
    luisa::uint count{};
    luisa::uint triangle_slot{};
    luisa::uint padding{};
};

struct InstanceGpu {
    luisa::uint geometry_index{};
    luisa::uint override_offset{};
    luisa::uint override_count{};
    float object_random{};
    luisa::uint particle_index{};
    float shadow_terminator_geometry_offset{};
    luisa::uint cycles_object_index{};
    std::int32_t cycles_light_group{};
};

struct MaterialBindingGpu {
    luisa::uint surface_tag{};
    luisa::uint parameter_block{};
    luisa::uint cycles_shader_index{};
    luisa::uint padding{};
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
    luisa::float3 normal_to_world_x{};
    luisa::float3 normal_to_world_y{};
    luisa::float3 normal_to_world_z{};
    luisa::float3 dpdu{};
    luisa::float3 dpdv{};
    luisa::float3 dPdx{};
    luisa::float3 dPdy{};
    luisa::float3 object_dPdx{};
    luisa::float3 object_dPdy{};
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
    luisa::uint ray_visibility{};
    luisa::uint ray_events{};
    luisa::uint ray_depth{};
    luisa::uint diffuse_depth{};
    luisa::uint glossy_depth{};
    luisa::uint transparent_depth{};
    luisa::uint transmission_depth{};
    float ray_length{};
    float time{};
    luisa::uint back_facing{};
};

struct SurfaceEvaluationCall {
    luisa::float3 f{};
    float pdf{};
    luisa::float3 diffuse_f{};
    float diffuse_pdf{};
    luisa::uint events{};
};

struct SurfaceSampleCall {
    luisa::float3 f{};
    float pdf{};
    luisa::float3 diffuse_f{};
    float diffuse_pdf{};
    luisa::uint events{};
    luisa::float3 wi{};
    float eta{};
    luisa::float2 roughness{};
    luisa::uint runtime_flags{};
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
    float diffuse_pdf{};
    luisa::uint events{};
    luisa::float3 wi{};
    float eta{};
    luisa::float2 roughness{};
    luisa::uint runtime_flags{};
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
    float camera_ortho_scale{};
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
    padding) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::AttributeBindingGpu,
    id,
    value_slot,
    padding) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::AttributeRangeGpu,
    offset,
    count,
    triangle_slot,
    padding) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::InstanceGpu,
    geometry_index,
    override_offset,
    override_count,
    object_random,
    particle_index,
    shadow_terminator_geometry_offset,
    cycles_object_index,
    cycles_light_group) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::MaterialBindingGpu,
    surface_tag,
    parameter_block,
    cycles_shader_index,
    padding) {};
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
    cycles_type,
    visibility_mask) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::EmissiveTriangleGpu,
    instance_index,
    geometry_index,
    primitive_index,
    surface_tag,
    parameter_block,
    emission_sampling) {};
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
    normal_to_world_x,
    normal_to_world_y,
    normal_to_world_z,
    dpdu,
    dpdv,
    dPdx,
    dPdy,
    object_dPdx,
    object_dPdy,
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
    ray_visibility,
    ray_events,
    ray_depth,
    diffuse_depth,
    glossy_depth,
    transparent_depth,
    transmission_depth,
    ray_length,
    time,
    back_facing) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceEvaluationCall,
    f,
    pdf,
    diffuse_f,
    diffuse_pdf,
    events) {};
LUISA_STRUCT(
    psycles::luisa_backend::detail::SurfaceSampleCall,
    f,
    pdf,
    diffuse_f,
    diffuse_pdf,
    events,
    wi,
    eta,
    roughness,
    runtime_flags,
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
    diffuse_pdf,
    events,
    wi,
    eta,
    roughness,
    runtime_flags,
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
    camera_ortho_scale,
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

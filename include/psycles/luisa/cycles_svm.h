/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_svm.h> through the Psycles::luisa target."
#endif

#include <psycles/compiler/cycles_svm_node_types.h>

#include <array>
#include <cstdint>

#include <luisa/luisa-compute.h>

LUISA_STRUCT(
    psycles::compiler::cycles_svm::packed_float3, x, y, z) {};
LUISA_STRUCT(
    psycles::compiler::cycles_svm::packed_normal, value) {};
LUISA_STRUCT(
    psycles::compiler::cycles_svm::uchar4, x, y, z, w) {};
LUISA_STRUCT(
    psycles::compiler::cycles_svm::AttributeMap,
    id,
    offset,
    element,
    type,
    pad) {};
LUISA_STRUCT(
    psycles::compiler::cycles_svm::KernelCurve,
    shader_id,
    first_key,
    num_keys,
    type) {};

namespace psycles::luisa_backend::cycles_svm {

/* Kernel feature flags copied from Cycles 5.2.1 kernel/features.h. They are
 * host/JIT specialization inputs, exactly like svm_eval_nodes'
 * node_feature_mask template parameter in Cycles. */
inline constexpr std::uint32_t kernel_feature_node_bsdf = 1u << 0u;
inline constexpr std::uint32_t kernel_feature_node_emission = 1u << 1u;
inline constexpr std::uint32_t kernel_feature_node_volume = 1u << 2u;
inline constexpr std::uint32_t kernel_feature_node_bump = 1u << 3u;
inline constexpr std::uint32_t kernel_feature_node_bump_state = 1u << 4u;
inline constexpr std::uint32_t kernel_feature_node_voronoi_extra = 1u << 5u;
inline constexpr std::uint32_t kernel_feature_node_raytrace = 1u << 6u;
inline constexpr std::uint32_t kernel_feature_node_aov = 1u << 7u;
inline constexpr std::uint32_t kernel_feature_node_light_path = 1u << 8u;
inline constexpr std::uint32_t kernel_feature_node_principled_hair = 1u << 9u;
inline constexpr std::uint32_t kernel_feature_node_portal = 1u << 10u;
inline constexpr std::uint32_t kernel_feature_path_tracing = 1u << 11u;
inline constexpr std::uint32_t kernel_feature_pointcloud = 1u << 12u;
inline constexpr std::uint32_t kernel_feature_hair_ribbon = 1u << 13u;
inline constexpr std::uint32_t kernel_feature_hair_thick = 1u << 14u;
inline constexpr std::uint32_t kernel_feature_hair =
    kernel_feature_hair_ribbon | kernel_feature_hair_thick;
inline constexpr std::uint32_t kernel_feature_object_motion = 1u << 15u;

inline constexpr std::uint32_t kernel_feature_node_mask_surface_light =
    kernel_feature_node_emission | kernel_feature_node_voronoi_extra |
    kernel_feature_node_light_path | kernel_feature_node_portal;
inline constexpr std::uint32_t kernel_feature_node_mask_surface_shadow =
    kernel_feature_node_bsdf | kernel_feature_node_emission |
    kernel_feature_node_bump | kernel_feature_node_bump_state |
    kernel_feature_node_voronoi_extra | kernel_feature_node_light_path |
    kernel_feature_node_principled_hair | kernel_feature_node_portal;
inline constexpr std::uint32_t kernel_feature_node_mask_surface =
    kernel_feature_node_mask_surface_shadow | kernel_feature_node_raytrace |
    kernel_feature_node_aov | kernel_feature_node_light_path;

/* Path visibility and path flags copied from Cycles 5.2.1 kernel/types.h. */
inline constexpr std::uint32_t path_ray_visibility_camera = 1u << 0u;
inline constexpr std::uint32_t path_ray_visibility_transmit = 1u << 1u;
inline constexpr std::uint32_t path_ray_visibility_diffuse = 1u << 2u;
inline constexpr std::uint32_t path_ray_visibility_glossy = 1u << 3u;
inline constexpr std::uint32_t path_ray_visibility_volume_scatter = 1u << 4u;
inline constexpr std::uint32_t path_ray_visibility_shadow_opaque = 1u << 5u;
inline constexpr std::uint32_t path_ray_visibility_shadow_transparent =
    1u << 6u;
inline constexpr std::uint32_t path_ray_visibility_shadow =
    path_ray_visibility_shadow_opaque |
    path_ray_visibility_shadow_transparent;

inline constexpr std::uint32_t path_ray_reflect = 1u << 0u;
inline constexpr std::uint32_t path_ray_singular = 1u << 1u;
inline constexpr std::uint32_t path_ray_emission = 1u << 5u;

/* Runtime ShaderData flags copied from Cycles 5.2.1 kernel/types.h. */
inline constexpr std::uint32_t shader_data_backfacing = 1u << 0u;
inline constexpr std::uint32_t shader_data_emission = 1u << 1u;
inline constexpr std::uint32_t shader_data_is_volume_shader_eval = 1u << 8u;
inline constexpr std::uint32_t shader_data_volume_cubic =
    static_cast<std::uint32_t>(compiler::cycles_svm::SD_VOLUME_CUBIC);

/* Object sentinel and object flag copied from Cycles 5.2.1 kernel/types.h. */
inline constexpr std::uint32_t object_none = ~0u;
inline constexpr std::uint32_t primitive_none = ~0u;
inline constexpr std::uint32_t primitive_triangle = 1u << 0u;
inline constexpr std::uint32_t primitive_curve_thick = 1u << 1u;
inline constexpr std::uint32_t primitive_curve_ribbon = 1u << 2u;
inline constexpr std::uint32_t primitive_curve =
    primitive_curve_thick | primitive_curve_ribbon;
inline constexpr std::uint32_t primitive_point = 1u << 3u;
inline constexpr std::uint32_t primitive_num_bits =
    static_cast<std::uint32_t>(compiler::cycles_svm::PRIMITIVE_NUM_BITS);
inline constexpr std::uint32_t primitive_volume = 1u << 4u;
inline constexpr std::uint32_t primitive_lamp = 1u << 5u;
inline constexpr std::uint32_t primitive_motion = 1u << 6u;
inline constexpr std::uint32_t shader_data_object_motion = 1u << 1u;
inline constexpr std::uint32_t shader_data_object_transform_applied = 1u << 2u;
inline constexpr std::uint32_t shader_data_object_has_corner_normals =
    1u << 12u;

/* CameraType values copied from Cycles 5.2.1 kernel/types.h. */
inline constexpr std::uint32_t camera_perspective = 0u;
inline constexpr std::uint32_t camera_orthographic = 1u;
inline constexpr std::uint32_t camera_panorama = 2u;
inline constexpr std::uint32_t camera_custom = 3u;

/* Shader id decoration bits copied from Cycles 5.2.1 kernel/types.h. */
inline constexpr std::uint32_t shader_smooth_normal = 1u << 31u;
inline constexpr std::uint32_t shader_cast_shadow = 1u << 30u;
inline constexpr std::uint32_t shader_use_mis = 1u << 28u;
inline constexpr std::uint32_t shader_exclude_diffuse = 1u << 27u;
inline constexpr std::uint32_t shader_exclude_glossy = 1u << 26u;
inline constexpr std::uint32_t shader_exclude_transmit = 1u << 25u;
inline constexpr std::uint32_t shader_exclude_camera = 1u << 24u;
inline constexpr std::uint32_t shader_exclude_scatter = 1u << 23u;
inline constexpr std::uint32_t shader_exclude_shadow_catcher = 1u << 22u;
inline constexpr std::uint32_t shader_exclude_any =
    shader_exclude_diffuse | shader_exclude_glossy |
    shader_exclude_transmit | shader_exclude_camera |
    shader_exclude_scatter | shader_exclude_shadow_catcher;
inline constexpr std::uint32_t shader_mask =
    ~(shader_smooth_normal | shader_cast_shadow | shader_use_mis |
      shader_exclude_any);

enum class EvaluationStatus : std::uint32_t {
  running = 0u,
  ended = 1u,
  unsupported_node = 2u,
  invalid_node = 3u,
};

/* Exact transform values consumed by Cycles' vector-transform handler. The
 * camera fields project kernel_data.cam; the static object fields project
 * object_fetch_transform() for ShaderData::object. Motion transforms remain
 * on ShaderData, as in Cycles. */
struct TransformState {
  luisa::compute::Float4x4 camera_to_world;
  luisa::compute::Float4x4 world_to_camera;
  luisa::compute::Float4x4 object_to_world;
  luisa::compute::Float4x4 world_to_object;

  TransformState(
      luisa::compute::Expr<luisa::float4x4> camera_to_world_transform,
      luisa::compute::Expr<luisa::float4x4> world_to_camera_transform,
      luisa::compute::Expr<luisa::float4x4> object_to_world_transform,
      luisa::compute::Expr<luisa::float4x4> world_to_object_transform) noexcept;
};

struct TriangleVertices {
  luisa::compute::Float3 v0;
  luisa::compute::Float3 v1;
  luisa::compute::Float3 v2;
};

struct TriangleNormals {
  luisa::compute::Float3 n0;
  luisa::compute::Float3 n1;
  luisa::compute::Float3 n2;
};

/* Direct Luisa projections of Cycles' differential and dual3 value types. */
struct Differential {
  luisa::compute::Float dx;
  luisa::compute::Float dy;
};

struct Dual1 {
  luisa::compute::Float val;
  luisa::compute::Float dx;
  luisa::compute::Float dy;
};

struct Dual2 {
  luisa::compute::Float2 val;
  luisa::compute::Float2 dx;
  luisa::compute::Float2 dy;
};

struct Dual3 {
  luisa::compute::Float3 val;
  luisa::compute::Float3 dx;
  luisa::compute::Float3 dy;
};

struct Dual4 {
  luisa::compute::Float4 val;
  luisa::compute::Float4 dx;
  luisa::compute::Float4 dy;
};

struct AttributeDescriptor {
  luisa::compute::UInt element;
  luisa::compute::UInt type;
  luisa::compute::Int offset;
};

struct ShaderData;

/* Host/JIT projection of the exact KernelGlobals services consumed by the
 * copied SVM handlers. Virtual dispatch happens while Luisa records the AST;
 * generated device code contains only the resulting buffer operations. */
class KernelGlobals {
public:
  virtual ~KernelGlobals() noexcept = default;

  [[nodiscard]] virtual TriangleVertices triangle_vertices(
      luisa::compute::Expr<std::uint32_t> object,
      luisa::compute::Expr<std::uint32_t> prim) const noexcept = 0;
  [[nodiscard]] virtual TriangleVertices
  motion_triangle_vertices(luisa::compute::Expr<std::uint32_t> object,
                           luisa::compute::Expr<std::uint32_t> prim,
                           luisa::compute::Expr<float> time) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  film_rgb_to_y() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  primitive_tangent(const ShaderData &shader_data) const noexcept = 0;
  [[nodiscard]] virtual Dual3 primitive_tangent_derivative(
      const ShaderData &shader_data) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::UInt object_attribute_map_offset(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Var<compiler::cycles_svm::AttributeMap>
  attribute_map(luisa::compute::Expr<std::uint32_t> offset) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float
  attribute_float(luisa::compute::Expr<std::int32_t> offset) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float2 attribute_float2(
      luisa::compute::Expr<std::int32_t> offset) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Var<compiler::cycles_svm::packed_float3>
  attribute_float3(
      luisa::compute::Expr<std::int32_t> offset) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float4 attribute_float4(
      luisa::compute::Expr<std::int32_t> offset) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Var<compiler::cycles_svm::uchar4>
  attribute_uchar4(
      luisa::compute::Expr<std::int32_t> offset) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Var<compiler::cycles_svm::packed_normal>
  attribute_normal(
      luisa::compute::Expr<std::int32_t> offset) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::UInt3 triangle_vertex_indices(
      luisa::compute::Expr<std::uint32_t> prim) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Int object_normal_offset(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::UInt object_num_geom_steps(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Int object_num_vertices(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Int object_num_primitives(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3 object_dupli_generated(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3 object_dupli_uv(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::UInt camera_type() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float camera_width() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float camera_height() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3 camera_world_to_ndc(
      const ShaderData &shader_data,
      luisa::compute::Expr<luisa::float3> position) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Var<compiler::cycles_svm::KernelCurve>
  curve(luisa::compute::Expr<std::uint32_t> prim) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Bool
  film_is_rec709() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  film_rec709_to_r() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  film_rec709_to_g() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  film_rec709_to_b() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  object_inverse_position_transform_if_object(
      const ShaderData &shader_data,
      luisa::compute::Expr<luisa::float3> value) const noexcept = 0;
  [[nodiscard]] virtual Dual3
  object_inverse_position_transform_if_object_derivative(
      const ShaderData &shader_data, const Dual3 &value) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  object_inverse_position_transform(
      const ShaderData &shader_data,
      luisa::compute::Expr<luisa::float3> value) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float4 kernel_image_interp_3d(
      ShaderData &shader_data,
      luisa::compute::Expr<std::int32_t> image_texture_id,
      luisa::compute::Expr<luisa::float3> position,
      luisa::compute::Expr<std::int32_t> interpolation,
      luisa::compute::Expr<bool> stochastic) const noexcept = 0;
};

/* The fields below are the exact ShaderData projection consumed by the first
 * copied SVM node families. More fields are added only when their Cycles
 * handler is copied; no alternative shader state is consulted. */
struct ShaderData {
  luisa::compute::Float3 P;
  luisa::compute::Float3 ray_P;
  luisa::compute::Float3 N;
  luisa::compute::Float3 Ng;
  luisa::compute::Float3 wi;
  luisa::compute::UInt type;
  luisa::compute::UInt shader;
  luisa::compute::UInt flag;
  luisa::compute::UInt object_flag;
  luisa::compute::UInt prim;
  luisa::compute::Float u;
  luisa::compute::Float v;
  luisa::compute::UInt object;
  luisa::compute::Float time;
  luisa::compute::Float ray_length;
  luisa::compute::Float dP;
  luisa::compute::Float dI;
  Differential du;
  Differential dv;
  luisa::compute::Float3 dPdu;
  luisa::compute::Float3 dPdv;
  luisa::compute::Float4x4 ob_tfm_motion;
  luisa::compute::Float4x4 ob_itfm_motion;
  luisa::compute::UInt lcg_state;
  luisa::compute::Float3 closure_emission_background;
  luisa::compute::Float3 closure_transparent_extinction;

  ShaderData(
      luisa::compute::Expr<luisa::float3> position,
      luisa::compute::Expr<luisa::float3> normal,
      luisa::compute::Expr<luisa::float3> geometric_normal,
      luisa::compute::Expr<luisa::float3> incoming,
      luisa::compute::Expr<std::uint32_t> primitive_type,
      luisa::compute::Expr<std::uint32_t> shader_id,
      luisa::compute::Expr<std::uint32_t> shader_flags,
      luisa::compute::Expr<std::uint32_t> object_flags,
      luisa::compute::Expr<std::uint32_t> primitive_id,
      luisa::compute::Expr<float> parametric_u,
      luisa::compute::Expr<float> parametric_v,
      luisa::compute::Expr<std::uint32_t> object_id,
      luisa::compute::Expr<float> motion_time,
      luisa::compute::Expr<float> length,
      luisa::compute::Expr<float> position_differential,
      luisa::compute::Expr<float> incoming_differential,
      luisa::compute::Expr<float> parametric_u_dx,
      luisa::compute::Expr<float> parametric_u_dy,
      luisa::compute::Expr<float> parametric_v_dx,
      luisa::compute::Expr<float> parametric_v_dy,
      luisa::compute::Expr<luisa::float3> position_u_derivative,
      luisa::compute::Expr<luisa::float3> position_v_derivative,
      luisa::compute::Expr<luisa::float4x4> motion_object_to_world,
      luisa::compute::Expr<luisa::float4x4> motion_world_to_object,
      luisa::compute::Expr<std::uint32_t> random_state = 0u) noexcept;
};

[[nodiscard]] luisa::compute::Float3 decode_packed_normal(
    luisa::compute::Var<compiler::cycles_svm::packed_normal> packed) noexcept;

[[nodiscard]] AttributeDescriptor
find_attribute(const KernelGlobals &kernel_globals,
               const ShaderData &shader_data,
               luisa::compute::Expr<luisa::ulong> id) noexcept;

[[nodiscard]] luisa::compute::Float primitive_surface_attribute_float(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept;
[[nodiscard]] Dual1 primitive_surface_attribute_float_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept;
[[nodiscard]] luisa::compute::Float2 primitive_surface_attribute_float2(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept;
[[nodiscard]] Dual2 primitive_surface_attribute_float2_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept;
[[nodiscard]] luisa::compute::Float3 primitive_surface_attribute_float3(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept;
[[nodiscard]] Dual3 primitive_surface_attribute_float3_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept;
[[nodiscard]] luisa::compute::Float4 primitive_surface_attribute_float4(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept;
[[nodiscard]] Dual4 primitive_surface_attribute_float4_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeDescriptor &descriptor) noexcept;

[[nodiscard]] luisa::compute::Bool
primitive_is_volume_attribute(const ShaderData &shader_data) noexcept;
[[nodiscard]] luisa::compute::Float4 volume_attribute_float4(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor,
    luisa::compute::Expr<bool> stochastic) noexcept;
[[nodiscard]] luisa::compute::Float primitive_volume_attribute_float(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor,
    luisa::compute::Expr<bool> stochastic) noexcept;
[[nodiscard]] luisa::compute::Float2 primitive_volume_attribute_float2(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor,
    luisa::compute::Expr<bool> stochastic) noexcept;
[[nodiscard]] luisa::compute::Float3 primitive_volume_attribute_float3(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor,
    luisa::compute::Expr<bool> stochastic) noexcept;
[[nodiscard]] luisa::compute::Float4 primitive_volume_attribute_float4(
    const KernelGlobals &kernel_globals, ShaderData &shader_data,
    const AttributeDescriptor &descriptor,
    luisa::compute::Expr<bool> stochastic) noexcept;

/* Integrator counters read by Cycles' Light Path node. */
struct PathState {
  luisa::compute::UInt visibility;
  luisa::compute::UInt flag;
  luisa::compute::UInt bounce;
  luisa::compute::UInt transparent_bounce;
  luisa::compute::UInt diffuse_bounce;
  luisa::compute::UInt glossy_bounce;
  luisa::compute::UInt transmission_bounce;
  luisa::compute::UInt portal_bounce;

  PathState(luisa::compute::Expr<std::uint32_t> path_visibility,
            luisa::compute::Expr<std::uint32_t> path_flag,
            luisa::compute::Expr<std::uint32_t> ray_bounce = 0u,
            luisa::compute::Expr<std::uint32_t> ray_transparent = 0u,
            luisa::compute::Expr<std::uint32_t> ray_diffuse = 0u,
            luisa::compute::Expr<std::uint32_t> ray_glossy = 0u,
            luisa::compute::Expr<std::uint32_t> ray_transmission = 0u,
            luisa::compute::Expr<std::uint32_t> ray_portal = 0u) noexcept;
};

struct EvaluationResult {
  luisa::compute::UInt final_offset;
  luisa::compute::UInt status;
  luisa::compute::Float3 closure_weight;

  EvaluationResult() noexcept;
};

/* Luisa DSL realization of Cycles 5.2.1 svm_eval_nodes. `node_feature_mask`
 * and `node_types_used` are host/JIT constants, corresponding respectively
 * to Cycles' template feature mask and kernel_data_svm_usage_NODE_* constants.
 * The device machine still has one word PC loop and one primary opcode switch. */
void eval_nodes(
    const KernelGlobals &kernel_globals,
    const luisa::compute::BufferUInt &words,
    compiler::cycles_svm::ShaderType shader_type,
    std::uint32_t kernel_features,
    std::uint32_t node_feature_mask,
    const std::array<bool, compiler::cycles_svm::NODE_NUM> &node_types_used,
    const TransformState &transform_state,
    ShaderData &shader_data,
    const PathState &path_state,
    EvaluationResult &result) noexcept;

} // namespace psycles::luisa_backend::cycles_svm

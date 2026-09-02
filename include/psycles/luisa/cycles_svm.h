/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_svm.h> through the Psycles::luisa target."
#endif

#include <psycles/compiler/cycles_svm_node_types.h>
#include <psycles/luisa/cycles_bsdf_tables.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <luisa/dsl/local.h>
#include <luisa/luisa-compute.h>

LUISA_STRUCT(psycles::compiler::cycles_svm::packed_float3, x, y, z){};
LUISA_STRUCT(psycles::compiler::cycles_svm::packed_float2, x, y){};
LUISA_STRUCT(psycles::compiler::cycles_svm::packed_float4, x, y, z, w){};
LUISA_STRUCT(psycles::compiler::cycles_svm::packed_uint3, x, y, z){};
LUISA_STRUCT(psycles::compiler::cycles_svm::packed_normal, value){};
LUISA_STRUCT(psycles::compiler::cycles_svm::uchar4, x, y, z, w){};
LUISA_STRUCT(psycles::compiler::cycles_svm::PackedTransform, x, y, z){};
LUISA_STRUCT(psycles::compiler::cycles_svm::KernelObject, tfm, itfm,
             volume_density, pass_id, random_number, color, alpha,
             particle_index, dupli_generated, dupli_uv, num_geom_steps,
             num_tfm_steps, numverts, numprims, attribute_map_offset,
             motion_offset, position_offset, normal_offset, cryptomatte_object,
             cryptomatte_asset, shadow_terminator_shading_offset,
             shadow_terminator_geometry_offset, ao_distance, lightgroup,
             visibility, primitive_type, velocity_scale,
             _pad_light_set_alignment, light_set_membership,
             receiver_light_set, _pad_shadow_set_alignment,
             shadow_set_membership, blocker_shadow_set, _pad_tail){};
LUISA_STRUCT(psycles::compiler::cycles_svm::AttributeMap, id, offset, element,
             type, pad){};
LUISA_STRUCT(psycles::compiler::cycles_svm::KernelCurve, shader_id, first_key,
             num_keys, type){};

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
inline constexpr std::uint32_t path_ray_visibility_shadow_transparent = 1u
                                                                        << 6u;
inline constexpr std::uint32_t path_ray_visibility_shadow =
    path_ray_visibility_shadow_opaque | path_ray_visibility_shadow_transparent;

inline constexpr std::uint32_t path_ray_reflect = 1u << 0u;
inline constexpr std::uint32_t path_ray_singular = 1u << 1u;
inline constexpr std::uint32_t path_ray_diffuse_ancestor = 1u << 4u;
inline constexpr std::uint32_t path_ray_emission = 1u << 5u;
inline constexpr std::uint32_t path_ray_terminate_on_next_surface = 1u << 10u;
inline constexpr std::uint32_t path_ray_terminate_in_next_volume = 1u << 11u;
inline constexpr std::uint32_t path_ray_terminate_after_transparent = 1u << 12u;
inline constexpr std::uint32_t path_ray_terminate_after_volume = 1u << 13u;
inline constexpr std::uint32_t path_ray_terminate =
    path_ray_terminate_on_next_surface | path_ray_terminate_in_next_volume |
    path_ray_terminate_after_transparent | path_ray_terminate_after_volume;

/* Runtime ShaderData flags copied from Cycles 5.2.1 kernel/types.h. */
inline constexpr std::uint32_t shader_data_backfacing = 1u << 0u;
inline constexpr std::uint32_t shader_data_emission = 1u << 1u;
inline constexpr std::uint32_t shader_data_bsdf = 1u << 2u;
inline constexpr std::uint32_t shader_data_bsdf_has_eval = 1u << 3u;
inline constexpr std::uint32_t shader_data_bssrdf = 1u << 4u;
inline constexpr std::uint32_t shader_data_is_volume_shader_eval = 1u << 8u;
inline constexpr std::uint32_t shader_data_transparent = 1u << 9u;
inline constexpr std::uint32_t shader_data_bsdf_has_transmission = 1u << 10u;
inline constexpr std::uint32_t shader_data_ray_portal = 1u << 11u;
inline constexpr std::uint32_t shader_data_volume_cubic =
    static_cast<std::uint32_t>(compiler::cycles_svm::SD_VOLUME_CUBIC);
inline constexpr std::uint32_t shader_data_use_bump_map_correction = 1u << 15u;

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
inline constexpr std::uint32_t shader_data_object_has_corner_normals = 1u
                                                                       << 12u;

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
    shader_exclude_diffuse | shader_exclude_glossy | shader_exclude_transmit |
    shader_exclude_camera | shader_exclude_scatter |
    shader_exclude_shadow_catcher;
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

/* Exact geometry/object-table services consumed by Cycles' Info nodes. The
 * interface is resolved while Luisa records the AST. A KernelGlobals provider
 * that does not expose this interface makes the corresponding transition
 * explicitly unsupported; it never substitutes legacy SurfacePoint fields. */
class InfoServices {
public:
  virtual ~InfoServices() noexcept = default;

  [[nodiscard]] virtual luisa::compute::Float3
  object_location(const ShaderData &shader_data) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  object_color(luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float
  object_alpha(luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float
  object_pass_id(luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float
  shader_pass_id(const ShaderData &shader_data) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float object_random_number(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;

  [[nodiscard]] virtual luisa::compute::Int object_particle_id(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::UInt particle_index(
      luisa::compute::Expr<std::int32_t> particle) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float
  particle_age(luisa::compute::Expr<std::int32_t> particle) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float particle_lifetime(
      luisa::compute::Expr<std::int32_t> particle) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float
  particle_size(luisa::compute::Expr<std::int32_t> particle) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3 particle_location(
      luisa::compute::Expr<std::int32_t> particle) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3 particle_velocity(
      luisa::compute::Expr<std::int32_t> particle) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3 particle_angular_velocity(
      luisa::compute::Expr<std::int32_t> particle) const noexcept = 0;

  [[nodiscard]] virtual luisa::compute::Float
  curve_thickness(const ShaderData &shader_data) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  point_position(const ShaderData &shader_data) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float
  point_radius(const ShaderData &shader_data) const noexcept = 0;
};

/* Host/JIT projection of the exact KernelGlobals services consumed by the
 * copied SVM handlers. Virtual dispatch happens while Luisa records the AST;
 * generated device code contains only the resulting buffer operations. */
class KernelGlobals : public CyclesBsdfTableReader {
public:
  virtual ~KernelGlobals() noexcept = default;

  /* kernel_data.integrator.caustics_* in Cycles. Returning true is the exact
   * no-caustics-tricks configuration; production scene services override
   * these when the corresponding integrator controls are enabled. */
  [[nodiscard]] virtual luisa::compute::Bool
  caustics_reflective() const noexcept {
    return true;
  }
  [[nodiscard]] virtual luisa::compute::Bool
  caustics_refractive() const noexcept {
    return true;
  }

  /* kernel_data.background.transparent_roughness_squared_threshold. The
   * disabled default is Cycles' negative sentinel; production scene services
   * override this when transparent-background roughness filtering is active. */
  [[nodiscard]] virtual luisa::compute::Float
  transparent_roughness_squared_threshold() const noexcept {
    return -1.0f;
  }

  /* kernel_data_fetch(objects, object).shadow_terminator_shading_offset. A
   * value of one is Cycles' no-op domain for the frequency correction. */
  [[nodiscard]] virtual luisa::compute::Float
  object_shadow_terminator_shading_offset(
      luisa::compute::Expr<std::uint32_t>) const noexcept {
    return 1.0f;
  }

  // Packed scene-owned IES table from Cycles LightManager. The default is
  // intentionally inert so node families that do not use NODE_IES do not
  // acquire a device binding while their AST is recorded.
  [[nodiscard]] virtual luisa::compute::Float
  ies(luisa::compute::Expr<std::uint32_t>) const noexcept {
    return luisa::compute::Expr<std::int32_t>{-1}.bitcast<float>();
  }

  [[nodiscard]] virtual const InfoServices *info_services() const noexcept {
    return nullptr;
  }

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
  [[nodiscard]] virtual luisa::compute::Float
  camera_height() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3 camera_world_to_ndc(
      const ShaderData &shader_data,
      luisa::compute::Expr<luisa::float3> position) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Var<compiler::cycles_svm::KernelCurve>
  curve(luisa::compute::Expr<std::uint32_t> prim) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Int object_position_offset(
      luisa::compute::Expr<std::uint32_t> object) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float4
  curve_key(luisa::compute::Expr<std::int32_t> key) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Bool
  film_is_rec709() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  film_xyz_to_r() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  film_xyz_to_g() const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float3
  film_xyz_to_b() const noexcept = 0;
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
  [[nodiscard]] virtual luisa::compute::Float4 kernel_image_interp_with_udim(
      ShaderData &shader_data,
      luisa::compute::Expr<std::int32_t> image_texture_id,
      const Dual2 &uv) const noexcept = 0;
  [[nodiscard]] virtual luisa::compute::Float4 kernel_image_interp_3d(
      ShaderData &shader_data,
      luisa::compute::Expr<std::int32_t> image_texture_id,
      luisa::compute::Expr<luisa::float3> position,
      luisa::compute::Expr<std::int32_t> interpolation,
      luisa::compute::Expr<bool> stochastic) const noexcept = 0;
};

/* Cycles kernel/types.h::MAX_CLOSURE. Scene analysis may instantiate a
 * smaller live pool, but the semantic allocation domain never exceeds this
 * ABI limit. */
inline constexpr std::size_t maximum_closure_capacity = 64u;

/* Common ShaderClosure prefix copied from Cycles 5.2.1 kernel/types.h. These
 * are host bundles of Luisa expressions; they do not invent a second closure
 * model or pre-evaluate a directional response. */
struct ShaderClosureCommon {
  luisa::compute::Float3 weight;
  luisa::compute::UInt type;
  luisa::compute::Float sample_weight;
  luisa::compute::Float3 N;
};

struct OrenNayarParam {
  luisa::compute::Float roughness;
  luisa::compute::Float a;
  luisa::compute::Float b;
  luisa::compute::Float3 multiscatter_term;
};

struct OrenNayarClosure {
  ShaderClosureCommon common;
  OrenNayarParam param;
};

struct SheenParam {
  luisa::compute::Float roughness;
  luisa::compute::Float transform_a;
  luisa::compute::Float transform_b;
  luisa::compute::Float3 T;
  luisa::compute::Float3 B;
};

struct SheenClosure {
  ShaderClosureCommon common;
  SheenParam param;
};

struct VelvetParam {
  luisa::compute::Float sigma;
  luisa::compute::Float invsigma2;
};

struct VelvetClosure {
  ShaderClosureCommon common;
  VelvetParam param;
};

/* Typed projection of Cycles 5.2.1 kernel/closure/bsdf_toon.h::ToonBsdf. */
struct ToonParam {
  luisa::compute::Float size;
  luisa::compute::Float smooth;
};

struct ToonClosure {
  ShaderClosureCommon common;
  ToonParam param;
};

/* Typed projection of Cycles 5.2.1
 * kernel/closure/bsdf_ray_portal.h::RayPortalClosure. */
struct RayPortalParam {
  luisa::compute::Float3 P;
  luisa::compute::Float3 D;
};

struct RayPortalClosure {
  ShaderClosureCommon common;
  RayPortalParam param;
};

/* Typed projection of Cycles 5.2.1 kernel/closure/bsdf_hair.h::HairBsdf. */
struct HairParam {
  luisa::compute::Float3 T;
  luisa::compute::Float roughness1;
  luisa::compute::Float roughness2;
  luisa::compute::Float offset;
};

struct HairClosure {
  ShaderClosureCommon common;
  HairParam param;
};

/* Typed projection of Cycles 5.2.1
 * kernel/closure/bsdf_principled_hair_chiang.h::ChiangHairBSDF. */
struct ChiangHairParam {
  luisa::compute::Float3 sigma;
  luisa::compute::Float v;
  luisa::compute::Float s;
  luisa::compute::Float alpha;
  luisa::compute::Float eta;
  luisa::compute::Float m0_roughness;
  luisa::compute::Float h;
};

struct ChiangHairClosure {
  ShaderClosureCommon common;
  ChiangHairParam param;
};

/* Typed projections of Cycles 5.2.1
 * kernel/closure/bsdf_principled_hair_huang.h. HuangHairExtra consumes one
 * closure-sized tail slot; the SoA pool associates that slot with its owning
 * ordinary closure while preserving the same two-slot allocator transition. */
struct HuangHairParam {
  luisa::compute::Float3 sigma;
  luisa::compute::Float roughness;
  luisa::compute::Float tilt;
  luisa::compute::Float eta;
  luisa::compute::Float aspect_ratio;
  luisa::compute::Float h;
};

struct HuangHairExtra {
  luisa::compute::Float R;
  luisa::compute::Float TT;
  luisa::compute::Float TRT;
  luisa::compute::Float3 Y;
  luisa::compute::Float3 Z;
  luisa::compute::Float3 wi;
  luisa::compute::Float radius;
  luisa::compute::Float e2;
  luisa::compute::Float pixel_coverage;
};

struct HuangHairClosure {
  ShaderClosureCommon common;
  HuangHairParam param;
  HuangHairExtra extra;
};

/* Typed projection of Cycles 5.2.1 kernel/closure/bssrdf.h::Bssrdf. */
struct BssrdfParam {
  luisa::compute::Float3 radius;
  luisa::compute::Float3 albedo;
  luisa::compute::Float anisotropy;
  luisa::compute::Float ior;
  luisa::compute::Float alpha;
};

struct BssrdfClosure {
  ShaderClosureCommon common;
  BssrdfParam param;
};

/* MicrofacetFresnel values copied from Cycles 5.2.1
 * kernel/closure/bsdf_microfacet.h. */
enum class MicrofacetFresnel : std::uint32_t {
  none = 0u,
  dielectric = 1u,
  dielectric_tint = 2u,
  conductor = 3u,
  generalized_schlick = 4u,
  f82_tint = 5u,
};

struct FresnelThinFilm {
  luisa::compute::Float thickness;
  luisa::compute::Float ior;
};

struct FresnelGeneralizedSchlick {
  FresnelThinFilm thin_film;
  luisa::compute::Float3 reflection_tint;
  luisa::compute::Float3 transmission_tint;
  luisa::compute::Float3 f0;
  luisa::compute::Float3 f90;
  luisa::compute::Float exponent;
};

struct FresnelConductor {
  FresnelThinFilm thin_film;
  luisa::compute::Float3 ior;
  luisa::compute::Float3 extinction;
};

struct FresnelF82Tint {
  FresnelThinFilm thin_film;
  luisa::compute::Float3 f0;
  luisa::compute::Float3 b;
};

struct MicrofacetParam {
  luisa::compute::Float alpha_x;
  luisa::compute::Float alpha_y;
  luisa::compute::Float ior;
  luisa::compute::Float energy_scale;
  luisa::compute::UInt fresnel_type;
  luisa::compute::Float3 T;
};

struct MicrofacetClosure {
  ShaderClosureCommon common;
  MicrofacetParam param;
  FresnelGeneralizedSchlick generalized_schlick;
};

struct MicrofacetConductorClosure {
  ShaderClosureCommon common;
  MicrofacetParam param;
  FresnelConductor conductor;
};

struct MicrofacetF82TintClosure {
  ShaderClosureCommon common;
  MicrofacetParam param;
  FresnelF82Tint f82_tint;
};

/* Device-local realization of ShaderData::closure[], num_closure and
 * num_closure_left. Storage is SoA, but its state machine is exactly Cycles'
 * prefix allocator: successful allocations append one common record, extra
 * payloads belong to the record's runtime ClosureType, and only [0, count)
 * is observable. */
class ClosurePool final {
public:
  struct Allocation {
    luisa::compute::UInt index;
    luisa::compute::Bool valid;
  };

private:
  std::size_t _capacity;
  luisa::compute::Local<luisa::float4> _weight_and_sample;
  luisa::compute::Local<luisa::float4> _normal;
  luisa::compute::Local<luisa::uint> _type;
  /* ShaderClosure is a tagged union in Cycles. These SoA rows are the same
   * union, not one allocation per closure family. The current largest live
   * member is MicrofacetBsdf plus its discriminated Fresnel extra payload;
   * Oren-Nayar and Sheen reuse its prefix rows. */
  luisa::compute::Local<luisa::float4> _payload0;
  luisa::compute::Local<luisa::float4> _payload1;
  luisa::compute::Local<luisa::float4> _payload2;
  luisa::compute::Local<luisa::float4> _payload3;
  luisa::compute::Local<luisa::float4> _payload4;
  luisa::compute::Local<luisa::float4> _payload5;
  luisa::compute::Local<luisa::float4> _payload6;
  luisa::compute::Local<luisa::uint> _payload_tag;
  luisa::compute::UInt _count;
  luisa::compute::UInt _left;

public:
  explicit ClosurePool(std::size_t capacity) noexcept;
  ClosurePool(const ClosurePool &) = delete;
  ClosurePool(ClosurePool &&) = delete;
  ClosurePool &operator=(const ClosurePool &) = delete;
  ClosurePool &operator=(ClosurePool &&) = delete;

  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] luisa::compute::UInt count() const noexcept;
  [[nodiscard]] luisa::compute::UInt left() const noexcept;

  [[nodiscard]] Allocation
  allocate(luisa::compute::Expr<std::uint32_t> type,
           luisa::compute::Expr<luisa::float3> weight) noexcept;
  /* Exact closure_alloc_extra transition. Extra records consume tail slots
   * without increasing count. Failure atomically rolls back the immediately
   * preceding ordinary allocation identified by owner. */
  [[nodiscard]] luisa::compute::Bool
  allocate_extra(const Allocation &owner,
                 luisa::compute::Expr<std::uint32_t> slot_count) noexcept;
  /* Exact inverse of a successful ordinary+extra allocation. This is used by
   * Huang setup when an ellipse hit lies outside its projected radius. */
  void rollback_with_extra(
      const Allocation &owner,
      luisa::compute::Expr<std::uint32_t> extra_slot_count) noexcept;
  void set_type(luisa::compute::Expr<std::uint32_t> index,
                luisa::compute::Expr<std::uint32_t> type) noexcept;
  void set_weight(luisa::compute::Expr<std::uint32_t> index,
                  luisa::compute::Expr<luisa::float3> weight) noexcept;
  void add_weight(luisa::compute::Expr<std::uint32_t> index,
                  luisa::compute::Expr<luisa::float3> weight) noexcept;
  void set_sample_weight(luisa::compute::Expr<std::uint32_t> index,
                         luisa::compute::Expr<float> sample_weight) noexcept;
  void add_sample_weight(luisa::compute::Expr<std::uint32_t> index,
                         luisa::compute::Expr<float> sample_weight) noexcept;
  void set_normal(luisa::compute::Expr<std::uint32_t> index,
                  luisa::compute::Expr<luisa::float3> normal) noexcept;
  void set_oren_nayar_param(luisa::compute::Expr<std::uint32_t> index,
                            const OrenNayarParam &param) noexcept;
  void set_sheen_param(luisa::compute::Expr<std::uint32_t> index,
                       const SheenParam &param) noexcept;
  void set_velvet_param(luisa::compute::Expr<std::uint32_t> index,
                        const VelvetParam &param) noexcept;
  void set_toon_param(luisa::compute::Expr<std::uint32_t> index,
                      const ToonParam &param) noexcept;
  void set_ray_portal_param(luisa::compute::Expr<std::uint32_t> index,
                            const RayPortalParam &param) noexcept;
  void set_hair_param(luisa::compute::Expr<std::uint32_t> index,
                      const HairParam &param) noexcept;
  void set_chiang_hair_param(luisa::compute::Expr<std::uint32_t> index,
                             const ChiangHairParam &param) noexcept;
  void set_huang_hair(luisa::compute::Expr<std::uint32_t> index,
                      const HuangHairParam &param,
                      const HuangHairExtra &extra) noexcept;
  void set_bssrdf_param(luisa::compute::Expr<std::uint32_t> index,
                        const BssrdfParam &param) noexcept;
  void set_microfacet_param(luisa::compute::Expr<std::uint32_t> index,
                            const MicrofacetParam &param) noexcept;
  void
  set_generalized_schlick(luisa::compute::Expr<std::uint32_t> index,
                          const FresnelGeneralizedSchlick &fresnel) noexcept;
  void set_fresnel_conductor(luisa::compute::Expr<std::uint32_t> index,
                             const FresnelConductor &fresnel) noexcept;
  void set_fresnel_f82_tint(luisa::compute::Expr<std::uint32_t> index,
                            const FresnelF82Tint &fresnel) noexcept;
  void set_left(luisa::compute::Expr<std::uint32_t> left) noexcept;

  [[nodiscard]] ShaderClosureCommon
  common(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] OrenNayarClosure
  oren_nayar(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] SheenClosure
  sheen(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] VelvetClosure
  velvet(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] ToonClosure
  toon(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] RayPortalClosure
  ray_portal(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] HairClosure
  hair(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] ChiangHairClosure
  chiang_hair(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] HuangHairClosure
  huang_hair(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] BssrdfClosure
  bssrdf(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] MicrofacetParam
  microfacet_param(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] MicrofacetClosure
  microfacet(luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] MicrofacetConductorClosure microfacet_conductor(
      luisa::compute::Expr<std::uint32_t> index) const noexcept;
  [[nodiscard]] MicrofacetF82TintClosure
  microfacet_f82_tint(luisa::compute::Expr<std::uint32_t> index) const noexcept;
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
  ClosurePool *closure;

  ShaderData(luisa::compute::Expr<luisa::float3> position,
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
             luisa::compute::Expr<std::uint32_t> random_state = 0u,
             ClosurePool *closure_pool = nullptr) noexcept;
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
[[nodiscard]] luisa::compute::Float4
volume_attribute_float4(const KernelGlobals &kernel_globals,
                        ShaderData &shader_data,
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
 * The device machine still has one word PC loop and one primary opcode switch.
 */
void eval_nodes(
    const KernelGlobals &kernel_globals,
    const luisa::compute::BufferUInt &words,
    compiler::cycles_svm::ShaderType shader_type, std::uint32_t kernel_features,
    std::uint32_t node_feature_mask,
    const std::array<bool, compiler::cycles_svm::NODE_NUM> &node_types_used,
    const TransformState &transform_state, ShaderData &shader_data,
    const PathState &path_state, EvaluationResult &result) noexcept;

} // namespace psycles::luisa_backend::cycles_svm

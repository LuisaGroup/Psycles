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

/* The fields below are the exact ShaderData projection consumed by the first
 * copied SVM node families. More fields are added only when their Cycles
 * handler is copied; no alternative shader state is consulted. */
struct ShaderData {
  luisa::compute::Float3 P;
  luisa::compute::Float3 N;
  luisa::compute::Float3 Ng;
  luisa::compute::Float3 wi;
  luisa::compute::UInt shader;
  luisa::compute::UInt flag;
  luisa::compute::Float u;
  luisa::compute::Float v;
  luisa::compute::Float ray_length;
  luisa::compute::Float3 closure_emission_background;
  luisa::compute::Float3 closure_transparent_extinction;

  ShaderData(luisa::compute::Expr<luisa::float3> position,
             luisa::compute::Expr<luisa::float3> normal,
             luisa::compute::Expr<luisa::float3> geometric_normal,
             luisa::compute::Expr<luisa::float3> incoming,
             luisa::compute::Expr<std::uint32_t> shader_id,
             luisa::compute::Expr<std::uint32_t> shader_flags,
             luisa::compute::Expr<float> parametric_u,
             luisa::compute::Expr<float> parametric_v,
             luisa::compute::Expr<float> length) noexcept;
};

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
    const luisa::compute::BufferUInt &words,
    compiler::cycles_svm::ShaderType shader_type,
    std::uint32_t node_feature_mask,
    const std::array<bool, compiler::cycles_svm::NODE_NUM> &node_types_used,
    ShaderData &shader_data,
    const PathState &path_state,
    EvaluationResult &result) noexcept;

} // namespace psycles::luisa_backend::cycles_svm

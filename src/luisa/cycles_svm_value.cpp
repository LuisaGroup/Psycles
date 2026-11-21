/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) \
  $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

[[nodiscard]] Float3 geometry_value(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    Expr<std::uint32_t> geometry_type) noexcept {
  const auto eval_one = [&](std::uint32_t gt) noexcept -> Float3 {
    switch (gt) {
      case NODE_GEOM_P: return Float3{shader_data.P};
      case NODE_GEOM_N: return Float3{shader_data.N};
      case NODE_GEOM_T: return kernel_globals.primitive_tangent(shader_data);
      case NODE_GEOM_I: return Float3{shader_data.wi};
      case NODE_GEOM_Ng: return Float3{shader_data.Ng};
      case NODE_GEOM_uv:
        return make_float3(1.0f - shader_data.u - shader_data.v,
                           shader_data.u, 0.0f);
      default: return make_float3(0.0f);
    }
  };
  if (const auto ct = try_const_uint(geometry_type)) {
    return eval_one(*ct);
  }
  Float3 data = make_float3(0.0f);
  $switch(geometry_type) {
    PSYCLES_SVM_CASE(NODE_GEOM_P) { data = eval_one(NODE_GEOM_P); };
    PSYCLES_SVM_CASE(NODE_GEOM_N) { data = eval_one(NODE_GEOM_N); };
    PSYCLES_SVM_CASE(NODE_GEOM_T) { data = eval_one(NODE_GEOM_T); };
    PSYCLES_SVM_CASE(NODE_GEOM_I) { data = eval_one(NODE_GEOM_I); };
    PSYCLES_SVM_CASE(NODE_GEOM_Ng) { data = eval_one(NODE_GEOM_Ng); };
    PSYCLES_SVM_CASE(NODE_GEOM_uv) { data = eval_one(NODE_GEOM_uv); };
    $default { data = make_float3(0.0f); };
  };
  return data;
}

[[nodiscard]] Dual3 geometry_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    Expr<std::uint32_t> geometry_type) noexcept {
  const auto eval_one = [&](std::uint32_t gt) noexcept -> Dual3 {
    Dual3 d{.val = make_float3(0.0f),
            .dx = make_float3(0.0f),
            .dy = make_float3(0.0f)};
    switch (gt) {
      case NODE_GEOM_P:
        d.val = shader_data.P;
        d.dx = shader_data.dPdu * shader_data.du.dx +
               shader_data.dPdv * shader_data.dv.dx;
        d.dy = shader_data.dPdu * shader_data.du.dy +
               shader_data.dPdv * shader_data.dv.dy;
        return d;
      case NODE_GEOM_N:
        d.val = shader_data.N;
        return d;
      case NODE_GEOM_T: {
        const auto tangent =
            kernel_globals.primitive_tangent_derivative(shader_data);
        d.val = tangent.val;
        d.dx = tangent.dx;
        d.dy = tangent.dy;
        return d;
      }
      case NODE_GEOM_I: {
        d.val = shader_data.wi;
        const auto incoming =
            differential_from_compact(shader_data.wi, shader_data.dI);
        d.dx = incoming.dx;
        d.dy = incoming.dy;
        return d;
      }
      case NODE_GEOM_Ng:
        d.val = shader_data.Ng;
        return d;
      case NODE_GEOM_uv:
        d.val = make_float3(1.0f - shader_data.u - shader_data.v,
                            shader_data.u, 0.0f);
        d.dx = make_float3(-shader_data.du.dx - shader_data.dv.dx,
                           shader_data.du.dx, 0.0f);
        d.dy = make_float3(-shader_data.du.dy - shader_data.dv.dy,
                           shader_data.du.dy, 0.0f);
        return d;
      default:
        return d;
    }
  };
  if (const auto ct = try_const_uint(geometry_type)) {
    return eval_one(*ct);
  }
  Dual3 data{.val = make_float3(0.0f),
             .dx = make_float3(0.0f),
             .dy = make_float3(0.0f)};
  $switch(geometry_type) {
    PSYCLES_SVM_CASE(NODE_GEOM_P) { data = eval_one(NODE_GEOM_P); };
    PSYCLES_SVM_CASE(NODE_GEOM_N) { data = eval_one(NODE_GEOM_N); };
    PSYCLES_SVM_CASE(NODE_GEOM_T) { data = eval_one(NODE_GEOM_T); };
    PSYCLES_SVM_CASE(NODE_GEOM_I) { data = eval_one(NODE_GEOM_I); };
    PSYCLES_SVM_CASE(NODE_GEOM_Ng) { data = eval_one(NODE_GEOM_Ng); };
    PSYCLES_SVM_CASE(NODE_GEOM_uv) { data = eval_one(NODE_GEOM_uv); };
    $default {};
  };
  return data;
}

} // namespace

void node_value_f(Cursor &cursor, Stack &stack,
                  bool use_derivatives) noexcept {
  const auto value = cursor.floating();
  const auto packed_output = cursor.word();
  const auto output = cursor.byte(packed_output, 0u);
  if (use_derivatives) {
    stack_store_dual1(
        stack, output,
        Dual1{.val = value, .dx = 0.0f, .dy = 0.0f});
  } else {
    stack_store_float(stack, output, value);
  }
}

void node_value_v(Cursor &cursor, Stack &stack,
                  bool use_derivatives) noexcept {
  const auto packed_output = cursor.word();
  const auto x = cursor.floating();
  const auto y = cursor.floating();
  const auto z = cursor.floating();
  const auto value = make_float3(x, y, z);
  const auto output = cursor.byte(packed_output, 0u);
  if (use_derivatives) {
    stack_store_dual3(
        stack, output,
        Dual3{.val = value,
              .dx = make_float3(0.0f),
              .dy = make_float3(0.0f)});
  } else {
    stack_store_float3(stack, output, value);
  }
}

void node_geometry(Cursor &cursor, Stack &stack,
                   const KernelGlobals &kernel_globals,
                   const ShaderData &shader_data,
                   bool use_derivatives) noexcept {
  const auto packed = cursor.word();
  const auto geometry_type = cursor.byte(packed, 0u);
  const auto bump_offset = cursor.byte(packed, 1u);
  const auto store_derivatives = cursor.byte(packed, 2u);
  const auto output_offset = cursor.byte(packed, 3u);
  const auto bump_filter_width = cursor.floating();

  if (use_derivatives) {
    const auto cbo = try_const_uint(bump_offset);
    const auto csd = try_const_uint(store_derivatives);
    if (cbo && *cbo == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_CENTER) &&
        csd && *csd == 0u) {
      stack_store_float3(
          stack, output_offset,
          geometry_value(kernel_globals, shader_data, geometry_type));
      return;
    }
    auto data =
        geometry_derivative(kernel_globals, shader_data, geometry_type);
    if (cbo) {
      if (*cbo == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
        data.val += data.dx * bump_filter_width;
      } else if (*cbo == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
        data.val += data.dy * bump_filter_width;
      }
    } else {
      $if(bump_offset == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
        data.val += data.dx * bump_filter_width;
      }
      $elif(bump_offset == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
        data.val += data.dy * bump_filter_width;
      };
    }
    if (csd) {
      if (*csd != 0u) {
        stack_store_dual3(stack, output_offset, data);
      } else {
        stack_store_float3(stack, output_offset, data.val);
      }
    } else {
      $if(store_derivatives != 0u) {
        stack_store_dual3(stack, output_offset, data);
      }
      $else { stack_store_float3(stack, output_offset, data.val); };
    }
  } else {
    stack_store_float3(
        stack, output_offset,
        geometry_value(kernel_globals, shader_data, geometry_type));
  }
}

void node_light_path(Cursor &cursor, Stack &stack,
                     const ShaderData &shader_data,
                     const PathState &path_state,
                     std::uint32_t node_feature_mask) noexcept {
  const auto path_type = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  const bool has_lp = (node_feature_mask & kernel_feature_node_light_path) != 0u;
  const auto eval_lp = [&](std::uint32_t lp) noexcept -> Float {
    switch (lp) {
      case NODE_LP_camera:
        return select(0.0f, 1.0f,
                      (path_state.visibility & path_ray_visibility_camera) != 0u);
      case NODE_LP_shadow:
        return select(0.0f, 1.0f,
                      (path_state.visibility & path_ray_visibility_shadow) != 0u);
      case NODE_LP_diffuse:
        return select(0.0f, 1.0f,
                      (path_state.visibility & path_ray_visibility_diffuse) != 0u);
      case NODE_LP_glossy:
        return select(0.0f, 1.0f,
                      (path_state.visibility & path_ray_visibility_glossy) != 0u);
      case NODE_LP_singular:
        return select(0.0f, 1.0f, (path_state.flag & path_ray_singular) != 0u);
      case NODE_LP_reflection:
        return select(0.0f, 1.0f, (path_state.flag & path_ray_reflect) != 0u);
      case NODE_LP_transmission:
        return select(
            0.0f, 1.0f,
            (path_state.visibility & path_ray_visibility_transmit) != 0u);
      case NODE_LP_volume_scatter:
        return select(
            0.0f, 1.0f,
            (path_state.visibility & path_ray_visibility_volume_scatter) != 0u);
      case NODE_LP_backfacing:
        return select(0.0f, 1.0f,
                      (shader_data.flag & shader_data_backfacing) != 0u);
      case NODE_LP_ray_length:
        return Float{shader_data.ray_length};
      case NODE_LP_ray_depth: {
        Float base = has_lp ? path_state.bounce.cast<float>() : Float{0.0f};
        return base +
               select(0.0f, 1.0f,
                      ((path_state.visibility & path_ray_visibility_shadow) !=
                       0u) |
                          ((path_state.flag & path_ray_emission) != 0u));
      }
      case NODE_LP_ray_transparent:
        return has_lp ? path_state.transparent_bounce.cast<float>()
                      : Float{0.0f};
      case NODE_LP_ray_diffuse:
        return has_lp ? path_state.diffuse_bounce.cast<float>() : Float{0.0f};
      case NODE_LP_ray_glossy:
        return has_lp ? path_state.glossy_bounce.cast<float>() : Float{0.0f};
      case NODE_LP_ray_transmission:
        return has_lp ? path_state.transmission_bounce.cast<float>()
                      : Float{0.0f};
      case NODE_LP_ray_portal:
        return has_lp ? path_state.portal_bounce.cast<float>() : Float{0.0f};
      default:
        return Float{0.0f};
    }
  };
  if (const auto ct = try_const_uint(path_type)) {
    stack_store_float(stack, output_offset, eval_lp(*ct));
    return;
  }
  Float info = 0.0f;
  $switch (path_type) {
    PSYCLES_SVM_CASE(NODE_LP_camera) { info = eval_lp(NODE_LP_camera); };
    PSYCLES_SVM_CASE(NODE_LP_shadow) { info = eval_lp(NODE_LP_shadow); };
    PSYCLES_SVM_CASE(NODE_LP_diffuse) { info = eval_lp(NODE_LP_diffuse); };
    PSYCLES_SVM_CASE(NODE_LP_glossy) { info = eval_lp(NODE_LP_glossy); };
    PSYCLES_SVM_CASE(NODE_LP_singular) { info = eval_lp(NODE_LP_singular); };
    PSYCLES_SVM_CASE(NODE_LP_reflection) { info = eval_lp(NODE_LP_reflection); };
    PSYCLES_SVM_CASE(NODE_LP_transmission) { info = eval_lp(NODE_LP_transmission); };
    PSYCLES_SVM_CASE(NODE_LP_volume_scatter) { info = eval_lp(NODE_LP_volume_scatter); };
    PSYCLES_SVM_CASE(NODE_LP_backfacing) { info = eval_lp(NODE_LP_backfacing); };
    PSYCLES_SVM_CASE(NODE_LP_ray_length) { info = eval_lp(NODE_LP_ray_length); };
    PSYCLES_SVM_CASE(NODE_LP_ray_depth) { info = eval_lp(NODE_LP_ray_depth); };
    PSYCLES_SVM_CASE(NODE_LP_ray_transparent) { info = eval_lp(NODE_LP_ray_transparent); };
    PSYCLES_SVM_CASE(NODE_LP_ray_diffuse) { info = eval_lp(NODE_LP_ray_diffuse); };
    PSYCLES_SVM_CASE(NODE_LP_ray_glossy) { info = eval_lp(NODE_LP_ray_glossy); };
    PSYCLES_SVM_CASE(NODE_LP_ray_transmission) { info = eval_lp(NODE_LP_ray_transmission); };
    PSYCLES_SVM_CASE(NODE_LP_ray_portal) { info = eval_lp(NODE_LP_ray_portal); };
    $default { info = 0.0f; };
  };

  stack_store_float(stack, output_offset, info);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE

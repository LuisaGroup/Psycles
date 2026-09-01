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
  Float3 data = make_float3(0.0f);
  $switch(geometry_type) {
    PSYCLES_SVM_CASE(NODE_GEOM_P) { data = shader_data.P; };
    PSYCLES_SVM_CASE(NODE_GEOM_N) { data = shader_data.N; };
    PSYCLES_SVM_CASE(NODE_GEOM_T) {
      data = kernel_globals.primitive_tangent(shader_data);
    };
    PSYCLES_SVM_CASE(NODE_GEOM_I) { data = shader_data.wi; };
    PSYCLES_SVM_CASE(NODE_GEOM_Ng) { data = shader_data.Ng; };
    PSYCLES_SVM_CASE(NODE_GEOM_uv) {
      data = make_float3(1.0f - shader_data.u - shader_data.v,
                         shader_data.u, 0.0f);
    };
    $default { data = make_float3(0.0f); };
  };
  return data;
}

[[nodiscard]] Dual3 geometry_derivative(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    Expr<std::uint32_t> geometry_type) noexcept {
  Dual3 data{.val = make_float3(0.0f),
             .dx = make_float3(0.0f),
             .dy = make_float3(0.0f)};
  $switch(geometry_type) {
    PSYCLES_SVM_CASE(NODE_GEOM_P) {
      data.val = shader_data.P;
      data.dx = shader_data.dPdu * shader_data.du.dx +
                shader_data.dPdv * shader_data.dv.dx;
      data.dy = shader_data.dPdu * shader_data.du.dy +
                shader_data.dPdv * shader_data.dv.dy;
    };
    PSYCLES_SVM_CASE(NODE_GEOM_N) { data.val = shader_data.N; };
    PSYCLES_SVM_CASE(NODE_GEOM_T) {
      const auto tangent =
          kernel_globals.primitive_tangent_derivative(shader_data);
      data.val = tangent.val;
      data.dx = tangent.dx;
      data.dy = tangent.dy;
    };
    PSYCLES_SVM_CASE(NODE_GEOM_I) {
      data.val = shader_data.wi;
      const auto incoming =
          differential_from_compact(shader_data.wi, shader_data.dI);
      data.dx = incoming.dx;
      data.dy = incoming.dy;
    };
    PSYCLES_SVM_CASE(NODE_GEOM_Ng) { data.val = shader_data.Ng; };
    PSYCLES_SVM_CASE(NODE_GEOM_uv) {
      data.val = make_float3(1.0f - shader_data.u - shader_data.v,
                             shader_data.u, 0.0f);
      data.dx = make_float3(-shader_data.du.dx - shader_data.dv.dx,
                            shader_data.du.dx, 0.0f);
      data.dy = make_float3(-shader_data.du.dy - shader_data.dv.dy,
                            shader_data.du.dy, 0.0f);
    };
    $default {
      data.val = make_float3(0.0f);
      data.dx = make_float3(0.0f);
      data.dy = make_float3(0.0f);
    };
  };
  return data;
}

} // namespace

void node_value_f(Cursor &cursor, Stack &stack) noexcept {
  const auto value = cursor.floating();
  const auto packed_output = cursor.word();
  stack_store_float(stack, cursor.byte(packed_output, 0u), value);
}

void node_value_v(Cursor &cursor, Stack &stack) noexcept {
  const auto packed_output = cursor.word();
  const auto x = cursor.floating();
  const auto y = cursor.floating();
  const auto z = cursor.floating();
  const auto value = make_float3(x, y, z);
  stack_store_float3(stack, cursor.byte(packed_output, 0u), value);
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
    auto data =
        geometry_derivative(kernel_globals, shader_data, geometry_type);
    $if(bump_offset == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
      data.val += data.dx * bump_filter_width;
    }
    $elif(bump_offset == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
      data.val += data.dy * bump_filter_width;
    };
    $if(store_derivatives != 0u) {
      stack_store_dual3(stack, output_offset, data);
    }
    $else { stack_store_float3(stack, output_offset, data.val); };
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
  Float info = 0.0f;

  $switch (path_type) {
    PSYCLES_SVM_CASE(NODE_LP_camera) {
      info = select(0.0f, 1.0f,
                    (path_state.visibility & path_ray_visibility_camera) !=
                        0u);
    };
    PSYCLES_SVM_CASE(NODE_LP_shadow) {
      info = select(0.0f, 1.0f,
                    (path_state.visibility & path_ray_visibility_shadow) !=
                        0u);
    };
    PSYCLES_SVM_CASE(NODE_LP_diffuse) {
      info = select(0.0f, 1.0f,
                    (path_state.visibility & path_ray_visibility_diffuse) !=
                        0u);
    };
    PSYCLES_SVM_CASE(NODE_LP_glossy) {
      info = select(0.0f, 1.0f,
                    (path_state.visibility & path_ray_visibility_glossy) !=
                        0u);
    };
    PSYCLES_SVM_CASE(NODE_LP_singular) {
      info = select(0.0f, 1.0f,
                    (path_state.flag & path_ray_singular) != 0u);
    };
    PSYCLES_SVM_CASE(NODE_LP_reflection) {
      info = select(0.0f, 1.0f,
                    (path_state.flag & path_ray_reflect) != 0u);
    };
    PSYCLES_SVM_CASE(NODE_LP_transmission) {
      info = select(
          0.0f, 1.0f,
          (path_state.visibility & path_ray_visibility_transmit) != 0u);
    };
    PSYCLES_SVM_CASE(NODE_LP_volume_scatter) {
      info = select(
          0.0f, 1.0f,
          (path_state.visibility & path_ray_visibility_volume_scatter) != 0u);
    };
    PSYCLES_SVM_CASE(NODE_LP_backfacing) {
      info = select(0.0f, 1.0f,
                    (shader_data.flag & shader_data_backfacing) != 0u);
    };
    PSYCLES_SVM_CASE(NODE_LP_ray_length) { info = shader_data.ray_length; };
    PSYCLES_SVM_CASE(NODE_LP_ray_depth) {
      if ((node_feature_mask & kernel_feature_node_light_path) != 0u) {
        info = path_state.bounce.cast<float>();
      }
      $if (((path_state.visibility & path_ray_visibility_shadow) != 0u) |
           ((path_state.flag & path_ray_emission) != 0u)) {
        info += 1.0f;
      };
    };
    PSYCLES_SVM_CASE(NODE_LP_ray_transparent) {
      if ((node_feature_mask & kernel_feature_node_light_path) != 0u) {
        info = path_state.transparent_bounce.cast<float>();
      }
    };
    PSYCLES_SVM_CASE(NODE_LP_ray_diffuse) {
      if ((node_feature_mask & kernel_feature_node_light_path) != 0u) {
        info = path_state.diffuse_bounce.cast<float>();
      }
    };
    PSYCLES_SVM_CASE(NODE_LP_ray_glossy) {
      if ((node_feature_mask & kernel_feature_node_light_path) != 0u) {
        info = path_state.glossy_bounce.cast<float>();
      }
    };
    PSYCLES_SVM_CASE(NODE_LP_ray_transmission) {
      if ((node_feature_mask & kernel_feature_node_light_path) != 0u) {
        info = path_state.transmission_bounce.cast<float>();
      }
    };
    PSYCLES_SVM_CASE(NODE_LP_ray_portal) {
      if ((node_feature_mask & kernel_feature_node_light_path) != 0u) {
        info = path_state.portal_bounce.cast<float>();
      }
    };
    $default { info = 0.0f; };
  };

  stack_store_float(stack, output_offset, info);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE

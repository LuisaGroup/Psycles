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

void node_geometry(Cursor &cursor, Stack &stack, ShaderData &shader_data,
                   Bool &supported) noexcept {
  const auto packed = cursor.word();
  const auto geometry_type = cursor.byte(packed, 0u);
  const auto output_offset = cursor.byte(packed, 3u);
  static_cast<void>(cursor.floating()); // bump_filter_width

  Float3 data = make_float3(0.0f);
  $switch (geometry_type) {
    PSYCLES_SVM_CASE(NODE_GEOM_P) { data = shader_data.P; };
    PSYCLES_SVM_CASE(NODE_GEOM_N) { data = shader_data.N; };
    PSYCLES_SVM_CASE(NODE_GEOM_T) {
      /* Cycles obtains this through primitive_tangent. That exact primitive
       * service is wired with the attribute family, not substituted here. */
      supported = false;
    };
    PSYCLES_SVM_CASE(NODE_GEOM_I) { data = shader_data.wi; };
    PSYCLES_SVM_CASE(NODE_GEOM_Ng) { data = shader_data.Ng; };
    PSYCLES_SVM_CASE(NODE_GEOM_uv) {
      data = make_float3(1.0f - shader_data.u - shader_data.v,
                         shader_data.u, 0.0f);
    };
    $default { data = make_float3(0.0f); };
  };
  stack_store_float3(stack, output_offset, data);
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
    PSYCLES_SVM_CASE(NODE_LP_ray_transparent) {
      if ((node_feature_mask & kernel_feature_node_light_path) != 0u) {
        info = path_state.transparent_bounce.cast<float>();
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

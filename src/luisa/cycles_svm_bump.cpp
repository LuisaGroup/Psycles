/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_set_bump(Cursor &cursor, Stack &stack,
                   const TransformState &transform_state,
                   const ShaderData &shader_data, bool bump_feature_enabled,
                   bool object_motion_enabled) noexcept {
  const auto scale_input = cursor.word();
  const auto strength_input = cursor.word();
  const auto bump_filter_width = cursor.floating();
  const auto packed_inputs = cursor.word();
  const auto packed_outputs = cursor.word();
  const auto normal_offset = cursor.byte(packed_inputs, 0u);
  const auto invert = cursor.byte(packed_inputs, 1u);
  const auto use_object_space = cursor.byte(packed_inputs, 2u);
  const auto center_offset = cursor.byte(packed_inputs, 3u);
  const auto dx_offset = cursor.byte(packed_outputs, 0u);
  const auto dy_offset = cursor.byte(packed_outputs, 1u);
  const auto output_offset = cursor.byte(packed_outputs, 2u);
  const auto bump_state_offset = cursor.byte(packed_outputs, 3u);

  if (bump_feature_enabled) {
    Float3 normal_in =
        stack_load_float3_default(stack, normal_offset, shader_data.N);
    Float3 dP_dx;
    Float3 dP_dy;
    $if(bump_state_offset == static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      const auto dP = differential_from_compact(shader_data.Ng, shader_data.dP);
      dP_dx = dP.dx;
      dP_dy = dP.dy;
    }
    $else {
      dP_dx = stack_load_float3(stack, bump_state_offset + 4u);
      dP_dy = stack_load_float3(stack, bump_state_offset + 7u);
    };

    $if(use_object_space != 0u) {
      object_inverse_normal_transform(normal_in, transform_state, shader_data,
                                      object_motion_enabled);
      object_inverse_dir_transform(dP_dx, transform_state, shader_data,
                                   object_motion_enabled);
      object_inverse_dir_transform(dP_dy, transform_state, shader_data,
                                   object_motion_enabled);
    };

    const auto rx = cross(dP_dy, normal_in);
    const auto ry = cross(normal_in, dP_dx);
    const auto height_center = stack_load_float(stack, center_offset);
    const auto height_x = stack_load_float(stack, dx_offset);
    const auto height_y = stack_load_float(stack, dy_offset);
    const auto determinant = dot(dP_dx, rx);
    const auto surface_gradient =
        (height_x - height_center) * rx + (height_y - height_center) * ry;
    const auto absolute_determinant = abs(determinant);
    Float strength = stack_load_input_float(stack, strength_input);
    Float scale = stack_load_input_float(stack, scale_input);
    $if(invert != 0u) { scale *= -1.0f; };
    strength = max(strength, 0.0f);

    const auto determinant_sign = select(1.0f, -1.0f, determinant < 0.0f);
    Float3 normal_out = safe_normalize_cycles(
        bump_filter_width * absolute_determinant * normal_in -
        scale * determinant_sign * surface_gradient);
    $if((normal_out.x == 0.0f) & (normal_out.y == 0.0f) &
        (normal_out.z == 0.0f)) {
      normal_out = normal_in;
    }
    $else {
      normal_out = normalize_cycles(strength * normal_out +
                                   (1.0f - strength) * normal_in);
    };

    $if(use_object_space != 0u) {
      object_normal_transform(normal_out, transform_state, shader_data,
                              object_motion_enabled);
    };
    stack_store_float3(stack, output_offset, normal_out);
  } else {
    stack_store_float3(stack, output_offset, make_float3(0.0f));
  }
}

} // namespace psycles::luisa_backend::cycles_svm::detail

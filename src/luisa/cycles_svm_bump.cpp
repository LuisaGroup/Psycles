/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

[[nodiscard]] Bool
is_attribute_found(const AttributeDescriptor &descriptor) noexcept {
  return descriptor.offset != static_cast<std::int32_t>(ATTR_STD_NOT_FOUND);
}

[[nodiscard]] Float3 packed_float3_value(
    Var<compiler::cycles_svm::packed_float3> value) noexcept {
  return make_float3(value.x, value.y, value.z);
}

[[nodiscard]] Float3 triangle_face_normal_undisplaced(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    Expr<std::int32_t> position_offset) noexcept {
  const auto indices =
      kernel_globals.triangle_vertex_indices(shader_data.prim);
  const auto v0 = packed_float3_value(kernel_globals.attribute_float3(
      position_offset + indices.x.cast<std::int32_t>()));
  const auto v1 = packed_float3_value(kernel_globals.attribute_float3(
      position_offset + indices.y.cast<std::int32_t>()));
  const auto v2 = packed_float3_value(kernel_globals.attribute_float3(
      position_offset + indices.z.cast<std::int32_t>()));
  const Bool negative_scale_applied =
      ((shader_data.object_flag &
        static_cast<std::uint32_t>(SD_OBJECT_NEGATIVE_SCALE)) != 0u) &
      ((shader_data.object_flag &
        static_cast<std::uint32_t>(SD_OBJECT_TRANSFORM_APPLIED)) != 0u);
  return normalize_cycles(select(cross(v1 - v0, v2 - v0),
                                 cross(v2 - v0, v1 - v0),
                                 negative_scale_applied));
}

void primitive_normal_set_undisplaced(
    const KernelGlobals &kernel_globals,
    const TransformState &transform_state, ShaderData &shader_data,
    Expr<std::int32_t> position_offset,
    bool object_motion_enabled) noexcept {
  Float3 normal = make_float3(0.0f);
  Bool found = true;
  $if((shader_data.shader & shader_smooth_normal) != 0u) {
    const auto descriptor = find_attribute(
        kernel_globals, shader_data,
        static_cast<luisa::ulong>(ATTR_STD_NORMAL_UNDISPLACED));
    $if(is_attribute_found(descriptor)) {
      normal = safe_normalize_cycles(primitive_surface_attribute_float3(
          kernel_globals, shader_data, descriptor));
    }
    $else { found = false; };
  }
  $else {
    normal = triangle_face_normal_undisplaced(
        kernel_globals, shader_data, position_offset);
  };
  $if(found) {
    object_normal_transform(normal, transform_state, shader_data,
                            object_motion_enabled);
    shader_data.N = select(
        normal, -normal,
        (shader_data.flag & shader_data_backfacing) != 0u);
  };
}

} // namespace

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

void node_set_normal(Cursor &cursor, Stack &stack,
                     ShaderData &shader_data) noexcept {
  const auto packed = cursor.word();
  const auto direction_offset = cursor.byte(packed, 0u);
  const auto normal_offset = cursor.byte(packed, 1u);
  const auto normal = stack_load_float3(stack, direction_offset);
  shader_data.N = normal;
  stack_store_float3(stack, normal_offset, normal);
}

void node_enter_bump_eval(Cursor &cursor, Stack &stack,
                          const KernelGlobals &kernel_globals,
                          const TransformState &transform_state,
                          ShaderData &shader_data,
                          bool object_motion_enabled) noexcept {
  const auto state_offset = cursor.byte(cursor.word(), 0u);

  stack_store_float3(stack, state_offset, shader_data.P);
  stack_store_float(stack, state_offset + 3u, shader_data.dP);

  const auto descriptor = find_attribute(
      kernel_globals, shader_data,
      static_cast<luisa::ulong>(ATTR_STD_POSITION_UNDISPLACED));
  $if(is_attribute_found(descriptor)) {
    auto position = primitive_surface_attribute_float3_derivative(
        kernel_globals, shader_data, descriptor);
    object_position_transform(position, transform_state, shader_data,
                              object_motion_enabled);

    shader_data.P = position.val;
    shader_data.dP =
        0.5f * (length(position.dx) + length(position.dy));
    stack_store_float3(stack, state_offset + 4u, position.dx);
    stack_store_float3(stack, state_offset + 7u, position.dy);

    primitive_normal_set_undisplaced(
        kernel_globals, transform_state, shader_data, descriptor.offset,
        object_motion_enabled);
  };
}

void node_leave_bump_eval(Cursor &cursor, Stack &stack,
                          ShaderData &shader_data) noexcept {
  const auto state_offset = cursor.byte(cursor.word(), 0u);
  shader_data.P = stack_load_float3(stack, state_offset);
  shader_data.dP = stack_load_float(stack, state_offset + 3u);
}

void node_set_displacement(Cursor &cursor, Stack &stack,
                           ShaderData &shader_data,
                           bool bump_feature_enabled) noexcept {
  const auto displacement_offset = cursor.byte(cursor.word(), 0u);
  if (bump_feature_enabled) {
    shader_data.P += stack_load_float3(stack, displacement_offset);
  }
}

void node_displacement(Cursor &cursor, Stack &stack,
                       const TransformState &transform_state,
                       const ShaderData &shader_data,
                       bool bump_feature_enabled,
                       bool object_motion_enabled) noexcept {
  const auto space = cursor.word();
  const auto height = stack_load_input_float(stack, cursor.word());
  const auto midlevel = stack_load_input_float(stack, cursor.word());
  const auto scale = stack_load_input_float(stack, cursor.word());
  const auto packed_offsets = cursor.word();
  const auto normal_offset = cursor.byte(packed_offsets, 0u);
  const auto output_offset = cursor.byte(packed_offsets, 1u);

  if (bump_feature_enabled) {
    auto displacement =
        stack_load_float3_default(stack, normal_offset, shader_data.N);
    $if(space == static_cast<std::uint32_t>(NODE_NORMAL_MAP_OBJECT)) {
      object_inverse_normal_transform(displacement, transform_state,
                                      shader_data, object_motion_enabled);
      displacement *= (height - midlevel) * scale;
      object_dir_transform(displacement, transform_state, shader_data,
                           object_motion_enabled);
    }
    $else { displacement *= (height - midlevel) * scale; };
    stack_store_float3(stack, output_offset, displacement);
  } else {
    stack_store_float3(stack, output_offset, make_float3(0.0f));
  }
}

} // namespace psycles::luisa_backend::cycles_svm::detail

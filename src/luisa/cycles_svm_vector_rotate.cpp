/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

[[nodiscard]] Float3 rotate_around_axis(Expr<luisa::float3> point,
                                        Expr<luisa::float3> axis,
                                        Expr<float> angle) noexcept {
  const Float cosine = cos(angle);
  const Float sine = sin(angle);
  const Float one_minus_cosine = 1.0f - cosine;
  return make_float3(
      ((cosine + one_minus_cosine * axis.x * axis.x) * point.x) +
          ((one_minus_cosine * axis.x * axis.y - axis.z * sine) * point.y) +
          ((one_minus_cosine * axis.x * axis.z + axis.y * sine) * point.z),
      ((one_minus_cosine * axis.x * axis.y + axis.z * sine) * point.x) +
          ((cosine + one_minus_cosine * axis.y * axis.y) * point.y) +
          ((one_minus_cosine * axis.y * axis.z - axis.x * sine) * point.z),
      ((one_minus_cosine * axis.x * axis.z - axis.y * sine) * point.x) +
          ((one_minus_cosine * axis.y * axis.z + axis.x * sine) * point.y) +
          ((cosine + one_minus_cosine * axis.z * axis.z) * point.z));
}

[[nodiscard]] Float3 rotate_euler(Expr<luisa::float3> point,
                                  Expr<luisa::float3> euler,
                                  Expr<bool> transposed) noexcept {
  const Float cx = cos(euler.x);
  const Float cy = cos(euler.y);
  const Float cz = cos(euler.z);
  const Float sx = sin(euler.x);
  const Float sy = sin(euler.y);
  const Float sz = sin(euler.z);

  const Float t_xx = cy * cz;
  const Float t_yx = cy * sz;
  const Float t_zx = -sy;
  const Float t_xy = sy * sx * cz - cx * sz;
  const Float t_yy = sy * sx * sz + cx * cz;
  const Float t_zy = cy * sx;
  const Float t_xz = sy * cx * cz + sx * sz;
  const Float t_yz = sy * cx * sz - sx * cz;
  const Float t_zz = cy * cx;

  Float3 result = make_float3(0.0f);
  $if (transposed) {
    result = make_float3(
        point.x * t_xx + point.y * t_yx + point.z * t_zx,
        point.x * t_xy + point.y * t_yy + point.z * t_zy,
        point.x * t_xz + point.y * t_yz + point.z * t_zz);
  }
  $else {
    result = make_float3(
        point.x * t_xx + point.y * t_xy + point.z * t_xz,
        point.x * t_yx + point.y * t_yy + point.z * t_yz,
        point.x * t_zx + point.y * t_zy + point.z * t_zz);
  };
  return result;
}

} // namespace

void node_vector_rotate(Cursor &cursor, Stack &stack) noexcept {
  const auto rotate_type = cursor.word();
  const auto vector_x = cursor.word();
  const auto vector_y = cursor.word();
  const auto vector_z = cursor.word();
  const auto center_x = cursor.word();
  const auto center_y = cursor.word();
  const auto center_z = cursor.word();
  const auto axis_x = cursor.word();
  const auto axis_y = cursor.word();
  const auto axis_z = cursor.word();
  const auto rotation_x = cursor.word();
  const auto rotation_y = cursor.word();
  const auto rotation_z = cursor.word();
  const auto angle_input = cursor.word();
  const auto packed = cursor.word();
  const auto invert = cursor.byte(packed, 0u);
  const auto result_offset = cursor.byte(packed, 1u);

  $if (result_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    const Float3 vector =
        stack_load_input_float3(stack, vector_x, vector_y, vector_z);
    const Float3 center =
        stack_load_input_float3(stack, center_x, center_y, center_z);
    Float3 result = make_float3(0.0f);

    $if (rotate_type == static_cast<std::uint32_t>(
                            NODE_VECTOR_ROTATE_TYPE_EULER_XYZ)) {
      const Float3 rotation = stack_load_input_float3(
          stack, rotation_x, rotation_y, rotation_z);
      result = rotate_euler(vector - center, rotation, invert != 0u) + center;
    }
    $else {
      Float3 axis = make_float3(0.0f);
      Float axis_length = 0.0f;
      $switch (rotate_type) {
        $case(static_cast<std::uint32_t>(NODE_VECTOR_ROTATE_TYPE_AXIS_X)) {
          axis = make_float3(1.0f, 0.0f, 0.0f);
          axis_length = 1.0f;
        };
        $case(static_cast<std::uint32_t>(NODE_VECTOR_ROTATE_TYPE_AXIS_Y)) {
          axis = make_float3(0.0f, 1.0f, 0.0f);
          axis_length = 1.0f;
        };
        $case(static_cast<std::uint32_t>(NODE_VECTOR_ROTATE_TYPE_AXIS_Z)) {
          axis = make_float3(0.0f, 0.0f, 1.0f);
          axis_length = 1.0f;
        };
        $default {
          axis = stack_load_input_float3(stack, axis_x, axis_y, axis_z);
          axis_length = sqrt(dot(axis, axis));
        };
      };
      Float angle = stack_load_input_float(stack, angle_input);
      $if (invert != 0u) { angle = -angle; };
      $if (axis_length != 0.0f) {
        result =
            rotate_around_axis(vector - center, axis / axis_length, angle) +
            center;
      }
      $else { result = vector; };
    };

    stack_store_float3(stack, result_offset, result);
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

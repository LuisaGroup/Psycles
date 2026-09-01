/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

[[nodiscard]] Bool is_zero(Expr<luisa::float3> value) noexcept {
  return (value.x == 0.0f) & (value.y == 0.0f) & (value.z == 0.0f);
}

[[nodiscard]] Bool is_finite(Expr<luisa::float3> value) noexcept {
  return !any(dsl::isnan(value)) & !any(dsl::isinf(value));
}

[[nodiscard]] Bool
is_attribute_found(const AttributeDescriptor &descriptor) noexcept {
  return descriptor.offset != static_cast<std::int32_t>(ATTR_STD_NOT_FOUND);
}

[[nodiscard]] Dual3 dual_cross_right(const Dual3 &a,
                                     Expr<luisa::float3> b) noexcept {
  return {.val = cross(a.val, b),
          .dx = cross(a.dx, b),
          .dy = cross(a.dy, b)};
}

[[nodiscard]] Dual3 dual_cross_left(Expr<luisa::float3> a,
                                    const Dual3 &b) noexcept {
  return {.val = cross(a, b.val),
          .dx = cross(a, b.dx),
          .dy = cross(a, b.dy)};
}

[[nodiscard]] Dual3 float2_to_dual3(const Dual2 &value) noexcept {
  return {.val = make_float3(value.val.x, value.val.y, 0.0f),
          .dx = make_float3(value.dx.x, value.dx.y, 0.0f),
          .dy = make_float3(value.dy.x, value.dy.y, 0.0f)};
}

[[nodiscard]] Dual3 radial_tangent(const Dual3 &generated,
                                   Expr<std::uint32_t> axis) noexcept {
  Dual3 tangent;
  $if(axis == static_cast<std::uint32_t>(NODE_TANGENT_AXIS_X)) {
    tangent.val = make_float3(0.0f, -(generated.val.z - 0.5f),
                             generated.val.y - 0.5f);
    tangent.dx = make_float3(0.0f, -generated.dx.z, generated.dx.y);
    tangent.dy = make_float3(0.0f, -generated.dy.z, generated.dy.y);
  }
  $elif(axis == static_cast<std::uint32_t>(NODE_TANGENT_AXIS_Y)) {
    tangent.val = make_float3(-(generated.val.z - 0.5f), 0.0f,
                             generated.val.x - 0.5f);
    tangent.dx = make_float3(-generated.dx.z, 0.0f, generated.dx.x);
    tangent.dy = make_float3(-generated.dy.z, 0.0f, generated.dy.x);
  }
  $else {
    tangent.val = make_float3(-(generated.val.y - 0.5f),
                             generated.val.x - 0.5f, 0.0f);
    tangent.dx = make_float3(-generated.dx.y, generated.dx.x, 0.0f);
    tangent.dy = make_float3(-generated.dy.y, generated.dy.x, 0.0f);
  };
  return tangent;
}

[[nodiscard]] Float3 radial_tangent(Expr<luisa::float3> generated,
                                    Expr<std::uint32_t> axis) noexcept {
  Float3 tangent;
  $if(axis == static_cast<std::uint32_t>(NODE_TANGENT_AXIS_X)) {
    tangent = make_float3(0.0f, -(generated.z - 0.5f),
                          generated.y - 0.5f);
  }
  $elif(axis == static_cast<std::uint32_t>(NODE_TANGENT_AXIS_Y)) {
    tangent = make_float3(-(generated.z - 0.5f), 0.0f,
                          generated.x - 0.5f);
  }
  $else {
    tangent = make_float3(-(generated.y - 0.5f),
                          generated.x - 0.5f, 0.0f);
  };
  return tangent;
}

void evaluate_tangent_plain(Stack &stack,
                            const KernelGlobals &kernel_globals,
                            const TransformState &transform_state,
                            const ShaderData &shader_data,
                            Expr<std::uint32_t> direction_type,
                            Expr<std::uint32_t> axis,
                            Expr<std::uint32_t> attr,
                            Expr<std::uint32_t> tangent_offset,
                            bool object_motion_enabled) noexcept {
  const auto descriptor = find_attribute(
      kernel_globals, shader_data, attr.cast<luisa::ulong>());
  Float3 tangent = make_float3(0.0f);
  Bool active = true;
  $if(direction_type == static_cast<std::uint32_t>(NODE_TANGENT_UVMAP)) {
    $if(!is_attribute_found(descriptor)) { active = false; }
    $else {
      $if(descriptor.type ==
          static_cast<std::uint32_t>(NODE_ATTR_FLOAT2)) {
        tangent = make_float3(primitive_surface_attribute_float2(
                                  kernel_globals, shader_data, descriptor),
                              0.0f);
      }
      $else {
        tangent = primitive_surface_attribute_float3(
            kernel_globals, shader_data, descriptor);
      };
    };
  }
  $else {
    Float3 generated;
    $if(!is_attribute_found(descriptor)) { generated = shader_data.P; }
    $elif(descriptor.type ==
          static_cast<std::uint32_t>(NODE_ATTR_FLOAT2)) {
      generated = make_float3(primitive_surface_attribute_float2(
                                  kernel_globals, shader_data, descriptor),
                              0.0f);
    }
    $else {
      generated = primitive_surface_attribute_float3(
          kernel_globals, shader_data, descriptor);
    };
    tangent = radial_tangent(generated, axis);
  };

  $if(active) {
    object_normal_transform(tangent, transform_state, shader_data,
                            object_motion_enabled);
    tangent = cross(shader_data.N,
                    normalize_cycles(cross(tangent, shader_data.N)));
  };
  stack_store_float3(stack, tangent_offset, tangent);
}

void evaluate_tangent_derivative(Stack &stack,
                                 const KernelGlobals &kernel_globals,
                                 const TransformState &transform_state,
                                 const ShaderData &shader_data,
                                 Expr<std::uint32_t> direction_type,
                                 Expr<std::uint32_t> axis,
                                 Expr<std::uint32_t> attr,
                                 Expr<std::uint32_t> tangent_offset,
                                 bool object_motion_enabled) noexcept {
  const auto descriptor = find_attribute(
      kernel_globals, shader_data, attr.cast<luisa::ulong>());
  Dual3 tangent{.val = make_float3(0.0f),
                .dx = make_float3(0.0f),
                .dy = make_float3(0.0f)};
  Bool active = true;
  $if(direction_type == static_cast<std::uint32_t>(NODE_TANGENT_UVMAP)) {
    $if(!is_attribute_found(descriptor)) { active = false; }
    $elif(descriptor.type ==
          static_cast<std::uint32_t>(NODE_ATTR_FLOAT2)) {
      tangent = float2_to_dual3(primitive_surface_attribute_float2_derivative(
          kernel_globals, shader_data, descriptor));
    }
    $else {
      tangent = primitive_surface_attribute_float3_derivative(
          kernel_globals, shader_data, descriptor);
    };
  }
  $else {
    Dual3 generated;
    $if(!is_attribute_found(descriptor)) {
      generated = shading_position_dual(shader_data);
    }
    $elif(descriptor.type ==
          static_cast<std::uint32_t>(NODE_ATTR_FLOAT2)) {
      generated = float2_to_dual3(
          primitive_surface_attribute_float2_derivative(
              kernel_globals, shader_data, descriptor));
    }
    $else {
      generated = primitive_surface_attribute_float3_derivative(
          kernel_globals, shader_data, descriptor);
    };
    tangent = radial_tangent(generated, axis);
  };

  $if(active) {
    object_normal_transform(tangent, transform_state, shader_data,
                            object_motion_enabled);
    tangent = dual_cross_left(
        shader_data.N,
        normalize_dual_cycles(dual_cross_right(tangent, shader_data.N)));
  };
  stack_store_dual3(stack, tangent_offset, tangent);
}

} // namespace

void node_normal(Cursor &cursor, Stack &stack) noexcept {
  const auto input_x = cursor.word();
  const auto input_y = cursor.word();
  const auto input_z = cursor.word();
  const auto normal = stack_load_input_float3(stack, input_x, input_y, input_z);
  const auto packed_outputs = cursor.word();
  const auto normal_output = cursor.byte(packed_outputs, 0u);
  const auto dot_output = cursor.byte(packed_outputs, 1u);

  // Cursor reads are sequenced explicitly. C++ does not specify a left-to-right
  // order for function arguments, so spelling three reads inside make_float3
  // can reverse the serialized Cycles x/y/z payload on a conforming compiler.
  const auto direction_x = cursor.floating();
  const auto direction_y = cursor.floating();
  const auto direction_z = cursor.floating();
  Float3 direction = make_float3(direction_x, direction_y, direction_z);
  direction = normalize_cycles(direction);

  $if(normal_output != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, normal_output, direction);
  };
  $if(dot_output != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, dot_output,
                      dot(direction, normalize_cycles(normal)));
  };
}

void node_normal_map(Cursor &cursor, Stack &stack,
                     const KernelGlobals &kernel_globals,
                     const TransformState &transform_state,
                     const ShaderData &shader_data,
                     bool object_motion_enabled) noexcept {
  const auto space = cursor.word();
  const auto invert_green = cursor.word();
  const auto use_original_base = cursor.word();
  const auto attr = cursor.word();
  const auto attr_sign = cursor.word();
  const auto color_x = cursor.word();
  const auto color_y = cursor.word();
  const auto color_z = cursor.word();
  const auto strength_input = cursor.word();
  const auto normal_offset = cursor.byte(cursor.word(), 0u);

  Float3 color = 2.0f *
                 (stack_load_input_float3(stack, color_x, color_y, color_z) -
                  make_float3(0.5f));
  $if(invert_green != 0u) { color.y = -color.y; };

  const Bool is_backfacing =
      (shader_data.flag & shader_data_backfacing) != 0u;
  Float3 normal = shader_data.N;
  Float strength = stack_load_input_float(stack, strength_input);
  Bool linear_interpolate_strength = false;
  Bool active = true;

  $if(space == static_cast<std::uint32_t>(NODE_NORMAL_MAP_TANGENT)) {
    const Bool eligible =
        (shader_data.object != object_none) &
        ((shader_data.type & primitive_triangle) != 0u);
    $if(!eligible) { active = false; }
    $else {
      const auto tangent_descriptor = find_attribute(
          kernel_globals, shader_data, attr.cast<luisa::ulong>());
      const auto sign_descriptor = find_attribute(
          kernel_globals, shader_data, attr_sign.cast<luisa::ulong>());
      $if(!is_attribute_found(tangent_descriptor) |
          !is_attribute_found(sign_descriptor)) {
        active = false;
      }
      $else {
        const auto tangent = primitive_surface_attribute_float3(
            kernel_globals, shader_data, tangent_descriptor);
        const auto sign = primitive_surface_attribute_float(
            kernel_globals, shader_data, sign_descriptor);
        Float3 base_normal;
        $if((shader_data.shader & shader_smooth_normal) != 0u) {
          $if(use_original_base != 0u) {
            const auto undisplaced = find_attribute(
                kernel_globals, shader_data,
                static_cast<luisa::ulong>(ATTR_STD_NORMAL_UNDISPLACED));
            $if(is_attribute_found(undisplaced)) {
              base_normal = primitive_surface_attribute_float3(
                  kernel_globals, shader_data, undisplaced);
              linear_interpolate_strength = true;
            }
            $else {
              base_normal = triangle_smooth_normal_unnormalized_object_space(
                  kernel_globals, transform_state, shader_data,
                  object_motion_enabled);
            };
          }
          $else {
            base_normal = triangle_smooth_normal_unnormalized_object_space(
                kernel_globals, transform_state, shader_data,
                object_motion_enabled);
          };
        }
        $else {
          base_normal = shader_data.Ng;
          $if(is_backfacing) { base_normal = -base_normal; };
          object_inverse_normal_transform(base_normal, transform_state,
                                          shader_data, object_motion_enabled);
        };

        $if(!linear_interpolate_strength) {
          color.x *= strength;
          color.y *= strength;
          color.z = 1.0f +
                    (color.z - 1.0f) * clamp(strength, 0.0f, 1.0f);
        };
        const auto bitangent = sign * cross(base_normal, tangent);
        normal = safe_normalize_cycles(color.x * tangent +
                                       color.y * bitangent +
                                       color.z * base_normal);
        object_normal_transform(normal, transform_state, shader_data,
                                object_motion_enabled);
        $if(is_backfacing) { normal = -normal; };
      };
    };
  }
  $else {
    linear_interpolate_strength = true;
    $if((space == static_cast<std::uint32_t>(NODE_NORMAL_MAP_BLENDER_OBJECT)) |
        (space == static_cast<std::uint32_t>(NODE_NORMAL_MAP_BLENDER_WORLD))) {
      color.y = -color.y;
      color.z = -color.z;
    };
    normal = color;
    $if((space == static_cast<std::uint32_t>(NODE_NORMAL_MAP_OBJECT)) |
        (space == static_cast<std::uint32_t>(NODE_NORMAL_MAP_BLENDER_OBJECT))) {
      object_normal_transform(normal, transform_state, shader_data,
                              object_motion_enabled);
    }
    $else { normal = safe_normalize_cycles(normal); };
    $if(is_backfacing) { normal = -normal; };
  };

  $if(active) {
    $if(linear_interpolate_strength & (strength != 1.0f)) {
      strength = max(strength, 0.0f);
      normal = safe_normalize_cycles(shader_data.N +
                                     (normal - shader_data.N) * strength);
    };
    $if(is_zero(normal) | !is_finite(normal)) { normal = shader_data.N; };
  };
  stack_store_float3(stack, normal_offset, normal);
}

void node_tangent(Cursor &cursor, Stack &stack,
                  const KernelGlobals &kernel_globals,
                  const TransformState &transform_state,
                  const ShaderData &shader_data,
                  bool use_derivatives,
                  bool object_motion_enabled) noexcept {
  const auto direction_type = cursor.word();
  const auto axis = cursor.word();
  const auto attr = cursor.word();
  const auto tangent_offset = cursor.byte(cursor.word(), 0u);
  if (use_derivatives) {
    evaluate_tangent_derivative(stack, kernel_globals, transform_state,
                                shader_data, direction_type, axis, attr,
                                tangent_offset, object_motion_enabled);
  } else {
    evaluate_tangent_plain(stack, kernel_globals, transform_state, shader_data,
                           direction_type, axis, attr, tangent_offset,
                           object_motion_enabled);
  }
}

} // namespace psycles::luisa_backend::cycles_svm::detail

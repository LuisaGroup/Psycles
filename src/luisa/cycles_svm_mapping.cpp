/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include "surface_vector_mapping.h"

#include <psycles/luisa/cycles_transform.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

[[nodiscard]] Dual3 map_point(const Dual3 &vector, Float3 location,
                              Float3 rotation, Float3 scale) noexcept {
  return {
      .val = luisa_backend::detail::map_vector_point_inline(
          vector.val, location, rotation, scale),
      .dx = luisa_backend::detail::map_vector_direction_inline(
          vector.dx, rotation, scale),
      .dy = luisa_backend::detail::map_vector_direction_inline(
          vector.dy, rotation, scale)};
}

[[nodiscard]] Dual3 map_texture(const Dual3 &vector, Float3 location,
                                Float3 rotation, Float3 scale) noexcept {
  return {
      .val = luisa_backend::detail::map_vector_texture_inline(
          vector.val, location, rotation, scale),
      .dx = luisa_backend::detail::safe_divide_components(
          luisa_backend::detail::rotate_euler_transposed(vector.dx,
                                                         rotation),
          scale),
      .dy = luisa_backend::detail::safe_divide_components(
          luisa_backend::detail::rotate_euler_transposed(vector.dy,
                                                         rotation),
          scale)};
}

[[nodiscard]] Dual3 map_vector(const Dual3 &vector, Float3 rotation,
                               Float3 scale) noexcept {
  return {
      .val = luisa_backend::detail::map_vector_direction_inline(
          vector.val, rotation, scale),
      .dx = luisa_backend::detail::map_vector_direction_inline(
          vector.dx, rotation, scale),
      .dy = luisa_backend::detail::map_vector_direction_inline(
          vector.dy, rotation, scale)};
}

[[nodiscard]] Dual3 map_normal(const Dual3 &vector, Float3 rotation,
                               Float3 scale) noexcept {
  const auto map_unnormalized = [&](Expr<luisa::float3> value) {
    return luisa_backend::detail::rotate_euler(
        luisa_backend::detail::safe_divide_components(value, scale),
        rotation);
  };
  return safe_normalize_dual({.val = map_unnormalized(vector.val),
                              .dx = map_unnormalized(vector.dx),
                              .dy = map_unnormalized(vector.dy)});
}

[[nodiscard]] Dual3 evaluate_mapping(Expr<std::uint32_t> mapping_type,
                                     const Dual3 &vector, Float3 location,
                                     Float3 rotation, Float3 scale) noexcept {
  Dual3 result{.val = make_float3(0.0f),
               .dx = make_float3(0.0f),
               .dy = make_float3(0.0f)};
  $switch (mapping_type) {
    $case(static_cast<std::uint32_t>(NODE_MAPPING_TYPE_POINT)) {
      result = map_point(vector, location, rotation, scale);
    };
    $case(static_cast<std::uint32_t>(NODE_MAPPING_TYPE_TEXTURE)) {
      result = map_texture(vector, location, rotation, scale);
    };
    $case(static_cast<std::uint32_t>(NODE_MAPPING_TYPE_VECTOR)) {
      result = map_vector(vector, rotation, scale);
    };
    $case(static_cast<std::uint32_t>(NODE_MAPPING_TYPE_NORMAL)) {
      result = map_normal(vector, rotation, scale);
    };
  };
  return result;
}

} // namespace

void node_mapping(Cursor &cursor, Stack &stack,
                  bool use_derivatives) noexcept {
  const auto mapping_type = cursor.word();
  const auto vector_x = cursor.word();
  const auto vector_y = cursor.word();
  const auto vector_z = cursor.word();
  const auto location_x = cursor.word();
  const auto location_y = cursor.word();
  const auto location_z = cursor.word();
  const auto rotation_x = cursor.word();
  const auto rotation_y = cursor.word();
  const auto rotation_z = cursor.word();
  const auto scale_x = cursor.word();
  const auto scale_y = cursor.word();
  const auto scale_z = cursor.word();
  const auto packed = cursor.word();
  const auto result_offset = cursor.byte(packed, 0u);

  $if (result_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    const auto location = stack_load_input_float3(
        stack, location_x, location_y, location_z);
    const auto rotation = stack_load_input_float3(
        stack, rotation_x, rotation_y, rotation_z);
    const auto scale =
        stack_load_input_float3(stack, scale_x, scale_y, scale_z);
    if (use_derivatives) {
      const auto vector = stack_load_input_dual_float3(
          stack, vector_x, vector_y, vector_z);
      stack_store_dual3(
          stack, result_offset,
          evaluate_mapping(mapping_type, vector, location, rotation, scale));
    } else {
      const auto vector = stack_load_input_float3(
          stack, vector_x, vector_y, vector_z);
      const Dual3 dual{.val = vector,
                       .dx = make_float3(0.0f),
                       .dy = make_float3(0.0f)};
      stack_store_float3(
          stack, result_offset,
          evaluate_mapping(mapping_type, dual, location, rotation, scale).val);
    }
  };
}

void node_texture_mapping(Cursor &cursor, Stack &stack,
                          bool use_derivatives) noexcept {
  const auto packed = cursor.word();
  const auto vector_offset = cursor.byte(packed, 0u);
  const auto result_offset = cursor.byte(packed, 1u);
  const auto transform = packed_transform(cursor);
  if (use_derivatives) {
    const Dual3 vector{
        .val = stack_load_float3(stack, vector_offset),
        .dx = stack_load_float3(stack, vector_offset + 3u),
        .dy = stack_load_float3(stack, vector_offset + 6u)};
    stack_store_dual3(stack, result_offset,
                      transform_point(transform, vector));
  } else {
    stack_store_float3(
        stack, result_offset,
        cycles_transform::point(
            transform, stack_load_float3(stack, vector_offset)));
  }
}

void node_min_max(Cursor &cursor, Stack &stack) noexcept {
  const auto packed = cursor.word();
  const auto vector_offset = cursor.byte(packed, 0u);
  const auto result_offset = cursor.byte(packed, 1u);
  const auto minimum_x = cursor.floating();
  const auto minimum_y = cursor.floating();
  const auto minimum_z = cursor.floating();
  const auto maximum_x = cursor.floating();
  const auto maximum_y = cursor.floating();
  const auto maximum_z = cursor.floating();
  const auto value = stack_load_float3(stack, vector_offset);
  stack_store_float3(stack, result_offset,
                     min(max(make_float3(minimum_x, minimum_y, minimum_z),
                             value),
                         make_float3(maximum_x, maximum_y, maximum_z)));
}

} // namespace psycles::luisa_backend::cycles_svm::detail

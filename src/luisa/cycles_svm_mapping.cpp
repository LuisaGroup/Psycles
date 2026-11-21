/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include "surface_vector_mapping.h"

#include <psycles/luisa/cycles_transform.h>

#include <bit>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

struct EulerTrig {
  Float sx, cx, sy, cy, sz, cz;
};

[[nodiscard]] EulerTrig make_euler_trig(Float3 rotation) noexcept {
  return {.sx = sin(rotation.x),
          .cx = cos(rotation.x),
          .sy = sin(rotation.y),
          .cy = cos(rotation.y),
          .sz = sin(rotation.z),
          .cz = cos(rotation.z)};
}

[[nodiscard]] Float3 apply_euler(Float3 value, const EulerTrig &t) noexcept {
  const auto x_rotated = make_float3(
      value.x,
      t.cx * value.y - t.sx * value.z,
      t.sx * value.y + t.cx * value.z);
  const auto y_rotated = make_float3(
      t.cy * x_rotated.x + t.sy * x_rotated.z,
      x_rotated.y,
      -t.sy * x_rotated.x + t.cy * x_rotated.z);
  return make_float3(
      t.cz * y_rotated.x - t.sz * y_rotated.y,
      t.sz * y_rotated.x + t.cz * y_rotated.y,
      y_rotated.z);
}

[[nodiscard]] Float3 apply_euler_transposed(Float3 value,
                                            const EulerTrig &t) noexcept {
  return make_float3(
      t.cy * t.cz * value.x + t.cy * t.sz * value.y - t.sy * value.z,
      (t.sy * t.sx * t.cz - t.cx * t.sz) * value.x +
          (t.sy * t.sx * t.sz + t.cx * t.cz) * value.y +
          t.cy * t.sx * value.z,
      (t.sy * t.cx * t.cz + t.sx * t.sz) * value.x +
          (t.sy * t.cx * t.sz - t.sx * t.cz) * value.y +
          t.cy * t.cx * value.z);
}

[[nodiscard]] Dual3 rotate_dual(const Dual3 &value, Float3 rotation,
                                bool may_rotate, bool transposed,
                                bool use_derivatives) noexcept {
  if (!may_rotate) {
    return value;
  }
  Dual3 result = value;
  $if (any(rotation != make_float3(0.0f))) {
    const auto trig = make_euler_trig(rotation);
    if (transposed) {
      result.val = apply_euler_transposed(value.val, trig);
      if (use_derivatives) {
        result.dx = apply_euler_transposed(value.dx, trig);
        result.dy = apply_euler_transposed(value.dy, trig);
      }
    } else {
      result.val = apply_euler(value.val, trig);
      if (use_derivatives) {
        result.dx = apply_euler(value.dx, trig);
        result.dy = apply_euler(value.dy, trig);
      }
    }
  };
  return result;
}

[[nodiscard]] Dual3 map_point(const Dual3 &vector, Float3 location,
                              Float3 rotation, Float3 scale, bool may_rotate,
                              bool use_derivatives) noexcept {
  const Dual3 scaled{
      .val = vector.val * scale,
      .dx = use_derivatives ? Float3{vector.dx * scale} : make_float3(0.0f),
      .dy = use_derivatives ? Float3{vector.dy * scale} : make_float3(0.0f)};
  auto rotated = rotate_dual(scaled, rotation, may_rotate, false, use_derivatives);
  rotated.val += location;
  return rotated;
}

[[nodiscard]] Dual3 map_texture(const Dual3 &vector, Float3 location,
                                Float3 rotation, Float3 scale, bool may_rotate,
                                bool use_derivatives) noexcept {
  const Dual3 shifted{
      .val = vector.val - location,
      .dx = vector.dx,
      .dy = vector.dy};
  const auto rotated =
      rotate_dual(shifted, rotation, may_rotate, true, use_derivatives);
  return {
      .val = luisa_backend::detail::safe_divide_components(rotated.val, scale),
      .dx = use_derivatives
                ? luisa_backend::detail::safe_divide_components(rotated.dx, scale)
                : make_float3(0.0f),
      .dy = use_derivatives
                ? luisa_backend::detail::safe_divide_components(rotated.dy, scale)
                : make_float3(0.0f)};
}

[[nodiscard]] Dual3 map_vector(const Dual3 &vector, Float3 rotation,
                               Float3 scale, bool may_rotate,
                               bool use_derivatives) noexcept {
  const Dual3 scaled{
      .val = vector.val * scale,
      .dx = use_derivatives ? Float3{vector.dx * scale} : make_float3(0.0f),
      .dy = use_derivatives ? Float3{vector.dy * scale} : make_float3(0.0f)};
  return rotate_dual(scaled, rotation, may_rotate, false, use_derivatives);
}

[[nodiscard]] Dual3 map_normal(const Dual3 &vector, Float3 rotation,
                               Float3 scale, bool may_rotate,
                               bool use_derivatives) noexcept {
  const Dual3 divided{
      .val = luisa_backend::detail::safe_divide_components(vector.val, scale),
      .dx = use_derivatives
                ? luisa_backend::detail::safe_divide_components(vector.dx, scale)
                : make_float3(0.0f),
      .dy = use_derivatives
                ? luisa_backend::detail::safe_divide_components(vector.dy, scale)
                : make_float3(0.0f)};
  const auto rotated =
      rotate_dual(divided, rotation, may_rotate, false, use_derivatives);
  if (use_derivatives) {
    return safe_normalize_dual(rotated);
  }
  const auto mapped_length = length(rotated.val);
  return {.val = rotated.val / select(1.0f, mapped_length, mapped_length != 0.0f),
          .dx = make_float3(0.0f),
          .dy = make_float3(0.0f)};
}

[[nodiscard]] Dual3 evaluate_mapping(Expr<std::uint32_t> mapping_type,
                                     const Dual3 &vector, Float3 location,
                                     Float3 rotation, Float3 scale,
                                     MappingStaticOptions options,
                                     bool use_derivatives) noexcept {
  const auto enabled = [&](NodeMappingType t) noexcept {
    const auto idx = static_cast<std::uint32_t>(t);
    return idx < 32u && (options.type_mask & (1u << idx)) != 0u;
  };
  const auto eval_one = [&](NodeMappingType t) noexcept -> Dual3 {
    switch (t) {
      case NODE_MAPPING_TYPE_POINT:
        return map_point(vector, location, rotation, scale, options.may_rotate,
                         use_derivatives);
      case NODE_MAPPING_TYPE_TEXTURE:
        return map_texture(vector, location, rotation, scale, options.may_rotate,
                           use_derivatives);
      case NODE_MAPPING_TYPE_VECTOR:
        return map_vector(vector, rotation, scale, options.may_rotate,
                          use_derivatives);
      case NODE_MAPPING_TYPE_NORMAL:
      default:
        return map_normal(vector, rotation, scale, options.may_rotate,
                          use_derivatives);
    }
  };
  if (const auto ct = try_const_uint(mapping_type)) {
    return eval_one(static_cast<NodeMappingType>(*ct));
  }
  static constexpr NodeMappingType all_types[] = {
      NODE_MAPPING_TYPE_POINT, NODE_MAPPING_TYPE_TEXTURE,
      NODE_MAPPING_TYPE_VECTOR, NODE_MAPPING_TYPE_NORMAL};
  std::uint32_t count = 0u;
  NodeMappingType single = NODE_MAPPING_TYPE_POINT;
  for (const auto t : all_types) {
    if (enabled(t)) {
      ++count;
      single = t;
    }
  }
  if (count == 1u && options.type_mask != ~0u) {
    return eval_one(single);
  }
  Dual3 result{.val = make_float3(0.0f),
               .dx = make_float3(0.0f),
               .dy = make_float3(0.0f)};
  $switch (mapping_type) {
    for (const auto t : all_types) {
      if (enabled(t)) {
        $case(static_cast<std::uint32_t>(t)) { result = eval_one(t); };
      }
    }
  };
  return result;
}

} // namespace

void node_mapping(Cursor &cursor, Stack &stack, bool use_derivatives,
                  MappingStaticOptions options) noexcept {
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

  const bool needs_location =
      (options.type_mask & ((1u << NODE_MAPPING_TYPE_POINT) |
                            (1u << NODE_MAPPING_TYPE_TEXTURE))) != 0u;
  auto node_options = options;
  if (node_options.may_rotate) {
    const auto rx = try_const_uint(rotation_x);
    const auto ry = try_const_uint(rotation_y);
    const auto rz = try_const_uint(rotation_z);
    if (rx && ry && rz &&
        ((*rx >> 8u) != (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) &&
        std::bit_cast<float>(*rx) == 0.0f &&
        std::bit_cast<float>(*ry) == 0.0f &&
        std::bit_cast<float>(*rz) == 0.0f) {
      node_options.may_rotate = false;
    }
  }
  const auto emit_body = [&](Expr<std::uint32_t> out_off) noexcept {
    const Float3 location = needs_location
                                ? stack_load_input_float3(stack, location_x,
                                                          location_y, location_z)
                                : Float3{make_float3(0.0f)};
    const Float3 rotation = node_options.may_rotate
                                ? stack_load_input_float3(stack, rotation_x,
                                                          rotation_y, rotation_z)
                                : Float3{make_float3(0.0f)};
    const auto scale =
        stack_load_input_float3(stack, scale_x, scale_y, scale_z);
    if (use_derivatives) {
      const auto vector = stack_load_input_dual_float3(
          stack, vector_x, vector_y, vector_z);
      stack_store_dual3(
          stack, out_off,
          evaluate_mapping(mapping_type, vector, location, rotation, scale,
                           node_options, true));
    } else {
      const auto vector = stack_load_input_float3(
          stack, vector_x, vector_y, vector_z);
      const Dual3 dual{.val = vector,
                       .dx = make_float3(0.0f),
                       .dy = make_float3(0.0f)};
      stack_store_float3(
          stack, out_off,
          evaluate_mapping(mapping_type, dual, location, rotation, scale,
                           node_options, false)
              .val);
    }
  };
  if (const auto cr = try_const_uint(result_offset)) {
    if (*cr != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      emit_body(*cr);
    }
    return;
  }
  $if (result_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    emit_body(result_offset);
  };
}

void node_texture_mapping(Cursor &cursor, Stack &stack,
                          bool use_derivatives) noexcept {
  const auto packed = cursor.word();
  const auto vector_offset = cursor.byte(packed, 0u);
  const auto result_offset = cursor.byte(packed, 1u);
  const auto x0 = cursor.floating();
  const auto x1 = cursor.floating();
  const auto x2 = cursor.floating();
  const auto x3 = cursor.floating();
  const auto y0 = cursor.floating();
  const auto y1 = cursor.floating();
  const auto y2 = cursor.floating();
  const auto y3 = cursor.floating();
  const auto z0 = cursor.floating();
  const auto z1 = cursor.floating();
  const auto z2 = cursor.floating();
  const auto z3 = cursor.floating();

  const auto cx0 = try_const_float(x0);
  const auto cx1 = try_const_float(x1);
  const auto cx2 = try_const_float(x2);
  const auto cx3 = try_const_float(x3);
  const auto cy0 = try_const_float(y0);
  const auto cy1 = try_const_float(y1);
  const auto cy2 = try_const_float(y2);
  const auto cy3 = try_const_float(y3);
  const auto cz0 = try_const_float(z0);
  const auto cz1 = try_const_float(z1);
  const auto cz2 = try_const_float(z2);
  const auto cz3 = try_const_float(z3);

  if (cx0 && cx1 && cx2 && cx3 && cy0 && cy1 && cy2 && cy3 && cz0 && cz1 &&
      cz2 && cz3) {
    const bool is_diag = (*cx1 == 0.0f && *cx2 == 0.0f && *cy0 == 0.0f &&
                          *cy2 == 0.0f && *cz0 == 0.0f && *cz1 == 0.0f);
    const auto apply_dir = [&](Expr<luisa::float3> v) noexcept -> Float3 {
      const Float3 p = v;
      if (is_diag) {
        if (*cx0 == 1.0f && *cy1 == 1.0f && *cz2 == 1.0f) {
          return p;
        }
        return p * make_float3(*cx0, *cy1, *cz2);
      }
      return make_float3(dot(p, make_float3(*cx0, *cx1, *cx2)),
                         dot(p, make_float3(*cy0, *cy1, *cy2)),
                         dot(p, make_float3(*cz0, *cz1, *cz2)));
    };
    const auto apply_pt = [&](Expr<luisa::float3> v) noexcept -> Float3 {
      Float3 res = apply_dir(v);
      if (*cx3 != 0.0f || *cy3 != 0.0f || *cz3 != 0.0f) {
        res += make_float3(*cx3, *cy3, *cz3);
      }
      return res;
    };
    if (use_derivatives) {
      const Dual3 out{
          .val = apply_pt(stack_load_float3(stack, vector_offset)),
          .dx = apply_dir(
              stack_load_float3(stack, stack_offset_add(vector_offset, 3u))),
          .dy = apply_dir(
              stack_load_float3(stack, stack_offset_add(vector_offset, 6u)))};
      stack_store_dual3(stack, result_offset, out);
    } else {
      stack_store_float3(
          stack, result_offset,
          apply_pt(stack_load_float3(stack, vector_offset)));
    }
    return;
  }

  const auto transform = transform_from_rows(
      make_float4(x0, x1, x2, x3),
      make_float4(y0, y1, y2, y3),
      make_float4(z0, z1, z2, z3));
  if (use_derivatives) {
    const Dual3 vector{
        .val = stack_load_float3(stack, vector_offset),
        .dx = stack_load_float3(stack, stack_offset_add(vector_offset, 3u)),
        .dy = stack_load_float3(stack, stack_offset_add(vector_offset, 6u))};
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

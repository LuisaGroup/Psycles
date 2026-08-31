/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

[[nodiscard]] Float4 table_read(Cursor &cursor,
                                Expr<std::uint32_t> element) noexcept {
  const UInt base = element * 4u;
  return make_float4(cursor.floating_at(base), cursor.floating_at(base + 1u),
                     cursor.floating_at(base + 2u),
                     cursor.floating_at(base + 3u));
}

[[nodiscard]] Float4 rgb_ramp_lookup_clamped(
    Cursor &cursor, Expr<float> factor, Expr<bool> interpolate,
    Expr<std::uint32_t> table_size) noexcept {
  const UInt last = table_size - 1u;
  const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
  // Keep the signed float_to_int followed by clamp from Cycles. In
  // particular, changing the conversion to unsigned changes the defined
  // finite-input range and backend handling of non-finite values before the
  // explicit clamp.
  const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
  const UInt element = cast<uint>(index);
  const Float t = scaled - cast<float>(index);
  Float4 result = table_read(cursor, element);
  $if(interpolate & (t > 0.0f)) {
    const Float4 next = table_read(cursor, element + 1u);
    result = (1.0f - t) * result + t * next;
  };
  return result;
}

[[nodiscard]] Float4 rgb_ramp_lookup_extrapolated(
    Cursor &cursor, Expr<float> factor, Expr<bool> interpolate,
    Expr<bool> extrapolate, Expr<std::uint32_t> table_size) noexcept {
  const UInt last = table_size - 1u;
  Float4 result = make_float4(0.0f);
  $if(((factor < 0.0f) | (factor > 1.0f)) & extrapolate) {
    Float4 value = make_float4(0.0f);
    Float4 delta = make_float4(0.0f);
    Float distance = factor;
    $if(factor < 0.0f) {
      value = table_read(cursor, 0u);
      delta = value - table_read(cursor, 1u);
      distance = -factor;
    }
    $else {
      value = table_read(cursor, last);
      delta = value - table_read(cursor, last - 1u);
      distance = factor - 1.0f;
    };
    result = value + delta * distance * cast<float>(last);
  }
  $else {
    result = rgb_ramp_lookup_clamped(cursor, factor, interpolate, table_size);
  };
  return result;
}

[[nodiscard]] Float
float_ramp_lookup_clamped(Cursor &cursor, Expr<float> factor,
                          Expr<bool> interpolate,
                          Expr<std::uint32_t> table_size) noexcept {
  const UInt last = table_size - 1u;
  const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
  const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
  const UInt element = cast<uint>(index);
  const Float t = scaled - cast<float>(index);
  Float result = cursor.floating_at(element);
  $if(interpolate & (t > 0.0f)) {
    const Float next = cursor.floating_at(element + 1u);
    result = (1.0f - t) * result + t * next;
  };
  return result;
}

[[nodiscard]] Float
float_ramp_lookup_extrapolated(Cursor &cursor, Expr<float> factor,
                               Expr<bool> interpolate, Expr<bool> extrapolate,
                               Expr<std::uint32_t> table_size) noexcept {
  const UInt last = table_size - 1u;
  Float result = 0.0f;
  $if(((factor < 0.0f) | (factor > 1.0f)) & extrapolate) {
    Float value = 0.0f;
    Float delta = 0.0f;
    Float distance = factor;
    $if(factor < 0.0f) {
      value = cursor.floating_at(0u);
      delta = value - cursor.floating_at(1u);
      distance = -factor;
    }
    $else {
      value = cursor.floating_at(last);
      delta = value - cursor.floating_at(last - 1u);
      distance = factor - 1.0f;
    };
    result = value + delta * distance * cast<float>(last);
  }
  $else {
    result = float_ramp_lookup_clamped(cursor, factor, interpolate, table_size);
  };
  return result;
}

} // namespace

void node_rgb_ramp(Cursor &cursor, Stack &stack) noexcept {
  const UInt table_size = cursor.word();
  const Float factor = stack_load_input_float(stack, cursor.word());
  const UInt packed = cursor.word();
  const UInt interpolate = cursor.byte(packed, 0u);
  const UInt color_offset = cursor.byte(packed, 1u);
  const UInt alpha_offset = cursor.byte(packed, 2u);

  const Float4 color = rgb_ramp_lookup_clamped(
      cursor, factor, interpolate != 0u, table_size);

  $if(color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, color_offset, color.xyz());
  };
  $if(alpha_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, alpha_offset, color.w);
  };
  cursor.advance(table_size * 4u);
}

void node_curves(Cursor &cursor, Stack &stack) noexcept {
  const UInt color_x = cursor.word();
  const UInt color_y = cursor.word();
  const UInt color_z = cursor.word();
  const Float3 color =
      stack_load_input_float3(stack, color_x, color_y, color_z);
  const Float factor = stack_load_input_float(stack, cursor.word());
  const Float min_x = cursor.floating();
  const Float max_x = cursor.floating();
  const UInt table_size = cursor.word();
  const UInt packed = cursor.word();
  const Bool extrapolate = cursor.byte(packed, 0u) != 0u;
  const UInt output_offset = cursor.byte(packed, 1u);

  const Float range_x = max_x - min_x;
  const Float3 relative = (color - min_x) / range_x;
  const Float red = rgb_ramp_lookup_extrapolated(
                        cursor, relative.x, true, extrapolate, table_size)
                        .x;
  const Float green = rgb_ramp_lookup_extrapolated(
                          cursor, relative.y, true, extrapolate, table_size)
                          .y;
  const Float blue = rgb_ramp_lookup_extrapolated(
                         cursor, relative.z, true, extrapolate, table_size)
                         .z;
  const Float3 mapped = make_float3(red, green, blue);
  stack_store_float3(stack, output_offset,
                     (1.0f - factor) * color + factor * mapped);
  cursor.advance(table_size * 4u);
}

void node_float_curve(Cursor &cursor, Stack &stack) noexcept {
  const Float factor = stack_load_input_float(stack, cursor.word());
  const Float input = stack_load_input_float(stack, cursor.word());
  const Float min_x = cursor.floating();
  const Float max_x = cursor.floating();
  const UInt table_size = cursor.word();
  const UInt packed = cursor.word();
  const Bool extrapolate = cursor.byte(packed, 0u) != 0u;
  const UInt output_offset = cursor.byte(packed, 1u);

  const Float relative = (input - min_x) / (max_x - min_x);
  const Float value = float_ramp_lookup_extrapolated(cursor, relative, true,
                                                     extrapolate, table_size);
  stack_store_float(stack, output_offset,
                    (1.0f - factor) * input + factor * value);
  cursor.advance(table_size);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

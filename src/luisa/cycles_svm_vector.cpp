/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

[[nodiscard]] Float vector_component(Expr<luisa::float3> value,
                                     Expr<std::uint32_t> index) noexcept {
  if (const auto ci = try_const_uint(index)) {
    if (*ci == 0u) {
      return value.x;
    }
    if (*ci == 1u) {
      return value.y;
    }
    return value.z;
  }
  Float result = value.z;
  $if (index == 0u) {
    result = value.x;
  }
  $elif (index == 1u) {
    result = value.y;
  };
  return result;
}

} // namespace

void node_separate_vector(Cursor &cursor, Stack &stack,
                          bool use_derivatives) noexcept {
  const auto x_bits = cursor.word();
  const auto y_bits = cursor.word();
  const auto z_bits = cursor.word();
  const auto packed = cursor.word();
  const auto vector_index = cursor.byte(packed, 0u);
  const auto output_offset = cursor.byte(packed, 1u);

  const auto emit_sep = [&](Expr<std::uint32_t> out_off) noexcept {
    if (use_derivatives) {
      const auto input =
          stack_load_input_dual_float3(stack, x_bits, y_bits, z_bits);
      stack_store_float(stack, out_off,
                        vector_component(input.val, vector_index));
      stack_store_float(stack, stack_offset_add(out_off, 1u),
                        vector_component(input.dx, vector_index));
      stack_store_float(stack, stack_offset_add(out_off, 2u),
                        vector_component(input.dy, vector_index));
    } else {
      const Float3 value =
          stack_load_input_float3(stack, x_bits, y_bits, z_bits);
      stack_store_float(stack, out_off,
                        vector_component(value, vector_index));
    }
  };
  if (const auto co = try_const_uint(output_offset)) {
    if (*co != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      emit_sep(*co);
    }
    return;
  }
  $if (output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    emit_sep(output_offset);
  };
}

void node_combine_vector(Cursor &cursor, Stack &stack,
                         bool use_derivatives) noexcept {
  const auto input_bits = cursor.word();
  const auto packed = cursor.word();
  const auto vector_index = cursor.byte(packed, 0u);
  const auto output_offset = cursor.byte(packed, 1u);

  const auto emit_comb = [&](Expr<std::uint32_t> out_off) noexcept {
    const auto slot_off = stack_offset_add(out_off, vector_index);
    if (use_derivatives) {
      const auto input = stack_load_input_dual_float(stack, input_bits);
      stack_store_float(stack, slot_off, input.val);
      stack_store_float(stack, stack_offset_add(slot_off, 3u), input.dx);
      stack_store_float(stack, stack_offset_add(slot_off, 6u), input.dy);
    } else {
      const Float value = stack_load_input_float(stack, input_bits);
      stack_store_float(stack, slot_off, value);
    }
  };
  if (const auto co = try_const_uint(output_offset)) {
    if (*co != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      emit_comb(*co);
    }
    return;
  }
  $if (output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    emit_comb(output_offset);
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

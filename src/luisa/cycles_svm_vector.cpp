/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

void load_input_dual_float(Stack &stack, Expr<std::uint32_t> bits,
                           Float &value, Float &dx, Float &dy) noexcept {
  value = bits.bitcast<float>();
  dx = 0.0f;
  dy = 0.0f;
  $if ((bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    const UInt offset = bits & 0xffu;
    value = stack_load_float(stack, offset);
    dx = stack_load_float(stack, offset + 1u);
    dy = stack_load_float(stack, offset + 2u);
  };
}

void load_input_dual_float3(Stack &stack, Expr<std::uint32_t> x_bits,
                            Expr<std::uint32_t> y_bits,
                            Expr<std::uint32_t> z_bits, Float3 &value,
                            Float3 &dx, Float3 &dy) noexcept {
  value = make_float3(x_bits.bitcast<float>(), y_bits.bitcast<float>(),
                      z_bits.bitcast<float>());
  dx = make_float3(0.0f);
  dy = make_float3(0.0f);
  $if ((x_bits >> 8u) == (SVM_INPUT_STACK_OFFSET_MASK >> 8u)) {
    const UInt offset = x_bits & 0xffu;
    value = stack_load_float3(stack, offset);
    dx = stack_load_float3(stack, offset + 3u);
    dy = stack_load_float3(stack, offset + 6u);
  };
}

[[nodiscard]] Float vector_component(Expr<luisa::float3> value,
                                     Expr<std::uint32_t> index) noexcept {
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

  $if (output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    if (use_derivatives) {
      Float3 value;
      Float3 dx;
      Float3 dy;
      load_input_dual_float3(stack, x_bits, y_bits, z_bits, value, dx, dy);
      stack_store_float(stack, output_offset,
                        vector_component(value, vector_index));
      stack_store_float(stack, output_offset + 1u,
                        vector_component(dx, vector_index));
      stack_store_float(stack, output_offset + 2u,
                        vector_component(dy, vector_index));
    } else {
      const Float3 value =
          stack_load_input_float3(stack, x_bits, y_bits, z_bits);
      stack_store_float(stack, output_offset,
                        vector_component(value, vector_index));
    }
  };
}

void node_combine_vector(Cursor &cursor, Stack &stack,
                         bool use_derivatives) noexcept {
  const auto input_bits = cursor.word();
  const auto packed = cursor.word();
  const auto vector_index = cursor.byte(packed, 0u);
  const auto output_offset = cursor.byte(packed, 1u);

  $if (output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    if (use_derivatives) {
      Float value;
      Float dx;
      Float dy;
      load_input_dual_float(stack, input_bits, value, dx, dy);
      stack_store_float(stack, output_offset + vector_index, value);
      stack_store_float(stack, output_offset + vector_index + 3u, dx);
      stack_store_float(stack, output_offset + vector_index + 6u, dy);
    } else {
      const Float value = stack_load_input_float(stack, input_bits);
      stack_store_float(stack, output_offset + vector_index, value);
    }
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

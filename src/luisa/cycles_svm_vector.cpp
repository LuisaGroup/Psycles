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
      const auto input =
          stack_load_input_dual_float3(stack, x_bits, y_bits, z_bits);
      stack_store_float(stack, output_offset,
                        vector_component(input.val, vector_index));
      stack_store_float(stack, output_offset + 1u,
                        vector_component(input.dx, vector_index));
      stack_store_float(stack, output_offset + 2u,
                        vector_component(input.dy, vector_index));
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
      const auto input = stack_load_input_dual_float(stack, input_bits);
      stack_store_float(stack, output_offset + vector_index, input.val);
      stack_store_float(stack, output_offset + vector_index + 3u, input.dx);
      stack_store_float(stack, output_offset + vector_index + 6u, input.dy);
    } else {
      const Float value = stack_load_input_float(stack, input_bits);
      stack_store_float(stack, output_offset + vector_index, value);
    }
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

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

} // namespace psycles::luisa_backend::cycles_svm::detail

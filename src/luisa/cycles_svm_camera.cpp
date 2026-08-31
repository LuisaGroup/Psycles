/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_transform.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_camera(Cursor &cursor, Stack &stack,
                 const TransformState &transform_state,
                 const ShaderData &shader_data) noexcept {
  const auto packed = cursor.word();
  const auto vector_offset = cursor.byte(packed, 0u);
  const auto zdepth_offset = cursor.byte(packed, 1u);
  const auto distance_offset = cursor.byte(packed, 2u);

  // Direct projection of Cycles 5.2.1 svm_node_camera: transform sd->P once,
  // derive all three values unconditionally, then guard only the stack stores.
  const Float3 vector =
      cycles_transform::point(transform_state.world_to_camera, shader_data.P);
  const Float zdepth = vector.z;
  const Float distance = length(vector);

  $if(vector_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, vector_offset, normalize_cycles(vector));
  };
  $if(zdepth_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, zdepth_offset, zdepth);
  };
  $if(distance_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, distance_offset, distance);
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

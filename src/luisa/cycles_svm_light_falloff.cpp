/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <limits>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_light_falloff(Cursor &cursor, Stack &stack,
                        const ShaderData &shader_data) noexcept {
  const auto falloff_type = cursor.word();
  const auto strength_input = cursor.word();
  const auto smooth_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto output_offset = cursor.byte(packed_output, 0u);
  Float strength = stack_load_input_float(stack, strength_input);

  // Cycles treats FLT_MAX as the distant-light sentinel and returns Strength
  // before loading Smooth or recording any distance arithmetic. Keep this as
  // structured control flow: select would still evaluate FLT_MAX * FLT_MAX.
  $if(shader_data.ray_length != std::numeric_limits<float>::max()) {
    $switch(falloff_type) {
      $case(static_cast<std::uint32_t>(NODE_LIGHT_FALLOFF_QUADRATIC)) {};
      $case(static_cast<std::uint32_t>(NODE_LIGHT_FALLOFF_LINEAR)) {
        strength *= shader_data.ray_length;
      };
      $case(static_cast<std::uint32_t>(NODE_LIGHT_FALLOFF_CONSTANT)) {
        strength *= shader_data.ray_length * shader_data.ray_length;
      };
    };

    const auto smooth = stack_load_input_float(stack, smooth_input);
    $if(smooth > 0.0f) {
      const auto squared = shader_data.ray_length * shader_data.ray_length;
      strength *= squared / (smooth + squared);
    };
  };
  stack_store_float(stack, output_offset, strength);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

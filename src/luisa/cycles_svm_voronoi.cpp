/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_voronoi.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_tex_voronoi(Cursor &cursor, Stack &stack,
                      bool voronoi_extra_enabled) noexcept {
  const auto dimensions = cursor.word();
  const auto feature = cursor.word();
  const auto metric = cursor.word();
  const auto w_bits = cursor.word();
  const auto scale_bits = cursor.word();
  const auto detail_bits = cursor.word();
  const auto roughness_bits = cursor.word();
  const auto lacunarity_bits = cursor.word();
  const auto smoothness_bits = cursor.word();
  const auto exponent_bits = cursor.word();
  const auto randomness_bits = cursor.word();
  const auto first_outputs = cursor.word();
  const auto normalize = cursor.byte(first_outputs, 0u) != 0u;
  const auto coordinate_offset = cursor.byte(first_outputs, 1u);
  const auto distance_offset = cursor.byte(first_outputs, 2u);
  const auto color_offset = cursor.byte(first_outputs, 3u);
  const auto second_outputs = cursor.word();
  const auto position_offset = cursor.byte(second_outputs, 0u);
  const auto w_output_offset = cursor.byte(second_outputs, 1u);
  const auto radius_offset = cursor.byte(second_outputs, 2u);

  const auto vector = stack_load_float3(stack, coordinate_offset);
  const auto w = stack_load_input_float(stack, w_bits);
  const auto scale = stack_load_input_float(stack, scale_bits);
  const auto detail = stack_load_input_float(stack, detail_bits);
  const auto roughness = stack_load_input_float(stack, roughness_bits);
  const auto lacunarity = stack_load_input_float(stack, lacunarity_bits);
  const auto smoothness = stack_load_input_float(stack, smoothness_bits);
  const auto exponent = stack_load_input_float(stack, exponent_bits);
  const auto randomness = stack_load_input_float(stack, randomness_bits);

  const auto output = cycles_voronoi::evaluate_runtime(
      voronoi_extra_enabled, dimensions, feature, metric, normalize, vector, w,
      scale, detail, roughness, lacunarity, smoothness, exponent, randomness);
  $if (distance_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, distance_offset, output.color_distance.w);
  };
  $if (color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, color_offset, output.color_distance.xyz());
  };
  $if (position_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, position_offset, output.position.xyz());
  };
  $if (w_output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, w_output_offset, output.position.w);
  };
  $if (radius_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, radius_offset, output.radius);
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

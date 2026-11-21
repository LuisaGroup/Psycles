/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_voronoi.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_tex_voronoi(Cursor &cursor, Stack &stack,
                      bool voronoi_extra_enabled,
                      VoronoiStaticOptions options) noexcept {
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
  const Float w = (options.dimension_mask & ((1u << 1u) | (1u << 4u))) != 0u
                      ? stack_load_input_float(stack, w_bits)
                      : def(0.0f);
  const auto scale = stack_load_input_float(stack, scale_bits);
  const Float detail = options.may_have_detail
                           ? stack_load_input_float(stack, detail_bits)
                           : def(0.0f);
  const Float roughness = options.may_have_detail
                              ? stack_load_input_float(stack, roughness_bits)
                              : def(0.0f);
  const Float lacunarity = options.may_have_detail
                               ? stack_load_input_float(stack, lacunarity_bits)
                               : def(0.0f);
  const Float smoothness =
      (options.feature_mask & (1u << NODE_VORONOI_SMOOTH_F1)) != 0u
          ? stack_load_input_float(stack, smoothness_bits)
          : def(0.0f);
  const Float exponent =
      (options.metric_mask & (1u << NODE_VORONOI_MINKOWSKI)) != 0u
          ? stack_load_input_float(stack, exponent_bits)
          : def(0.0f);
  const auto randomness = stack_load_input_float(stack, randomness_bits);

  const cycles_voronoi::RuntimeFilterOptions filter{
      .dimension_mask = options.dimension_mask,
      .feature_mask = options.feature_mask,
      .metric_mask = options.metric_mask,
      .may_have_detail = options.may_have_detail,
      .may_normalize = options.may_normalize,
      .may_need_color = options.may_need_color,
      .may_need_position = options.may_need_position,
      .may_need_radius = options.may_need_radius};
  const auto output = cycles_voronoi::evaluate_runtime_filtered(
      voronoi_extra_enabled, filter, dimensions, feature, metric, normalize,
      vector, w, scale, detail, roughness, lacunarity, smoothness, exponent,
      randomness);
  stack_store_float_if_valid(stack, distance_offset, output.color_distance.w);
  if (options.may_need_color) {
    stack_store_float3_if_valid(stack, color_offset, output.color_distance.xyz());
  }
  if (options.may_need_position) {
    stack_store_float3_if_valid(stack, position_offset, output.position.xyz());
    stack_store_float_if_valid(stack, w_output_offset, output.position.w);
  }
  if (options.may_need_radius) {
    stack_store_float_if_valid(stack, radius_offset, output.radius);
  }
}

} // namespace psycles::luisa_backend::cycles_svm::detail

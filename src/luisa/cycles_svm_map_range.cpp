/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {
using namespace luisa::compute;
using namespace compiler::cycles_svm;

[[nodiscard]] Float3 vector_input(Cursor &cursor, Stack &stack) noexcept {
  const auto x = cursor.word();
  const auto y = cursor.word();
  const auto z = cursor.word();
  return stack_load_input_float3(stack, x, y, z);
}

[[nodiscard]] Float3 safe_divide(Float3 a, Float3 b) noexcept {
  return select(make_float3(0.0f), a / b, b != make_float3(0.0f));
}

[[nodiscard]] Float scalar_smoothstep(Float edge0, Float edge1, Float value,
                                      bool smoother) noexcept {
  // The scalar node calls this only inside its unequal-bounds branch.
  if (smoother) {
    const Float x = clamp((value - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f);
  }
  Float result = 0.0f;
  $if(value < edge0) { result = 0.0f; }
  $elif(value >= edge1) { result = 1.0f; }
  $else {
    const Float t = (value - edge0) / (edge1 - edge0);
    result = (3.0f - 2.0f * t) * (t * t);
  };
  return result;
}
} // namespace

void node_map_range(Cursor &cursor, Stack &stack) noexcept {
  const auto type = cursor.word();
  const Float value = stack_load_input_float(stack, cursor.word());
  const Float from_min = stack_load_input_float(stack, cursor.word());
  const Float from_max = stack_load_input_float(stack, cursor.word());
  const Float to_min = stack_load_input_float(stack, cursor.word());
  const Float to_max = stack_load_input_float(stack, cursor.word());
  const Float steps = stack_load_input_float(stack, cursor.word());
  const auto output = cursor.byte(cursor.word(), 0u);
  Float result = 0.0f;
  $if(from_max != from_min) {
    Float factor = value;
    $switch(type) {
      $case(static_cast<unsigned>(NODE_MAP_RANGE_STEPPED)) {
        factor = (value - from_min) / (from_max - from_min);
        $if(steps > 0.0f) { factor = floor(factor * (steps + 1.0f)) / steps; }
        $else { factor = 0.0f; };
      };
      $case(static_cast<unsigned>(NODE_MAP_RANGE_SMOOTHSTEP)) {
        $if(from_min > from_max) {
          factor = 1.0f - scalar_smoothstep(from_max, from_min, factor, false);
        }
        $else {
          factor = scalar_smoothstep(from_min, from_max, factor, false);
        };
      };
      $case(static_cast<unsigned>(NODE_MAP_RANGE_SMOOTHERSTEP)) {
        $if(from_min > from_max) {
          factor = 1.0f - scalar_smoothstep(from_max, from_min, factor, true);
        }
        $else { factor = scalar_smoothstep(from_min, from_max, factor, true); };
      };
      $default { factor = (value - from_min) / (from_max - from_min); };
    };
    result = to_min + factor * (to_max - to_min);
  };
  // Scalar Clamp is a separate NODE_CLAMP, even for smooth interpolation.
  stack_store_float(stack, output, result);
}

void node_vector_map_range(Cursor &cursor, Stack &stack) noexcept {
  const auto type = cursor.word();
  const auto use_clamp = cursor.byte(cursor.word(), 0u);
  const auto value = vector_input(cursor, stack);
  const auto from_min = vector_input(cursor, stack);
  const auto from_max = vector_input(cursor, stack);
  const auto to_min = vector_input(cursor, stack);
  const auto to_max = vector_input(cursor, stack);
  const auto steps = vector_input(cursor, stack);
  const auto output = cursor.byte(cursor.word(), 0u);
  Float3 factor = value;
  $switch(type) {
    $case(static_cast<unsigned>(NODE_MAP_RANGE_STEPPED)) {
      factor = safe_divide(value - from_min, from_max - from_min);
      factor = select(make_float3(0.0f), floor(factor * (steps + 1.0f)) / steps,
                      steps > make_float3(0.0f));
    };
    $case(static_cast<unsigned>(NODE_MAP_RANGE_SMOOTHSTEP)) {
      factor = safe_divide(value - from_min, from_max - from_min);
      factor = clamp(factor, make_float3(0.0f), make_float3(1.0f));
      factor = (make_float3(3.0f) - 2.0f * factor) * (factor * factor);
    };
    $case(static_cast<unsigned>(NODE_MAP_RANGE_SMOOTHERSTEP)) {
      factor = safe_divide(value - from_min, from_max - from_min);
      factor = clamp(factor, make_float3(0.0f), make_float3(1.0f));
      factor =
          factor * factor * factor * (factor * (factor * 6.0f - 15.0f) + 10.0f);
    };
    $default { factor = safe_divide(value - from_min, from_max - from_min); };
  };
  Float3 result = to_min + factor * (to_max - to_min);
  // Unlike scalar, vector clamps internally and ignores Clamp for both smooth
  // modes. Its degenerate input interval maps to To Min, not scalar's zero.
  $if((type != static_cast<unsigned>(NODE_MAP_RANGE_SMOOTHSTEP)) &
      (type != static_cast<unsigned>(NODE_MAP_RANGE_SMOOTHERSTEP)) &
      (use_clamp > 0u)) {
    result = select(clamp(result, to_min, to_max),
                    clamp(result, to_max, to_min), to_min > to_max);
  };
  stack_store_float3(stack, output, result);
}
} // namespace psycles::luisa_backend::cycles_svm::detail

/* SPDX-FileCopyrightText: 2024 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_gabor.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_tex_gabor(Cursor &cursor, Stack &stack) noexcept {
  const auto gabor_type = cursor.word();
  const auto orientation_x = cursor.word();
  const auto orientation_y = cursor.word();
  const auto orientation_z = cursor.word();
  const auto scale_bits = cursor.word();
  const auto frequency_bits = cursor.word();
  const auto anisotropy_bits = cursor.word();
  const auto orientation_2d_bits = cursor.word();
  const auto offsets = cursor.word();
  const auto coordinates_offset = cursor.byte(offsets, 0u);
  const auto value_offset = cursor.byte(offsets, 1u);
  const auto phase_offset = cursor.byte(offsets, 2u);
  const auto intensity_offset = cursor.byte(offsets, 3u);

  const auto coordinates = stack_load_float3(stack, coordinates_offset);
  const auto orientation_3d = stack_load_input_float3(
      stack, orientation_x, orientation_y, orientation_z);
  const auto scale = stack_load_input_float(stack, scale_bits);
  const auto frequency = stack_load_input_float(stack, frequency_bits);
  const auto anisotropy = stack_load_input_float(stack, anisotropy_bits);
  const auto orientation_2d =
      stack_load_input_float(stack, orientation_2d_bits);
  const auto output = cycles_gabor::evaluate(
      gabor_type, coordinates, orientation_3d, scale, frequency, anisotropy,
      orientation_2d);

  $if (value_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, value_offset, output.x);
  };
  $if (phase_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, phase_offset, output.y);
  };
  $if (intensity_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, intensity_offset, output.z);
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

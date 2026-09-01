/* SPDX-FileCopyrightText: 2009-2010 Sony Pictures Imageworks Inc., et al. All Rights Reserved.
 * SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Adapted code from Open Shading Language. */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_color_nodes.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;

Float3 rec709_to_rgb(const KernelGlobals &kernel_globals,
                     Expr<luisa::float3> rec709) noexcept {
  Float3 result = rec709;
  $if (!kernel_globals.film_is_rec709()) {
    result = make_float3(dot(kernel_globals.film_rec709_to_r(), rec709),
                         dot(kernel_globals.film_rec709_to_g(), rec709),
                         dot(kernel_globals.film_rec709_to_b(), rec709));
  };
  return result;
}

Float3 xyz_to_rgb(const KernelGlobals &kernel_globals,
                  Expr<luisa::float3> xyz) noexcept {
  return make_float3(dot(kernel_globals.film_xyz_to_r(), xyz),
                     dot(kernel_globals.film_xyz_to_g(), xyz),
                     dot(kernel_globals.film_xyz_to_b(), xyz));
}

void node_blackbody(Cursor &cursor, Stack &stack,
                    const KernelGlobals &kernel_globals) noexcept {
  const auto temperature_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto color_offset = cursor.byte(packed_output, 0u);
  const Float temperature =
      stack_load_input_float(stack, temperature_input);
  const Float3 color = luisa::compute::max(
      rec709_to_rgb(kernel_globals,
                    cycles_color_nodes::blackbody_rec709(temperature)),
      make_float3(0.0f));
  stack_store_float3(stack, color_offset, color);
}

void node_wavelength(Cursor &cursor, Stack &stack,
                     const KernelGlobals &kernel_globals) noexcept {
  const auto wavelength_input = cursor.word();
  const auto packed_output = cursor.word();
  const auto color_offset = cursor.byte(packed_output, 0u);
  const Float wavelength =
      stack_load_input_float(stack, wavelength_input);
  const Float3 color = luisa::compute::max(
      xyz_to_rgb(kernel_globals,
                 cycles_color_nodes::wavelength_xyz(wavelength)) *
          (1.0f / 2.52f),
      make_float3(0.0f));
  stack_store_float3(stack, color_offset, color);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

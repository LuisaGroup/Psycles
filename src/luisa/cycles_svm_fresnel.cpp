/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"
#include "surface_fresnel.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_fresnel(Cursor &cursor, Stack &stack,
                  const ShaderData &shader_data) noexcept {
  Float eta = stack_load_input_float(stack, cursor.word());
  const auto packed = cursor.word();
  const auto normal_offset = cursor.byte(packed, 0u);
  const auto output_offset = cursor.byte(packed, 1u);
  const auto normal =
      stack_load_float3_default(stack, normal_offset, shader_data.N);

  eta = max(eta, 1.0e-5f);
  eta = select(eta, 1.0f / eta,
               (shader_data.flag & shader_data_backfacing) != 0u);
  const auto factor = psycles::luisa_backend::detail::fresnel_dielectric_cos(
      dot(shader_data.wi, normal), eta);
  stack_store_float(stack, output_offset, factor);
}

void node_layer_weight(Cursor &cursor, Stack &stack,
                       const ShaderData &shader_data) noexcept {
  const auto weight_type = cursor.word();
  Float blend = stack_load_input_float(stack, cursor.word());
  const auto packed = cursor.word();
  const auto normal_offset = cursor.byte(packed, 0u);
  const auto output_offset = cursor.byte(packed, 1u);
  const auto normal =
      stack_load_float3_default(stack, normal_offset, shader_data.N);

  Float factor;
  $if(weight_type == static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FRESNEL)) {
    Float eta = max(1.0f - blend, 1.0e-5f);
    eta = select(1.0f / eta, eta,
                 (shader_data.flag & shader_data_backfacing) != 0u);
    factor = psycles::luisa_backend::detail::fresnel_dielectric_cos(
        dot(shader_data.wi, normal), eta);
  }
  $else {
    factor = abs(dot(shader_data.wi, normal));
    $if(blend != 0.5f) {
      blend = clamp(blend, 0.0f, 1.0f - 1.0e-5f);
      blend = select(0.5f / (1.0f - blend), 2.0f * blend, blend < 0.5f);
      factor = pow(factor, blend);
    };
    factor = 1.0f - factor;
  };
  stack_store_float(stack, output_offset, factor);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

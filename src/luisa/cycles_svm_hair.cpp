/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_hair.h"

#include "cycles_svm_microfacet.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_hair(Cursor &cursor, Stack &stack, Expr<std::uint32_t> type,
               Expr<luisa::float3> closure_weight, Expr<float> mix_weight,
               ShaderData &shader_data) noexcept {
  /* svm_node_get consumes the complete typed payload before bsdf_alloc. This
   * is observable when the weight is cut off or the closure pool is full. */
  const auto roughness1_input = cursor.word();
  const auto roughness2_input = cursor.word();
  const auto offset_input = cursor.word();
  const auto tangent_packed = cursor.word();
  const auto tangent_offset = cursor.byte(tangent_packed, 0u);

  const auto allocated =
      bsdf_allocate(shader_data, closure_weight * mix_weight);
  $if(allocated.valid) {
    auto &pool = *shader_data.closure;
    const auto normal =
        maybe_ensure_valid_specular_reflection(shader_data, shader_data.N);
    const auto roughness1 = stack_load_input_float(stack, roughness1_input);
    const auto roughness2 = stack_load_input_float(stack, roughness2_input);
    Float offset = -stack_load_input_float(stack, offset_input);
    Float3 tangent;

    $if(tangent_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      tangent = normalize_cycles(stack_load_float3(stack, tangent_offset));
    }
    $elif((shader_data.type & primitive_curve) == 0u) {
      tangent = normalize_cycles(shader_data.dPdv);
      offset = 0.0f;
    }
    $else { tangent = normalize_cycles(shader_data.dPdu); };

    pool.set_type(allocated.index, type);
    pool.set_normal(allocated.index, normal);
    pool.set_hair_param(allocated.index,
                        {.T = tangent,
                         .roughness1 = clamp(roughness1, 0.001f, 1.0f),
                         .roughness2 = clamp(roughness2, 0.001f, 1.0f),
                         .offset = offset});
    shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
    $if(type == static_cast<std::uint32_t>(CLOSURE_BSDF_HAIR_TRANSMISSION_ID)) {
      shader_data.flag |= shader_data_bsdf_has_transmission;
    };
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

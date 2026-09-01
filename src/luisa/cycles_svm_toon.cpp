/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_toon.h"

#include "cycles_svm_microfacet.h"

#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

inline constexpr float half_pi = 1.57079632679489661923f;

} // namespace

void node_toon(const KernelGlobals &kernel_globals, Cursor &cursor,
               Stack &stack, Expr<std::uint32_t> type,
               Expr<luisa::float3> closure_weight, Expr<float> mix_weight,
               ShaderData &shader_data, const PathState &path_state) noexcept {
  /* svm_node_get advances over the complete typed record before Cycles tests
   * the Glossy-caustics branch. Preserve that cursor transition even when no
   * closure is allocated. */
  const auto size_input = cursor.word();
  const auto smooth_input = cursor.word();
  const auto normal_packed = cursor.word();

  const Bool glossy =
      type == static_cast<std::uint32_t>(CLOSURE_BSDF_GLOSSY_TOON_ID);
  const Bool diffuse_visibility =
      (path_state.visibility & path_ray_visibility_diffuse) != 0u;
  const Bool active =
      !glossy | kernel_globals.caustics_reflective() | !diffuse_visibility;

  $if(active) {
    const auto normal_offset = cursor.byte(normal_packed, 0u);
    auto normal =
        stack_load_float3_default(stack, normal_offset, shader_data.N);
    normal =
        native_vector_math::safe_normalize_nonzero_or(normal, shader_data.N);

    auto &pool = *shader_data.closure;
    const auto allocated =
        bsdf_allocate(shader_data, closure_weight * mix_weight);
    $if(allocated.valid) {
      /* bsdf_toon_setup_common performs these maps after allocation, for both
       * closure tags and in this order. */
      const auto size =
          clamp(stack_load_input_float(stack, size_input), 1.0e-5f, 1.0f) *
          half_pi;
      const auto smooth =
          clamp(stack_load_input_float(stack, smooth_input), 0.0f, 1.0f) *
          half_pi;
      pool.set_normal(allocated.index, normal);
      pool.set_toon_param(allocated.index, {.size = size, .smooth = smooth});
      pool.set_type(allocated.index, type);
      shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
    };
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

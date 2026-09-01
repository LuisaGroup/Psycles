/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_sheen.h"

#include "cycles_svm_microfacet.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_sample_mapping.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;
namespace table_detail = ::psycles::luisa_backend::detail;

namespace {

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

} // namespace

Float3 principled_sheen_setup(const KernelGlobals &kernel_globals,
                              ShaderData &shader_data,
                              const PathState &path_state,
                              Expr<luisa::float3> input_weight,
                              Expr<luisa::float3> normal,
                              Expr<float> roughness) noexcept {
  auto &pool = *shader_data.closure;
  const Float3 weight = max(input_weight, make_float3(0.0f));
  const Float sample_weight = abs(average(weight));
  const Bool survives_cutoff =
      ((sample_weight >= CLOSURE_WEIGHT_CUTOFF) |
       ((shader_data.flag & shader_data_is_volume_shader_eval) != 0u)) &
      (sample_weight > 0.0f);
  const Bool emission_path = (path_state.flag & path_ray_emission) != 0u;

  ClosurePool::Allocation allocated{.index = 0u, .valid = false};
  Bool setup_active = false;
  $if(emission_path) { setup_active = survives_cutoff; }
  $else {
    allocated = bsdf_allocate(shader_data, weight);
    setup_active = allocated.valid;
  };

  Float3 layer_albedo = make_float3(0.0f);
  $if(setup_active) {
    const auto canonical_roughness = clamp(roughness, 1.0e-3f, 1.0f);
    const auto basis = cycles_sample_mapping::make_orthonormals_safe_tangent(
        normal, shader_data.wi);
    const auto cosine = dot(normal, shader_data.wi);
    const auto transform_a = table_detail::cycles_table_2d(
        kernel_globals, cosine, canonical_roughness,
        UInt{cycles45_tables::sheen_ltc_offset}, 32u, 32u);
    const auto transform_b = table_detail::cycles_table_2d(
        kernel_globals, cosine, canonical_roughness,
        UInt{cycles45_tables::sheen_ltc_offset + 32u * 32u}, 32u, 32u);
    const auto albedo = table_detail::cycles_table_2d(
        kernel_globals, cosine, canonical_roughness,
        UInt{cycles45_tables::sheen_ltc_offset + 2u * 32u * 32u}, 32u, 32u);
    const SheenParam param{.roughness = canonical_roughness,
                           .transform_a = transform_a,
                           .transform_b = transform_b,
                           .T = basis.tangent,
                           .B = basis.bitangent};

    $if(!emission_path) {
      pool.set_normal(allocated.index, normal);
      pool.set_sheen_param(allocated.index, param);
    };

    /* Preserve Cycles' comparison form. With a malformed NaN table entry,
     * neither less-than predicate fires; changing this to >= would define a
     * different state transition. Scene tables themselves are finite. */
    const Bool invalid = (abs(transform_a) < 1.0e-5f) | (albedo < 1.0e-5f);
    $if(invalid) {
      $if(!emission_path) {
        /* bsdf_alloc already wrote CLOSURE_NONE. Sheen setup clears only the
         * sample weight and leaves the consumed common slot observable. */
        pool.set_sample_weight(allocated.index, 0.0f);
      };
    }
    $else {
      layer_albedo = weight * albedo;
      $if(!emission_path) {
        pool.set_type(allocated.index,
                      static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID));
        pool.set_weight(allocated.index, layer_albedo);
        pool.set_sample_weight(allocated.index, sample_weight * albedo);
      };
      shader_data.flag |= shader_data_bsdf | shader_data_bsdf_has_eval;
    };
  };
  return layer_albedo;
}

} // namespace psycles::luisa_backend::cycles_svm::detail

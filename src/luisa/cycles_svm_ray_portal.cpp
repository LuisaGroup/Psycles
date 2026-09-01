/* SPDX-FileCopyrightText: 2023 Blender Authors
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_ray_portal.h"

#include <psycles/luisa/native_vector_math.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Bool is_zero(Expr<luisa::float3> value) noexcept {
  return (value.x == 0.0f) & (value.y == 0.0f) & (value.z == 0.0f);
}

} // namespace

void node_ray_portal(Cursor &cursor, Stack &stack,
                     Expr<luisa::float3> closure_weight, Expr<float> mix_weight,
                     ShaderData &shader_data) noexcept {
  const auto direction_x = cursor.word();
  const auto direction_y = cursor.word();
  const auto direction_z = cursor.word();
  const auto position_packed = cursor.word();
  const auto position_offset = cursor.byte(position_packed, 0u);

  const Float3 weight = closure_weight * mix_weight;
  const Float3 position =
      stack_load_float3_default(stack, position_offset, shader_data.P);
  Float3 direction =
      stack_load_input_float3(stack, direction_x, direction_y, direction_z);
  const Float sample_weight = abs(average(weight));

  /* bsdf_ray_portal_setup uses direct closure_alloc rather than bsdf_alloc:
   * signed weights remain intact, and extinction changes before a possible
   * capacity failure. Keep the source comparison form so NaN is rejected. */
  $if(sample_weight >= CLOSURE_WEIGHT_CUTOFF) {
    shader_data.closure_transparent_extinction += weight;
    auto &pool = *shader_data.closure;
    const auto allocated = pool.allocate(
        static_cast<std::uint32_t>(CLOSURE_BSDF_RAY_PORTAL_ID), weight);
    $if(allocated.valid) {
      $if(is_zero(direction)) { direction = -shader_data.wi; };
      pool.set_sample_weight(allocated.index, sample_weight);
      pool.set_normal(allocated.index, shader_data.N);
      pool.set_ray_portal_param(
          allocated.index,
          {.P = position,
           .D = native_vector_math::safe_normalize_nonzero(direction)});
      shader_data.flag |= shader_data_bsdf | shader_data_ray_portal;
    };
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

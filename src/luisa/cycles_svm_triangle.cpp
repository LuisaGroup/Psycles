/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;

namespace {

[[nodiscard]] Int normal_index(Expr<std::uint32_t> index) noexcept {
  return index.cast<std::int32_t>();
}

[[nodiscard]] Bool is_zero(Expr<luisa::float3> value) noexcept {
  return (value.x == 0.0f) & (value.y == 0.0f) & (value.z == 0.0f);
}

} // namespace

Dual3 shading_position_dual(const ShaderData &shader_data) noexcept {
  return {.val = shader_data.P,
          .dx = shader_data.dPdu * shader_data.du.dx +
                shader_data.dPdv * shader_data.dv.dx,
          .dy = shader_data.dPdu * shader_data.du.dy +
                shader_data.dPdv * shader_data.dv.dy};
}

TriangleNormals triangle_normals(
    const KernelGlobals &kernel_globals,
    const ShaderData &shader_data) noexcept {
  UInt i0;
  UInt i1;
  UInt i2;
  $if((shader_data.object_flag &
       shader_data_object_has_corner_normals) != 0u) {
    i0 = shader_data.prim * 3u;
    i1 = i0 + 1u;
    i2 = i0 + 2u;
  }
  $else {
    const auto indices =
        kernel_globals.triangle_vertex_indices(shader_data.prim);
    i0 = indices.x;
    i1 = indices.y;
    i2 = indices.z;
  };
  const auto offset =
      kernel_globals.object_normal_offset(shader_data.object);
  return {
      .n0 = decode_packed_normal(
          kernel_globals.attribute_normal(offset + normal_index(i0))),
      .n1 = decode_packed_normal(
          kernel_globals.attribute_normal(offset + normal_index(i1))),
      .n2 = decode_packed_normal(
          kernel_globals.attribute_normal(offset + normal_index(i2)))};
}

Float3 triangle_smooth_normal_unnormalized_object_space(
    const KernelGlobals &kernel_globals,
    const TransformState &transform_state,
    const ShaderData &shader_data,
    bool object_motion_enabled) noexcept {
  auto normals = triangle_normals(kernel_globals, shader_data);
  $if((shader_data.object_flag &
       shader_data_object_transform_applied) != 0u) {
    object_inverse_normal_transform(normals.n0, transform_state, shader_data,
                                    object_motion_enabled);
    object_inverse_normal_transform(normals.n1, transform_state, shader_data,
                                    object_motion_enabled);
    object_inverse_normal_transform(normals.n2, transform_state, shader_data,
                                    object_motion_enabled);
  };
  const auto interpolated =
      safe_normalize_cycles((1.0f - shader_data.u - shader_data.v) * normals.n0 +
                            shader_data.u * normals.n1 +
                            shader_data.v * normals.n2);
  return select(interpolated, shader_data.Ng, is_zero(interpolated));
}

} // namespace psycles::luisa_backend::cycles_svm::detail

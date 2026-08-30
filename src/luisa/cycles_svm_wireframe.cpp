/* SPDX-FileCopyrightText: 2009-2010 Sony Pictures Imageworks Inc., et al. All
 * Rights Reserved. SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: BSD-3-Clause */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

Differential3 differential_from_compact(Expr<luisa::float3> direction,
                                        Expr<float> differential) noexcept {
  Float3 orthogonal;
  $if((direction.x != direction.y) | (direction.x != direction.z)) {
    orthogonal =
        make_float3(direction.z - direction.y, direction.x - direction.z,
                    direction.y - direction.x);
  }
  $else {
    orthogonal =
        make_float3(direction.z - direction.y, direction.x + direction.z,
                    -direction.y - direction.x);
  };
  orthogonal = normalize_exact(orthogonal);
  return {.dx = differential * orthogonal,
          .dy = differential * cross(direction, orthogonal)};
}

namespace {

[[nodiscard]] Float wireframe(
    const KernelGlobals &kernel_globals, const TransformState &transform_state,
    const ShaderData &shader_data, const Differential3 &dP, Expr<float> size,
    Expr<std::uint32_t> use_pixel_size, Expr<luisa::float3> position,
    bool require_triangle_primitive, bool object_motion_enabled) noexcept {
  Bool valid_primitive = shader_data.prim != primitive_none;
  if (require_triangle_primitive) {
    valid_primitive &= (shader_data.type & primitive_triangle) != 0u;
  }

  Float result = 0.0f;
  $if(valid_primitive) {
    Float3 co0;
    Float3 co1;
    Float3 co2;
    $if((shader_data.type & primitive_motion) != 0u) {
      const auto vertices = kernel_globals.motion_triangle_vertices(
          shader_data.object, shader_data.prim, shader_data.time);
      co0 = vertices.v0;
      co1 = vertices.v1;
      co2 = vertices.v2;
    }
    $else {
      const auto vertices = kernel_globals.triangle_vertices(shader_data.object,
                                                             shader_data.prim);
      co0 = vertices.v0;
      co1 = vertices.v1;
      co2 = vertices.v2;
    };

    $if((shader_data.object_flag & shader_data_object_transform_applied) ==
        0u) {
      object_position_transform(co0, transform_state, shader_data,
                                object_motion_enabled);
      object_position_transform(co1, transform_state, shader_data,
                                object_motion_enabled);
      object_position_transform(co2, transform_state, shader_data,
                                object_motion_enabled);
    };

    Float pixelwidth = 1.0f;
    $if(use_pixel_size != 0u) {
      const auto projected_x =
          dP.dx - dot(dP.dx, shader_data.wi) * shader_data.wi;
      const auto projected_y =
          dP.dy - dot(dP.dy, shader_data.wi) * shader_data.wi;
      const auto pixelwidth_x = sqrt(dot(projected_x, projected_x));
      const auto pixelwidth_y = sqrt(dot(projected_y, projected_y));
      pixelwidth = (pixelwidth_x + pixelwidth_y) * 0.5f;
    };

    pixelwidth *= 0.5f * size;
    pixelwidth *= pixelwidth;
    const auto test_edge = [&](Expr<luisa::float3> current,
                               Expr<luisa::float3> previous) noexcept {
      $if(result == 0.0f) {
        const auto direction = position - current;
        const auto edge = current - previous;
        const auto area = cross(edge, direction);
        $if(dot(area, area) < dot(edge, edge) * pixelwidth) { result = 1.0f; };
      };
    };
    test_edge(co0, co2);
    test_edge(co1, co0);
    test_edge(co2, co1);
  };
  return result;
}

} // namespace

void node_wireframe(Cursor &cursor, Stack &stack,
                    const KernelGlobals &kernel_globals,
                    const TransformState &transform_state,
                    const ShaderData &shader_data,
                    bool require_triangle_primitive,
                    bool object_motion_enabled) noexcept {
  const auto size = stack_load_input_float(stack, cursor.word());
  const auto bump_filter_width = cursor.floating();
  const auto packed = cursor.word();
  const auto use_pixel_size = cursor.byte(packed, 0u);
  const auto bump_offset = cursor.byte(packed, 1u);
  const auto output_offset = cursor.byte(packed, 2u);

  const auto dP = differential_from_compact(shader_data.Ng, shader_data.dP);
  Float3 position = shader_data.P;
  $if(bump_offset == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
    position += dP.dx * bump_filter_width;
  }
  $elif(bump_offset == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
    position += dP.dy * bump_filter_width;
  };

  const auto factor = wireframe(
      kernel_globals, transform_state, shader_data, dP, size, use_pixel_size,
      position, require_triangle_primitive, object_motion_enabled);
  $if(output_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, output_offset, factor);
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_transform.h>

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) \
  $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

[[nodiscard]] Bool is_zero(Expr<luisa::float3> value) noexcept {
  return (value.x == 0.0f) & (value.y == 0.0f) & (value.z == 0.0f);
}

[[nodiscard]] Float3 safe_normalize_or(
    Expr<luisa::float3> value, Expr<luisa::float3> fallback) noexcept {
  const auto normalized = safe_normalize_cycles(value);
  return select(normalized, fallback, is_zero(normalized));
}

/* Cycles math_dual.h: safe_normalize(a) =
 * is_zero(len_squared(a)) ? 0 : a * inversesqrt(len_squared(a)). The guarded
 * denominator is semantic (zero duals stay zero), not a last-bit emulation. */
[[nodiscard]] Dual3 safe_normalize_dual(const Dual3 &value) noexcept {
  const Float length_squared = dot(value.val, value.val);
  const Bool nonzero = length_squared != 0.0f;
  const Float safe_length_squared = select(1.0f, length_squared, nonzero);
  const Float inverse_length =
      select(0.0f, rsqrt(safe_length_squared), nonzero);
  const Float inverse_derivative =
      -0.5f * inverse_length / safe_length_squared;
  const Float inverse_dx =
      (2.0f * dot(value.val, value.dx)) * inverse_derivative;
  const Float inverse_dy =
      (2.0f * dot(value.val, value.dy)) * inverse_derivative;
  return {.val = value.val * inverse_length,
          .dx = value.val * inverse_dx + value.dx * inverse_length,
          .dy = value.val * inverse_dy + value.dy * inverse_length};
}

[[nodiscard]] Float4x4 transform_from_rows(Expr<luisa::float4> x,
                                           Expr<luisa::float4> y,
                                           Expr<luisa::float4> z) noexcept {
  return make_float4x4(make_float4(x.x, y.x, z.x, 0.0f),
                       make_float4(x.y, y.y, z.y, 0.0f),
                       make_float4(x.z, y.z, z.z, 0.0f),
                       make_float4(x.w, y.w, z.w, 1.0f));
}

[[nodiscard]] Float4x4 packed_transform(Cursor &cursor) noexcept {
  // Cursor reads are state transitions. C++ does not specify an evaluation
  // order for function arguments, so each word must be consumed by a separate
  // full-expression to preserve Cycles' program-counter order.
  const auto x0 = cursor.floating();
  const auto x1 = cursor.floating();
  const auto x2 = cursor.floating();
  const auto x3 = cursor.floating();
  const auto y0 = cursor.floating();
  const auto y1 = cursor.floating();
  const auto y2 = cursor.floating();
  const auto y3 = cursor.floating();
  const auto z0 = cursor.floating();
  const auto z1 = cursor.floating();
  const auto z2 = cursor.floating();
  const auto z3 = cursor.floating();
  const auto x = make_float4(x0, x1, x2, x3);
  const auto y = make_float4(y0, y1, y2, y3);
  const auto z = make_float4(z0, z1, z2, z3);
  return transform_from_rows(x, y, z);
}

[[nodiscard]] Dual3 shading_position(const ShaderData &shader_data) noexcept {
  return {.val = shader_data.P,
          .dx = shader_data.dPdu * shader_data.du.dx +
                shader_data.dPdv * shader_data.dv.dx,
          .dy = shader_data.dPdu * shader_data.du.dy +
                shader_data.dPdv * shader_data.dv.dy};
}

[[nodiscard]] Dual3 shading_incoming(const ShaderData &shader_data) noexcept {
  const auto differential =
      differential_from_compact(shader_data.wi, shader_data.dI);
  return {.val = shader_data.wi,
          .dx = differential.dx,
          .dy = differential.dy};
}

[[nodiscard]] Dual3 transform_point(Expr<luisa::float4x4> transform,
                                    const Dual3 &value) noexcept {
  return {.val = cycles_transform::point(transform, value.val),
          .dx = cycles_transform::direction(transform, value.dx),
          .dy = cycles_transform::direction(transform, value.dy)};
}

[[nodiscard]] Float3 camera_position(
    const TransformState &transform_state) noexcept {
  return transform_state.camera_to_world[3u].xyz();
}

[[nodiscard]] Float3 texco_camera(const TransformState &transform_state,
                                  const ShaderData &shader_data,
                                  Expr<luisa::float3> position) noexcept {
  Float3 data = position;
  $if(shader_data.object == object_none) {
    data += camera_position(transform_state);
  };
  return cycles_transform::point(transform_state.world_to_camera, data);
}

[[nodiscard]] Dual3 texco_camera(const TransformState &transform_state,
                                 const ShaderData &shader_data,
                                 const Dual3 &position) noexcept {
  Dual3 data{.val = position.val, .dx = position.dx, .dy = position.dy};
  $if(shader_data.object == object_none) {
    data.val += camera_position(transform_state);
  };
  return transform_point(transform_state.world_to_camera, data);
}

[[nodiscard]] Float3 texco_reflection(
    const ShaderData &shader_data) noexcept {
  Float3 data = shader_data.wi;
  $if(shader_data.object != object_none) {
    data = -reflect(data, shader_data.N);
  };
  return data;
}

[[nodiscard]] Dual3 texco_reflection_derivative(
    const ShaderData &shader_data) noexcept {
  auto data = shading_incoming(shader_data);
  $if(shader_data.object != object_none) {
    data.val = -reflect(data.val, shader_data.N);
    data.dx = -reflect(data.dx, shader_data.N);
    data.dy = -reflect(data.dy, shader_data.N);
  };
  return data;
}

[[nodiscard]] Int normal_index(Expr<std::uint32_t> index) noexcept {
  return index.cast<std::int32_t>();
}

[[nodiscard]] TriangleNormals triangle_normals(
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

[[nodiscard]] Int motion_normal_step_offset(
    const KernelGlobals &kernel_globals,
    const ShaderData &shader_data,
    Expr<std::int32_t> step,
    Expr<std::int32_t> num_steps) noexcept {
  Int offset = kernel_globals.object_normal_offset(shader_data.object);
  const Int center_step = (num_steps - 1) / 2;
  $if(step != center_step) {
    Int storage_step = step;
    $if(storage_step < center_step) { storage_step += 1; };
    const Int stride = select(
        kernel_globals.object_num_vertices(shader_data.object),
        kernel_globals.object_num_primitives(shader_data.object) * 3,
        (shader_data.object_flag &
         shader_data_object_has_corner_normals) != 0u);
    offset += storage_step * stride;
  };
  return offset;
}

[[nodiscard]] TriangleNormals motion_normals_at_step(
    const KernelGlobals &kernel_globals,
    const ShaderData &shader_data,
    Expr<std::int32_t> step,
    Expr<std::int32_t> num_steps) noexcept {
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
  const auto offset = motion_normal_step_offset(
      kernel_globals, shader_data, step, num_steps);
  return {
      .n0 = decode_packed_normal(
          kernel_globals.attribute_normal(offset + normal_index(i0))),
      .n1 = decode_packed_normal(
          kernel_globals.attribute_normal(offset + normal_index(i1))),
      .n2 = decode_packed_normal(
          kernel_globals.attribute_normal(offset + normal_index(i2)))};
}

[[nodiscard]] TriangleNormals motion_triangle_normals(
    const KernelGlobals &kernel_globals,
    const ShaderData &shader_data) noexcept {
  const Int num_steps = kernel_globals
                            .object_num_geom_steps(shader_data.object)
                            .cast<std::int32_t>();
  const Int max_step = num_steps - 1;
  const Int step = min((shader_data.time * max_step.cast<float>())
                           .cast<std::int32_t>(),
                       max_step - 1);
  const Float t = shader_data.time * max_step.cast<float>() -
                  step.cast<float>();
  const auto current =
      motion_normals_at_step(kernel_globals, shader_data, step, num_steps);
  const auto next =
      motion_normals_at_step(kernel_globals, shader_data, step + 1, num_steps);
  return {
      .n0 = normalize_cycles((1.0f - t) * current.n0 + t * next.n0),
      .n1 = normalize_cycles((1.0f - t) * current.n1 + t * next.n1),
      .n2 = normalize_cycles((1.0f - t) * current.n2 + t * next.n2)};
}

[[nodiscard]] Float3 interpolate_normal(
    const TriangleNormals &normal, Expr<float> u, Expr<float> v,
    Expr<luisa::float3> fallback) noexcept {
  return safe_normalize_or((1.0f - u - v) * normal.n0 + u * normal.n1 +
                               v * normal.n2,
                           fallback);
}

[[nodiscard]] Dual3 smooth_normal(
    const KernelGlobals &kernel_globals,
    const TransformState &transform_state,
    const ShaderData &shader_data,
    bool object_motion_enabled) noexcept {
  Dual3 result{.val = shader_data.N,
               .dx = make_float3(0.0f),
               .dy = make_float3(0.0f)};
  const Bool smooth_triangle =
      ((shader_data.type & primitive_triangle) != 0u) &
      ((shader_data.shader & shader_smooth_normal) != 0u);
  $if(smooth_triangle) {
    TriangleNormals normal;
    $if(shader_data.type == primitive_triangle) {
      normal = triangle_normals(kernel_globals, shader_data);
    }
    $else {
      normal = motion_triangle_normals(kernel_globals, shader_data);
    };
    auto center = interpolate_normal(normal, shader_data.u, shader_data.v,
                                     shader_data.Ng);
    auto x = interpolate_normal(normal, shader_data.u + shader_data.du.dx,
                                shader_data.v + shader_data.dv.dx,
                                shader_data.Ng);
    auto y = interpolate_normal(normal, shader_data.u + shader_data.du.dy,
                                shader_data.v + shader_data.dv.dy,
                                shader_data.Ng);
    $if((shader_data.flag & shader_data_backfacing) != 0u) {
      center = -center;
      x = -x;
      y = -y;
    };
    $if((shader_data.object_flag &
         shader_data_object_transform_applied) != 0u) {
      object_inverse_normal_transform(center, transform_state, shader_data,
                                      object_motion_enabled);
      object_inverse_normal_transform(x, transform_state, shader_data,
                                      object_motion_enabled);
      object_inverse_normal_transform(y, transform_state, shader_data,
                                      object_motion_enabled);
    };
    result.val = center;
    result.dx = x - center;
    result.dy = y - center;
  }
  $else {
    object_inverse_normal_transform(result.val, transform_state, shader_data,
                                    object_motion_enabled);
  };
  return result;
}

[[nodiscard]] Float4x4 generated_transform(
    const KernelGlobals &kernel_globals,
    const AttributeDescriptor &descriptor) noexcept {
  const auto x = kernel_globals.attribute_float4(descriptor.offset);
  const auto y = kernel_globals.attribute_float4(descriptor.offset + 1);
  const auto z = kernel_globals.attribute_float4(descriptor.offset + 2);
  return transform_from_rows(x, y, z);
}

[[nodiscard]] Float3 volume_normalized_position(
    const KernelGlobals &kernel_globals,
    const ShaderData &shader_data,
    Expr<luisa::float3> position) noexcept {
  const auto descriptor = find_attribute(
      kernel_globals, shader_data,
      static_cast<luisa::ulong>(ATTR_STD_GENERATED_TRANSFORM));
  Float3 result = kernel_globals.object_inverse_position_transform_if_object(
      shader_data, position);
  $if(descriptor.offset !=
      static_cast<std::int32_t>(ATTR_STD_NOT_FOUND)) {
    result = cycles_transform::point(
        generated_transform(kernel_globals, descriptor), result);
  };
  return result;
}

[[nodiscard]] Dual3 volume_normalized_position(
    const KernelGlobals &kernel_globals,
    const ShaderData &shader_data,
    const Dual3 &position) noexcept {
  const auto descriptor = find_attribute(
      kernel_globals, shader_data,
      static_cast<luisa::ulong>(ATTR_STD_GENERATED_TRANSFORM));
  auto result =
      kernel_globals.object_inverse_position_transform_if_object_derivative(
          shader_data, position);
  $if(descriptor.offset !=
      static_cast<std::int32_t>(ATTR_STD_NOT_FOUND)) {
    result = transform_point(generated_transform(kernel_globals, descriptor),
                             result);
  };
  return result;
}

[[nodiscard]] Float3 dupli_generated(
    const KernelGlobals &kernel_globals,
    const ShaderData &shader_data) noexcept {
  Float3 result = make_float3(0.0f);
  $if(shader_data.object != object_none) {
    result = kernel_globals.object_dupli_generated(shader_data.object);
  };
  return result;
}

[[nodiscard]] Float3 dupli_uv(const KernelGlobals &kernel_globals,
                              const ShaderData &shader_data) noexcept {
  Float3 result = make_float3(0.0f);
  $if(shader_data.object != object_none) {
    result = kernel_globals.object_dupli_uv(shader_data.object);
  };
  return result;
}

[[nodiscard]] Float3 evaluate_plain(
    Cursor &cursor, Expr<std::uint32_t> type,
    const KernelGlobals &kernel_globals,
    const TransformState &transform_state,
    const ShaderData &shader_data,
    const PathState &path_state,
    bool volume_enabled,
    bool object_motion_enabled) noexcept {
  Float3 data = make_float3(0.0f);
  $switch(type) {
    PSYCLES_SVM_CASE(NODE_TEXCO_OBJECT) {
      data = kernel_globals.object_inverse_position_transform_if_object(
          shader_data, shader_data.P);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_OBJECT_WITH_TRANSFORM) {
      data = cycles_transform::point(packed_transform(cursor), shader_data.P);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_NORMAL) {
      data = shader_data.N;
      object_inverse_normal_transform(data, transform_state, shader_data,
                                      object_motion_enabled);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_CAMERA) {
      data = texco_camera(transform_state, shader_data, shader_data.P);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_WINDOW) {
      const Bool orthographic_background_camera =
          ((path_state.visibility & path_ray_visibility_camera) != 0u) &
          (shader_data.object == object_none) &
          (kernel_globals.camera_type() == camera_orthographic);
      data = kernel_globals.camera_world_to_ndc(
          shader_data,
          select(shader_data.P, shader_data.ray_P,
                 orthographic_background_camera));
      data.z = 0.0f;
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_REFLECTION) {
      data = texco_reflection(shader_data);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_DUPLI_GENERATED) {
      data = dupli_generated(kernel_globals, shader_data);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_DUPLI_UV) {
      data = dupli_uv(kernel_globals, shader_data);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_VOLUME_GENERATED) {
      data = shader_data.P;
      if (volume_enabled) {
        $if(shader_data.object != object_none) {
          data = volume_normalized_position(kernel_globals, shader_data, data);
        };
      }
    };
    $default { data = make_float3(0.0f); };
  };
  return data;
}

[[nodiscard]] Dual3 evaluate_derivative(
    Cursor &cursor, Expr<std::uint32_t> type,
    const KernelGlobals &kernel_globals,
    const TransformState &transform_state,
    const ShaderData &shader_data,
    const PathState &path_state,
    bool volume_enabled,
    bool object_motion_enabled) noexcept {
  Dual3 data{.val = make_float3(0.0f),
             .dx = make_float3(0.0f),
             .dy = make_float3(0.0f)};
  $switch(type) {
    PSYCLES_SVM_CASE(NODE_TEXCO_OBJECT) {
      data =
          kernel_globals.object_inverse_position_transform_if_object_derivative(
              shader_data, shading_position(shader_data));
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_OBJECT_WITH_TRANSFORM) {
      data = transform_point(packed_transform(cursor),
                             shading_position(shader_data));
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_NORMAL) {
      data = smooth_normal(kernel_globals, transform_state, shader_data,
                           object_motion_enabled);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_CAMERA) {
      data = texco_camera(transform_state, shader_data,
                          shading_position(shader_data));
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_WINDOW) {
      const Bool orthographic_background_camera =
          ((path_state.visibility & path_ray_visibility_camera) != 0u) &
          (shader_data.object == object_none) &
          (kernel_globals.camera_type() == camera_orthographic);
      $if(orthographic_background_camera) {
        data.val = kernel_globals.camera_world_to_ndc(shader_data,
                                                      shader_data.ray_P);
      }
      $else {
        data.val = kernel_globals.camera_world_to_ndc(shader_data,
                                                      shader_data.P);
        data.dx = make_float3(1.0f / kernel_globals.camera_width(), 0.0f,
                             0.0f);
        data.dy = make_float3(0.0f, 1.0f / kernel_globals.camera_height(),
                             0.0f);
      };
      data.val.z = 0.0f;
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_REFLECTION) {
      data = texco_reflection_derivative(shader_data);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_DUPLI_GENERATED) {
      data.val = dupli_generated(kernel_globals, shader_data);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_DUPLI_UV) {
      data.val = dupli_uv(kernel_globals, shader_data);
    };
    PSYCLES_SVM_CASE(NODE_TEXCO_VOLUME_GENERATED) {
      data = shading_position(shader_data);
      if (volume_enabled) {
        $if(shader_data.object != object_none) {
          data = volume_normalized_position(kernel_globals, shader_data, data);
        };
      }
    };
    $default {
      data.val = make_float3(0.0f);
      data.dx = make_float3(0.0f);
      data.dy = make_float3(0.0f);
    };
  };
  return data;
}

} // namespace

void node_tex_coord(Cursor &cursor, Stack &stack,
                    const KernelGlobals &kernel_globals,
                    const TransformState &transform_state,
                    const ShaderData &shader_data,
                    const PathState &path_state,
                    bool use_derivatives,
                    bool volume_enabled,
                    bool object_motion_enabled) noexcept {
  const auto packed = cursor.word();
  const auto type = cursor.byte(packed, 0u);
  const auto bump_offset = cursor.byte(packed, 1u);
  const auto store_derivatives = cursor.byte(packed, 2u);
  const auto output_offset = cursor.byte(packed, 3u);
  const auto bump_filter_width = cursor.floating();

  if (use_derivatives) {
    auto data = evaluate_derivative(
        cursor, type, kernel_globals, transform_state, shader_data, path_state,
        volume_enabled, object_motion_enabled);
    $if(bump_offset == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
      data.val += data.dx * bump_filter_width;
    }
    $elif(bump_offset == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
      data.val += data.dy * bump_filter_width;
    };
    $if(type == static_cast<std::uint32_t>(NODE_TEXCO_NORMAL)) {
      data = safe_normalize_dual(data);
    };
    $if(store_derivatives != 0u) {
      stack_store_dual3(stack, output_offset, data);
    }
    $else { stack_store_float3(stack, output_offset, data.val); };
  } else {
    stack_store_float3(
        stack, output_offset,
        evaluate_plain(cursor, type, kernel_globals, transform_state,
                       shader_data, path_state, volume_enabled,
                       object_motion_enabled));
  }
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE

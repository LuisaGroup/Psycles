/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <bit>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

struct AttributeNode {
  Expr<std::int32_t> attr;
  Expr<std::uint32_t> out_offset;
  Expr<std::uint32_t> output_type;
  Expr<std::uint32_t> bump_offset;
  Expr<std::uint32_t> store_derivatives;
  Expr<float> bump_filter_width;
};

[[nodiscard]] AttributeNode read_attribute_node(Cursor &cursor) noexcept {
  const auto attr_w = cursor.word();
  const auto attr = [&]() noexcept -> Expr<std::int32_t> {
    if (const auto cw = try_const_uint(attr_w)) {
      return Expr<std::int32_t>{std::bit_cast<std::int32_t>(*cw)};
    }
    return attr_w.bitcast<std::int32_t>();
  }();
  const auto packed = cursor.word();
  return {.attr = attr,
          .out_offset = cursor.byte(packed, 0u),
          .output_type = cursor.byte(packed, 1u),
          .bump_offset = cursor.byte(packed, 2u),
          .store_derivatives = cursor.byte(packed, 3u),
          .bump_filter_width = cursor.floating()};
}

class AttributeScopedKernelGlobals final : public DelegatingKernelGlobals {
  SurfaceAttributeSummary _summary;

public:
  AttributeScopedKernelGlobals(const KernelGlobals &inner,
                               SurfaceAttributeSummary summary) noexcept
      : DelegatingKernelGlobals{inner}, _summary{summary} {}

  [[nodiscard]] SurfaceAttributeSummary
  surface_attribute_summary() const noexcept override {
    return _summary;
  }
};

[[nodiscard]] AttributeDescriptor attribute_not_found() noexcept {
  return {.element = static_cast<std::uint32_t>(ATTR_ELEMENT_NONE),
          .type = static_cast<std::uint32_t>(NODE_ATTR_FLOAT),
          .offset = static_cast<std::int32_t>(ATTR_STD_NOT_FOUND)};
}

[[nodiscard]] Bool
is_attribute_found(const AttributeDescriptor &descriptor) noexcept {
  return descriptor.offset !=
         static_cast<std::int32_t>(ATTR_STD_NOT_FOUND);
}

[[nodiscard]] AttributeDescriptor node_attr_init(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeNode &node, bool known_surface_absent = false) noexcept {
  auto descriptor = attribute_not_found();
  if (known_surface_absent) {
    descriptor.type = node.output_type;
    return descriptor;
  }
  const Expr<luisa::ulong> attr_id = [&]() noexcept -> Expr<luisa::ulong> {
    if (const auto ca = try_const_int(node.attr)) {
      return Expr<luisa::ulong>{
          static_cast<luisa::ulong>(static_cast<std::int64_t>(*ca))};
    }
    return node.attr.cast<luisa::ulong>();
  }();
  descriptor = find_attribute(kernel_globals, shader_data, attr_id);
  descriptor.type =
      select(descriptor.type, node.output_type, !is_attribute_found(descriptor));
  return descriptor;
}

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Dual1 average(const Dual3 &value) noexcept {
  return {.val = average(value.val),
          .dx = average(value.dx),
          .dy = average(value.dy)};
}

[[nodiscard]] Dual3 dual3_from_dual1(const Dual1 &value) noexcept {
  return {.val = make_float3(value.val),
          .dx = make_float3(value.dx),
          .dy = make_float3(value.dy)};
}

[[nodiscard]] Dual3 dual3_from_dual2(const Dual2 &value) noexcept {
  return {.val = make_float3(value.val, 0.0f),
          .dx = make_float3(value.dx, 0.0f),
          .dy = make_float3(value.dy, 0.0f)};
}

[[nodiscard]] Dual3 dual3_from_dual4(const Dual4 &value) noexcept {
  return {.val = value.val.xyz(),
          .dx = value.dx.xyz(),
          .dy = value.dy.xyz()};
}

[[nodiscard]] Dual3 dual3_from_dual4_alpha(const Dual4 &value) noexcept {
  return {.val = make_float3(value.val.w),
          .dx = make_float3(value.dx.w),
          .dy = make_float3(value.dy.w)};
}

[[nodiscard]] Dual3 dual3_one() noexcept {
  return {.val = make_float3(1.0f),
          .dx = make_float3(0.0f),
          .dy = make_float3(0.0f)};
}

[[nodiscard]] Dual3 shading_position(const ShaderData &shader_data) noexcept {
  return {
      .val = shader_data.P,
      .dx = shader_data.dPdu * shader_data.du.dx +
            shader_data.dPdv * shader_data.dv.dx,
      .dy = shader_data.dPdu * shader_data.du.dy +
            shader_data.dPdv * shader_data.dv.dy};
}

[[nodiscard]] Float3 convert_surface_float(Expr<std::uint32_t> output_type,
                                           Float value) noexcept {
  if (const auto cot = try_const_uint(output_type)) {
    return *cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)
               ? make_float3(1.0f)
               : make_float3(value);
  }
  return select(
      make_float3(value), make_float3(1.0f),
      output_type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA));
}

[[nodiscard]] Float3 convert_surface_float2(Expr<std::uint32_t> output_type,
                                            Float2 value) noexcept {
  if (const auto cot = try_const_uint(output_type)) {
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      return make_float3(value.x);
    }
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      return make_float3(1.0f);
    }
    return make_float3(value, 0.0f);
  }
  Float3 result = make_float3(value, 0.0f);
  $if(output_type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
    result = make_float3(value.x);
  }
  $elif(output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
    result = make_float3(1.0f);
  };
  return result;
}

[[nodiscard]] Float3 convert_surface_float3(Expr<std::uint32_t> output_type,
                                            Float3 value) noexcept {
  if (const auto cot = try_const_uint(output_type)) {
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      return make_float3(average(value));
    }
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      return make_float3(1.0f);
    }
    return value;
  }
  Float3 result = value;
  $if(output_type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
    result = make_float3(average(value));
  }
  $elif(output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
    result = make_float3(1.0f);
  };
  return result;
}

[[nodiscard]] Float3 convert_surface_float4(Expr<std::uint32_t> output_type,
                                            Float4 value) noexcept {
  if (const auto cot = try_const_uint(output_type)) {
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      return make_float3(average(value.xyz()));
    }
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      return make_float3(value.w);
    }
    return value.xyz();
  }
  Float3 result = value.xyz();
  $if(output_type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
    result = make_float3(average(value.xyz()));
  }
  $elif(output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
    result = make_float3(value.w);
  };
  return result;
}

[[nodiscard]] Dual3 convert_dual_float(Expr<std::uint32_t> output_type,
                                       const Dual1 &value) noexcept {
  if (const auto cot = try_const_uint(output_type)) {
    return *cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)
               ? dual3_one()
               : dual3_from_dual1(value);
  }
  Dual3 result = dual3_from_dual1(value);
  $if(output_type ==
      static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
    result = dual3_one();
  };
  return result;
}

[[nodiscard]] Dual3 convert_dual_float2(Expr<std::uint32_t> output_type,
                                        const Dual2 &value) noexcept {
  if (const auto cot = try_const_uint(output_type)) {
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      return {.val = make_float3(value.val.x),
              .dx = make_float3(value.dx.x),
              .dy = make_float3(value.dy.x)};
    }
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      return dual3_one();
    }
    return dual3_from_dual2(value);
  }
  Dual3 result = dual3_from_dual2(value);
  $if(output_type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
    result = {.val = make_float3(value.val.x),
              .dx = make_float3(value.dx.x),
              .dy = make_float3(value.dy.x)};
  }
  $elif(output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
    result = dual3_one();
  };
  return result;
}

[[nodiscard]] Dual3 convert_dual_float3(Expr<std::uint32_t> output_type,
                                        const Dual3 &value) noexcept {
  if (const auto cot = try_const_uint(output_type)) {
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      return dual3_from_dual1(average(value));
    }
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      return dual3_one();
    }
    return value;
  }
  Dual3 result = value;
  $if(output_type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
    result = dual3_from_dual1(average(value));
  }
  $elif(output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
    result = dual3_one();
  };
  return result;
}

[[nodiscard]] Dual3 convert_dual_float4(Expr<std::uint32_t> output_type,
                                        const Dual4 &value) noexcept {
  if (const auto cot = try_const_uint(output_type)) {
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      return dual3_from_dual1(average(dual3_from_dual4(value)));
    }
    if (*cot == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      return dual3_from_dual4_alpha(value);
    }
    return dual3_from_dual4(value);
  }
  Dual3 result = dual3_from_dual4(value);
  $if(output_type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
    result = dual3_from_dual1(average(dual3_from_dual4(value)));
  }
  $elif(output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
    result = dual3_from_dual4_alpha(value);
  };
  return result;
}

[[nodiscard]] Float3 node_attr_surface_eval(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeNode &node,
    const AttributeDescriptor &descriptor,
    bool known_surface_absent = false) noexcept {
  const auto summary = kernel_globals.surface_attribute_summary();
  const auto type_mask = summary.type_mask;
  const bool may_float = (type_mask & (1u << NODE_ATTR_FLOAT)) != 0u;
  const bool may_float2 = (type_mask & (1u << NODE_ATTR_FLOAT2)) != 0u;
  const bool may_float3 = (type_mask & (1u << NODE_ATTR_FLOAT3)) != 0u;
  const bool may_float4_or_rgba =
      (type_mask & ((1u << NODE_ATTR_FLOAT4) | (1u << NODE_ATTR_RGBA))) != 0u;
  const auto const_attr = try_const_int(node.attr);
  const auto const_out = try_const_uint(node.output_type);
  const bool may_be_uv =
      !const_attr.has_value() ||
      *const_attr == static_cast<std::int32_t>(ATTR_STD_UV);
  const bool may_be_generated =
      !const_attr.has_value() ||
      *const_attr == static_cast<std::int32_t>(ATTR_STD_GENERATED);

  if (known_surface_absent) {
    Float3 result = make_float3(0.0f);
    if (may_be_uv) {
      $if((shader_data.type == primitive_lamp) &
          (node.attr == static_cast<std::int32_t>(ATTR_STD_UV))) {
        result = make_float3(1.0f - shader_data.u - shader_data.v,
                             shader_data.u, 0.0f);
      }
      $else {
        if (may_be_generated) {
          result = kernel_globals.object_inverse_position_transform_if_object(
              shader_data, shader_data.P);
        } else if (const_out &&
                   *const_out == static_cast<std::uint32_t>(
                                     NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
          result = make_float3(1.0f);
        } else if (!const_out) {
          $if(node.output_type ==
              static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
            result = make_float3(1.0f);
          };
        }
      };
      return result;
    }
    if (may_be_generated) {
      return kernel_globals.object_inverse_position_transform_if_object(
          shader_data, shader_data.P);
    }
    if (const_out) {
      return *const_out ==
                     static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)
                 ? make_float3(1.0f)
                 : make_float3(0.0f);
    }
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = make_float3(1.0f);
    };
    return result;
  }

  const auto eval_found = [&]() noexcept {
    const unsigned family_count =
        (may_float ? 1u : 0u) + (may_float2 ? 1u : 0u) +
        (may_float4_or_rgba ? 1u : 0u) + (may_float3 ? 1u : 0u);
    if (family_count == 1u) {
      if (may_float) {
        return convert_surface_float(
            node.output_type, primitive_surface_attribute_float(
                                  kernel_globals, shader_data, descriptor));
      }
      if (may_float2) {
        return convert_surface_float2(
            node.output_type, primitive_surface_attribute_float2(
                                  kernel_globals, shader_data, descriptor));
      }
      if (may_float4_or_rgba) {
        return convert_surface_float4(
            node.output_type, primitive_surface_attribute_float4(
                                  kernel_globals, shader_data, descriptor));
      }
      return convert_surface_float3(
          node.output_type, primitive_surface_attribute_float3(
                                kernel_globals, shader_data, descriptor));
    }
    Float3 found_val = make_float3(0.0f);
    Bool handled = false;
    if (may_float) {
      $if(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT)) {
        found_val = convert_surface_float(
            node.output_type, primitive_surface_attribute_float(
                                  kernel_globals, shader_data, descriptor));
        handled = true;
      };
    }
    if (may_float2) {
      $if((!handled) &
          (descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT2))) {
        found_val = convert_surface_float2(
            node.output_type, primitive_surface_attribute_float2(
                                  kernel_globals, shader_data, descriptor));
        handled = true;
      };
    }
    if (may_float4_or_rgba) {
      $if((!handled) &
          ((descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT4)) |
           (descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_RGBA)))) {
        found_val = convert_surface_float4(
            node.output_type, primitive_surface_attribute_float4(
                                  kernel_globals, shader_data, descriptor));
        handled = true;
      };
    }
    if (may_float3) {
      $if(!handled) {
        found_val = convert_surface_float3(
            node.output_type, primitive_surface_attribute_float3(
                                  kernel_globals, shader_data, descriptor));
      };
    }
    return found_val;
  };

  Float3 result = make_float3(0.0f);
  const auto emit_missing_alpha = [&]() noexcept {
    if (const_out) {
      if (*const_out ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
        result = make_float3(1.0f);
      }
    } else {
      $if(node.output_type ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
        result = make_float3(1.0f);
      };
    }
  };
  if (!may_be_uv && !may_be_generated) {
    $if(!is_attribute_found(descriptor)) { emit_missing_alpha(); }
    $else { result = eval_found(); };
    return result;
  }
  if (may_be_uv && !may_be_generated) {
    $if(shader_data.type == primitive_lamp) {
      result = make_float3(1.0f - shader_data.u - shader_data.v,
                           shader_data.u, 0.0f);
    }
    $elif(!is_attribute_found(descriptor)) { emit_missing_alpha(); }
    $else { result = eval_found(); };
    return result;
  }
  if (!may_be_uv && may_be_generated) {
    $if(!is_attribute_found(descriptor)) {
      result = kernel_globals.object_inverse_position_transform_if_object(
          shader_data, shader_data.P);
    }
    $else { result = eval_found(); };
    return result;
  }

  $if((shader_data.type == primitive_lamp) &
      (node.attr == static_cast<std::int32_t>(ATTR_STD_UV))) {
    result = make_float3(1.0f - shader_data.u - shader_data.v,
                         shader_data.u, 0.0f);
  }
  $elif((node.attr == static_cast<std::int32_t>(ATTR_STD_GENERATED)) &
        !is_attribute_found(descriptor)) {
    result = kernel_globals.object_inverse_position_transform_if_object(
        shader_data, shader_data.P);
  }
  $elif(!is_attribute_found(descriptor)) { emit_missing_alpha(); }
  $else { result = eval_found(); };
  return result;
}

[[nodiscard]] Dual3 node_attr_derivative_eval(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeNode &node,
    const AttributeDescriptor &descriptor,
    bool known_surface_absent = false) noexcept {
  const auto summary = kernel_globals.surface_attribute_summary();
  const auto type_mask = summary.type_mask;
  const bool may_float = (type_mask & (1u << NODE_ATTR_FLOAT)) != 0u;
  const bool may_float2 = (type_mask & (1u << NODE_ATTR_FLOAT2)) != 0u;
  const bool may_float3 = (type_mask & (1u << NODE_ATTR_FLOAT3)) != 0u;
  const bool may_float4_or_rgba =
      (type_mask & ((1u << NODE_ATTR_FLOAT4) | (1u << NODE_ATTR_RGBA))) != 0u;
  const auto const_attr = try_const_int(node.attr);
  const auto const_out = try_const_uint(node.output_type);
  const bool may_be_uv =
      !const_attr.has_value() ||
      *const_attr == static_cast<std::int32_t>(ATTR_STD_UV);
  const bool may_be_generated =
      !const_attr.has_value() ||
      *const_attr == static_cast<std::int32_t>(ATTR_STD_GENERATED);

  Dual3 result{.val = make_float3(0.0f),
               .dx = make_float3(0.0f),
               .dy = make_float3(0.0f)};
  if (known_surface_absent) {
    if (may_be_uv) {
      $if((shader_data.type == primitive_lamp) &
          (node.attr == static_cast<std::int32_t>(ATTR_STD_UV))) {
        result.val = make_float3(1.0f - shader_data.u - shader_data.v,
                                 shader_data.u, 0.0f);
        result.dx = make_float3(-shader_data.du.dx - shader_data.dv.dx,
                                shader_data.du.dx, 0.0f);
        result.dy = make_float3(-shader_data.du.dy - shader_data.dv.dy,
                                shader_data.du.dy, 0.0f);
      };
      return result;
    }
    if (may_be_generated) {
      return kernel_globals
          .object_inverse_position_transform_if_object_derivative(
              shader_data, shading_position(shader_data));
    }
    if (const_out) {
      if (*const_out ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
        return dual3_one();
      }
      return result;
    }
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = dual3_one();
    };
    return result;
  }

  const auto eval_found = [&]() noexcept {
    const unsigned family_count =
        (may_float ? 1u : 0u) + (may_float2 ? 1u : 0u) +
        (may_float4_or_rgba ? 1u : 0u) + (may_float3 ? 1u : 0u);
    if (family_count == 1u) {
      if (may_float) {
        return convert_dual_float(
            node.output_type,
            primitive_surface_attribute_float_derivative(
                kernel_globals, shader_data, descriptor));
      }
      if (may_float2) {
        return convert_dual_float2(
            node.output_type,
            primitive_surface_attribute_float2_derivative(
                kernel_globals, shader_data, descriptor));
      }
      if (may_float4_or_rgba) {
        return convert_dual_float4(
            node.output_type,
            primitive_surface_attribute_float4_derivative(
                kernel_globals, shader_data, descriptor));
      }
      return convert_dual_float3(
          node.output_type,
          primitive_surface_attribute_float3_derivative(
              kernel_globals, shader_data, descriptor));
    }
    Dual3 found_val{.val = make_float3(0.0f),
                    .dx = make_float3(0.0f),
                    .dy = make_float3(0.0f)};
    Bool handled = false;
    if (may_float) {
      $if(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT)) {
        found_val = convert_dual_float(
            node.output_type,
            primitive_surface_attribute_float_derivative(
                kernel_globals, shader_data, descriptor));
        handled = true;
      };
    }
    if (may_float2) {
      $if((!handled) &
          (descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT2))) {
        found_val = convert_dual_float2(
            node.output_type,
            primitive_surface_attribute_float2_derivative(
                kernel_globals, shader_data, descriptor));
        handled = true;
      };
    }
    if (may_float4_or_rgba) {
      $if((!handled) &
          ((descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT4)) |
           (descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_RGBA)))) {
        found_val = convert_dual_float4(
            node.output_type,
            primitive_surface_attribute_float4_derivative(
                kernel_globals, shader_data, descriptor));
        handled = true;
      };
    }
    if (may_float3) {
      $if(!handled) {
        found_val = convert_dual_float3(
            node.output_type,
            primitive_surface_attribute_float3_derivative(
                kernel_globals, shader_data, descriptor));
      };
    }
    return found_val;
  };

  const auto emit_missing_alpha = [&]() noexcept {
    if (const_out) {
      if (*const_out ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
        result = dual3_one();
      }
    } else {
      $if(node.output_type ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
        result = dual3_one();
      };
    }
  };
  if (!may_be_uv && !may_be_generated) {
    $if(!is_attribute_found(descriptor)) { emit_missing_alpha(); }
    $else { result = eval_found(); };
    return result;
  }
  if (may_be_uv && !may_be_generated) {
    $if(shader_data.type == primitive_lamp) {
      result.val = make_float3(1.0f - shader_data.u - shader_data.v,
                               shader_data.u, 0.0f);
      result.dx = make_float3(-shader_data.du.dx - shader_data.dv.dx,
                              shader_data.du.dx, 0.0f);
      result.dy = make_float3(-shader_data.du.dy - shader_data.dv.dy,
                              shader_data.du.dy, 0.0f);
    }
    $elif(!is_attribute_found(descriptor)) { emit_missing_alpha(); }
    $else { result = eval_found(); };
    return result;
  }
  if (!may_be_uv && may_be_generated) {
    $if(!is_attribute_found(descriptor)) {
      result =
          kernel_globals.object_inverse_position_transform_if_object_derivative(
              shader_data, shading_position(shader_data));
    }
    $else { result = eval_found(); };
    return result;
  }

  $if((shader_data.type == primitive_lamp) &
      (node.attr == static_cast<std::int32_t>(ATTR_STD_UV))) {
    result.val = make_float3(1.0f - shader_data.u - shader_data.v,
                             shader_data.u, 0.0f);
    result.dx = make_float3(-shader_data.du.dx - shader_data.dv.dx,
                            shader_data.du.dx, 0.0f);
    result.dy = make_float3(-shader_data.du.dy - shader_data.dv.dy,
                            shader_data.du.dy, 0.0f);
  }
  $elif((node.attr == static_cast<std::int32_t>(ATTR_STD_GENERATED)) &
        !is_attribute_found(descriptor)) {
    result =
        kernel_globals.object_inverse_position_transform_if_object_derivative(
            shader_data, shading_position(shader_data));
  }
  $elif(!is_attribute_found(descriptor)) { emit_missing_alpha(); }
  $else { result = eval_found(); };
  return result;
}

void node_attr_store(Expr<std::uint32_t> type, Stack &stack,
                     Expr<std::uint32_t> output_offset,
                     Expr<luisa::float3> value) noexcept {
  if (const auto ct = try_const_uint(type)) {
    if (*ct == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT3)) {
      stack_store_float3(stack, output_offset, value);
    } else {
      stack_store_float(stack, output_offset, average(value));
    }
    return;
  }
  $if(type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT3)) {
    stack_store_float3(stack, output_offset, value);
  }
  $else { stack_store_float(stack, output_offset, average(value)); };
}

void node_attr_store(Expr<std::uint32_t> type, Stack &stack,
                     Expr<std::uint32_t> output_offset,
                     const Dual3 &value) noexcept {
  if (const auto ct = try_const_uint(type)) {
    if (*ct == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT3)) {
      stack_store_dual3(stack, output_offset, value);
    } else {
      stack_store_dual1(stack, output_offset, average(value));
    }
    return;
  }
  $if(type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT3)) {
    stack_store_dual3(stack, output_offset, value);
  }
  $else { stack_store_dual1(stack, output_offset, average(value)); };
}

} // namespace

void node_attr_surface(Cursor &cursor, Stack &stack,
                       const KernelGlobals &kernel_globals,
                       const ShaderData &shader_data) noexcept {
  const auto node = read_attribute_node(cursor);
  const auto ca = try_const_int(node.attr);
  const auto cot = try_const_uint(node.output_type);
  const bool cacheable =
      ca.has_value() && *ca >= 0 &&
      *ca != static_cast<std::int32_t>(ATTR_STD_GENERATED) &&
      cot.has_value();
  if (cacheable) {
    if (auto cached = lookup_cached_surface_attr(
            shader_data, static_cast<std::uint64_t>(*ca), *cot)) {
      node_attr_store(node.output_type, stack, node.out_offset, *cached);
      return;
    }
  }
  if (ca && *ca >= 0) {
    const auto scoped_summary = kernel_globals.attribute_summary_for_id(
        static_cast<std::uint64_t>(*ca));
    const AttributeScopedKernelGlobals scoped_kg{kernel_globals,
                                                 scoped_summary};
    const bool absent = scoped_summary.element_mask == 0u;
    const auto descriptor =
        node_attr_init(scoped_kg, shader_data, node, absent);
    const auto data =
        node_attr_surface_eval(scoped_kg, shader_data, node, descriptor, absent);
    if (cacheable) {
      store_cached_surface_attr(shader_data, static_cast<std::uint64_t>(*ca),
                                *cot, data);
    }
    node_attr_store(node.output_type, stack, node.out_offset, data);
    return;
  }
  const auto descriptor = node_attr_init(kernel_globals, shader_data, node);
  const auto data =
      node_attr_surface_eval(kernel_globals, shader_data, node, descriptor);
  node_attr_store(node.output_type, stack, node.out_offset, data);
}

void node_attr_derivative(Cursor &cursor, Stack &stack,
                          const KernelGlobals &kernel_globals,
                          const ShaderData &shader_data) noexcept {
  const auto node = read_attribute_node(cursor);
  const auto cbo = try_const_uint(node.bump_offset);
  const auto csd = try_const_uint(node.store_derivatives);
  const auto ca = try_const_int(node.attr);
  const auto cot = try_const_uint(node.output_type);
  const bool cacheable =
      ca.has_value() && *ca >= 0 &&
      *ca != static_cast<std::int32_t>(ATTR_STD_GENERATED) &&
      cot.has_value();
  if (cbo && *cbo == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_CENTER) &&
      csd && *csd == 0u) {
    if (cacheable) {
      if (auto cached = lookup_cached_surface_attr(
              shader_data, static_cast<std::uint64_t>(*ca), *cot)) {
        node_attr_store(node.output_type, stack, node.out_offset, *cached);
        return;
      }
    }
    if (ca && *ca >= 0) {
      const auto scoped_summary = kernel_globals.attribute_summary_for_id(
          static_cast<std::uint64_t>(*ca));
      const AttributeScopedKernelGlobals scoped_kg{kernel_globals,
                                                   scoped_summary};
      const bool absent = scoped_summary.element_mask == 0u;
      const auto descriptor =
          node_attr_init(scoped_kg, shader_data, node, absent);
      const auto data = node_attr_surface_eval(scoped_kg, shader_data, node,
                                               descriptor, absent);
      if (cacheable) {
        store_cached_surface_attr(shader_data, static_cast<std::uint64_t>(*ca),
                                  *cot, data);
      }
      node_attr_store(node.output_type, stack, node.out_offset, data);
      return;
    }
    const auto descriptor = node_attr_init(kernel_globals, shader_data, node);
    const auto data =
        node_attr_surface_eval(kernel_globals, shader_data, node, descriptor);
    node_attr_store(node.output_type, stack, node.out_offset, data);
    return;
  }
  const auto eval_with_kg = [&](const KernelGlobals &kg,
                                bool absent) noexcept {
    const auto descriptor = node_attr_init(kg, shader_data, node, absent);
    return node_attr_derivative_eval(kg, shader_data, node, descriptor, absent);
  };
  Dual3 data = [&]() noexcept {
    if (cacheable) {
      if (auto cached = lookup_cached_deriv_attr(
              shader_data, static_cast<std::uint64_t>(*ca), *cot)) {
        return *cached;
      }
    }
    if (ca && *ca >= 0) {
      const auto scoped_summary = kernel_globals.attribute_summary_for_id(
          static_cast<std::uint64_t>(*ca));
      const AttributeScopedKernelGlobals scoped_kg{kernel_globals,
                                                   scoped_summary};
      auto evaluated =
          eval_with_kg(scoped_kg, scoped_summary.element_mask == 0u);
      if (cacheable) {
        store_cached_deriv_attr(shader_data, static_cast<std::uint64_t>(*ca),
                                *cot, evaluated);
      }
      return evaluated;
    }
    return eval_with_kg(kernel_globals, false);
  }();
  if (cbo) {
    if (*cbo == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
      data.val += data.dx * node.bump_filter_width;
    } else if (*cbo == static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
      data.val += data.dy * node.bump_filter_width;
    }
  } else {
    $if(node.bump_offset ==
        static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
      data.val += data.dx * node.bump_filter_width;
    }
    $elif(node.bump_offset ==
          static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
      data.val += data.dy * node.bump_filter_width;
    };
  }
  if (csd) {
    if (*csd != 0u) {
      node_attr_store(node.output_type, stack, node.out_offset, data);
    } else {
      node_attr_store(node.output_type, stack, node.out_offset, data.val);
    }
  } else {
    $if(node.store_derivatives != 0u) {
      node_attr_store(node.output_type, stack, node.out_offset, data);
    }
    $else {
      node_attr_store(node.output_type, stack, node.out_offset, data.val);
    };
  }
}

void node_attr_volume(Cursor &cursor, Stack &stack,
                      const KernelGlobals &kernel_globals,
                      ShaderData &shader_data) noexcept {
  device_assert(primitive_is_volume_attribute(shader_data));
  const auto node = read_attribute_node(cursor);
  const auto descriptor =
      node_attr_init(kernel_globals, shader_data, node);
  const auto stochastic =
      node.bump_filter_width.bitcast<luisa::uint>() != 0u;
  const auto value = volume_attribute_float4(
      kernel_globals, shader_data, descriptor, stochastic);
  $if(node.output_type ==
      static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
    stack_store_float(stack, node.out_offset, average(value.xyz()));
  }
  $elif(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT3)) {
    stack_store_float3(stack, node.out_offset, value.xyz());
  }
  $else { stack_store_float(stack, node.out_offset, value.w); };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

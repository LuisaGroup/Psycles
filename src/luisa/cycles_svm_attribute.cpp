/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

struct AttributeNode {
  Int attr;
  UInt out_offset;
  UInt output_type;
  UInt bump_offset;
  UInt store_derivatives;
  Float bump_filter_width;
};

[[nodiscard]] AttributeNode read_attribute_node(Cursor &cursor) noexcept {
  const auto attr = cursor.word().bitcast<std::int32_t>();
  const auto packed = cursor.word();
  return {.attr = attr,
          .out_offset = cursor.byte(packed, 0u),
          .output_type = cursor.byte(packed, 1u),
          .bump_offset = cursor.byte(packed, 2u),
          .store_derivatives = cursor.byte(packed, 3u),
          .bump_filter_width = cursor.floating()};
}

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
    const AttributeNode &node) noexcept {
  auto descriptor = attribute_not_found();
  $if(shader_data.object != object_none) {
    descriptor = find_attribute(kernel_globals, shader_data,
                                node.attr.cast<luisa::ulong>());
    $if(!is_attribute_found(descriptor)) {
      descriptor = attribute_not_found();
      descriptor.type = node.output_type;
    };
  }
  $else {
    descriptor = attribute_not_found();
    descriptor.type = node.output_type;
  };
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

[[nodiscard]] Float3 node_attr_surface_eval(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeNode &node,
    const AttributeDescriptor &descriptor) noexcept {
  Float3 result = make_float3(0.0f);
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
  $elif(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT)) {
    const auto value = primitive_surface_attribute_float(
        kernel_globals, shader_data, descriptor);
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = make_float3(1.0f);
    }
    $else { result = make_float3(value); };
  }
  $elif(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT2)) {
    const auto value = primitive_surface_attribute_float2(
        kernel_globals, shader_data, descriptor);
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      result = make_float3(value.x);
    }
    $elif(node.output_type ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = make_float3(1.0f);
    }
    $else { result = make_float3(value, 0.0f); };
  }
  $elif((descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT4)) |
        (descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_RGBA))) {
    const auto value = primitive_surface_attribute_float4(
        kernel_globals, shader_data, descriptor);
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      result = make_float3(average(value.xyz()));
    }
    $elif(node.output_type ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = make_float3(value.w);
    }
    $else { result = value.xyz(); };
  }
  $else {
    const auto value = primitive_surface_attribute_float3(
        kernel_globals, shader_data, descriptor);
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      result = make_float3(average(value));
    }
    $elif(node.output_type ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = make_float3(1.0f);
    }
    $else { result = value; };
  };
  return result;
}

[[nodiscard]] Dual3 node_attr_derivative_eval(
    const KernelGlobals &kernel_globals, const ShaderData &shader_data,
    const AttributeNode &node,
    const AttributeDescriptor &descriptor) noexcept {
  Dual3 result{.val = make_float3(0.0f),
               .dx = make_float3(0.0f),
               .dy = make_float3(0.0f)};
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
  $elif(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT)) {
    const auto value = primitive_surface_attribute_float_derivative(
        kernel_globals, shader_data, descriptor);
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = dual3_one();
    }
    $else { result = dual3_from_dual1(value); };
  }
  $elif(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT2)) {
    const auto value = primitive_surface_attribute_float2_derivative(
        kernel_globals, shader_data, descriptor);
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      result = {.val = make_float3(value.val.x),
                .dx = make_float3(value.dx.x),
                .dy = make_float3(value.dy.x)};
    }
    $elif(node.output_type ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = dual3_one();
    }
    $else { result = dual3_from_dual2(value); };
  }
  $elif((descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT4)) |
        (descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_RGBA))) {
    const auto value = primitive_surface_attribute_float4_derivative(
        kernel_globals, shader_data, descriptor);
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      result = dual3_from_dual1(average(dual3_from_dual4(value)));
    }
    $elif(node.output_type ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = dual3_from_dual4_alpha(value);
    }
    $else { result = dual3_from_dual4(value); };
  }
  $else {
    const auto value = primitive_surface_attribute_float3_derivative(
        kernel_globals, shader_data, descriptor);
    $if(node.output_type ==
        static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT)) {
      result = dual3_from_dual1(average(value));
    }
    $elif(node.output_type ==
          static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT_ALPHA)) {
      result = dual3_one();
    }
    $else { result = value; };
  };
  return result;
}

void node_attr_store(Expr<std::uint32_t> type, Stack &stack,
                     Expr<std::uint32_t> output_offset,
                     Expr<luisa::float3> value) noexcept {
  $if(type == static_cast<std::uint32_t>(NODE_ATTR_OUTPUT_FLOAT3)) {
    stack_store_float3(stack, output_offset, value);
  }
  $else { stack_store_float(stack, output_offset, average(value)); };
}

void node_attr_store(Expr<std::uint32_t> type, Stack &stack,
                     Expr<std::uint32_t> output_offset,
                     const Dual3 &value) noexcept {
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
  const auto descriptor =
      node_attr_init(kernel_globals, shader_data, node);
  const auto data = node_attr_surface_eval(kernel_globals, shader_data, node,
                                           descriptor);
  node_attr_store(node.output_type, stack, node.out_offset, data);
}

void node_attr_derivative(Cursor &cursor, Stack &stack,
                          const KernelGlobals &kernel_globals,
                          const ShaderData &shader_data) noexcept {
  const auto node = read_attribute_node(cursor);
  const auto descriptor =
      node_attr_init(kernel_globals, shader_data, node);
  auto data = node_attr_derivative_eval(kernel_globals, shader_data, node,
                                        descriptor);
  $if(node.bump_offset ==
      static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
    data.val += data.dx * node.bump_filter_width;
  }
  $elif(node.bump_offset ==
        static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
    data.val += data.dy * node.bump_filter_width;
  };
  $if(node.store_derivatives != 0u) {
    node_attr_store(node.output_type, stack, node.out_offset, data);
  }
  $else {
    node_attr_store(node.output_type, stack, node.out_offset, data.val);
  };
}

void node_attr_volume(Cursor &cursor, Stack &stack,
                      const KernelGlobals &kernel_globals,
                      const ShaderData &shader_data) noexcept {
  const auto node = read_attribute_node(cursor);
  const auto descriptor =
      node_attr_init(kernel_globals, shader_data, node);
  const auto stochastic =
      node.bump_filter_width.bitcast<luisa::uint>() != 0u;
  const auto value = kernel_globals.volume_attribute_float4(
      shader_data, descriptor, stochastic);
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

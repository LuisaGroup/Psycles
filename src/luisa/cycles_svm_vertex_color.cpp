/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

struct VertexColorNode {
  UInt layer_id;
  UInt color_offset;
  UInt alpha_offset;
  UInt bump_offset;
  Float bump_filter_width;
};

[[nodiscard]] VertexColorNode
read_vertex_color_node(Cursor &cursor) noexcept {
  const auto packed = cursor.word();
  return {.layer_id = cursor.byte(packed, 0u),
          .color_offset = cursor.byte(packed, 1u),
          .alpha_offset = cursor.byte(packed, 2u),
          .bump_offset = cursor.byte(packed, 3u),
          .bump_filter_width = cursor.floating()};
}

[[nodiscard]] Bool
is_attribute_found(const AttributeDescriptor &descriptor) noexcept {
  return descriptor.offset !=
         static_cast<std::int32_t>(ATTR_STD_NOT_FOUND);
}

void store_vertex_color(Stack &stack, const VertexColorNode &node,
                        Expr<luisa::float3> color,
                        Expr<float> alpha) noexcept {
  $if(node.color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, node.color_offset, color);
  };
  $if(node.alpha_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, node.alpha_offset, alpha);
  };
}

template<bool use_derivatives>
void node_vertex_color_impl(Cursor &cursor, Stack &stack,
                            const KernelGlobals &kernel_globals,
                            const ShaderData &shader_data) noexcept {
  const auto node = read_vertex_color_node(cursor);
  Float3 color = make_float3(0.0f);
  Float alpha = 0.0f;
  const auto descriptor = find_attribute(
      kernel_globals, shader_data, node.layer_id.cast<luisa::ulong>());

  $if(is_attribute_found(descriptor)) {
    $if((descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT4)) |
        (descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_RGBA))) {
      if constexpr (use_derivatives) {
        auto vertex_color = primitive_surface_attribute_float4_derivative(
            kernel_globals, shader_data, descriptor);
        $if(node.bump_offset ==
            static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
          vertex_color.val += vertex_color.dx * node.bump_filter_width;
        }
        $elif(node.bump_offset ==
              static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
          vertex_color.val += vertex_color.dy * node.bump_filter_width;
        };
        color = vertex_color.val.xyz();
        alpha = vertex_color.val.w;
      } else {
        const Float4 vertex_color = primitive_surface_attribute_float4(
            kernel_globals, shader_data, descriptor);
        color = vertex_color.xyz();
        alpha = vertex_color.w;
      }
    }
    $else {
      if constexpr (use_derivatives) {
        auto vertex_color = primitive_surface_attribute_float3_derivative(
            kernel_globals, shader_data, descriptor);
        $if(node.bump_offset ==
            static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DX)) {
          vertex_color.val += vertex_color.dx * node.bump_filter_width;
        }
        $elif(node.bump_offset ==
              static_cast<std::uint32_t>(NODE_BUMP_OFFSET_DY)) {
          vertex_color.val += vertex_color.dy * node.bump_filter_width;
        };
        color = vertex_color.val;
      } else {
        color = primitive_surface_attribute_float3(kernel_globals, shader_data,
                                                   descriptor);
      }
      alpha = 1.0f;
    };
  };

  store_vertex_color(stack, node, color, alpha);
}

} // namespace

void node_vertex_color(Cursor &cursor, Stack &stack,
                       const KernelGlobals &kernel_globals,
                       const ShaderData &shader_data) noexcept {
  node_vertex_color_impl<false>(cursor, stack, kernel_globals, shader_data);
}

void node_vertex_color_derivative(Cursor &cursor, Stack &stack,
                                  const KernelGlobals &kernel_globals,
                                  const ShaderData &shader_data) noexcept {
  node_vertex_color_impl<true>(cursor, stack, kernel_globals, shader_data);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Float linear_rgb_to_gray(const KernelGlobals &kernel_globals,
                                       Expr<luisa::float3> value) noexcept {
  return dot(value, kernel_globals.film_rgb_to_y());
}

void store_dual1(Stack &stack, Expr<std::uint32_t> offset, Expr<float> value,
                 Expr<float> dx, Expr<float> dy,
                 bool use_derivatives) noexcept {
  stack_store_float(stack, offset, value);
  if (use_derivatives) {
    stack_store_float(stack, offset + 1u, dx);
    stack_store_float(stack, offset + 2u, dy);
  }
}

void store_dual3(Stack &stack, Expr<std::uint32_t> offset,
                 Expr<luisa::float3> value, Expr<luisa::float3> dx,
                 Expr<luisa::float3> dy, bool use_derivatives) noexcept {
  stack_store_float3(stack, offset, value);
  if (use_derivatives) {
    stack_store_float3(stack, offset + 3u, dx);
    stack_store_float3(stack, offset + 6u, dy);
  }
}

} // namespace

void node_convert(Cursor &cursor, Stack &stack,
                  const KernelGlobals &kernel_globals,
                  bool use_derivatives) noexcept {
  const auto convert_type = cursor.word();
  const auto packed_offsets = cursor.word();
  const auto from_offset = cursor.byte(packed_offsets, 0u);
  const auto to_offset = cursor.byte(packed_offsets, 1u);

  $switch(convert_type) {
    PSYCLES_SVM_CASE(NODE_CONVERT_FI) {
      const auto value = stack_load_float(stack, from_offset);
      stack_store_int(stack, to_offset, value.cast<std::int32_t>());
    };
    PSYCLES_SVM_CASE(NODE_CONVERT_FV) {
      const auto value = stack_load_float(stack, from_offset);
      Float dx = 0.0f;
      Float dy = 0.0f;
      if (use_derivatives) {
        dx = stack_load_float(stack, from_offset + 1u);
        dy = stack_load_float(stack, from_offset + 2u);
      }
      store_dual3(stack, to_offset, make_float3(value), make_float3(dx),
                  make_float3(dy), use_derivatives);
    };
    PSYCLES_SVM_CASE(NODE_CONVERT_CF) {
      const auto value = stack_load_float3(stack, from_offset);
      Float dx = 0.0f;
      Float dy = 0.0f;
      if (use_derivatives) {
        dx = linear_rgb_to_gray(kernel_globals,
                                stack_load_float3(stack, from_offset + 3u));
        dy = linear_rgb_to_gray(kernel_globals,
                                stack_load_float3(stack, from_offset + 6u));
      }
      store_dual1(stack, to_offset, linear_rgb_to_gray(kernel_globals, value),
                  dx, dy, use_derivatives);
    };
    PSYCLES_SVM_CASE(NODE_CONVERT_CI) {
      const auto value = linear_rgb_to_gray(
          kernel_globals, stack_load_float3(stack, from_offset));
      stack_store_int(stack, to_offset, value.cast<std::int32_t>());
    };
    PSYCLES_SVM_CASE(NODE_CONVERT_VF) {
      const auto value = stack_load_float3(stack, from_offset);
      Float dx = 0.0f;
      Float dy = 0.0f;
      if (use_derivatives) {
        dx = average(stack_load_float3(stack, from_offset + 3u));
        dy = average(stack_load_float3(stack, from_offset + 6u));
      }
      store_dual1(stack, to_offset, average(value), dx, dy, use_derivatives);
    };
    PSYCLES_SVM_CASE(NODE_CONVERT_VI) {
      const auto value = average(stack_load_float3(stack, from_offset));
      stack_store_int(stack, to_offset, value.cast<std::int32_t>());
    };
    PSYCLES_SVM_CASE(NODE_CONVERT_IF) {
      const auto value = stack_load_int(stack, from_offset).cast<float>();
      store_dual1(stack, to_offset, value, 0.0f, 0.0f, use_derivatives);
    };
    PSYCLES_SVM_CASE(NODE_CONVERT_IV) {
      const auto value = stack_load_int(stack, from_offset).cast<float>();
      store_dual3(stack, to_offset, make_float3(value), make_float3(0.0f),
                  make_float3(0.0f), use_derivatives);
    };
    $default{};
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE

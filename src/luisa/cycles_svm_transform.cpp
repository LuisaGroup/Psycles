/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <psycles/luisa/cycles_transform.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;

Float4x4 transform_from_rows(Expr<luisa::float4> x,
                             Expr<luisa::float4> y,
                             Expr<luisa::float4> z) noexcept {
  return make_float4x4(make_float4(x.x, y.x, z.x, 0.0f),
                       make_float4(x.y, y.y, z.y, 0.0f),
                       make_float4(x.z, y.z, z.z, 0.0f),
                       make_float4(x.w, y.w, z.w, 1.0f));
}

Float4x4 packed_transform(Cursor &cursor) noexcept {
  // Cursor reads are state transitions. C++ does not specify argument
  // evaluation order, so every bytecode word is consumed in its own
  // full-expression.
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

Dual3 transform_point(Expr<luisa::float4x4> transform,
                      const Dual3 &value) noexcept {
  return {.val = cycles_transform::point(transform, value.val),
          .dx = cycles_transform::direction(transform, value.dx),
          .dy = cycles_transform::direction(transform, value.dy)};
}

} // namespace psycles::luisa_backend::cycles_svm::detail

/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

void node_rgb_ramp(Cursor &cursor, Stack &stack) noexcept {
  const UInt table_size = cursor.word();
  const Float factor = stack_load_input_float(stack, cursor.word());
  const UInt packed = cursor.word();
  const UInt interpolate = cursor.byte(packed, 0u);
  const UInt color_offset = cursor.byte(packed, 1u);
  const UInt alpha_offset = cursor.byte(packed, 2u);

  const UInt last = table_size - 1u;
  const Float scaled = clamp(factor, 0.0f, 1.0f) * cast<float>(last);
  // Keep the signed float_to_int followed by clamp from Cycles. In
  // particular, changing the conversion to unsigned changes the defined
  // finite-input range and backend handling of non-finite values before the
  // explicit clamp.
  const Int index = clamp(cast<int>(scaled), 0, cast<int>(last));
  const UInt element = cast<uint>(index);
  const Float t = scaled - cast<float>(index);
  const auto read = [&](Expr<std::uint32_t> element) noexcept {
    const UInt base = element * 4u;
    return make_float4(cursor.floating_at(base), cursor.floating_at(base + 1u),
                       cursor.floating_at(base + 2u),
                       cursor.floating_at(base + 3u));
  };

  Float4 color = read(element);
  $if((interpolate != 0u) & (t > 0.0f)) {
    const Float4 next = read(element + 1u);
    color = (1.0f - t) * color + t * next;
  };

  $if(color_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float3(stack, color_offset, color.xyz());
  };
  $if(alpha_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, alpha_offset, color.w);
  };
  cursor.advance(table_size * 4u);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

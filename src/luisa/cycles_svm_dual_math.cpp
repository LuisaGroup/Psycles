/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;

Dual3 safe_normalize_dual(const Dual3 &value) noexcept {
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

} // namespace psycles::luisa_backend::cycles_svm::detail

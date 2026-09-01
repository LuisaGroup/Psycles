/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <luisa/dsl/sugar.h>

#include <cstdint>

namespace psycles::luisa_backend::cycles_checker {

[[nodiscard]] inline luisa::compute::Float
evaluate(luisa::compute::Float3 point) noexcept {
  using namespace luisa::compute;
  point = (point + 0.000001f) * 0.999999f;
  const Int x = abs(floor(point.x).cast<std::int32_t>());
  const Int y = abs(floor(point.y).cast<std::int32_t>());
  const Int z = abs(floor(point.z).cast<std::int32_t>());
  const auto xy_parity = select(0, 1, (x % 2) == (y % 2));
  return select(0.0f, 1.0f, xy_parity == (z % 2));
}

} // namespace psycles::luisa_backend::cycles_checker

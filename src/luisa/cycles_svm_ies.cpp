/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

namespace {

[[nodiscard]] Float cubic_interp(Float a, Float b, Float c, Float d,
                                 Float x) noexcept {
  // Exact Cycles util/math_base.h polynomial and association.
  return 0.5f *
             (((d + 3.0f * (b - c) - a) * x +
               (2.0f * a - 5.0f * b + 4.0f * c - d)) *
                  x +
              (c - a)) *
             x +
         b;
}

[[nodiscard]] Float ies_lookup(const KernelGlobals &kernel_globals,
                               Int offset) noexcept {
  return kernel_globals.ies(offset.cast<std::uint32_t>());
}

[[nodiscard]] Float interpolate_vertical(
    const KernelGlobals &kernel_globals, Int offset, Bool wrap_low,
    Bool wrap_high, Int vertical, Int vertical_count, Float fraction,
    Int horizontal) noexcept {
  const auto base = offset + horizontal * vertical_count;
  const Float c = ies_lookup(kernel_globals, base + vertical + 1);
  const Float b = ies_lookup(kernel_globals, base + vertical);

  Float a = b;
  $if(vertical > 0) { a = ies_lookup(kernel_globals, base + vertical - 1); }
  $else {
    $if(wrap_low) { a = ies_lookup(kernel_globals, base + 1); };
  };

  Float d = c;
  $if(vertical + 2 < vertical_count) {
    d = ies_lookup(kernel_globals, base + vertical + 2);
  }
  $else {
    $if(wrap_high) {
      d = ies_lookup(kernel_globals, base + vertical_count - 2);
    };
  };
  return cubic_interp(a, b, c, d, fraction);
}

[[nodiscard]] Float kernel_ies_interp(const KernelGlobals &kernel_globals,
                                      UInt slot, Float horizontal_angle,
                                      Float vertical_angle) noexcept {
  Int offset = kernel_globals.ies(slot).bitcast<std::int32_t>();
  Float result = 100.0f;
  $if(offset != -1) {
    const Int horizontal_count =
        ies_lookup(kernel_globals, offset).bitcast<std::int32_t>();
    offset += 1;
    const Int vertical_count =
        ies_lookup(kernel_globals, offset).bitcast<std::int32_t>();
    offset += 1;

    const auto horizontal = [&](Int index) noexcept {
      return ies_lookup(kernel_globals, offset + index);
    };
    const auto vertical = [&](Int index) noexcept {
      return ies_lookup(kernel_globals, offset + horizontal_count + index);
    };

    const Float vertical_low = vertical(0);
    const Float vertical_high = vertical(vertical_count - 1);
    const Float horizontal_low = horizontal(0);
    const Float horizontal_high = horizontal(horizontal_count - 1);
    const Bool inside = (vertical_angle >= vertical_low) &
                        (vertical_angle < vertical_high) &
                        (horizontal_angle >= horizontal_low) &
                        (horizontal_angle < horizontal_high);
    result = 0.0f;
    $if(inside) {
      constexpr auto pi = 3.14159265358979323846f;
      constexpr auto two_pi = 2.0f * pi;
      const Bool wrap_horizontal =
          (horizontal_low < 1.0e-7f) &
          (horizontal_high > two_pi - 1.0e-7f);
      const Bool wrap_vertical_low = vertical_low < 1.0e-7f;
      const Bool wrap_vertical_high = vertical_high > pi - 1.0e-7f;

      Int horizontal_index = 0;
      $while(horizontal(horizontal_index + 1) < horizontal_angle) {
        horizontal_index += 1;
      };
      Int vertical_index = 0;
      $while(vertical(vertical_index + 1) < vertical_angle) {
        vertical_index += 1;
      };

      const Float horizontal_fraction =
          (horizontal_angle - horizontal(horizontal_index)) /
          (horizontal(horizontal_index + 1) -
           horizontal(horizontal_index));
      const Float vertical_fraction =
          (vertical_angle - vertical(vertical_index)) /
          (vertical(vertical_index + 1) - vertical(vertical_index));

      offset += horizontal_count + vertical_count;
      const Float b = interpolate_vertical(
          kernel_globals, offset, wrap_vertical_low, wrap_vertical_high,
          vertical_index, vertical_count, vertical_fraction,
          horizontal_index);
      const Float c = interpolate_vertical(
          kernel_globals, offset, wrap_vertical_low, wrap_vertical_high,
          vertical_index, vertical_count, vertical_fraction,
          horizontal_index + 1);

      Float a = b;
      $if(horizontal_index > 0) {
        a = interpolate_vertical(
            kernel_globals, offset, wrap_vertical_low, wrap_vertical_high,
            vertical_index, vertical_count, vertical_fraction,
            horizontal_index - 1);
      }
      $else {
        $if(wrap_horizontal) {
          a = interpolate_vertical(
              kernel_globals, offset, wrap_vertical_low, wrap_vertical_high,
              vertical_index, vertical_count, vertical_fraction,
              horizontal_count - 2);
        };
      };

      // This is intentionally initialized from b, matching Cycles 5.2.1.
      Float d = b;
      $if(horizontal_index + 2 < horizontal_count) {
        d = interpolate_vertical(
            kernel_globals, offset, wrap_vertical_low, wrap_vertical_high,
            vertical_index, vertical_count, vertical_fraction,
            horizontal_index + 2);
      }
      $else {
        $if(wrap_horizontal) {
          d = interpolate_vertical(
              kernel_globals, offset, wrap_vertical_low, wrap_vertical_high,
              vertical_index, vertical_count, vertical_fraction, 1);
        };
      };
      result = max(cubic_interp(a, b, c, d, horizontal_fraction), 0.0f);
    };
  };
  return result;
}

} // namespace

void node_ies(Cursor &cursor, Stack &stack,
              const KernelGlobals &kernel_globals) noexcept {
  const auto strength_input = cursor.word();
  const auto slot = cursor.word();
  const auto packed_offsets = cursor.word();
  const auto vector_offset = cursor.byte(packed_offsets, 0u);
  const auto factor_offset = cursor.byte(packed_offsets, 1u);

  auto vector = stack_load_float3(stack, vector_offset);
  const auto strength = stack_load_input_float(stack, strength_input);
  vector = normalize_cycles(vector);
  constexpr auto pi = 3.14159265358979323846f;
  const auto vertical_angle = acos(clamp(-vector.z, -1.0f, 1.0f));
  const auto horizontal_angle = atan2(vector.x, vector.y) + pi;
  const auto factor =
      strength * kernel_ies_interp(kernel_globals, slot, horizontal_angle,
                                   vertical_angle);
  $if(factor_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
    stack_store_float(stack, factor_offset, factor);
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail

/* SPDX-FileCopyrightText: 2018-2023 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_principled_hair_math.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail::principled_hair_math {

using namespace luisa::compute;

namespace {

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

} // namespace

Float bessel_I0(Expr<float> argument) noexcept {
  const auto squared_argument = square(argument);
  Float value = 1.0f + 0.25f * squared_argument;
  Float power = square(squared_argument);
  ULong factorial_squared = 1ull;
  UInt power_of_four = 16u;
  UInt index = 2u;
  Bool active = true;
  $while((index < 10u) & active) {
    const auto index64 = index.cast<luisa::ulong>();
    factorial_squared *= index64 * index64;
    const auto next = value + power / (power_of_four.cast<float>() *
                                       factorial_squared.cast<float>());
    active = value != next;
    value = next;
    power *= squared_argument;
    power_of_four *= 4u;
    index += 1u;
  };
  return value;
}

Float log_bessel_I0(Expr<float> argument) noexcept {
  Float result;
  $if(argument > 12.0f) {
    result = argument + 0.5f * (1.0f / (8.0f * argument) -
                                1.83787706640934548356f - log(argument));
  }
  $else { result = log(bessel_I0(argument)); };
  return result;
}

Float longitudinal_scattering(Expr<float> sine_incoming,
                              Expr<float> cosine_incoming,
                              Expr<float> sine_outgoing,
                              Expr<float> cosine_outgoing,
                              Expr<float> variance) noexcept {
  const auto inverse_variance = 1.0f / variance;
  const auto cosine_argument =
      cosine_incoming * cosine_outgoing * inverse_variance;
  const auto sine_argument = sine_incoming * sine_outgoing * inverse_variance;
  Float result;
  $if(variance <= 0.1f) {
    result = exp(log_bessel_I0(cosine_argument) - sine_argument -
                 inverse_variance + 0.6931f + log(0.5f * inverse_variance));
  }
  $else {
    result = exp(-sine_argument) * bessel_I0(cosine_argument) /
             (sinh(inverse_variance) * 2.0f * variance);
  };
  return result;
}

} // namespace psycles::luisa_backend::cycles_svm::detail::principled_hair_math

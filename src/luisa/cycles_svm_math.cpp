/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include <luisa/dsl/sugar.h>

#define PSYCLES_SVM_CASE(node) \
  $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

[[nodiscard]] Float safe_divide(Float a, Float b) noexcept {
  Float result = 0.0f;
  $if (b != 0.0f) { result = a / b; };
  return result;
}

[[nodiscard]] Float compatible_pow(Float x, Float y) noexcept {
  Float result = 0.0f;
  $if (y == 0.0f) {
    result = 1.0f;
  } $elif (x != 0.0f) {
    $if (x < 0.0f) {
      $if (fmod(-y, 2.0f) == 0.0f) {
        result = pow(-x, y);
      } $else {
        result = -pow(-x, y);
      };
    } $else {
      result = pow(x, y);
    };
  };
  return result;
}

[[nodiscard]] Float safe_pow(Float a, Float b) noexcept {
  Float result = 0.0f;
  const auto integral_b = b.cast<int>().cast<float>();
  $if (!((a < 0.0f) & (b != integral_b))) {
    result = compatible_pow(a, b);
  };
  return result;
}

[[nodiscard]] Float safe_log(Float a, Float b) noexcept {
  Float result = 0.0f;
  $if ((a > 0.0f) & (b > 0.0f)) {
    result = safe_divide(log(a), log(b));
  };
  return result;
}

[[nodiscard]] Float inverse_sqrt(Float value) noexcept {
  Float result = 0.0f;
  $if (value > 0.0f) { result = 1.0f / sqrt(value); };
  return result;
}

[[nodiscard]] Float safe_modulo(Float a, Float b) noexcept {
  Float result = 0.0f;
  $if (b != 0.0f) { result = fmod(a, b); };
  return result;
}

[[nodiscard]] Float safe_floored_modulo(Float a, Float b) noexcept {
  Float result = 0.0f;
  $if (b != 0.0f) { result = a - floor(a / b) * b; };
  return result;
}

[[nodiscard]] Float wrap(Float value, Float maximum, Float minimum) noexcept {
  const Float range = maximum - minimum;
  Float result = minimum;
  $if (range != 0.0f) {
    result = value - range * floor((value - minimum) / range);
  };
  return result;
}

[[nodiscard]] Float pingpong(Float a, Float b) noexcept {
  Float result = 0.0f;
  $if (b != 0.0f) {
    const auto fraction = (a - b) / (b * 2.0f);
    result = abs((fraction - floor(fraction)) * b * 2.0f - b);
  };
  return result;
}

[[nodiscard]] Float smooth_min(Float a, Float b, Float k) noexcept {
  Float result = luisa::compute::min(a, b);
  $if (k != 0.0f) {
    const Float h = luisa::compute::max(k - abs(a - b), 0.0f) / k;
    result = luisa::compute::min(a, b) -
             h * h * h * k * (1.0f / 6.0f);
  };
  return result;
}

[[nodiscard]] Float compatible_atan2(Float y, Float x) noexcept {
  Float result = 0.0f;
  $if ((x != 0.0f) | (y != 0.0f)) { result = atan2(y, x); };
  return result;
}

[[nodiscard]] Float compatible_sign(Float value) noexcept {
  Float result = 0.0f;
  $if (value != 0.0f) { result = select(1.0f, -1.0f, value < 0.0f); };
  return result;
}

} // namespace

luisa::compute::Float svm_math(luisa::compute::Expr<std::uint32_t> type,
                               luisa::compute::Expr<float> a_value,
                               luisa::compute::Expr<float> b_value,
                               luisa::compute::Expr<float> c_value) noexcept {
  using namespace luisa::compute;
  const Float a = a_value;
  const Float b = b_value;
  const Float c = c_value;
  Float result = 0.0f;
  $switch (type) {
    PSYCLES_SVM_CASE(NODE_MATH_ADD) { result = a + b; };
    PSYCLES_SVM_CASE(NODE_MATH_SUBTRACT) { result = a - b; };
    PSYCLES_SVM_CASE(NODE_MATH_MULTIPLY) { result = a * b; };
    PSYCLES_SVM_CASE(NODE_MATH_DIVIDE) { result = safe_divide(a, b); };
    PSYCLES_SVM_CASE(NODE_MATH_SINE) { result = sin(a); };
    PSYCLES_SVM_CASE(NODE_MATH_COSINE) { result = cos(a); };
    PSYCLES_SVM_CASE(NODE_MATH_TANGENT) { result = tan(a); };
    PSYCLES_SVM_CASE(NODE_MATH_ARCSINE) { result = asin(clamp(a, -1.0f, 1.0f)); };
    PSYCLES_SVM_CASE(NODE_MATH_ARCCOSINE) { result = acos(clamp(a, -1.0f, 1.0f)); };
    PSYCLES_SVM_CASE(NODE_MATH_ARCTANGENT) { result = atan(a); };
    PSYCLES_SVM_CASE(NODE_MATH_POWER) { result = safe_pow(a, b); };
    PSYCLES_SVM_CASE(NODE_MATH_LOGARITHM) { result = safe_log(a, b); };
    PSYCLES_SVM_CASE(NODE_MATH_MINIMUM) {
      result = luisa::compute::min(a, b);
    };
    PSYCLES_SVM_CASE(NODE_MATH_MAXIMUM) {
      result = luisa::compute::max(a, b);
    };
    PSYCLES_SVM_CASE(NODE_MATH_ROUND) { result = floor(a + 0.5f); };
    PSYCLES_SVM_CASE(NODE_MATH_LESS_THAN) {
      result = select(0.0f, 1.0f, a < b);
    };
    PSYCLES_SVM_CASE(NODE_MATH_GREATER_THAN) {
      result = select(0.0f, 1.0f, a > b);
    };
    PSYCLES_SVM_CASE(NODE_MATH_MODULO) { result = safe_modulo(a, b); };
    PSYCLES_SVM_CASE(NODE_MATH_ABSOLUTE) { result = abs(a); };
    PSYCLES_SVM_CASE(NODE_MATH_ARCTAN2) { result = compatible_atan2(a, b); };
    PSYCLES_SVM_CASE(NODE_MATH_FLOOR) { result = floor(a); };
    PSYCLES_SVM_CASE(NODE_MATH_CEIL) { result = ceil(a); };
    PSYCLES_SVM_CASE(NODE_MATH_FRACTION) { result = a - floor(a); };
    PSYCLES_SVM_CASE(NODE_MATH_SQRT) {
      result = sqrt(luisa::compute::max(a, 0.0f));
    };
    PSYCLES_SVM_CASE(NODE_MATH_INV_SQRT) { result = inverse_sqrt(a); };
    PSYCLES_SVM_CASE(NODE_MATH_SIGN) { result = compatible_sign(a); };
    PSYCLES_SVM_CASE(NODE_MATH_EXPONENT) { result = exp(a); };
    PSYCLES_SVM_CASE(NODE_MATH_RADIANS) {
      result = a * (3.14159265358979323846f / 180.0f);
    };
    PSYCLES_SVM_CASE(NODE_MATH_DEGREES) {
      result = a * (180.0f / 3.14159265358979323846f);
    };
    PSYCLES_SVM_CASE(NODE_MATH_SINH) { result = sinh(a); };
    PSYCLES_SVM_CASE(NODE_MATH_COSH) { result = cosh(a); };
    PSYCLES_SVM_CASE(NODE_MATH_TANH) { result = tanh(a); };
    PSYCLES_SVM_CASE(NODE_MATH_TRUNC) {
      result = select(ceil(a), floor(a), a >= 0.0f);
    };
    PSYCLES_SVM_CASE(NODE_MATH_SNAP) { result = floor(safe_divide(a, b)) * b; };
    PSYCLES_SVM_CASE(NODE_MATH_WRAP) { result = wrap(a, b, c); };
    PSYCLES_SVM_CASE(NODE_MATH_COMPARE) {
      result = select(0.0f, 1.0f,
                      (a == b) |
                          (abs(a - b) <= luisa::compute::max(
                                             c, 1.1920928955078125e-7f)));
    };
    PSYCLES_SVM_CASE(NODE_MATH_MULTIPLY_ADD) { result = a * b + c; };
    PSYCLES_SVM_CASE(NODE_MATH_PINGPONG) { result = pingpong(a, b); };
    PSYCLES_SVM_CASE(NODE_MATH_SMOOTH_MIN) { result = smooth_min(a, b, c); };
    PSYCLES_SVM_CASE(NODE_MATH_SMOOTH_MAX) {
      result = -smooth_min(-a, -b, c);
    };
    PSYCLES_SVM_CASE(NODE_MATH_FLOORED_MODULO) {
      result = safe_floored_modulo(a, b);
    };
    $default { result = 0.0f; };
  };
  return result;
}

void node_math(Cursor &cursor, Stack &stack) noexcept {
  const auto math_type = cursor.word();
  const auto value1 = cursor.word();
  const auto value2 = cursor.word();
  const auto value3 = cursor.word();
  const auto result_word = cursor.word();
  const auto result_offset = cursor.byte(result_word, 0u);

  const auto a = stack_load_input_float(stack, value1);
  const auto b = stack_load_input_float(stack, value2);
  const auto c = stack_load_input_float(stack, value3);
  stack_store_float(stack, result_offset, svm_math(math_type, a, b, c));
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE

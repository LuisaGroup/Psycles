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
  $if (value > 0.0f) { result = rsqrt(value); };
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

[[nodiscard]] Float svm_math_masked(Expr<std::uint32_t> type,
                                    Expr<float> a_value,
                                    Expr<float> b_value,
                                    Expr<float> c_value,
                                    std::uint64_t type_mask) noexcept {
  const Float a = a_value;
  const Float b = b_value;
  const Float c = c_value;
  const auto enabled = [&](NodeMathType op) noexcept {
    const auto idx = static_cast<std::uint32_t>(op);
    return idx < 64u && (type_mask & (1ull << idx)) != 0ull;
  };
  const auto eval_one = [&](NodeMathType op) noexcept -> Float {
    switch (op) {
      case NODE_MATH_ADD: return a + b;
      case NODE_MATH_SUBTRACT: return a - b;
      case NODE_MATH_MULTIPLY: return a * b;
      case NODE_MATH_DIVIDE: return safe_divide(a, b);
      case NODE_MATH_SINE: return sin(a);
      case NODE_MATH_COSINE: return cos(a);
      case NODE_MATH_TANGENT: return tan(a);
      case NODE_MATH_ARCSINE: return asin(clamp(a, -1.0f, 1.0f));
      case NODE_MATH_ARCCOSINE: return acos(clamp(a, -1.0f, 1.0f));
      case NODE_MATH_ARCTANGENT: return atan(a);
      case NODE_MATH_POWER: return safe_pow(a, b);
      case NODE_MATH_LOGARITHM: return safe_log(a, b);
      case NODE_MATH_MINIMUM: return luisa::compute::min(a, b);
      case NODE_MATH_MAXIMUM: return luisa::compute::max(a, b);
      case NODE_MATH_ROUND: return floor(a + 0.5f);
      case NODE_MATH_LESS_THAN: return select(0.0f, 1.0f, a < b);
      case NODE_MATH_GREATER_THAN: return select(0.0f, 1.0f, a > b);
      case NODE_MATH_MODULO: return safe_modulo(a, b);
      case NODE_MATH_ABSOLUTE: return abs(a);
      case NODE_MATH_ARCTAN2: return compatible_atan2(a, b);
      case NODE_MATH_FLOOR: return floor(a);
      case NODE_MATH_CEIL: return ceil(a);
      case NODE_MATH_FRACTION: return a - floor(a);
      case NODE_MATH_SQRT: return sqrt(luisa::compute::max(a, 0.0f));
      case NODE_MATH_INV_SQRT: return inverse_sqrt(a);
      case NODE_MATH_SIGN: return compatible_sign(a);
      case NODE_MATH_EXPONENT: return exp(a);
      case NODE_MATH_RADIANS: return a * (3.14159265358979323846f / 180.0f);
      case NODE_MATH_DEGREES: return a * (180.0f / 3.14159265358979323846f);
      case NODE_MATH_SINH: return sinh(a);
      case NODE_MATH_COSH: return cosh(a);
      case NODE_MATH_TANH: return tanh(a);
      case NODE_MATH_TRUNC: return select(ceil(a), floor(a), a >= 0.0f);
      case NODE_MATH_SNAP: return floor(safe_divide(a, b)) * b;
      case NODE_MATH_WRAP: return wrap(a, b, c);
      case NODE_MATH_COMPARE:
        return select(0.0f, 1.0f,
                      (a == b) |
                          (abs(a - b) <= luisa::compute::max(
                                             c, 1.1920928955078125e-7f)));
      case NODE_MATH_MULTIPLY_ADD: return a * b + c;
      case NODE_MATH_PINGPONG: return pingpong(a, b);
      case NODE_MATH_SMOOTH_MIN: return smooth_min(a, b, c);
      case NODE_MATH_SMOOTH_MAX: return -smooth_min(-a, -b, c);
      case NODE_MATH_FLOORED_MODULO: return safe_floored_modulo(a, b);
      default: return 0.0f;
    }
  };
  static constexpr NodeMathType all_ops[] = {
      NODE_MATH_ADD,          NODE_MATH_SUBTRACT,     NODE_MATH_MULTIPLY,
      NODE_MATH_DIVIDE,       NODE_MATH_SINE,         NODE_MATH_COSINE,
      NODE_MATH_TANGENT,      NODE_MATH_ARCSINE,      NODE_MATH_ARCCOSINE,
      NODE_MATH_ARCTANGENT,   NODE_MATH_POWER,        NODE_MATH_LOGARITHM,
      NODE_MATH_MINIMUM,      NODE_MATH_MAXIMUM,      NODE_MATH_ROUND,
      NODE_MATH_LESS_THAN,    NODE_MATH_GREATER_THAN, NODE_MATH_MODULO,
      NODE_MATH_ABSOLUTE,     NODE_MATH_ARCTAN2,      NODE_MATH_FLOOR,
      NODE_MATH_CEIL,         NODE_MATH_FRACTION,     NODE_MATH_SQRT,
      NODE_MATH_INV_SQRT,     NODE_MATH_SIGN,         NODE_MATH_EXPONENT,
      NODE_MATH_RADIANS,      NODE_MATH_DEGREES,      NODE_MATH_SINH,
      NODE_MATH_COSH,         NODE_MATH_TANH,         NODE_MATH_TRUNC,
      NODE_MATH_SNAP,         NODE_MATH_WRAP,         NODE_MATH_COMPARE,
      NODE_MATH_MULTIPLY_ADD, NODE_MATH_PINGPONG,     NODE_MATH_SMOOTH_MIN,
      NODE_MATH_SMOOTH_MAX,   NODE_MATH_FLOORED_MODULO};
  std::uint32_t active_count = 0u;
  NodeMathType single_op = NODE_MATH_ADD;
  for (const auto op : all_ops) {
    if (enabled(op)) {
      ++active_count;
      single_op = op;
    }
  }
  if (active_count == 0u) {
    return 0.0f;
  }
  if (active_count == 1u && type_mask != ~0ull) {
    return eval_one(single_op);
  }
  Float result = 0.0f;
  $switch (type) {
    for (const auto op : all_ops) {
      if (enabled(op)) {
        PSYCLES_SVM_CASE(op) { result = eval_one(op); };
      }
    }
    $default { result = 0.0f; };
  };
  return result;
}

} // namespace

luisa::compute::Float svm_math(luisa::compute::Expr<std::uint32_t> type,
                               luisa::compute::Expr<float> a_value,
                               luisa::compute::Expr<float> b_value,
                               luisa::compute::Expr<float> c_value) noexcept {
  return svm_math_masked(type, a_value, b_value, c_value, ~0ull);
}

void node_math(Cursor &cursor, Stack &stack,
               std::uint64_t type_mask) noexcept {
  const auto math_type = cursor.word();
  const auto value1 = cursor.word();
  const auto value2 = cursor.word();
  const auto value3 = cursor.word();
  const auto result_word = cursor.word();
  const auto result_offset = cursor.byte(result_word, 0u);

  if (type_mask == 0ull) {
    stack_store_float(stack, result_offset, 0.0f);
    return;
  }
  static constexpr std::uint64_t ternary_mask =
      (1ull << NODE_MATH_WRAP) | (1ull << NODE_MATH_COMPARE) |
      (1ull << NODE_MATH_MULTIPLY_ADD) | (1ull << NODE_MATH_SMOOTH_MIN) |
      (1ull << NODE_MATH_SMOOTH_MAX);
  static constexpr std::uint64_t unary_only_mask =
      (1ull << NODE_MATH_SINE) | (1ull << NODE_MATH_COSINE) |
      (1ull << NODE_MATH_TANGENT) | (1ull << NODE_MATH_ARCSINE) |
      (1ull << NODE_MATH_ARCCOSINE) | (1ull << NODE_MATH_ARCTANGENT) |
      (1ull << NODE_MATH_ROUND) | (1ull << NODE_MATH_ABSOLUTE) |
      (1ull << NODE_MATH_FLOOR) | (1ull << NODE_MATH_CEIL) |
      (1ull << NODE_MATH_FRACTION) | (1ull << NODE_MATH_SQRT) |
      (1ull << NODE_MATH_INV_SQRT) | (1ull << NODE_MATH_SIGN) |
      (1ull << NODE_MATH_EXPONENT) | (1ull << NODE_MATH_RADIANS) |
      (1ull << NODE_MATH_DEGREES) | (1ull << NODE_MATH_SINH) |
      (1ull << NODE_MATH_COSH) | (1ull << NODE_MATH_TANH) |
      (1ull << NODE_MATH_TRUNC);
  const bool needs_b = (type_mask & ~unary_only_mask) != 0ull;
  const bool needs_c = (type_mask & ternary_mask) != 0ull;

  const auto a = stack_load_input_float(stack, value1);
  const Float b = needs_b ? stack_load_input_float(stack, value2) : Float{0.0f};
  const Float c = needs_c ? stack_load_input_float(stack, value3) : Float{0.0f};
  stack_store_float(stack, result_offset,
                    svm_math_masked(math_type, a, b, c, type_mask));
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_SVM_CASE

/* SPDX-FileCopyrightText: 2011-2025 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_internal.h"

#include "surface_math.h"
#include "surface_vector_math.h"

#include <luisa/dsl/sugar.h>

#define PSYCLES_VECTOR_MATH_CASE(node) \
  $case(static_cast<std::uint32_t>(node))

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

struct DualVectorMathResult {
  Dual1 value;
  Dual3 vector;
};

[[nodiscard]] Dual1 zero_dual1() noexcept {
  return {.val = 0.0f, .dx = 0.0f, .dy = 0.0f};
}

[[nodiscard]] Dual3 zero_dual3() noexcept {
  return {.val = make_float3(0.0f),
          .dx = make_float3(0.0f),
          .dy = make_float3(0.0f)};
}

[[nodiscard]] Dual1 constant_dual1(Float value) noexcept {
  return {.val = value, .dx = 0.0f, .dy = 0.0f};
}

[[nodiscard]] Dual3 constant_dual3(Float3 value) noexcept {
  return {.val = value,
          .dx = make_float3(0.0f),
          .dy = make_float3(0.0f)};
}

[[nodiscard]] Dual3 select_dual3_components(Bool3 condition,
                                            const Dual3 &if_true,
                                            const Dual3 &if_false) noexcept {
  return {.val = select(if_false.val, if_true.val, condition),
          .dx = select(if_false.dx, if_true.dx, condition),
          .dy = select(if_false.dy, if_true.dy, condition)};
}

[[nodiscard]] Dual3 select_dual3(Bool condition, const Dual3 &if_true,
                                 const Dual3 &if_false) noexcept {
  return {.val = select(if_false.val, if_true.val, condition),
          .dx = select(if_false.dx, if_true.dx, condition),
          .dy = select(if_false.dy, if_true.dy, condition)};
}

[[nodiscard]] Dual3 dual_negate(const Dual3 &value) noexcept {
  return {.val = -value.val, .dx = -value.dx, .dy = -value.dy};
}

[[nodiscard]] Dual1 dual_add(const Dual1 &a, const Dual1 &b) noexcept {
  return {.val = a.val + b.val, .dx = a.dx + b.dx, .dy = a.dy + b.dy};
}

[[nodiscard]] Dual3 dual_add(const Dual3 &a, const Dual3 &b) noexcept {
  return {.val = a.val + b.val, .dx = a.dx + b.dx, .dy = a.dy + b.dy};
}

[[nodiscard]] Dual1 dual_subtract(const Dual1 &a,
                                  const Dual1 &b) noexcept {
  return {.val = a.val - b.val, .dx = a.dx - b.dx, .dy = a.dy - b.dy};
}

[[nodiscard]] Dual3 dual_subtract(const Dual3 &a,
                                  const Dual3 &b) noexcept {
  return {.val = a.val - b.val, .dx = a.dx - b.dx, .dy = a.dy - b.dy};
}

[[nodiscard]] Dual1 dual_multiply(const Dual1 &a,
                                  const Dual1 &b) noexcept {
  return {.val = a.val * b.val,
          .dx = a.val * b.dx + a.dx * b.val,
          .dy = a.val * b.dy + a.dy * b.val};
}

[[nodiscard]] Dual3 dual_multiply(const Dual3 &a,
                                  const Dual3 &b) noexcept {
  return {.val = a.val * b.val,
          .dx = a.val * b.dx + a.dx * b.val,
          .dy = a.val * b.dy + a.dy * b.val};
}

[[nodiscard]] Dual3 dual_multiply(const Dual3 &a,
                                  const Dual1 &b) noexcept {
  return {.val = a.val * b.val,
          .dx = a.val * b.dx + a.dx * b.val,
          .dy = a.val * b.dy + a.dy * b.val};
}

[[nodiscard]] Dual3 dual_scale_components(const Dual3 &a,
                                          Float3 b) noexcept {
  return {.val = a.val * b, .dx = a.dx * b, .dy = a.dy * b};
}

[[nodiscard]] Dual3 dual_scale(const Dual3 &a, Float b) noexcept {
  return {.val = a.val * b, .dx = a.dx * b, .dy = a.dy * b};
}

[[nodiscard]] Dual1 dual_divide_unchecked(const Dual1 &numerator,
                                          const Dual1 &denominator) noexcept {
  const Float inverse = 1.0f / denominator.val;
  const Float quotient = numerator.val / denominator.val;
  return {.val = quotient,
          .dx = (numerator.dx - quotient * denominator.dx) * inverse,
          .dy = (numerator.dy - quotient * denominator.dy) * inverse};
}

[[nodiscard]] Dual3 dual_divide_unchecked(const Dual3 &numerator,
                                          const Dual3 &denominator) noexcept {
  const Float3 inverse = 1.0f / denominator.val;
  const Float3 quotient = numerator.val / denominator.val;
  return {.val = quotient,
          .dx = (numerator.dx - quotient * denominator.dx) * inverse,
          .dy = (numerator.dy - quotient * denominator.dy) * inverse};
}

[[nodiscard]] Dual3 safe_dual_divide(const Dual3 &numerator,
                                     const Dual3 &denominator) noexcept {
  const Bool3 nonzero = denominator.val != make_float3(0.0f);
  const Dual3 safe_denominator{
      .val = select(make_float3(1.0f), denominator.val, nonzero),
      .dx = denominator.dx,
      .dy = denominator.dy};
  return select_dual3_components(
      nonzero, dual_divide_unchecked(numerator, safe_denominator),
      zero_dual3());
}

[[nodiscard]] Dual1 dual_dot(const Dual3 &a, const Dual3 &b) noexcept {
  return {.val = dot(a.val, b.val),
          .dx = dot(a.val, b.dx) + dot(a.dx, b.val),
          .dy = dot(a.val, b.dy) + dot(a.dy, b.val)};
}

[[nodiscard]] Dual1 dual_sqrt(const Dual1 &value) noexcept {
  const Float result = sqrt(value.val);
  const Float derivative = 0.5f / result;
  return {.val = result,
          .dx = derivative * value.dx,
          .dy = derivative * value.dy};
}

[[nodiscard]] Dual1 dual_length(const Dual3 &value) noexcept {
  return dual_sqrt(dual_dot(value, value));
}

[[nodiscard]] Dual3 dual_cross(const Dual3 &a, const Dual3 &b) noexcept {
  return {.val = cross(a.val, b.val),
          .dx = cross(a.val, b.dx) + cross(a.dx, b.val),
          .dy = cross(a.val, b.dy) + cross(a.dy, b.val)};
}

[[nodiscard]] Dual3 dual_floor(const Dual3 &value) noexcept {
  return constant_dual3(floor(value.val));
}

[[nodiscard]] Dual3 dual_ceil(const Dual3 &value) noexcept {
  return constant_dual3(ceil(value.val));
}

[[nodiscard]] Float3 safe_fmod_vector(Float3 a, Float3 b) noexcept {
  const Bool3 nonzero = b != make_float3(0.0f);
  const Float3 safe_b = select(make_float3(1.0f), b, nonzero);
  return select(make_float3(0.0f), fmod(a, safe_b), nonzero);
}

[[nodiscard]] Dual3 dual_safe_fmod(const Dual3 &a,
                                   const Dual3 &b) noexcept {
  // Cycles intentionally treats modulo's derivatives as a passthrough from
  // the dividend; the divisor derivatives do not participate.
  return {.val = safe_fmod_vector(a.val, b.val), .dx = a.dx, .dy = a.dy};
}

[[nodiscard]] Dual3 dual_safe_floored_fmod(const Dual3 &a,
                                           const Dual3 &b) noexcept {
  const Bool3 nonzero = b.val != make_float3(0.0f);
  const Float3 safe_b = select(make_float3(1.0f), b.val, nonzero);
  const Float3 quotient_floor = floor(a.val / safe_b);
  const auto candidate =
      dual_subtract(a, dual_scale_components(b, quotient_floor));
  return select_dual3_components(nonzero, candidate, zero_dual3());
}

[[nodiscard]] Dual3 dual_wrap(const Dual3 &value, const Dual3 &maximum,
                              const Dual3 &minimum) noexcept {
  return dual_add(
      dual_safe_floored_fmod(dual_subtract(value, minimum),
                             dual_subtract(maximum, minimum)),
      minimum);
}

[[nodiscard]] Dual3 dual_minimum(const Dual3 &a, const Dual3 &b) noexcept {
  return select_dual3_components(a.val < b.val, a, b);
}

[[nodiscard]] Dual3 dual_maximum(const Dual3 &a, const Dual3 &b) noexcept {
  return select_dual3_components(a.val > b.val, a, b);
}

[[nodiscard]] Dual3 dual_absolute(const Dual3 &value) noexcept {
  return select_dual3_components(value.val > make_float3(0.0f), value,
                                 dual_negate(value));
}

[[nodiscard]] Dual3 dual_project(const Dual3 &value,
                                 const Dual3 &onto) noexcept {
  const Dual1 length_squared = dual_dot(onto, onto);
  const Bool nonzero = length_squared.val != 0.0f;
  const Dual1 safe_length_squared{
      .val = select(1.0f, length_squared.val, nonzero),
      .dx = length_squared.dx,
      .dy = length_squared.dy};
  const auto projected = dual_multiply(
      onto, dual_divide_unchecked(dual_dot(value, onto),
                                  safe_length_squared));
  return select_dual3(nonzero, projected, zero_dual3());
}

[[nodiscard]] Dual3 dual_reflect(const Dual3 &incident,
                                 const Dual3 &normal) noexcept {
  const Dual3 unit_normal = safe_normalize_dual(normal);
  return dual_subtract(
      incident,
      dual_scale(dual_multiply(unit_normal,
                               dual_dot(incident, unit_normal)),
                 Float{2.0f}));
}

[[nodiscard]] Dual3 dual_refract(const Dual3 &incident,
                                 const Dual3 &normal,
                                 const Dual1 &eta) noexcept {
  const Dual3 unit_normal = safe_normalize_dual(normal);
  const Dual1 normal_incident = dual_dot(incident, unit_normal);
  const Dual1 k = dual_subtract(
      constant_dual1(1.0f),
      dual_multiply(dual_multiply(eta, eta),
                    dual_subtract(constant_dual1(1.0f),
                                  dual_multiply(normal_incident,
                                                normal_incident))));
  Dual3 result = zero_dual3();
  $if (k.val >= 0.0f) {
    result = dual_subtract(
        dual_multiply(incident, eta),
        dual_multiply(
            unit_normal,
            dual_add(dual_multiply(eta, normal_incident), dual_sqrt(k))));
  };
  return result;
}

[[nodiscard]] Dual3 dual_faceforward(const Dual3 &vector,
                                     const Dual3 &incident,
                                     const Dual3 &reference) noexcept {
  return select_dual3(dual_dot(reference, incident).val < 0.0f, vector,
                      dual_negate(vector));
}

[[nodiscard]] Dual3 dual_sine(const Dual3 &value) noexcept {
  const Float3 sine = sin(value.val);
  const Float3 derivative = cos(value.val);
  return {.val = sine,
          .dx = derivative * value.dx,
          .dy = derivative * value.dy};
}

[[nodiscard]] Dual3 dual_cosine(const Dual3 &value) noexcept {
  const Float3 cosine = cos(value.val);
  const Float3 derivative = -sin(value.val);
  return {.val = cosine,
          .dx = derivative * value.dx,
          .dy = derivative * value.dy};
}

[[nodiscard]] Dual3 dual_tangent(const Dual3 &value) noexcept {
  const Float3 cosine = cos(value.val);
  const Bool3 nonzero = cosine != make_float3(0.0f);
  const Float3 safe_cosine = select(make_float3(1.0f), cosine, nonzero);
  const Float3 secant =
      select(make_float3(0.0f), 1.0f / safe_cosine, nonzero);
  const Float3 derivative = secant * secant;
  return {.val = tan(value.val),
          .dx = derivative * value.dx,
          .dy = derivative * value.dy};
}

[[nodiscard]] Float3 safe_power_vector(Float3 base,
                                       Float3 exponent) noexcept {
  return make_float3(
      ::psycles::luisa_backend::detail::cycles_safe_power(base.x, exponent.x),
      ::psycles::luisa_backend::detail::cycles_safe_power(base.y, exponent.y),
      ::psycles::luisa_backend::detail::cycles_safe_power(base.z,
                                                          exponent.z));
}

[[nodiscard]] Float3 safe_log_vector(Float3 value) noexcept {
  const Bool3 positive = value > make_float3(0.0f);
  const Float3 safe_value = select(make_float3(1.0f), value, positive);
  return select(make_float3(0.0f), log(safe_value), positive);
}

[[nodiscard]] Dual3 dual_safe_power(const Dual3 &base,
                                    const Dual3 &exponent) noexcept {
  const Float3 base_to_exponent_minus_one =
      safe_power_vector(base.val, exponent.val - 1.0f);
  const Float3 result = base.val * base_to_exponent_minus_one;
  const Float3 derivative_base =
      exponent.val * base_to_exponent_minus_one;
  const Float3 derivative_exponent = result * safe_log_vector(base.val);
  return {.val = result,
          .dx = derivative_base * base.dx +
                derivative_exponent * exponent.dx,
          .dy = derivative_base * base.dy +
                derivative_exponent * exponent.dy};
}

[[nodiscard]] ::psycles::luisa_backend::detail::SurfaceVectorMathResult
evaluate_primal(UInt type, Float3 a, Float3 b, Float3 c,
                Float param1) noexcept {
  using GraphOperation = compiler::VectorMathOperation;
  ::psycles::luisa_backend::detail::SurfaceVectorMathResult result{
      .value = 0.0f, .vector = make_float3(0.0f)};
#define PSYCLES_PRIMAL_CASE(node, operation)                                \
  PSYCLES_VECTOR_MATH_CASE(node) {                                         \
    result = ::psycles::luisa_backend::detail::                            \
        evaluate_surface_vector_math_operation(GraphOperation::operation, \
                                                a, b, c, param1);          \
  };
  $switch (type) {
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_ADD, add)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_SUBTRACT, subtract)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_MULTIPLY, multiply)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_DIVIDE, divide)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_CROSS_PRODUCT, cross_product)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_PROJECT, project)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_REFLECT, reflect)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_DOT_PRODUCT, dot_product)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_DISTANCE, distance)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_LENGTH, length)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_SCALE, scale)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_NORMALIZE, normalize)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_SNAP, snap)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_FLOOR, floor)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_CEIL, ceil)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_MODULO, modulo)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_FRACTION, fraction)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_ABSOLUTE, absolute)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_MINIMUM, minimum)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_MAXIMUM, maximum)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_WRAP, wrap)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_SINE, sine)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_COSINE, cosine)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_TANGENT, tangent)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_REFRACT, refract)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_FACEFORWARD, faceforward)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_MULTIPLY_ADD, multiply_add)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_POWER, power)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_SIGN, sign)
    PSYCLES_PRIMAL_CASE(NODE_VECTOR_MATH_ROUND, round)
    $default {};
  };
#undef PSYCLES_PRIMAL_CASE
  return result;
}

[[nodiscard]] DualVectorMathResult evaluate_dual(
    UInt type, const Dual3 &a, const Dual3 &b, const Dual3 &c,
    const Dual1 &param1) noexcept {
  DualVectorMathResult result{.value = zero_dual1(),
                              .vector = zero_dual3()};
  $switch (type) {
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_ADD) {
      result.vector = dual_add(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_SUBTRACT) {
      result.vector = dual_subtract(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_MULTIPLY) {
      result.vector = dual_multiply(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_DIVIDE) {
      result.vector = safe_dual_divide(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_CROSS_PRODUCT) {
      result.vector = dual_cross(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_PROJECT) {
      result.vector = dual_project(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_REFLECT) {
      result.vector = dual_reflect(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_DOT_PRODUCT) {
      result.value = dual_dot(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_DISTANCE) {
      result.value = dual_length(dual_subtract(a, b));
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_LENGTH) {
      result.value = dual_length(a);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_SCALE) {
      result.vector = dual_multiply(a, param1);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_NORMALIZE) {
      result.vector = safe_normalize_dual(a);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_SNAP) {
      result.vector = dual_multiply(dual_floor(safe_dual_divide(a, b)), b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_FLOOR) {
      result.vector = dual_floor(a);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_CEIL) {
      result.vector = dual_ceil(a);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_MODULO) {
      result.vector = dual_safe_fmod(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_FRACTION) {
      result.vector = dual_subtract(a, dual_floor(a));
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_ABSOLUTE) {
      result.vector = dual_absolute(a);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_MINIMUM) {
      result.vector = dual_minimum(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_MAXIMUM) {
      result.vector = dual_maximum(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_WRAP) {
      result.vector = dual_wrap(a, b, c);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_SINE) {
      result.vector = dual_sine(a);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_COSINE) {
      result.vector = dual_cosine(a);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_TANGENT) {
      result.vector = dual_tangent(a);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_REFRACT) {
      result.vector = dual_refract(a, b, param1);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_FACEFORWARD) {
      result.vector = dual_faceforward(a, b, c);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_MULTIPLY_ADD) {
      result.vector = dual_add(dual_multiply(a, b), c);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_POWER) {
      result.vector = dual_safe_power(a, b);
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_SIGN) {
      result.vector = constant_dual3(
          make_float3(select(select(1.0f, -1.0f, a.val.x < 0.0f), 0.0f,
                             a.val.x == 0.0f),
                      select(select(1.0f, -1.0f, a.val.y < 0.0f), 0.0f,
                             a.val.y == 0.0f),
                      select(select(1.0f, -1.0f, a.val.z < 0.0f), 0.0f,
                             a.val.z == 0.0f)));
    };
    PSYCLES_VECTOR_MATH_CASE(NODE_VECTOR_MATH_ROUND) {
      result.vector = dual_floor(
          dual_add(a, constant_dual3(make_float3(0.5f))));
    };
    $default {};
  };
  return result;
}

} // namespace

void node_vector_math(Cursor &cursor, Stack &stack,
                      bool use_derivatives) noexcept {
  const UInt math_type = cursor.word();
  const UInt a_x = cursor.word();
  const UInt a_y = cursor.word();
  const UInt a_z = cursor.word();
  const UInt b_x = cursor.word();
  const UInt b_y = cursor.word();
  const UInt b_z = cursor.word();
  const UInt c_x = cursor.word();
  const UInt c_y = cursor.word();
  const UInt c_z = cursor.word();
  const UInt param1_bits = cursor.word();
  const UInt packed_outputs = cursor.word();
  const UInt value_offset = cursor.byte(packed_outputs, 0u);
  const UInt vector_offset = cursor.byte(packed_outputs, 1u);

  if (use_derivatives) {
    const auto result = evaluate_dual(
        math_type, stack_load_input_dual_float3(stack, a_x, a_y, a_z),
        stack_load_input_dual_float3(stack, b_x, b_y, b_z),
        stack_load_input_dual_float3(stack, c_x, c_y, c_z),
        stack_load_input_dual_float(stack, param1_bits));
    $if (value_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      stack_store_dual1(stack, value_offset, result.value);
    };
    $if (vector_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      stack_store_dual3(stack, vector_offset, result.vector);
    };
  } else {
    const auto result = evaluate_primal(
        math_type, stack_load_input_float3(stack, a_x, a_y, a_z),
        stack_load_input_float3(stack, b_x, b_y, b_z),
        stack_load_input_float3(stack, c_x, c_y, c_z),
        stack_load_input_float(stack, param1_bits));
    $if (value_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      stack_store_float(stack, value_offset, result.value);
    };
    $if (vector_offset != static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      stack_store_float3(stack, vector_offset, result.vector);
    };
  }
}

} // namespace psycles::luisa_backend::cycles_svm::detail

#undef PSYCLES_VECTOR_MATH_CASE

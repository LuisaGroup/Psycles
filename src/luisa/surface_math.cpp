#include "surface_math.h"
#include "surface_math_constants.h"

#include <cstdlib>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

using namespace luisa::compute;

Float3 safe_normalize(
    Float3 value,
    Float3 fallback) noexcept {
    const auto valid = dot(value, value) > 1.0e-20f;
    auto selected = select(fallback, value, valid);
    const auto fallback_valid =
        dot(selected, selected) > 1.0e-20f;
    selected = select(
        make_float3(0.0f, 0.0f, 1.0f),
        selected,
        fallback_valid);
    return normalize(selected);
}

Float cycles_safe_power(
    Float base,
    Float exponent) noexcept {
    const auto base_is_zero = base == 0.0f;
    const auto exponent_is_zero = exponent == 0.0f;
    const auto integer_exponent = exponent == trunc(exponent);

    // Cycles returns before powf for a zero base. Selecting a benign operand
    // models that total control-flow semantics without recording an invalid
    // 0^negative operation into a branchless backend AST.
    const auto magnitude = pow(select(abs(base), 1.0f, base_is_zero), exponent);
    const auto odd_exponent = fmod(abs(exponent), 2.0f) != 0.0f;
    const auto signed_magnitude = select(
        magnitude,
        -magnitude,
        (base < 0.0f) & odd_exponent);
    auto result = select(
        0.0f,
        signed_magnitude,
        (base >= 0.0f) | integer_exponent);
    result = select(
        result,
        0.0f,
        base_is_zero & (exponent != 0.0f));
    return select(result, 1.0f, exponent_is_zero);
}

Float evaluate_surface_math_operation(
    compiler::MathOperation operation,
    Float a,
    Float b,
    Float c) noexcept {
    switch (operation) {
        case compiler::MathOperation::add:
            return a + b;
        case compiler::MathOperation::subtract:
            return a - b;
        case compiler::MathOperation::multiply:
            return a * b;
        case compiler::MathOperation::divide:
            return select(0.0f, a / b, b != 0.0f);
        case compiler::MathOperation::multiply_add:
            return a * b + c;
        case compiler::MathOperation::power:
            return cycles_safe_power(a, b);
        case compiler::MathOperation::logarithm: {
            auto denominator = log(b);
            return select(0.0f,
                          log(a) / denominator,
                          (a > 0.0f) & (b > 0.0f) & (denominator != 0.0f));
        }
        case compiler::MathOperation::square_root:
            return sqrt(max(a, 0.0f));
        case compiler::MathOperation::inverse_square_root: {
            const auto positive = a > 0.0f;
            return select(
                0.0f,
                rsqrt(select(1.0f, a, positive)),
                positive);
        }
        case compiler::MathOperation::absolute:
            return abs(a);
        case compiler::MathOperation::exponent:
            return exp(a);
        case compiler::MathOperation::minimum:
            return min(a, b);
        case compiler::MathOperation::maximum:
            return max(a, b);
        case compiler::MathOperation::less_than:
            return select(0.0f, 1.0f, a < b);
        case compiler::MathOperation::greater_than:
            return select(0.0f, 1.0f, a > b);
        case compiler::MathOperation::sign:
            return select(select(1.0f, -1.0f, a < 0.0f), 0.0f, a == 0.0f);
        case compiler::MathOperation::compare:
            return select(0.0f,
                          1.0f,
                          (a == b) |
                              (abs(a - b) <=
                               max(c, 1.1920928955078125e-7f)));
        case compiler::MathOperation::smooth_minimum: {
            auto nonzero = c != 0.0f;
            auto h = max(c - abs(a - b), 0.0f) / c;
            auto smooth = min(a, b) - h * h * h * c * (1.0f / 6.0f);
            return select(min(a, b), smooth, nonzero);
        }
        case compiler::MathOperation::smooth_maximum: {
            auto nonzero = c != 0.0f;
            auto h = max(c - abs(a - b), 0.0f) / c;
            auto smooth = max(a, b) + h * h * h * c * (1.0f / 6.0f);
            return select(max(a, b), smooth, nonzero);
        }
        case compiler::MathOperation::round:
            return floor(a + 0.5f);
        case compiler::MathOperation::floor:
            return floor(a);
        case compiler::MathOperation::ceil:
            return ceil(a);
        case compiler::MathOperation::trunc:
            return trunc(a);
        case compiler::MathOperation::fraction:
            return a - floor(a);
        case compiler::MathOperation::modulo:
            return select(0.0f, fmod(a, b), b != 0.0f);
        case compiler::MathOperation::floored_modulo:
            return select(0.0f, a - floor(a / b) * b, b != 0.0f);
        case compiler::MathOperation::wrap: {
            auto range = b - c;
            return select(c,
                          a - range * floor((a - c) / range),
                          range != 0.0f);
        }
        case compiler::MathOperation::snap:
            return floor(select(0.0f, a / b, b != 0.0f)) * b;
        case compiler::MathOperation::ping_pong:
            return select(0.0f,
                          abs(fract((a - b) / (b * 2.0f)) * b * 2.0f - b),
                          b != 0.0f);
        case compiler::MathOperation::sine:
            return sin(a);
        case compiler::MathOperation::cosine:
            return cos(a);
        case compiler::MathOperation::tangent:
            return tan(a);
        case compiler::MathOperation::arcsine:
            return asin(clamp(a, -1.0f, 1.0f));
        case compiler::MathOperation::arccosine:
            return acos(clamp(a, -1.0f, 1.0f));
        case compiler::MathOperation::arctangent:
            return atan(a);
        case compiler::MathOperation::arctangent2:
            return select(atan2(a, b), 0.0f, (a == 0.0f) & (b == 0.0f));
        case compiler::MathOperation::hyperbolic_sine:
            return sinh(a);
        case compiler::MathOperation::hyperbolic_cosine:
            return cosh(a);
        case compiler::MathOperation::hyperbolic_tangent:
            return tanh(a);
        case compiler::MathOperation::radians:
            return a * (pi / 180.0f);
        case compiler::MathOperation::degrees:
            return a * (180.0f / pi);
    }
    std::abort();
}

} // namespace psycles::luisa_backend::detail

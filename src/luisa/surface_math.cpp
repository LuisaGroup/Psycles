#include "surface_math.h"
#include "surface_math_constants.h"

#include <array>
#include <cstdlib>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

using SurfaceMathSvmCallable =
    luisa::compute::Callable<float(luisa::uint, float, float, float)>;

[[nodiscard]] auto active_math_operations(
    std::span<const std::uint16_t> immediate_domain) noexcept {
    std::array<bool, compiler::math_operation_count> result{};
    for (const auto encoded : immediate_domain) {
        if (encoded >= compiler::math_operation_count) {
            std::abort();
        }
        result[encoded] = true;
    }
    return result;
}

} // namespace

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
        case compiler::MathOperation::power: {
            auto integer_exponent = b == trunc(b);
            auto powered = pow(abs(a), b);
            auto odd_exponent = fmod(abs(b), 2.0f) != 0.0f;
            powered = select(powered, -powered, (a < 0.0f) & odd_exponent);
            return select(0.0f, powered, (a >= 0.0f) | integer_exponent);
        }
        case compiler::MathOperation::logarithm: {
            auto denominator = log(b);
            return select(0.0f,
                          log(a) / denominator,
                          (a > 0.0f) & (b > 0.0f) & (denominator != 0.0f));
        }
        case compiler::MathOperation::square_root:
            return sqrt(max(a, 0.0f));
        case compiler::MathOperation::inverse_square_root:
            return select(0.0f, 1.0f / sqrt(a), a > 0.0f);
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

Float evaluate_surface_math_svm(
    UInt immediate,
    std::span<const std::uint16_t> immediate_domain,
    Float a,
    Float b,
    Float c) noexcept {
    const auto active_operations = active_math_operations(immediate_domain);
    SurfaceMathSvmCallable callable =
        [active_operations](UInt operation,
                            Float input_a,
                            Float input_b,
                            Float input_c) noexcept {
            Float result = 0.0f;
            luisa::compute::detail::SwitchStmtBuilder{operation} % [&] {
                for (auto index = std::size_t{0u};
                     index < active_operations.size();
                     ++index) {
                    if (!active_operations[index]) {
                        continue;
                    }
                    luisa::compute::detail::SwitchCaseStmtBuilder{
                        static_cast<luisa::uint>(index)} %
                        [&, index] {
                            result = evaluate_surface_math_operation(
                                static_cast<compiler::MathOperation>(index),
                                input_a,
                                input_b,
                                input_c);
                        };
                }
                luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                    luisa::compute::dsl::unreachable(
                        "invalid compact surface Math operation");
                };
            };
            return result;
        };
    callable.set_name("surface_math_svm");
    return callable(immediate, a, b, c);
}

} // namespace psycles::luisa_backend::detail

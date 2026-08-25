#include "graph_surface_internal.h"
#include "surface_math.h"
#include "surface_mix.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] bool supports_math_value(
    compiler::ValueOperation operation) noexcept {
    switch (operation) {
        case compiler::ValueOperation::parameter:
        case compiler::ValueOperation::passthrough:
        case compiler::ValueOperation::scalar_to_color:
        case compiler::ValueOperation::scalar_to_boolean:
        case compiler::ValueOperation::color_to_scalar:
        case compiler::ValueOperation::vector_to_scalar:
        case compiler::ValueOperation::add:
        case compiler::ValueOperation::subtract:
        case compiler::ValueOperation::multiply:
        case compiler::ValueOperation::divide:
        case compiler::ValueOperation::minimum:
        case compiler::ValueOperation::maximum:
        case compiler::ValueOperation::power:
        case compiler::ValueOperation::math:
        case compiler::ValueOperation::absolute:
        case compiler::ValueOperation::clamp01:
        case compiler::ValueOperation::clamp_range:
        case compiler::ValueOperation::map_range_float:
        case compiler::ValueOperation::map_range_vector:
        case compiler::ValueOperation::vector_math_value:
        case compiler::ValueOperation::vector_math_vector:
        case compiler::ValueOperation::mix_float:
        case compiler::ValueOperation::mix_vector:
        case compiler::ValueOperation::mix:
            return true;
        default:
            return false;
    }
}

class MathValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        [[maybe_unused]] const auto &services = context.services;
        [[maybe_unused]] const auto &point = context.point;
        [[maybe_unused]] auto &result = context.result;
        const auto &instruction = this->instruction();
        Float4 value = make_float4(0.0f);
        switch (instruction.operation) {
                case compiler::ValueOperation::parameter:
                    switch (surface_value_category(
                        instruction.result_type)) {
                        case SurfaceValueCategory::scalar: {
                            auto parameter = services.parameter_float(
                                point.parameter_block,
                                instruction.parameter.value);
                            return SurfaceValueExpression::from_scalar(
                                Expr<float>{parameter.expression()});
                        }
                        case SurfaceValueCategory::vector: {
                            auto parameter = services.parameter_float3(
                                point.parameter_block,
                                instruction.parameter.value);
                            return SurfaceValueExpression::from_vector(
                                Expr<luisa::float3>{
                                    parameter.expression()});
                        }
                        case SurfaceValueCategory::unsigned_integer: {
                            auto parameter = services.parameter_uint64(
                                point.parameter_block,
                                instruction.parameter.value);
                            return SurfaceValueExpression::from_unsigned_integer(
                                Expr<luisa::ulong>{
                                    parameter.expression()});
                        }
                    }
                    std::abort();
                case compiler::ValueOperation::passthrough:
                    return get(
                        instruction.operand(operand::unary::input),
                        result.values);
                case compiler::ValueOperation::scalar_to_color: {
                    auto x = scalar(
                        instruction.operand(operand::unary::input), result);
                    value = make_float4(x, x, x, 1.0f);
                    break;
                }
                case compiler::ValueOperation::scalar_to_boolean: {
                    // Cycles maps Blender Boolean sockets to INT and uses
                    // NODE_CONVERT_FI for a linked float. Preserve its
                    // truncation boundary exactly; values in (-1, 1) become
                    // false even though Blender's generic field conversion
                    // uses a different predicate.
                    auto x = scalar(
                        instruction.operand(operand::unary::input), result);
                    value = make_float4(select(
                        0.0f,
                        1.0f,
                        cast<int>(x) != 0));
                    break;
                }
                case compiler::ValueOperation::color_to_scalar: {
                    auto color = vector(
                        instruction.operand(operand::unary::input), result);
                    value = make_float4(
                        dot(
                            color,
                            // Blender 4.5 default scene-linear
                            // Film::rgb_to_y coefficients.
                            make_float3(
                                0.21267404f,
                                0.7151516f,
                                0.07217542f)));
                    break;
                }
                case compiler::ValueOperation::vector_to_scalar: {
                    auto vector_value = vector(
                        instruction.operand(operand::unary::input), result);
                    value = make_float4(
                        (vector_value.x +
                         vector_value.y +
                         vector_value.z) /
                        3.0f);
                    break;
                }
                case compiler::ValueOperation::add:
                    value = make_float4(
                        scalar(
                            instruction.operand(operand::binary::a), result) +
                        scalar(
                            instruction.operand(operand::binary::b), result));
                    break;
                case compiler::ValueOperation::subtract:
                    value = make_float4(
                        scalar(
                            instruction.operand(operand::binary::a), result) -
                        scalar(
                            instruction.operand(operand::binary::b), result));
                    break;
                case compiler::ValueOperation::multiply:
                    value = make_float4(
                        scalar(
                            instruction.operand(operand::binary::a), result) *
                        scalar(
                            instruction.operand(operand::binary::b), result));
                    break;
                case compiler::ValueOperation::divide: {
                    auto denominator =
                        scalar(
                            instruction.operand(operand::binary::b), result);
                    value = make_float4(select(
                        0.0f,
                        scalar(
                            instruction.operand(operand::binary::a), result) /
                            denominator,
                        abs(denominator) > 1.0e-20f));
                    break;
                }
                case compiler::ValueOperation::minimum:
                    value = make_float4(min(
                        scalar(
                            instruction.operand(operand::binary::a), result),
                        scalar(
                            instruction.operand(operand::binary::b), result)));
                    break;
                case compiler::ValueOperation::maximum:
                    value = make_float4(max(
                        scalar(
                            instruction.operand(operand::binary::a), result),
                        scalar(
                            instruction.operand(operand::binary::b), result)));
                    break;
                case compiler::ValueOperation::power:
                    value = make_float4(pow(
                        max(
                            scalar(
                                instruction.operand(operand::binary::a),
                                result),
                            0.0f),
                        scalar(
                            instruction.operand(operand::binary::b), result)));
                    break;
                case compiler::ValueOperation::math: {
                    auto a = scalar(
                        instruction.operand(operand::ternary::a), result);
                    auto b = scalar(
                        instruction.operand(operand::ternary::b), result);
                    auto c = scalar(
                        instruction.operand(operand::ternary::c), result);
                    Float evaluated = 0.0f;
                    if (context.svm_immediate_override != nullptr) {
                        evaluated = evaluate_surface_math_svm(
                            *context.svm_immediate_override,
                            context.svm_immediate_domain,
                            a,
                            b,
                            c);
                    } else {
                        evaluated = evaluate_surface_math_operation(
                            static_cast<compiler::MathOperation>(
                                instruction.static_u0),
                            a,
                            b,
                            c);
                    }
                    value = make_float4(evaluated);
                    break;
                }
                case compiler::ValueOperation::absolute:
                    value = make_float4(abs(
                        scalar(
                            instruction.operand(operand::unary::input),
                            result)));
                    break;
                case compiler::ValueOperation::clamp01:
                    value = make_float4(clamp(
                        scalar(
                            instruction.operand(operand::unary::input),
                            result),
                        0.0f,
                        1.0f));
                    break;
                case compiler::ValueOperation::clamp_range: {
                    auto input = scalar(
                        instruction.operand(operand::clamp_range::value),
                        result);
                    auto minimum = scalar(
                        instruction.operand(operand::clamp_range::minimum),
                        result);
                    auto maximum = scalar(
                        instruction.operand(operand::clamp_range::maximum),
                        result);
                    if (instruction.static_u0 == 1u) {
                        auto reverse = minimum > maximum;
                        auto original_minimum = minimum;
                        minimum = select(
                            minimum, maximum, reverse);
                        maximum = select(
                            maximum,
                            original_minimum,
                            reverse);
                    }
                    value = make_float4(
                        min(max(input, minimum), maximum));
                    break;
                }
                case compiler::ValueOperation::map_range_float: {
                    auto input = scalar(
                        instruction.operand(operand::map_range::value),
                        result);
                    auto from_min = scalar(
                        instruction.operand(operand::map_range::from_min),
                        result);
                    auto from_max = scalar(
                        instruction.operand(operand::map_range::from_max),
                        result);
                    auto to_min = scalar(
                        instruction.operand(operand::map_range::to_min),
                        result);
                    auto to_max = scalar(
                        instruction.operand(operand::map_range::to_max),
                        result);
                    auto steps = scalar(
                        instruction.operand(operand::map_range::steps),
                        result);
                    auto denominator = from_max - from_min;
                    auto has_range = denominator != 0.0f;
                    auto factor =
                        (input - from_min) /
                        select(1.0f, denominator, has_range);
                    if (instruction.static_u0 == 1u) {
                        factor = select(
                            0.0f,
                            floor(
                                factor * (steps + 1.0f)) /
                                select(
                                    1.0f,
                                    steps,
                                    steps > 0.0f),
                            steps > 0.0f);
                    } else if (
                        instruction.static_u0 == 2u) {
                        factor = clamp(
                            factor, 0.0f, 1.0f);
                        factor =
                            (3.0f - 2.0f * factor) *
                            (factor * factor);
                    } else if (
                        instruction.static_u0 == 3u) {
                        factor = clamp(
                            factor, 0.0f, 1.0f);
                        factor =
                            factor * factor * factor *
                            (factor *
                                     (factor * 6.0f - 15.0f) +
                             10.0f);
                    }
                    auto mapped =
                        to_min + factor * (to_max - to_min);
                    mapped = select(
                        0.0f, mapped, has_range);
                    if (instruction.static_u1 != 0u) {
                        auto minimum = min(to_min, to_max);
                        auto maximum = max(to_min, to_max);
                        mapped = min(
                            max(mapped, minimum), maximum);
                    }
                    value = make_float4(mapped);
                    break;
                }
                case compiler::ValueOperation::map_range_vector: {
                    auto input = vector(
                        instruction.operand(operand::map_range::value),
                        result);
                    auto from_min = vector(
                        instruction.operand(operand::map_range::from_min),
                        result);
                    auto from_max = vector(
                        instruction.operand(operand::map_range::from_max),
                        result);
                    auto to_min = vector(
                        instruction.operand(operand::map_range::to_min),
                        result);
                    auto to_max = vector(
                        instruction.operand(operand::map_range::to_max),
                        result);
                    auto steps = vector(
                        instruction.operand(operand::map_range::steps),
                        result);
                    auto numerator = input - from_min;
                    auto denominator = from_max - from_min;
                    auto safe_divide = [](
                                           Float numerator_component,
                                           Float denominator_component) {
                        auto nonzero =
                            denominator_component != 0.0f;
                        return select(
                            0.0f,
                            numerator_component /
                                select(
                                    1.0f,
                                    denominator_component,
                                    nonzero),
                            nonzero);
                    };
                    auto factor = make_float3(
                        safe_divide(
                            numerator.x, denominator.x),
                        safe_divide(
                            numerator.y, denominator.y),
                        safe_divide(
                            numerator.z, denominator.z));
                    if (instruction.static_u0 == 1u) {
                        auto stepped = [](
                                           Float factor_component,
                                           Float steps_component) {
                            auto valid =
                                steps_component > 0.0f;
                            return select(
                                0.0f,
                                floor(
                                    factor_component *
                                    (steps_component + 1.0f)) /
                                    select(
                                        1.0f,
                                        steps_component,
                                        valid),
                                valid);
                        };
                        factor = make_float3(
                            stepped(factor.x, steps.x),
                            stepped(factor.y, steps.y),
                            stepped(factor.z, steps.z));
                    } else if (
                        instruction.static_u0 == 2u) {
                        factor = clamp(
                            factor, 0.0f, 1.0f);
                        factor =
                            (make_float3(3.0f) -
                             2.0f * factor) *
                            (factor * factor);
                    } else if (
                        instruction.static_u0 == 3u) {
                        factor = clamp(
                            factor, 0.0f, 1.0f);
                        factor =
                            factor * factor * factor *
                            (factor *
                                     (factor * 6.0f - 15.0f) +
                             10.0f);
                    }
                    auto mapped =
                        to_min + factor * (to_max - to_min);
                    if (instruction.static_u1 != 0u &&
                        instruction.static_u0 < 2u) {
                        mapped = min(
                            max(mapped, min(to_min, to_max)),
                            max(to_min, to_max));
                    }
                    value = make_float4(mapped, 0.0f);
                    break;
                }
                case compiler::ValueOperation::vector_math_value:
                case compiler::ValueOperation::vector_math_vector: {
                    auto a = vector(
                        instruction.operand(operand::vector_math::a), result);
                    auto b = vector(
                        instruction.operand(operand::vector_math::b), result);
                    auto c = vector(
                        instruction.operand(operand::vector_math::c), result);
                    auto scale = scalar(
                        instruction.operand(operand::vector_math::scale),
                        result);
                    auto safe_divide = [](
                                           Float numerator,
                                           Float denominator) {
                        auto valid = denominator != 0.0f;
                        return select(
                            0.0f,
                            numerator /
                                select(
                                    1.0f,
                                    denominator,
                                    valid),
                            valid);
                    };
                    auto safe_divide_vector =
                        [&](Float3 numerator, Float3 denominator) {
                            return make_float3(
                                safe_divide(
                                    numerator.x,
                                    denominator.x),
                                safe_divide(
                                    numerator.y,
                                    denominator.y),
                                safe_divide(
                                    numerator.z,
                                    denominator.z));
                        };
                    auto safe_normalize_zero =
                        [](Float3 input) {
                            auto input_length =
                                sqrt(dot(input, input));
                            auto valid =
                                input_length != 0.0f;
                            return select(
                                input,
                                input /
                                    select(
                                        1.0f,
                                        input_length,
                                        valid),
                                valid);
                        };
                    auto safe_power = [](Float base, Float exponent) {
                        auto integer_exponent =
                            exponent == trunc(exponent);
                        auto powered =
                            pow(abs(base), exponent);
                        auto odd_exponent =
                            fmod(abs(exponent), 2.0f) != 0.0f;
                        powered = select(
                            powered,
                            -powered,
                            (base < 0.0f) & odd_exponent);
                        return select(
                            0.0f,
                            powered,
                            (base >= 0.0f) |
                                integer_exponent);
                    };
                    auto wrap_component = [](
                                              Float input,
                                              Float maximum,
                                              Float minimum) {
                        auto range = maximum - minimum;
                        auto valid = range != 0.0f;
                        return select(
                            minimum,
                            input -
                                range *
                                    floor(
                                        (input - minimum) /
                                        select(
                                            1.0f,
                                            range,
                                            valid)),
                            valid);
                    };

                    Float scalar_result = 0.0f;
                    Float3 vector_result =
                        make_float3(0.0f);
                    switch (
                        static_cast<compiler::VectorMathOperation>(
                            instruction.static_u0)) {
                        case compiler::VectorMathOperation::add:
                            vector_result = a + b;
                            break;
                        case compiler::VectorMathOperation::subtract:
                            vector_result = a - b;
                            break;
                        case compiler::VectorMathOperation::multiply:
                            vector_result = a * b;
                            break;
                        case compiler::VectorMathOperation::divide:
                            vector_result =
                                safe_divide_vector(a, b);
                            break;
                        case compiler::VectorMathOperation::
                            multiply_add:
                            vector_result = a * b + c;
                            break;
                        case compiler::VectorMathOperation::
                            cross_product:
                            vector_result = cross(a, b);
                            break;
                        case compiler::VectorMathOperation::project: {
                            auto length_squared = dot(b, b);
                            auto valid =
                                length_squared != 0.0f;
                            vector_result = select(
                                make_float3(0.0f),
                                safe_divide(
                                    dot(a, b),
                                    length_squared) *
                                    b,
                                valid);
                            break;
                        }
                        case compiler::VectorMathOperation::reflect: {
                            auto normal =
                                safe_normalize_zero(b);
                            vector_result =
                                a -
                                2.0f * normal *
                                    dot(a, normal);
                            break;
                        }
                        case compiler::VectorMathOperation::refract: {
                            auto normal =
                                safe_normalize_zero(b);
                            auto cosine = dot(normal, a);
                            auto k =
                                1.0f -
                                scale * scale *
                                    (1.0f -
                                     cosine * cosine);
                            vector_result = select(
                                make_float3(0.0f),
                                scale * a -
                                    (scale * cosine +
                                     sqrt(max(k, 0.0f))) *
                                        normal,
                                k >= 0.0f);
                            break;
                        }
                        case compiler::VectorMathOperation::
                            faceforward:
                            vector_result = select(
                                -a,
                                a,
                                dot(c, b) < 0.0f);
                            break;
                        case compiler::VectorMathOperation::
                            dot_product:
                            scalar_result = dot(a, b);
                            break;
                        case compiler::VectorMathOperation::distance: {
                            auto delta = a - b;
                            scalar_result =
                                sqrt(dot(delta, delta));
                            break;
                        }
                        case compiler::VectorMathOperation::length:
                            scalar_result = sqrt(dot(a, a));
                            break;
                        case compiler::VectorMathOperation::scale:
                            vector_result = a * scale;
                            break;
                        case compiler::VectorMathOperation::normalize:
                            vector_result =
                                safe_normalize_zero(a);
                            break;
                        case compiler::VectorMathOperation::absolute:
                            vector_result = abs(a);
                            break;
                        case compiler::VectorMathOperation::power:
                            vector_result = make_float3(
                                safe_power(a.x, b.x),
                                safe_power(a.y, b.y),
                                safe_power(a.z, b.z));
                            break;
                        case compiler::VectorMathOperation::sign: {
                            auto sign_component = [](Float input) {
                                return select(
                                    select(
                                        1.0f,
                                        -1.0f,
                                        input < 0.0f),
                                    0.0f,
                                    input == 0.0f);
                            };
                            vector_result = make_float3(
                                sign_component(a.x),
                                sign_component(a.y),
                                sign_component(a.z));
                            break;
                        }
                        case compiler::VectorMathOperation::minimum:
                            vector_result = min(a, b);
                            break;
                        case compiler::VectorMathOperation::maximum:
                            vector_result = max(a, b);
                            break;
                        case compiler::VectorMathOperation::floor:
                            vector_result = floor(a);
                            break;
                        case compiler::VectorMathOperation::ceil:
                            vector_result = ceil(a);
                            break;
                        case compiler::VectorMathOperation::fraction:
                            vector_result = a - floor(a);
                            break;
                        case compiler::VectorMathOperation::modulo:
                            vector_result = make_float3(
                                select(
                                    0.0f,
                                    fmod(a.x, b.x),
                                    b.x != 0.0f),
                                select(
                                    0.0f,
                                    fmod(a.y, b.y),
                                    b.y != 0.0f),
                                select(
                                    0.0f,
                                    fmod(a.z, b.z),
                                    b.z != 0.0f));
                            break;
                        case compiler::VectorMathOperation::wrap:
                            vector_result = make_float3(
                                wrap_component(
                                    a.x, b.x, c.x),
                                wrap_component(
                                    a.y, b.y, c.y),
                                wrap_component(
                                    a.z, b.z, c.z));
                            break;
                        case compiler::VectorMathOperation::snap:
                            vector_result =
                                floor(
                                    safe_divide_vector(a, b)) *
                                b;
                            break;
                        case compiler::VectorMathOperation::sine:
                            vector_result = make_float3(
                                sin(a.x), sin(a.y), sin(a.z));
                            break;
                        case compiler::VectorMathOperation::cosine:
                            vector_result = make_float3(
                                cos(a.x), cos(a.y), cos(a.z));
                            break;
                        case compiler::VectorMathOperation::tangent:
                            vector_result = make_float3(
                                tan(a.x), tan(a.y), tan(a.z));
                            break;
                    }
                    value =
                        instruction.operation ==
                                compiler::ValueOperation::
                                    vector_math_value
                            ? make_float4(scalar_result)
                            : make_float4(
                                  vector_result, 0.0f);
                    break;
                }
                case compiler::ValueOperation::mix_float: {
                    auto t = scalar(
                        instruction.operand(operand::mix::factor), result);
                    if (instruction.static_u0 != 0u) {
                        t = clamp(t, 0.0f, 1.0f);
                    }
                    value = make_float4(lerp(
                        scalar(
                            instruction.operand(operand::mix::a), result),
                        scalar(
                            instruction.operand(operand::mix::b), result),
                        t));
                    break;
                }
                case compiler::ValueOperation::mix_vector: {
                    auto t = instruction.static_u0 != 0u
                                 ? vector(
                                       instruction.operand(
                                           operand::mix::factor),
                                       result)
                                 : make_float3(
                                       scalar(
                                           instruction.operand(
                                               operand::mix::factor),
                                           result));
                    if (instruction.static_u1 != 0u) {
                        t = clamp(t, 0.0f, 1.0f);
                    }
                    value = make_float4(
                        lerp(
                            vector(
                                instruction.operand(operand::mix::a), result),
                            vector(
                                instruction.operand(operand::mix::b), result),
                            t),
                        1.0f);
                    break;
                }
                case compiler::ValueOperation::mix: {
                    auto t = scalar(
                        instruction.operand(operand::mix::factor), result);
                    auto a = vector(
                        instruction.operand(operand::mix::a), result);
                    auto b = vector(
                        instruction.operand(operand::mix::b), result);
                    const auto mixed =
                        context.svm_immediate_override != nullptr
                            ? evaluate_surface_mix_svm(
                                  services,
                                  *context.svm_immediate_override,
                                  context.svm_immediate_domain,
                                  t,
                                  a,
                                  b)
                            : evaluate_surface_mix(
                                  services,
                                  static_cast<compiler::BlendOperation>(
                                      instruction.static_u0),
                                  (instruction.static_u1 & 1u) != 0u,
                                  (instruction.static_u1 & 2u) != 0u,
                                  t,
                                  a,
                                  b);
                    value = make_float4(mixed, 1.0f);
                    break;
                }
            default:
                break;
        }
        return project_surface_value(
            instruction.result_type, value);
    }
};

}// namespace

std::unique_ptr<ValueNode> try_make_math_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    if (!supports_math_value(instruction.operation)) {
        return nullptr;
    }
    return std::make_unique<MathValueNode>(instruction);
}

}// namespace psycles::luisa_backend::detail

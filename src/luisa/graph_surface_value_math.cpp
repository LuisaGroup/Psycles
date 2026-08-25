#include "graph_surface_internal.h"
#include "surface_math.h"
#include "surface_mix.h"
#include "surface_vector_math.h"

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
                    const auto a = vector(
                        instruction.operand(operand::vector_math::a), result);
                    const auto b = vector(
                        instruction.operand(operand::vector_math::b), result);
                    const auto c = vector(
                        instruction.operand(operand::vector_math::c), result);
                    const auto scale = scalar(
                        instruction.operand(operand::vector_math::scale),
                        result);
                    if (context.svm_immediate_override != nullptr) {
                        value = instruction.operation ==
                                        compiler::ValueOperation::
                                            vector_math_value
                                    ? make_float4(
                                          evaluate_surface_vector_math_value_svm(
                                              *context.svm_immediate_override,
                                              context.svm_immediate_domain,
                                              a,
                                              b,
                                              c,
                                              scale))
                                    : make_float4(
                                          evaluate_surface_vector_math_vector_svm(
                                              *context.svm_immediate_override,
                                              context.svm_immediate_domain,
                                              a,
                                              b,
                                              c,
                                              scale),
                                          0.0f);
                    } else {
                        const auto evaluated =
                            evaluate_surface_vector_math_operation(
                                static_cast<compiler::VectorMathOperation>(
                                    instruction.static_u0),
                                a,
                                b,
                                c,
                                scale);
                        value = instruction.operation ==
                                        compiler::ValueOperation::
                                            vector_math_value
                                    ? make_float4(evaluated.value)
                                    : make_float4(evaluated.vector, 0.0f);
                    }
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

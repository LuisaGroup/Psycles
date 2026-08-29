#include "path_tracer_surface_value_numeric_family.h"

#include "surface_map_range.h"
#include "surface_math.h"
#include "surface_vector_math.h"

#include <cstdlib>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] compiler::SurfaceValueBank
result_bank(const compiler::SurfaceValueStaticVariant &variant) noexcept {
    auto bank = compiler::SurfaceValueBank::scalar;
    if (!compiler::classify_surface_value_type(variant.instruction.result_type,
                                               bank)) {
        std::abort();
    }
    return bank;
}

void emit_math_family(
    SurfaceValueOperandReader &operands, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
    Float result = 0.0f;
    switch (variant.instruction.operation) {
        case compiler::ValueOperation::add: {
            const auto a = operands.scalar(operand::binary::a);
            const auto b = operands.scalar(operand::binary::b);
            result = a + b;
            break;
        }
        case compiler::ValueOperation::subtract: {
            const auto a = operands.scalar(operand::binary::a);
            const auto b = operands.scalar(operand::binary::b);
            result = a - b;
            break;
        }
        case compiler::ValueOperation::multiply: {
            const auto a = operands.scalar(operand::binary::a);
            const auto b = operands.scalar(operand::binary::b);
            result = a * b;
            break;
        }
        case compiler::ValueOperation::divide: {
            const auto numerator = operands.scalar(operand::binary::a);
            const auto denominator = operands.scalar(operand::binary::b);
            result = select(0.0f, numerator / denominator,
                            abs(denominator) > 1.0e-20f);
            break;
        }
        case compiler::ValueOperation::minimum: {
            const auto a = operands.scalar(operand::binary::a);
            const auto b = operands.scalar(operand::binary::b);
            result = min(a, b);
            break;
        }
        case compiler::ValueOperation::maximum: {
            const auto a = operands.scalar(operand::binary::a);
            const auto b = operands.scalar(operand::binary::b);
            result = max(a, b);
            break;
        }
        case compiler::ValueOperation::power: {
            const auto a = operands.scalar(operand::binary::a);
            const auto b = operands.scalar(operand::binary::b);
            result = pow(max(a, 0.0f), b);
            break;
        }
        case compiler::ValueOperation::math: {
            const auto a = operands.scalar(operand::ternary::a);
            const auto b = operands.scalar(operand::ternary::b);
            const auto c = operands.scalar(operand::ternary::c);
            result = evaluate_surface_math_svm(
                surface_value_runtime_immediate(instruction), variant.svm_immediates,
                a, b, c);
            break;
        }
        case compiler::ValueOperation::absolute: {
            const auto input = operands.scalar(operand::unary::input);
            result = abs(input);
            break;
        }
        default:
            std::abort();
    }
    write_surface_value_scalar(locals, instruction, std::move(result));
}

void emit_vector_math_family(
    SurfaceValueOperandReader &operands, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
    const auto bank = result_bank(variant);
    const auto value_output =
        variant.instruction.operation ==
        compiler::ValueOperation::vector_math_value;
    if ((!value_output && variant.instruction.operation !=
                              compiler::ValueOperation::vector_math_vector) ||
        (value_output && bank != compiler::SurfaceValueBank::scalar) ||
        (!value_output && bank != compiler::SurfaceValueBank::vector)) {
        std::abort();
    }

    // Cycles' SVMNodeVectorMath ABI reads A, B, C and Scale in this order.
    // Keep each staged read in its own full-expression so C++ host evaluation
    // order cannot change the recorded Luisa AST or extend live ranges.
    const auto a = operands.vector(operand::vector_math::a);
    const auto b = operands.vector(operand::vector_math::b);
    const auto c = operands.vector(operand::vector_math::c);
    const auto scale = operands.scalar(operand::vector_math::scale);
    if (value_output) {
        const auto value = evaluate_surface_vector_math_value_svm(
            surface_value_runtime_immediate(instruction), variant.svm_immediates, a, b,
            c, scale);
        write_surface_value_scalar(locals, instruction, value);
    } else {
        const auto value = evaluate_surface_vector_math_vector_svm(
            surface_value_runtime_immediate(instruction), variant.svm_immediates, a, b,
            c, scale);
        write_surface_value_vector(locals, instruction, value);
    }
}

void emit_clamp_family(
    SurfaceValueOperandReader &operands, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
    if (result_bank(variant) != compiler::SurfaceValueBank::scalar) {
        std::abort();
    }
    Float value = 0.0f;
    switch (variant.instruction.operation) {
        case compiler::ValueOperation::clamp01: {
            const auto input = operands.scalar(operand::unary::input);
            value = clamp(input, 0.0f, 1.0f);
            break;
        }
        case compiler::ValueOperation::clamp_range: {
            const auto input = operands.scalar(operand::clamp_range::value);
            const auto minimum = operands.scalar(operand::clamp_range::minimum);
            const auto maximum = operands.scalar(operand::clamp_range::maximum);
            value = evaluate_surface_clamp_svm(
                surface_value_runtime_immediate(instruction), variant.svm_immediates,
                input, minimum, maximum);
            break;
        }
        default:
            std::abort();
    }
    write_surface_value_scalar(locals, instruction, std::move(value));
}

} // namespace

void emit_direct_surface_numeric_family(
    compiler::SurfaceSvmValueOpcode family,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    switch (family) {
        case compiler::SurfaceSvmValueOpcode::math:
            emit_math_family(operands, locals, instruction, variant);
            return;
        case compiler::SurfaceSvmValueOpcode::vector_math:
            emit_vector_math_family(operands, locals, instruction, variant);
            return;
        case compiler::SurfaceSvmValueOpcode::clamp:
            emit_clamp_family(operands, locals, instruction, variant);
            return;
        default:
            std::abort();
    }
}

} // namespace psycles::luisa_backend::detail

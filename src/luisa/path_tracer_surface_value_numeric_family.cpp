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

void emit_map_range_family(
    compiler::SurfaceSvmValueOpcode family,
    const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    const auto immediate = surface_value_runtime_immediate(instruction);
    if (family == compiler::SurfaceSvmValueOpcode::map_range) {
        if (variant.instruction.operation !=
                compiler::ValueOperation::map_range_float ||
            result_bank(variant) != compiler::SurfaceValueBank::scalar) {
            std::abort();
        }
        const auto value = operands.scalar(operand::map_range::value);
        const auto from_min = operands.scalar(operand::map_range::from_min);
        const auto from_max = operands.scalar(operand::map_range::from_max);
        const auto to_min = operands.scalar(operand::map_range::to_min);
        const auto to_max = operands.scalar(operand::map_range::to_max);
        const auto steps = operands.scalar(operand::map_range::steps);
        write_surface_value_scalar(
            locals, instruction,
            evaluate_surface_map_range_float_svm(
                immediate, variant.svm_immediates, value, from_min, from_max,
                to_min, to_max, steps));
        return;
    }
    if (family == compiler::SurfaceSvmValueOpcode::vector_map_range) {
        if (variant.instruction.operation !=
                compiler::ValueOperation::map_range_vector ||
            result_bank(variant) != compiler::SurfaceValueBank::vector) {
            std::abort();
        }
        const auto value = operands.vector(operand::map_range::value);
        const auto from_min = operands.vector(operand::map_range::from_min);
        const auto from_max = operands.vector(operand::map_range::from_max);
        const auto to_min = operands.vector(operand::map_range::to_min);
        const auto to_max = operands.vector(operand::map_range::to_max);
        const auto steps = operands.vector(operand::map_range::steps);
        write_surface_value_vector(
            locals, instruction,
            evaluate_surface_map_range_vector_svm(
                immediate, variant.svm_immediates, value, from_min, from_max,
                to_min, to_max, steps));
        return;
    }
    std::abort();
}

void emit_mix_float_family(
    const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    if (variant.instruction.operation != compiler::ValueOperation::mix_float ||
        result_bank(variant) != compiler::SurfaceValueBank::scalar ||
        variant.svm_immediates.empty()) {
        std::abort();
    }
    for (const auto immediate : variant.svm_immediates) {
        if ((immediate & ~compiler::surface_value_mix_float_clamp_bit) != 0u) {
            std::abort();
        }
    }
    const auto a = operands.scalar(operand::mix::a);
    const auto b = operands.scalar(operand::mix::b);
    auto factor = operands.scalar(operand::mix::factor);
    factor = select(
        factor, clamp(factor, 0.0f, 1.0f),
        (surface_value_runtime_immediate(instruction) &
         compiler::surface_value_mix_float_clamp_bit) != 0u);
    write_surface_value_scalar(locals, instruction, lerp(a, b, factor));
}

void emit_mix_vector_family(
    compiler::SurfaceSvmValueOpcode family,
    const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    const auto non_uniform =
        family == compiler::SurfaceSvmValueOpcode::mix_vector_non_uniform;
    if ((!non_uniform && family !=
                             compiler::SurfaceSvmValueOpcode::mix_vector) ||
        variant.instruction.operation != compiler::ValueOperation::mix_vector ||
        result_bank(variant) != compiler::SurfaceValueBank::vector ||
        variant.svm_immediates.empty()) {
        std::abort();
    }
    constexpr auto configuration_mask =
        compiler::surface_value_mix_vector_non_uniform_bit |
        compiler::surface_value_mix_vector_clamp_bit;
    for (const auto immediate : variant.svm_immediates) {
        if ((immediate & ~configuration_mask) != 0u ||
            ((immediate &
              compiler::surface_value_mix_vector_non_uniform_bit) != 0u) !=
                non_uniform) {
            std::abort();
        }
    }
    const auto a = operands.vector(operand::mix::a);
    const auto b = operands.vector(operand::mix::b);
    auto factor = non_uniform
                      ? operands.vector(operand::mix::factor)
                      : make_float3(operands.scalar(operand::mix::factor));
    factor = select(
        factor, clamp(factor, 0.0f, 1.0f),
        (surface_value_runtime_immediate(instruction) &
         compiler::surface_value_mix_vector_clamp_bit) != 0u);
    write_surface_value_vector(locals, instruction, lerp(a, b, factor));
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
        case compiler::SurfaceSvmValueOpcode::map_range:
        case compiler::SurfaceSvmValueOpcode::vector_map_range:
            emit_map_range_family(family, locals, instruction, variant,
                                  operands);
            return;
        case compiler::SurfaceSvmValueOpcode::mix_float:
            emit_mix_float_family(locals, instruction, variant, operands);
            return;
        case compiler::SurfaceSvmValueOpcode::mix_vector:
        case compiler::SurfaceSvmValueOpcode::mix_vector_non_uniform:
            emit_mix_vector_family(family, locals, instruction, variant,
                                   operands);
            return;
        default:
            std::abort();
    }
}

} // namespace psycles::luisa_backend::detail

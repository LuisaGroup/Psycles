#include "path_tracer_surface_value_family.h"

#include "surface_math.h"

#include <cstdlib>
#include <utility>
#include <vector>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

class SurfaceValueOperandReader {
private:
    const compiler::SurfaceValueStaticVariant &_variant;
    const ShaderServices &_services;
    const SurfacePoint &_point;
    const SurfaceValueLocalsView &_locals;
    std::vector<UInt> _addresses;

private:
    [[nodiscard]] compiler::SurfaceValueBank
    bank(std::size_t index) const noexcept {
        if (index >= _variant.operand_types.size()) {
            std::abort();
        }
        auto result = compiler::SurfaceValueBank::scalar;
        if (!compiler::classify_surface_value_type(_variant.operand_types[index],
                                                   result)) {
            std::abort();
        }
        return result;
    }

    [[nodiscard]] compiler::SurfaceValueOperandRoute
    route(std::size_t index) const noexcept {
        if (index >= _variant.operand_routes.size()) {
            std::abort();
        }
        return _variant.operand_routes[index];
    }

    [[nodiscard]] UInt address(std::size_t index) const noexcept {
        if (index >= _addresses.size()) {
            std::abort();
        }
        return UInt{_addresses[index].expression()};
    }

public:
    SurfaceValueOperandReader(
        const SurfaceValueRuntime &runtime,
        SurfaceValueRuntimeBufferSlot operand_slot,
        const ShaderServices &services, const SurfacePoint &point,
        const SurfaceValueLocalsView &locals, Var<luisa::uint4> instruction,
        const compiler::SurfaceValueStaticVariant &variant) noexcept
        : _variant{variant}, _services{services}, _point{point}, _locals{locals} {
        if (variant.operand_routes.size() != variant.operand_types.size()) {
            std::abort();
        }
        _addresses.reserve(variant.operand_types.size());
        const auto inline_operands =
            variant.operand_types.size() <=
            compiler::surface_value_inline_operand_capacity;
        const auto word_count = compiler::surface_value_operand_word_count(
            variant.operand_types.size());
        for (auto word_index = std::size_t{}; word_index < word_count;
             ++word_index) {
            auto word =
                inline_operands ? UInt{instruction.z.expression()} : surface_value_runtime_buffer<luisa::uint>(runtime, operand_slot).read(instruction.z + static_cast<std::uint32_t>(word_index));
            for (auto lane = std::size_t{};
                 lane < compiler::surface_value_operands_per_word; ++lane) {
                const auto operand_index =
                    word_index * compiler::surface_value_operands_per_word + lane;
                if (operand_index >= variant.operand_types.size()) {
                    break;
                }
                auto compact =
                    (word >> static_cast<std::uint32_t>(
                                 compiler::surface_value_operand_lane_bits * lane)) &
                    0xffffu;
                _addresses.emplace_back(
                    (compact & static_cast<std::uint32_t>(
                                   compiler::SurfaceValueOperandAddress::index_mask)) |
                    ((compact &
                      (static_cast<std::uint32_t>(
                           compiler::SurfaceValueOperandAddress::parameter_bit) |
                       static_cast<std::uint32_t>(
                           compiler::SurfaceValueOperandAddress::bank_mask)))
                     << (compiler::SurfaceValueAddress::bank_shift -
                         compiler::SurfaceValueOperandAddress::bank_shift)));
            }
        }
    }

    [[nodiscard]] Float scalar(std::size_t index) const noexcept {
        if (bank(index) != compiler::SurfaceValueBank::scalar) {
            std::abort();
        }
        auto operand = address(index);
        switch (route(index)) {
            case compiler::SurfaceValueOperandRoute::local:
                return _locals.scalars.read(operand &
                                            compiler::SurfaceValueAddress::index_mask);
            case compiler::SurfaceValueOperandRoute::parameter:
                return _services.parameter_float(
                    _point.parameter_block,
                    operand & compiler::SurfaceValueAddress::index_mask);
            case compiler::SurfaceValueOperandRoute::dynamic:
                return read_scalar_dynamic(_services, _point, _locals,
                                           std::move(operand));
        }
        std::abort();
    }

    [[nodiscard]] Float3 vector(std::size_t index) const noexcept {
        if (bank(index) != compiler::SurfaceValueBank::vector) {
            std::abort();
        }
        auto operand = address(index);
        switch (route(index)) {
            case compiler::SurfaceValueOperandRoute::local:
                return _locals.vectors.read(operand &
                                            compiler::SurfaceValueAddress::index_mask);
            case compiler::SurfaceValueOperandRoute::parameter:
                return _services.parameter_float3(
                    _point.parameter_block,
                    operand & compiler::SurfaceValueAddress::index_mask);
            case compiler::SurfaceValueOperandRoute::dynamic:
                return read_vector_dynamic(_services, _point, _locals,
                                           std::move(operand));
        }
        std::abort();
    }
};

void write_scalar(const SurfaceValueLocalsView &locals,
                  Var<luisa::uint4> instruction, Float value) noexcept {
    locals.scalars.write(instruction.y &
                             compiler::SurfaceValueAddress::index_mask,
                         std::move(value));
}

void write_vector(const SurfaceValueLocalsView &locals,
                  Var<luisa::uint4> instruction, Float3 value) noexcept {
    locals.vectors.write(instruction.y &
                             compiler::SurfaceValueAddress::index_mask,
                         std::move(value));
}

void emit_convert_family(SurfaceValueOperandReader &operands,
                         const SurfaceValueLocalsView &locals,
                         Var<luisa::uint4> instruction,
                         compiler::ValueOperation operation) noexcept {
    namespace operand = compiler::value_operand;
    switch (operation) {
        case compiler::ValueOperation::scalar_to_color: {
            const auto x = operands.scalar(operand::unary::input);
            write_vector(locals, instruction, make_float3(x));
            return;
        }
        case compiler::ValueOperation::scalar_to_boolean: {
            const auto x = operands.scalar(operand::unary::input);
            write_scalar(locals, instruction, select(0.0f, 1.0f, cast<int>(x) != 0));
            return;
        }
        case compiler::ValueOperation::color_to_scalar: {
            const auto color = operands.vector(operand::unary::input);
            write_scalar(locals, instruction,
                         dot(color, make_float3(0.21267404f, 0.7151516f, 0.07217542f)));
            return;
        }
        case compiler::ValueOperation::vector_to_scalar: {
            const auto value = operands.vector(operand::unary::input);
            write_scalar(locals, instruction, (value.x + value.y + value.z) / 3.0f);
            return;
        }
        case compiler::ValueOperation::parameter:
        case compiler::ValueOperation::passthrough:
        default:
            // Parameters and passthroughs are eliminated before executable
            // evaluator interning; every other operation belongs elsewhere.
            std::abort();
    }
}

void emit_math_family(
    SurfaceValueOperandReader &operands, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
    namespace operand = compiler::value_operand;
    Float result = 0.0f;
    switch (variant.instruction.operation) {
        case compiler::ValueOperation::add:
            result = operands.scalar(operand::binary::a) +
                     operands.scalar(operand::binary::b);
            break;
        case compiler::ValueOperation::subtract:
            result = operands.scalar(operand::binary::a) -
                     operands.scalar(operand::binary::b);
            break;
        case compiler::ValueOperation::multiply:
            result = operands.scalar(operand::binary::a) *
                     operands.scalar(operand::binary::b);
            break;
        case compiler::ValueOperation::divide: {
            const auto denominator = operands.scalar(operand::binary::b);
            result = select(0.0f, operands.scalar(operand::binary::a) / denominator,
                            abs(denominator) > 1.0e-20f);
            break;
        }
        case compiler::ValueOperation::minimum:
            result = min(operands.scalar(operand::binary::a),
                         operands.scalar(operand::binary::b));
            break;
        case compiler::ValueOperation::maximum:
            result = max(operands.scalar(operand::binary::a),
                         operands.scalar(operand::binary::b));
            break;
        case compiler::ValueOperation::power:
            result = pow(max(operands.scalar(operand::binary::a), 0.0f),
                         operands.scalar(operand::binary::b));
            break;
        case compiler::ValueOperation::math: {
            const auto immediate =
                (instruction.x & compiler::surface_value_svm_immediate_mask) >>
                compiler::surface_value_svm_immediate_shift;
            result = evaluate_surface_math_svm(immediate, variant.svm_immediates,
                                               operands.scalar(operand::ternary::a),
                                               operands.scalar(operand::ternary::b),
                                               operands.scalar(operand::ternary::c));
            break;
        }
        case compiler::ValueOperation::absolute:
            result = abs(operands.scalar(operand::unary::input));
            break;
        default:
            std::abort();
    }
    write_scalar(locals, instruction, std::move(result));
}

}// namespace

bool emit_direct_surface_value_variant(
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot operand_slot, const ShaderServices &services,
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
    if (variant.svm_immediates.empty()) {
        std::abort();
    }
    const auto family = compiler::surface_svm_value_opcode(
        variant.instruction.operation, variant.svm_immediates.front());
    if (!surface_value_family_has_direct_evaluator(family)) {
        return false;
    }
    for (const auto immediate : variant.svm_immediates) {
        if (compiler::surface_svm_value_opcode(variant.instruction.operation,
                                               immediate) != family) {
            std::abort();
        }
    }
    SurfaceValueOperandReader operands{runtime, operand_slot, services, point,
                                       locals, instruction, variant};
    switch (family) {
        case compiler::SurfaceSvmValueOpcode::convert:
            emit_convert_family(operands, locals, instruction,
                                variant.instruction.operation);
            return true;
        case compiler::SurfaceSvmValueOpcode::math:
            emit_math_family(operands, locals, instruction, variant);
            return true;
        default:
            std::abort();
    }
}

}// namespace psycles::luisa_backend::detail

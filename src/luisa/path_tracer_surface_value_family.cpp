#include "path_tracer_surface_value_family.h"
#include "path_tracer_surface_value_numeric_family.h"
#include "path_tracer_surface_value_state_family.h"
#include "path_tracer_surface_value_texture_family.h"

#include <cstdlib>
#include <utility>
#include <vector>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
compiler::SurfaceValueBank
SurfaceValueOperandReader::bank(std::size_t index) const noexcept {
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

compiler::SurfaceValueOperandRoute
SurfaceValueOperandReader::route(std::size_t index) const noexcept {
    if (index >= _variant.operand_routes.size()) {
        std::abort();
    }
    return _variant.operand_routes[index];
}

UInt SurfaceValueOperandReader::address(std::size_t index) const noexcept {
    if (index >= _addresses.size()) {
        std::abort();
    }
    return UInt{_addresses[index].expression()};
}

void SurfaceValueOperandReader::begin_read(std::size_t index) const noexcept {
    // Luisa DSL calls below construct the device AST on the host. Requiring a
    // monotonically increasing, single read of each logical operand makes the
    // emitted load order equal to the bytecode ABI instead of depending on the
    // host compiler's unspecified expression-operand evaluation order.
    if (!advance_surface_value_operand_read_order(index,
                                                  _next_operand_index)) {
        std::abort();
    }
}

SurfaceValueOperandReader::SurfaceValueOperandReader(
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot operand_slot, const ShaderServices &services,
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
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
        const auto word =
            inline_operands ? UInt{instruction.z.expression()}
                            : surface_value_runtime_buffer<luisa::uint>(
                                  runtime, operand_slot)
                                  .read(instruction.z +
                                        static_cast<std::uint32_t>(word_index));
        for (auto lane = std::size_t{};
             lane < compiler::surface_value_operands_per_word; ++lane) {
            const auto operand_index =
                word_index * compiler::surface_value_operands_per_word + lane;
            if (operand_index >= variant.operand_types.size()) {
                break;
            }
            const auto compact =
                (word >>
                 static_cast<std::uint32_t>(
                     compiler::surface_value_operand_lane_bits * lane)) &
                0xffffu;
            _addresses.emplace_back(
                (compact &
                 static_cast<std::uint32_t>(
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

Float SurfaceValueOperandReader::scalar(std::size_t index) const noexcept {
    begin_read(index);
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

Float3 SurfaceValueOperandReader::vector(std::size_t index) const noexcept {
    begin_read(index);
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

ULong SurfaceValueOperandReader::unsigned_integer(
    std::size_t index) const noexcept {
    begin_read(index);
    if (bank(index) != compiler::SurfaceValueBank::unsigned_integer) {
        std::abort();
    }
    auto operand = address(index);
    switch (route(index)) {
    case compiler::SurfaceValueOperandRoute::local:
        return _locals.unsigned_integers.read(
            operand & compiler::SurfaceValueAddress::index_mask);
    case compiler::SurfaceValueOperandRoute::parameter:
        return _services.parameter_uint64(
            _point.parameter_block,
            operand & compiler::SurfaceValueAddress::index_mask);
    case compiler::SurfaceValueOperandRoute::dynamic:
        return read_unsigned_integer_dynamic(_services, _point, _locals,
                                             std::move(operand));
    }
    std::abort();
}

void write_surface_value_scalar(const SurfaceValueLocalsView &locals,
                                Var<luisa::uint4> instruction,
                                Float value) noexcept {
    locals.scalars.write(instruction.y &
                             compiler::SurfaceValueAddress::index_mask,
                         std::move(value));
}

void write_surface_value_vector(const SurfaceValueLocalsView &locals,
                                Var<luisa::uint4> instruction,
                                Float3 value) noexcept {
    locals.vectors.write(instruction.y &
                             compiler::SurfaceValueAddress::index_mask,
                         std::move(value));
}

namespace {

void emit_convert_family(SurfaceValueOperandReader &operands,
                         const SurfaceValueLocalsView &locals,
                         Var<luisa::uint4> instruction,
                         compiler::ValueOperation operation) noexcept {
    namespace operand = compiler::value_operand;
    switch (operation) {
        case compiler::ValueOperation::scalar_to_color: {
            const auto x = operands.scalar(operand::unary::input);
            write_surface_value_vector(locals, instruction, make_float3(x));
            return;
        }
        case compiler::ValueOperation::scalar_to_boolean: {
            const auto x = operands.scalar(operand::unary::input);
            write_surface_value_scalar(locals, instruction,
                                       select(0.0f, 1.0f, cast<int>(x) != 0));
            return;
        }
        case compiler::ValueOperation::color_to_scalar: {
            const auto color = operands.vector(operand::unary::input);
            write_surface_value_scalar(
                locals, instruction,
                dot(color, make_float3(0.21267404f, 0.7151516f, 0.07217542f)));
            return;
        }
        case compiler::ValueOperation::vector_to_scalar: {
            const auto value = operands.vector(operand::unary::input);
            write_surface_value_scalar(locals, instruction,
                                       (value.x + value.y + value.z) / 3.0f);
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

} // namespace

bool emit_direct_surface_value_variant(
    const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots, const ShaderServices &services,
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
    SurfaceValueOperandReader operands{
        runtime, bytecode_slots.operand, services, point, locals, instruction,
        variant};
    switch (family) {
        case compiler::SurfaceSvmValueOpcode::convert:
            emit_convert_family(operands, locals, instruction,
                                variant.instruction.operation);
            return true;
        case compiler::SurfaceSvmValueOpcode::math:
        case compiler::SurfaceSvmValueOpcode::vector_math:
        case compiler::SurfaceSvmValueOpcode::clamp:
            emit_direct_surface_numeric_family(family, locals, instruction,
                                               variant, operands);
            return true;
        case compiler::SurfaceSvmValueOpcode::geometry:
        case compiler::SurfaceSvmValueOpcode::geometry_derivative:
        case compiler::SurfaceSvmValueOpcode::tex_coord:
        case compiler::SurfaceSvmValueOpcode::tex_coord_derivative:
        case compiler::SurfaceSvmValueOpcode::bump_support:
            emit_direct_surface_state_family(
                family, runtime, bytecode_slots, services, point, locals,
                instruction, variant, operands);
            return true;
        case compiler::SurfaceSvmValueOpcode::mix_color:
        case compiler::SurfaceSvmValueOpcode::rgb_ramp:
        case compiler::SurfaceSvmValueOpcode::mapping:
        case compiler::SurfaceSvmValueOpcode::tex_image:
        case compiler::SurfaceSvmValueOpcode::tex_image_box:
            emit_direct_surface_texture_family(family, runtime, bytecode_slots,
                                               services, point, locals,
                                               instruction, variant, operands);
            return true;
        default:
            std::abort();
    }
}

}// namespace psycles::luisa_backend::detail

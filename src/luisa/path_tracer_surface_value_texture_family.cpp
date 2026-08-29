#include "path_tracer_surface_value_texture_family.h"

#include "surface_image_svm.h"
#include "surface_mix.h"
#include "surface_shader_table_evaluation.h"
#include "surface_vector_mapping.h"

#include <cstdlib>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] UInt
surface_value_immediate(Var<luisa::uint4> instruction) noexcept {
    return (instruction.x & compiler::surface_value_svm_immediate_mask) >>
           compiler::surface_value_svm_immediate_shift;
}

[[nodiscard]] compiler::SurfaceValueBank
result_bank(const compiler::SurfaceValueStaticVariant &variant) noexcept {
    auto bank = compiler::SurfaceValueBank::scalar;
    if (!compiler::classify_surface_value_type(variant.instruction.result_type,
                                               bank)) {
        std::abort();
    }
    return bank;
}

void emit_mix_color_family(const ShaderServices &services,
                           const SurfaceValueLocalsView &locals,
                           Var<luisa::uint4> instruction,
                           const compiler::SurfaceValueStaticVariant &variant,
                           SurfaceValueOperandReader &operands) noexcept {
    const auto a = operands.vector(operand::mix::a);
    const auto b = operands.vector(operand::mix::b);
    auto factor = operands.scalar(operand::mix::factor);
    Float3 mixed = make_float3(0.0f);
    switch (variant.instruction.operation) {
    case compiler::ValueOperation::mix:
        mixed = evaluate_surface_mix_svm(services,
                                         surface_value_immediate(instruction),
                                         variant.svm_immediates, factor, a, b);
        break;
    case compiler::ValueOperation::multiply_color:
        for (const auto immediate : variant.svm_immediates) {
            if (immediate != 0u) {
                std::abort();
            }
        }
        factor = clamp(factor, 0.0f, 1.0f);
        mixed = lerp(a, a * b, factor);
        break;
    default:
        std::abort();
    }
    write_surface_value_vector(locals, instruction, std::move(mixed));
}

void emit_rgb_ramp_family(const SurfaceValueRuntime &runtime,
                          SurfaceValueBytecodeSlots bytecode_slots,
                          const ShaderServices &services,
                          const SurfacePoint &point,
                          const SurfaceValueLocalsView &locals,
                          Var<luisa::uint4> instruction,
                          const compiler::SurfaceValueStaticVariant &variant,
                          SurfaceValueOperandReader &operands) noexcept {
    if (variant.instruction.operation != compiler::ValueOperation::color_ramp) {
        std::abort();
    }
    const auto bank = result_bank(variant);
    if (bank != compiler::SurfaceValueBank::scalar &&
        bank != compiler::SurfaceValueBank::vector) {
        std::abort();
    }
    const auto alpha_output = bank == compiler::SurfaceValueBank::scalar;
    for (const auto immediate : variant.svm_immediates) {
        const auto encoded_alpha =
            (immediate & compiler::surface_value_color_ramp_alpha_bit) != 0u;
        if (encoded_alpha != alpha_output) {
            std::abort();
        }
    }
    const auto parameter = surface_value_runtime_buffer<luisa::uint>(
                               runtime, bytecode_slots.metadata_parameter)
                               .read(instruction.w);
    const auto table = surface_shader_table_view(
        services, point, Expr<std::uint32_t>{parameter.expression()});
    const auto ramp = evaluate_surface_color_ramp_svm(
        services, surface_value_immediate(instruction), variant.svm_immediates,
        table, operands.scalar(operand::color_ramp::factor));
    if (alpha_output) {
        write_surface_value_scalar(locals, instruction, ramp.w);
    } else {
        write_surface_value_vector(locals, instruction, ramp.xyz());
    }
}

void emit_mapping_family(const ShaderServices &services,
                         const SurfaceValueLocalsView &locals,
                         Var<luisa::uint4> instruction,
                         const compiler::SurfaceValueStaticVariant &variant,
                         SurfaceValueOperandReader &operands) noexcept {
    if (variant.instruction.operation != compiler::ValueOperation::mapping ||
        result_bank(variant) != compiler::SurfaceValueBank::vector) {
        std::abort();
    }
    const auto vector = operands.vector(operand::mapping::vector);
    const auto location = operands.vector(operand::mapping::location);
    const auto rotation = operands.vector(operand::mapping::rotation);
    const auto scale = operands.vector(operand::mapping::scale);
    const auto mapped = evaluate_surface_mapping_svm(
        services, surface_value_immediate(instruction), variant.svm_immediates,
        vector, location, rotation, scale);
    write_surface_value_vector(locals, instruction, std::move(mapped));
}

void emit_image_family(compiler::SurfaceSvmValueOpcode family,
                       const ShaderServices &services,
                       const SurfacePoint &point,
                       const SurfaceValueLocalsView &locals,
                       Var<luisa::uint4> instruction,
                       const compiler::SurfaceValueStaticVariant &variant,
                       SurfaceValueOperandReader &operands) noexcept {
    const auto color_output =
        variant.instruction.operation == compiler::ValueOperation::image_color;
    if (!color_output && variant.instruction.operation !=
                             compiler::ValueOperation::image_alpha) {
        std::abort();
    }
    const auto bank = result_bank(variant);
    if ((color_output && bank != compiler::SurfaceValueBank::vector) ||
        (!color_output && bank != compiler::SurfaceValueBank::scalar)) {
        std::abort();
    }
    const auto shape = family == compiler::SurfaceSvmValueOpcode::tex_image_box
                           ? SurfaceImageSvmShape::image_box
                           : SurfaceImageSvmShape::image;
    // Emit operand reads in bytecode order. Callable argument evaluation order
    // is unspecified in C++, and letting the host compiler choose it can keep
    // the 64-bit texture handle live across the coordinate read in the shader
    // AST. Explicit statements make the staged program independent of the host
    // compiler's argument-order choice.
    const auto coordinate = operands.vector(operand::image_texture::vector);
    const auto texture_handle = cast<std::uint32_t>(
        operands.unsigned_integer(operand::image_texture::image));
    Float projection_blend = 0.0f;
    if (shape == SurfaceImageSvmShape::image_box) {
        projection_blend =
            operands.scalar(operand::image_texture::projection_blend);
    }
    const auto sampled = evaluate_surface_image_svm(
        services, point, shape, surface_value_immediate(instruction),
        variant.svm_immediates, coordinate, texture_handle, projection_blend);
    if (color_output) {
        write_surface_value_vector(locals, instruction, sampled.xyz());
    } else {
        write_surface_value_scalar(locals, instruction, sampled.w);
    }
}

} // namespace

void emit_direct_surface_texture_family(
    compiler::SurfaceSvmValueOpcode family, const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots, const ShaderServices &services,
    const SurfacePoint &point, const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    switch (family) {
    case compiler::SurfaceSvmValueOpcode::mix_color:
        emit_mix_color_family(services, locals, instruction, variant, operands);
        return;
    case compiler::SurfaceSvmValueOpcode::rgb_ramp:
        emit_rgb_ramp_family(runtime, bytecode_slots, services, point, locals,
                             instruction, variant, operands);
        return;
    case compiler::SurfaceSvmValueOpcode::mapping:
        emit_mapping_family(services, locals, instruction, variant, operands);
        return;
    case compiler::SurfaceSvmValueOpcode::tex_image:
    case compiler::SurfaceSvmValueOpcode::tex_image_box:
        emit_image_family(family, services, point, locals, instruction, variant,
                          operands);
        return;
    default:
        std::abort();
    }
}

} // namespace psycles::luisa_backend::detail

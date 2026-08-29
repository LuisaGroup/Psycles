#include "path_tracer_surface_value_state_family.h"

#include "surface_bump.h"

#include <cstdlib>
#include <utility>

#include <luisa/dsl/sugar.h>
#include <psycles/contract/scene.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] UInt
surface_value_immediate(Var<luisa::uint4> instruction) noexcept {
    return (instruction.x & compiler::surface_value_svm_immediate_mask) >>
           compiler::surface_value_svm_immediate_shift;
}

void require_immediate_subset(
    const compiler::SurfaceValueStaticVariant &variant,
    std::uint16_t mask) noexcept {
    for (const auto immediate : variant.svm_immediates) {
        if ((immediate & ~mask) != 0u) {
            std::abort();
        }
    }
}

[[nodiscard]] Float3 transformed_object_coordinate(
    const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots,
    const SurfacePoint &point,
    Var<luisa::uint4> instruction) noexcept {
    // The record owns one range, exactly like Cycles advances one SVM offset
    // before reading PackedTransform. Do not re-read the metadata for each
    // matrix element: the 16 payload reads are all relative to this one base.
    const auto range =
        surface_value_runtime_buffer<luisa::uint2>(
            runtime, bytecode_slots.metadata_static_range)
            .read(instruction.w);
    const auto static_data = surface_value_runtime_buffer<float>(
        runtime, bytecode_slots.static_data);
    const auto m = [&](std::uint32_t index) noexcept {
        return static_data.read(range.x + index);
    };
    const auto world_to_object =
        make_float4x4(make_float4(m(0u), m(1u), m(2u), m(3u)),
                      make_float4(m(4u), m(5u), m(6u), m(7u)),
                      make_float4(m(8u), m(9u), m(10u), m(11u)),
                      make_float4(m(12u), m(13u), m(14u), m(15u)));
    return (world_to_object * make_float4(point.position, 1.0f)).xyz();
}

void emit_uv_coordinate(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands,
    std::size_t map_operand) noexcept {
    require_immediate_subset(
        variant,
        static_cast<std::uint16_t>(
            compiler::surface_value_uv_named_immediate_bit));
    auto value = make_float3(point.uv.x, point.uv.y, 0.0f);
    auto has_default = false;
    auto has_named = false;
    for (const auto immediate : variant.svm_immediates) {
        has_named |=
            (immediate & compiler::surface_value_uv_named_immediate_bit) != 0u;
        has_default |=
            (immediate & compiler::surface_value_uv_named_immediate_bit) == 0u;
    }
    if (has_named && !has_default) {
        const auto map = operands.unsigned_integer(map_operand);
        value = services.attribute(map, point).value.xyz();
    } else if (has_named) {
        const auto named =
            (surface_value_immediate(instruction) &
             compiler::surface_value_uv_named_immediate_bit) != 0u;
        $if(named) {
            const auto map = operands.unsigned_integer(map_operand);
            value = services.attribute(map, point).value.xyz();
        };
    }
    write_surface_value_vector(locals, instruction, std::move(value));
}

void emit_geometry_family(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
    switch (variant.instruction.operation) {
        case compiler::ValueOperation::surface_position:
            write_surface_value_vector(locals, instruction, point.position);
            return;
        case compiler::ValueOperation::shading_normal:
            write_surface_value_vector(locals, instruction,
                                       point.shading_normal);
            return;
        case compiler::ValueOperation::geometric_normal:
            write_surface_value_vector(locals, instruction,
                                       point.geometric_normal);
            return;
        case compiler::ValueOperation::incoming:
            write_surface_value_vector(locals, instruction, point.incoming);
            return;
        case compiler::ValueOperation::pointiness:
            write_surface_value_scalar(
                locals, instruction,
                services
                    .attribute(contract::cycles_pointiness_attribute_id, point)
                    .value.x);
            return;
        default:
            std::abort();
    }
}

void emit_geometry_derivative_family(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    const auto dx = operands.scalar(operand::sampled_nullary::dx);
    const auto dy = operands.scalar(operand::sampled_nullary::dy);
    const auto sampled = surface_differential_sample_point(point, dx, dy);
    switch (variant.instruction.operation) {
        case compiler::ValueOperation::sampled_surface_position:
            write_surface_value_vector(locals, instruction, sampled.position);
            return;
        case compiler::ValueOperation::sampled_pointiness:
            write_surface_value_scalar(
                locals, instruction,
                services
                    .attribute(contract::cycles_pointiness_attribute_id,
                               sampled)
                    .value.x);
            return;
        default:
            std::abort();
    }
}

void emit_tex_coord_family(
    const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    switch (variant.instruction.operation) {
        case compiler::ValueOperation::uv:
            emit_uv_coordinate(services, point, locals, instruction, variant,
                               operands, operand::uv::map);
            return;
        case compiler::ValueOperation::generated:
            write_surface_value_vector(locals, instruction, point.generated);
            return;
        case compiler::ValueOperation::object_position:
            write_surface_value_vector(locals, instruction,
                                       point.object_position);
            return;
        case compiler::ValueOperation::object_position_with_transform:
            write_surface_value_vector(
                locals, instruction,
                transformed_object_coordinate(runtime, bytecode_slots, point,
                                              instruction));
            return;
        default:
            std::abort();
    }
}

void emit_tex_coord_derivative_family(
    const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    const auto dx = operands.scalar(operand::sampled_nullary::dx);
    const auto dy = operands.scalar(operand::sampled_nullary::dy);
    const auto sampled = surface_differential_sample_point(point, dx, dy);
    switch (variant.instruction.operation) {
        case compiler::ValueOperation::sampled_uv:
            emit_uv_coordinate(services, sampled, locals, instruction, variant,
                               operands, operand::sampled_uv::map);
            return;
        case compiler::ValueOperation::sampled_generated:
            write_surface_value_vector(locals, instruction, sampled.generated);
            return;
        case compiler::ValueOperation::sampled_object_position:
            write_surface_value_vector(locals, instruction,
                                       sampled.object_position);
            return;
        case compiler::ValueOperation::sampled_object_position_with_transform:
            write_surface_value_vector(
                locals, instruction,
                transformed_object_coordinate(runtime, bytecode_slots, sampled,
                                              instruction));
            return;
        default:
            std::abort();
    }
}

void emit_bump_support_family(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    switch (variant.instruction.operation) {
        case compiler::ValueOperation::bump_offset_zero:
            write_surface_value_scalar(locals, instruction, 0.0f);
            return;
        case compiler::ValueOperation::bump_filter_width: {
            const auto width = operands.scalar(operand::unary::input);
            write_surface_value_scalar(locals, instruction, max(width, 0.0f));
            return;
        }
        case compiler::ValueOperation::bump_samples: {
            require_immediate_subset(variant, 0x7u);
            const auto height_center =
                operands.scalar(operand::bump_samples::height_center);
            const auto height_x =
                operands.scalar(operand::bump_samples::height_x);
            const auto height_y =
                operands.scalar(operand::bump_samples::height_y);
            const auto strength =
                operands.scalar(operand::bump_samples::strength);
            const auto distance =
                operands.scalar(operand::bump_samples::distance);
            const auto filter_width =
                operands.scalar(operand::bump_samples::filter_width);
            const auto linked_normal =
                operands.vector(operand::bump_samples::normal);
            const auto encoded = surface_value_immediate(instruction);
            const auto configuration = SurfaceBumpSvmConfiguration{
                .invert = (encoded & 1u) != 0u,
                .normal_linked = (encoded & 2u) != 0u,
                .object_space = (encoded & 4u) != 0u};
            const auto normal = select(point.shading_normal, linked_normal,
                                       configuration.normal_linked);
            const auto value = evaluate_surface_bump(
                services, point, configuration, normal, filter_width,
                height_center, height_x, height_y, distance, strength);
            write_surface_value_vector(locals, instruction, value);
            return;
        }
        default:
            std::abort();
    }
}

} // namespace

void emit_direct_surface_state_family(
    compiler::SurfaceSvmValueOpcode family,
    const SurfaceValueRuntime &runtime,
    SurfaceValueBytecodeSlots bytecode_slots,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction,
    const compiler::SurfaceValueStaticVariant &variant,
    SurfaceValueOperandReader &operands) noexcept {
    switch (family) {
        case compiler::SurfaceSvmValueOpcode::geometry:
            emit_geometry_family(services, point, locals, instruction, variant);
            return;
        case compiler::SurfaceSvmValueOpcode::geometry_derivative:
            emit_geometry_derivative_family(services, point, locals,
                                            instruction, variant, operands);
            return;
        case compiler::SurfaceSvmValueOpcode::tex_coord:
            emit_tex_coord_family(runtime, bytecode_slots, services, point,
                                  locals, instruction, variant, operands);
            return;
        case compiler::SurfaceSvmValueOpcode::tex_coord_derivative:
            emit_tex_coord_derivative_family(
                runtime, bytecode_slots, services, point, locals, instruction,
                variant, operands);
            return;
        case compiler::SurfaceSvmValueOpcode::bump_support:
            emit_bump_support_family(services, point, locals, instruction,
                                     variant, operands);
            return;
        default:
            std::abort();
    }
}

} // namespace psycles::luisa_backend::detail

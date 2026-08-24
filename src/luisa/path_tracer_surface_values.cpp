#include "path_tracer_surface_values.h"

#include "graph_surface_internal.h"
#include "path_tracer_attribute_lookup.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surface_closure_setup.h"
#include "path_tracer_surfaces.h"
#include "path_tracer_texture_sampling.h"
#include "principled_layer_component.h"
#include "surface_bump.h"
#include "surface_preparation_accumulator.h"

#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/cycles_closure.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include <luisa/core/logging.h>
#include <luisa/dsl/local.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace value_operand = compiler::value_operand;

struct SurfaceValueLocals {
    luisa::compute::Local<float> scalars{
        SurfaceValueRuntime::scalar_capacity};
    luisa::compute::Local<luisa::float3> vectors{
        SurfaceValueRuntime::vector_capacity};
    luisa::compute::Local<luisa::ulong> unsigned_integers{
        SurfaceValueRuntime::unsigned_integer_capacity};

    void define_all() const noexcept {
        // The host compiler proves that every legal bytecode operand reads a
        // slot written earlier in the same program. Materialize the stronger
        // device-local invariant only where the bank is in coroutine root
        // scope: it is fully defined at its lexical lifetime boundary. This
        // lets the alloca-scope proof move the bank below shade_surface
        // without depending on immutable buffer contents that are
        // intentionally opaque to XIR. Callable-local banks already have the
        // smaller lexical lifetime and do not need these stores.
        for (auto i = 0u; i < SurfaceValueRuntime::scalar_capacity; ++i) {
            scalars.write(i, 0.0f);
        }
        for (auto i = 0u; i < SurfaceValueRuntime::vector_capacity; ++i) {
            vectors.write(i, make_float3(0.0f));
        }
        for (auto i = 0u;
             i < SurfaceValueRuntime::unsigned_integer_capacity; ++i) {
            unsigned_integers.write(i, 0ull);
        }
    }
};

enum class SurfaceValueBankDefinition {
    program_prefix,
    full_bank,
};

template<typename T>
[[nodiscard]] auto surface_value_runtime_buffer(
    const SurfaceValueRuntime &runtime,
    SurfaceValueRuntimeBufferSlot slot) noexcept {
    return runtime.device_view->buffer<T>(
        surface_value_runtime_buffer_slot(slot), false, true);
}

using SurfaceValueHeightCallable = Callable<float(
    Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
    BindlessArray, SurfacePointCall, luisa::uint)>;

using SurfaceValueNodes =
    std::vector<std::unique_ptr<ValueNode>>;

[[nodiscard]] Float read_scalar_dynamic(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    UInt address) noexcept {
    Float result = 0.0f;
    $if((address &
         compiler::SurfaceValueAddress::parameter_bit) != 0u) {
        result = services.parameter_float(
            point.parameter_block,
            address & compiler::SurfaceValueAddress::index_mask);
    }
    $else {
        result = locals.scalars.read(
            address & compiler::SurfaceValueAddress::index_mask);
    };
    return result;
}

[[nodiscard]] Float3 read_vector_dynamic(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    UInt address) noexcept {
    Float3 result = make_float3(0.0f);
    $if((address &
         compiler::SurfaceValueAddress::parameter_bit) != 0u) {
        result = services.parameter_float3(
            point.parameter_block,
            address & compiler::SurfaceValueAddress::index_mask);
    }
    $else {
        result = locals.vectors.read(
            address & compiler::SurfaceValueAddress::index_mask);
    };
    return result;
}

[[nodiscard]] ULong read_unsigned_integer_dynamic(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    UInt address) noexcept {
    ULong result = 0ull;
    $if((address &
         compiler::SurfaceValueAddress::parameter_bit) != 0u) {
        result = services.parameter_uint64(
            point.parameter_block,
            address & compiler::SurfaceValueAddress::index_mask);
    }
    $else {
        result = locals.unsigned_integers.read(
            address & compiler::SurfaceValueAddress::index_mask);
    };
    return result;
}

[[nodiscard]] SurfaceValueExpression read_dynamic_value(
    contract::SocketType type,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    UInt address) noexcept {
    switch (surface_value_category(type)) {
        case SurfaceValueCategory::scalar: {
            const auto value = read_scalar_dynamic(
                services, point, locals, address);
            return SurfaceValueExpression::from_scalar(
                Expr<float>{value.expression()});
        }
        case SurfaceValueCategory::vector: {
            const auto value = read_vector_dynamic(
                services, point, locals, address);
            return SurfaceValueExpression::from_vector(
                Expr<luisa::float3>{value.expression()});
        }
        case SurfaceValueCategory::unsigned_integer: {
            const auto value = read_unsigned_integer_dynamic(
                services, point, locals, address);
            return SurfaceValueExpression::from_unsigned_integer(
                Expr<luisa::ulong>{value.expression()});
        }
    }
    std::abort();
}

void write_dynamic_value(
    contract::SocketType type,
    const SurfaceValueLocals &locals,
    UInt address,
    const SurfaceValueExpression &value) noexcept {
    const auto index =
        address & compiler::SurfaceValueAddress::index_mask;
    switch (surface_value_category(type)) {
        case SurfaceValueCategory::scalar:
            locals.scalars.write(index, value.scalar());
            return;
        case SurfaceValueCategory::vector:
            locals.vectors.write(index, value.vector());
            return;
        case SurfaceValueCategory::unsigned_integer:
            locals.unsigned_integers.write(
                index, value.unsigned_integer());
            return;
    }
    std::abort();
}

[[nodiscard]] TracedValues load_variant_operands(
    const compiler::SurfaceValueStaticVariant &variant,
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    Var<luisa::uint4> instruction) noexcept {
    TracedValues operands;
    operands.shading_normal = point.shading_normal;
    operands.values.reserve(variant.operand_types.size());
    for (auto operand_index = std::size_t{0u};
         operand_index < variant.operand_types.size(); ++operand_index) {
        auto address = surface_value_runtime_buffer<luisa::uint>(
                           runtime,
                           SurfaceValueRuntimeBufferSlot::operand)
                           .read(
            instruction.z + static_cast<std::uint32_t>(operand_index));
        operands.values.emplace_back(read_dynamic_value(
            variant.operand_types[operand_index],
            services,
            point,
            locals,
            std::move(address)));
    }
    return operands;
}

[[nodiscard]] SurfaceValueExpression evaluate_non_bump_variant(
    const compiler::SurfaceValueStaticVariant &variant,
    const ValueNode &node,
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    TracedValues &operands,
    Var<luisa::uint4> instruction) noexcept {
    const auto table_parameter =
        variant.instruction.operation ==
            compiler::ValueOperation::color_ramp ||
        variant.instruction.operation ==
            compiler::ValueOperation::rgb_curve;
    const auto static_table = !variant.instruction.static_table.empty();
    if (table_parameter || static_table) {
        auto parameter = surface_value_runtime_buffer<luisa::uint>(
                             runtime,
                             SurfaceValueRuntimeBufferSlot::metadata_parameter)
                             .read(instruction.w);
        auto static_range =
            surface_value_runtime_buffer<luisa::uint2>(
                runtime,
                SurfaceValueRuntimeBufferSlot::metadata_static_range)
                .read(instruction.w);
        const Expr<std::uint32_t> parameter_expression{
            parameter.expression()};
        const ValueStaticTableView static_table_view{
            .resources = Expr<BindlessArray>{runtime.device_view},
            .buffer_slot = surface_value_runtime_buffer_slot(
                SurfaceValueRuntimeBufferSlot::static_data),
            .begin = Expr<std::uint32_t>{static_range.x.expression()}};
        ValueEvaluationContext context{
            .services = services,
            .point = point,
            .result = operands,
            .surface = nullptr,
            .parameter_override = table_parameter
                                      ? &parameter_expression
                                      : nullptr,
            .static_table_override = static_table
                                         ? &static_table_view
                                         : nullptr};
        return node.evaluate(context);
    }
    ValueEvaluationContext context{
        .services = services,
        .point = point,
        .result = operands,
        .surface = nullptr};
    return node.evaluate(context);
}

[[nodiscard]] SurfaceValueExpression evaluate_bump_variant(
    const compiler::SurfaceValueStaticVariant &variant,
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    TracedValues &operands,
    Var<luisa::uint4> instruction,
    UInt instruction_index,
    const SurfaceValueHeightCallable &height,
    Expr<Buffer<float>> scalar_parameters,
    Expr<Buffer<luisa::float3>> vector_parameters,
    Expr<Buffer<float>> cycles_bsdf_tables,
    Expr<BindlessArray> textures,
    Expr<BindlessArray> geometry_heap) noexcept {
    const auto configuration =
        decode_surface_bump_configuration(
            variant.instruction.static_u0);
    const auto normal = configuration.normal_linked
                            ? vector(
                                  variant.instruction.operand(
                                      value_operand::bump::normal),
                                  operands)
                            : operands.shading_normal;
    const auto domain = make_surface_bump_evaluation_domain(
        point,
        scalar(
            variant.instruction.operand(
                value_operand::bump::filter_width),
            operands));
    const auto height_program =
        surface_value_runtime_buffer<luisa::uint>(
            runtime,
            SurfaceValueRuntimeBufferSlot::bump_height_program)
            .read(instruction_index);
    const auto height_x = height(
        scalar_parameters,
        vector_parameters,
        cycles_bsdf_tables,
        textures,
        geometry_heap,
        pack_surface_point(domain.point_x),
        height_program);
    const auto height_y = height(
        scalar_parameters,
        vector_parameters,
        cycles_bsdf_tables,
        textures,
        geometry_heap,
        pack_surface_point(domain.point_y),
        height_program);
    const auto result = evaluate_surface_bump(
        services,
        point,
        configuration,
        normal,
        domain,
        scalar(
            variant.instruction.operand(
                value_operand::bump::height),
            operands),
        Expr<float>{height_x.expression()},
        Expr<float>{height_y.expression()},
        scalar(
            variant.instruction.operand(
                value_operand::bump::distance),
            operands),
        scalar(
            variant.instruction.operand(
                value_operand::bump::strength),
            operands));
    return SurfaceValueExpression::from_vector(
        Expr<luisa::float3>{result.expression()});
}

void emit_surface_value_program(
    const SurfaceValueRuntime &runtime,
    const SurfaceValueNodes &nodes,
    const ShaderServices &services,
    const SurfacePoint &point,
    UInt program,
    const SurfaceValueLocals &locals,
    const SurfaceValueHeightCallable *height,
    Expr<Buffer<float>> scalar_parameters,
    Expr<Buffer<luisa::float3>> vector_parameters,
    Expr<Buffer<float>> cycles_bsdf_tables,
    Expr<BindlessArray> textures,
    Expr<BindlessArray> geometry_heap) noexcept {
    auto range = surface_value_runtime_buffer<luisa::uint4>(
                     runtime,
                     SurfaceValueRuntimeBufferSlot::program)
                     .read(program);
    UInt instruction_index = range.x;
    const auto instruction_end = range.x + range.y;
    $while(instruction_index < instruction_end) {
        Var<luisa::uint4> instruction =
            surface_value_runtime_buffer<luisa::uint4>(
                runtime,
                SurfaceValueRuntimeBufferSlot::instruction)
                .read(instruction_index);
        auto variant_index =
            surface_value_runtime_buffer<luisa::uint>(
                runtime,
                SurfaceValueRuntimeBufferSlot::instruction_variant)
                .read(instruction_index);
        luisa::compute::detail::SwitchStmtBuilder{variant_index} % [&] {
            for (auto index = std::size_t{0u}; index < nodes.size();
                 ++index) {
                luisa::compute::detail::SwitchCaseStmtBuilder{
                    static_cast<luisa::uint>(index)} %
                    [&, index] {
                        const auto &variant =
                            runtime.executable.executable.variants[index];
                        auto operands = load_variant_operands(
                            variant,
                            runtime,
                            services,
                            point,
                            locals,
                            instruction);
                        if (variant.instruction.operation ==
                            compiler::ValueOperation::bump) {
                            if (height == nullptr) {
                                luisa::compute::dsl::unreachable(
                                    "Bump reached the Bump-free height evaluator");
                                return;
                            }
                            auto value = evaluate_bump_variant(
                                variant,
                                runtime,
                                services,
                                point,
                                operands,
                                instruction,
                                instruction_index,
                                *height,
                                scalar_parameters,
                                vector_parameters,
                                cycles_bsdf_tables,
                                textures,
                                geometry_heap);
                            write_dynamic_value(
                                variant.instruction.result_type,
                                locals,
                                instruction.y,
                                value);
                            return;
                        }
                        auto value = evaluate_non_bump_variant(
                            variant,
                            *nodes[index],
                            runtime,
                            services,
                            point,
                            operands,
                            instruction);
                        write_dynamic_value(
                            variant.instruction.result_type,
                            locals,
                            instruction.y,
                            value);
                    };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable(
                    "invalid compact surface value variant");
            };
        };
        instruction_index += 1u;
    };
}

[[nodiscard]] Float read_closure_scalar_or(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    Var<luisa::uint4> instruction,
    std::size_t operand_index,
    Float fallback) noexcept {
    const auto address =
        surface_value_runtime_buffer<luisa::uint>(
            runtime,
            SurfaceValueRuntimeBufferSlot::closure_operand)
            .read(
                instruction.y +
                static_cast<std::uint32_t>(operand_index));
    Float result = fallback;
    $if(address != compiler::SurfaceValueAddress::invalid_value) {
        result = read_scalar_dynamic(
            services, point, locals, address);
    };
    return result;
}

[[nodiscard]] Float3 read_closure_vector_or(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    Var<luisa::uint4> instruction,
    std::size_t operand_index,
    Float3 fallback) noexcept {
    const auto address =
        surface_value_runtime_buffer<luisa::uint>(
            runtime,
            SurfaceValueRuntimeBufferSlot::closure_operand)
            .read(
                instruction.y +
                static_cast<std::uint32_t>(operand_index));
    Float3 result = fallback;
    $if(address != compiler::SurfaceValueAddress::invalid_value) {
        result = read_vector_dynamic(
            services, point, locals, address);
    };
    return result;
}

[[nodiscard]] Float closure_mix_weight(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    Var<luisa::uint4> instruction) noexcept {
    Float weight = 1.0f;
    UInt term_index = instruction.z;
    const auto term_end = instruction.z + instruction.w;
    $while(term_index < term_end) {
        const auto term =
            surface_value_runtime_buffer<luisa::uint2>(
                runtime,
                SurfaceValueRuntimeBufferSlot::closure_mix_term)
                .read(term_index);
        const auto factor = clamp(
            read_scalar_dynamic(
                services, point, locals, term.x),
            0.0f,
            1.0f);
        const auto complement =
            (term.y & compiler::surface_closure_mix_complement) != 0u;
        weight *= select(factor, 1.0f - factor, complement);
        term_index += 1u;
    };
    return weight;
}

[[nodiscard]] TracedClosure decode_surface_closure(
    std::uint32_t static_variant,
    compiler::PrincipledClosureFeatureMask scene_features,
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    Var<luisa::uint4> instruction,
    Float mix_weight) noexcept {
    namespace operand = compiler::surface_closure_operand;
    const auto operation = static_cast<compiler::ClosureOperation>(
        static_variant & compiler::surface_closure_opcode_mask);
    const auto bssrdf_method = static_cast<compiler::BssrdfMethod>(
        (static_variant & compiler::surface_closure_bssrdf_method_mask) >>
        compiler::surface_closure_bssrdf_method_shift);
    const auto coat_normal_linked =
        (static_variant &
         compiler::surface_closure_coat_normal_linked) != 0u;
    const auto preserve_ggx_energy =
        (static_variant &
         compiler::surface_closure_preserve_ggx_energy) != 0u;
    const auto beckmann =
        (static_variant & compiler::surface_closure_beckmann) != 0u;

    switch (operation) {
        case compiler::ClosureOperation::diffuse:
        case compiler::ClosureOperation::glossy: {
            const auto glossy =
                operation == compiler::ClosureOperation::glossy;
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::diffuse::color,
                make_float3(0.0f));
            const auto roughness = read_closure_scalar_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::diffuse::roughness,
                0.0f);
            return TracedClosure{
                .operation = operation,
                .weight = bsdf_allocated_weight(color * mix_weight),
                .color = color,
                .normal = safe_normalize(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::diffuse::normal,
                        point.shading_normal),
                    point.shading_normal),
                .roughness = roughness,
                .diffuse_roughness = roughness,
                .metallic = glossy ? 1.0f : 0.0f,
                .ior = 1.5f,
                .specular_ior_level = 0.5f,
                .specular_tint = make_float3(1.0f),
                .preserve_ggx_energy =
                    glossy && preserve_ggx_energy,
                .beckmann = glossy && beckmann};
        }
        case compiler::ClosureOperation::translucent: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::translucent::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = bsdf_allocated_weight(color * mix_weight),
                .color = color,
                .normal = safe_normalize(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::translucent::normal,
                        point.shading_normal),
                    point.shading_normal),
                .roughness = 0.0f,
                .diffuse_roughness = 0.0f,
                .metallic = 0.0f,
                .ior = 1.0f,
                .specular_ior_level = 0.0f,
                .specular_tint = make_float3(1.0f)};
        }
        case compiler::ClosureOperation::principled: {
            const auto scalar = [&](std::size_t index, float fallback) noexcept {
                return read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    index,
                    Float{fallback});
            };
            const auto vector = [&](std::size_t index,
                                    luisa::float3 fallback) noexcept {
                return read_closure_vector_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    index,
                    Float3{fallback});
            };
            const auto emission_color = vector(
                operand::principled::emission_color,
                luisa::make_float3(0.0f));
            const auto emission_strength = scalar(
                operand::principled::emission_strength, 0.0f);
            return TracedClosure{
                .operation = operation,
                .principled_features = scene_features,
                .weight = make_float3(mix_weight),
                .color = vector(
                    operand::principled::color,
                    luisa::make_float3(0.0f)),
                .normal = safe_normalize(
                    vector(
                        operand::principled::normal,
                        luisa::make_float3(0.0f)),
                    point.shading_normal),
                .roughness = scalar(
                    operand::principled::roughness, 0.0f),
                .diffuse_roughness = scalar(
                    operand::principled::diffuse_roughness, 0.0f),
                .subsurface_weight = scalar(
                    operand::principled::subsurface_weight, 0.0f),
                .subsurface_radius = vector(
                    operand::principled::subsurface_radius,
                    luisa::make_float3(0.0f)),
                .subsurface_scale = scalar(
                    operand::principled::subsurface_scale, 0.0f),
                .subsurface_method = bssrdf_method,
                .subsurface_ior = scalar(
                    operand::principled::subsurface_ior, 1.4f),
                .subsurface_anisotropy = scalar(
                    operand::principled::subsurface_anisotropy, 0.0f),
                .transmission_weight = scalar(
                    operand::principled::transmission_weight, 0.0f),
                .metallic = clamp(
                    scalar(operand::principled::metallic, 0.0f),
                    0.0f,
                    1.0f),
                .ior = max(
                    scalar(operand::principled::ior, 1.0f),
                    1.0e-5f),
                .specular_ior_level = max(
                    scalar(
                        operand::principled::specular_ior_level,
                        0.0f),
                    0.0f),
                .specular_tint = max(
                    vector(
                        operand::principled::specular_tint,
                        luisa::make_float3(1.0f)),
                    make_float3(0.0f)),
                .alpha = scalar(operand::principled::alpha, 1.0f),
                .thin_wall =
                    scalar(operand::principled::thin_wall, 0.0f) != 0.0f,
                .sheen_weight = scalar(
                    operand::principled::sheen_weight, 0.0f),
                .sheen_roughness = scalar(
                    operand::principled::sheen_roughness, 0.5f),
                .sheen_tint = vector(
                    operand::principled::sheen_tint,
                    luisa::make_float3(1.0f)),
                .coat_weight = scalar(
                    operand::principled::coat_weight, 0.0f),
                .coat_roughness = scalar(
                    operand::principled::coat_roughness, 0.03f),
                .coat_ior = scalar(
                    operand::principled::coat_ior, 1.5f),
                .coat_tint = vector(
                    operand::principled::coat_tint,
                    luisa::make_float3(1.0f)),
                .coat_normal = vector(
                    operand::principled::coat_normal,
                    luisa::make_float3(0.0f)),
                .coat_normal_linked = coat_normal_linked,
                .emission = emission_color * emission_strength,
                .preserve_ggx_energy = preserve_ggx_energy,
                .beckmann = false};
        }
        case compiler::ClosureOperation::glass:
        case compiler::ClosureOperation::refraction: {
            const auto glass =
                operation == compiler::ClosureOperation::glass;
            const auto color = max(
                read_closure_vector_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::glass::color,
                    make_float3(0.0f)),
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = glass ? make_float3(mix_weight)
                                : color * mix_weight,
                .color = color,
                .normal = safe_normalize(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::glass::normal,
                        point.shading_normal),
                    point.shading_normal),
                .roughness = read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::glass::roughness,
                    0.0f),
                .diffuse_roughness = 0.0f,
                .metallic = 0.0f,
                .ior = max(
                    read_closure_scalar_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::glass::ior,
                        1.0f),
                    1.0e-5f),
                .specular_ior_level = 0.5f,
                .specular_tint = make_float3(1.0f),
                .preserve_ggx_energy =
                    glass && preserve_ggx_energy,
                .beckmann = beckmann};
        }
        case compiler::ClosureOperation::emission: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::emission::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = color *
                          read_closure_scalar_or(
                              runtime,
                              services,
                              point,
                              locals,
                              instruction,
                              operand::emission::strength,
                              0.0f) *
                          mix_weight,
                .color = color,
                .normal = make_float3(0.0f, 0.0f, 1.0f),
                .roughness = 0.0f,
                .metallic = 0.0f,
                .ior = 1.0f};
        }
        case compiler::ClosureOperation::transparent: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::transparent::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = color * mix_weight,
                .color = color,
                .normal = point.shading_normal,
                .roughness = 0.0f,
                .metallic = 0.0f,
                .ior = 1.0f};
        }
        case compiler::ClosureOperation::subsurface: {
            const auto color = read_closure_vector_or(
                runtime,
                services,
                point,
                locals,
                instruction,
                operand::subsurface::color,
                make_float3(0.0f));
            return TracedClosure{
                .operation = operation,
                .weight = color * mix_weight,
                .color = color,
                .normal = safe_normalize(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::subsurface::normal,
                        point.shading_normal),
                    point.shading_normal),
                .roughness = read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::subsurface::roughness,
                    0.0f),
                .subsurface_radius = max(
                    read_closure_vector_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::subsurface::radius,
                        make_float3(0.0f)),
                    make_float3(0.0f)),
                .subsurface_scale = max(
                    read_closure_scalar_or(
                        runtime,
                        services,
                        point,
                        locals,
                        instruction,
                        operand::subsurface::scale,
                        0.0f),
                    0.0f),
                .subsurface_method = bssrdf_method,
                .subsurface_ior = read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::subsurface::ior,
                    1.4f),
                .subsurface_anisotropy = read_closure_scalar_or(
                    runtime,
                    services,
                    point,
                    locals,
                    instruction,
                    operand::subsurface::anisotropy,
                    0.0f),
                .ior = 1.0f};
        }
        case compiler::ClosureOperation::null_closure:
        case compiler::ClosureOperation::add:
        case compiler::ClosureOperation::mix:
            break;
    }
    std::abort();
}

template<typename Visitor>
void emit_surface_closure_program(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    UInt instruction_begin,
    UInt instruction_end,
    Visitor &&visit) noexcept {
    UInt instruction_index = instruction_begin;
    $while(instruction_index < instruction_end) {
        Var<luisa::uint4> instruction =
            surface_value_runtime_buffer<luisa::uint4>(
                runtime,
                SurfaceValueRuntimeBufferSlot::closure_instruction)
                .read(instruction_index);
        const auto static_variant =
            instruction.x &
            compiler::surface_closure_static_variant_mask;
        const auto endpoints =
            (instruction.x &
             compiler::surface_closure_endpoint_mask) >>
            compiler::surface_closure_endpoint_shift;
        const auto mix_weight = closure_mix_weight(
            runtime,
            services,
            point,
            locals,
            instruction);
        luisa::compute::detail::SwitchStmtBuilder{static_variant} % [&] {
            for (const auto variant :
                 runtime.closure_static_variants) {
                luisa::compute::detail::SwitchCaseStmtBuilder{variant} %
                    [&, variant] {
                        const auto closure = decode_surface_closure(
                            variant,
                            runtime.used_principled_closure_features,
                            runtime,
                            services,
                            point,
                            locals,
                            instruction,
                            mix_weight);
                        visit(closure, endpoints, instruction_index);
                    };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable(
                    "invalid compact surface closure variant");
            };
        };
        instruction_index += 1u;
    };
}

[[nodiscard]] SurfaceClosureRecord merged_transparent_closure(
    const SurfacePoint &point,
    Float3 weight,
    Float sample_weight_value) noexcept {
    return canonical_surface_closure(TracedClosure{
        .operation = compiler::ClosureOperation::transparent,
        .weight = weight,
        .allocation_weight = sample_weight_value,
        .sample_weight = sample_weight_value,
        .setup_valid = true,
        .albedo = weight,
        .color = make_float3(1.0f),
        .normal = point.shading_normal,
        .roughness = 0.0f,
        .ior = 1.0f,
        .evaluation_scale = make_float3(1.0f)});
}

template<typename PhysicalClosureSink>
[[nodiscard]] SurfacePopulation execute_surface_closure_program(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    UInt program,
    Expr<bool> emission_reflective_caustics,
    Expr<bool> reflective_caustics,
    Expr<bool> refractive_caustics,
    PhysicalClosureSink &&emit_physical) noexcept {
    const auto range = surface_value_runtime_buffer<luisa::uint4>(
                           runtime,
                           SurfaceValueRuntimeBufferSlot::program)
                           .read(program);
    const auto closure_begin = range.z;
    const auto closure_end = range.z + range.w;
    const PrincipledLayerComponent principled_layers{services, point};
    Float3 emission = make_float3(0.0f);
    Float3 transparent_weight = make_float3(0.0f);
    Float transparent_sample_weight = 0.0f;
    Bool transparent_pending = false;
    UInt replay_begin = closure_end;

    emit_surface_closure_program(
        runtime,
        services,
        point,
        locals,
        closure_begin,
        closure_end,
        [&](const TracedClosure &raw,
            UInt endpoints,
            UInt instruction_index) noexcept {
            const auto emission_endpoint =
                (endpoints & compiler::surface_closure_endpoint_bit(
                                 compiler::SurfaceClosureEndpoint::emission)) !=
                0u;
            if (raw.operation ==
                compiler::ClosureOperation::emission) {
                $if(emission_endpoint) {
                    emission += raw.weight;
                };
            } else if (
                raw.operation ==
                    compiler::ClosureOperation::principled &&
                (runtime.used_principled_closure_features &
                 compiler::principled_closure_feature_bit(
                     compiler::PrincipledClosureFeature::emission)) != 0u) {
                const auto contribution = principled_layers
                                              .evaluate_emission(
                                                  raw,
                                                  raw.principled_features,
                                                  emission_reflective_caustics)
                                              .radiance;
                $if(emission_endpoint) {
                    emission += contribution;
                };
            }

            const auto physical_endpoint =
                (endpoints & compiler::surface_closure_endpoint_bit(
                                 compiler::SurfaceClosureEndpoint::physical)) !=
                0u;
            $if(physical_endpoint) {
                expand_physical_surface_closure(
                    services,
                    point,
                    raw,
                    reflective_caustics,
                    refractive_caustics,
                    [&](const TracedClosure &physical) noexcept {
                        if (physical.operation ==
                            compiler::ClosureOperation::transparent) {
                            transparent_weight += physical.weight;
                            transparent_sample_weight +=
                                physical.sample_weight;
                            const auto allocated =
                                physical.sample_weight >=
                                cycles_closure::closure_weight_cutoff;
                            $if(!transparent_pending & allocated) {
                                replay_begin = instruction_index;
                            };
                            transparent_pending |= allocated;
                            return;
                        }
                        $if(!transparent_pending) {
                            emit_physical(
                                canonical_surface_closure(physical));
                        };
                    });
            };
        });

    $if(transparent_pending) {
        emit_physical(merged_transparent_closure(
            point,
            transparent_weight,
            transparent_sample_weight));
    };

    // The first pass has already evaluated values and found the exact raw
    // instruction containing the first allocated transparent output. Replaying
    // only from that leaf reconstructs the non-transparent suffix in source
    // order without retaining a per-thread physical closure arena. Pure setup
    // is deterministic, so the retained sequence and left-fold order are
    // identical to the expanded route.
    Bool reached_first_transparent = false;
    $if(transparent_pending) {
        emit_surface_closure_program(
            runtime,
            services,
            point,
            locals,
            replay_begin,
            closure_end,
            [&](const TracedClosure &raw,
                UInt endpoints,
                UInt) noexcept {
                const auto physical_endpoint =
                    (endpoints & compiler::surface_closure_endpoint_bit(
                                     compiler::SurfaceClosureEndpoint::physical)) !=
                    0u;
                $if(physical_endpoint) {
                    expand_physical_surface_closure(
                        services,
                        point,
                        raw,
                        reflective_caustics,
                        refractive_caustics,
                        [&](const TracedClosure &physical) noexcept {
                            if (physical.operation ==
                                compiler::ClosureOperation::transparent) {
                                reached_first_transparent |=
                                    physical.sample_weight >=
                                    cycles_closure::closure_weight_cutoff;
                                return;
                            }
                            $if(reached_first_transparent) {
                                emit_physical(
                                    canonical_surface_closure(physical));
                            };
                        });
                };
            });
    };
    return {
        .emission = std::move(emission),
        .shading_normal = point.shading_normal};
}

[[nodiscard]] SurfacePreparation prepare_surface_closure_program(
    const SurfaceValueRuntime &runtime,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    UInt program,
    const SurfacePreparationQuery &query,
    const SurfaceClosureIdentityCallable &identity,
    const SurfaceClosureAovCallable &aov_operation) noexcept {
    SurfacePreparationAccumulator accumulator{
        point,
        maximum_surface_closure_capacity,
        query.glossy_filter_roughness,
        query.include_runtime_flags,
        query.include_aov,
        identity,
        aov_operation};
    const auto population = execute_surface_closure_program(
        runtime,
        services,
        point,
        locals,
        program,
        query.emission_reflective_caustics,
        query.reflective_caustics,
        query.refractive_caustics,
        [&](const SurfaceClosureRecord &closure) noexcept {
            accumulator.add(closure);
        });
    accumulator.finish();
    return accumulator.preparation(population.emission);
}

[[nodiscard]] SurfacePoint automatic_normal_point(
    const SurfaceValueRuntime &runtime,
    UInt surface_tag,
    const SurfacePoint &point) noexcept {
    auto result = point;
    const auto use_undisplaced =
        surface_value_runtime_buffer<luisa::uint>(
            runtime,
            SurfaceValueRuntimeBufferSlot::topology_flag)
            .read(surface_tag) &
        surface_value_runtime_topology_flag(
            SurfaceValueRuntimeTopologyFlag::
                automatic_bump_uses_undisplaced_geometry);
    $if (use_undisplaced != 0u) {
        result.position = point.undisplaced_position;
        result.object_position = point.undisplaced_object_position;
        result.shading_normal = point.undisplaced_shading_normal;
        result.object_shading_normal =
            point.undisplaced_object_shading_normal;
        result.dPdx = point.undisplaced_dPdx;
        result.dPdy = point.undisplaced_dPdy;
        result.object_dPdx = point.undisplaced_object_dPdx;
        result.object_dPdy = point.undisplaced_object_dPdy;
    };
    return result;
}

[[nodiscard]] Float3 evaluate_compact_surface_normal(
    const SurfaceValueRuntime &runtime,
    const SurfaceValueNodes &nodes,
    const ShaderServices &services,
    UInt surface_tag,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    const SurfaceValueHeightCallable &height,
    Expr<Buffer<float>> scalar_parameters,
    Expr<Buffer<luisa::float3>> vector_parameters,
    Expr<Buffer<float>> cycles_bsdf_tables,
    Expr<BindlessArray> textures,
    Expr<BindlessArray> geometry_heap) noexcept {
    Float3 result = point.shading_normal;
    $if(surface_tag <
        static_cast<luisa::uint>(runtime.topologies.size())) {
        const auto normal_output =
            surface_value_runtime_buffer<luisa::uint>(
                runtime,
                SurfaceValueRuntimeBufferSlot::normal_output_address)
                .read(surface_tag);
        $if(normal_output !=
            compiler::SurfaceValueAddress::invalid_value) {
            const auto normal_point = automatic_normal_point(
                runtime, surface_tag, point);
            const auto normal_program =
                surface_tag * SurfaceValueRuntime::programs_per_topology +
                SurfaceValueRuntime::normal_program_offset;
            emit_surface_value_program(
                runtime,
                nodes,
                services,
                normal_point,
                normal_program,
                locals,
                &height,
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap);
            result = read_vector_dynamic(
                services, normal_point, locals, normal_output);
        };
    };
    return result;
}

// Execute the exact dependency-union value schedule once and expose its
// surface-local typed banks only to a continuation recorded in the same
// lexical/device control-flow region. Locals never cross the population
// boundary, and both preparation-only and populate-once routes therefore
// share automatic-normal, Bump, and value-numbering semantics.
template<typename Continuation>
void emit_compact_surface_values(
    const SurfaceValueRuntime &runtime,
    const SurfaceValueNodes &nodes,
    const ShaderServices &services,
    UInt surface_tag,
    SurfacePoint point,
    const SurfaceValueHeightCallable &height,
    Expr<Buffer<float>> scalar_parameters,
    Expr<Buffer<luisa::float3>> vector_parameters,
    Expr<Buffer<float>> cycles_bsdf_tables,
    Expr<BindlessArray> textures,
    Expr<BindlessArray> geometry_heap,
    SurfaceValueBankDefinition bank_definition,
    Continuation &&continuation) noexcept {
    $if(surface_tag <
        static_cast<luisa::uint>(runtime.topologies.size())) {
        SurfaceValueLocals locals;
        if (bank_definition == SurfaceValueBankDefinition::full_bank) {
            locals.define_all();
        }
        point.shading_normal = evaluate_compact_surface_normal(
            runtime,
            nodes,
            services,
            surface_tag,
            point,
            locals,
            height,
            scalar_parameters,
            vector_parameters,
            cycles_bsdf_tables,
            textures,
            geometry_heap);

        const auto preparation_program =
            surface_tag * SurfaceValueRuntime::programs_per_topology +
            SurfaceValueRuntime::preparation_program_offset;
        emit_surface_value_program(
            runtime,
            nodes,
            services,
            point,
            preparation_program,
            locals,
            &height,
            scalar_parameters,
            vector_parameters,
            cycles_bsdf_tables,
            textures,
            geometry_heap);
        continuation(point, locals, preparation_program);
    };
}

[[nodiscard]] std::shared_ptr<SurfaceValueNodes>
make_surface_value_nodes(
    const SurfaceValueRuntime &runtime) noexcept {
    auto nodes = std::make_shared<SurfaceValueNodes>();
    nodes->reserve(
        runtime.executable.executable.variants.size());
    for (const auto &variant :
         runtime.executable.executable.variants) {
        nodes->emplace_back(
            make_value_node(variant.instruction));
    }
    return nodes;
}

[[nodiscard]] SurfaceValueHeightCallable
make_surface_value_height_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    std::uint32_t maximum_bump_depth) noexcept {
    const auto make_stratum =
        [scene, nodes, texture_sampling, attribute_lookup](
            std::optional<SurfaceValueHeightCallable> lower,
            std::uint32_t stratum) noexcept {
      SurfaceValueHeightCallable height =
        [scene, nodes, texture_sampling, attribute_lookup,
         lower = std::move(lower)](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            Var<SurfacePointCall> packed_point,
            UInt program) noexcept {
            CallableTexture2DSamplingProvider texture_provider{
                textures, texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space,
                nullptr,
                &texture_provider,
                &attribute_provider};
            const auto point = unpack_surface_point(packed_point);
            SurfaceValueLocals locals;
            emit_surface_value_program(
                *scene->surface_values,
                *nodes,
                services,
                point,
                program,
                locals,
                lower ? &*lower : nullptr,
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap);
            const auto output =
                surface_value_runtime_buffer<luisa::uint>(
                    *scene->surface_values,
                    SurfaceValueRuntimeBufferSlot::program_output)
                    .read(program);
            return read_scalar_dynamic(
                services, point, locals, output);
        };
      const auto name = luisa::format(
          "surface_value_height_stratum_{}", stratum);
      height.set_name(name);
      return height;
    };
    const auto stratum_count = std::max(maximum_bump_depth, 1u);
    auto height = make_stratum(std::nullopt, 0u);
    for (auto stratum = std::uint32_t{1u}; stratum < stratum_count;
         ++stratum) {
        height = make_stratum(
            std::optional<SurfaceValueHeightCallable>{height}, stratum);
    }
    return height;
}

} // namespace

namespace {

class CompactSurfacePopulationProgramImpl final
    : public SurfacePopulationProgram {

  private:
    std::shared_ptr<LuisaSceneData> _scene;
    std::shared_ptr<SurfaceValueNodes> _nodes;
    SurfaceValueHeightCallable _height;

  public:
    CompactSurfacePopulationProgramImpl(
        std::shared_ptr<LuisaSceneData> scene,
        std::shared_ptr<SurfaceValueNodes> nodes,
        SurfaceValueHeightCallable height) noexcept
        : _scene{std::move(scene)},
          _nodes{std::move(nodes)},
          _height{std::move(height)} {}

    [[nodiscard]] SurfacePopulation populate(
        Expr<std::uint32_t> surface_tag,
        const ShaderServices &services,
        const SurfacePoint &point,
        const SurfacePopulationQuery &query,
        SurfaceClosureCollector &collector) const noexcept override {
        SurfacePopulation result{
            .emission = make_float3(0.0f),
            .shading_normal = point.shading_normal};

        // begin/finish form one transaction even for an invalid runtime tag.
        // A valid compact program refines begin() with the automatic-normal
        // result before emitting its first retained physical closure.
        collector.begin(point.shading_normal);
        emit_compact_surface_values(
            *_scene->surface_values,
            *_nodes,
            services,
            UInt{surface_tag},
            point,
            _height,
            Expr<Buffer<float>>{_scene->scalar_parameter_buffer},
            Expr<Buffer<luisa::float3>>{
                _scene->vector_parameter_buffer},
            Expr<Buffer<float>>{_scene->cycles_bsdf_table_buffer},
            Expr<BindlessArray>{_scene->texture_heap},
            Expr<BindlessArray>{_scene->heap},
            SurfaceValueBankDefinition::full_bank,
            [&](const SurfacePoint &evaluated_point,
                const SurfaceValueLocals &locals,
                UInt preparation_program) noexcept {
                collector.begin(evaluated_point.shading_normal);
                const auto population =
                    execute_surface_closure_program(
                        *_scene->surface_values,
                        services,
                        evaluated_point,
                        locals,
                        preparation_program,
                        query.emission_reflective_caustics,
                        query.reflective_caustics,
                        query.refractive_caustics,
                        [&](const SurfaceClosureRecord &closure) noexcept {
                            collector.add(closure);
                        });
                result.emission = population.emission;
                result.shading_normal = population.shading_normal;
            });
        collector.finish();
        return result;
    }
};

} // namespace

std::shared_ptr<const SurfacePopulationProgram>
make_compact_surface_population_program(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene->surface_values ||
        scene->surface_values->topologies.size() !=
            scene->surfaces.size()) {
        std::abort();
    }
    auto nodes = make_surface_value_nodes(*scene->surface_values);
    const auto texture_sampling =
        make_texture_2d_sampling_callables();
    const auto attribute_lookup =
        make_surface_attribute_lookup_callable(
            scene->attribute_binding_slot,
            scene->attribute_range_slot);
    auto height = make_surface_value_height_callable(
        scene, nodes, texture_sampling, attribute_lookup,
        scene->surface_values->executable.maximum_bump_depth);
    return std::make_shared<
        CompactSurfacePopulationProgramImpl>(
        scene, std::move(nodes), std::move(height));
}

SurfacePreparationCallable
make_compact_surface_preparation_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene->surface_values ||
        scene->surface_values->topologies.size() !=
            scene->surfaces.size()) {
        std::abort();
    }
    const auto &runtime = *scene->surface_values;
    auto nodes = make_surface_value_nodes(runtime);

    const auto closure_setup =
        make_surface_closure_setup_callables();
    const auto closure_identity =
        make_surface_closure_identity_callable();
    const auto closure_aov =
        make_surface_closure_aov_callable();
    const auto texture_sampling =
        make_texture_2d_sampling_callables();
    const auto attribute_lookup =
        make_surface_attribute_lookup_callable(
            scene->attribute_binding_slot,
            scene->attribute_range_slot);

    auto height = make_surface_value_height_callable(
        scene, nodes, texture_sampling, attribute_lookup,
        runtime.executable.maximum_bump_depth);

    SurfacePreparationCallable preparation =
        [scene,
         nodes,
         height = std::move(height),
         closure_setup,
         closure_identity,
         closure_aov,
         texture_sampling,
         attribute_lookup](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Var<SurfacePreparationQueryCall> packed_query) noexcept {
            CallableSurfaceClosureSetupProvider setup_provider{
                cycles_bsdf_tables, closure_setup};
            CallableTexture2DSamplingProvider texture_provider{
                textures, texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                scene->attribute_binding_slot,
                scene->attribute_range_slot,
                scene->nishita_texture_bindings,
                scene->shader_color_space,
                &setup_provider,
                &texture_provider,
                &attribute_provider};
            auto point = unpack_surface_point(packed_point);
            Var<SurfacePreparationCall> result =
                pack_surface_preparation(
                    SurfacePreparation::zero(point));
            emit_compact_surface_values(
                *scene->surface_values,
                *nodes,
                services,
                surface_tag,
                point,
                height,
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap,
                SurfaceValueBankDefinition::program_prefix,
                [&](const SurfacePoint &evaluated_point,
                    const SurfaceValueLocals &locals,
                    UInt preparation_program) noexcept {
                    const auto query =
                        unpack_surface_preparation_query(packed_query);
                    result = pack_surface_preparation(
                        prepare_surface_closure_program(
                            *scene->surface_values,
                            services,
                            evaluated_point,
                            locals,
                            preparation_program,
                            query,
                            closure_identity,
                            closure_aov));
                });
            return result;
        };
    preparation.set_name("surface_prepare_compact_values");
    return preparation;
}

SurfaceBssrdfNormalCallable make_compact_surface_bssrdf_normal_callable(
    const std::shared_ptr<LuisaSceneData> &scene) noexcept {
    if (!scene->surface_values ||
        scene->surface_values->topologies.size() != scene->surfaces.size()) {
        std::abort();
    }
    const auto &runtime = *scene->surface_values;
    auto nodes = make_surface_value_nodes(runtime);
    const auto closure_setup = make_surface_closure_setup_callables();
    const auto texture_sampling = make_texture_2d_sampling_callables();
    const auto attribute_lookup = make_surface_attribute_lookup_callable(
        scene->attribute_binding_slot, scene->attribute_range_slot);
    auto height = make_surface_value_height_callable(
        scene, nodes, texture_sampling, attribute_lookup,
        runtime.executable.maximum_bump_depth);

    SurfaceBssrdfNormalCallable bssrdf_normal =
        [scene, nodes, height = std::move(height), closure_setup,
         texture_sampling, attribute_lookup](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, UInt surface_tag,
            Var<SurfacePointCall> packed_point, Bool has_bssrdf_bump,
            Bool reflective_caustics, Bool refractive_caustics) noexcept {
            CallableSurfaceClosureSetupProvider setup_provider{cycles_bsdf_tables,
                                                               closure_setup};
            CallableTexture2DSamplingProvider texture_provider{textures,
                                                               texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{scalar_parameters,
                                          vector_parameters,
                                          cycles_bsdf_tables,
                                          textures,
                                          geometry_heap,
                                          scene->attribute_binding_slot,
                                          scene->attribute_range_slot,
                                          scene->nishita_texture_bindings,
                                          scene->shader_color_space,
                                          &setup_provider,
                                          &texture_provider,
                                          &attribute_provider};
            const auto point = unpack_surface_point(packed_point);
            Float3 result = point.shading_normal;
            $if (has_bssrdf_bump) {
                $if (surface_tag < static_cast<luisa::uint>(
                                       scene->surface_values->topologies.size())) {
                    emit_compact_surface_values(
                        *scene->surface_values, *nodes, services, surface_tag, point,
                        height, scalar_parameters, vector_parameters,
                        cycles_bsdf_tables, textures, geometry_heap,
                        SurfaceValueBankDefinition::program_prefix,
                        [&](const SurfacePoint &evaluated_point,
                            const SurfaceValueLocals &locals,
                            UInt preparation_program) noexcept {
                            SurfaceBssrdfNormalAccumulator accumulator{
                                evaluated_point.shading_normal,
                                scene->volume_metadata.closure_allocation_budget};
                            static_cast<void>(execute_surface_closure_program(
                                *scene->surface_values, services, evaluated_point, locals,
                                preparation_program, reflective_caustics,
                                reflective_caustics, refractive_caustics,
                                [&](const SurfaceClosureRecord &closure) noexcept {
                                    accumulator.add(closure.kind, closure.weight,
                                                    closure.allocation_weight,
                                                    closure.normal);
                                }));
                            result = accumulator.result();
                        });
                };
            };
            return result;
        };
    bssrdf_normal.set_name("surface_bssrdf_normal_compact_values");
    return bssrdf_normal;
}

} // namespace psycles::luisa_backend::detail

#include "path_tracer_surface_values.h"

#include "graph_surface_internal.h"
#include "path_tracer_attribute_lookup.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surface_closure_setup.h"
#include "path_tracer_surfaces.h"
#include "path_tracer_texture_sampling.h"
#include "surface_bump.h"

#include <psycles/luisa/graph_surface.h>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>

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
};

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
        auto address = runtime.operand_buffer->read(
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
    if (table_parameter) {
        auto parameter = runtime.metadata_parameter_buffer->read(
            instruction.w);
        const Expr<std::uint32_t> parameter_expression{
            parameter.expression()};
        ValueEvaluationContext context{
            .services = services,
            .point = point,
            .result = operands,
            .surface = nullptr,
            .parameter_override = &parameter_expression};
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
    const BufferFloat &scalar_parameters,
    const BufferFloat3 &vector_parameters,
    const BufferFloat &cycles_bsdf_tables,
    const BindlessVar &textures,
    const BindlessVar &geometry_heap) noexcept {
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
        runtime.bump_height_program_buffer->read(instruction_index);
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
    const BufferFloat &scalar_parameters,
    const BufferFloat3 &vector_parameters,
    const BufferFloat &cycles_bsdf_tables,
    const BindlessVar &textures,
    const BindlessVar &geometry_heap) noexcept {
    auto range = runtime.program_buffer->read(program);
    UInt instruction_index = range.x;
    const auto instruction_end = range.x + range.y;
    $while(instruction_index < instruction_end) {
        Var<luisa::uint4> instruction =
            runtime.instruction_buffer->read(instruction_index);
        auto variant_index =
            runtime.instruction_variant_buffer->read(
                instruction_index);
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

[[nodiscard]] SurfaceValueExpression read_static_value(
    contract::SocketType type,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals,
    std::uint32_t encoded) noexcept {
    const compiler::SurfaceValueAddress address{encoded};
    if (!address.valid()) {
        return SurfaceValueExpression::zero(type);
    }
    if (address.parameter()) {
        switch (surface_value_category(type)) {
            case SurfaceValueCategory::scalar: {
                const auto value = services.parameter_float(
                    point.parameter_block, address.index());
                return SurfaceValueExpression::from_scalar(
                    Expr<float>{value.expression()});
            }
            case SurfaceValueCategory::vector: {
                const auto value = services.parameter_float3(
                    point.parameter_block, address.index());
                return SurfaceValueExpression::from_vector(
                    Expr<luisa::float3>{value.expression()});
            }
            case SurfaceValueCategory::unsigned_integer: {
                const auto value = services.parameter_uint64(
                    point.parameter_block, address.index());
                return SurfaceValueExpression::from_unsigned_integer(
                    Expr<luisa::ulong>{value.expression()});
            }
        }
        std::abort();
    }
    switch (surface_value_category(type)) {
        case SurfaceValueCategory::scalar: {
            const auto value = locals.scalars.read(address.index());
            return SurfaceValueExpression::from_scalar(
                Expr<float>{value.expression()});
        }
        case SurfaceValueCategory::vector: {
            const auto value = locals.vectors.read(address.index());
            return SurfaceValueExpression::from_vector(
                Expr<luisa::float3>{value.expression()});
        }
        case SurfaceValueCategory::unsigned_integer: {
            const auto value =
                locals.unsigned_integers.read(address.index());
            return SurfaceValueExpression::from_unsigned_integer(
                Expr<luisa::ulong>{value.expression()});
        }
    }
    std::abort();
}

[[nodiscard]] TracedValues materialize_preparation_values(
    const SurfaceValueRuntimeTopology &topology,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocals &locals) noexcept {
    TracedValues values;
    values.shading_normal = point.shading_normal;
    const auto &instructions =
        topology.program->value_instructions();
    values.values.reserve(instructions.size());
    for (auto index = std::size_t{0u}; index < instructions.size();
         ++index) {
        values.values.emplace_back(read_static_value(
            instructions[index].result_type,
            services,
            point,
            locals,
            topology.preparation_addresses[index]));
    }
    return values;
}

[[nodiscard]] SurfacePoint automatic_normal_point(
    const SurfaceValueRuntime &runtime,
    UInt surface_tag,
    const SurfacePoint &point) noexcept {
    auto result = point;
    const auto use_undisplaced =
        runtime.normal_undisplaced_flag_buffer->read(surface_tag);
    $if(use_undisplaced != 0u) {
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

[[nodiscard]] std::vector<bool> dependency_mask(
    const compiler::SurfaceProgram &program,
    compiler::ValueExpressionId root) {
    std::vector<bool> active(
        program.value_instructions().size(), false);
    std::vector<compiler::ValueExpressionId> pending;
    if (root.valid()) {
        pending.emplace_back(root);
    }
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        if (!id.valid() ||
            id.value >= active.size() ||
            active[id.value]) {
            continue;
        }
        active[id.value] = true;
        for (const auto operand :
             program.value_instructions()[id.value].operands) {
            if (operand.valid()) {
                pending.emplace_back(operand);
            }
        }
    }
    return active;
}

[[nodiscard]] bool fits_runtime_capacity(
    const compiler::SurfaceValueProgramDescriptor &program) noexcept {
    return program.scalar_slots <= SurfaceValueRuntime::scalar_capacity &&
           program.vector_slots <= SurfaceValueRuntime::vector_capacity &&
           program.unsigned_integer_slots <=
               SurfaceValueRuntime::unsigned_integer_capacity;
}

template<typename T>
void provide_dummy_if_empty(luisa::vector<T> &values, T dummy) {
    if (values.empty()) {
        values.emplace_back(std::move(dummy));
    }
}

} // namespace

std::unique_ptr<SurfaceValueRuntime> build_surface_value_runtime(
    luisa::compute::Device &device,
    std::span<const std::shared_ptr<const compiler::SurfaceProgram>> programs,
    std::span<const compiler::SurfaceClosurePlan> closure_plans,
    std::string &diagnostic) {
    diagnostic.clear();
    if (programs.empty() || programs.size() != closure_plans.size()) {
        diagnostic = "surface topology programs and closure plans do not form a non-empty bijection";
        return nullptr;
    }
    if (programs.size() >
        std::numeric_limits<std::uint32_t>::max() /
            SurfaceValueRuntime::programs_per_topology) {
        diagnostic = "surface topology count exceeds compact device program ids";
        return nullptr;
    }

    auto runtime = std::make_unique<SurfaceValueRuntime>();
    runtime->topologies.reserve(programs.size());
    std::vector<compiler::SurfaceValueStoragePlan> root_storage;
    root_storage.reserve(
        programs.size() * SurfaceValueRuntime::programs_per_topology);

    for (auto topology = std::size_t{0u}; topology < programs.size();
         ++topology) {
        const auto &program_ptr = programs[topology];
        if (!program_ptr || !closure_plans[topology].compatible(*program_ptr)) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " has no compatible closure plan";
            return nullptr;
        }
        const auto &program = *program_ptr;
        auto normal_active =
            dependency_mask(program, program.surface_normal_root());
        auto normal_outputs =
            std::vector<bool>(normal_active.size(), false);
        if (program.surface_normal_root().valid()) {
            if (program.surface_normal_root().value >= normal_outputs.size()) {
                diagnostic = "surface topology " + std::to_string(topology) +
                             " has an invalid automatic-normal root";
                return nullptr;
            }
            normal_outputs[program.surface_normal_root().value] = true;
        }
        auto normal_storage = compiler::plan_surface_value_storage(
            program, normal_active, normal_outputs);
        if (!normal_storage.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " automatic-normal plan: " +
                         normal_storage.diagnostic;
            return nullptr;
        }
        auto normal_image = compiler::lower_surface_value_program(
            program, normal_storage);
        if (!normal_image.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " automatic-normal lowering: " +
                         normal_image.diagnostic;
            return nullptr;
        }

        const auto dependencies =
            compiler::analyze_surface_value_dependencies(
                program, closure_plans[topology]);
        auto preparation_storage =
            compiler::plan_surface_value_storage(
                program,
                dependencies.preparation,
                dependencies.preparation_outputs);
        if (!preparation_storage.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " preparation plan: " +
                         preparation_storage.diagnostic;
            return nullptr;
        }
        auto preparation_image =
            compiler::lower_surface_value_program(
                program, preparation_storage);
        if (!preparation_image.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " preparation lowering: " +
                         preparation_image.diagnostic;
            return nullptr;
        }

        auto normal_output =
            compiler::SurfaceValueAddress::invalid_value;
        if (program.surface_normal_root().valid()) {
            normal_output = normal_image.value_addresses[
                program.surface_normal_root().value];
            if (normal_output ==
                compiler::SurfaceValueAddress::invalid_value) {
                diagnostic = "surface topology " +
                             std::to_string(topology) +
                             " automatic-normal root has no typed address";
                return nullptr;
            }
        }
        auto uses_undisplaced_geometry = false;
        for (auto index = std::size_t{0u};
             index < normal_active.size(); ++index) {
            const auto &instruction =
                program.value_instructions()[index];
            uses_undisplaced_geometry |=
                normal_active[index] &&
                instruction.operation ==
                    compiler::ValueOperation::bump &&
                (instruction.static_u0 & 4u) != 0u;
        }
        runtime->topologies.emplace_back(
            SurfaceValueRuntimeTopology{
                .program = program_ptr,
                .preparation_addresses =
                    std::move(preparation_image.value_addresses),
                .normal_output_address = normal_output,
                .automatic_bump_uses_undisplaced_geometry =
                    uses_undisplaced_geometry});
        runtime->normal_output_addresses.emplace_back(normal_output);
        runtime->normal_undisplaced_flags.emplace_back(
            uses_undisplaced_geometry ? 1u : 0u);
        root_storage.emplace_back(std::move(normal_storage));
        root_storage.emplace_back(std::move(preparation_storage));
    }

    std::vector<compiler::SurfaceValueExecutionInput> roots;
    roots.reserve(root_storage.size());
    for (auto topology = std::size_t{0u}; topology < programs.size();
         ++topology) {
        roots.emplace_back(compiler::SurfaceValueExecutionInput{
            .program = programs[topology].get(),
            .storage = &root_storage[
                topology * SurfaceValueRuntime::programs_per_topology +
                SurfaceValueRuntime::normal_program_offset]});
        roots.emplace_back(compiler::SurfaceValueExecutionInput{
            .program = programs[topology].get(),
            .storage = &root_storage[
                topology * SurfaceValueRuntime::programs_per_topology +
                SurfaceValueRuntime::preparation_program_offset]});
    }
    runtime->executable =
        compiler::build_surface_value_bump_executable_scene(roots);
    if (!runtime->executable.valid) {
        diagnostic = runtime->executable.diagnostic;
        return nullptr;
    }
    if (runtime->executable.root_program_count != roots.size()) {
        diagnostic = "compact surface root program count changed during Bump stratification";
        return nullptr;
    }
    const auto &image = runtime->executable.executable.values;
    for (auto index = std::size_t{0u}; index < image.programs.size();
         ++index) {
        if (!fits_runtime_capacity(image.programs[index])) {
            diagnostic = "compact surface program " + std::to_string(index) +
                         " requires typed slots beyond the 8 scalar / 12 vector / 1 uint64 validation capacity";
            return nullptr;
        }
        runtime->program_ranges.emplace_back(luisa::make_uint2(
            image.programs[index].instruction_begin,
            image.programs[index].instruction_count));
    }
    runtime->instructions.reserve(image.instructions.size());
    for (const auto &instruction : image.instructions) {
        runtime->instructions.emplace_back(luisa::make_uint4(
            instruction.control,
            instruction.result,
            instruction.operand_begin,
            instruction.metadata_index));
    }
    runtime->operands.assign(
        image.operands.begin(), image.operands.end());
    runtime->instruction_variants.assign(
        runtime->executable.executable.instruction_variants.begin(),
        runtime->executable.executable.instruction_variants.end());
    runtime->metadata_parameters.reserve(image.metadata.size());
    for (const auto &metadata : image.metadata) {
        runtime->metadata_parameters.emplace_back(metadata.parameter);
    }
    runtime->bump_height_programs.assign(
        runtime->executable.bump_height_programs.begin(),
        runtime->executable.bump_height_programs.end());
    runtime->program_outputs.assign(
        runtime->executable.program_outputs.begin(),
        runtime->executable.program_outputs.end());

    provide_dummy_if_empty(
        runtime->program_ranges, luisa::make_uint2(0u));
    provide_dummy_if_empty(
        runtime->instructions, luisa::make_uint4(0u));
    provide_dummy_if_empty(runtime->operands, 0u);
    provide_dummy_if_empty(runtime->instruction_variants, 0u);
    provide_dummy_if_empty(
        runtime->metadata_parameters,
        compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(
        runtime->bump_height_programs,
        compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(
        runtime->program_outputs,
        compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(
        runtime->normal_output_addresses,
        compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(runtime->normal_undisplaced_flags, 0u);

    runtime->program_buffer =
        device.create_buffer<luisa::uint2>(runtime->program_ranges.size());
    runtime->instruction_buffer =
        device.create_buffer<luisa::uint4>(runtime->instructions.size());
    runtime->operand_buffer =
        device.create_buffer<luisa::uint>(runtime->operands.size());
    runtime->instruction_variant_buffer =
        device.create_buffer<luisa::uint>(
            runtime->instruction_variants.size());
    runtime->metadata_parameter_buffer =
        device.create_buffer<luisa::uint>(
            runtime->metadata_parameters.size());
    runtime->bump_height_program_buffer =
        device.create_buffer<luisa::uint>(
            runtime->bump_height_programs.size());
    runtime->program_output_buffer =
        device.create_buffer<luisa::uint>(runtime->program_outputs.size());
    runtime->normal_output_address_buffer =
        device.create_buffer<luisa::uint>(
            runtime->normal_output_addresses.size());
    runtime->normal_undisplaced_flag_buffer =
        device.create_buffer<luisa::uint>(
            runtime->normal_undisplaced_flags.size());
    return runtime;
}

void upload_surface_value_runtime(
    Stream &stream,
    const SurfaceValueRuntime &runtime) noexcept {
    stream << runtime.program_buffer.copy_from(
                  luisa::span{runtime.program_ranges})
           << runtime.instruction_buffer.copy_from(
                  luisa::span{runtime.instructions})
           << runtime.operand_buffer.copy_from(
                  luisa::span{runtime.operands})
           << runtime.instruction_variant_buffer.copy_from(
                  luisa::span{runtime.instruction_variants})
           << runtime.metadata_parameter_buffer.copy_from(
                  luisa::span{runtime.metadata_parameters})
           << runtime.bump_height_program_buffer.copy_from(
                  luisa::span{runtime.bump_height_programs})
           << runtime.program_output_buffer.copy_from(
                  luisa::span{runtime.program_outputs})
           << runtime.normal_output_address_buffer.copy_from(
                  luisa::span{runtime.normal_output_addresses})
           << runtime.normal_undisplaced_flag_buffer.copy_from(
                  luisa::span{runtime.normal_undisplaced_flags});
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
    auto nodes = std::make_shared<SurfaceValueNodes>();
    nodes->reserve(
        runtime.executable.executable.variants.size());
    for (const auto &variant :
         runtime.executable.executable.variants) {
        nodes->emplace_back(
            make_value_node(variant.instruction));
    }

    const auto closure_setup =
        make_surface_closure_setup_callables();
    const auto texture_sampling =
        make_texture_2d_sampling_callables();
    const auto attribute_lookup =
        make_surface_attribute_lookup_callable(
            scene->attribute_binding_slot,
            scene->attribute_range_slot);

    SurfaceValueHeightCallable height =
        [scene, nodes, texture_sampling, attribute_lookup](
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
                nullptr,
                scalar_parameters,
                vector_parameters,
                cycles_bsdf_tables,
                textures,
                geometry_heap);
            const auto output =
                scene->surface_values->program_output_buffer->read(
                    program);
            return read_scalar_dynamic(
                services, point, locals, output);
        };
    height.set_name("surface_value_height");

    std::vector<const GraphSurfaceImplementation *> implementations;
    implementations.reserve(scene->surfaces.size());
    for (auto topology = std::size_t{0u};
         topology < scene->surfaces.size(); ++topology) {
        // compile_scene is the sole producer of this dispatch and registers
        // GraphSurface for every dense tag before constructing callables.
        const auto *surface = static_cast<const GraphSurface *>(
            scene->surfaces.implementation(topology));
        implementations.emplace_back(
            surface->internal_implementation());
    }

    SurfacePreparationCallable preparation =
        [scene,
         nodes,
         implementations = std::move(implementations),
         height = std::move(height),
         closure_setup,
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
            $if(surface_tag <
                static_cast<luisa::uint>(implementations.size())) {
                SurfaceValueLocals locals;
                const auto normal_program =
                    surface_tag *
                        SurfaceValueRuntime::programs_per_topology +
                    SurfaceValueRuntime::normal_program_offset;
                const auto normal_output =
                    scene->surface_values
                        ->normal_output_address_buffer->read(
                            surface_tag);
                $if(normal_output !=
                    compiler::SurfaceValueAddress::invalid_value) {
                    const auto normal_point = automatic_normal_point(
                        *scene->surface_values,
                        surface_tag,
                        point);
                    emit_surface_value_program(
                        *scene->surface_values,
                        *nodes,
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
                    point.shading_normal = read_vector_dynamic(
                        services,
                        normal_point,
                        locals,
                        normal_output);
                };

                const auto preparation_program =
                    surface_tag *
                        SurfaceValueRuntime::programs_per_topology +
                    SurfaceValueRuntime::preparation_program_offset;
                emit_surface_value_program(
                    *scene->surface_values,
                    *nodes,
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
                const auto query =
                    unpack_surface_preparation_query(packed_query);
                if (implementations.size() == 1u) {
                    const auto values =
                        materialize_preparation_values(
                            scene->surface_values->topologies.front(),
                            services,
                            point,
                            locals);
                    result = pack_surface_preparation(
                        implementations.front()->prepare_traced_values(
                            services, point, values, query));
                } else {
                    luisa::compute::detail::SwitchStmtBuilder{surface_tag} % [&] {
                        for (auto topology = std::size_t{0u};
                             topology < implementations.size(); ++topology) {
                            luisa::compute::detail::SwitchCaseStmtBuilder{
                                static_cast<luisa::uint>(topology)} %
                                [&, topology] {
                                    const auto values =
                                        materialize_preparation_values(
                                            scene->surface_values
                                                ->topologies[topology],
                                            services,
                                            point,
                                            locals);
                                    result = pack_surface_preparation(
                                        implementations[topology]
                                            ->prepare_traced_values(
                                                services,
                                                point,
                                                values,
                                                query));
                                };
                        }
                        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                            luisa::compute::dsl::unreachable(
                                "invalid compact surface topology tag");
                        };
                    };
                }
            };
            return result;
        };
    preparation.set_name("surface_prepare_compact_values");
    return preparation;
}

} // namespace psycles::luisa_backend::detail

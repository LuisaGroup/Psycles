#include "path_tracer_surface_value_program.h"

#include "graph_surface_internal.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surfaces.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

SurfaceValueLocalsView SurfaceValueLocals::view() const noexcept {
    static_assert(SurfaceValueRuntime::unsigned_integer_capacity == 1u);
    return {
        .scalars = {
            luisa::compute::detail::Ref<SurfaceValueScalarBank>{
                scalars.expression()}},
        .vectors = {
            luisa::compute::detail::Ref<SurfaceValueVectorBank>{
                vectors.expression()}},
        .unsigned_integers = {
            luisa::compute::detail::Ref<luisa::ulong>{
                unsigned_integers.expression()}}};
}

void SurfaceValueLocals::define_all() const noexcept {
    // The host compiler proves read-before-write for every legal bytecode
    // operand. Full definition is needed only when the bank starts in a
    // coroutine root scope, so XIR can move the alloca below shade_surface
    // without depending on opaque immutable-buffer contents.
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

Float read_scalar_dynamic(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
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

Float3 read_vector_dynamic(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
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

namespace {

using SurfaceValueNodes =
    std::vector<std::unique_ptr<ValueNode>>;

struct SurfaceValueHandlerGroup {
    std::uint32_t key{};
    std::vector<std::uint32_t> variants;
};

[[nodiscard]] std::uint32_t handler_key(
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
    compiler::SurfaceValueBank result_bank{};
    if (!compiler::classify_surface_value_type(
            variant.instruction.result_type, result_bank) ||
        variant.svm_immediates.empty()) {
        std::abort();
    }
    const auto key = compiler::make_surface_value_handler_key(
        variant.instruction.operation,
        result_bank,
        variant.svm_immediates.front());
    for (const auto immediate : variant.svm_immediates) {
        if (compiler::make_surface_value_handler_key(
                variant.instruction.operation,
                result_bank,
                immediate) != key) {
            std::abort();
        }
    }
    return key;
}

[[nodiscard]] std::vector<SurfaceValueHandlerGroup>
make_handler_groups(
    const SurfaceValueRuntime &runtime,
    std::span<const std::uint32_t> active_variants) noexcept {
    std::vector<SurfaceValueHandlerGroup> groups;
    groups.reserve(active_variants.size());
    for (const auto variant_index : active_variants) {
        if (variant_index >=
            runtime.executable.variants.size()) {
            std::abort();
        }
        const auto key = handler_key(
            runtime.executable.variants[variant_index]);
        const auto group = std::find_if(
            groups.begin(), groups.end(), [key](const auto &candidate) {
                return candidate.key == key;
            });
        if (group == groups.end()) {
            groups.emplace_back(SurfaceValueHandlerGroup{
                .key = key, .variants = {variant_index}});
        } else {
            group->variants.emplace_back(variant_index);
        }
    }
    std::sort(groups.begin(), groups.end(), [](const auto &lhs,
                                                const auto &rhs) {
        return lhs.key < rhs.key;
    });
    return groups;
}

[[nodiscard]] UInt device_handler_key(UInt control) noexcept {
    const auto operation = control & compiler::surface_value_opcode_mask;
    UInt key = control &
               (compiler::surface_value_opcode_mask |
                compiler::surface_value_result_bank_mask);
    const auto image =
        (operation == static_cast<std::uint32_t>(
                          compiler::ValueOperation::image_color)) |
        (operation == static_cast<std::uint32_t>(
                          compiler::ValueOperation::image_alpha));
    const auto immediate =
        (control & compiler::surface_value_svm_immediate_mask) >>
        compiler::surface_value_svm_immediate_shift;
    const auto projection =
        (immediate & compiler::surface_value_image_projection_mask) >>
        compiler::surface_value_image_projection_shift;
    key |= select(
        0u,
        compiler::surface_value_handler_image_box_bit,
        image & (projection == 1u));
    return key;
}

[[nodiscard]] ULong read_unsigned_integer_dynamic(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
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
    const SurfaceValueLocalsView &locals,
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
    const SurfaceValueLocalsView &locals,
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
    const SurfaceValueLocalsView &locals,
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
    // Instruction-owned fields form a product, not an exclusive sum. In
    // particular, Color Ramp owns both an SVM mode immediate and a late-bound
    // ParameterId. Dropping either projection makes two otherwise shareable
    // records observe the first host variant's stale field. Build one context
    // from every independently present record component, then evaluate once.
    std::optional<UInt> immediate;
    if (compiler::surface_value_operation_uses_svm_immediate(
            variant.instruction.operation)) {
        immediate.emplace(
            (instruction.x &
             compiler::surface_value_svm_immediate_mask) >>
            compiler::surface_value_svm_immediate_shift);
    }
    const auto table_parameter =
        variant.instruction.operation ==
            compiler::ValueOperation::color_ramp ||
        variant.instruction.operation ==
            compiler::ValueOperation::rgb_curve;
    const auto static_table = !variant.instruction.static_table.empty();
    std::optional<Expr<std::uint32_t>> parameter_expression;
    if (table_parameter) {
        auto parameter = surface_value_runtime_buffer<luisa::uint>(
                             runtime,
                             SurfaceValueRuntimeBufferSlot::metadata_parameter)
                             .read(instruction.w);
        parameter_expression.emplace(parameter.expression());
    }
    std::optional<ValueStaticTableView> static_table_view;
    if (static_table) {
        auto static_range =
            surface_value_runtime_buffer<luisa::uint2>(
                runtime,
                SurfaceValueRuntimeBufferSlot::metadata_static_range)
                .read(instruction.w);
        static_table_view.emplace(ValueStaticTableView{
            .resources = Expr<BindlessArray>{runtime.device_view},
            .buffer_slot = surface_value_runtime_buffer_slot(
                SurfaceValueRuntimeBufferSlot::static_data),
            .begin = Expr<std::uint32_t>{static_range.x.expression()}});
    }
    ValueEvaluationContext context{
        .services = services,
        .point = point,
        .result = operands,
        .surface = nullptr,
        .parameter_override = parameter_expression
                                  ? &*parameter_expression
                                  : nullptr,
        .static_table_override = static_table_view
                                     ? &*static_table_view
                                     : nullptr,
        .svm_immediate_override = immediate ? &*immediate : nullptr,
        .svm_immediate_domain = immediate
                                    ? std::span<const std::uint16_t>{
                                          variant.svm_immediates}
                                    : std::span<const std::uint16_t>{}};
    return node.evaluate(context);
}

[[nodiscard]] SurfacePoint
automatic_normal_point(UInt program_flags, const SurfacePoint &point) noexcept;

void emit_surface_value_program(const SurfaceValueRuntime &runtime,
                                const SurfaceValueNodes &nodes,
                                std::span<const std::uint32_t> active_variants,
                                const ShaderServices &services,
                                SurfacePoint &point, UInt program,
                                const SurfaceValueLocalsView &locals) noexcept {
    const auto handler_groups = make_handler_groups(runtime, active_variants);
    auto range = surface_value_runtime_buffer<luisa::uint4>(
                     runtime, SurfaceValueRuntimeBufferSlot::program)
                     .read(program);
    const auto program_flags =
        surface_value_runtime_buffer<luisa::uint>(
            runtime, SurfaceValueRuntimeBufferSlot::program_flag)
            .read(program);
    auto transaction_point = automatic_normal_point(program_flags, point);
    UInt instruction_index = range.x;
    const auto instruction_end = range.x + range.y;
    $while(instruction_index < instruction_end) {
        Var<luisa::uint4> instruction =
            surface_value_runtime_buffer<luisa::uint4>(
                runtime, SurfaceValueRuntimeBufferSlot::instruction)
                .read(instruction_index);
        $if(instruction.x ==
            compiler::surface_value_surface_normal_transition_control) {
            const auto normal = read_vector_dynamic(services, transaction_point,
                                                    locals, instruction.y);
            // The automatic-normal prefix may temporarily substitute only
            // these Cycles ShaderData fields. Commit consumes its output
            // before restoring the endpoint root's geometric state.
            transaction_point.position = point.position;
            transaction_point.object_position = point.object_position;
            transaction_point.shading_normal = normal;
            transaction_point.object_shading_normal =
                point.object_shading_normal;
            transaction_point.dPdx = point.dPdx;
            transaction_point.dPdy = point.dPdy;
            transaction_point.object_dPdx = point.object_dPdx;
            transaction_point.object_dPdy = point.object_dPdy;
        }
        $else {
            const auto emit_variant = [&](std::uint32_t index) noexcept {
                if (index >= nodes.size() ||
                    index >= runtime.executable.variants.size()) {
                    std::abort();
                }
                const auto &variant =
                    runtime.executable.variants[index];
                auto operands = load_variant_operands(
                    variant, runtime, services, transaction_point, locals,
                    instruction);
                auto value = evaluate_non_bump_variant(
                    variant, *nodes[index], runtime, services,
                    transaction_point, operands, instruction);
                write_dynamic_value(variant.instruction.result_type, locals,
                                    instruction.y, value);
            };
            const auto primary_key = device_handler_key(instruction.x);
            luisa::compute::detail::SwitchStmtBuilder{primary_key} % [&] {
                for (const auto &group : handler_groups) {
                    luisa::compute::detail::SwitchCaseStmtBuilder{group.key} %
                        [&] {
                            if (group.variants.size() == 1u) {
                                emit_variant(group.variants.front());
                                return;
                            }
                            // The primary key is only an instruction-local
                            // projection. If two exact evaluator families
                            // inhabit one fiber, preserve the complete compiler
                            // discriminator inside that one opcode branch. No
                            // instruction outside an ambiguous fiber pays this
                            // immutable-buffer read.
                            const auto variant_index =
                                surface_value_runtime_buffer<luisa::uint>(
                                    runtime, SurfaceValueRuntimeBufferSlot::
                                                 instruction_variant)
                                    .read(instruction_index);
                            luisa::compute::detail::SwitchStmtBuilder{
                                variant_index} %
                                [&] {
                                    for (const auto index : group.variants) {
                                        luisa::compute::detail::
                                                SwitchCaseStmtBuilder{index} %
                                            [&, index] { emit_variant(index); };
                                    }
                                    luisa::compute::detail::
                                            SwitchDefaultStmtBuilder{} %
                                        [] {
                                            luisa::compute::dsl::unreachable(
                                                "invalid compact surface "
                                                "evaluator "
                                                "fiber");
                                        };
                                };
                        };
                }
                luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                    luisa::compute::dsl::unreachable(
                        "invalid compact surface value handler");
                };
            };
        };
        instruction_index += 1u;
    };
    point.shading_normal = transaction_point.shading_normal;
}

[[nodiscard]] SurfacePoint
automatic_normal_point(UInt program_flags, const SurfacePoint &point) noexcept {
    auto result = point;
    const auto use_undisplaced =
        program_flags &
        compiler::
            surface_value_program_automatic_normal_uses_undisplaced_geometry;
    $if(use_undisplaced != 0u) {
        result.position = point.undisplaced_position;
        result.object_position = point.undisplaced_object_position;
        result.shading_normal = point.undisplaced_shading_normal;
        result.object_shading_normal = point.undisplaced_object_shading_normal;
        result.dPdx = point.undisplaced_dPdx;
        result.dPdy = point.undisplaced_dPdy;
        result.object_dPdx = point.undisplaced_object_dPdx;
        result.object_dPdy = point.undisplaced_object_dPdy;
    };
    return result;
}

[[nodiscard]] std::shared_ptr<SurfaceValueNodes>
make_surface_value_nodes(const SurfaceValueRuntime &runtime) noexcept {
    auto nodes = std::make_shared<SurfaceValueNodes>();
    nodes->reserve(runtime.executable.variants.size());
    for (const auto &variant : runtime.executable.variants) {
        nodes->emplace_back(make_value_node(variant.instruction));
    }
    return nodes;
}

[[nodiscard]] SurfaceValueProgramCallable
make_surface_value_program_callable_impl(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueProgramDomain domain) noexcept {
    const auto domain_view =
        surface_value_program_domain(*scene->surface_values, domain);
    std::vector<std::uint32_t> value_variants{
        domain_view.value_variants.begin(), domain_view.value_variants.end()};
    const auto program_offset = domain_view.program_offset;

    SurfaceValueProgramCallable value_program =
        [scene, nodes, texture_sampling, attribute_lookup,
         value_variants = std::move(value_variants), program_offset](
            BufferFloat scalar_parameters, BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables, BindlessVar textures,
            BindlessVar geometry_heap, UInt surface_tag,
            Var<SurfacePointCall> packed_point,
            Var<SurfaceValueScalarBank> &scalar_bank,
            Var<SurfaceValueVectorBank> &vector_bank,
            ULong &unsigned_integer_bank) noexcept {
            CallableTexture2DSamplingProvider texture_provider{
                textures, texture_sampling};
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
                                          nullptr,
                                          &texture_provider,
                                          &attribute_provider};
            SurfaceValueLocalsView locals{
                .scalars = {luisa::compute::detail::Ref<SurfaceValueScalarBank>{
                    scalar_bank.expression()}},
                .vectors = {luisa::compute::detail::Ref<SurfaceValueVectorBank>{
                    vector_bank.expression()}},
                .unsigned_integers = {luisa::compute::detail::Ref<luisa::ulong>{
                    unsigned_integer_bank.expression()}}};
            auto point = unpack_surface_point(packed_point);
            const auto program =
                surface_tag * SurfaceValueRuntime::programs_per_topology +
                program_offset;
            emit_surface_value_program(
                *scene->surface_values, *nodes, value_variants, services, point,
                program, locals);
            return pack_surface_point(point);
        };
    switch (domain) {
    case SurfaceValueProgramDomain::preparation:
        value_program.set_name("surface_value_program_preparation");
        break;
    case SurfaceValueProgramDomain::emission:
        value_program.set_name("surface_value_program_emission");
        break;
    case SurfaceValueProgramDomain::bssrdf:
        value_program.set_name("surface_value_program_bssrdf");
        break;
    }
    return value_program;
}

} // namespace

SurfaceValueProgramCallable make_surface_value_program_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueProgramDomain domain) noexcept {
    auto nodes = make_surface_value_nodes(*scene->surface_values);
    return make_surface_value_program_callable_impl(
        scene, nodes, texture_sampling, attribute_lookup, domain);
}

} // namespace psycles::luisa_backend::detail

#include "path_tracer_surface_value_program.h"

#include "graph_surface_internal.h"
#include "path_tracer_ambient_occlusion.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surface_value_family.h"
#include "path_tracer_surfaces.h"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <luisa/core/stl/format.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

SurfaceValueLocals::SurfaceValueLocals(std::uint32_t capacity) noexcept
    : stack{capacity} {
    if (std::find(surface_value_stack_lane_buckets.begin(),
                  surface_value_stack_lane_buckets.end(),
                  capacity) == surface_value_stack_lane_buckets.end()) {
        std::abort();
    }
    // The host bytecode verifier proves read-before-write for every legal
    // operand. This one aggregate must-definition starts a fresh device-side
    // lifetime wherever the interpreter stack is instantiated. Encoding the
    // invariant in construction prevents a caller from accidentally exposing
    // the undefined root-scope contents to coroutine liveness.
    auto builder = luisa::compute::detail::FunctionBuilder::current();
    const auto undefined = builder->call(
        stack.type(), luisa::compute::CallOp::UNDEFINED, {});
    builder->assign(stack.expression(), undefined);
}

SurfaceValueLocalsView SurfaceValueLocals::view() const noexcept {
    return SurfaceValueLocalsView{stack.expression()};
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

ULong read_unsigned_integer_dynamic(const ShaderServices &services,
                                    const SurfacePoint &point,
                                    const SurfaceValueLocalsView &locals,
                                    UInt address) noexcept {
    ULong result = 0ull;
    $if((address & compiler::SurfaceValueAddress::parameter_bit) != 0u) {
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

namespace {

using SurfaceValueNodes =
    std::vector<std::unique_ptr<ValueNode>>;

[[nodiscard]] SurfaceValueExpression disabled_surface_value(
    const compiler::ValueInstruction &instruction) noexcept {
    switch (compiler::cycles_node_feature_semantics(instruction.operation)
                .disabled_value) {
    case compiler::CyclesNodeDisabledValue::zero:
      return SurfaceValueExpression::zero(instruction.result_type);
    case compiler::CyclesNodeDisabledValue::one:
      if (surface_value_category(instruction.result_type) !=
          SurfaceValueCategory::scalar) {
        std::abort();
      }
      return SurfaceValueExpression::from_scalar(1.0f);
    case compiler::CyclesNodeDisabledValue::evaluate:
      std::abort();
    }
    std::abort();
}

// One callable owns one Cycles-aligned SVM execution family. Semantic
// operations within that family are runtime subtypes in the instruction, not
// separate callable identities. LLVM remains free to inline profitable
// handlers; no inline/noinline policy is encoded here.
template<std::size_t StackCapacity>
using SurfaceValueHandlerCallable = Callable<void(
    Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
    BindlessArray, SurfacePointCall &, luisa::float3, bool, luisa::uint4,
    SurfaceValueStackBankFor<StackCapacity> &)>;

template<std::size_t StackCapacity>
using SurfaceValueHandlers = std::vector<std::optional<
    SurfaceValueHandlerCallable<StackCapacity>>>;

template<std::size_t StackCapacity>
using SurfaceValueAmbientOcclusionHandlerCallable = Callable<void(
    Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
    BindlessArray, SurfacePointCall &, luisa::float3, bool, luisa::uint4,
    SurfaceValueStackBankFor<StackCapacity> &, Buffer<luisa::float4>,
    luisa::uint4, luisa::uint2)>;

template<std::size_t StackCapacity>
using SurfaceValueAmbientOcclusionHandlers = std::vector<std::optional<
    SurfaceValueAmbientOcclusionHandlerCallable<StackCapacity>>>;

struct SurfaceValueHandlerGroup {
    std::uint32_t key{};
    std::vector<std::uint32_t> variants;
};

inline constexpr SurfaceValueBytecodeSlots svm_value_bytecode_slots{
    .operand = SurfaceValueRuntimeBufferSlot::svm_value_operand,
    .metadata_static_u0 = SurfaceValueRuntimeBufferSlot::svm_metadata_static_u0,
    .metadata_parameter = SurfaceValueRuntimeBufferSlot::svm_metadata_parameter,
    .metadata_static_range =
        SurfaceValueRuntimeBufferSlot::svm_metadata_static_range,
    .static_data = SurfaceValueRuntimeBufferSlot::svm_static_data};

[[nodiscard]] bool surface_value_variant_is_external_query(
    const SurfaceValueRuntime &runtime,
    std::uint32_t variant_index) noexcept {
    if (variant_index >= runtime.value_variants.size()) {
        std::abort();
    }
    return compiler::surface_value_operation_is_external_query(
        runtime.value_variants[variant_index].instruction.operation);
}

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

[[nodiscard]] std::uint32_t family_subtype_key(
    const compiler::SurfaceValueStaticVariant &variant) noexcept {
    compiler::SurfaceValueBank result_bank{};
    if (!compiler::classify_surface_value_type(
            variant.instruction.result_type, result_bank)) {
        std::abort();
    }
    return compiler::make_surface_value_family_subtype_key(
        variant.instruction.operation, result_bank);
}

[[nodiscard]] std::vector<SurfaceValueHandlerGroup>
make_handler_groups(
    const SurfaceValueRuntime &runtime,
    std::span<const std::uint32_t> active_variants) noexcept {
    std::vector<SurfaceValueHandlerGroup> groups;
    groups.reserve(active_variants.size());
    for (const auto variant_index : active_variants) {
        if (variant_index >= runtime.value_variants.size()) {
            std::abort();
        }
        const auto key = handler_key(
            runtime.value_variants[variant_index]);
        const auto group = std::find_if(
            groups.begin(), groups.end(), [key](const auto &candidate) {
                return candidate.key == key;
            });
        if (group == groups.end()) {
            groups.emplace_back(SurfaceValueHandlerGroup{
                .key = key, .variants = {variant_index}});
        } else {
            const auto candidate_subtype = family_subtype_key(
                runtime.value_variants[variant_index]);
            const auto duplicate = std::find_if(
                group->variants.begin(), group->variants.end(),
                [&](std::uint32_t existing) noexcept {
                    return family_subtype_key(
                               runtime.value_variants[existing]) ==
                           candidate_subtype;
                });
            if (duplicate != group->variants.end()) {
                // A semantic subtype must name one fixed typed ABI within its
                // family. Reaching two host provenance variants for one
                // subtype means the bytecode omitted an execution-shape
                // discriminator.
                const auto &existing = runtime.value_variants[*duplicate];
                const auto &candidate =
                    runtime.value_variants[variant_index];
                LUISA_ERROR_WITH_LOCATION(
                    "Surface SVM family {} semantic subtype {} has multiple "
                    "typed ABIs: variants {} and {}, result types {} and {}, "
                    "operand counts {} and {}, evaluator fields ({}, {}) and "
                    "({}, {}).",
                    key, candidate_subtype, *duplicate,
                    variant_index,
                    static_cast<std::uint32_t>(
                        existing.instruction.result_type),
                    static_cast<std::uint32_t>(
                        candidate.instruction.result_type),
                    existing.operand_types.size(),
                    candidate.operand_types.size(),
                    existing.instruction.static_u0,
                    existing.instruction.static_u1,
                    candidate.instruction.static_u0,
                    candidate.instruction.static_u1);
            }
            group->variants.emplace_back(variant_index);
        }
    }
    std::sort(groups.begin(), groups.end(), [](const auto &lhs,
                                                const auto &rhs) {
        return lhs.key < rhs.key;
    });
    for (auto &group : groups) {
        std::sort(group.variants.begin(), group.variants.end(),
                  [&](std::uint32_t lhs, std::uint32_t rhs) noexcept {
                      return family_subtype_key(
                                 runtime.value_variants[lhs]) <
                             family_subtype_key(
                                 runtime.value_variants[rhs]);
                  });
    }
    return groups;
}

[[nodiscard]] UInt device_handler_key(UInt control) noexcept {
    return control & compiler::surface_value_opcode_mask;
}

[[nodiscard]] Float read_scalar_routed(
    compiler::SurfaceValueOperandRoute route,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt address) noexcept {
    switch (route) {
        case compiler::SurfaceValueOperandRoute::local:
            return locals.scalars.read(
                address & compiler::SurfaceValueAddress::index_mask);
        case compiler::SurfaceValueOperandRoute::parameter:
            return services.parameter_float(
                point.parameter_block,
                address & compiler::SurfaceValueAddress::index_mask);
        case compiler::SurfaceValueOperandRoute::dynamic:
            return read_scalar_dynamic(
                services, point, locals, std::move(address));
    }
    std::abort();
}

[[nodiscard]] Float3 read_vector_routed(
    compiler::SurfaceValueOperandRoute route,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt address) noexcept {
    switch (route) {
        case compiler::SurfaceValueOperandRoute::local:
            return locals.vectors.read(
                address & compiler::SurfaceValueAddress::index_mask);
        case compiler::SurfaceValueOperandRoute::parameter:
            return services.parameter_float3(
                point.parameter_block,
                address & compiler::SurfaceValueAddress::index_mask);
        case compiler::SurfaceValueOperandRoute::dynamic:
            return read_vector_dynamic(
                services, point, locals, std::move(address));
    }
    std::abort();
}

[[nodiscard]] ULong read_unsigned_integer_routed(
    compiler::SurfaceValueOperandRoute route,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt address) noexcept {
    switch (route) {
        case compiler::SurfaceValueOperandRoute::local:
            return locals.unsigned_integers.read(
                address & compiler::SurfaceValueAddress::index_mask);
        case compiler::SurfaceValueOperandRoute::parameter:
            return services.parameter_uint64(
                point.parameter_block,
                address & compiler::SurfaceValueAddress::index_mask);
        case compiler::SurfaceValueOperandRoute::dynamic:
            return read_unsigned_integer_dynamic(
                services, point, locals, std::move(address));
    }
    std::abort();
}

[[nodiscard]] SurfaceValueExpression read_routed_value(
    contract::SocketType type,
    compiler::SurfaceValueOperandRoute route,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    UInt address) noexcept {
    switch (surface_value_category(type)) {
        case SurfaceValueCategory::scalar: {
            const auto value = read_scalar_routed(
                route, services, point, locals, address);
            return SurfaceValueExpression::from_scalar(
                Expr<float>{value.expression()});
        }
        case SurfaceValueCategory::vector: {
            const auto value = read_vector_routed(
                route, services, point, locals, address);
            return SurfaceValueExpression::from_vector(
                Expr<luisa::float3>{value.expression()});
        }
        case SurfaceValueCategory::unsigned_integer: {
            const auto value = read_unsigned_integer_routed(
                route, services, point, locals, address);
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

[[nodiscard]] UInt expand_compact_surface_value_operand(UInt compact) noexcept {
    return (compact & static_cast<std::uint32_t>(
                          compiler::SurfaceValueOperandAddress::index_mask)) |
           ((compact &
             (static_cast<std::uint32_t>(
                  compiler::SurfaceValueOperandAddress::parameter_bit) |
              static_cast<std::uint32_t>(
                  compiler::SurfaceValueOperandAddress::bank_mask)))
            << (compiler::SurfaceValueAddress::bank_shift -
                compiler::SurfaceValueOperandAddress::bank_shift));
}

[[nodiscard]] TracedValues load_variant_operands(
    const compiler::SurfaceValueStaticVariant &variant,
    const SurfaceValueRuntime &runtime,
    const SurfaceValueBytecodeSlots &slots,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceValueLocalsView &locals,
    Var<luisa::uint4> instruction) noexcept {
    TracedValues operands;
    operands.shading_normal = point.shading_normal;
    if (variant.operand_routes.size() !=
        variant.operand_types.size()) {
        std::abort();
    }
    operands.values.reserve(variant.operand_types.size());
    const auto inline_operands =
        variant.operand_types.size() <=
        compiler::surface_value_inline_operand_capacity;
    for (auto word_index = std::size_t{0u};
         word_index < compiler::surface_value_operand_word_count(
                          variant.operand_types.size());
         ++word_index) {
        // Arity is an opcode/variant invariant, so this host branch does not
        // enter the shader AST. Small nodes consume the instruction's third
        // word directly; larger nodes perform one global read for each pair
        // of addresses in the packed overflow stream.
        auto word = inline_operands
                        ? UInt{instruction.z.expression()}
                        : surface_value_runtime_buffer<luisa::uint>(
                              runtime, slots.operand)
                              .read(instruction.z +
                                    static_cast<std::uint32_t>(word_index));
        for (auto lane = std::size_t{0u};
             lane < compiler::surface_value_operands_per_word;
             ++lane) {
            const auto operand_index =
                word_index * compiler::surface_value_operands_per_word + lane;
            if (operand_index >= variant.operand_types.size()) {
                break;
            }
            auto compact =
                (word >> static_cast<std::uint32_t>(
                             compiler::surface_value_operand_lane_bits * lane)) &
                0xffffu;
            auto address =
                expand_compact_surface_value_operand(std::move(compact));
            operands.values.emplace_back(read_routed_value(
                variant.operand_types[operand_index],
                variant.operand_routes[operand_index],
                services,
                point,
                locals,
                std::move(address)));
        }
    }
    return operands;
}

[[nodiscard]] SurfaceValueExpression evaluate_non_bump_variant(
    const compiler::SurfaceValueStaticVariant &variant,
    const ValueNode &node,
    const SurfaceValueRuntime &runtime,
    const SurfaceValueBytecodeSlots &slots,
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
    std::optional<UInt> static_u0_expression;
    if (compiler::surface_value_operation_uses_metadata_static_u0(
            variant.instruction.operation)) {
      static_u0_expression.emplace(surface_value_runtime_buffer<luisa::uint>(
                                       runtime, slots.metadata_static_u0)
                                       .read(instruction.w));
    }
    if (table_parameter) {
        auto parameter = surface_value_runtime_buffer<luisa::uint>(
                             runtime, slots.metadata_parameter)
                             .read(instruction.w);
        parameter_expression.emplace(parameter.expression());
    }
    std::optional<ValueStaticTableView> static_table_view;
    if (static_table) {
        auto static_range =
            surface_value_runtime_buffer<luisa::uint2>(
                runtime, slots.metadata_static_range)
                .read(instruction.w);
        static_table_view.emplace(ValueStaticTableView{
            .resources = Expr<BindlessArray>{runtime.device_view},
            .buffer_slot = surface_value_runtime_buffer_slot(
                slots.static_data),
            .begin = Expr<std::uint32_t>{static_range.x.expression()}});
    }
    ValueEvaluationContext context{
        .services = services,
        .point = point,
        .result = operands,
        .surface = nullptr,
        .parameter_override =
            parameter_expression ? &*parameter_expression : nullptr,
        .static_table_override =
            static_table_view ? &*static_table_view : nullptr,
        .static_u0_override =
            static_u0_expression ? &*static_u0_expression : nullptr,
        .svm_immediate_override = immediate ? &*immediate : nullptr,
        .svm_immediate_domain =
            immediate ? std::span<const std::uint16_t>{variant.svm_immediates}
                      : std::span<const std::uint16_t>{}};
    return node.evaluate(context);
}

[[nodiscard]] SurfacePoint surface_value_point(
    const Var<SurfacePointCall> &packed_base_point,
    Float3 transaction_shading_normal,
    Bool use_undisplaced_geometry) noexcept {
    const auto base = unpack_surface_point(packed_base_point);
    auto point = base;
    // This is the exact field projection of automatic_normal_point. Keeping
    // the phase as one Boolean and reconstructing the projection inside each
    // statically typed handler lets dead-field elimination retain only the
    // SurfacePoint members that the selected node can observe.
    point.position = select(
        base.position, base.undisplaced_position, use_undisplaced_geometry);
    point.object_position = select(
        base.object_position, base.undisplaced_object_position,
        use_undisplaced_geometry);
    point.shading_normal = transaction_shading_normal;
    point.object_shading_normal = select(
        base.object_shading_normal, base.undisplaced_object_shading_normal,
        use_undisplaced_geometry);
    point.dPdx = select(
        base.dPdx, base.undisplaced_dPdx, use_undisplaced_geometry);
    point.dPdy = select(
        base.dPdy, base.undisplaced_dPdy, use_undisplaced_geometry);
    point.object_dPdx = select(
        base.object_dPdx, base.undisplaced_object_dPdx,
        use_undisplaced_geometry);
    point.object_dPdy = select(
        base.object_dPdy, base.undisplaced_object_dPdy,
        use_undisplaced_geometry);
    return point;
}

template<std::size_t StackCapacity>
void emit_surface_value_variant(
    const SurfaceValueRuntime &runtime,
    const SurfaceValueNodes &nodes,
    SurfaceValueBytecodeSlots bytecode_slots,
    const ShaderServices &services,
    const SurfacePoint &point,
    Var<luisa::uint4> instruction,
    Var<SurfaceValueStackBankFor<StackCapacity>> &stack,
    compiler::CyclesNodeFeatureMask node_feature_mask,
    std::uint32_t variant_index) noexcept {
    if (variant_index >= runtime.value_variants.size() ||
        variant_index >= nodes.size()) {
      std::abort();
    }
    SurfaceValueLocalsView locals{stack.expression()};
    const auto &variant = runtime.value_variants[variant_index];
    if (!compiler::cycles_node_features_enable(
            variant.instruction.operation, node_feature_mask)) {
        write_dynamic_value(
            variant.instruction.result_type,
            locals,
            instruction.y,
            disabled_surface_value(variant.instruction));
        return;
    }
    if (emit_direct_surface_value_variant(runtime, bytecode_slots, services,
                                          point, locals, instruction,
                                          variant)) {
        return;
    }
    auto operands = load_variant_operands(
        variant, runtime, bytecode_slots, services, point, locals,
        instruction);
    const auto value = evaluate_non_bump_variant(
        variant, *nodes[variant_index], runtime, bytecode_slots, services,
        point, operands, instruction);
    write_dynamic_value(
        variant.instruction.result_type, locals, instruction.y, value);
}

template<std::size_t StackCapacity>
[[nodiscard]] SurfaceValueHandlerCallable<StackCapacity>
make_surface_value_handler_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueBytecodeSlots bytecode_slots,
    const SurfaceValueHandlerGroup &group,
    compiler::CyclesNodeFeatureMask node_feature_mask) noexcept {
    if (!scene->surface_values || group.variants.empty()) {
        std::abort();
    }
    const auto *runtime = scene->surface_values.get();
    for (const auto variant : group.variants) {
        if (variant >= runtime->value_variants.size() ||
            variant >= nodes->size() ||
            handler_key(runtime->value_variants[variant]) != group.key) {
        std::abort();
      }
    }
    SurfaceValueHandlerCallable<StackCapacity> handler =
        [scene, nodes, texture_sampling, attribute_lookup, bytecode_slots,
         group, runtime, node_feature_mask](BufferFloat scalar_parameters,
                  BufferFloat3 vector_parameters,
                  BufferFloat cycles_bsdf_tables,
                  BindlessVar textures,
                  BindlessVar geometry_heap,
                  Var<SurfacePointCall> &packed_base_point,
                  Float3 transaction_shading_normal,
                  Bool use_undisplaced_geometry,
                  Var<luisa::uint4> instruction,
                  Var<SurfaceValueStackBankFor<StackCapacity>> &stack) noexcept {
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
            const auto point = surface_value_point(
                packed_base_point,
                transaction_shading_normal,
                use_undisplaced_geometry);
            const auto emit_variant = [&](std::uint32_t variant) noexcept {
              emit_surface_value_variant<StackCapacity>(
                  *runtime, *nodes, bytecode_slots, services, point,
                  instruction, stack, node_feature_mask, variant);
            };
            if (group.variants.size() == 1u) {
              emit_variant(group.variants.front());
            } else {
              const auto subtype =
                  instruction.x & compiler::surface_value_family_subtype_mask;
              luisa::compute::detail::SwitchStmtBuilder{subtype} % [&] {
                for (const auto variant : group.variants) {
                  const auto semantic = family_subtype_key(
                      runtime->value_variants[variant]);
                  luisa::compute::detail::SwitchCaseStmtBuilder{semantic} %
                      [&] { emit_variant(variant); };
                }
                luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                  luisa::compute::dsl::unreachable(
                      "invalid surface SVM family subtype");
                };
              };
            }
        };
    handler.set_name(luisa::format(
        "surface_value_family_{}", group.key));
    return handler;
}

template<std::size_t StackCapacity>
[[nodiscard]] SurfaceValueAmbientOcclusionHandlerCallable<StackCapacity>
make_surface_ambient_occlusion_handler_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    const std::shared_ptr<const SceneTraversalComponent> &traversal,
    SurfaceValueBytecodeSlots bytecode_slots,
    std::uint32_t variant_index) noexcept {
    if (!scene->surface_values || !traversal ||
        variant_index >= scene->surface_values->value_variants.size() ||
        variant_index >= nodes->size() ||
        !surface_value_variant_is_external_query(
            *scene->surface_values, variant_index)) {
        std::abort();
    }
    const auto *runtime = scene->surface_values.get();
    const auto operation =
        runtime->value_variants[variant_index].instruction.operation;
    SurfaceValueAmbientOcclusionHandlerCallable<StackCapacity> handler =
        [scene, nodes, texture_sampling, attribute_lookup, traversal,
         bytecode_slots, variant_index, runtime](
            BufferFloat scalar_parameters,
            BufferFloat3 vector_parameters,
            BufferFloat cycles_bsdf_tables,
            BindlessVar textures,
            BindlessVar geometry_heap,
            Var<SurfacePointCall> &packed_base_point,
            Float3 transaction_shading_normal,
            Bool use_undisplaced_geometry,
            Var<luisa::uint4> instruction,
            Var<SurfaceValueStackBankFor<StackCapacity>> &stack,
            BufferFloat4 sobol_table,
            Var<luisa::uint4> random_state,
            UInt2 source) noexcept {
            CallableTexture2DSamplingProvider texture_provider{
                textures, texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            const PathSurfaceAmbientOcclusionContext ambient_occlusion{
                .sobol_table = sobol_table,
                .sobol_sequence_size = random_state.x,
                .sample_index = random_state.y,
                .rng_hash = random_state.z,
                .rng_offset = random_state.w,
                .source_object = source.x,
                .source_primitive = source.y};
            PathSurfaceAmbientOcclusionProvider ambient_occlusion_provider{
                scene, traversal, ambient_occlusion};
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
                &attribute_provider,
                &ambient_occlusion_provider};
            SurfaceValueLocalsView locals{stack.expression()};
            const auto &variant =
                runtime->value_variants[variant_index];
            const auto point = surface_value_point(
                packed_base_point,
                transaction_shading_normal,
                use_undisplaced_geometry);
            auto operands = load_variant_operands(
                variant, *runtime, bytecode_slots, services, point, locals,
                instruction);
            const auto value = evaluate_non_bump_variant(
                variant, *(*nodes)[variant_index], *runtime, bytecode_slots,
                services, point, operands, instruction);
            write_dynamic_value(
                variant.instruction.result_type, locals, instruction.y,
                value);
        };
    handler.set_name(luisa::format(
        "surface_value_handler_{}_{}_ao",
        static_cast<std::uint32_t>(operation), variant_index));
    return handler;
}

template<std::size_t StackCapacity>
[[nodiscard]] SurfaceValueHandlers<StackCapacity>
make_surface_value_handlers(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    std::span<const SurfaceValueHandlerGroup> groups,
    SurfaceValueBytecodeSlots bytecode_slots,
    bool include_external_queries,
    compiler::CyclesNodeFeatureMask node_feature_mask) noexcept {
    SurfaceValueHandlers<StackCapacity> handlers(groups.size());
    for (auto group_index = std::size_t{}; group_index < groups.size();
         ++group_index) {
        const auto &group = groups[group_index];
        const auto external = std::any_of(
            group.variants.begin(), group.variants.end(),
            [&](std::uint32_t variant) noexcept {
              return surface_value_variant_is_external_query(
                  *scene->surface_values, variant);
            });
        if (external &&
            !include_external_queries) {
            continue;
        }
        if (handlers[group_index].has_value()) {
            std::abort();
        }
        handlers[group_index].emplace(
            make_surface_value_handler_callable<StackCapacity>(
                scene, nodes, texture_sampling, attribute_lookup,
                bytecode_slots, group, node_feature_mask));
    }
    return handlers;
}

template<std::size_t StackCapacity>
[[nodiscard]] SurfaceValueAmbientOcclusionHandlers<StackCapacity>
make_surface_ambient_occlusion_handlers(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    std::span<const SurfaceValueHandlerGroup> groups,
    SurfaceValueBytecodeSlots bytecode_slots) noexcept {
    SurfaceValueAmbientOcclusionHandlers<StackCapacity> handlers(groups.size());
    const auto traversal = make_scene_traversal_component(
        make_scene_traversal_stage_plan(
            scene->geometries.size(), scene->curve_geometries.size()));
    for (auto group_index = std::size_t{}; group_index < groups.size();
         ++group_index) {
        const auto &group = groups[group_index];
        const auto external = std::count_if(
            group.variants.begin(), group.variants.end(),
            [&](std::uint32_t variant) noexcept {
              return surface_value_variant_is_external_query(
                  *scene->surface_values, variant);
            });
        if (external == 0) {
            continue;
        }
        if (static_cast<std::size_t>(external) != group.variants.size() ||
            group.variants.size() != 1u ||
            handlers[group_index].has_value()) {
            std::abort();
        }
        handlers[group_index].emplace(
            make_surface_ambient_occlusion_handler_callable<StackCapacity>(
                scene, nodes, texture_sampling, attribute_lookup, traversal,
                bytecode_slots, group.variants.front()));
    }
    return handlers;
}

[[nodiscard]] std::shared_ptr<SurfaceValueNodes>
make_surface_value_nodes(const SurfaceValueRuntime &runtime) noexcept {
    auto nodes = std::make_shared<SurfaceValueNodes>();
    nodes->reserve(runtime.value_variants.size());
    for (const auto &variant : runtime.value_variants) {
        nodes->emplace_back(make_value_node(variant.instruction));
    }
    return nodes;
}

[[nodiscard]] bool surface_value_program_domain_has_external_query(
    const SurfaceValueRuntime &runtime,
    SurfaceValueProgramDomain domain) noexcept {
    const auto view = surface_value_program_domain(runtime, domain);
    return std::any_of(
        view.value_variants.begin(), view.value_variants.end(),
        [&](std::uint32_t variant) noexcept {
            return compiler::cycles_node_features_enable(
                       runtime.value_variants[variant].instruction.operation,
                       view.node_feature_mask) &&
                   surface_value_variant_is_external_query(runtime, variant);
        });
}

} // namespace

struct SurfaceValueInstructionDispatcher::Impl {
    const SurfaceValueRuntime *runtime{};
    bool ambient_occlusion{};
    std::uint32_t stack_capacity{};

    Impl(const SurfaceValueRuntime *runtime, bool ambient_occlusion,
         std::uint32_t stack_capacity) noexcept
        : runtime{runtime}, ambient_occlusion{ambient_occlusion},
          stack_capacity{stack_capacity} {}

    virtual ~Impl() noexcept = default;

    virtual void dispatch(
        Expr<Buffer<float>> scalar_parameters,
        Expr<Buffer<luisa::float3>> vector_parameters,
        Expr<Buffer<float>> cycles_bsdf_tables, Expr<BindlessArray> textures,
        Expr<BindlessArray> geometry_heap, Var<SurfacePointCall> &point,
        Float3 transaction_shading_normal, Bool use_undisplaced_geometry,
        Var<luisa::uint4> instruction, const SurfaceValueLocalsView &locals,
        const PathSurfaceAmbientOcclusionContext *ambient_occlusion_context)
        const noexcept = 0;
};

namespace {

template<std::size_t StackCapacity>
struct SurfaceValueInstructionDispatcherImpl final
    : SurfaceValueInstructionDispatcher::Impl {
    SurfaceValueHandlers<StackCapacity> handlers;
    SurfaceValueAmbientOcclusionHandlers<StackCapacity>
        ambient_occlusion_handlers;
    std::vector<SurfaceValueHandlerGroup> handler_groups;

    SurfaceValueInstructionDispatcherImpl(
        const SurfaceValueRuntime *runtime,
        SurfaceValueHandlers<StackCapacity> handlers,
        SurfaceValueAmbientOcclusionHandlers<StackCapacity>
            ambient_occlusion_handlers,
        std::vector<SurfaceValueHandlerGroup> handler_groups,
        bool ambient_occlusion) noexcept
        : Impl{runtime, ambient_occlusion,
               static_cast<std::uint32_t>(StackCapacity)},
          handlers(std::move(handlers)),
          ambient_occlusion_handlers(
              std::move(ambient_occlusion_handlers)),
          handler_groups(std::move(handler_groups)) {}

    void dispatch(
        Expr<Buffer<float>> scalar_parameters,
        Expr<Buffer<luisa::float3>> vector_parameters,
        Expr<Buffer<float>> cycles_bsdf_tables, Expr<BindlessArray> textures,
        Expr<BindlessArray> geometry_heap, Var<SurfacePointCall> &point,
        Float3 transaction_shading_normal, Bool use_undisplaced_geometry,
        Var<luisa::uint4> instruction, const SurfaceValueLocalsView &locals,
        const PathSurfaceAmbientOcclusionContext *ambient_occlusion_context)
        const noexcept override {
      using Stack = SurfaceValueStackBankFor<StackCapacity>;
      if (locals.storage_expression() == nullptr ||
          locals.storage_expression()->type() !=
              luisa::compute::Type::of<Stack>()) {
        std::abort();
      }
      auto stack =
          luisa::compute::detail::Ref<Stack>{locals.storage_expression()};
      const auto emit_group =
          [&](std::uint32_t index) noexcept {
            if (index >= handlers.size()) {
              std::abort();
            }
            if (handlers[index].has_value()) {
              (*handlers[index])(scalar_parameters, vector_parameters,
                                 cycles_bsdf_tables, textures, geometry_heap,
                                 point, transaction_shading_normal,
                                 use_undisplaced_geometry, instruction, stack);
              return;
            }
            if (!ambient_occlusion ||
                ambient_occlusion_context == nullptr ||
                index >= ambient_occlusion_handlers.size() ||
                !ambient_occlusion_handlers[index].has_value()) {
              std::abort();
            }
            auto random_state = luisa::compute::make_uint4(
                ambient_occlusion_context->sobol_sequence_size,
                ambient_occlusion_context->sample_index,
                ambient_occlusion_context->rng_hash,
                ambient_occlusion_context->rng_offset);
            const auto source = make_uint2(
                ambient_occlusion_context->source_object,
                ambient_occlusion_context->source_primitive);
            (*ambient_occlusion_handlers[index])(
                scalar_parameters, vector_parameters, cycles_bsdf_tables,
                textures, geometry_heap, point,
                transaction_shading_normal, use_undisplaced_geometry,
                instruction, stack,
                ambient_occlusion_context->sobol_table,
                random_state, source);
          };
      const auto primary_key = device_handler_key(instruction.x);
      luisa::compute::detail::SwitchStmtBuilder{primary_key} %
          [&] {
            for (auto group_index = std::size_t{};
                 group_index < handler_groups.size(); ++group_index) {
              const auto &group = handler_groups[group_index];
              luisa::compute::detail::SwitchCaseStmtBuilder{group.key} %
                  [&] {
                    emit_group(static_cast<std::uint32_t>(group_index));
                  };
            }
            luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                luisa::compute::dsl::unreachable(
                    "invalid unified surface value handler");
            };
          };
    }
};

template<std::size_t StackCapacity>
[[nodiscard]] SurfaceValueInstructionDispatcher
make_surface_value_instruction_dispatcher_with_capacity(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    std::span<const std::uint32_t> variants,
    bool needs_ambient_occlusion,
    compiler::CyclesNodeFeatureMask node_feature_mask) noexcept {
    auto groups = make_handler_groups(*scene->surface_values, variants);
    auto handlers = make_surface_value_handlers<StackCapacity>(
        scene, nodes, texture_sampling, attribute_lookup, groups,
        svm_value_bytecode_slots, !needs_ambient_occlusion,
        node_feature_mask);
    SurfaceValueAmbientOcclusionHandlers<StackCapacity>
        ambient_occlusion_handlers;
    if (needs_ambient_occlusion) {
        ambient_occlusion_handlers =
            make_surface_ambient_occlusion_handlers<StackCapacity>(
                scene, nodes, texture_sampling, attribute_lookup, groups,
                svm_value_bytecode_slots);
    }
    return SurfaceValueInstructionDispatcher{std::make_shared<
        SurfaceValueInstructionDispatcherImpl<StackCapacity>>(
        scene->surface_values.get(), std::move(handlers),
        std::move(ambient_occlusion_handlers), std::move(groups),
        needs_ambient_occlusion)};
}

} // namespace

SurfaceValueInstructionDispatcher::SurfaceValueInstructionDispatcher(
    std::shared_ptr<const Impl> impl) noexcept
    : _impl{std::move(impl)} {
    if (!_impl || _impl->runtime == nullptr) {
        std::abort();
    }
}

bool SurfaceValueInstructionDispatcher::requires_ambient_occlusion()
    const noexcept {
    return _impl->ambient_occlusion;
}

std::uint32_t SurfaceValueInstructionDispatcher::stack_capacity()
    const noexcept {
    return _impl->stack_capacity;
}

void SurfaceValueInstructionDispatcher::operator()(
    Expr<Buffer<float>> scalar_parameters,
    Expr<Buffer<luisa::float3>> vector_parameters,
    Expr<Buffer<float>> cycles_bsdf_tables, Expr<BindlessArray> textures,
    Expr<BindlessArray> geometry_heap, Var<SurfacePointCall> &point,
    Float3 transaction_shading_normal, Bool use_undisplaced_geometry,
    Var<luisa::uint4> instruction, const SurfaceValueLocalsView &locals,
    const PathSurfaceAmbientOcclusionContext *ambient_occlusion)
    const noexcept {
  _impl->dispatch(scalar_parameters, vector_parameters, cycles_bsdf_tables,
                  textures, geometry_heap, point, transaction_shading_normal,
                  use_undisplaced_geometry, instruction, locals,
                  ambient_occlusion);
}

SurfaceValueInstructionDispatcher make_surface_value_instruction_dispatcher(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueProgramDomain domain,
    bool enable_external_queries) noexcept {
    if (!scene || !scene->surface_values) {
        std::abort();
    }
    auto nodes = make_surface_value_nodes(*scene->surface_values);
    const auto domain_view =
        surface_value_program_domain(*scene->surface_values, domain);
    std::vector<std::uint32_t> variants{
        domain_view.value_variants.begin(),
        domain_view.value_variants.end()};
    const auto needs_ambient_occlusion =
        enable_external_queries &&
        surface_value_program_domain_has_external_query(
            *scene->surface_values, domain);
    if (needs_ambient_occlusion &&
        (!scene->has_ambient_occlusion ||
         !scene->ambient_occlusion_distance_buffer)) {
        std::abort();
    }
    const auto stack_capacity = surface_value_stack_storage_lanes(
        scene->surface_values->svm_scene.maximum_stack_lanes);
    switch (stack_capacity) {
        case 32u:
            return make_surface_value_instruction_dispatcher_with_capacity<
                32u>(scene, nodes, texture_sampling, attribute_lookup,
                     variants, needs_ambient_occlusion,
                     domain_view.node_feature_mask);
        case 64u:
            return make_surface_value_instruction_dispatcher_with_capacity<
                64u>(scene, nodes, texture_sampling, attribute_lookup,
                     variants, needs_ambient_occlusion,
                     domain_view.node_feature_mask);
        case 128u:
            return make_surface_value_instruction_dispatcher_with_capacity<
                128u>(scene, nodes, texture_sampling, attribute_lookup,
                      variants, needs_ambient_occlusion,
                      domain_view.node_feature_mask);
        case SurfaceValueRuntime::stack_capacity:
            return make_surface_value_instruction_dispatcher_with_capacity<
                SurfaceValueRuntime::stack_capacity>(
                scene, nodes, texture_sampling, attribute_lookup, variants,
                needs_ambient_occlusion, domain_view.node_feature_mask);
        default: std::abort();
    }
}

} // namespace psycles::luisa_backend::detail

#include "path_tracer_surface_value_program.h"

#include "graph_surface_internal.h"
#include "path_tracer_ambient_occlusion.h"
#include "path_tracer_shader_services.h"
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

SurfaceValueLocalsView SurfaceValueLocals::view() const noexcept {
    const auto storage =
        luisa::compute::detail::Ref<SurfaceValueStackBank>{stack.expression()};
    return {
        .scalars = {storage},
        .vectors = {storage},
        .unsigned_integers = {storage}};
}

void SurfaceValueLocals::define_all() const noexcept {
    // The host compiler proves read-before-write for every legal bytecode
    // operand. These aggregate assignments are therefore lifetime witnesses,
    // not observable values: XIR can move each alloca below shade_surface and
    // native backends can erase the seed instead of materializing 255 zeros.
    auto storage = luisa::compute::detail::Ref<SurfaceValueStackBank>{
        stack.expression()};
    storage = luisa::compute::undefined<SurfaceValueStackBank>();
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

// A compact value instruction has one statically selected semantic handler.
// Keeping that handler as a real Luisa callable exposes the same control/data
// boundary to every device compiler: the interpreter owns program order and
// typed-bank addresses, while a handler owns only the temporaries needed to
// evaluate one instruction. LLVM remains free to inline profitable handlers;
// no inline/noinline policy is encoded here.
using SurfaceValueHandlerCallable = Callable<void(
    Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
    BindlessArray, SurfacePointCall &, luisa::float3, bool, luisa::uint4,
    SurfaceValueStackBank &)>;

using SurfaceValueHandlers =
    std::vector<std::optional<SurfaceValueHandlerCallable>>;

using SurfaceValueAmbientOcclusionHandlerCallable = Callable<void(
    Buffer<float>, Buffer<luisa::float3>, Buffer<float>, BindlessArray,
    BindlessArray, SurfacePointCall &, luisa::float3, bool, luisa::uint4,
    SurfaceValueStackBank &, Buffer<luisa::float4>, luisa::uint4,
    luisa::uint2)>;

using SurfaceValueAmbientOcclusionHandlers =
    std::vector<std::optional<SurfaceValueAmbientOcclusionHandlerCallable>>;

struct SurfaceValueHandlerGroup {
    std::uint32_t key{};
    std::vector<std::uint32_t> variants;
};

struct SurfaceValueBytecodeSlots {
    SurfaceValueRuntimeBufferSlot operand;
    SurfaceValueRuntimeBufferSlot metadata_parameter;
    SurfaceValueRuntimeBufferSlot metadata_static_range;
    SurfaceValueRuntimeBufferSlot static_data;
};

inline constexpr SurfaceValueBytecodeSlots svm_value_bytecode_slots{
    .operand = SurfaceValueRuntimeBufferSlot::svm_value_operand,
    .metadata_parameter =
        SurfaceValueRuntimeBufferSlot::svm_metadata_parameter,
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

[[nodiscard]] SurfaceValueHandlerCallable
make_surface_value_handler_callable(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueBytecodeSlots bytecode_slots,
    std::uint32_t variant_index) noexcept {
    if (!scene->surface_values ||
        variant_index >= scene->surface_values->value_variants.size() ||
        variant_index >= nodes->size()) {
        std::abort();
    }
    const auto *runtime = scene->surface_values.get();
    const auto operation =
        runtime->value_variants[variant_index].instruction.operation;
    SurfaceValueHandlerCallable handler =
        [scene, nodes, texture_sampling, attribute_lookup, bytecode_slots,
         variant_index, runtime](BufferFloat scalar_parameters,
                  BufferFloat3 vector_parameters,
                  BufferFloat cycles_bsdf_tables,
                  BindlessVar textures,
                  BindlessVar geometry_heap,
                  Var<SurfacePointCall> &packed_base_point,
                  Float3 transaction_shading_normal,
                  Bool use_undisplaced_geometry,
                  Var<luisa::uint4> instruction,
                  Var<SurfaceValueStackBank> &stack) noexcept {
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
            const auto storage =
                luisa::compute::detail::Ref<SurfaceValueStackBank>{
                    stack.expression()};
            SurfaceValueLocalsView locals{
                .scalars = {storage},
                .vectors = {storage},
                .unsigned_integers = {storage}};
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
        "surface_value_handler_{}_{}",
        static_cast<std::uint32_t>(operation), variant_index));
    return handler;
}

[[nodiscard]] SurfaceValueAmbientOcclusionHandlerCallable
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
    SurfaceValueAmbientOcclusionHandlerCallable handler =
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
            Var<SurfaceValueStackBank> &stack,
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
            const auto storage =
                luisa::compute::detail::Ref<SurfaceValueStackBank>{
                    stack.expression()};
            SurfaceValueLocalsView locals{
                .scalars = {storage},
                .vectors = {storage},
                .unsigned_integers = {storage}};
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

[[nodiscard]] SurfaceValueHandlers make_surface_value_handlers(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    std::span<const std::uint32_t> active_variants,
    SurfaceValueBytecodeSlots bytecode_slots,
    bool include_external_queries) noexcept {
    SurfaceValueHandlers handlers(
        scene->surface_values->value_variants.size());
    for (const auto variant_index : active_variants) {
        if (surface_value_variant_is_external_query(
                *scene->surface_values, variant_index) &&
            !include_external_queries) {
            continue;
        }
        if (variant_index >= handlers.size() ||
            handlers[variant_index].has_value()) {
            std::abort();
        }
        handlers[variant_index].emplace(
            make_surface_value_handler_callable(
                scene, nodes, texture_sampling, attribute_lookup,
                bytecode_slots, variant_index));
    }
    return handlers;
}

[[nodiscard]] SurfaceValueAmbientOcclusionHandlers
make_surface_ambient_occlusion_handlers(
    const std::shared_ptr<LuisaSceneData> &scene,
    const std::shared_ptr<SurfaceValueNodes> &nodes,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    std::span<const std::uint32_t> active_variants,
    SurfaceValueBytecodeSlots bytecode_slots) noexcept {
    SurfaceValueAmbientOcclusionHandlers handlers(
        scene->surface_values->value_variants.size());
    const auto traversal = make_scene_traversal_component(
        make_scene_traversal_stage_plan(
            scene->geometries.size(), scene->curve_geometries.size()));
    for (const auto variant_index : active_variants) {
        if (!surface_value_variant_is_external_query(
                *scene->surface_values, variant_index)) {
            continue;
        }
        if (variant_index >= handlers.size() ||
            handlers[variant_index].has_value()) {
            std::abort();
        }
        handlers[variant_index].emplace(
            make_surface_ambient_occlusion_handler_callable(
                scene, nodes, texture_sampling, attribute_lookup, traversal,
                bytecode_slots, variant_index));
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
            return surface_value_variant_is_external_query(runtime, variant);
        });
}

} // namespace

struct SurfaceValueInstructionDispatcher::Impl {
    const SurfaceValueRuntime *runtime{};
    SurfaceValueHandlers handlers;
    SurfaceValueAmbientOcclusionHandlers ambient_occlusion_handlers;
    std::vector<SurfaceValueHandlerGroup> handler_groups;
    bool ambient_occlusion{};
};

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

void SurfaceValueInstructionDispatcher::operator()(
    Expr<Buffer<float>> scalar_parameters,
    Expr<Buffer<luisa::float3>> vector_parameters,
    Expr<Buffer<float>> cycles_bsdf_tables,
    Expr<BindlessArray> textures,
    Expr<BindlessArray> geometry_heap,
    Var<SurfacePointCall> &point,
    Float3 transaction_shading_normal,
    Bool use_undisplaced_geometry,
    Var<luisa::uint4> instruction,
    UInt instruction_index,
    luisa::compute::detail::Ref<SurfaceValueStackBank> stack,
    const PathSurfaceAmbientOcclusionContext
        *ambient_occlusion) const noexcept {
    const auto &impl = *_impl;
    const auto emit_variant = [&](std::uint32_t index) noexcept {
        if (index >= impl.handlers.size()) {
            std::abort();
        }
        if (impl.handlers[index].has_value()) {
            (*impl.handlers[index])(
                scalar_parameters, vector_parameters, cycles_bsdf_tables,
                textures, geometry_heap, point, transaction_shading_normal,
                use_undisplaced_geometry, instruction, stack);
            return;
        }
        if (!impl.ambient_occlusion || ambient_occlusion == nullptr ||
            index >= impl.ambient_occlusion_handlers.size() ||
            !impl.ambient_occlusion_handlers[index].has_value()) {
            std::abort();
        }
        auto random_state = luisa::compute::make_uint4(
            ambient_occlusion->sobol_sequence_size,
            ambient_occlusion->sample_index,
            ambient_occlusion->rng_hash,
            ambient_occlusion->rng_offset);
        const auto source = make_uint2(
            ambient_occlusion->source_object,
            ambient_occlusion->source_primitive);
        (*impl.ambient_occlusion_handlers[index])(
            scalar_parameters, vector_parameters, cycles_bsdf_tables,
            textures, geometry_heap, point, transaction_shading_normal,
            use_undisplaced_geometry, instruction, stack,
            ambient_occlusion->sobol_table,
            random_state, source);
    };
    const auto primary_key = device_handler_key(instruction.x);
    luisa::compute::detail::SwitchStmtBuilder{primary_key} % [&] {
        for (const auto &group : impl.handler_groups) {
            luisa::compute::detail::SwitchCaseStmtBuilder{group.key} % [&] {
                if (group.variants.size() == 1u) {
                    emit_variant(group.variants.front());
                    return;
                }
                const auto variant =
                    surface_value_runtime_buffer<luisa::uint>(
                        *impl.runtime,
                        SurfaceValueRuntimeBufferSlot::svm_instruction_variant)
                        .read(instruction_index);
                luisa::compute::detail::SwitchStmtBuilder{variant} % [&] {
                    for (const auto index : group.variants) {
                        luisa::compute::detail::SwitchCaseStmtBuilder{index} %
                            [&, index] { emit_variant(index); };
                    }
                    luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                        luisa::compute::dsl::unreachable(
                            "invalid unified surface evaluator fiber");
                    };
                };
            };
        }
        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
            luisa::compute::dsl::unreachable(
                "invalid unified surface value handler");
        };
    };
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
    auto handlers = make_surface_value_handlers(
        scene, nodes, texture_sampling, attribute_lookup, variants,
        svm_value_bytecode_slots, !needs_ambient_occlusion);
    SurfaceValueAmbientOcclusionHandlers ambient_occlusion_handlers;
    if (needs_ambient_occlusion) {
        ambient_occlusion_handlers = make_surface_ambient_occlusion_handlers(
            scene, nodes, texture_sampling, attribute_lookup, variants,
            svm_value_bytecode_slots);
    }
    auto groups = make_handler_groups(*scene->surface_values, variants);
    return SurfaceValueInstructionDispatcher{std::make_shared<
        SurfaceValueInstructionDispatcher::Impl>(
        SurfaceValueInstructionDispatcher::Impl{
            .runtime = scene->surface_values.get(),
            .handlers = std::move(handlers),
            .ambient_occlusion_handlers =
                std::move(ambient_occlusion_handlers),
            .handler_groups = std::move(groups),
            .ambient_occlusion = needs_ambient_occlusion})};
}

} // namespace psycles::luisa_backend::detail

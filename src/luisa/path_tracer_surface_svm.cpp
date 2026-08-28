#include "path_tracer_surface_svm.h"

#include "path_tracer_surface_closure_decode.h"

#include <algorithm>
#include <cstdlib>
#include <utility>
#include <vector>

#include <luisa/dsl/sugar.h>
#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {
namespace {

struct SurfaceSvmClosureHandlerGroup {
    std::uint32_t static_variant{};
    std::vector<compiler::PrincipledClosureFeatureMask> principled_features;
};

[[nodiscard]] std::span<const SurfaceSvmClosureVariant>
surface_svm_closure_variants(
    const SurfaceValueRuntime &runtime,
    SurfaceValueProgramDomain domain) noexcept {
    switch (domain) {
        case SurfaceValueProgramDomain::preparation:
            return runtime.preparation_svm_closure_variants;
        case SurfaceValueProgramDomain::emission:
            return runtime.emission_svm_closure_variants;
        case SurfaceValueProgramDomain::bssrdf:
            return runtime.bssrdf_svm_closure_variants;
    }
    std::abort();
}

[[nodiscard]] std::vector<SurfaceSvmClosureHandlerGroup>
make_surface_svm_closure_handler_groups(
    std::span<const SurfaceSvmClosureVariant> variants) {
    if (!std::is_sorted(variants.begin(), variants.end()) ||
        std::adjacent_find(variants.begin(), variants.end()) !=
            variants.end()) {
        std::abort();
    }
    std::vector<SurfaceSvmClosureHandlerGroup> groups;
    for (const auto &variant : variants) {
        if (groups.empty() ||
            groups.back().static_variant != variant.static_variant) {
            groups.emplace_back(SurfaceSvmClosureHandlerGroup{
                .static_variant = variant.static_variant});
        }
        groups.back().principled_features.emplace_back(
            variant.principled_features);
    }
    return groups;
}

[[nodiscard]] SurfacePoint evaluated_surface_point(
    const SurfacePoint &base_point,
    Float3 shading_normal) noexcept {
    auto point = base_point;
    point.shading_normal = std::move(shading_normal);
    return point;
}

} // namespace

struct SurfaceSvmInterpreter::Impl {
    std::shared_ptr<LuisaSceneData> scene;
    const SurfaceValueRuntime *runtime{};
    SurfaceValueInstructionDispatcher values;
    std::vector<SurfaceSvmClosureHandlerGroup> closure_groups;
    std::uint32_t program_offset{};
};

SurfaceSvmInterpreter::SurfaceSvmInterpreter(
    std::shared_ptr<const Impl> impl) noexcept
    : _impl{std::move(impl)} {
    if (!_impl || !_impl->scene || _impl->runtime == nullptr) {
        std::abort();
    }
}

bool SurfaceSvmInterpreter::requires_ambient_occlusion() const noexcept {
    return _impl->values.requires_ambient_occlusion();
}

Float3 SurfaceSvmInterpreter::execute(
    const SurfaceSvmExecutionResources &resources,
    const ShaderServices &services,
    UInt surface_tag,
    const SurfacePoint &base_point,
    SurfaceSvmClosureConsumer &consumer,
    const PathSurfaceAmbientOcclusionContext *ambient_occlusion) const noexcept {
    const auto &impl = *_impl;
    const auto &runtime = *impl.runtime;
    Float3 final_shading_normal = base_point.shading_normal;
    $if(surface_tag <
        static_cast<luisa::uint>(runtime.topologies.size())) {
        const auto program =
            surface_tag * SurfaceValueRuntime::programs_per_topology +
            impl.program_offset;
        const auto descriptors =
            surface_value_runtime_buffer<luisa::uint4>(
                runtime, SurfaceValueRuntimeBufferSlot::svm_program);
        const auto descriptor0 = descriptors.read(program * 2u);
        const auto descriptor1 = descriptors.read(program * 2u + 1u);

        SurfaceValueLocals locals;
        const auto locals_view = locals.view();
        auto packed_base_point = pack_surface_point(base_point);
        Bool use_undisplaced_geometry =
            (descriptor1.y &
             compiler::
                 surface_value_program_automatic_normal_uses_undisplaced_geometry) !=
            0u;
        Float3 transaction_shading_normal = select(
            base_point.shading_normal,
            base_point.undisplaced_shading_normal,
            use_undisplaced_geometry);

        const auto read_weight = [&](UInt slot) noexcept {
            const auto stored =
                slot != compiler::surface_svm_root_weight_slot;
            const auto safe_slot = select(0u, slot, stored);
            return select(1.0f, locals_view.scalars.read(safe_slot), stored);
        };
        const auto transaction_point = [&] noexcept {
            return evaluated_surface_point(
                base_point,
                Float3{transaction_shading_normal.expression()});
        };

        UInt instruction_index = descriptor0.x;
        const auto instruction_end = descriptor0.x + descriptor0.y;
        $while(instruction_index < instruction_end) {
            Var<luisa::uint4> instruction =
                surface_value_runtime_buffer<luisa::uint4>(
                    runtime, SurfaceValueRuntimeBufferSlot::svm_instruction)
                    .read(instruction_index);
            const auto opcode =
                instruction.x & compiler::surface_svm_opcode_mask;
            UInt next_instruction = instruction_index + 1u;

            $if(opcode <= static_cast<std::uint32_t>(
                              compiler::ValueOperation::ambient_occlusion)) {
                impl.values(
                    resources.scalar_parameters,
                    resources.vector_parameters,
                    resources.cycles_bsdf_tables,
                    resources.textures,
                    resources.geometry_heap,
                    packed_base_point,
                    transaction_shading_normal,
                    use_undisplaced_geometry,
                    instruction,
                    instruction_index,
                    locals_view.scalars.storage,
                    ambient_occlusion);
            }
            $elif(opcode == compiler::surface_svm_mix_closure_opcode) {
                const auto point = transaction_point();
                const auto factor = clamp(
                    read_scalar_dynamic(
                        services, point, locals_view, instruction.y),
                    0.0f,
                    1.0f);
                const auto parent = read_weight(instruction.z);
                $if((instruction.x &
                     compiler::surface_svm_mix_left_result_bit) != 0u) {
                    const auto slot =
                        instruction.w &
                        compiler::surface_svm_packed_weight_slot_mask;
                    locals_view.scalars.write(
                        slot, parent * (1.0f - factor));
                };
                $if((instruction.x &
                     compiler::surface_svm_mix_right_result_bit) != 0u) {
                    const auto slot = instruction.w >> 16u;
                    locals_view.scalars.write(slot, parent * factor);
                };
            }
            $elif(opcode ==
                  compiler::surface_svm_add_closure_weight_opcode) {
                locals_view.scalars.write(
                    instruction.w,
                    read_weight(instruction.y) +
                        read_weight(instruction.z));
            }
            $elif(opcode == compiler::surface_svm_jump_if_one_opcode) {
                const auto point = transaction_point();
                const auto factor = read_scalar_dynamic(
                    services, point, locals_view, instruction.y);
                next_instruction = select(
                    next_instruction, instruction.z, factor >= 1.0f);
            }
            $elif(opcode == compiler::surface_svm_jump_if_zero_opcode) {
                const auto point = transaction_point();
                const auto factor = read_scalar_dynamic(
                    services, point, locals_view, instruction.y);
                next_instruction = select(
                    next_instruction, instruction.z, factor <= 0.0f);
            }
            $elif(opcode == compiler::surface_svm_closure_leaf_opcode) {
                const auto mix_weight = read_weight(instruction.z);
                // Preserve Cycles and the established interpreter's NaN rule:
                // `!(NaN <= 0)` keeps the leaf reachable.
                $if(!(mix_weight <= 0.0f)) {
                    const auto closure_control =
                        instruction.x >>
                        compiler::surface_svm_closure_control_shift;
                    const auto static_variant =
                        closure_control &
                        compiler::surface_closure_static_variant_mask;
                    const auto endpoints =
                        (closure_control &
                         compiler::surface_closure_endpoint_mask) >>
                        compiler::surface_closure_endpoint_shift;
                    Var<luisa::uint4> closure_instruction = make_uint4(
                        closure_control, instruction.y, 0u, 0u);
                    const auto point = transaction_point();
                    const auto emit =
                        [&](std::uint32_t variant,
                            compiler::PrincipledClosureFeatureMask
                                principled_features) noexcept {
                            const auto closure = decode_surface_closure(
                                variant,
                                principled_features,
                                runtime,
                                SurfaceValueRuntimeBufferSlot::
                                    svm_closure_operand,
                                services,
                                point,
                                locals_view,
                                closure_instruction,
                                mix_weight);
                            consumer.visit(
                                point, closure, endpoints, instruction_index);
                        };
                    luisa::compute::detail::SwitchStmtBuilder{
                        static_variant} % [&] {
                        for (const auto &group : impl.closure_groups) {
                            luisa::compute::detail::SwitchCaseStmtBuilder{
                                group.static_variant} % [&] {
                                if (group.principled_features.size() == 1u) {
                                    emit(group.static_variant,
                                         group.principled_features.front());
                                } else {
                                    luisa::compute::detail::SwitchStmtBuilder{
                                        instruction.w} % [&] {
                                        for (const auto features :
                                             group.principled_features) {
                                            luisa::compute::detail::
                                                SwitchCaseStmtBuilder{
                                                    features} % [&, features] {
                                                emit(group.static_variant,
                                                     features);
                                            };
                                        }
                                        luisa::compute::detail::
                                            SwitchDefaultStmtBuilder{} % [] {
                                            luisa::compute::dsl::unreachable(
                                                "invalid unified Principled "
                                                "feature mask");
                                        };
                                    };
                                }
                            };
                        }
                        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
                            luisa::compute::dsl::unreachable(
                                "invalid unified surface closure variant");
                        };
                    };
                };
            }
            $elif(opcode == compiler::surface_svm_set_normal_opcode) {
                const auto normal = read_vector_dynamic(
                    services, base_point, locals_view, instruction.y);
                use_undisplaced_geometry = false;
                transaction_shading_normal = normal;
                consumer.set_shading_normal(normal);
            }
            $elif(opcode == compiler::surface_svm_end_opcode) {
                next_instruction = instruction_end;
            }
            $else {
                luisa::compute::dsl::unreachable(
                    "invalid unified surface SVM opcode");
            };
            instruction_index = next_instruction;
        };
        final_shading_normal = transaction_shading_normal;
    };
    return final_shading_normal;
}

SurfaceSvmInterpreter make_surface_svm_interpreter(
    const std::shared_ptr<LuisaSceneData> &scene,
    const Texture2DSamplingCallables &texture_sampling,
    const SurfaceAttributeLookupCallable &attribute_lookup,
    SurfaceValueProgramDomain domain,
    bool enable_external_queries) noexcept {
    if (!scene || !scene->surface_values) {
        std::abort();
    }
    const auto domain_view =
        surface_value_program_domain(*scene->surface_values, domain);
    auto values = make_surface_value_instruction_dispatcher(
        scene, texture_sampling, attribute_lookup, domain,
        enable_external_queries);
    auto closure_groups = make_surface_svm_closure_handler_groups(
        surface_svm_closure_variants(*scene->surface_values, domain));
    return SurfaceSvmInterpreter{
        std::make_shared<SurfaceSvmInterpreter::Impl>(
            SurfaceSvmInterpreter::Impl{
                .scene = scene,
                .runtime = scene->surface_values.get(),
                .values = std::move(values),
                .closure_groups = std::move(closure_groups),
                .program_offset = domain_view.program_offset})};
}

} // namespace psycles::luisa_backend::detail

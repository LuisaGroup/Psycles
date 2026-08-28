#include "path_tracer_surface_values.h"
#include "path_tracer_surface_route_policy.h"

#include <psycles/compiler/surface_bump_expansion.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/core/logging.h>

namespace psycles::luisa_backend::detail {
namespace {

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

struct SurfaceSvmRuntimeProgram {
    compiler::SurfaceSvmProgramImage image;
    // Parallel to `image.instructions`. Value records name their immutable
    // source expression; every control and closure record uses the invalid
    // sentinel. This host-only relation is discarded after exact evaluator
    // variants have been assigned to the aggregated scene.
    std::vector<std::uint32_t> instruction_sources;
};

void collect_surface_svm_closure_variants(
    const compiler::SurfaceSvmProgramImage &program,
    std::vector<SurfaceSvmClosureVariant> &variants) {
    for (const auto &instruction : program.instructions) {
        if (compiler::surface_svm_bytecode_kind(instruction) !=
            compiler::SurfaceSvmBytecodeKind::closure_leaf) {
            continue;
        }
        const auto control =
            compiler::surface_svm_closure_control(instruction);
        variants.emplace_back(SurfaceSvmClosureVariant{
            .static_variant =
                control & compiler::surface_closure_static_variant_mask,
            .principled_features = instruction.payload2});
    }
}

void canonicalize_surface_svm_closure_variants(
    std::vector<SurfaceSvmClosureVariant> &variants) {
    std::sort(variants.begin(), variants.end());
    variants.erase(std::unique(variants.begin(), variants.end()),
                   variants.end());
}

[[nodiscard]] SurfaceSvmRuntimeProgram build_surface_svm_runtime_program(
    const compiler::SurfaceProgram &program,
    const compiler::SurfaceClosurePlan &closure_plan,
    const compiler::SurfaceValueDependencyPlan &dependencies,
    compiler::SurfaceClosureEndpointMask endpoints,
    compiler::SurfaceValueStorageCapacity capacity) {
    SurfaceSvmRuntimeProgram result;
    const auto schedule = compiler::plan_surface_svm_schedule(
        program, closure_plan, dependencies, endpoints);
    if (!schedule.valid) {
        result.image.diagnostic = "schedule: " + schedule.diagnostic;
        return result;
    }
    const auto storage =
        compiler::plan_surface_svm_storage(program, schedule, capacity);
    if (!storage.valid) {
        result.image.diagnostic = "storage: " + storage.diagnostic;
        return result;
    }
    result.image = compiler::lower_surface_svm_program(
        program, closure_plan, dependencies, schedule, storage);
    if (!result.image.valid) {
        result.image.diagnostic = "lowering: " + result.image.diagnostic;
        return result;
    }

    result.instruction_sources.reserve(result.image.instructions.size());
    for (const auto &instruction : schedule.instructions) {
        if (instruction.kind ==
            compiler::SurfaceSvmScheduleInstructionKind::value) {
            if (instruction.source >= storage.representatives.size()) {
                result.image.valid = false;
                result.image.diagnostic =
                    "scheduled value has no quotient representative";
                result.instruction_sources.clear();
                return result;
            }
            if (storage.representatives[instruction.source] ==
                instruction.source) {
                result.instruction_sources.emplace_back(instruction.source);
            }
            continue;
        }
        result.instruction_sources.emplace_back(
            compiler::SurfaceValueAddress::invalid_value);
    }
    if (result.instruction_sources.size() != result.image.instructions.size()) {
        result.image.valid = false;
        result.image.diagnostic =
            "source provenance is not parallel to unified bytecode";
        result.instruction_sources.clear();
        return result;
    }
    for (auto index = std::size_t{};
         index < result.image.instructions.size(); ++index) {
        const auto is_value = compiler::surface_svm_bytecode_kind(
                                  result.image.instructions[index]) ==
                              compiler::SurfaceSvmBytecodeKind::value;
        const auto has_source = result.instruction_sources[index] !=
                                compiler::SurfaceValueAddress::invalid_value;
        if (is_value != has_source) {
            result.image.valid = false;
            result.image.diagnostic =
                "source provenance disagrees with unified bytecode kind";
            result.instruction_sources.clear();
            return result;
        }
    }
    return result;
}

[[nodiscard]] SurfaceSvmRuntimeProgram compose_surface_svm_runtime_program(
    const compiler::SurfaceProgram &program,
    const compiler::SurfaceValueStoragePlan *normal_storage,
    compiler::ValueExpressionId normal_output,
    bool uses_undisplaced_geometry,
    SurfaceSvmRuntimeProgram root) {
    if (normal_storage == nullptr) {
        return root;
    }
    if (!normal_storage->compatible(program) || !normal_output.valid()) {
        root.image.valid = false;
        root.image.diagnostic =
            "automatic-normal provenance is incomplete or incompatible";
        root.instruction_sources.clear();
        return root;
    }
    const auto normal =
        compiler::lower_surface_value_program(program, *normal_storage);
    if (!normal.valid || normal_output.value >= normal.value_addresses.size()) {
        root.image.valid = false;
        root.image.diagnostic = "automatic-normal prefix cannot be lowered";
        root.instruction_sources.clear();
        return root;
    }
    const auto encoded_output = normal.value_addresses[normal_output.value];
    if (encoded_output == compiler::SurfaceValueAddress::invalid_value) {
        root.image.valid = false;
        root.image.diagnostic =
            "automatic-normal output has no typed address";
        root.instruction_sources.clear();
        return root;
    }
    auto composed = compiler::compose_surface_svm_normal_transaction(
        normal, encoded_output, root.image, uses_undisplaced_geometry);
    if (!composed.valid) {
        root.image = std::move(composed);
        root.instruction_sources.clear();
        return root;
    }

    std::vector<std::uint32_t> sources;
    sources.reserve(composed.instructions.size());
    for (const auto source : normal_storage->instructions) {
        sources.emplace_back(source.value);
    }
    sources.emplace_back(compiler::SurfaceValueAddress::invalid_value);
    sources.insert(sources.end(), root.instruction_sources.begin(),
                   root.instruction_sources.end());
    if (sources.size() != composed.instructions.size()) {
        composed.valid = false;
        composed.diagnostic =
            "automatic-normal composition changed provenance cardinality";
        sources.clear();
    }
    root.image = std::move(composed);
    root.instruction_sources = std::move(sources);
    return root;
}

[[nodiscard]] bool build_surface_svm_runtime_scene(
    const compiler::SurfaceValueExecutableScene &executable,
    std::span<const compiler::SurfaceValueExecutionInput> inputs,
    std::vector<SurfaceSvmRuntimeProgram> &programs,
    compiler::SurfaceSvmSceneImage &scene,
    std::vector<std::uint32_t> &instruction_variants,
    std::string &diagnostic) {
    const auto &legacy = executable.values;
    if (!executable.valid || inputs.size() != programs.size() ||
        legacy.programs.size() != programs.size() ||
        executable.instruction_variants.size() !=
            legacy.instructions.size()) {
        diagnostic =
            "unified and established runtime programs do not form a "
            "bijection";
        return false;
    }

    instruction_variants.clear();
    instruction_variants.reserve([&] {
        auto count = std::size_t{};
        for (const auto &program : programs) {
            count += program.image.instructions.size();
        }
        return count;
    }());
    for (auto program_index = std::size_t{};
         program_index < programs.size(); ++program_index) {
        const auto &input = inputs[program_index];
        const auto &range = legacy.programs[program_index];
        auto &program = programs[program_index];
        if (input.program == nullptr || input.storage == nullptr ||
            range.instruction_begin > legacy.instructions.size() ||
            range.instruction_count >
                legacy.instructions.size() - range.instruction_begin ||
            program.instruction_sources.size() !=
                program.image.instructions.size()) {
            diagnostic = "surface program " +
                         std::to_string(program_index) +
                         " has an incomplete evaluator provenance relation";
            return false;
        }

        std::vector<std::uint32_t> source_variants(
            input.program->value_instructions().size(),
            compiler::SurfaceValueAddress::invalid_value);
        auto cursor = range.instruction_begin;
        const auto end = range.instruction_begin + range.instruction_count;
        const auto consume = [&](const compiler::SurfaceValueStoragePlan &plan,
                                 std::string_view phase) {
            if (!plan.compatible(*input.program)) {
                diagnostic = "surface program " +
                             std::to_string(program_index) + " " +
                             std::string{phase} +
                             " storage is incompatible";
                return false;
            }
            for (const auto source : plan.instructions) {
                if (!source.valid() ||
                    source.value >= source_variants.size() || cursor >= end ||
                    compiler::is_surface_value_surface_normal_transition(
                        legacy.instructions[cursor])) {
                    diagnostic = "surface program " +
                                 std::to_string(program_index) + " " +
                                 std::string{phase} +
                                 " lost its instruction/source ordering";
                    return false;
                }
                const auto variant =
                    executable.instruction_variants[cursor];
                if (variant >= executable.variants.size() ||
                    compiler::surface_value_operation(
                        legacy.instructions[cursor]) !=
                        input.program->value_instructions()[source.value]
                            .operation) {
                    diagnostic = "surface program " +
                                 std::to_string(program_index) + " " +
                                 std::string{phase} +
                                 " has an invalid evaluator variant";
                    return false;
                }
                auto &assigned = source_variants[source.value];
                if (assigned != compiler::SurfaceValueAddress::invalid_value &&
                    assigned != variant) {
                    diagnostic = "surface program " +
                                 std::to_string(program_index) +
                                 " assigns two semantic variants to one "
                                 "source value";
                    return false;
                }
                assigned = variant;
                ++cursor;
            }
            return true;
        };

        if (input.surface_normal_storage != nullptr) {
            if (!consume(*input.surface_normal_storage, "automatic-normal") ||
                cursor >= end ||
                !compiler::is_surface_value_surface_normal_transition(
                    legacy.instructions[cursor]) ||
                executable.instruction_variants[cursor] !=
                    compiler::SurfaceValueAddress::invalid_value) {
                if (diagnostic.empty()) {
                    diagnostic = "surface program " +
                                 std::to_string(program_index) +
                                 " has no exact automatic-normal boundary";
                }
                return false;
            }
            ++cursor;
        }
        if (!consume(*input.storage, "root") || cursor != end) {
            if (diagnostic.empty()) {
                diagnostic = "surface program " +
                             std::to_string(program_index) +
                             " established stream has an unowned suffix";
            }
            return false;
        }

        for (auto instruction_index = std::size_t{};
             instruction_index < program.image.instructions.size();
             ++instruction_index) {
            const auto &instruction =
                program.image.instructions[instruction_index];
            const auto kind =
                compiler::surface_svm_bytecode_kind(instruction);
            const auto source =
                program.instruction_sources[instruction_index];
            if (kind != compiler::SurfaceSvmBytecodeKind::value) {
                if (source != compiler::SurfaceValueAddress::invalid_value) {
                    diagnostic = "surface program " +
                                 std::to_string(program_index) +
                                 " assigns a value source to a control record";
                    return false;
                }
                instruction_variants.emplace_back(
                    compiler::SurfaceValueAddress::invalid_value);
                continue;
            }
            if (source >= source_variants.size()) {
                diagnostic = "surface program " +
                             std::to_string(program_index) +
                             " has a value record without a source";
                return false;
            }
            const auto variant = source_variants[source];
            if (variant >= executable.variants.size()) {
                diagnostic = "surface program " +
                             std::to_string(program_index) +
                             " has a value record outside the semantic "
                             "variant domain";
                return false;
            }
            const auto value =
                compiler::surface_svm_value_instruction(instruction);
            const auto &static_variant = executable.variants[variant];
            const auto immediate = compiler::surface_value_svm_immediate(value);
            if (compiler::surface_value_operation(value) !=
                    static_variant.instruction.operation ||
                compiler::surface_value_operand_count(value) !=
                    static_variant.operand_types.size() ||
                std::find(static_variant.svm_immediates.begin(),
                          static_variant.svm_immediates.end(),
                          immediate) == static_variant.svm_immediates.end()) {
                diagnostic = "surface program " +
                             std::to_string(program_index) +
                             " value record disagrees with its exact "
                             "evaluator variant";
                return false;
            }
            instruction_variants.emplace_back(variant);
        }
    }

    std::vector<compiler::SurfaceSvmProgramImage> images;
    images.reserve(programs.size());
    for (auto &program : programs) {
        images.emplace_back(std::move(program.image));
    }
    scene = compiler::build_surface_svm_scene_image(images);
    if (!scene.valid) {
        diagnostic = "unified surface SVM scene: " + scene.diagnostic;
        return false;
    }
    if (instruction_variants.size() != scene.instructions.size()) {
        diagnostic =
            "unified evaluator variants are not parallel to the scene";
        return false;
    }
    if (const auto validation =
            compiler::validate_surface_svm_scene_image(scene);
        !validation.empty()) {
        diagnostic = "unified surface SVM scene validation: " + validation;
        return false;
    }
    return true;
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
    std::span<const std::uint32_t> bssrdf_bump_tags,
    std::string &diagnostic,
    std::uint32_t region_handler_site_budget) {
    diagnostic.clear();
    if (programs.empty() || programs.size() != closure_plans.size()) {
        diagnostic = "surface topology programs and closure plans do not form a "
                     "non-empty bijection";
        return nullptr;
    }
    if (programs.size() >
        std::numeric_limits<std::uint32_t>::max() /
            SurfaceValueRuntime::programs_per_topology) {
        diagnostic = "surface topology count exceeds compact device program ids";
        return nullptr;
    }

    std::vector<bool> bssrdf_topologies(programs.size(), false);
    auto bssrdf_topology_count = std::size_t{0u};
    for (const auto tag : bssrdf_bump_tags) {
        if (tag >= programs.size()) {
            diagnostic =
                "BSSRDF-bump topology tag exceeds compact surface programs";
            return nullptr;
        }
        if (!bssrdf_topologies[tag]) {
            bssrdf_topologies[tag] = true;
            ++bssrdf_topology_count;
        }
    }

    // Refine each topology before any dependency or storage analysis. The
    // closure plan remains valid because refinement preserves closure-tree
    // indices and remaps every ValueExpressionId endpoint transactionally.
    // Keeping the expanded programs alive in the runtime also guarantees that
    // static evaluator variants never retain pointers into temporary IR.
    std::vector<std::shared_ptr<const compiler::SurfaceProgram>>
        execution_programs;
    execution_programs.reserve(programs.size());
    for (auto topology = std::size_t{0u}; topology < programs.size();
         ++topology) {
        const auto &source = programs[topology];
        if (!source || !closure_plans[topology].compatible(*source)) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " has no compatible closure plan";
            return nullptr;
        }
        auto expansion = compiler::expand_surface_bump_program(*source);
        if (!expansion.valid || !expansion.program) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " Bump graph expansion: " + expansion.diagnostic;
            return nullptr;
        }
        if (!closure_plans[topology].compatible(*expansion.program)) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " Bump graph expansion changed closure topology";
            return nullptr;
        }
        execution_programs.emplace_back(std::move(expansion.program));
    }

    auto runtime = std::make_unique<SurfaceValueRuntime>();
    runtime->topologies.reserve(programs.size());
    std::vector<compiler::SurfaceValueStoragePlan> normal_storage;
    normal_storage.reserve(programs.size());
    std::vector<compiler::SurfaceValueStoragePlan> root_storage;
    root_storage.reserve(programs.size() *
                         SurfaceValueRuntime::programs_per_topology);
    std::vector<SurfaceSvmRuntimeProgram> svm_programs;
    svm_programs.reserve(programs.size() *
                         SurfaceValueRuntime::programs_per_topology);
    std::vector<bool> automatic_normal_uses_undisplaced_geometry;
    automatic_normal_uses_undisplaced_geometry.reserve(programs.size());
    std::vector<bool> emission_uses_automatic_normal;
    emission_uses_automatic_normal.reserve(programs.size());

    for (auto topology = std::size_t{0u}; topology < programs.size();
         ++topology) {
        const auto &program_ptr = execution_programs[topology];
        const auto &program = *program_ptr;
        auto normal_active =
            dependency_mask(program, program.surface_normal_root());
        auto normal_outputs = std::vector<bool>(normal_active.size(), false);
        if (program.surface_normal_root().valid()) {
            if (program.surface_normal_root().value >= normal_outputs.size()) {
                diagnostic = "surface topology " + std::to_string(topology) +
                             " has an invalid automatic-normal root";
                return nullptr;
            }
            normal_outputs[program.surface_normal_root().value] = true;
        }
        auto topology_normal_storage = compiler::plan_surface_value_storage(
            program, normal_active, normal_outputs,
            SurfaceValueRuntime::storage_capacity);
        if (!topology_normal_storage.valid) {
            diagnostic =
                "surface topology " + std::to_string(topology) +
                " automatic-normal plan: " + topology_normal_storage.diagnostic;
            return nullptr;
        }

        const auto dependencies = compiler::analyze_surface_value_dependencies(
            program, closure_plans[topology]);
        auto preparation_storage = compiler::plan_surface_value_storage(
            program, dependencies.preparation,
            dependencies.preparation_outputs,
            SurfaceValueRuntime::storage_capacity);
        if (!preparation_storage.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " preparation plan: " + preparation_storage.diagnostic;
            return nullptr;
        }
        auto preparation_image =
            compiler::lower_surface_value_program(program, preparation_storage);
        if (!preparation_image.valid) {
            diagnostic =
                "surface topology " + std::to_string(topology) +
                " preparation lowering: " + preparation_image.diagnostic;
            return nullptr;
        }
        auto emission_storage = compiler::plan_surface_value_storage(
            program, dependencies.emission, dependencies.emission_outputs,
            SurfaceValueRuntime::storage_capacity);
        if (!emission_storage.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " emission plan: " + emission_storage.diagnostic;
            return nullptr;
        }
        const auto emission_image =
            compiler::lower_surface_value_program(program, emission_storage);
        if (!emission_image.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " emission lowering: " + emission_image.diagnostic;
            return nullptr;
        }

        auto topology_uses_undisplaced_geometry = false;
        for (auto index = std::size_t{0u}; index < normal_active.size();
             ++index) {
            const auto &instruction = program.value_instructions()[index];
            topology_uses_undisplaced_geometry |=
                normal_active[index] &&
                instruction.operation ==
                    compiler::ValueOperation::bump_samples &&
                (instruction.static_u0 & 4u) != 0u;
        }
        const auto has_automatic_normal =
            program.surface_normal_root().valid();
        const auto emission_has_automatic_normal =
            dependencies.emission_observes_shading_normal() &&
            has_automatic_normal;
        auto preparation_svm = build_surface_svm_runtime_program(
            program, closure_plans[topology], dependencies,
            compiler::all_surface_closure_endpoints,
            SurfaceValueRuntime::storage_capacity);
        if (!preparation_svm.image.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " unified preparation " +
                         preparation_svm.image.diagnostic;
            return nullptr;
        }
        preparation_svm = compose_surface_svm_runtime_program(
            program, has_automatic_normal ? &topology_normal_storage : nullptr,
            program.surface_normal_root(),
            topology_uses_undisplaced_geometry, std::move(preparation_svm));
        if (!preparation_svm.image.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " unified preparation " +
                         preparation_svm.image.diagnostic;
            return nullptr;
        }
        auto emission_svm = build_surface_svm_runtime_program(
            program, closure_plans[topology], dependencies,
            compiler::surface_closure_endpoint_bit(
                compiler::SurfaceClosureEndpoint::emission),
            SurfaceValueRuntime::storage_capacity);
        if (!emission_svm.image.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " unified emission " +
                         emission_svm.image.diagnostic;
            return nullptr;
        }
        emission_svm = compose_surface_svm_runtime_program(
            program,
            emission_has_automatic_normal ? &topology_normal_storage : nullptr,
            emission_has_automatic_normal ? program.surface_normal_root()
                                          : compiler::ValueExpressionId{},
            emission_has_automatic_normal &&
                topology_uses_undisplaced_geometry,
            std::move(emission_svm));
        if (!emission_svm.image.valid) {
            diagnostic = "surface topology " + std::to_string(topology) +
                         " unified emission " + emission_svm.image.diagnostic;
            return nullptr;
        }
        collect_surface_svm_closure_variants(
            preparation_svm.image,
            runtime->preparation_svm_closure_variants);
        collect_surface_svm_closure_variants(
            emission_svm.image,
            runtime->emission_svm_closure_variants);
        if (bssrdf_topologies[topology]) {
            collect_surface_svm_closure_variants(
                preparation_svm.image,
                runtime->bssrdf_svm_closure_variants);
        }
        runtime->topologies.emplace_back(SurfaceValueRuntimeTopology{
            .program = program_ptr,
            .preparation_addresses =
                std::move(preparation_image.value_addresses)});
        normal_storage.emplace_back(std::move(topology_normal_storage));
        automatic_normal_uses_undisplaced_geometry.emplace_back(
            topology_uses_undisplaced_geometry);
        emission_uses_automatic_normal.emplace_back(
            emission_has_automatic_normal);
        root_storage.emplace_back(std::move(preparation_storage));
        root_storage.emplace_back(std::move(emission_storage));
        svm_programs.emplace_back(std::move(preparation_svm));
        svm_programs.emplace_back(std::move(emission_svm));
    }
    canonicalize_surface_svm_closure_variants(
        runtime->preparation_svm_closure_variants);
    canonicalize_surface_svm_closure_variants(
        runtime->emission_svm_closure_variants);
    canonicalize_surface_svm_closure_variants(
        runtime->bssrdf_svm_closure_variants);

    std::vector<compiler::SurfaceValueExecutionInput> roots;
    roots.reserve(root_storage.size());
    for (auto topology = std::size_t{0u}; topology < programs.size();
         ++topology) {
        const auto has_automatic_normal =
            execution_programs[topology]->surface_normal_root().valid();
        const auto *normal =
            has_automatic_normal ? &normal_storage[topology] : nullptr;
        roots.emplace_back(compiler::SurfaceValueExecutionInput{
            .program = execution_programs[topology].get(),
            .storage =
                &root_storage[topology *
                                  SurfaceValueRuntime::programs_per_topology +
                              SurfaceValueRuntime::preparation_program_offset],
            .surface_normal_storage = normal,
            .surface_normal_output =
                execution_programs[topology]->surface_normal_root(),
            .surface_normal_uses_undisplaced_geometry =
                has_automatic_normal &&
                automatic_normal_uses_undisplaced_geometry[topology],
            .closure_plan = &closure_plans[topology]});
        const auto emission_has_automatic_normal =
            emission_uses_automatic_normal[topology];
        roots.emplace_back(compiler::SurfaceValueExecutionInput{
            .program = execution_programs[topology].get(),
            .storage =
                &root_storage[topology *
                                  SurfaceValueRuntime::programs_per_topology +
                              SurfaceValueRuntime::emission_program_offset],
            .surface_normal_storage =
                emission_has_automatic_normal ? normal : nullptr,
            .surface_normal_output =
                emission_has_automatic_normal
                    ? execution_programs[topology]->surface_normal_root()
                    : compiler::ValueExpressionId{},
            .surface_normal_uses_undisplaced_geometry =
                emission_has_automatic_normal &&
                automatic_normal_uses_undisplaced_geometry[topology],
            .closure_plan = &closure_plans[topology],
            .closure_endpoints = compiler::surface_closure_endpoint_bit(
                compiler::SurfaceClosureEndpoint::emission)});
    }
    runtime->executable = compiler::build_surface_value_executable_scene(roots);
    if (!runtime->executable.valid) {
        diagnostic = runtime->executable.diagnostic;
        return nullptr;
    }
    if (runtime->executable.values.programs.size() != roots.size()) {
        diagnostic = "one-stream compact surface executable does not preserve "
                     "the root-program bijection";
        return nullptr;
    }
    if (!build_surface_svm_runtime_scene(
            runtime->executable, roots, svm_programs, runtime->svm_scene,
            runtime->svm_instruction_variants, diagnostic)) {
        return nullptr;
    }
    auto svm_value_count = std::uint32_t{};
    auto svm_guard_count = std::uint32_t{};
    auto svm_leaf_count = std::uint32_t{};
    for (const auto &instruction : runtime->svm_scene.instructions) {
        const auto kind = compiler::surface_svm_bytecode_kind(instruction);
        svm_value_count += kind == compiler::SurfaceSvmBytecodeKind::value;
        svm_guard_count +=
            kind == compiler::SurfaceSvmBytecodeKind::jump_if_one ||
            kind == compiler::SurfaceSvmBytecodeKind::jump_if_zero;
        svm_leaf_count +=
            kind == compiler::SurfaceSvmBytecodeKind::closure_leaf;
    }
    LUISA_INFO(
        "Built replacement surface SVM scene: {} programs, {} records ({} "
        "values, {} guards, {} closure leaves), typed slots {}/{}/{}; every "
        "record has an exact evaluator/source proof.",
        runtime->svm_scene.programs.size(),
        runtime->svm_scene.instructions.size(), svm_value_count,
        svm_guard_count, svm_leaf_count,
        runtime->svm_scene.maximum_scalar_slots,
        runtime->svm_scene.maximum_vector_slots,
        runtime->svm_scene.maximum_unsigned_integer_slots);
    static_assert(sizeof(compiler::SurfaceSvmProgramDescriptor) == 32u);
    runtime->svm_program_descriptors.reserve(
        runtime->svm_scene.programs.size() * 2u);
    for (const auto &program : runtime->svm_scene.programs) {
        runtime->svm_program_descriptors.emplace_back(luisa::make_uint4(
            program.instruction_begin, program.instruction_count,
            program.scalar_slots, program.vector_slots));
        runtime->svm_program_descriptors.emplace_back(luisa::make_uint4(
            program.unsigned_integer_slots, program.flags,
            program.endpoints, program.reserved));
    }
    runtime->svm_instructions.reserve(
        runtime->svm_scene.instructions.size());
    for (const auto &instruction : runtime->svm_scene.instructions) {
        runtime->svm_instructions.emplace_back(luisa::make_uint4(
            instruction.control, instruction.payload0, instruction.payload1,
            instruction.payload2));
    }
    runtime->svm_value_operands.assign(
        runtime->svm_scene.value_operands.begin(),
        runtime->svm_scene.value_operands.end());
    runtime->svm_variants.assign(runtime->svm_instruction_variants.begin(),
                                 runtime->svm_instruction_variants.end());
    runtime->svm_metadata_parameters.reserve(
        runtime->svm_scene.value_metadata.size());
    runtime->svm_metadata_static_ranges.reserve(
        runtime->svm_scene.value_metadata.size());
    for (const auto &metadata : runtime->svm_scene.value_metadata) {
        runtime->svm_metadata_parameters.emplace_back(metadata.parameter);
        runtime->svm_metadata_static_ranges.emplace_back(luisa::make_uint2(
            metadata.static_table_begin, metadata.static_table_count));
    }
    runtime->svm_static_data.assign(runtime->svm_scene.static_data.begin(),
                                    runtime->svm_scene.static_data.end());
    runtime->svm_closure_operands.assign(
        runtime->svm_scene.closure_operands.begin(),
        runtime->svm_scene.closure_operands.end());
    runtime->region_specializations =
        compiler::plan_surface_value_region_specializations(
            runtime->executable, region_handler_site_budget);
    if (!runtime->region_specializations.valid) {
        diagnostic = "surface value region specialization: " +
                     runtime->region_specializations.diagnostic;
        return nullptr;
    }
    const auto &image = runtime->executable.values;
    if (runtime->executable.instruction_variants.size() !=
            image.instructions.size() ||
        image.closure_principled_features.size() !=
            image.closure_instructions.size()) {
        diagnostic = "compact surface semantic side streams are not parallel";
        return nullptr;
    }
    runtime->used_principled_closure_features =
        image.used_principled_closure_features;
    std::uint32_t anisotropic_closure_operations = 0u;
    std::uint32_t anisotropic_principled_features = 0u;
    std::uint32_t thin_film_closure_operations = 0u;
    std::uint32_t thin_film_principled_features = 0u;
    runtime->closure_static_variants.reserve(image.closure_instructions.size());
    for (auto instruction_index = std::size_t{0u};
         instruction_index < image.closure_instructions.size();
         ++instruction_index) {
        const auto &instruction =
            image.closure_instructions[instruction_index];
        if (!compiler::surface_closure_is_leaf(instruction)) {
            continue;
        }
        if ((instruction.control &
             compiler::surface_closure_microfacet_anisotropy) != 0u) {
            const auto operation =
                compiler::surface_closure_operation(instruction);
            anisotropic_closure_operations |=
                std::uint32_t{1u} << static_cast<std::uint32_t>(operation);
            if (operation == compiler::ClosureOperation::principled) {
                anisotropic_principled_features |=
                    image.closure_principled_features[instruction_index];
            }
        }
        if ((instruction.control & compiler::surface_closure_thin_film) != 0u) {
            const auto operation =
                compiler::surface_closure_operation(instruction);
            thin_film_closure_operations |=
                std::uint32_t{1u} << static_cast<std::uint32_t>(operation);
            if (operation == compiler::ClosureOperation::principled) {
                thin_film_principled_features |=
                    image.closure_principled_features[instruction_index];
            }
        }
        const auto key =
            instruction.control & compiler::surface_closure_static_variant_mask;
        if (std::find(runtime->closure_static_variants.begin(),
                      runtime->closure_static_variants.end(),
                      key) == runtime->closure_static_variants.end()) {
            runtime->closure_static_variants.emplace_back(key);
        }
    }
    std::sort(runtime->closure_static_variants.begin(),
              runtime->closure_static_variants.end());
    runtime->physical_closure_reachability = reachable_surface_closures(
        image.used_closure_operations,
        image.used_principled_closure_features,
        anisotropic_closure_operations,
        anisotropic_principled_features,
        thin_film_closure_operations,
        thin_film_principled_features);
    runtime->maximum_closure_mix_slots = image.maximum_closure_mix_slots;
    LUISA_INFO(
        "Surface physical-closure reachability: operations=0x{:08x}, "
        "Principled features=0x{:08x}, anisotropic operations=0x{:08x}, "
        "anisotropic Principled features=0x{:08x}, thin-film operations="
        "0x{:08x}, thin-film Principled features=0x{:08x}, kinds=0x{:08x}, "
        "Principled lobes=0x{:08x}, anisotropic kinds=0x{:08x}, "
        "thin-film kinds=0x{:08x}, thin-film lobes=0x{:08x}.",
        image.used_closure_operations,
        image.used_principled_closure_features,
        anisotropic_closure_operations,
        anisotropic_principled_features,
        thin_film_closure_operations,
        thin_film_principled_features,
        runtime->physical_closure_reachability.kinds,
        runtime->physical_closure_reachability.principled_lobes,
        runtime->physical_closure_reachability
            .anisotropic_microfacet_kinds,
        runtime->physical_closure_reachability.thin_film_kinds,
        runtime->physical_closure_reachability.thin_film_principled_lobes);
    const auto append_unique = [](auto &values, auto value) noexcept {
        if (std::find(values.begin(), values.end(), value) == values.end()) {
            values.emplace_back(value);
        }
    };
    const auto valid_program_range = [&image](std::uint32_t program) noexcept {
        if (program >= image.programs.size()) {
            return false;
        }
        const auto &range = image.programs[program];
        return range.instruction_begin <= image.instructions.size() &&
               range.instruction_count <=
                   image.instructions.size() - range.instruction_begin &&
               range.closure_begin <= image.closure_instructions.size() &&
               range.closure_count <=
                   image.closure_instructions.size() - range.closure_begin;
    };
    const auto collect_values =
        [&](std::uint32_t program,
            std::vector<std::uint32_t> &variants) noexcept {
            if (!valid_program_range(program)) {
                return false;
            }
            const auto &range = image.programs[program];
            for (auto offset = std::uint32_t{0u};
                 offset < range.instruction_count; ++offset) {
                const auto instruction = range.instruction_begin + offset;
                const auto variant =
                    runtime->executable.instruction_variants[instruction];
                if (variant == compiler::SurfaceValueAddress::invalid_value) {
                    if (!compiler::is_surface_value_surface_normal_transition(
                            image.instructions[instruction])) {
                        return false;
                    }
                    continue;
                }
                append_unique(variants, variant);
            }
            return true;
        };
    for (auto topology = std::size_t{0u}; topology < programs.size();
         ++topology) {
        const auto base = static_cast<std::uint32_t>(
            topology * SurfaceValueRuntime::programs_per_topology);
        const auto preparation_program =
            base + SurfaceValueRuntime::preparation_program_offset;
        const auto emission_program =
            base + SurfaceValueRuntime::emission_program_offset;
        if (!collect_values(preparation_program,
                            runtime->preparation_value_static_variants) ||
            !collect_values(emission_program,
                            runtime->emission_value_static_variants)) {
            diagnostic = "surface root program range exceeds its stream";
            return nullptr;
        }
        if (bssrdf_topologies[topology] &&
            !collect_values(preparation_program,
                            runtime->bssrdf_value_static_variants)) {
            diagnostic = "BSSRDF topology program range exceeds its stream";
            return nullptr;
        }
        const auto &range = image.programs[emission_program];
        for (auto offset = std::uint32_t{0u}; offset < range.closure_count;
             ++offset) {
            const auto closure = range.closure_begin + offset;
            const auto &instruction = image.closure_instructions[closure];
            if (!compiler::surface_closure_is_leaf(instruction)) {
                continue;
            }
            const auto endpoints =
                compiler::surface_closure_endpoints(instruction);
            const auto operation =
                compiler::surface_closure_operation(instruction);
            if (endpoints != compiler::surface_closure_endpoint_bit(
                                 compiler::SurfaceClosureEndpoint::emission) ||
                (operation != compiler::ClosureOperation::emission &&
                 operation != compiler::ClosureOperation::principled)) {
                diagnostic =
                    "emission projection retained a non-emission closure";
                return nullptr;
            }
            append_unique(
                runtime->emission_closure_static_variants,
                instruction.control &
                    compiler::surface_closure_emission_static_variant_mask);
            runtime->emission_principled_closure_features |=
                image.closure_principled_features[closure];
        }
        if (bssrdf_topologies[topology]) {
            const auto &bssrdf_range = image.programs[preparation_program];
            for (auto offset = std::uint32_t{0u};
                 offset < bssrdf_range.closure_count; ++offset) {
                const auto closure = bssrdf_range.closure_begin + offset;
                const auto &instruction = image.closure_instructions[closure];
                if (!compiler::surface_closure_is_leaf(instruction)) {
                    continue;
                }
                if (compiler::surface_closure_endpoints(instruction) == 0u) {
                    diagnostic =
                        "BSSRDF topology retained an endpoint-free closure";
                    return nullptr;
                }
                append_unique(
                    runtime->bssrdf_closure_static_variants,
                    instruction.control &
                        compiler::surface_closure_static_variant_mask);
                runtime->bssrdf_principled_closure_features |=
                    image.closure_principled_features[closure];
            }
        }
    }
    const auto sort_variants = [](auto &variants) noexcept {
        std::sort(variants.begin(), variants.end());
    };
    sort_variants(runtime->preparation_value_static_variants);
    sort_variants(runtime->emission_value_static_variants);
    sort_variants(runtime->bssrdf_value_static_variants);
    std::sort(runtime->emission_closure_static_variants.begin(),
              runtime->emission_closure_static_variants.end());
    std::sort(runtime->bssrdf_closure_static_variants.begin(),
              runtime->bssrdf_closure_static_variants.end());

    for (auto index = std::size_t{0u}; index < image.programs.size(); ++index) {
        if (!fits_runtime_capacity(image.programs[index])) {
            diagnostic =
                "compact surface program " + std::to_string(index) +
                " requires typed slots beyond the 8 scalar / 12 vector / 1 "
                "uint64 validation capacity";
            return nullptr;
        }
        runtime->program_ranges.emplace_back(
            luisa::make_uint4(image.programs[index].instruction_begin,
                              image.programs[index].instruction_count,
                              image.programs[index].closure_begin,
                              image.programs[index].closure_count));
        runtime->program_flags.emplace_back(image.programs[index].flags);
    }
    runtime->region_specializations_use_inline_tags =
        !runtime->region_specializations.specializations.empty() &&
        runtime->region_specializations.specializations.size() <=
            compiler::surface_value_region_specialization_tag_capacity;
    runtime->instructions.reserve(image.instructions.size());
    for (auto instruction_index = std::size_t{};
         instruction_index < image.instructions.size(); ++instruction_index) {
        const auto &instruction = image.instructions[instruction_index];
        auto control = instruction.control;
        if (runtime->region_specializations_use_inline_tags) {
            const auto specialization = runtime->region_specializations
                                            .instruction_specialization_indices[
                                                instruction_index];
            if (specialization != compiler::surface_value_no_region) {
                if (specialization >=
                        runtime->region_specializations.specializations.size() ||
                    !compiler::
                        surface_value_region_specialization_has_inline_tag(
                            specialization)) {
                    diagnostic =
                        "surface value region specialization cannot be encoded "
                        "in the validated runtime instruction";
                    return nullptr;
                }
                control |= compiler::make_surface_value_region_specialization_tag(
                    specialization);
            }
        }
        runtime->instructions.emplace_back(luisa::make_uint4(
            control, instruction.result, instruction.operand_payload,
            instruction.metadata_index));
    }
    runtime->operands.assign(image.operands.begin(), image.operands.end());
    runtime->instruction_variants.assign(
        runtime->executable.instruction_variants.begin(),
        runtime->executable.instruction_variants.end());
    // Inline-tag plans and the production budget-zero route never read the
    // parallel side stream. Keep only the later dummy binding in those cases;
    // copying and uploading one uint per instruction would be pure overhead.
    if (!runtime->region_specializations.specializations.empty() &&
        !runtime->region_specializations_use_inline_tags) {
        runtime->instruction_region_specializations.assign(
            runtime->region_specializations.instruction_specialization_indices
                .begin(),
            runtime->region_specializations.instruction_specialization_indices
                .end());
    }
    runtime->metadata_parameters.reserve(image.metadata.size());
    runtime->metadata_static_ranges.reserve(image.metadata.size());
    for (const auto &metadata : image.metadata) {
        runtime->metadata_parameters.emplace_back(metadata.parameter);
        runtime->metadata_static_ranges.emplace_back(luisa::make_uint2(
            metadata.static_table_begin, metadata.static_table_count));
    }
    runtime->static_data.assign(image.static_data.begin(),
                                image.static_data.end());
    runtime->closure_instructions.reserve(image.closure_instructions.size());
    for (const auto &instruction : image.closure_instructions) {
        runtime->closure_instructions.emplace_back(luisa::make_uint4(
            instruction.control, instruction.payload0,
            instruction.payload1, instruction.payload2));
    }
    runtime->closure_operands.assign(image.closure_operands.begin(),
                                     image.closure_operands.end());
    provide_dummy_if_empty(runtime->program_ranges, luisa::make_uint4(0u));
    provide_dummy_if_empty(runtime->program_flags, 0u);
    provide_dummy_if_empty(runtime->instructions, luisa::make_uint4(0u));
    provide_dummy_if_empty(runtime->operands, 0u);
    provide_dummy_if_empty(runtime->instruction_variants, 0u);
    provide_dummy_if_empty(runtime->instruction_region_specializations,
                           compiler::surface_value_no_region);
    provide_dummy_if_empty(runtime->metadata_parameters,
                           compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(runtime->metadata_static_ranges,
                           luisa::make_uint2(0u));
    provide_dummy_if_empty(runtime->static_data, 0.0f);
    provide_dummy_if_empty(runtime->closure_instructions,
                           luisa::make_uint4(0u));
    provide_dummy_if_empty(runtime->closure_operands,
                           compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(runtime->svm_program_descriptors,
                           luisa::make_uint4(0u));
    provide_dummy_if_empty(runtime->svm_instructions,
                           luisa::make_uint4(0u));
    provide_dummy_if_empty(runtime->svm_value_operands, 0u);
    provide_dummy_if_empty(runtime->svm_variants,
                           compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(runtime->svm_metadata_parameters,
                           compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(runtime->svm_metadata_static_ranges,
                           luisa::make_uint2(0u));
    provide_dummy_if_empty(runtime->svm_static_data, 0.0f);
    provide_dummy_if_empty(runtime->svm_closure_operands,
                           compiler::SurfaceValueAddress::invalid_value);

    auto maximum_instruction_count = std::uint32_t{0u};
    auto maximum_scalar_slots = std::uint32_t{0u};
    auto maximum_vector_slots = std::uint32_t{0u};
    auto maximum_unsigned_integer_slots = std::uint32_t{0u};
    for (const auto &program : image.programs) {
        maximum_instruction_count =
            std::max(maximum_instruction_count, program.instruction_count);
        maximum_scalar_slots =
            std::max(maximum_scalar_slots, program.scalar_slots);
        maximum_vector_slots =
            std::max(maximum_vector_slots, program.vector_slots);
        maximum_unsigned_integer_slots = std::max(
            maximum_unsigned_integer_slots, program.unsigned_integer_slots);
    }
    LUISA_INFO(
        "Built one-stream compact surface runtime: {} programs, {} "
        "instructions, {} operands, {} metadata records, {} static floats, "
        "{} semantic variants (population {}, emission {}, BSSRDF {} tags / "
        "{} values), {} closure instructions "
        "(population/emission/BSSRDF variants {}/{}/{}), maximum program "
        "length {}, typed slots {}/{}/{}, linear closure-weight slots {}.",
        image.programs.size(), image.instructions.size(), image.operands.size(),
        image.metadata.size(), image.static_data.size(),
        runtime->executable.variants.size(),
        runtime->preparation_value_static_variants.size(),
        runtime->emission_value_static_variants.size(),
        bssrdf_topology_count, runtime->bssrdf_value_static_variants.size(),
        image.closure_instructions.size(),
        runtime->closure_static_variants.size(),
        runtime->emission_closure_static_variants.size(),
        runtime->bssrdf_closure_static_variants.size(),
        maximum_instruction_count, maximum_scalar_slots, maximum_vector_slots,
        maximum_unsigned_integer_slots,
        runtime->maximum_closure_mix_slots);
    LUISA_INFO(
        "Selected {} typed surface regions: {} static occurrences, {} "
        "handler sites / {} budget, {} eliminated local-bank accesses, {} "
        "dispatch.",
        runtime->region_specializations.specializations.size(),
        [&] {
            auto count = std::uint64_t{};
            for (const auto &specialization :
                 runtime->region_specializations.specializations) {
                count += specialization.static_occurrences;
            }
            return count;
        }(),
        runtime->region_specializations.selected_handler_sites,
        runtime->region_specializations.handler_site_budget,
        runtime->region_specializations.eliminated_bank_accesses,
        runtime->region_specializations.specializations.empty()
            ? "disabled"
            : runtime->region_specializations_use_inline_tags
                  ? "inline instruction-tag"
                  : "side-stream");
    if (surface_value_region_diagnostics_requested()) {
        const auto list = [](const auto &values, const auto &project) {
            std::string text;
            for (const auto &value : values) {
                if (!text.empty()) {
                    text += ',';
                }
                text += std::to_string(project(value));
            }
            return text;
        };
        for (auto index = std::size_t{};
             index < runtime->region_specializations.specializations.size();
             ++index) {
            const auto &specialization =
                runtime->region_specializations.specializations[index];
            const auto &shape = specialization.shape;
            std::vector<std::uint32_t> operand_offsets{0u};
            std::vector<std::uint32_t> source_kinds;
            std::vector<std::uint32_t> source_indices;
            for (const auto &operands : shape.operand_sources) {
                for (const auto &source : operands) {
                    source_kinds.emplace_back(
                        static_cast<std::uint32_t>(source.kind));
                    source_indices.emplace_back(source.index);
                }
                operand_offsets.emplace_back(
                    static_cast<std::uint32_t>(source_kinds.size()));
            }
            LUISA_INFO(
                "Typed surface region {}: static_occurrences={} "
                "handler_sites={} eliminated_accesses={} variants=[{}] "
                "operand_offsets=[{}] source_kinds=[{}] source_indices=[{}] "
                "live_input_banks=[{}] live_outputs=[{}].",
                index, specialization.static_occurrences,
                specialization.handler_site_cost,
                specialization.eliminated_bank_accesses,
                list(shape.variant_indices,
                     [](auto value) { return value; }),
                list(operand_offsets, [](auto value) { return value; }),
                list(source_kinds, [](auto value) { return value; }),
                list(source_indices, [](auto value) { return value; }),
                list(shape.live_input_banks,
                     [](auto value) {
                         return static_cast<std::uint32_t>(value);
                     }),
                list(shape.live_output_instruction_offsets,
                     [](auto value) { return value; }));
        }
    }

    runtime->program_buffer =
        device.create_buffer<luisa::uint4>(runtime->program_ranges.size());
    runtime->program_flag_buffer =
        device.create_buffer<luisa::uint>(runtime->program_flags.size());
    runtime->instruction_buffer =
        device.create_buffer<luisa::uint4>(runtime->instructions.size());
    runtime->operand_buffer =
        device.create_buffer<luisa::uint>(runtime->operands.size());
    runtime->instruction_variant_buffer =
        device.create_buffer<luisa::uint>(runtime->instruction_variants.size());
    runtime->instruction_region_specialization_buffer =
        device.create_buffer<luisa::uint>(
            runtime->instruction_region_specializations.size());
    runtime->metadata_parameter_buffer =
        device.create_buffer<luisa::uint>(runtime->metadata_parameters.size());
    runtime->metadata_static_range_buffer = device.create_buffer<luisa::uint2>(
        runtime->metadata_static_ranges.size());
    runtime->static_data_buffer =
        device.create_buffer<float>(runtime->static_data.size());
    runtime->closure_instruction_buffer = device.create_buffer<luisa::uint4>(
        runtime->closure_instructions.size());
    runtime->closure_operand_buffer =
        device.create_buffer<luisa::uint>(runtime->closure_operands.size());
    runtime->svm_program_buffer = device.create_buffer<luisa::uint4>(
        runtime->svm_program_descriptors.size());
    runtime->svm_instruction_buffer = device.create_buffer<luisa::uint4>(
        runtime->svm_instructions.size());
    runtime->svm_value_operand_buffer = device.create_buffer<luisa::uint>(
        runtime->svm_value_operands.size());
    runtime->svm_instruction_variant_buffer =
        device.create_buffer<luisa::uint>(runtime->svm_variants.size());
    runtime->svm_metadata_parameter_buffer =
        device.create_buffer<luisa::uint>(
            runtime->svm_metadata_parameters.size());
    runtime->svm_metadata_static_range_buffer =
        device.create_buffer<luisa::uint2>(
            runtime->svm_metadata_static_ranges.size());
    runtime->svm_static_data_buffer =
        device.create_buffer<float>(runtime->svm_static_data.size());
    runtime->svm_closure_operand_buffer =
        device.create_buffer<luisa::uint>(
            runtime->svm_closure_operands.size());
    runtime->device_view =
        device.create_bindless_array(surface_value_runtime_buffer_slot_count);
    const auto bind = [&](SurfaceValueRuntimeBufferSlot slot,
                          const auto &buffer) noexcept {
        runtime->device_view.emplace_on_update(
            surface_value_runtime_buffer_slot(slot), buffer);
    };
    bind(SurfaceValueRuntimeBufferSlot::program, runtime->program_buffer);
    bind(SurfaceValueRuntimeBufferSlot::program_flag,
         runtime->program_flag_buffer);
    bind(SurfaceValueRuntimeBufferSlot::instruction,
         runtime->instruction_buffer);
    bind(SurfaceValueRuntimeBufferSlot::operand, runtime->operand_buffer);
    bind(SurfaceValueRuntimeBufferSlot::instruction_variant,
         runtime->instruction_variant_buffer);
    bind(SurfaceValueRuntimeBufferSlot::instruction_region_specialization,
         runtime->instruction_region_specialization_buffer);
    bind(SurfaceValueRuntimeBufferSlot::metadata_parameter,
         runtime->metadata_parameter_buffer);
    bind(SurfaceValueRuntimeBufferSlot::metadata_static_range,
         runtime->metadata_static_range_buffer);
    bind(SurfaceValueRuntimeBufferSlot::static_data,
         runtime->static_data_buffer);
    bind(SurfaceValueRuntimeBufferSlot::closure_instruction,
         runtime->closure_instruction_buffer);
    bind(SurfaceValueRuntimeBufferSlot::closure_operand,
         runtime->closure_operand_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_program,
         runtime->svm_program_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_instruction,
         runtime->svm_instruction_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_value_operand,
         runtime->svm_value_operand_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_instruction_variant,
         runtime->svm_instruction_variant_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_metadata_parameter,
         runtime->svm_metadata_parameter_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_metadata_static_range,
         runtime->svm_metadata_static_range_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_static_data,
         runtime->svm_static_data_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_closure_operand,
         runtime->svm_closure_operand_buffer);
    return runtime;
}

void upload_surface_value_runtime(Stream &stream,
                                  SurfaceValueRuntime &runtime) noexcept {
    stream << runtime.program_buffer.copy_from(
                  luisa::span{runtime.program_ranges})
           << runtime.program_flag_buffer.copy_from(
                  luisa::span{runtime.program_flags})
           << runtime.instruction_buffer.copy_from(
                  luisa::span{runtime.instructions})
           << runtime.operand_buffer.copy_from(luisa::span{runtime.operands})
           << runtime.instruction_variant_buffer.copy_from(
                  luisa::span{runtime.instruction_variants})
           << runtime.instruction_region_specialization_buffer.copy_from(
                  luisa::span{runtime.instruction_region_specializations})
           << runtime.metadata_parameter_buffer.copy_from(
                  luisa::span{runtime.metadata_parameters})
           << runtime.metadata_static_range_buffer.copy_from(
                  luisa::span{runtime.metadata_static_ranges})
           << runtime.static_data_buffer.copy_from(
                  luisa::span{runtime.static_data})
           << runtime.closure_instruction_buffer.copy_from(
                  luisa::span{runtime.closure_instructions})
           << runtime.closure_operand_buffer.copy_from(
                  luisa::span{runtime.closure_operands})
           << runtime.svm_program_buffer.copy_from(
                  luisa::span{runtime.svm_program_descriptors})
           << runtime.svm_instruction_buffer.copy_from(
                  luisa::span{runtime.svm_instructions})
           << runtime.svm_value_operand_buffer.copy_from(
                  luisa::span{runtime.svm_value_operands})
           << runtime.svm_instruction_variant_buffer.copy_from(
                  luisa::span{runtime.svm_variants})
           << runtime.svm_metadata_parameter_buffer.copy_from(
                  luisa::span{runtime.svm_metadata_parameters})
           << runtime.svm_metadata_static_range_buffer.copy_from(
                  luisa::span{runtime.svm_metadata_static_ranges})
           << runtime.svm_static_data_buffer.copy_from(
                  luisa::span{runtime.svm_static_data})
           << runtime.svm_closure_operand_buffer.copy_from(
                  luisa::span{runtime.svm_closure_operands})
           << runtime.device_view.update();
}

} // namespace psycles::luisa_backend::detail

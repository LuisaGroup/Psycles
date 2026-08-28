#include "path_tracer_surface_values.h"

#include <psycles/compiler/surface_bump_expansion.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
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

void canonicalize_u32_domain(std::vector<std::uint32_t> &values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

[[nodiscard]] bool collect_surface_svm_value_domain(
    const compiler::SurfaceSvmSceneImage &scene,
    std::span<const std::uint32_t> instruction_variants,
    std::size_t evaluator_count,
    std::uint32_t program_index,
    std::vector<std::uint32_t> &domain,
    std::string &diagnostic) {
    if (instruction_variants.size() != scene.instructions.size() ||
        program_index >= scene.programs.size()) {
        diagnostic = "unified surface evaluator side stream is not total";
        return false;
    }
    const auto &program = scene.programs[program_index];
    if (program.instruction_begin > scene.instructions.size() ||
        program.instruction_count >
            scene.instructions.size() - program.instruction_begin) {
        diagnostic =
            "unified surface program range exceeds its instruction stream";
        return false;
    }
    for (auto offset = std::uint32_t{}; offset < program.instruction_count;
         ++offset) {
        const auto instruction_index = program.instruction_begin + offset;
        const auto kind = compiler::surface_svm_bytecode_kind(
            scene.instructions[instruction_index]);
        const auto variant = instruction_variants[instruction_index];
        if (kind == compiler::SurfaceSvmBytecodeKind::value) {
            if (variant >= evaluator_count) {
                diagnostic =
                    "unified surface value record has no exact evaluator";
                return false;
            }
            domain.emplace_back(variant);
        } else if (variant != compiler::SurfaceValueAddress::invalid_value) {
            diagnostic =
                "unified surface control record names a value evaluator";
            return false;
        }
    }
    return true;
}

void project_surface_svm_closure_domain(
    std::span<const SurfaceSvmClosureVariant> variants,
    std::uint32_t static_variant_mask,
    std::vector<std::uint32_t> &static_variants,
    compiler::PrincipledClosureFeatureMask &principled_features) {
    static_variants.reserve(static_variants.size() + variants.size());
    for (const auto &variant : variants) {
        static_variants.emplace_back(
            variant.static_variant & static_variant_mask);
        principled_features |= variant.principled_features;
    }
    canonicalize_u32_domain(static_variants);
}

struct SurfaceSvmClosureCapabilities {
    std::uint32_t operations{};
    compiler::PrincipledClosureFeatureMask principled_features{};
    std::uint32_t anisotropic_operations{};
    compiler::PrincipledClosureFeatureMask anisotropic_principled_features{};
    std::uint32_t thin_film_operations{};
    compiler::PrincipledClosureFeatureMask thin_film_principled_features{};
};

[[nodiscard]] SurfaceSvmClosureCapabilities
analyze_surface_svm_closure_capabilities(
    std::span<const SurfaceSvmClosureVariant> variants) noexcept {
    static_assert(static_cast<std::uint32_t>(
                      compiler::ClosureOperation::refraction) < 32u);
    SurfaceSvmClosureCapabilities result;
    for (const auto &variant : variants) {
        const auto operation = static_cast<compiler::ClosureOperation>(
            variant.static_variant & compiler::surface_closure_opcode_mask);
        const auto operation_index = static_cast<std::uint32_t>(operation);
        const auto operation_bit = std::uint32_t{1u} << operation_index;
        result.operations |= operation_bit;
        result.principled_features |= variant.principled_features;
        if ((variant.static_variant &
             compiler::surface_closure_microfacet_anisotropy) != 0u) {
            result.anisotropic_operations |= operation_bit;
            if (operation == compiler::ClosureOperation::principled) {
                result.anisotropic_principled_features |=
                    variant.principled_features;
            }
        }
        if ((variant.static_variant & compiler::surface_closure_thin_film) !=
            0u) {
            result.thin_film_operations |= operation_bit;
            if (operation == compiler::ClosureOperation::principled) {
                result.thin_film_principled_features |=
                    variant.principled_features;
            }
        }
    }
    return result;
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
    std::string &diagnostic) {
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
    std::vector<SurfaceSvmRuntimeProgram> svm_programs;
    svm_programs.reserve(programs.size() *
                         SurfaceValueRuntime::programs_per_topology);
    std::vector<compiler::ValueExpressionId> svm_surface_normal_outputs;
    svm_surface_normal_outputs.reserve(
        programs.size() * SurfaceValueRuntime::programs_per_topology);

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
            .program = program_ptr});
        svm_surface_normal_outputs.emplace_back(
            has_automatic_normal ? program.surface_normal_root()
                                 : compiler::ValueExpressionId{});
        svm_surface_normal_outputs.emplace_back(
            emission_has_automatic_normal ? program.surface_normal_root()
                                          : compiler::ValueExpressionId{});
        svm_programs.emplace_back(std::move(preparation_svm));
        svm_programs.emplace_back(std::move(emission_svm));
    }
    canonicalize_surface_svm_closure_variants(
        runtime->preparation_svm_closure_variants);
    canonicalize_surface_svm_closure_variants(
        runtime->emission_svm_closure_variants);
    canonicalize_surface_svm_closure_variants(
        runtime->bssrdf_svm_closure_variants);

    std::vector<compiler::SurfaceSvmEvaluatorProgramInput> evaluator_inputs;
    evaluator_inputs.reserve(svm_programs.size());
    for (auto program_index = std::size_t{};
         program_index < svm_programs.size(); ++program_index) {
        const auto topology =
            program_index / SurfaceValueRuntime::programs_per_topology;
        evaluator_inputs.emplace_back(
            compiler::SurfaceSvmEvaluatorProgramInput{
                .program = execution_programs[topology].get(),
                .image = &svm_programs[program_index].image,
                .instruction_sources =
                    svm_programs[program_index].instruction_sources,
                .surface_normal_output =
                    svm_surface_normal_outputs[program_index]});
    }
    auto svm_executable =
        compiler::build_surface_svm_executable_scene(evaluator_inputs);
    if (!svm_executable.valid) {
        diagnostic = svm_executable.diagnostic;
        return nullptr;
    }
    runtime->svm_scene = std::move(svm_executable.image);
    runtime->value_variants = std::move(svm_executable.value_variants);
    runtime->svm_instruction_variants =
        std::move(svm_executable.instruction_variants);

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
        "values, {} guards, {} closure leaves), stack lanes {}, typed colors "
        "{}/{}/{}; every "
        "record has an exact evaluator/source proof.",
        runtime->svm_scene.programs.size(),
        runtime->svm_scene.instructions.size(), svm_value_count,
        svm_guard_count, svm_leaf_count,
        runtime->svm_scene.maximum_stack_lanes,
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
            program.endpoints, program.stack_lanes));
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
    runtime->svm_metadata_static_u0.reserve(
        runtime->svm_scene.value_metadata.size());
    runtime->svm_metadata_parameters.reserve(
        runtime->svm_scene.value_metadata.size());
    runtime->svm_metadata_static_ranges.reserve(
        runtime->svm_scene.value_metadata.size());
    for (const auto &metadata : runtime->svm_scene.value_metadata) {
      runtime->svm_metadata_static_u0.emplace_back(
          static_cast<std::uint32_t>(metadata.static_u0));
      runtime->svm_metadata_parameters.emplace_back(metadata.parameter);
      runtime->svm_metadata_static_ranges.emplace_back(luisa::make_uint2(
          metadata.static_table_begin, metadata.static_table_count));
    }
    runtime->svm_static_data.assign(runtime->svm_scene.static_data.begin(),
                                    runtime->svm_scene.static_data.end());
    runtime->svm_closure_operands.assign(
        runtime->svm_scene.closure_operands.begin(),
        runtime->svm_scene.closure_operands.end());
    const auto &image = runtime->svm_scene;
    const auto expected_program_count =
        programs.size() * SurfaceValueRuntime::programs_per_topology;
    if (!image.valid || image.programs.size() != expected_program_count ||
        runtime->svm_instruction_variants.size() !=
            image.instructions.size()) {
        diagnostic =
            "unified surface scene does not preserve the topology/program "
            "bijection";
        return nullptr;
    }

    const auto capabilities = analyze_surface_svm_closure_capabilities(
        runtime->preparation_svm_closure_variants);
    if (capabilities.operations != image.used_closure_operations ||
        capabilities.principled_features !=
            image.used_principled_features) {
        diagnostic =
            "unified surface closure domain disagrees with scene aggregate "
            "capabilities";
        return nullptr;
    }
    project_surface_svm_closure_domain(
        runtime->preparation_svm_closure_variants,
        compiler::surface_closure_static_variant_mask,
        runtime->closure_static_variants,
        runtime->used_principled_closure_features);
    project_surface_svm_closure_domain(
        runtime->emission_svm_closure_variants,
        compiler::surface_closure_emission_static_variant_mask,
        runtime->emission_closure_static_variants,
        runtime->emission_principled_closure_features);
    project_surface_svm_closure_domain(
        runtime->bssrdf_svm_closure_variants,
        compiler::surface_closure_static_variant_mask,
        runtime->bssrdf_closure_static_variants,
        runtime->bssrdf_principled_closure_features);
    if (runtime->used_principled_closure_features !=
        capabilities.principled_features) {
        diagnostic =
            "unified surface closure projection lost a Principled feature";
        return nullptr;
    }

    runtime->physical_closure_reachability = reachable_surface_closures(
        capabilities.operations,
        capabilities.principled_features,
        capabilities.anisotropic_operations,
        capabilities.anisotropic_principled_features,
        capabilities.thin_film_operations,
        capabilities.thin_film_principled_features);
    LUISA_INFO(
        "Surface physical-closure reachability from unified SVM: "
        "operations=0x{:08x}, Principled features=0x{:08x}, anisotropic "
        "operations=0x{:08x}, anisotropic Principled features=0x{:08x}, "
        "thin-film operations=0x{:08x}, thin-film Principled "
        "features=0x{:08x}, kinds=0x{:08x}, Principled lobes=0x{:08x}, "
        "anisotropic kinds=0x{:08x}, thin-film kinds=0x{:08x}, thin-film "
        "lobes=0x{:08x}.",
        capabilities.operations,
        capabilities.principled_features,
        capabilities.anisotropic_operations,
        capabilities.anisotropic_principled_features,
        capabilities.thin_film_operations,
        capabilities.thin_film_principled_features,
        runtime->physical_closure_reachability.kinds,
        runtime->physical_closure_reachability.principled_lobes,
        runtime->physical_closure_reachability
            .anisotropic_microfacet_kinds,
        runtime->physical_closure_reachability.thin_film_kinds,
        runtime->physical_closure_reachability.thin_film_principled_lobes);

    for (auto topology = std::size_t{}; topology < programs.size();
         ++topology) {
        const auto base = static_cast<std::uint32_t>(
            topology * SurfaceValueRuntime::programs_per_topology);
        const auto preparation_program =
            base + SurfaceValueRuntime::preparation_program_offset;
        const auto emission_program =
            base + SurfaceValueRuntime::emission_program_offset;
        const auto emission_endpoint = compiler::surface_closure_endpoint_bit(
            compiler::SurfaceClosureEndpoint::emission);
        if (image.programs[preparation_program].endpoints !=
                compiler::all_surface_closure_endpoints ||
            image.programs[emission_program].endpoints !=
                emission_endpoint) {
            diagnostic =
                "unified surface program endpoint projection is not exact";
            return nullptr;
        }
        if (!collect_surface_svm_value_domain(
                image, runtime->svm_instruction_variants,
                runtime->value_variants.size(), preparation_program,
                runtime->preparation_value_static_variants, diagnostic) ||
            !collect_surface_svm_value_domain(
                image, runtime->svm_instruction_variants,
                runtime->value_variants.size(), emission_program,
                runtime->emission_value_static_variants, diagnostic)) {
            return nullptr;
        }
        if (bssrdf_topologies[topology] &&
            !collect_surface_svm_value_domain(
                image, runtime->svm_instruction_variants,
                runtime->value_variants.size(), preparation_program,
                runtime->bssrdf_value_static_variants, diagnostic)) {
            return nullptr;
        }
    }
    canonicalize_u32_domain(runtime->preparation_value_static_variants);
    canonicalize_u32_domain(runtime->emission_value_static_variants);
    canonicalize_u32_domain(runtime->bssrdf_value_static_variants);

    provide_dummy_if_empty(runtime->svm_program_descriptors,
                           luisa::make_uint4(0u));
    provide_dummy_if_empty(runtime->svm_instructions,
                           luisa::make_uint4(0u));
    provide_dummy_if_empty(runtime->svm_value_operands, 0u);
    provide_dummy_if_empty(runtime->svm_metadata_static_u0, 0u);
    provide_dummy_if_empty(runtime->svm_metadata_parameters,
                           compiler::SurfaceValueAddress::invalid_value);
    provide_dummy_if_empty(runtime->svm_metadata_static_ranges,
                           luisa::make_uint2(0u));
    provide_dummy_if_empty(runtime->svm_static_data, 0.0f);
    provide_dummy_if_empty(runtime->svm_closure_operands,
                           compiler::SurfaceValueAddress::invalid_value);

    LUISA_INFO(
        "Built unified surface SVM runtime: {} programs, {} records, {} value "
        "operands, {} metadata records, {} static floats, {} semantic "
        "evaluators (preparation {}, emission {}, BSSRDF {} tags / {} "
        "evaluators), {} closure leaves "
        "(preparation/emission/BSSRDF variants {}/{}/{}), maximum program "
        "length {}, stack lanes {}, typed colors {}/{}/{}.",
        image.programs.size(), image.instructions.size(),
        image.value_operands.size(), image.value_metadata.size(),
        image.static_data.size(), runtime->value_variants.size(),
        runtime->preparation_value_static_variants.size(),
        runtime->emission_value_static_variants.size(),
        bssrdf_topology_count, runtime->bssrdf_value_static_variants.size(),
        svm_leaf_count, runtime->preparation_svm_closure_variants.size(),
        runtime->emission_svm_closure_variants.size(),
        runtime->bssrdf_svm_closure_variants.size(),
        image.maximum_instruction_count, image.maximum_stack_lanes,
        image.maximum_scalar_slots,
        image.maximum_vector_slots, image.maximum_unsigned_integer_slots);
    runtime->svm_program_buffer = device.create_buffer<luisa::uint4>(
        runtime->svm_program_descriptors.size());
    runtime->svm_instruction_buffer = device.create_buffer<luisa::uint4>(
        runtime->svm_instructions.size());
    runtime->svm_value_operand_buffer = device.create_buffer<luisa::uint>(
        runtime->svm_value_operands.size());
    runtime->svm_metadata_static_u0_buffer = device.create_buffer<luisa::uint>(
        runtime->svm_metadata_static_u0.size());
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
    bind(SurfaceValueRuntimeBufferSlot::svm_program,
         runtime->svm_program_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_instruction,
         runtime->svm_instruction_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_value_operand,
         runtime->svm_value_operand_buffer);
    bind(SurfaceValueRuntimeBufferSlot::svm_metadata_static_u0,
         runtime->svm_metadata_static_u0_buffer);
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
  stream << runtime.svm_program_buffer.copy_from(
                luisa::span{runtime.svm_program_descriptors})
         << runtime.svm_instruction_buffer.copy_from(
                luisa::span{runtime.svm_instructions})
         << runtime.svm_value_operand_buffer.copy_from(
                luisa::span{runtime.svm_value_operands})
         << runtime.svm_metadata_static_u0_buffer.copy_from(
                luisa::span{runtime.svm_metadata_static_u0})
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

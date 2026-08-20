#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/material_library.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/contract/cycles_pointiness.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

void print_value(const psycles::contract::SocketValue &value) {
    using psycles::contract::SocketType;
    switch (value.type) {
        case SocketType::floating:
            std::cout << std::get<float>(value.value);
            break;
        case SocketType::unsigned_integer:
            std::cout << std::get<std::uint64_t>(value.value);
            break;
        case SocketType::boolean:
            std::cout << (std::get<bool>(value.value) ? "true" : "false");
            break;
        case SocketType::color:
        case SocketType::float3:
        case SocketType::point:
        case SocketType::vector:
        case SocketType::normal:
        case SocketType::spectrum: {
            const auto vector =
                std::get<psycles::Vec3f>(value.value);
            std::cout << '(' << vector.x << ", " << vector.y << ", "
                      << vector.z << ')';
            break;
        }
        default:
            std::cout << "<type " << static_cast<unsigned>(value.type)
                      << '>';
            break;
    }
}

[[nodiscard]] psycles::Vec3f transform_point(
    const psycles::Mat4f &transform,
    psycles::Vec3f point) noexcept {
    const auto &e = transform.elements;
    return {
        e[0u] * point.x + e[4u] * point.y +
            e[8u] * point.z + e[12u],
        e[1u] * point.x + e[5u] * point.y +
            e[9u] * point.z + e[13u],
        e[2u] * point.x + e[6u] * point.y +
            e[10u] * point.z + e[14u]};
}

[[nodiscard]] bool near(
    float actual,
    float expected) noexcept {
    return std::abs(actual - expected) <=
           1.0e-6f *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

}// namespace

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cerr
            << "usage: psycles_inspect_blender_material "
               "<export-directory> <material-name> "
               "[--require-generated-transform|"
               "--require-pointiness-source]\n";
        return EXIT_FAILURE;
    }
    const auto option =
        argc == 4 ? std::string_view{argv[3]} : std::string_view{};
    const auto require_generated_transform =
        option == "--require-generated-transform";
    const auto require_pointiness_source =
        option == "--require-pointiness-source";
    if (argc == 4 && !require_generated_transform &&
        !require_pointiness_source) {
        std::cerr
            << "unknown option: " << argv[3] << '\n';
        return EXIT_FAILURE;
    }
    auto imported =
        psycles::adapter::load_blender_scene_bundle(argv[1]);
    for (const auto &diagnostic : imported.diagnostics) {
        std::cerr
            << (diagnostic.severity ==
                        psycles::adapter::
                            BlenderSceneDiagnosticSeverity::error
                    ? "error: "
                    : "warning: ")
            << diagnostic.message << '\n';
    }
    if (!imported.ok()) {
        return EXIT_FAILURE;
    }
    if (require_generated_transform) {
        for (const auto &[geometry_id, geometry] :
             imported.scene->geometries) {
            static_cast<void>(geometry_id);
            if (!geometry.generated_transform) {
                std::cerr
                    << "geometry '" << geometry.name
                    << "' has no volume Generated transform\n";
                return EXIT_FAILURE;
            }
            if (geometry.generated.domain !=
                    psycles::contract::
                        MeshAttributeDomain::point ||
                geometry.generated.values.size() !=
                    geometry.positions.size()) {
                std::cerr
                    << "geometry '" << geometry.name
                    << "' cannot validate its point-domain "
                       "Generated transform\n";
                return EXIT_FAILURE;
            }
            for (std::size_t index = 0u;
                 index < geometry.positions.size();
                 ++index) {
                const auto expected = transform_point(
                    *geometry.generated_transform,
                    geometry.positions[index]);
                const auto actual =
                    geometry.generated.values[index];
                if (!near(actual.x, expected.x) ||
                    !near(actual.y, expected.y) ||
                    !near(actual.z, expected.z)) {
                    std::cerr
                        << "geometry '" << geometry.name
                        << "' has inconsistent surface and "
                           "volume Generated coordinates at point "
                        << index << '\n';
                    return EXIT_FAILURE;
                }
            }
        }
    }
    if (require_pointiness_source) {
        auto found_source = false;
        for (const auto &[geometry_id, geometry] :
             imported.scene->geometries) {
            static_cast<void>(geometry_id);
            if (!geometry.pointiness_source) {
                continue;
            }
            found_source = true;
            try {
                const auto values =
                    psycles::contract::
                        make_cycles_pointiness_attribute(
                            geometry.positions,
                            geometry.pointiness_source
                                ->point_normals,
                            geometry.pointiness_source->edges);
                if (values.size() != geometry.positions.size()) {
                    std::cerr
                        << "geometry '" << geometry.name
                        << "' has an invalid Pointiness extent\n";
                    return EXIT_FAILURE;
                }
            } catch (const std::invalid_argument &error) {
                std::cerr
                    << "geometry '" << geometry.name
                    << "' has an invalid Pointiness source: "
                    << error.what() << '\n';
                return EXIT_FAILURE;
            }
        }
        if (!found_source) {
            std::cerr << "scene has no Cycles Pointiness source\n";
            return EXIT_FAILURE;
        }
    }
    psycles::compiler::ShaderCompiler shader_compiler{
        psycles::compiler::make_core_node_registry()};
    psycles::compiler::MaterialLibrary materials;
    const auto update =
        materials.update(*imported.scene, shader_compiler);
    if (!update.committed) {
        for (const auto &diagnostic : update.diagnostics) {
            std::cerr << "material " << diagnostic.material.value;
            if (const auto iter = imported.scene->materials.find(
                    diagnostic.material);
                iter != imported.scene->materials.end()) {
                std::cerr << " '" << iter->second.name << "'";
            }
            std::cerr << ": " << diagnostic.message << '\n';
        }
        return EXIT_FAILURE;
    }

    const auto requested = std::string_view{argv[2]};
    const auto inspect_all = requested == "*";
    std::map<std::uint64_t, psycles::compiler::SurfaceClosurePlan>
        closure_plans;
    std::map<std::uint64_t, const psycles::compiler::SurfaceProgram *>
        representative_programs;
    if (inspect_all) {
        for (const auto &[id, material] : materials.materials()) {
            static_cast<void>(id);
            const auto &program = *material.surface_program();
            representative_programs.try_emplace(
                program.structure_signature(), &program);
            closure_plans[program.structure_signature()].merge(
                psycles::compiler::analyze_surface_closure_plan(
                    program, material.parameters()));
        }
    }
    for (const auto &[id, material] : materials.materials()) {
        if (!inspect_all && material.name() != requested) {
            continue;
        }
        const auto &program = *material.surface_program();
        std::cout << "material " << id.value << " '" << material.name()
                  << "' signature " << program.structure_signature()
                  << "\nparameters " << program.parameters().size()
                  << "\n";
        if (inspect_all) {
            std::set<std::uint32_t> source_nodes;
            for (const auto &instruction :
                 program.value_instructions()) {
                if (instruction.source_node) {
                    source_nodes.emplace(
                        instruction.source_node.value);
                }
            }
            for (const auto &instruction :
                 program.closure_instructions()) {
                if (instruction.source_node) {
                    source_nodes.emplace(
                        instruction.source_node.value);
                }
            }
            for (const auto &instruction :
                 program.volume_instructions()) {
                if (instruction.source_node) {
                    source_nodes.emplace(
                        instruction.source_node.value);
                }
            }
            const auto source =
                imported.scene->materials.find(id);
            std::cout << "nodes "
                      << (source != imported.scene->materials.end()
                              ? source->second.shader.nodes().size()
                              : 0u)
                      << "\nsource_nodes " << source_nodes.size()
                      << "\n";
            std::cout << "values "
                      << program.value_instructions().size()
                      << "\nclosures "
                      << program.closure_instructions().size()
                      << "\nvolumes "
                      << program.volume_instructions().size()
                      << "\n";
            continue;
        }
        for (const auto &parameter : program.parameters()) {
            std::cout << "  [" << parameter.id.value << "] node "
                      << parameter.node.value << ' ' << parameter.socket
                      << " = ";
            const auto *bound =
                material.parameters().find(parameter.id);
            print_value(
                bound == nullptr ? parameter.default_value : *bound);
            std::cout << '\n';
        }
        std::cout << "values "
                  << program.value_instructions().size() << "\n";
        for (std::size_t i = 0u;
             i < program.value_instructions().size();
             ++i) {
            const auto &instruction =
                program.value_instructions()[i];
            std::cout << "  [" << i << "] op "
                      << static_cast<unsigned>(instruction.operation)
                      << " node " << instruction.source_node.value
                      << " operands=[";
            for (std::size_t operand_index = 0u;
                 operand_index < instruction.operands.size();
                 ++operand_index) {
                if (operand_index != 0u) {
                    std::cout << ',';
                }
                std::cout << instruction.operands[operand_index].value;
            }
            std::cout << "] u0=" << instruction.static_u0
                      << " u1=" << instruction.static_u1 << '\n';
        }
        std::cout << "closures "
                  << program.closure_instructions().size() << "\n";
        for (std::size_t i = 0u;
             i < program.closure_instructions().size();
             ++i) {
            const auto &closure =
                program.closure_instructions()[i];
            std::cout << "  [" << i << "] op "
                      << static_cast<unsigned>(closure.operation)
                      << " color=" << closure.color.value
                      << " strength=" << closure.strength.value << '\n';
        }
        return EXIT_SUCCESS;
    }
    if (inspect_all) {
        using psycles::compiler::PrincipledClosureFeature;
        using psycles::compiler::ValueOperation;
        std::map<ValueOperation, std::size_t> value_operations;
        auto unique_values = std::size_t{};
        auto unique_closures = std::size_t{};
        auto reachable_closures = std::size_t{};
        auto preparation_active_values = std::size_t{};
        auto preparation_parameter_references = std::size_t{};
        auto preparation_runtime_instructions = std::size_t{};
        auto maximum_scalar_slots = std::uint32_t{};
        auto maximum_vector_slots = std::uint32_t{};
        auto maximum_unsigned_integer_slots = std::uint32_t{};
        auto maximum_typed_payload_bytes = std::size_t{};
        auto topology_typed_payload_bytes = std::size_t{};
        auto value_bytecode_instruction_bytes = std::size_t{};
        auto value_bytecode_operand_bytes = std::size_t{};
        auto value_bytecode_metadata_bytes = std::size_t{};
        auto value_bytecode_static_bytes = std::size_t{};
        std::vector<psycles::compiler::SurfaceValueStoragePlan>
            value_storage_plans;
        std::vector<psycles::compiler::SurfaceValueExecutionInput>
            value_execution_inputs;
        value_storage_plans.reserve(representative_programs.size());
        value_execution_inputs.reserve(representative_programs.size());
        std::map<PrincipledClosureFeature, std::size_t>
            feature_occurrences;
        for (const auto &[signature, program] :
             representative_programs) {
            unique_values += program->value_instructions().size();
            unique_closures +=
                program->closure_instructions().size();
            for (const auto &instruction :
                 program->value_instructions()) {
                value_operations[instruction.operation] += 1u;
            }
            const auto &plan = closure_plans.at(signature);
            const auto dependencies =
                psycles::compiler::analyze_surface_value_dependencies(
                    *program, plan);
            const auto storage =
                psycles::compiler::plan_surface_value_storage(
                    *program,
                    dependencies.preparation,
                    dependencies.preparation_outputs);
            if (!storage.valid) {
                std::cerr << "surface execution planning failed for topology "
                          << signature << ": " << storage.diagnostic << '\n';
                return EXIT_FAILURE;
            }
            const auto image =
                psycles::compiler::lower_surface_value_program(
                    *program, storage);
            if (!image.valid) {
                std::cerr << "surface bytecode lowering failed for topology "
                          << signature << ": " << image.diagnostic << '\n';
                return EXIT_FAILURE;
            }
            preparation_active_values += storage.active_values;
            preparation_parameter_references += storage.parameter_values;
            preparation_runtime_instructions += storage.instructions.size();
            maximum_scalar_slots =
                std::max(maximum_scalar_slots, storage.scalar_slots);
            maximum_vector_slots =
                std::max(maximum_vector_slots, storage.vector_slots);
            maximum_unsigned_integer_slots = std::max(
                maximum_unsigned_integer_slots,
                storage.unsigned_integer_slots);
            maximum_typed_payload_bytes = std::max(
                maximum_typed_payload_bytes, storage.payload_bytes());
            topology_typed_payload_bytes += storage.payload_bytes();
            value_bytecode_instruction_bytes +=
                image.instructions.size() *
                sizeof(psycles::compiler::SurfaceValueBytecodeInstruction);
            value_bytecode_operand_bytes +=
                image.operands.size() * sizeof(std::uint32_t);
            value_bytecode_metadata_bytes +=
                image.metadata.size() *
                sizeof(psycles::compiler::SurfaceValueBytecodeMetadata);
            value_bytecode_static_bytes +=
                image.static_data.size() * sizeof(float);
            value_storage_plans.emplace_back(storage);
            value_execution_inputs.emplace_back(
                psycles::compiler::SurfaceValueExecutionInput{
                    .program = program,
                    .storage = &value_storage_plans.back()});
            for (const auto &entry : plan.entries()) {
                reachable_closures += entry.reachable ? 1u : 0u;
                for (const auto feature : {
                         PrincipledClosureFeature::alpha,
                         PrincipledClosureFeature::sheen,
                         PrincipledClosureFeature::coat,
                         PrincipledClosureFeature::metallic,
                         PrincipledClosureFeature::thick_transmission,
                         PrincipledClosureFeature::thin_transmission,
                         PrincipledClosureFeature::dielectric,
                         PrincipledClosureFeature::thick_subsurface,
                         PrincipledClosureFeature::thin_subsurface,
                         PrincipledClosureFeature::diffuse}) {
                    feature_occurrences[feature] +=
                        (entry.principled_features &
                         psycles::compiler::
                             principled_closure_feature_bit(feature)) != 0u
                            ? 1u
                            : 0u;
                }
            }
        }
        const auto value_executable_scene =
            psycles::compiler::build_surface_value_executable_scene(
                value_execution_inputs);
        if (!value_executable_scene.valid) {
            std::cerr << "surface executable bytecode aggregation failed: "
                      << value_executable_scene.diagnostic << '\n';
            return EXIT_FAILURE;
        }
        const auto &value_scene_image = value_executable_scene.values;
        std::map<ValueOperation, std::size_t> value_variants_by_operation;
        for (const auto &variant : value_executable_scene.variants) {
            ++value_variants_by_operation[variant.instruction.operation];
        }
        const auto value_scene_descriptor_bytes =
            value_scene_image.programs.size() *
            sizeof(psycles::compiler::SurfaceValueProgramDescriptor);
        const auto value_scene_total_bytes =
            value_scene_descriptor_bytes +
            value_bytecode_instruction_bytes +
            value_bytecode_operand_bytes +
            value_bytecode_metadata_bytes +
            value_bytecode_static_bytes;
        const auto feature_count =
            [&](PrincipledClosureFeature feature) {
                const auto iter = feature_occurrences.find(feature);
                return iter == feature_occurrences.end()
                           ? std::size_t{}
                           : iter->second;
            };
        std::cout
            << "scene_summary\nunique_topologies "
            << representative_programs.size()
            << "\nunique_values " << unique_values
            << "\nunique_closures " << unique_closures
            << "\nreachable_closures " << reachable_closures
            << "\nvalue_opcode_kinds " << value_operations.size()
            << "\nvalue_static_variants "
            << value_executable_scene.variants.size()
            << "\npreparation_active_values "
            << preparation_active_values
            << "\npreparation_parameter_references "
            << preparation_parameter_references
            << "\npreparation_runtime_instructions "
            << preparation_runtime_instructions
            << "\nmaximum_scalar_slots " << maximum_scalar_slots
            << "\nmaximum_vector_slots " << maximum_vector_slots
            << "\nmaximum_unsigned_integer_slots "
            << maximum_unsigned_integer_slots
            << "\nmaximum_typed_payload_bytes "
            << maximum_typed_payload_bytes
            << "\ntopology_typed_payload_bytes "
            << topology_typed_payload_bytes
            << "\nvalue_bytecode_instruction_bytes "
            << value_bytecode_instruction_bytes
            << "\nvalue_bytecode_operand_bytes "
            << value_bytecode_operand_bytes
            << "\nvalue_bytecode_metadata_bytes "
            << value_bytecode_metadata_bytes
            << "\nvalue_bytecode_static_bytes "
            << value_bytecode_static_bytes
            << "\nvalue_scene_descriptor_bytes "
            << value_scene_descriptor_bytes
            << "\nvalue_scene_total_bytes "
            << value_scene_total_bytes
            << "\nprincipled_alpha "
            << feature_count(PrincipledClosureFeature::alpha)
            << "\nprincipled_sheen "
            << feature_count(PrincipledClosureFeature::sheen)
            << "\nprincipled_coat "
            << feature_count(PrincipledClosureFeature::coat)
            << "\nprincipled_metallic "
            << feature_count(PrincipledClosureFeature::metallic)
            << "\nprincipled_thick_transmission "
            << feature_count(
                   PrincipledClosureFeature::thick_transmission)
            << "\nprincipled_thin_transmission "
            << feature_count(
                   PrincipledClosureFeature::thin_transmission)
            << "\nprincipled_dielectric "
            << feature_count(PrincipledClosureFeature::dielectric)
            << "\nprincipled_thick_subsurface "
            << feature_count(
                   PrincipledClosureFeature::thick_subsurface)
            << "\nprincipled_thin_subsurface "
            << feature_count(
                   PrincipledClosureFeature::thin_subsurface)
            << "\nprincipled_diffuse "
            << feature_count(PrincipledClosureFeature::diffuse)
            << '\n';
        for (const auto &[operation, count] : value_operations) {
            std::cout << "value_opcode "
                      << static_cast<unsigned>(operation)
                      << ' ' << count << '\n';
        }
        for (const auto &[operation, count] :
             value_variants_by_operation) {
            std::cout << "value_variant "
                      << static_cast<unsigned>(operation)
                      << ' ' << count << '\n';
        }
        return EXIT_SUCCESS;
    }
    std::cerr << "material not found: " << requested << '\n';
    return EXIT_FAILURE;
}

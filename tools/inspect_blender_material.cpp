#include <psycles/adapter/blender_scene.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/material_library.h>
#include <psycles/compiler/surface_bump_expansion.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/contract/cycles_pointiness.h>
#include <psycles/luisa/surface_value_runtime_limits.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
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

struct ImageProducerCensus {
    std::size_t producers{};
    std::size_t output_instructions{};
    std::size_t dual_output_producers{};
    std::size_t exact_fusable_pairs{};
    std::size_t exact_fusable_box_pairs{};
    std::size_t exact_fusable_environment_pairs{};

    ImageProducerCensus &operator+=(
        const ImageProducerCensus &other) noexcept {
        producers += other.producers;
        output_instructions += other.output_instructions;
        dual_output_producers += other.dual_output_producers;
        exact_fusable_pairs += other.exact_fusable_pairs;
        exact_fusable_box_pairs += other.exact_fusable_box_pairs;
        exact_fusable_environment_pairs +=
            other.exact_fusable_environment_pairs;
        return *this;
    }
};

struct ImageProducerOutputs {
    bool environment{};
    std::size_t representative{};
    std::optional<std::size_t> color;
    std::optional<std::size_t> alpha;
};

struct VectorMathProducerCensus {
    std::size_t producers{};
    std::size_t output_instructions{};
    std::size_t dual_output_producers{};
    std::size_t exact_fusable_pairs{};

    VectorMathProducerCensus &operator+=(
        const VectorMathProducerCensus &other) noexcept {
        producers += other.producers;
        output_instructions += other.output_instructions;
        dual_output_producers += other.dual_output_producers;
        exact_fusable_pairs += other.exact_fusable_pairs;
        return *this;
    }
};

struct VectorMathProducerOutputs {
    std::size_t representative{};
    std::optional<std::size_t> value;
    std::optional<std::size_t> vector;
};

struct ClosureControlCensus {
    std::size_t reachable_adds{};
    std::size_t reachable_mixes{};
    // A live two-sided Mix admits Cycles' runtime zero/one branch skip. A
    // one-sided endpoint projection still needs its factor, but has no
    // alternative closure subtree in that execution domain.
    std::size_t live_two_sided_mixes{};
    std::size_t live_one_sided_mixes{};
    // Structured control evaluates each contributing Mix exactly once. This
    // source-graph census is the independent formal expectation checked
    // against the emitted bytecode, regardless of the numerical factor value
    // at any shading point.
    std::size_t incremental_mix_factor_evaluations{};

    ClosureControlCensus &operator+=(
        const ClosureControlCensus &other) noexcept {
        reachable_adds += other.reachable_adds;
        reachable_mixes += other.reachable_mixes;
        live_two_sided_mixes += other.live_two_sided_mixes;
        live_one_sided_mixes += other.live_one_sided_mixes;
        incremental_mix_factor_evaluations +=
            other.incremental_mix_factor_evaluations;
        return *this;
    }
};

struct SurfaceOperandRouteCensus {
    bool valid{};
    std::string diagnostic;
    std::size_t references{};
    std::size_t parameter_references{};
    std::size_t local_references{};
    std::size_t zero_arity_instructions{};
    std::size_t all_parameter_instructions{};
    std::size_t all_local_instructions{};
    std::size_t mixed_source_instructions{};
    std::size_t variant_route_pairs{};
    std::size_t polymorphic_variants{};
    std::size_t maximum_routes_per_variant{};
    std::size_t statically_routed_references{};
    std::size_t dynamically_routed_references{};
    std::size_t statically_parameter_references{};
    std::size_t statically_local_references{};
    std::vector<std::set<std::uint16_t>> routes_by_variant;
};

[[nodiscard]] SurfaceOperandRouteCensus census_surface_operand_routes(
    const psycles::compiler::SurfaceValueExecutableScene &scene) {
    SurfaceOperandRouteCensus result;
    if (!scene.valid ||
        scene.instruction_variants.size() != scene.values.instructions.size()) {
        result.diagnostic = "surface value executable scene is not parallel";
        return result;
    }
    result.routes_by_variant.resize(scene.variants.size());

    for (auto instruction_index = std::size_t{};
         instruction_index < scene.values.instructions.size();
         ++instruction_index) {
        const auto &instruction = scene.values.instructions[instruction_index];
        if (psycles::compiler::is_surface_value_surface_normal_transition(
                instruction)) {
            continue;
        }
        const auto variant_index =
            scene.instruction_variants[instruction_index];
        if (variant_index >= scene.variants.size()) {
            result.diagnostic =
                "surface value instruction has an invalid static variant";
            return result;
        }
        const auto operand_count =
            psycles::compiler::surface_value_operand_count(instruction);
        if (operand_count >
            psycles::compiler::surface_value_max_operand_count) {
            result.diagnostic =
                "surface value instruction exceeds the closed operand-arity "
                "contract";
            return result;
        }

        auto parameter_mask = std::uint16_t{};
        for (auto operand_index = std::size_t{}; operand_index < operand_count;
             ++operand_index) {
            const auto word_index =
                operand_index /
                psycles::compiler::surface_value_operands_per_word;
            const auto lane =
                operand_index %
                psycles::compiler::surface_value_operands_per_word;
            const auto word =
                operand_count <=
                        psycles::compiler::surface_value_inline_operand_capacity
                    ? instruction.operand_payload
                    : scene.values.operands.at(instruction.operand_payload +
                                               word_index);
            const auto operand =
                psycles::compiler::surface_value_operand_from_word(word, lane);
            if (!operand.valid()) {
                result.diagnostic =
                    "surface value instruction has an invalid compact "
                    "operand";
                return result;
            }
            ++result.references;
            if (operand.parameter()) {
                parameter_mask |= static_cast<std::uint16_t>(std::uint16_t{1u}
                                                             << operand_index);
                ++result.parameter_references;
            } else {
                ++result.local_references;
            }
        }
        result.routes_by_variant[variant_index].insert(parameter_mask);
        if (operand_count == 0u) {
            ++result.zero_arity_instructions;
        } else {
            const auto all_parameter_mask = static_cast<std::uint16_t>(
                (std::uint32_t{1u} << operand_count) - 1u);
            if (parameter_mask == 0u) {
                ++result.all_local_instructions;
            } else if (parameter_mask == all_parameter_mask) {
                ++result.all_parameter_instructions;
            } else {
                ++result.mixed_source_instructions;
            }
        }
    }

    for (const auto &routes : result.routes_by_variant) {
        if (routes.empty()) {
            result.diagnostic =
                "surface value executable scene has an unobserved variant";
            return result;
        }
        result.variant_route_pairs += routes.size();
        result.polymorphic_variants += routes.size() > 1u ? 1u : 0u;
        result.maximum_routes_per_variant =
            std::max(result.maximum_routes_per_variant, routes.size());
    }

    // Independently reconstruct the same finite product-domain join used by
    // the compiler. This is a census, not a read of the published route, so a
    // broken compiler projection remains visible in the diagnostic output.
    for (auto instruction_index = std::size_t{};
         instruction_index < scene.values.instructions.size();
         ++instruction_index) {
        const auto &instruction = scene.values.instructions[instruction_index];
        if (psycles::compiler::is_surface_value_surface_normal_transition(
                instruction)) {
            continue;
        }
        const auto variant_index =
            scene.instruction_variants[instruction_index];
        const auto operand_count =
            psycles::compiler::surface_value_operand_count(instruction);
        const auto &routes = result.routes_by_variant[variant_index];
        auto parameter_union = std::uint16_t{};
        auto parameter_intersection = static_cast<std::uint16_t>(
            (std::uint32_t{1u} << operand_count) - 1u);
        for (const auto route : routes) {
            parameter_union |= route;
            parameter_intersection &= route;
        }
        const auto dynamic_mask = static_cast<std::uint16_t>(
            parameter_union ^ parameter_intersection);
        for (auto operand_index = std::size_t{}; operand_index < operand_count;
             ++operand_index) {
            const auto bit =
                static_cast<std::uint16_t>(std::uint16_t{1u} << operand_index);
            if ((dynamic_mask & bit) != 0u) {
                ++result.dynamically_routed_references;
            } else {
                ++result.statically_routed_references;
                if ((parameter_intersection & bit) != 0u) {
                    ++result.statically_parameter_references;
                } else {
                    ++result.statically_local_references;
                }
            }
        }
    }
    result.valid = true;
    return result;
}

[[nodiscard]] ClosureControlCensus census_closure_control(
    const psycles::compiler::SurfaceProgram &program,
    const psycles::compiler::SurfaceClosurePlan &closure_plan,
    const psycles::compiler::SurfaceValueDependencyPlan &dependencies) {
    using psycles::compiler::ClosureExpressionId;
    using psycles::compiler::ClosureOperation;
    if (!closure_plan.compatible(program) ||
        !dependencies.compatible(program)) {
        std::abort();
    }
    const auto live = [&](ClosureExpressionId id) noexcept {
        return id.valid() &&
               id.value < program.closure_instructions().size() &&
               closure_plan.entry(id).reachable &&
               (dependencies.physical_closures[id.value] ||
                dependencies.emission_closures[id.value]);
    };

    ClosureControlCensus result;
    for (auto index = std::size_t{};
         index < program.closure_instructions().size(); ++index) {
        const auto id = ClosureExpressionId{
            static_cast<std::uint32_t>(index)};
        if (!live(id)) {
            continue;
        }
        const auto &closure = program.closure_instructions()[index];
        if (closure.operation == ClosureOperation::add) {
            ++result.reachable_adds;
            continue;
        }
        if (closure.operation != ClosureOperation::mix) {
            continue;
        }
        ++result.reachable_mixes;
        const auto a_live = live(closure.a);
        const auto b_live = live(closure.b);
        // lower_surface_closure_program contributes a factor term exactly
        // when both topology branches survived host reachability and at least
        // one branch is live in the selected endpoint domain.
        const auto contributes_factor =
            closure.a.valid() && closure.b.valid() &&
            closure_plan.entry(closure.a).reachable &&
            closure_plan.entry(closure.b).reachable &&
            (a_live || b_live);
        if (!contributes_factor) {
            continue;
        }
        ++result.incremental_mix_factor_evaluations;
        if (a_live && b_live) {
            ++result.live_two_sided_mixes;
        } else {
            ++result.live_one_sided_mixes;
        }
    }
    return result;
}

[[nodiscard]] bool bitwise_equal(
    std::span<const float> lhs,
    std::span<const float> rhs) noexcept {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (auto index = std::size_t{}; index < lhs.size(); ++index) {
        if (std::bit_cast<std::uint32_t>(lhs[index]) !=
            std::bit_cast<std::uint32_t>(rhs[index])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool exact_same_producer_configuration(
    const psycles::compiler::ValueInstruction &lhs,
    const psycles::compiler::ValueInstruction &rhs) noexcept {
    return static_cast<bool>(lhs.source_node) &&
           lhs.source_node == rhs.source_node && lhs.operands == rhs.operands &&
           lhs.parameter == rhs.parameter && lhs.static_u0 == rhs.static_u0 &&
           lhs.static_u1 == rhs.static_u1 &&
           std::bit_cast<std::uint32_t>(lhs.static_f0) ==
               std::bit_cast<std::uint32_t>(rhs.static_f0) &&
           std::bit_cast<std::uint32_t>(lhs.static_f1) ==
               std::bit_cast<std::uint32_t>(rhs.static_f1) &&
           bitwise_equal(lhs.static_table, rhs.static_table);
}

[[nodiscard]] ImageProducerCensus census_image_producers(
    std::span<const psycles::compiler::ValueInstruction> instructions,
    const std::vector<bool> *active = nullptr) {
    using psycles::compiler::ValueOperation;
    if (active != nullptr && active->size() != instructions.size()) {
        std::abort();
    }
    // Producer equivalence is source provenance plus the complete projected
    // configuration. Graph expansion intentionally clones the same authored
    // node under different differential contexts, whose remapped operands
    // make them distinct equivalence classes. Synthetic instructions have no
    // provenance and therefore remain singleton classes.
    std::vector<ImageProducerOutputs> producers;
    for (auto index = std::size_t{}; index < instructions.size(); ++index) {
        if (active != nullptr && !(*active)[index]) {
            continue;
        }
        const auto &instruction = instructions[index];
        auto environment = false;
        auto color = false;
        switch (instruction.operation) {
            case ValueOperation::image_color:
                color = true;
                break;
            case ValueOperation::image_alpha:
                break;
            case ValueOperation::environment_color:
                environment = true;
                color = true;
                break;
            case ValueOperation::environment_alpha:
                environment = true;
                break;
            default:
                continue;
        }
        auto producer = std::find_if(
            producers.begin(), producers.end(), [&](const auto &candidate) {
                return candidate.environment == environment &&
                       exact_same_producer_configuration(
                           instructions[candidate.representative], instruction);
            });
        if (producer == producers.end()) {
            producer = producers.emplace(
                producer, ImageProducerOutputs{
                              .environment = environment,
                              .representative = index});
        }
        auto &output = color ? producer->color : producer->alpha;
        if (output) {
            std::abort();
        }
        output = index;
    }

    ImageProducerCensus result;
    result.producers = producers.size();
    for (const auto &outputs : producers) {
        result.output_instructions += outputs.color.has_value() ? 1u : 0u;
        result.output_instructions += outputs.alpha.has_value() ? 1u : 0u;
        if (!outputs.color || !outputs.alpha) {
            continue;
        }
        ++result.dual_output_producers;
        const auto &color = instructions[*outputs.color];
        const auto &alpha = instructions[*outputs.alpha];
        if (!exact_same_producer_configuration(color, alpha)) {
            continue;
        }
        ++result.exact_fusable_pairs;
        if (outputs.environment) {
            ++result.exact_fusable_environment_pairs;
        } else if (((color.static_u1 >> 12u) & 0x3u) == 1u) {
            ++result.exact_fusable_box_pairs;
        }
    }
    return result;
}

[[nodiscard]] VectorMathProducerCensus census_vector_math_producers(
    std::span<const psycles::compiler::ValueInstruction> instructions,
    const std::vector<bool> *active = nullptr) {
    using psycles::compiler::ValueOperation;
    if (active != nullptr && active->size() != instructions.size()) {
        std::abort();
    }

    // Use the same exact equivalence relation as the image census. Source-node
    // identity alone is insufficient after differential-context expansion.
    std::vector<VectorMathProducerOutputs> producers;
    for (auto index = std::size_t{}; index < instructions.size(); ++index) {
        if (active != nullptr && !(*active)[index]) {
            continue;
        }
        const auto &instruction = instructions[index];
        const auto is_value =
            instruction.operation == ValueOperation::vector_math_value;
        if (!is_value &&
            instruction.operation != ValueOperation::vector_math_vector) {
            continue;
        }
        auto producer = std::find_if(
            producers.begin(), producers.end(), [&](const auto &candidate) {
                return exact_same_producer_configuration(
                    instructions[candidate.representative], instruction);
            });
        if (producer == producers.end()) {
            producer = producers.emplace(
                producer,
                VectorMathProducerOutputs{.representative = index});
        }
        auto &output = is_value ? producer->value : producer->vector;
        if (output) {
            std::abort();
        }
        output = index;
    }

    VectorMathProducerCensus result;
    result.producers = producers.size();
    for (const auto &outputs : producers) {
        result.output_instructions += outputs.value.has_value() ? 1u : 0u;
        result.output_instructions += outputs.vector.has_value() ? 1u : 0u;
        if (!outputs.value || !outputs.vector) {
            continue;
        }
        ++result.dual_output_producers;
        if (exact_same_producer_configuration(
                instructions[*outputs.value], instructions[*outputs.vector])) {
            ++result.exact_fusable_pairs;
        }
    }
    return result;
}

[[nodiscard]] bool producer_partition_regression() {
    using psycles::compiler::ValueExpressionId;
    using psycles::compiler::ValueInstruction;
    using psycles::compiler::ValueOperation;
    using psycles::contract::NodeId;
    using psycles::contract::SocketType;

    const auto make_output = [](ValueOperation operation, NodeId source,
                                SocketType type,
                                ValueExpressionId context) {
        return ValueInstruction{
            .operation = operation,
            .source_node = source,
            .result_type = type,
            .operands = {context, ValueExpressionId{2u}},
            .static_u0 = 3u,
            .static_u1 = 5u};
    };
    const std::vector image_instructions{
        make_output(ValueOperation::image_color, NodeId{7u},
                    SocketType::color, ValueExpressionId{0u}),
        make_output(ValueOperation::image_alpha, NodeId{7u},
                    SocketType::floating, ValueExpressionId{0u}),
        make_output(ValueOperation::image_color, NodeId{7u},
                    SocketType::color, ValueExpressionId{1u}),
        make_output(ValueOperation::image_alpha, NodeId{7u},
                    SocketType::floating, ValueExpressionId{1u})};
    const auto image = census_image_producers(image_instructions);

    const std::vector vector_math_instructions{
        make_output(ValueOperation::vector_math_vector, NodeId{8u},
                    SocketType::vector, ValueExpressionId{0u}),
        make_output(ValueOperation::vector_math_value, NodeId{8u},
                    SocketType::floating, ValueExpressionId{0u}),
        make_output(ValueOperation::vector_math_vector, NodeId{8u},
                    SocketType::vector, ValueExpressionId{1u}),
        make_output(ValueOperation::vector_math_value, NodeId{8u},
                    SocketType::floating, ValueExpressionId{1u})};
    const auto vector_math =
        census_vector_math_producers(vector_math_instructions);
    return image.producers == 2u && image.output_instructions == 4u &&
           image.dual_output_producers == 2u &&
           image.exact_fusable_pairs == 2u &&
           vector_math.producers == 2u &&
           vector_math.output_instructions == 4u &&
           vector_math.dual_output_producers == 2u &&
           vector_math.exact_fusable_pairs == 2u;
}

}// namespace

int main(int argc, char **argv) {
    if (argc == 2 &&
        std::string_view{argv[1]} == "--self-test-producer-partition") {
        if (!producer_partition_regression()) {
            std::cerr << "exact producer partition regression failed\n";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
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
    std::map<psycles::compiler::ClosureOperation, std::size_t>
        material_reachable_closure_operations;
    std::map<psycles::compiler::PrincipledClosureFeatureMask, std::size_t>
        material_principled_feature_masks;
    using SurfaceExecutionClassKey =
        std::pair<std::uint64_t, std::vector<std::uint32_t>>;
    std::set<SurfaceExecutionClassKey> material_execution_classes;
    auto execution_class_reachable_leaves = std::size_t{};
    auto execution_class_reachable_mixes = std::size_t{};
    auto execution_class_reachable_adds = std::size_t{};
    auto material_reachable_mixes = std::size_t{};
    auto material_reachable_adds = std::size_t{};
    if (inspect_all) {
        for (const auto &[id, material] : materials.materials()) {
            static_cast<void>(id);
            const auto &program = *material.surface_program();
            representative_programs.try_emplace(
                program.structure_signature(), &program);
            const auto material_plan =
                psycles::compiler::analyze_surface_closure_plan(
                    program, material.parameters());
            auto class_key = SurfaceExecutionClassKey{
                program.structure_signature(), {}};
            class_key.second.reserve(material_plan.entries().size());
            for (const auto &entry : material_plan.entries()) {
                class_key.second.emplace_back(
                    (entry.principled_features << 1u) |
                    static_cast<std::uint32_t>(entry.reachable));
            }
            const auto new_execution_class =
                material_execution_classes.emplace(
                    std::move(class_key)).second;
            closure_plans[program.structure_signature()].merge(
                material_plan);
            for (auto index = std::size_t{0u};
                 index < material_plan.entries().size(); ++index) {
                const auto &entry = material_plan.entries()[index];
                if (!entry.reachable) {
                    continue;
                }
                const auto operation =
                    program.closure_instructions()[index].operation;
                if (operation ==
                    psycles::compiler::ClosureOperation::mix) {
                    ++material_reachable_mixes;
                    execution_class_reachable_mixes +=
                        new_execution_class ? 1u : 0u;
                    continue;
                }
                if (operation ==
                    psycles::compiler::ClosureOperation::add) {
                    ++material_reachable_adds;
                    execution_class_reachable_adds +=
                        new_execution_class ? 1u : 0u;
                    continue;
                }
                if (operation !=
                    psycles::compiler::ClosureOperation::null_closure) {
                    ++material_reachable_closure_operations[operation];
                    execution_class_reachable_leaves +=
                        new_execution_class ? 1u : 0u;
                }
                if (operation ==
                    psycles::compiler::ClosureOperation::principled) {
                    ++material_principled_feature_masks
                        [entry.principled_features];
                }
            }
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
        std::map<ValueOperation, std::size_t>
            preparation_runtime_operations;
        auto unique_values = std::size_t{};
        auto expanded_values = std::size_t{};
        auto expanded_bump_nodes = std::size_t{};
        auto expanded_bump_sampled_instructions = std::size_t{};
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
        auto value_bytecode_inline_operand_instructions = std::size_t{};
        auto value_bytecode_inline_operands = std::size_t{};
        auto value_bytecode_packed_operand_loads = std::size_t{};
        ImageProducerCensus reachable_image_producers;
        ImageProducerCensus preparation_image_producers;
        VectorMathProducerCensus reachable_vector_math_producers;
        VectorMathProducerCensus physical_vector_math_producers;
        VectorMathProducerCensus emission_vector_math_producers;
        VectorMathProducerCensus preparation_vector_math_producers;
        ClosureControlCensus closure_control;
        std::vector<psycles::compiler::SurfaceValueStoragePlan>
            value_storage_plans;
        std::vector<std::shared_ptr<const psycles::compiler::SurfaceProgram>>
            value_execution_programs;
        std::vector<psycles::compiler::SurfaceValueExecutionInput>
            value_execution_inputs;
        value_storage_plans.reserve(representative_programs.size());
        value_execution_programs.reserve(representative_programs.size());
        value_execution_inputs.reserve(representative_programs.size());
        std::map<PrincipledClosureFeature, std::size_t>
            feature_occurrences;
        std::map<psycles::compiler::PrincipledClosureFeatureMask, std::size_t>
            principled_feature_masks;
        std::map<psycles::compiler::ClosureOperation, std::size_t>
            closure_bytecode_operations;
        for (const auto &[signature, program] :
             representative_programs) {
            unique_values += program->value_instructions().size();
            unique_closures +=
                program->closure_instructions().size();
            for (const auto &instruction :
                 program->value_instructions()) {
                value_operations[instruction.operation] += 1u;
            }
            reachable_image_producers +=
                census_image_producers(program->value_instructions());
            reachable_vector_math_producers +=
                census_vector_math_producers(program->value_instructions());
            const auto &plan = closure_plans.at(signature);
            auto expansion =
                psycles::compiler::expand_surface_bump_program(*program);
            if (!expansion.valid || !expansion.program) {
                std::cerr << "surface Bump graph expansion failed for topology "
                          << signature << ": " << expansion.diagnostic << '\n';
                return EXIT_FAILURE;
            }
            if (!plan.compatible(*expansion.program)) {
                std::cerr << "surface Bump graph expansion changed closure "
                             "topology "
                          << signature << '\n';
                return EXIT_FAILURE;
            }
            expanded_values += expansion.program->value_instructions().size();
            expanded_bump_nodes += expansion.bump_count;
            expanded_bump_sampled_instructions +=
                expansion.sampled_instruction_count;
            value_execution_programs.emplace_back(
                std::move(expansion.program));
            const auto *execution_program =
                value_execution_programs.back().get();
            const auto dependencies =
                psycles::compiler::analyze_surface_value_dependencies(
                    *execution_program, plan);
            closure_control += census_closure_control(
                *execution_program, plan, dependencies);
            preparation_image_producers += census_image_producers(
                execution_program->value_instructions(),
                &dependencies.preparation);
            physical_vector_math_producers += census_vector_math_producers(
                execution_program->value_instructions(),
                &dependencies.physical);
            emission_vector_math_producers += census_vector_math_producers(
                execution_program->value_instructions(),
                &dependencies.emission);
            preparation_vector_math_producers += census_vector_math_producers(
                execution_program->value_instructions(),
                &dependencies.preparation);
            const auto storage =
                psycles::compiler::plan_surface_value_storage(
                    *execution_program,
                    dependencies.preparation,
                    dependencies.preparation_outputs,
                    psycles::luisa_backend::
                        compact_surface_value_storage_capacity);
            if (!storage.valid) {
                std::cerr << "surface execution planning failed for topology "
                          << signature << ": " << storage.diagnostic << '\n';
                return EXIT_FAILURE;
            }
            const auto image =
                psycles::compiler::lower_surface_value_program(
                    *execution_program, storage);
            if (!image.valid) {
                std::cerr << "surface bytecode lowering failed for topology "
                          << signature << ": " << image.diagnostic << '\n';
                return EXIT_FAILURE;
            }
            preparation_active_values += storage.active_values;
            preparation_parameter_references += storage.parameter_values;
            preparation_runtime_instructions += storage.instructions.size();
            for (const auto id : storage.instructions) {
                if (!id.valid() ||
                    id.value >=
                        execution_program->value_instructions().size()) {
                    std::cerr
                        << "surface execution schedule contains an invalid "
                           "instruction for topology "
                        << signature << '\n';
                    return EXIT_FAILURE;
                }
                ++preparation_runtime_operations
                    [execution_program->value_instructions()[id.value]
                         .operation];
            }
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
            for (const auto &instruction : image.instructions) {
                // Two compact addresses occupy the instruction's operand
                // payload word. Metadata remains independent in the fourth
                // word; opcode-derived arity makes the layout canonical.
                const auto operand_count =
                    psycles::compiler::surface_value_operand_count(
                        instruction);
                if (operand_count <=
                    psycles::compiler::surface_value_inline_operand_capacity) {
                    ++value_bytecode_inline_operand_instructions;
                    value_bytecode_inline_operands += operand_count;
                }
                if (operand_count >
                    psycles::compiler::surface_value_inline_operand_capacity) {
                    value_bytecode_packed_operand_loads +=
                        psycles::compiler::surface_value_operand_word_count(
                            operand_count);
                }
            }
            value_storage_plans.emplace_back(storage);
            value_execution_inputs.emplace_back(
                psycles::compiler::SurfaceValueExecutionInput{
                    .program = execution_program,
                    .storage = &value_storage_plans.back(),
                    .closure_plan = &plan,
                    .closure_endpoints =
                        psycles::compiler::all_surface_closure_endpoints});
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
        const auto value_operand_routes =
            census_surface_operand_routes(value_executable_scene);
        if (!value_operand_routes.valid) {
            std::cerr << value_operand_routes.diagnostic << '\n';
            return EXIT_FAILURE;
        }
        const auto linear_mix_factor_evaluations =
            static_cast<std::size_t>(std::count_if(
                value_scene_image.closure_instructions.begin(),
                value_scene_image.closure_instructions.end(),
                [](const auto &instruction) noexcept {
                    return !psycles::compiler::surface_closure_is_leaf(
                        instruction);
                }));
        const auto binary_closure_mixes =
            static_cast<std::size_t>(std::count_if(
                value_scene_image.closure_instructions.begin(),
                value_scene_image.closure_instructions.end(),
                [](const auto &instruction) noexcept {
                    return psycles::compiler::surface_closure_instruction_kind(
                               instruction) ==
                           psycles::compiler::
                               SurfaceClosureInstructionKind::mix_both;
                }));
        const auto left_only_closure_mixes =
            static_cast<std::size_t>(std::count_if(
                value_scene_image.closure_instructions.begin(),
                value_scene_image.closure_instructions.end(),
                [](const auto &instruction) noexcept {
                    return psycles::compiler::surface_closure_instruction_kind(
                               instruction) ==
                               psycles::compiler::
                                   SurfaceClosureInstructionKind::mix_left;
                }));
        const auto right_only_closure_mixes =
            static_cast<std::size_t>(std::count_if(
                value_scene_image.closure_instructions.begin(),
                value_scene_image.closure_instructions.end(),
                [](const auto &instruction) noexcept {
                    return psycles::compiler::surface_closure_instruction_kind(
                               instruction) ==
                           psycles::compiler::
                               SurfaceClosureInstructionKind::mix_right;
                }));
        if (linear_mix_factor_evaluations !=
                closure_control.incremental_mix_factor_evaluations ||
            linear_mix_factor_evaluations !=
                binary_closure_mixes + left_only_closure_mixes +
                    right_only_closure_mixes) {
            std::cerr << "linear closure weights are not in bijection with "
                         "the formally contributing Mix nodes\n";
            return EXIT_FAILURE;
        }
        std::map<ValueOperation, std::size_t> value_variants_by_operation;
        for (const auto &variant : value_executable_scene.variants) {
            ++value_variants_by_operation[variant.instruction.operation];
        }
        for (const auto features :
             value_scene_image.closure_principled_features) {
            if (features != 0u) {
                ++principled_feature_masks[features];
            }
        }
        for (const auto &instruction :
             value_scene_image.closure_instructions) {
            if (psycles::compiler::surface_closure_is_leaf(instruction)) {
                ++closure_bytecode_operations
                    [psycles::compiler::surface_closure_operation(
                        instruction)];
            }
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
        const auto value_executable_scene_image_bytes =
            value_scene_image.programs.size() *
                sizeof(psycles::compiler::SurfaceValueProgramDescriptor) +
            value_scene_image.instructions.size() *
                sizeof(psycles::compiler::SurfaceValueBytecodeInstruction) +
            value_scene_image.operands.size() * sizeof(std::uint32_t) +
            value_scene_image.metadata.size() *
                sizeof(psycles::compiler::SurfaceValueBytecodeMetadata) +
            value_scene_image.static_data.size() * sizeof(float) +
            value_executable_scene.instruction_variants.size() *
                sizeof(std::uint32_t);
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
            << "\nexpanded_values " << expanded_values
            << "\nexpanded_bump_nodes " << expanded_bump_nodes
            << "\nexpanded_bump_sampled_instructions "
            << expanded_bump_sampled_instructions
            << "\nunique_closures " << unique_closures
            << "\nreachable_closures " << reachable_closures
            << "\nmaterial_execution_classes "
            << material_execution_classes.size()
            << "\nexecution_class_reachable_leaves "
            << execution_class_reachable_leaves
            << "\nexecution_class_reachable_mixes "
            << execution_class_reachable_mixes
            << "\nexecution_class_reachable_adds "
            << execution_class_reachable_adds
            << "\nmaterial_reachable_mixes "
            << material_reachable_mixes
            << "\nmaterial_reachable_adds "
            << material_reachable_adds
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
            << "\nvalue_bytecode_inline_operand_instructions "
            << value_bytecode_inline_operand_instructions
            << "\nvalue_bytecode_inline_operands "
            << value_bytecode_inline_operands
            << "\nvalue_bytecode_packed_operand_loads "
            << value_bytecode_packed_operand_loads
            << "\nvalue_operand_references "
            << value_operand_routes.references
            << "\nvalue_operand_parameter_references "
            << value_operand_routes.parameter_references
            << "\nvalue_operand_local_references "
            << value_operand_routes.local_references
            << "\nvalue_operand_zero_arity_instructions "
            << value_operand_routes.zero_arity_instructions
            << "\nvalue_operand_all_parameter_instructions "
            << value_operand_routes.all_parameter_instructions
            << "\nvalue_operand_all_local_instructions "
            << value_operand_routes.all_local_instructions
            << "\nvalue_operand_mixed_source_instructions "
            << value_operand_routes.mixed_source_instructions
            << "\nvalue_operand_variant_route_pairs "
            << value_operand_routes.variant_route_pairs
            << "\nvalue_operand_route_polymorphic_variants "
            << value_operand_routes.polymorphic_variants
            << "\nmaximum_value_operand_routes_per_variant "
            << value_operand_routes.maximum_routes_per_variant
            << "\nvalue_operand_statically_routed_references "
            << value_operand_routes.statically_routed_references
            << "\nvalue_operand_dynamically_routed_references "
            << value_operand_routes.dynamically_routed_references
            << "\nvalue_operand_statically_parameter_references "
            << value_operand_routes.statically_parameter_references
            << "\nvalue_operand_statically_local_references "
            << value_operand_routes.statically_local_references
            << "\nreachable_image_producers "
            << reachable_image_producers.producers
            << "\nreachable_image_output_instructions "
            << reachable_image_producers.output_instructions
            << "\nreachable_dual_image_producers "
            << reachable_image_producers.dual_output_producers
            << "\nreachable_exact_fusable_image_pairs "
            << reachable_image_producers.exact_fusable_pairs
            << "\npreparation_image_producers "
            << preparation_image_producers.producers
            << "\npreparation_image_output_instructions "
            << preparation_image_producers.output_instructions
            << "\npreparation_dual_image_producers "
            << preparation_image_producers.dual_output_producers
            << "\npreparation_exact_fusable_image_pairs "
            << preparation_image_producers.exact_fusable_pairs
            << "\npreparation_exact_fusable_box_pairs "
            << preparation_image_producers.exact_fusable_box_pairs
            << "\npreparation_exact_fusable_environment_pairs "
            << preparation_image_producers.exact_fusable_environment_pairs
            << "\nreachable_vector_math_producers "
            << reachable_vector_math_producers.producers
            << "\nreachable_vector_math_output_instructions "
            << reachable_vector_math_producers.output_instructions
            << "\nreachable_dual_vector_math_producers "
            << reachable_vector_math_producers.dual_output_producers
            << "\nreachable_exact_fusable_vector_math_pairs "
            << reachable_vector_math_producers.exact_fusable_pairs
            << "\nphysical_vector_math_producers "
            << physical_vector_math_producers.producers
            << "\nphysical_vector_math_output_instructions "
            << physical_vector_math_producers.output_instructions
            << "\nphysical_dual_vector_math_producers "
            << physical_vector_math_producers.dual_output_producers
            << "\nphysical_exact_fusable_vector_math_pairs "
            << physical_vector_math_producers.exact_fusable_pairs
            << "\nemission_vector_math_producers "
            << emission_vector_math_producers.producers
            << "\nemission_vector_math_output_instructions "
            << emission_vector_math_producers.output_instructions
            << "\nemission_dual_vector_math_producers "
            << emission_vector_math_producers.dual_output_producers
            << "\nemission_exact_fusable_vector_math_pairs "
            << emission_vector_math_producers.exact_fusable_pairs
            << "\npreparation_vector_math_producers "
            << preparation_vector_math_producers.producers
            << "\npreparation_vector_math_output_instructions "
            << preparation_vector_math_producers.output_instructions
            << "\npreparation_dual_vector_math_producers "
            << preparation_vector_math_producers.dual_output_producers
            << "\npreparation_exact_fusable_vector_math_pairs "
            << preparation_vector_math_producers.exact_fusable_pairs
            << "\nreachable_closure_adds "
            << closure_control.reachable_adds
            << "\nreachable_closure_mixes "
            << closure_control.reachable_mixes
            << "\nlive_two_sided_closure_mixes "
            << closure_control.live_two_sided_mixes
            << "\nlive_one_sided_closure_mixes "
            << closure_control.live_one_sided_mixes
            << "\nincremental_closure_mix_factor_evaluations "
            << closure_control.incremental_mix_factor_evaluations
            << "\nlinear_closure_mix_factor_evaluations "
            << linear_mix_factor_evaluations
            << "\nbinary_closure_mixes "
            << binary_closure_mixes
            << "\nleft_only_closure_mixes "
            << left_only_closure_mixes
            << "\nright_only_closure_mixes "
            << right_only_closure_mixes
            << "\nclosure_bytecode_instructions "
            << value_scene_image.closure_instructions.size()
            << "\nmaximum_closure_mix_depth "
            << value_scene_image.maximum_closure_mix_depth
            << "\nmaximum_closure_mix_slots "
            << value_scene_image.maximum_closure_mix_slots
            << "\nvalue_scene_descriptor_bytes "
            << value_scene_descriptor_bytes
            << "\nvalue_scene_total_bytes "
            << value_scene_total_bytes
            << "\nvalue_executable_scene_programs "
            << value_scene_image.programs.size()
            << "\nvalue_executable_scene_instructions "
            << value_scene_image.instructions.size()
            << "\nvalue_executable_scene_variants "
            << value_executable_scene.variants.size()
            << "\nvalue_executable_scene_image_bytes "
            << value_executable_scene_image_bytes
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
             preparation_runtime_operations) {
            std::cout << "preparation_runtime_opcode "
                      << static_cast<unsigned>(operation)
                      << ' ' << count << '\n';
        }
        for (const auto &[operation, count] :
             value_variants_by_operation) {
            std::cout << "value_variant "
                      << static_cast<unsigned>(operation)
                      << ' ' << count << '\n';
        }
        for (const auto &[features, count] : principled_feature_masks) {
            std::cout << "principled_feature_mask " << features << ' '
                      << count << '\n';
        }
        for (const auto &[operation, count] :
             closure_bytecode_operations) {
            std::cout << "closure_bytecode_opcode "
                      << static_cast<unsigned>(operation)
                      << ' ' << count << '\n';
        }
        for (const auto &[operation, count] :
             material_reachable_closure_operations) {
            std::cout << "material_reachable_closure_opcode "
                      << static_cast<unsigned>(operation)
                      << ' ' << count << '\n';
        }
        for (const auto &[features, count] :
             material_principled_feature_masks) {
            std::cout << "material_principled_feature_mask " << features
                      << ' ' << count << '\n';
        }
        for (auto index = std::size_t{0u};
             index < value_executable_scene.variants.size(); ++index) {
            const auto &variant = value_executable_scene.variants[index];
            std::cout << "value_variant_detail " << index << ' '
                      << static_cast<unsigned>(
                             variant.instruction.operation)
                      << ' '
                      << static_cast<unsigned>(
                             variant.instruction.result_type)
                      << ' ' << variant.instruction.static_u0 << ' '
                      << variant.instruction.static_u1 << ' '
                      << std::bit_cast<std::uint32_t>(
                             variant.instruction.static_f0)
                      << ' '
                      << std::bit_cast<std::uint32_t>(
                             variant.instruction.static_f1)
                      << ' ' << variant.instruction.static_table.size()
                      << ' ' << variant.svm_immediates.size();
            for (const auto type : variant.operand_types) {
                auto bank =
                    psycles::compiler::SurfaceValueBank::scalar;
                const auto valid =
                    psycles::compiler::classify_surface_value_type(
                        type, bank);
                std::cout << ' '
                          << (valid
                                  ? static_cast<unsigned>(bank)
                                  : 3u + static_cast<unsigned>(type));
            }
            std::cout << '\n';
        }
        for (auto variant = std::size_t{};
             variant < value_operand_routes.routes_by_variant.size();
             ++variant) {
            const auto &routes =
                value_operand_routes.routes_by_variant[variant];
            std::cout << "value_variant_operand_routes " << variant << ' '
                      << routes.size();
            for (const auto route : routes) {
                std::cout << ' ' << route;
            }
            std::cout << '\n';
        }
        return EXIT_SUCCESS;
    }
    std::cerr << "material not found: " << requested << '\n';
    return EXIT_FAILURE;
}

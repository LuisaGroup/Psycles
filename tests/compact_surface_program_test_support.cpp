#include "compact_surface_program_test_support.h"

#include <psycles/compiler/core_nodes.h>

#include "path_tracer_internal.h"
#include "path_tracer_surface_execution_domain.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace psycles::test_support {
namespace {

using namespace compiler;
using namespace contract;
using namespace luisa_backend::detail;

[[nodiscard]] bool domain_matches(const SurfaceValueProgramDomainView &view,
                                  const std::vector<std::uint32_t> &values,
                                  const std::vector<std::uint32_t> &heights,
                                  std::uint32_t offset) noexcept {
    return std::equal(view.value_variants.begin(), view.value_variants.end(),
                      values.begin(), values.end()) &&
           std::equal(view.height_variants.begin(), view.height_variants.end(),
                      heights.begin(), heights.end()) &&
           view.program_offset == offset;
}

[[nodiscard]] std::size_t
count_bump_configuration(const SurfaceValueSceneImage &image,
                         std::uint32_t program_begin, std::uint32_t program_end,
                         std::uint16_t configuration) noexcept {
    auto count = std::size_t{0u};
    for (auto program = program_begin; program < program_end; ++program) {
        const auto &range = image.programs[program];
        for (auto instruction = range.instruction_begin;
             instruction < range.instruction_begin + range.instruction_count;
             ++instruction) {
            const auto &record = image.instructions[instruction];
            if (is_surface_value_surface_normal_transition(record)) {
                continue;
            }
            count += surface_value_operation(record) == ValueOperation::bump &&
                     surface_value_svm_immediate(record) == configuration;
        }
    }
    return count;
}

} // namespace

ShaderGraph make_minimal_principled_graph() {
    ShaderGraph graph;
    const auto principled =
        graph.add_node(node_type::principled_bsdf, "Minimal Principled");
    const auto configured =
        graph.set_input(
            principled, "BaseColor",
            SocketValue::color({0.18f, 0.42f, 0.73f})) &&
        graph.set_input(
            principled, "Roughness", SocketValue::floating(0.31f)) &&
        graph.set_input(
            principled, "Metallic", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "TransmissionWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "SubsurfaceWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "SheenWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "CoatWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "Alpha", SocketValue::floating(1.0f)) &&
        graph.set_input(
            principled, "EmissionStrength", SocketValue::floating(0.0f));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure minimal Principled graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

ShaderGraph make_sampled_color_ramp_graph(
    std::string table, bool shifted_parameter) {
    ShaderGraph graph;
    NodeId factor_node{};
    if (shifted_parameter) {
        factor_node = graph.add_node(
            node_type::add_float,
            "Color Ramp parameter shift");
    }
    const auto ramp = graph.add_node(
        node_type::color_ramp,
        "Late-bound sampled Color Ramp");
    const auto emission = graph.add_node(
        node_type::emission,
        "Color Ramp emission");
    auto configured =
        graph.set_property(
            ramp, "Sampled", SocketValue::boolean(true)) &&
        graph.set_property(
            ramp, "Table", SocketValue::string(std::move(table))) &&
        graph.set_input(
            emission, "Strength", SocketValue::floating(1.0f)) &&
        graph.connect(
            {.node = ramp, .socket = "Color"}, emission, "Color");
    if (shifted_parameter) {
        configured = configured &&
                     graph.set_input(
                         factor_node, "A", SocketValue::floating(0.17f)) &&
                     graph.set_input(
                         factor_node, "B", SocketValue::floating(0.29f)) &&
                     graph.connect(
                         {.node = factor_node, .socket = "Value"},
                         ramp,
                         "Factor");
    } else {
        configured = configured &&
                     graph.set_input(
                         ramp, "Factor", SocketValue::floating(0.46f));
    }
    if (!configured) {
        throw std::runtime_error{
            "failed to configure sampled Color Ramp graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

ShaderGraph make_typed_clamp_graph() {
    ShaderGraph graph;
    const auto geometry = graph.add_node(
        node_type::geometry,
        "Clamp source geometry");
    const auto minmax = graph.add_node(
        node_type::clamp_range,
        "MINMAX clamp");
    const auto range = graph.add_node(
        node_type::clamp_range,
        "RANGE clamp");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Typed-clamp Principled");
    const auto configured =
        graph.set_property(
            minmax, "Mode", SocketValue::string("MINMAX")) &&
        graph.set_input(
            minmax, "Min", SocketValue::floating(0.12f)) &&
        graph.set_input(
            minmax, "Max", SocketValue::floating(0.68f)) &&
        graph.set_property(
            range, "Mode", SocketValue::string("RANGE")) &&
        graph.set_input(
            range, "Min", SocketValue::floating(0.61f)) &&
        graph.set_input(
            range, "Max", SocketValue::floating(0.17f)) &&
        graph.set_input(
            principled, "BaseColor",
            SocketValue::color({0.36f, 0.19f, 0.73f})) &&
        graph.set_input(
            principled, "TransmissionWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "SubsurfaceWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "SheenWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "CoatWeight", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "Alpha", SocketValue::floating(1.0f)) &&
        graph.set_input(
            principled, "EmissionStrength", SocketValue::floating(0.0f)) &&
        graph.connect(
            {.node = geometry, .socket = "Backfacing"}, minmax, "Value") &&
        graph.connect(
            {.node = geometry, .socket = "Backfacing"}, range, "Value") &&
        graph.connect(
            {.node = minmax, .socket = "Result"},
            principled, "Roughness") &&
        graph.connect(
            {.node = range, .socket = "Result"},
            principled, "Metallic");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure typed Clamp graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

namespace {

[[nodiscard]] ShaderGraph make_typed_map_range_graph(
    bool vector_mode,
    std::string interpolation,
    bool clamp_result) {
    ShaderGraph graph;
    const auto geometry = graph.add_node(
        node_type::geometry,
        vector_mode ? "Vector Map Range geometry"
                    : "Scalar Map Range geometry");
    const auto map_range = graph.add_node(
        node_type::map_range,
        vector_mode ? "Typed Vector Map Range"
                    : "Typed Scalar Map Range");
    const auto convert = graph.add_node(
        vector_mode ? node_type::vector_to_color
                    : node_type::scalar_to_color,
        "Map Range result color");
    const auto emission = graph.add_node(
        node_type::emission,
        "Map Range emission");
    auto configured =
        graph.set_property(
            map_range,
            "DataType",
            SocketValue::string(vector_mode ? "FLOAT_VECTOR" : "FLOAT")) &&
        graph.set_property(
            map_range,
            "Interpolation",
            SocketValue::string(std::move(interpolation))) &&
        graph.set_property(
            map_range,
            "Clamp",
            SocketValue::boolean(clamp_result));
    if (vector_mode) {
        configured = configured &&
                     graph.set_input(
                         map_range,
                         "FromMinVector",
                         SocketValue::vector({-0.2f, 0.1f, 0.4f})) &&
                     graph.set_input(
                         map_range,
                         "FromMaxVector",
                         SocketValue::vector({0.6f, 0.9f, -0.4f})) &&
                     graph.set_input(
                         map_range,
                         "ToMinVector",
                         SocketValue::vector({0.15f, 0.7f, 0.2f})) &&
                     graph.set_input(
                         map_range,
                         "ToMaxVector",
                         SocketValue::vector({0.75f, 0.1f, 0.9f})) &&
                     graph.set_input(
                         map_range,
                         "StepsVector",
                         SocketValue::vector({2.0f, 3.0f, 4.0f})) &&
                     graph.connect(
                         {.node = geometry, .socket = "Incoming"},
                         map_range,
                         "Vector") &&
                     graph.connect(
                         {.node = map_range, .socket = "Vector"},
                         convert,
                         "Vector");
    } else {
        configured = configured &&
                     graph.set_input(
                         map_range,
                         "FromMin",
                         SocketValue::floating(0.25f)) &&
                     graph.set_input(
                         map_range,
                         "FromMax",
                         SocketValue::floating(0.75f)) &&
                     graph.set_input(
                         map_range,
                         "ToMin",
                         SocketValue::floating(0.2f)) &&
                     graph.set_input(
                         map_range,
                         "ToMax",
                         SocketValue::floating(0.8f)) &&
                     graph.set_input(
                         map_range,
                         "Steps",
                         SocketValue::floating(3.0f)) &&
                     graph.connect(
                         {.node = geometry, .socket = "Backfacing"},
                         map_range,
                         "Value") &&
                     graph.connect(
                         {.node = map_range, .socket = "Result"},
                         convert,
                         "Value");
    }
    configured = configured &&
                 graph.connect(
                     {.node = convert, .socket = "Color"},
                     emission,
                     "Color");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure typed Map Range graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

} // namespace

std::vector<ShaderGraph> make_typed_map_range_graphs() {
    constexpr std::array interpolations{
        "LINEAR", "STEPPED", "SMOOTHSTEP", "SMOOTHERSTEP"};
    std::vector<ShaderGraph> result;
    result.reserve(interpolations.size() * 4u);
    for (const auto *interpolation : interpolations) {
        for (const auto clamp_result : {false, true}) {
            result.emplace_back(make_typed_map_range_graph(
                false, interpolation, clamp_result));
            result.emplace_back(make_typed_map_range_graph(
                true, interpolation, clamp_result));
        }
    }
    return result;
}

bool has_typed_clamp_record_domain(
    const SurfaceValueRuntime &runtime) noexcept {
    const SurfaceValueStaticVariant *clamp_variant = nullptr;
    constexpr auto expected_key = make_surface_value_handler_key(
        ValueOperation::clamp_range,
        SurfaceValueBank::scalar,
        0u);
    for (const auto &variant : runtime.executable.executable.variants) {
        if (variant.instruction.operation != ValueOperation::clamp_range) {
            continue;
        }
        if (clamp_variant != nullptr) {
            return false;
        }
        auto bank = SurfaceValueBank::scalar;
        if (!classify_surface_value_type(
                variant.instruction.result_type, bank) ||
            variant.svm_immediates.empty() ||
            make_surface_value_handler_key(
                variant.instruction.operation,
                bank,
                variant.svm_immediates.front()) != expected_key) {
            return false;
        }
        clamp_variant = &variant;
    }
    return clamp_variant != nullptr &&
           clamp_variant->instruction.static_u0 == 0u &&
           clamp_variant->instruction.static_u1 == 0u &&
           clamp_variant->svm_immediates ==
               std::vector<std::uint16_t>{0u, 1u};
}

bool has_typed_map_range_record_domains(
    const SurfaceValueRuntime &runtime) noexcept {
    constexpr std::array expected_immediates{
        std::uint16_t{0u}, std::uint16_t{1u},
        std::uint16_t{2u}, std::uint16_t{3u},
        std::uint16_t{4u}, std::uint16_t{5u},
        std::uint16_t{6u}, std::uint16_t{7u}};
    auto scalar_count = std::size_t{0u};
    auto vector_count = std::size_t{0u};
    for (const auto &variant : runtime.executable.executable.variants) {
        if (variant.instruction.operation != ValueOperation::map_range_float &&
            variant.instruction.operation != ValueOperation::map_range_vector) {
            continue;
        }
        if (variant.instruction.static_u0 != 0u ||
            variant.instruction.static_u1 != 0u ||
            !std::equal(
                variant.svm_immediates.begin(),
                variant.svm_immediates.end(),
                expected_immediates.begin(),
                expected_immediates.end())) {
            return false;
        }
        if (variant.instruction.operation == ValueOperation::map_range_float) {
            ++scalar_count;
        } else {
            ++vector_count;
        }
    }
    return scalar_count == 1u && vector_count == 1u;
}

bool has_color_ramp_record_product(
    const SurfaceValueRuntime &runtime) noexcept {
    const auto &executable = runtime.executable.executable;
    const auto variant_count = std::count_if(
        executable.variants.begin(), executable.variants.end(),
        [](const auto &variant) noexcept {
            return variant.instruction.operation == ValueOperation::color_ramp;
        });
    std::vector<std::uint32_t> parameters;
    for (const auto &instruction : executable.values.instructions) {
        if (surface_value_operation(instruction) !=
            ValueOperation::color_ramp) {
            continue;
        }
        if (instruction.metadata_index >=
            executable.values.metadata.size()) {
            return false;
        }
        const auto parameter =
            executable.values.metadata[instruction.metadata_index].parameter;
        if (parameter == ~std::uint32_t{0u}) {
            return false;
        }
        if (std::find(parameters.begin(), parameters.end(), parameter) ==
            parameters.end()) {
            parameters.emplace_back(parameter);
        }
    }
    return variant_count == 1u && parameters.size() >= 2u;
}

CompactSurfaceProgramEvidence
inspect_compact_surface_program(const SurfaceValueRuntime &runtime) noexcept {
    CompactSurfaceProgramEvidence result;
    const auto preparation = surface_value_program_domain(
        runtime, SurfaceValueProgramDomain::preparation);
    const auto emission = surface_value_program_domain(
        runtime, SurfaceValueProgramDomain::emission);
    const auto bssrdf = surface_value_program_domain(
        runtime, SurfaceValueProgramDomain::bssrdf);
    result.domains_match =
        domain_matches(preparation, runtime.preparation_value_static_variants,
                       runtime.height_value_static_variants,
                       SurfaceValueRuntime::preparation_program_offset) &&
        domain_matches(emission, runtime.emission_value_static_variants,
                       runtime.emission_height_value_static_variants,
                       SurfaceValueRuntime::emission_program_offset) &&
        domain_matches(bssrdf, runtime.bssrdf_value_static_variants,
                       runtime.bssrdf_height_value_static_variants,
                       SurfaceValueRuntime::preparation_program_offset);

    const auto &executable = runtime.executable;
    const auto &scene = executable.executable;
    const auto &image = scene.values;
    const auto root_end = executable.root_program_count;
    const auto all_end = static_cast<std::uint32_t>(image.programs.size());

    // A normal transaction has one consuming boundary in a root program and
    // none in any Bump-height subprogram. The invalid entries are not optional
    // metadata: they prove the variant and Bump-call side streams remain
    // parallel without assigning executable semantics to the boundary.
    auto transactions_exact =
        image.programs.size() >= root_end &&
        executable.bump_height_programs.size() == image.instructions.size() &&
        scene.instruction_variants.size() == image.instructions.size();
    for (auto program = std::uint32_t{0u};
         transactions_exact && program < all_end; ++program) {
        const auto &range = image.programs[program];
        auto transition_count = std::uint32_t{0u};
        for (auto offset = std::uint32_t{0u}; offset < range.instruction_count;
             ++offset) {
            const auto instruction = range.instruction_begin + offset;
            if (!is_surface_value_surface_normal_transition(
                    image.instructions[instruction])) {
                continue;
            }
            ++transition_count;
            transactions_exact &=
                scene.instruction_variants[instruction] ==
                    SurfaceValueAddress::invalid_value &&
                executable.bump_height_programs[instruction] ==
                    SurfaceValueAddress::invalid_value;
        }
        const auto known_flags =
            (range.flags & ~surface_value_program_flag_mask) == 0u;
        const auto flags_require_transition =
            range.flags == 0u || transition_count == 1u;
        const auto height_has_no_transition =
            program < root_end || transition_count == 0u;
        transactions_exact &= transition_count <= 1u && known_flags &&
                              flags_require_transition &&
                              height_has_no_transition;
    }
    for (auto topology = std::size_t{0u};
         transactions_exact && topology < runtime.topologies.size();
         ++topology) {
        const auto program = static_cast<std::uint32_t>(
            topology * SurfaceValueRuntime::programs_per_topology +
            SurfaceValueRuntime::preparation_program_offset);
        const auto &range = image.programs[program];
        const auto transition_count =
            std::count_if(image.instructions.begin() + range.instruction_begin,
                          image.instructions.begin() + range.instruction_begin +
                              range.instruction_count,
                          is_surface_value_surface_normal_transition);
        transactions_exact &=
            (transition_count == 1u) ==
            runtime.topologies[topology].program->surface_normal_root().valid();
    }
    result.normal_transactions_exact = transactions_exact;
    const auto bump = std::find_if(
        scene.variants.begin(), scene.variants.end(), [](const auto &variant) {
            return variant.instruction.operation == ValueOperation::bump;
        });
    if (bump == scene.variants.end()) {
        return result;
    }
    result.bump_variant =
        static_cast<std::uint32_t>(std::distance(scene.variants.begin(), bump));

    const auto contains = [variant = result.bump_variant](
                              const auto &domain) noexcept {
        return std::find(domain.begin(), domain.end(), variant) != domain.end();
    };
    const auto targets_are_height_programs = [&] {
        for (auto instruction = std::size_t{0u};
             instruction < image.instructions.size(); ++instruction) {
            if (surface_value_operation(image.instructions[instruction]) !=
                ValueOperation::bump) {
                continue;
            }
            const auto target = executable.bump_height_programs[instruction];
            if (target < root_end || target >= all_end) {
                return false;
            }
        }
        return true;
    }();

    result.bump_partition_exact =
        executable.maximum_bump_depth == 2u &&
        contains(runtime.preparation_value_static_variants) &&
        contains(runtime.height_value_static_variants) &&
        count_bump_configuration(image, 0u, root_end, 1u) != 0u &&
        count_bump_configuration(image, root_end, all_end, 1u) == 0u &&
        count_bump_configuration(image, 0u, root_end, 0u) != 0u &&
        count_bump_configuration(image, root_end, all_end, 0u) != 0u &&
        targets_are_height_programs;
    return result;
}

} // namespace psycles::test_support

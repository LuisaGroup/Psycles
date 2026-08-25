#include "compact_surface_program_test_support.h"

#include <psycles/compiler/core_nodes.h>

#include "path_tracer_internal.h"
#include "path_tracer_surface_execution_domain.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace psycles::test_support {
namespace {

using namespace compiler;
using namespace contract;
using namespace luisa_backend::detail;

[[nodiscard]] bool domain_matches(
    const SurfaceValueProgramDomainView &view,
    const std::vector<std::uint32_t> &values,
    const std::vector<std::uint32_t> &normals,
    const std::vector<std::uint32_t> &heights,
    std::uint32_t offset,
    bool conditional_normal) noexcept {
    return std::equal(view.value_variants.begin(),
                      view.value_variants.end(), values.begin(), values.end()) &&
           std::equal(view.normal_variants.begin(),
                      view.normal_variants.end(), normals.begin(), normals.end()) &&
           std::equal(view.height_variants.begin(),
                      view.height_variants.end(), heights.begin(), heights.end()) &&
           view.program_offset == offset &&
           view.automatic_normal_is_conditional == conditional_normal;
}

[[nodiscard]] std::size_t count_bump_configuration(
    const SurfaceValueSceneImage &image,
    std::uint32_t program_begin,
    std::uint32_t program_end,
    std::uint16_t configuration) noexcept {
    auto count = std::size_t{0u};
    for (auto program = program_begin; program < program_end; ++program) {
        const auto &range = image.programs[program];
        for (auto instruction = range.instruction_begin;
             instruction < range.instruction_begin + range.instruction_count;
             ++instruction) {
            const auto &record = image.instructions[instruction];
            count += surface_value_operation(record) == ValueOperation::bump &&
                     surface_value_svm_immediate(record) == configuration;
        }
    }
    return count;
}

} // namespace

ShaderGraph make_ambiguous_clamp_graph() {
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
        "Ambiguous-clamp Principled");
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
            "failed to configure ambiguous Clamp graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

bool has_ambiguous_clamp_handler_fiber(
    const SurfaceValueRuntime &runtime) noexcept {
    auto count = std::size_t{0u};
    constexpr auto expected_key = make_surface_value_handler_key(
        ValueOperation::clamp_range,
        SurfaceValueBank::scalar,
        0u);
    for (const auto &variant : runtime.executable.executable.variants) {
        if (variant.instruction.operation != ValueOperation::clamp_range) {
            continue;
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
        ++count;
    }
    return count == 2u;
}

CompactSurfaceProgramEvidence inspect_compact_surface_program(
    const SurfaceValueRuntime &runtime) noexcept {
    CompactSurfaceProgramEvidence result;
    const auto preparation = surface_value_program_domain(
        runtime, SurfaceValueProgramDomain::preparation);
    const auto emission = surface_value_program_domain(
        runtime, SurfaceValueProgramDomain::emission);
    const auto bssrdf = surface_value_program_domain(
        runtime, SurfaceValueProgramDomain::bssrdf);
    result.domains_match =
        domain_matches(preparation,
                       runtime.preparation_value_static_variants,
                       runtime.normal_value_static_variants,
                       runtime.height_value_static_variants,
                       SurfaceValueRuntime::preparation_program_offset,
                       false) &&
        domain_matches(emission,
                       runtime.emission_value_static_variants,
                       runtime.emission_normal_value_static_variants,
                       runtime.emission_height_value_static_variants,
                       SurfaceValueRuntime::emission_program_offset,
                       true) &&
        domain_matches(bssrdf,
                       runtime.bssrdf_value_static_variants,
                       runtime.bssrdf_normal_value_static_variants,
                       runtime.bssrdf_height_value_static_variants,
                       SurfaceValueRuntime::preparation_program_offset,
                       false);

    const auto &executable = runtime.executable;
    const auto &scene = executable.executable;
    const auto bump = std::find_if(
        scene.variants.begin(), scene.variants.end(), [](const auto &variant) {
            return variant.instruction.operation == ValueOperation::bump;
        });
    if (bump == scene.variants.end()) {
        return result;
    }
    result.bump_variant = static_cast<std::uint32_t>(
        std::distance(scene.variants.begin(), bump));

    const auto contains = [variant = result.bump_variant](
                              const auto &domain) noexcept {
        return std::find(domain.begin(), domain.end(), variant) != domain.end();
    };
    const auto &image = scene.values;
    const auto root_end = executable.root_program_count;
    const auto all_end = static_cast<std::uint32_t>(image.programs.size());
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

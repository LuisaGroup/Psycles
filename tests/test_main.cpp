#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/contract/render.h>
#include <psycles/contract/scene.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using namespace psycles;
using namespace psycles::adapter;
using namespace psycles::compiler;
using namespace psycles::contract;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] ShaderGraph make_diffuse_graph(Vec3f color) {
    ShaderGraph graph;
    const auto color_node = graph.add_node(node_type::constant_color, "Base Color");
    const auto diffuse_node = graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    expect(
        graph.set_input(color_node, "Color", SocketValue::color(color)),
        "failed to set color");
    expect(
        graph.connect(
            {.node = color_node, .socket = "Color"},
            diffuse_node,
            "Color"),
        "failed to connect color");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse_node, .socket = "Closure"});
    return graph;
}

void test_shader_graph_and_invalidation() {
    ShaderCompiler compiler{make_core_node_registry()};

    auto red = compiler.compile(make_diffuse_graph({1.0f, 0.0f, 0.0f}));
    auto blue = compiler.compile(make_diffuse_graph({0.0f, 0.0f, 1.0f}));

    expect(red.ok(), "valid diffuse graph failed to compile");
    expect(blue.ok(), "second valid diffuse graph failed to compile");
    expect(
        red.program->analysis().evaluation_order.size() == 2u,
        "reachable evaluation order is incorrect");
    expect(
        (red.program->analysis().required_features &
         feature_bit(ShaderFeature::surface)) != 0u,
        "surface feature was not discovered");
    expect(
        red.program->analysis().structure_signature ==
            blue.program->analysis().structure_signature,
        "parameter-only edit changed the structure signature");
    expect(
        red.program->analysis().parameter_signature !=
            blue.program->analysis().parameter_signature,
        "parameter-only edit did not change the parameter signature");
}

void test_closure_tree_is_preserved() {
    ShaderCompiler compiler{make_core_node_registry()};
    ShaderGraph graph;
    const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    const auto mix = graph.add_node(node_type::mix_closure, "Mix");

    expect(
        graph.connect(
            {.node = diffuse, .socket = "Closure"},
            mix,
            "A"),
        "failed to connect diffuse closure");
    expect(
        graph.connect(
            {.node = emission, .socket = "Closure"},
            mix,
            "B"),
        "failed to connect emission closure");
    expect(
        graph.set_input(mix, "Factor", SocketValue::floating(0.25f)),
        "failed to set closure mix factor");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = mix, .socket = "Closure"});

    auto compiled = compiler.compile(graph);
    expect(compiled.ok(), "closure mix graph failed to compile");
    expect(
        compiled.program->analysis().evaluation_order.size() == 3u,
        "closure composition was collapsed or lost");
    expect(
        (compiled.program->analysis().required_features &
         feature_bit(ShaderFeature::emission)) != 0u,
        "emission feature was not preserved through the closure tree");
}

void test_volume_closure_tree_is_preserved() {
    ShaderGraph graph;
    const auto surface =
        graph.add_node(node_type::diffuse_bsdf, "Surface");
    const auto absorption =
        graph.add_node(
            node_type::volume_absorption,
            "Absorption");
    const auto scatter =
        graph.add_node(node_type::volume_scatter, "Scatter");
    const auto coefficients =
        graph.add_node(
            node_type::volume_coefficients,
            "Coefficients");
    const auto principled =
        graph.add_node(
            node_type::principled_volume,
            "Principled Volume");
    const auto additive =
        graph.add_node(node_type::add_volume, "Add Volumes");
    const auto mixed =
        graph.add_node(node_type::mix_volume, "Mix Volumes");
    const auto root =
        graph.add_node(node_type::add_volume, "Volume Root");

    expect(
        graph.set_input(
            absorption,
            "Density",
            SocketValue::floating(0.25f)),
        "failed to set absorption density");
    expect(
        graph.set_property(
            scatter,
            "Phase",
            SocketValue::string("DRAINE")),
        "failed to set Draine phase");
    expect(
        graph.set_property(
            coefficients,
            "Phase",
            SocketValue::string("MIE")),
        "failed to set Mie phase");
    expect(
        graph.connect(
            {.node = absorption, .socket = "Volume"},
            additive,
            "A") &&
            graph.connect(
                {.node = scatter, .socket = "Volume"},
                additive,
                "B") &&
            graph.connect(
                {.node = coefficients, .socket = "Volume"},
                mixed,
                "A") &&
            graph.connect(
                {.node = principled, .socket = "Volume"},
                mixed,
                "B") &&
            graph.set_input(
                mixed,
                "Factor",
                SocketValue::floating(0.375f)) &&
            graph.connect(
                {.node = additive, .socket = "Volume"},
                root,
                "A") &&
            graph.connect(
                {.node = mixed, .socket = "Volume"},
                root,
                "B"),
        "failed to preserve the volume closure tree");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = surface, .socket = "Closure"});
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{.node = root, .socket = "Volume"});

    ShaderCompiler compiler{make_core_node_registry()};
    const auto shader = compiler.compile(graph);
    expect(shader.ok(), "volume graph failed to compile");
    expect(
        (shader.program->analysis().required_features &
         feature_bit(ShaderFeature::volume)) != 0u,
        "volume feature was not discovered");
}

void test_cycles_normalized_graph_adapter() {
    CyclesNormalizedShaderGraph source;
    source.nodes = {
        {
            .id = 10u,
            .type = "geometry",
            .variant = {},
            .label = "Geometry",
            .inputs = {},
            .properties = {}},
        {
            .id = 11u,
            .type = "rgb",
            .variant = {},
            .label = "Base Color",
            .inputs = {{
                .socket = "Color",
                .source = std::nullopt,
                .value = SocketValue::color({0.8f, 0.4f, 0.2f})}},
            .properties = {}},
        {
            .id = 12u,
            .type = "diffuse_bsdf",
            .variant = {},
            .label = "Diffuse",
            .inputs = {
                {
                    .socket = "Color",
                    .source = CyclesOutputRef{
                        .node = 11u,
                        .socket = "Color"},
                    .value = std::nullopt},
                {
                    .socket = "Roughness",
                    .source = std::nullopt,
                    .value = SocketValue::floating(0.0f)},
                {
                    .socket = "Normal",
                    .source = CyclesOutputRef{
                        .node = 10u,
                        .socket = "Normal"},
                    .value = std::nullopt}},
            .properties = {}},
        {
            .id = 15u,
            .type = "value",
            .variant = {},
            .label = "Emission Scale A",
            .inputs = {{
                .socket = "Value",
                .source = std::nullopt,
                .value = SocketValue::floating(2.0f)}},
            .properties = {}},
        {
            .id = 16u,
            .type = "value",
            .variant = {},
            .label = "Emission Scale B",
            .inputs = {{
                .socket = "Value",
                .source = std::nullopt,
                .value = SocketValue::floating(2.0f)}},
            .properties = {}},
        {
            .id = 17u,
            .type = "math",
            .variant = "multiply",
            .label = "Emission Strength",
            .inputs = {
                {
                    .socket = "Value1",
                    .source = CyclesOutputRef{
                        .node = 15u,
                        .socket = "Value"},
                    .value = std::nullopt},
                {
                    .socket = "Value2",
                    .source = CyclesOutputRef{
                        .node = 16u,
                        .socket = "Value"},
                    .value = std::nullopt}},
            .properties = {}},
        {
            .id = 13u,
            .type = "emission",
            .variant = {},
            .label = "Emission",
            .inputs = {
                {
                    .socket = "Color",
                    .source = std::nullopt,
                    .value = SocketValue::color({0.2f, 0.5f, 1.0f})},
                {
                    .socket = "Strength",
                    .source = CyclesOutputRef{
                        .node = 17u,
                        .socket = "Value"},
                    .value = std::nullopt}},
            .properties = {}},
        {
            .id = 14u,
            .type = "mix_closure",
            .variant = {},
            .label = "Mix",
            .inputs = {
                {
                    .socket = "Fac",
                    .source = std::nullopt,
                    .value = SocketValue::floating(0.25f)},
                {
                    .socket = "Closure1",
                    .source = CyclesOutputRef{
                        .node = 12u,
                        .socket = "BSDF"},
                    .value = std::nullopt},
                {
                    .socket = "Closure2",
                    .source = CyclesOutputRef{
                        .node = 13u,
                        .socket = "Emission"},
                    .value = std::nullopt}},
            .properties = {}}};
    source.set_root(
        ShaderDomain::surface,
        CyclesOutputRef{.node = 14u, .socket = "Closure"});

    auto adapted = adapt_cycles_shader_graph(
        source, make_core_cycles_node_mappings());
    expect(adapted.ok(), "normalized Cycles graph failed to adapt");

    ShaderCompiler shader_compiler{make_core_node_registry()};
    auto shader = shader_compiler.compile(*adapted.graph);
    expect(shader.ok(), "adapted Cycles graph failed validation");
}

void test_cycles_principled_emission_adapter() {
    CyclesNormalizedShaderGraph source;
    source.nodes = {
        {
            .id = 9u,
            .type = "geometry",
            .variant = {},
            .label = "Linked coat normal",
            .inputs = {},
            .properties = {}},
        {
            .id = 10u,
            .type = "principled_bsdf",
            .variant = {},
            .label = "Layered Principled emission",
            .inputs = {
                {
                    .socket = "Alpha",
                    .source = std::nullopt,
                    .value = SocketValue::floating(0.73f)},
                {
                    .socket = "Thin Wall",
                    .source = std::nullopt,
                    .value = SocketValue::boolean(true)},
                {
                    .socket = "Sheen Weight",
                    .source = std::nullopt,
                    .value = SocketValue::floating(0.55f)},
                {
                    .socket = "Sheen Roughness",
                    .source = std::nullopt,
                    .value = SocketValue::floating(0.38f)},
                {
                    .socket = "Sheen Tint",
                    .source = std::nullopt,
                    .value = SocketValue::color(
                        {0.9f, 0.35f, 0.12f})},
                {
                    .socket = "Transmission Weight",
                    .source = std::nullopt,
                    .value = SocketValue::floating(0.42f)},
                {
                    .socket = "Coat Weight",
                    .source = std::nullopt,
                    .value = SocketValue::floating(0.75f)},
                {
                    .socket = "Coat Roughness",
                    .source = std::nullopt,
                    .value = SocketValue::floating(0.23f)},
                {
                    .socket = "Coat IOR",
                    .source = std::nullopt,
                    .value = SocketValue::floating(1.7f)},
                {
                    .socket = "Coat Tint",
                    .source = std::nullopt,
                    .value = SocketValue::color(
                        {0.3f, 0.7f, 0.95f})},
                {
                    .socket = "Coat Normal",
                    .source = CyclesOutputRef{
                        .node = 9u,
                        .socket = "Normal"},
                    .value = std::nullopt},
                {
                    .socket = "Emission Color",
                    .source = std::nullopt,
                    .value = SocketValue::color(
                        {0.2f, 0.4f, 0.8f})},
                {
                    .socket = "Emission Strength",
                    .source = std::nullopt,
                    .value = SocketValue::floating(3.0f)}},
            .properties = {{
                "distribution",
                SocketValue::string("MULTI_GGX")}}}};
    source.set_root(
        ShaderDomain::surface,
        CyclesOutputRef{.node = 10u, .socket = "BSDF"});

    auto adapted = adapt_cycles_shader_graph(
        source, make_core_cycles_node_mappings());
    expect(
        adapted.ok(),
        "normalized Principled emission graph failed to adapt");

    ShaderCompiler shader_compiler{make_core_node_registry()};
    auto shader = shader_compiler.compile(*adapted.graph);
    expect(
        shader.ok(),
        "adapted Principled emission graph failed validation");
}

void test_cycles_adapter_rejects_svm_lowered_graph() {
    CyclesNormalizedShaderGraph source;
    source.stage = CyclesGraphStage::svm_lowered;
    auto adapted = adapt_cycles_shader_graph(
        source, make_core_cycles_node_mappings());
    expect(
        !adapted.ok() &&
            !adapted.diagnostics.empty() &&
            adapted.diagnostics.front().code ==
                CyclesAdapterDiagnosticCode::graph_already_svm_lowered,
        "adapter accepted a graph after SVM closure lowering");
}

}// namespace

int main() {
    try {
        test_shader_graph_and_invalidation();
        test_closure_tree_is_preserved();
        test_volume_closure_tree_is_preserved();
        test_cycles_normalized_graph_adapter();
        test_cycles_principled_emission_adapter();
        test_cycles_adapter_rejects_svm_lowered_graph();
        std::cout << "All Psycles contract tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Psycles test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

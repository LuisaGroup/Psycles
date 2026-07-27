#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/material_library.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/render.h>
#include <psycles/contract/scene.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

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

    auto red_surface = compile_surface_program(*red.program);
    expect(red_surface.ok(), "red surface failed to lower");
    auto blue_binding = bind_surface_parameters(
        *red_surface.program, *blue.program);
    expect(
        blue_binding.ok(),
        "same-topology blue material could not reuse red program");
    const auto color = std::find_if(
        red_surface.program->parameters().begin(),
        red_surface.program->parameters().end(),
        [](const ParameterDesc &parameter) {
            return parameter.socket == "Color";
        });
    expect(
        color != red_surface.program->parameters().end(),
        "lowered color parameter is missing");
    const auto *value = blue_binding.parameters->find(color->id);
    expect(
        value != nullptr &&
            std::get<Vec3f>(value->value) ==
                Vec3f{0.0f, 0.0f, 1.0f},
        "rebound parameter block retained the red color");
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

void test_combine_color_lowers_to_surface_program() {
    ShaderGraph graph;
    const auto combine =
        graph.add_node(node_type::combine_color, "Combine RGB");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    expect(
        graph.set_input(
            combine, "R", SocketValue::floating(0.25f)) &&
            graph.set_input(
                combine, "G", SocketValue::floating(0.5f)) &&
            graph.set_input(
                combine, "B", SocketValue::floating(0.75f)),
        "failed to configure Combine Color");
    expect(
        graph.set_property(
            combine, "Mode", SocketValue::string("HSL")),
        "failed to configure Combine Color mode");
    expect(
        graph.connect(
            {.node = combine, .socket = "Color"},
            diffuse,
            "Color"),
        "failed to connect Combine Color");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});

    ShaderCompiler compiler{make_core_node_registry()};
    auto shader = compiler.compile(graph);
    expect(shader.ok(), "Combine Color graph failed to compile");
    auto surface = compile_surface_program(*shader.program);
    expect(surface.ok(), "Combine Color graph failed to lower");
    expect(
        std::ranges::any_of(
            surface.program->value_instructions(),
            [](const ValueInstruction &instruction) {
                return instruction.operation ==
                           ValueOperation::combine_color &&
                       instruction.static_u0 == 2u;
            }),
        "HSL Combine Color instruction is missing");
}

void test_cycles_color_value_nodes_lower_to_surface_program() {
    ShaderGraph graph;
    const auto source =
        graph.add_node(node_type::constant_color, "Source");
    const auto gamma =
        graph.add_node(node_type::gamma_color, "Gamma");
    const auto bright = graph.add_node(
        node_type::brightness_contrast,
        "Brightness/Contrast");
    const auto gray =
        graph.add_node(node_type::color_to_scalar, "RGB to BW");
    const auto clamp =
        graph.add_node(node_type::clamp_range, "Clamp");
    const auto color =
        graph.add_node(node_type::scalar_to_color, "To Color");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");

    expect(
        graph.connect(
            {.node = source, .socket = "Color"},
            gamma,
            "Color") &&
            graph.connect(
                {.node = gamma, .socket = "Color"},
                bright,
                "Color") &&
            graph.connect(
                {.node = bright, .socket = "Color"},
                gray,
                "Color") &&
            graph.connect(
                {.node = gray, .socket = "Value"},
                clamp,
                "Value") &&
            graph.connect(
                {.node = clamp, .socket = "Result"},
                color,
                "Value") &&
            graph.connect(
                {.node = color, .socket = "Color"},
                diffuse,
                "Color"),
        "failed to connect Cycles color/value test graph");
    expect(
        graph.set_input(
            gamma, "Gamma", SocketValue::floating(2.2f)) &&
            graph.set_input(
                bright, "Bright", SocketValue::floating(0.1f)) &&
            graph.set_input(
                bright,
                "Contrast",
                SocketValue::floating(-0.25f)) &&
            graph.set_input(
                clamp, "Min", SocketValue::floating(0.8f)) &&
            graph.set_input(
                clamp, "Max", SocketValue::floating(0.2f)) &&
            graph.set_property(
                clamp, "Mode", SocketValue::string("RANGE")),
        "failed to configure Cycles color/value test graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});

    ShaderCompiler compiler{make_core_node_registry()};
    auto shader = compiler.compile(graph);
    expect(
        shader.ok(),
        "Cycles color/value graph failed to compile");
    auto surface = compile_surface_program(*shader.program);
    expect(
        surface.ok(),
        "Cycles color/value graph failed to lower");
    const auto &instructions =
        surface.program->value_instructions();
    for (const auto operation : {
             ValueOperation::gamma,
             ValueOperation::brightness_contrast,
             ValueOperation::color_to_scalar}) {
        expect(
            std::ranges::any_of(
                instructions,
                [operation](const ValueInstruction &instruction) {
                    return instruction.operation == operation;
                }),
            "Cycles color/value instruction is missing");
    }
    expect(
        std::ranges::any_of(
            instructions,
            [](const ValueInstruction &instruction) {
                return instruction.operation ==
                           ValueOperation::clamp_range &&
                       instruction.static_u0 == 1u;
            }),
        "range-swapping Clamp instruction is missing");
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

    auto surface = compile_surface_program(*shader.program);
    expect(surface.ok(), "adapted Cycles graph failed surface lowering");
    expect(
        surface.program->closure_instructions().size() == 3u,
        "closure tree did not lower to three semantic instructions");

    expect(
        surface.program->closure_instructions()[0u].operation ==
                ClosureOperation::diffuse &&
            surface.program->closure_instructions()[1u].operation ==
                ClosureOperation::emission &&
            surface.program->closure_instructions()[2u].operation ==
                ClosureOperation::mix,
        "normalized closure tree changed during lowering");

    auto factor = std::find_if(
        surface.program->parameters().begin(),
        surface.program->parameters().end(),
        [](const auto &parameter) {
            return parameter.socket == "Factor";
        });
    expect(
        factor != surface.program->parameters().end(),
        "mix factor did not become a runtime parameter");
    SurfaceParameterBlock parameters{*surface.program};
    expect(
        parameters.set(
            *surface.program,
            factor->id,
            SocketValue::floating(1.0f)),
        "failed to update mix factor parameter");
    expect(
        surface.program->closure_instructions().size() == 3u,
        "parameter update changed closure topology");
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

void test_incremental_material_library() {
    ShaderCompiler shader_compiler{make_core_node_registry()};
    MaterialLibrary library;
    const auto material_id = MaterialId{42u};

    SceneSnapshot red_scene;
    red_scene.revision = 1u;
    red_scene.materials.emplace(
        material_id,
        MaterialDesc{
            .name = "Editable",
            .shader = make_diffuse_graph({1.0f, 0.0f, 0.0f})});
    auto added = library.update(red_scene, shader_compiler);
    expect(added.committed, "initial material library update failed");
    expect(
        added.updates.size() == 1u &&
            added.updates.front().kind ==
                MaterialUpdateKind::added,
        "initial material was not reported as added");
    const auto *red_material = library.find(material_id);
    expect(red_material != nullptr, "compiled red material is missing");
    auto original_program = red_material->surface_program();

    auto blue_scene = red_scene;
    blue_scene.revision = 2u;
    blue_scene.materials.at(material_id).shader =
        make_diffuse_graph({0.0f, 0.0f, 1.0f});
    auto rebound = library.update(blue_scene, shader_compiler);
    expect(rebound.committed, "parameter-only material update failed");
    expect(
        rebound.updates.front().kind ==
            MaterialUpdateKind::parameters_rebound,
        "parameter-only edit was not classified as a rebind");
    const auto *blue_material = library.find(material_id);
    expect(
        blue_material != nullptr &&
            blue_material->surface_program().get() ==
                original_program.get(),
        "parameter-only edit regenerated the surface program");
    const auto color = std::find_if(
        blue_material->surface_program()->parameters().begin(),
        blue_material->surface_program()->parameters().end(),
        [](const ParameterDesc &parameter) {
            return parameter.socket == "Color";
        });
    const auto *blue_value =
        color == blue_material->surface_program()->parameters().end()
            ? nullptr
            : blue_material->parameters().find(color->id);
    expect(
        blue_value != nullptr &&
            std::get<Vec3f>(blue_value->value) ==
                Vec3f{0.0f, 0.0f, 1.0f},
        "material parameter rebind did not reach the parameter block");

    ShaderGraph mixed_graph;
    const auto diffuse =
        mixed_graph.add_node(node_type::diffuse_bsdf);
    const auto emission =
        mixed_graph.add_node(node_type::emission);
    const auto add_node =
        mixed_graph.add_node(node_type::add_closure);
    expect(
        mixed_graph.connect(
            {.node = diffuse, .socket = "Closure"},
            add_node,
            "A"),
        "failed to build structural material edit");
    expect(
        mixed_graph.connect(
            {.node = emission, .socket = "Closure"},
            add_node,
            "B"),
        "failed to build structural material edit");
    mixed_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = add_node, .socket = "Closure"});

    auto mixed_scene = blue_scene;
    mixed_scene.revision = 3u;
    mixed_scene.materials.at(material_id).shader =
        std::move(mixed_graph);
    auto recompiled = library.update(mixed_scene, shader_compiler);
    expect(recompiled.committed, "structural material update failed");
    expect(
        recompiled.updates.front().kind ==
            MaterialUpdateKind::program_recompiled,
        "structural edit was not classified as a recompile");
    const auto *mixed_material = library.find(material_id);
    expect(
        mixed_material != nullptr &&
            mixed_material->surface_program().get() !=
                original_program.get(),
        "structural edit reused an incompatible surface program");
    auto mixed_program = mixed_material->surface_program();

    auto invalid_scene = mixed_scene;
    invalid_scene.revision = 4u;
    ShaderGraph invalid_graph;
    const auto unknown =
        invalid_graph.add_node("cycles.future.unsupported");
    invalid_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = unknown, .socket = "Closure"});
    invalid_scene.materials.at(material_id).shader =
        std::move(invalid_graph);
    auto rejected = library.update(invalid_scene, shader_compiler);
    expect(
        !rejected.committed && !rejected.diagnostics.empty(),
        "invalid material library update was committed");
    expect(
        library.source_revision() == 3u &&
            library.find(material_id)
                    ->surface_program()
                    .get() == mixed_program.get(),
        "rejected material update partially mutated the library");
}

void test_graph_errors_are_explicit() {
    {
        ShaderCompiler compiler{make_core_node_registry()};
        ShaderGraph graph;
        const auto color = graph.add_node(node_type::constant_color);
        graph.set_root(
            ShaderDomain::surface,
            OutputRef{.node = color, .socket = "Color"});
        auto compiled = compiler.compile(graph);
        expect(!compiled.ok(), "invalid surface root type was accepted");
    }

    {
        auto registry = make_core_node_registry();
        expect(
            registry.register_schema(NodeSchema{
                .type = "test.closure.pass",
                .inputs = {{
                    .name = "In",
                    .type = SocketType::closure,
                    .required = true,
                    .default_value = std::nullopt}},
                .outputs = {{
                    .name = "Out",
                    .type = SocketType::closure,
                    .required = false,
                    .default_value = std::nullopt}},
                .properties = {},
                .required_features = {}}),
            "failed to register cycle-test node");

        ShaderCompiler compiler{std::move(registry)};
        ShaderGraph graph;
        const auto a = graph.add_node("test.closure.pass");
        const auto b = graph.add_node("test.closure.pass");
        expect(
            graph.connect({.node = b, .socket = "Out"}, a, "In"),
            "failed to create first cycle edge");
        expect(
            graph.connect({.node = a, .socket = "Out"}, b, "In"),
            "failed to create second cycle edge");
        graph.set_root(
            ShaderDomain::surface,
            OutputRef{.node = a, .socket = "Out"});
        auto compiled = compiler.compile(graph);
        expect(!compiled.ok(), "cyclic graph was accepted");
    }
}

void test_scene_delta_is_atomic() {
    SceneDatabase scene;
    SceneDelta initial{.base_revision = 0u, .commands = {}};

    auto material = MaterialId{1u};
    auto geometry = GeometryId{2u};
    auto instance = InstanceId{3u};
    auto camera = CameraId{4u};

    initial.commands.emplace_back(UpsertMaterial{
        .id = material,
        .value = {
            .name = "Material",
            .shader = make_diffuse_graph({0.8f, 0.8f, 0.8f})}});
    initial.commands.emplace_back(UpsertGeometry{
        .id = geometry,
        .value = {
            .name = "Triangle",
            .positions = {
                {-1.0f, 0.0f, 0.0f},
                {1.0f, 0.0f, 0.0f},
                {0.0f, 1.0f, 0.0f}},
            .normals = {},
            .uv = {},
            .triangles = {{{0u, 1u, 2u}}},
            .material_slots = {material},
            .triangle_material_slots = {0u}}});
    initial.commands.emplace_back(UpsertInstance{
        .id = instance,
        .value = {
            .name = "Triangle Instance",
            .geometry = geometry,
            .transform = {},
            .motion = {},
            .material_overrides = {},
            .visibility_mask = ~std::uint32_t{0u}}});
    initial.commands.emplace_back(UpsertCamera{
        .id = camera,
        .value = {
            .name = "Camera",
            .projection = CameraProjection::perspective,
            .transform = {},
            .field_of_view = 0.78539816339f,
            .orthographic_scale = 1.0f,
            .near_clip = 1.0e-4f,
            .far_clip = 1.0e5f}});
    initial.commands.emplace_back(SetActiveCamera{.id = camera});

    auto first = scene.apply(initial);
    expect(first.committed, "valid initial scene failed to commit");
    expect(scene.snapshot().revision == 1u, "scene revision did not advance");

    SceneDelta invalid{.base_revision = 1u, .commands = {}};
    invalid.commands.emplace_back(RemoveMaterial{.id = material});
    auto rejected = scene.apply(invalid);
    expect(!rejected.committed, "dangling material removal was committed");
    expect(
        scene.snapshot().revision == 1u,
        "rejected scene delta changed the revision");
    expect(
        scene.snapshot().materials.contains(material),
        "rejected scene delta partially mutated the scene");

    SceneDelta stale{.base_revision = 0u, .commands = {}};
    auto stale_result = scene.apply(stale);
    expect(!stale_result.committed, "stale scene delta was committed");
}

}// namespace

int main() {
    try {
        test_shader_graph_and_invalidation();
        test_closure_tree_is_preserved();
        test_combine_color_lowers_to_surface_program();
        test_cycles_color_value_nodes_lower_to_surface_program();
        test_cycles_normalized_graph_adapter();
        test_cycles_adapter_rejects_svm_lowered_graph();
        test_incremental_material_library();
        test_graph_errors_are_explicit();
        test_scene_delta_is_atomic();
        std::cout << "All Psycles contract tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Psycles test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

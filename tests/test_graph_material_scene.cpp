#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/material_library.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
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
    const auto color_node =
        graph.add_node(node_type::constant_color, "Base Color");
    const auto diffuse_node =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    expect(
        graph.set_input(
            color_node, "Color", SocketValue::color(color)),
        "failed to set color");
    expect(
        graph.connect(
            {.node = color_node, .socket = "Color"},
            diffuse_node,
            "Color"),
        "failed to connect color");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = diffuse_node,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] const ParameterDesc *find_property_parameter(
    const SurfaceProgram &program,
    std::string_view property) noexcept {
    const auto found = std::find_if(
        program.parameters().begin(),
        program.parameters().end(),
        [&](const ParameterDesc &parameter) {
            return parameter.source ==
                       ParameterSource::property &&
                   parameter.socket == property;
        });
    return found == program.parameters().end()
               ? nullptr
               : &*found;
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

void test_material_library_compiles_only_the_selected_domain() {
    ShaderCompiler shader_compiler{make_core_node_registry()};
    MaterialLibrary library;
    constexpr MaterialId reachable{1u};
    constexpr MaterialId unreachable{2u};

    SceneSnapshot scene;
    scene.revision = 1u;
    scene.materials.emplace(
        reachable,
        MaterialDesc{
            .name = "reachable",
            .shader = make_diffuse_graph({0.2f, 0.4f, 0.8f})});
    ShaderGraph invalid_graph;
    const auto unknown =
        invalid_graph.add_node("cycles.future.unreachable");
    invalid_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = unknown, .socket = "Closure"});
    scene.materials.emplace(
        unreachable,
        MaterialDesc{
            .name = "unreachable invalid graph",
            .shader = std::move(invalid_graph)});

    const auto selected = library.update(
        scene, shader_compiler, std::set{reachable});
    expect(
        selected.committed && selected.diagnostics.empty() &&
            library.materials().size() == 1u &&
            library.find(reachable) != nullptr &&
            library.find(unreachable) == nullptr,
        "unselected invalid graph entered the compiled material domain");

    scene.revision = 2u;
    const auto rejected = library.update(
        scene, shader_compiler, std::set{unreachable});
    expect(
        !rejected.committed && !rejected.diagnostics.empty() &&
            library.source_revision() == 1u &&
            library.find(reachable) != nullptr &&
            library.find(unreachable) == nullptr,
        "failed selected-domain update partially mutated the library");

    const auto missing = MaterialId{99u};
    const auto missing_root = library.update(
        scene, shader_compiler, std::set{missing});
    expect(
        !missing_root.committed &&
            missing_root.diagnostics.size() == 1u &&
            missing_root.diagnostics.front().material == missing &&
            library.find(reachable) != nullptr,
        "missing material root was not rejected transactionally");

    scene.materials.at(unreachable).shader =
        make_diffuse_graph({0.8f, 0.3f, 0.1f});
    const auto switched = library.update(
        scene, shader_compiler, std::set{unreachable});
    expect(
        switched.committed && library.materials().size() == 1u &&
            library.find(reachable) == nullptr &&
            library.find(unreachable) != nullptr,
        "selected-domain switch did not remove the former root");

    scene.revision = 3u;
    const auto emptied = library.update(
        scene,
        shader_compiler,
        std::set<MaterialId>{});
    expect(
        emptied.committed && library.materials().empty() &&
            library.source_revision() == 3u,
        "empty selected domain did not remove every compiled material");
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

void test_linked_inputs_retain_cycles_socket_defaults() {
    ShaderCompiler compiler{make_core_node_registry()};
    ShaderGraph graph;
    const auto color = graph.add_node(node_type::constant_color, "Color");
    const auto mix = graph.add_node(node_type::mix_color, "Mix Color");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    const auto authored = SocketValue::color({0.25f, 0.5f, 0.75f});

    expect(
        graph.set_input(mix, "A", authored) &&
            graph.connect({color, "Color"}, mix, "A") &&
            graph.connect({color, "Color"}, mix, "B") &&
            graph.connect({mix, "Color"}, emission, "Color"),
        "failed to construct linked socket-default regression graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});

    const auto compiled = compiler.compile(graph);
    expect(compiled.ok(), "linked socket-default graph did not compile");
    const auto *normalized_mix = compiled.program->graph().find(mix);
    expect(normalized_mix != nullptr, "normalized Mix Color node is absent");
    const auto &a = normalized_mix->inputs.at("A");
    const auto &b = normalized_mix->inputs.at("B");
    expect(
        a.source.has_value() && a.value == authored,
        "connect discarded an authored Cycles socket fallback");
    expect(
        b.source.has_value() && b.value.has_value() &&
            b.value->type == SocketType::color &&
            std::get<Vec3f>(b.value->value) == Vec3f{},
        "normalization did not retain the Cycles schema fallback while linked");
}

void test_sampled_color_ramp_is_part_of_the_graph_contract() {
    ShaderCompiler compiler{make_core_node_registry()};
    ShaderGraph graph;
    const auto ramp = graph.add_node(node_type::color_ramp, "Sampled Ramp");
    const auto emission = graph.add_node(node_type::emission, "Emission");

    expect(
        graph.set_input(
            ramp,
            "Factor",
            SocketValue::floating(0.375f)),
        "failed to set sampled ramp factor");
    expect(
        graph.set_property(
            ramp,
            "Sampled",
            SocketValue::boolean(true)),
        "failed to mark Color Ramp as sampled");
    expect(
        graph.set_property(
            ramp,
            "Table",
            SocketValue::string(
                "0,0.1,0.2,0.3,0.4;"
                "0.5,0.5,0.6,0.7,0.8;"
                "1,0.9,1,0.1,0.2")),
        "failed to set sampled Color Ramp table");
    expect(
        graph.connect(
            {.node = ramp, .socket = "Color"},
            emission,
            "Color"),
        "failed to connect sampled Color Ramp");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});

    auto compiled = compiler.compile(graph);
    expect(
        compiled.ok(),
        "sampled Color Ramp was rejected by the graph contract");
    auto surface = compile_surface_program(*compiled.program);
    expect(
        surface.ok(),
        "sampled Color Ramp failed to lower");
    const auto ramp_instruction = std::find_if(
        surface.program->value_instructions().begin(),
        surface.program->value_instructions().end(),
        [](const ValueInstruction &instruction) {
            return instruction.operation == ValueOperation::color_ramp;
        });
    expect(
        ramp_instruction !=
            surface.program->value_instructions().end(),
        "sampled Color Ramp instruction is missing");
    expect(
        (ramp_instruction->static_u0 & 2u) != 0u,
        "sampled Color Ramp lost its normalized-table flag");
    const auto table_parameter = find_property_parameter(
        *surface.program, "Table");
    expect(
        table_parameter != nullptr &&
            table_parameter->type == SocketType::string &&
            ramp_instruction->parameter == table_parameter->id &&
            ramp_instruction->static_table.empty(),
        "sampled Color Ramp table was embedded in shader topology");

    auto different_table_graph = graph;
    expect(
        different_table_graph.set_property(
            ramp,
            "Table",
            SocketValue::string(
                "0,1,0,0,1;0.25,0,1,0,0.75;"
                "0.5,0,0,1,0.5;0.75,1,1,0,0.25;1,1,1,1,0")),
        "failed to author a second sampled Color Ramp table");
    const auto different = compiler.compile(different_table_graph);
    expect(
        different.ok() &&
            compiled.program->analysis().structure_signature ==
                different.program->analysis().structure_signature &&
            compiled.program->analysis().parameter_signature !=
                different.program->analysis().parameter_signature,
        "Color Ramp table length/content fragmented shader topology");
    const auto rebound = bind_surface_parameters(
        *surface.program, *different.program);
    const auto *rebound_table =
        rebound.ok()
            ? rebound.parameters->find(table_parameter->id)
            : nullptr;
    expect(
        rebound_table != nullptr &&
            std::get<std::string>(rebound_table->value).find("0.25") !=
                std::string::npos,
        "same-topology Color Ramp did not rebind its runtime table");
}

void test_point_to_vector_conversion_lowers() {
    ShaderCompiler compiler{make_core_node_registry()};
    ShaderGraph graph;
    const auto geometry = graph.add_node(node_type::geometry, "Geometry");
    const auto conversion = graph.add_node(
        node_type::point_to_vector, "Position to Vector");
    const auto noise = graph.add_node(node_type::noise_texture, "Noise");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    expect(
        graph.set_property(
            noise,
            "NeedsColor",
            SocketValue::boolean(true)),
        "failed to request the Noise Color output");
    expect(
        graph.connect(
            {.node = geometry, .socket = "Position"},
            conversion,
            "Point"),
        "failed to connect Geometry Position to point conversion");
    expect(
        graph.connect(
            {.node = conversion, .socket = "Vector"},
            noise,
            "Vector"),
        "failed to connect converted Position to vector input");
    expect(
        graph.connect(
            {.node = noise, .socket = "Color"},
            emission,
            "Color"),
        "failed to connect noise color to emission");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});

    auto compiled = compiler.compile(graph);
    expect(compiled.ok(), "point-to-vector graph failed to compile");
    auto surface = compile_surface_program(*compiled.program);
    expect(
        surface.ok(),
        "point-to-vector graph failed to lower" +
            (surface.diagnostics.empty()
                 ? std::string{}
                 : ": " + surface.diagnostics.front().message));
    expect(
        std::ranges::any_of(
            surface.program->value_instructions(),
            [](const ValueInstruction &instruction) {
                return instruction.operation ==
                       ValueOperation::passthrough;
            }),
        "point-to-vector passthrough instruction is missing");
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

    auto byte_color_geometry =
        scene.snapshot().geometries.at(geometry);
    byte_color_geometry.cycles_byte_color_attributes.emplace(
        "Byte Color",
        MeshAttribute<std::array<std::uint8_t, 4u>>{
            .domain = MeshAttributeDomain::corner,
            .values = {
                {1u, 2u, 3u, 4u},
                {5u, 6u, 7u, 8u},
                {9u, 10u, 11u, 12u}}});
    byte_color_geometry.default_color_attribute = "Byte Color";
    SceneDelta valid_byte_color{
        .base_revision = 1u,
        .commands = {}};
    valid_byte_color.commands.emplace_back(UpsertGeometry{
        .id = geometry,
        .value = byte_color_geometry});
    auto byte_color_result = scene.apply(valid_byte_color);
    expect(
        byte_color_result.committed,
        "valid Cycles BYTE_COLOR source failed to commit");
    expect(
        scene.snapshot().revision == 2u,
        "Cycles BYTE_COLOR commit did not advance the revision");

    auto invalid_byte_color_geometry = byte_color_geometry;
    invalid_byte_color_geometry.cycles_byte_color_attributes
        .at("Byte Color")
        .domain = MeshAttributeDomain::face;
    invalid_byte_color_geometry.cycles_byte_color_attributes
        .at("Byte Color")
        .values.resize(1u);
    SceneDelta invalid_byte_color{
        .base_revision = 2u,
        .commands = {}};
    invalid_byte_color.commands.emplace_back(UpsertGeometry{
        .id = geometry,
        .value = invalid_byte_color_geometry});
    auto rejected_byte_color = scene.apply(invalid_byte_color);
    expect(
        !rejected_byte_color.committed,
        "non-corner Cycles BYTE_COLOR source was committed");

    invalid_byte_color_geometry = byte_color_geometry;
    invalid_byte_color_geometry.cycles_byte_color_attributes
        .at("Byte Color")
        .values.pop_back();
    invalid_byte_color = SceneDelta{
        .base_revision = 2u,
        .commands = {}};
    invalid_byte_color.commands.emplace_back(UpsertGeometry{
        .id = geometry,
        .value = invalid_byte_color_geometry});
    rejected_byte_color = scene.apply(invalid_byte_color);
    expect(
        !rejected_byte_color.committed,
        "truncated Cycles BYTE_COLOR source was committed");

    auto invalid_default_geometry = byte_color_geometry;
    invalid_default_geometry.default_color_attribute = "Missing";
    SceneDelta invalid_default{
        .base_revision = 2u,
        .commands = {}};
    invalid_default.commands.emplace_back(UpsertGeometry{
        .id = geometry,
        .value = invalid_default_geometry});
    auto rejected_default = scene.apply(invalid_default);
    expect(
        !rejected_default.committed,
        "missing default color attribute was committed");

    SceneDelta invalid{.base_revision = 2u, .commands = {}};
    invalid.commands.emplace_back(RemoveMaterial{.id = material});
    auto rejected = scene.apply(invalid);
    expect(!rejected.committed, "dangling material removal was committed");
    expect(
        scene.snapshot().revision == 2u,
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
        test_incremental_material_library();
        test_material_library_compiles_only_the_selected_domain();
        test_graph_errors_are_explicit();
        test_linked_inputs_retain_cycles_socket_defaults();
        test_sampled_color_ramp_is_part_of_the_graph_contract();
        test_point_to_vector_conversion_lowers();
        test_scene_delta_is_atomic();
        std::cout
            << "All Psycles graph/material/scene tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Psycles graph/material/scene test failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

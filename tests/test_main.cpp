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

[[nodiscard]] ShaderGraph make_image_emission_graph(
    std::string interpolation,
    std::string extension,
    std::string projection,
    float projection_blend) {
    ShaderGraph graph;
    const auto image =
        graph.add_node(node_type::image_texture, "Image Texture");
    const auto emission =
        graph.add_node(node_type::emission, "Emission");
    expect(
        graph.set_property(
            image,
            "Interpolation",
            SocketValue::string(std::move(interpolation))),
        "failed to set image interpolation");
    expect(
        graph.set_property(
            image,
            "Extension",
            SocketValue::string(std::move(extension))),
        "failed to set image extension");
    expect(
        graph.set_property(
            image,
            "Projection",
            SocketValue::string(std::move(projection))),
        "failed to set image projection");
    expect(
        graph.set_property(
            image,
            "ProjectionBlend",
            SocketValue::floating(projection_blend)),
        "failed to set image projection blend");
    expect(
        graph.connect(
            {.node = image, .socket = "Color"},
            emission,
            "Color"),
        "failed to connect image color to emission");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_environment_emission_graph(
    std::string interpolation,
    std::string projection) {
    ShaderGraph graph;
    const auto environment = graph.add_node(
        node_type::environment_texture,
        "Environment Texture");
    const auto emission =
        graph.add_node(node_type::emission, "Emission");
    expect(
        graph.set_property(
            environment,
            "Interpolation",
            SocketValue::string(std::move(interpolation))),
        "failed to set environment interpolation");
    expect(
        graph.set_property(
            environment,
            "Projection",
            SocketValue::string(std::move(projection))),
        "failed to set environment projection");
    expect(
        graph.connect(
            {.node = environment, .socket = "Color"},
            emission,
            "Color"),
        "failed to connect environment color to emission");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
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

void test_image_texture_modes_are_structural() {
    ShaderCompiler compiler{make_core_node_registry()};
    const auto baseline = compiler.compile(
        make_image_emission_graph(
            "Linear", "REPEAT", "FLAT", 0.0f));
    const auto closest = compiler.compile(
        make_image_emission_graph(
            "Closest", "REPEAT", "FLAT", 0.0f));
    const auto mirror = compiler.compile(
        make_image_emission_graph(
            "Linear", "MIRROR", "FLAT", 0.0f));
    const auto sphere = compiler.compile(
        make_image_emission_graph(
            "Linear", "REPEAT", "SPHERE", 0.0f));
    const auto box_blend = compiler.compile(
        make_image_emission_graph(
            "Linear", "REPEAT", "BOX", 0.55f));

    expect(baseline.ok(), "baseline image graph failed to compile");
    expect(closest.ok(), "closest image graph failed to compile");
    expect(mirror.ok(), "mirror image graph failed to compile");
    expect(sphere.ok(), "sphere image graph failed to compile");
    expect(box_blend.ok(), "box image graph failed to compile");

    const auto baseline_signature =
        baseline.program->analysis().structure_signature;
    expect(
        closest.program->analysis().structure_signature !=
            baseline_signature,
        "image interpolation did not invalidate shader structure");
    expect(
        mirror.program->analysis().structure_signature !=
            baseline_signature,
        "image extension did not invalidate shader structure");
    expect(
        sphere.program->analysis().structure_signature !=
            baseline_signature,
        "image projection did not invalidate shader structure");
    expect(
        box_blend.program->analysis().structure_signature !=
            baseline_signature,
        "image projection blend did not invalidate shader structure");
}

void test_environment_texture_modes_are_structural() {
    ShaderCompiler compiler{make_core_node_registry()};
    const auto baseline = compiler.compile(
        make_environment_emission_graph(
            "Linear", "EQUIRECTANGULAR"));
    const auto closest = compiler.compile(
        make_environment_emission_graph(
            "Closest", "EQUIRECTANGULAR"));
    const auto mirror_ball = compiler.compile(
        make_environment_emission_graph(
            "Linear", "MIRROR_BALL"));

    expect(
        baseline.ok(),
        "baseline environment graph failed to compile");
    expect(
        closest.ok(),
        "closest environment graph failed to compile");
    expect(
        mirror_ball.ok(),
        "mirror-ball environment graph failed to compile");
    const auto baseline_signature =
        baseline.program->analysis().structure_signature;
    expect(
        closest.program->analysis().structure_signature !=
            baseline_signature,
        "environment interpolation did not invalidate shader structure");
    expect(
        mirror_ball.program->analysis().structure_signature !=
            baseline_signature,
        "environment projection did not invalidate shader structure");

    const auto lowered = compile_surface_program(*baseline.program);
    expect(
        lowered.ok(),
        "environment graph did not lower to a surface program");
    const auto has_environment_color = std::any_of(
        lowered.program->value_instructions().begin(),
        lowered.program->value_instructions().end(),
        [](const auto &instruction) {
            return instruction.operation ==
                   ValueOperation::environment_color;
        });
    expect(
        has_environment_color,
        "environment graph emitted no typed environment operation");
}

void test_wave_texture_configuration_lowers_structurally() {
    ShaderCompiler compiler{make_core_node_registry()};
    ShaderGraph graph;
    const auto wave = graph.add_node(
        node_type::wave_texture,
        "Wave Texture");
    const auto emission = graph.add_node(
        node_type::emission,
        "Emission");
    for (const auto &[name, value] : {
             std::pair{"Scale", 7.25f},
             std::pair{"Distortion", 1.5f},
             std::pair{"Detail", 3.75f},
             std::pair{"DetailScale", -2.0f},
             std::pair{"DetailRoughness", 0.625f},
             std::pair{"PhaseOffset", -0.75f}}) {
        expect(
            graph.set_input(
                wave,
                name,
                SocketValue::floating(value)),
            std::string{"failed to set Wave input "} + name);
    }
    expect(
        graph.set_input(
            wave,
            "Vector",
            SocketValue::vector({0.25f, -0.5f, 1.75f})),
        "failed to set Wave Vector");
    for (const auto &[name, value] : {
             std::pair{"WaveType", "RINGS"},
             std::pair{"BandsDirection", "DIAGONAL"},
             std::pair{"RingsDirection", "SPHERICAL"},
             std::pair{"Profile", "TRI"}}) {
        expect(
            graph.set_property(
                wave,
                name,
                SocketValue::string(value)),
            std::string{"failed to set Wave property "} + name);
    }
    expect(
        graph.set_property(
            wave,
            "NeedsColor",
            SocketValue::boolean(true)),
        "failed to select the Wave Color output");
    expect(
        graph.connect(
            {.node = wave, .socket = "Color"},
            emission,
            "Color"),
        "failed to connect Wave Color");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});

    auto compiled = compiler.compile(graph);
    expect(compiled.ok(), "Wave graph failed contract compilation");
    auto surface = compile_surface_program(*compiled.program);
    expect(surface.ok(), "Wave graph failed typed lowering");
    const auto wave_color = std::ranges::find_if(
        surface.program->value_instructions(),
        [](const ValueInstruction &instruction) {
            return instruction.operation == ValueOperation::wave_color;
        });
    expect(
        wave_color != surface.program->value_instructions().end(),
        "Wave Color instruction is missing");
    constexpr std::uint64_t expected_configuration =
        1u | (3u << 8u) | (3u << 16u) | (2u << 24u);
    expect(
        wave_color->static_u0 == expected_configuration,
        "Wave static configuration was not encoded structurally");
    expect(
        wave_color->a.valid() && wave_color->b.valid() &&
            wave_color->c.valid() && wave_color->d.valid() &&
            wave_color->e.valid() && wave_color->f.valid() &&
            wave_color->g.valid(),
        "Wave dynamic inputs were not preserved as typed dependencies");
    expect(
        std::ranges::none_of(
            surface.program->value_instructions(),
            [](const ValueInstruction &instruction) {
                return instruction.operation ==
                       ValueOperation::wave_factor;
            }),
        "unused Wave Factor duplicated the procedural AST");
}

void test_voronoi_texture_configuration_lowers_structurally() {
    ShaderCompiler compiler{make_core_node_registry()};
    ShaderGraph graph;
    const auto voronoi = graph.add_node(
        node_type::voronoi_texture,
        "Voronoi Texture");
    const auto emission = graph.add_node(
        node_type::emission,
        "Emission");
    expect(
        graph.set_input(
            voronoi,
            "Vector",
            SocketValue::vector({0.25f, -0.5f, 1.75f})),
        "failed to set Voronoi Vector");
    for (const auto &[name, value] : {
             std::pair{"W", -0.375f},
             std::pair{"Scale", 7.25f},
             std::pair{"Detail", 3.75f},
             std::pair{"Roughness", 0.625f},
             std::pair{"Lacunarity", 2.25f},
             std::pair{"Smoothness", 0.35f},
             std::pair{"Exponent", 1.75f},
             std::pair{"Randomness", 0.83f}}) {
        expect(
            graph.set_input(
                voronoi,
                name,
                SocketValue::floating(value)),
            std::string{"failed to set Voronoi input "} + name);
    }
    expect(
        graph.set_property(
            voronoi,
            "Dimensions",
            SocketValue::unsigned_integer(4u)),
        "failed to set Voronoi dimensions");
    expect(
        graph.set_property(
            voronoi,
            "Feature",
            SocketValue::string("SMOOTH_F1")),
        "failed to set Voronoi feature");
    expect(
        graph.set_property(
            voronoi,
            "DistanceMetric",
            SocketValue::string("MINKOWSKI")),
        "failed to set Voronoi metric");
    expect(
        graph.set_property(
            voronoi,
            "Normalize",
            SocketValue::boolean(true)),
        "failed to set Voronoi normalization");
    expect(
        graph.set_property(
            voronoi,
            "Output",
            SocketValue::string("Color")),
        "failed to select Voronoi Color");
    expect(
        graph.connect(
            {.node = voronoi, .socket = "Color"},
            emission,
            "Color"),
        "failed to connect Voronoi Color");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});

    auto compiled = compiler.compile(graph);
    expect(compiled.ok(), "Voronoi graph failed contract compilation");
    auto surface = compile_surface_program(*compiled.program);
    expect(surface.ok(), "Voronoi graph failed typed lowering");
    const auto instruction = std::ranges::find_if(
        surface.program->value_instructions(),
        [](const ValueInstruction &value) {
            return value.operation == ValueOperation::voronoi_color;
        });
    expect(
        instruction != surface.program->value_instructions().end(),
        "Voronoi Color instruction is missing");
    constexpr std::uint64_t expected_configuration =
        4u |
        (static_cast<std::uint64_t>(VoronoiFeature::smooth_f1) << 8u) |
        (static_cast<std::uint64_t>(
             VoronoiDistanceMetric::minkowski)
            << 16u) |
        (1ull << 24u);
    expect(
        instruction->static_u0 == expected_configuration,
        "Voronoi static configuration was not encoded structurally");
    expect(
        instruction->a.valid() && instruction->b.valid() &&
            instruction->c.valid() && instruction->d.valid() &&
            instruction->e.valid() && instruction->f.valid() &&
            instruction->g.valid() && instruction->h.valid() &&
            instruction->i.valid(),
        "Voronoi dynamic inputs were not preserved as typed dependencies");
    expect(
        std::ranges::none_of(
            surface.program->value_instructions(),
            [](const ValueInstruction &value) {
                return value.operation ==
                           ValueOperation::voronoi_distance ||
                       value.operation ==
                           ValueOperation::voronoi_position ||
                       value.operation == ValueOperation::voronoi_w ||
                       value.operation == ValueOperation::voronoi_radius;
            }),
        "unused Voronoi outputs duplicated the procedural AST");
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

void test_cycles_emission_evaluation_mode() {
    ShaderCompiler compiler{make_core_node_registry()};
    const auto compile_mode = [&](const ShaderGraph &graph) {
        const auto shader = compiler.compile(graph);
        expect(shader.ok(), "emission scheduling graph failed to compile");
        const auto surface =
            compile_surface_program(*shader.program);
        expect(surface.ok(), "emission scheduling graph failed to lower");
        return surface.program->emission_evaluation();
    };

    ShaderGraph diffuse_graph;
    const auto diffuse =
        diffuse_graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    diffuse_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    expect(
        compile_mode(diffuse_graph) == EmissionEvaluationMode::none,
        "non-emitting closure was scheduled as an emitter");

    ShaderGraph constant_graph;
    const auto constant_emission =
        constant_graph.add_node(node_type::emission, "Emission");
    constant_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = constant_emission, .socket = "Closure"});
    expect(
        compile_mode(constant_graph) ==
            EmissionEvaluationMode::constant,
        "unconnected Emission parameters missed Cycles' constant path");

    ShaderGraph linked_color_graph;
    const auto color = linked_color_graph.add_node(
        node_type::constant_color, "Linked Constant Color");
    const auto linked_color_emission = linked_color_graph.add_node(
        node_type::emission, "Emission");
    expect(
        linked_color_graph.connect(
            {.node = color, .socket = "Color"},
            linked_color_emission,
            "Color"),
        "failed to connect constant emission color");
    linked_color_graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = linked_color_emission,
            .socket = "Closure"});
    expect(
        compile_mode(linked_color_graph) ==
            EmissionEvaluationMode::deferred,
        "linked Emission color was incorrectly folded on the host");

    ShaderGraph additive_graph;
    const auto additive_diffuse = additive_graph.add_node(
        node_type::diffuse_bsdf, "Diffuse");
    const auto additive_emission = additive_graph.add_node(
        node_type::emission, "Emission");
    const auto add = additive_graph.add_node(
        node_type::add_closure, "Add");
    expect(
        additive_graph.connect(
            {.node = additive_diffuse, .socket = "Closure"},
            add,
            "A") &&
            additive_graph.connect(
                {.node = additive_emission, .socket = "Closure"},
                add,
                "B"),
        "failed to construct additive emission graph");
    additive_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = add, .socket = "Closure"});
    expect(
        compile_mode(additive_graph) ==
            EmissionEvaluationMode::constant,
        "non-emitting Add branch invalidated constant emission");

    ShaderGraph linked_factor_graph;
    const auto factor = linked_factor_graph.add_node(
        node_type::constant_float, "Linked Constant Factor");
    const auto factor_diffuse = linked_factor_graph.add_node(
        node_type::diffuse_bsdf, "Diffuse");
    const auto factor_emission = linked_factor_graph.add_node(
        node_type::emission, "Emission");
    const auto mix = linked_factor_graph.add_node(
        node_type::mix_closure, "Mix");
    expect(
        linked_factor_graph.connect(
            {.node = factor, .socket = "Value"},
            mix,
            "Factor") &&
            linked_factor_graph.connect(
                {.node = factor_diffuse, .socket = "Closure"},
                mix,
                "A") &&
            linked_factor_graph.connect(
                {.node = factor_emission, .socket = "Closure"},
                mix,
                "B"),
        "failed to construct linked-factor emission graph");
    linked_factor_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = mix, .socket = "Closure"});
    expect(
        compile_mode(linked_factor_graph) ==
            EmissionEvaluationMode::deferred,
        "linked Mix factor was incorrectly folded into constant emission");
}

void test_cycles_principled_emission_metadata() {
    ShaderCompiler compiler{make_core_node_registry()};
    const auto make_graph = [](Vec3f color, float strength) {
        ShaderGraph graph;
        const auto principled = graph.add_node(
            node_type::principled_bsdf,
            "Principled emission metadata");
        expect(
            graph.set_input(
                principled,
                "EmissionColor",
                SocketValue::color(color)) &&
                graph.set_input(
                    principled,
                    "EmissionStrength",
                    SocketValue::floating(strength)),
            "failed to configure Principled emission metadata graph");
        graph.set_root(
            ShaderDomain::surface,
            OutputRef{.node = principled, .socket = "Closure"});
        return graph;
    };

    const auto zero_shader = compiler.compile(
        make_graph({1.0f, 1.0f, 1.0f}, 0.0f));
    const auto emitting_shader = compiler.compile(
        make_graph({0.25f, 0.5f, 1.0f}, 4.0f));
    expect(
        zero_shader.ok() && emitting_shader.ok(),
        "Principled emission metadata graph failed to compile");
    expect(
        zero_shader.program->analysis().structure_signature ==
            emitting_shader.program->analysis().structure_signature,
        "Principled emission parameters changed reusable topology");
    expect(
        (emitting_shader.program->analysis().required_features &
         feature_bit(ShaderFeature::emission)) != 0u,
        "Principled emission feature was not retained");

    const auto surface =
        compile_surface_program(*zero_shader.program);
    expect(surface.ok(), "Principled emission surface failed to lower");
    expect(
        surface.program->emission_evaluation() ==
            EmissionEvaluationMode::deferred,
        "Principled emission entered the constant scheduling path");
    expect(
        surface.program->closure_instructions().size() == 1u &&
            surface.program->closure_instructions().front()
                .emission_color.valid() &&
            surface.program->closure_instructions().front()
                .emission_strength.valid(),
        "Principled raw emission expressions were dropped during lowering");

    const SurfaceParameterBlock zero_parameters{*surface.program};
    expect(
        estimate_surface_emission(
            *surface.program,
            zero_parameters) == Vec3f{},
        "zero Principled emission entered emitter metadata");
    const auto rebound = bind_surface_parameters(
        *surface.program,
        *emitting_shader.program);
    expect(
        rebound.ok(),
        "Principled emission parameters could not reuse the surface AST");
    expect(
        estimate_surface_emission(
            *surface.program,
            *rebound.parameters) == Vec3f{1.0f, 2.0f, 4.0f},
        "Principled emission estimate diverged from Cycles metadata");

    ShaderGraph linked_graph;
    const auto linked_color = linked_graph.add_node(
        node_type::constant_color,
        "Linked zero color");
    const auto linked_principled = linked_graph.add_node(
        node_type::principled_bsdf,
        "Linked Principled emission");
    expect(
        linked_graph.set_input(
            linked_color,
            "Color",
            SocketValue::color({0.0f, 0.0f, 0.0f})) &&
            linked_graph.set_input(
                linked_principled,
                "EmissionStrength",
                SocketValue::floating(2.0f)) &&
            linked_graph.connect(
                {.node = linked_color, .socket = "Color"},
                linked_principled,
                "EmissionColor"),
        "failed to link Principled emission metadata graph");
    linked_graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = linked_principled,
            .socket = "Closure"});
    const auto linked_shader = compiler.compile(linked_graph);
    expect(linked_shader.ok(), "linked Principled graph failed to compile");
    const auto linked_surface =
        compile_surface_program(*linked_shader.program);
    expect(linked_surface.ok(), "linked Principled surface failed to lower");
    const SurfaceParameterBlock linked_parameters{
        *linked_surface.program};
    expect(
        estimate_surface_emission(
            *linked_surface.program,
            linked_parameters) == Vec3f{2.0f, 2.0f, 2.0f},
        "linked Principled color was host-evaluated instead of estimated");
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

    const auto program =
        compile_surface_program(*shader.program);
    expect(program.ok(), "volume graph failed to lower");
    expect(
        program.program->volume_root().valid(),
        "lowered volume root is missing");
    expect(
        program.program->volume_instructions().size() == 7u,
        "volume closure tree was flattened or lost");
    const auto has_phase =
        [&](VolumeOperation operation, VolumePhase phase) {
            return std::ranges::any_of(
                program.program->volume_instructions(),
                [=](const VolumeInstruction &instruction) {
                    return instruction.operation == operation &&
                           instruction.phase == phase;
                });
        };
    expect(
        has_phase(
            VolumeOperation::scatter,
            VolumePhase::draine),
        "Draine phase was not preserved structurally");
    expect(
        has_phase(
            VolumeOperation::coefficients,
            VolumePhase::mie),
        "Mie phase was not preserved structurally");
    const auto &root_instruction =
        program.program->volume_instructions()
            [program.program->volume_root().value];
    expect(
        root_instruction.operation == VolumeOperation::add &&
            root_instruction.a.valid() &&
            root_instruction.b.valid(),
        "volume root composition is not explicit");
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

void test_spectral_color_nodes_lower_to_surface_program() {
    const auto verify =
        [](const char *type_name,
           const char *input_name,
           float input_value,
           ValueOperation expected_operation) {
            ShaderGraph graph;
            const auto spectral =
                graph.add_node(type_name, "Spectral Color");
            const auto emission =
                graph.add_node(node_type::emission, "Emission");
            expect(
                graph.set_input(
                    spectral,
                    input_name,
                    SocketValue::floating(input_value)),
                "failed to configure spectral color node");
            expect(
                graph.connect(
                    {.node = spectral, .socket = "Color"},
                    emission,
                    "Color"),
                "failed to connect spectral color node");
            graph.set_root(
                ShaderDomain::surface,
                OutputRef{
                    .node = emission,
                    .socket = "Closure"});

            ShaderCompiler compiler{make_core_node_registry()};
            auto shader = compiler.compile(graph);
            expect(
                shader.ok(),
                "spectral color graph failed to compile");
            auto surface =
                compile_surface_program(*shader.program);
            expect(
                surface.ok(),
                "spectral color graph failed to lower");
            expect(
                std::ranges::any_of(
                    surface.program->value_instructions(),
                    [expected_operation](
                        const ValueInstruction &instruction) {
                        return instruction.operation ==
                               expected_operation;
                    }),
                "spectral color instruction is missing");
        };

    verify(
        node_type::blackbody,
        "Temperature",
        6500.0f,
        ValueOperation::blackbody);
    verify(
        node_type::wavelength,
        "Wavelength",
        555.0f,
        ValueOperation::wavelength);
}

void test_map_range_modes_lower_to_surface_program() {
    ShaderCompiler compiler{make_core_node_registry()};

    ShaderGraph scalar_graph;
    const auto scalar_map =
        scalar_graph.add_node(node_type::map_range, "Scalar Map Range");
    const auto scalar_color = scalar_graph.add_node(
        node_type::scalar_to_color, "Scalar Map Range Color");
    const auto scalar_emission =
        scalar_graph.add_node(node_type::emission, "Scalar Emission");
    expect(
        scalar_graph.set_property(
            scalar_map,
            "DataType",
            SocketValue::string("FLOAT")) &&
            scalar_graph.set_property(
                scalar_map,
                "Interpolation",
                SocketValue::string("STEPPED")) &&
            scalar_graph.set_property(
                scalar_map,
                "Clamp",
                SocketValue::boolean(true)) &&
            scalar_graph.set_input(
                scalar_map,
                "Steps",
                SocketValue::floating(3.0f)) &&
            scalar_graph.connect(
                {.node = scalar_map, .socket = "Result"},
                scalar_color,
                "Value") &&
            scalar_graph.connect(
                {.node = scalar_color, .socket = "Color"},
                scalar_emission,
                "Color"),
        "failed to construct scalar Map Range graph");
    scalar_graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = scalar_emission,
            .socket = "Closure"});
    auto compiled_scalar = compiler.compile(scalar_graph);
    expect(
        compiled_scalar.ok(),
        "scalar Map Range graph failed to compile");
    auto scalar_surface =
        compile_surface_program(*compiled_scalar.program);
    expect(
        scalar_surface.ok(),
        "scalar Map Range graph failed to lower");
    expect(
        std::ranges::any_of(
            scalar_surface.program->value_instructions(),
            [](const ValueInstruction &instruction) {
                return instruction.operation ==
                           ValueOperation::map_range_float &&
                       instruction.static_u0 == 1u &&
                       instruction.static_u1 == 1u;
            }),
        "scalar stepped/clamped Map Range instruction is missing");

    ShaderGraph vector_graph;
    const auto vector_map =
        vector_graph.add_node(node_type::map_range, "Vector Map Range");
    const auto vector_color = vector_graph.add_node(
        node_type::vector_to_color, "Vector Map Range Color");
    const auto vector_emission =
        vector_graph.add_node(node_type::emission, "Vector Emission");
    expect(
        vector_graph.set_property(
            vector_map,
            "DataType",
            SocketValue::string("FLOAT_VECTOR")) &&
            vector_graph.set_property(
                vector_map,
                "Interpolation",
                SocketValue::string("SMOOTHERSTEP")) &&
            vector_graph.set_property(
                vector_map,
                "Clamp",
                SocketValue::boolean(false)) &&
            vector_graph.set_input(
                vector_map,
                "FromMaxVector",
                SocketValue::vector({2.0f, 1.0f, 0.0f})) &&
            vector_graph.connect(
                {.node = vector_map, .socket = "Vector"},
                vector_color,
                "Vector") &&
            vector_graph.connect(
                {.node = vector_color, .socket = "Color"},
                vector_emission,
                "Color"),
        "failed to construct vector Map Range graph");
    vector_graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = vector_emission,
            .socket = "Closure"});
    auto compiled_vector = compiler.compile(vector_graph);
    expect(
        compiled_vector.ok(),
        "vector Map Range graph failed to compile");
    auto vector_surface =
        compile_surface_program(*compiled_vector.program);
    expect(
        vector_surface.ok(),
        "vector Map Range graph failed to lower");
    expect(
        std::ranges::any_of(
            vector_surface.program->value_instructions(),
            [](const ValueInstruction &instruction) {
                return instruction.operation ==
                           ValueOperation::map_range_vector &&
                       instruction.static_u0 == 3u &&
                       instruction.static_u1 == 0u;
            }),
        "vector smootherstep Map Range instruction is missing");
}

void test_vector_math_modes_lower_to_surface_program() {
    ShaderCompiler compiler{make_core_node_registry()};

    ShaderGraph graph;
    const auto vector_math = graph.add_node(
        node_type::vector_math, "Vector Math Refract");
    const auto vector_color = graph.add_node(
        node_type::vector_to_color, "Vector Math Color");
    const auto emission =
        graph.add_node(node_type::emission, "Vector Math Emission");
    expect(
        graph.set_property(
            vector_math,
            "Operation",
            SocketValue::string("REFRACT")) &&
            graph.set_input(
                vector_math,
                "Scale",
                SocketValue::floating(1.33f)) &&
            graph.connect(
                {.node = vector_math, .socket = "Vector"},
                vector_color,
                "Vector") &&
            graph.connect(
                {.node = vector_color, .socket = "Color"},
                emission,
                "Color"),
        "failed to construct Vector Math graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = emission,
            .socket = "Closure"});
    auto compiled = compiler.compile(graph);
    expect(compiled.ok(), "Vector Math graph failed to compile");
    auto surface = compile_surface_program(*compiled.program);
    expect(surface.ok(), "Vector Math graph failed to lower");
    expect(
        std::ranges::any_of(
            surface.program->value_instructions(),
            [](const ValueInstruction &instruction) {
                return instruction.operation ==
                           ValueOperation::vector_math_vector &&
                       instruction.static_u0 ==
                           static_cast<std::uint64_t>(
                               VectorMathOperation::refract);
            }),
        "Vector Math refract instruction is missing");
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

void test_cycles_glass_closure_lowering() {
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
            .type = "glass_bsdf",
            .variant = {},
            .label = "Glass",
            .inputs = {
                {
                    .socket = "Color",
                    .source = std::nullopt,
                    .value = SocketValue::color({0.8f, 0.9f, 1.0f})},
                {
                    .socket = "Roughness",
                    .source = std::nullopt,
                    .value = SocketValue::floating(0.17320508f)},
                {
                    .socket = "IOR",
                    .source = std::nullopt,
                    .value = SocketValue::floating(1.45f)},
                {
                    .socket = "Normal",
                    .source = CyclesOutputRef{
                        .node = 10u,
                        .socket = "Normal"},
                    .value = std::nullopt}},
            .properties = {{
                "distribution",
                SocketValue::string("BECKMANN")}}}};
    source.set_root(
        ShaderDomain::surface,
        CyclesOutputRef{.node = 11u, .socket = "BSDF"});

    auto adapted = adapt_cycles_shader_graph(
        source, make_core_cycles_node_mappings());
    expect(adapted.ok(), "normalized Glass graph failed to adapt");

    ShaderCompiler shader_compiler{make_core_node_registry()};
    auto shader = shader_compiler.compile(*adapted.graph);
    expect(shader.ok(), "adapted Glass graph failed validation");
    auto surface = compile_surface_program(*shader.program);
    expect(surface.ok(), "adapted Glass graph failed surface lowering");
    expect(
        surface.program->closure_instructions().size() == 1u,
        "Glass graph did not lower to one closure");
    const auto &closure =
        surface.program->closure_instructions().front();
    expect(
        closure.operation == ClosureOperation::glass &&
            closure.beckmann && !closure.preserve_ggx_energy &&
            closure.ior.valid(),
        "Glass closure lost its type, IOR, or Beckmann distribution");
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
    auto surface = compile_surface_program(*shader.program);
    expect(
        surface.ok(),
        "adapted Principled emission graph failed surface lowering");
    const SurfaceParameterBlock parameters{*surface.program};
    expect(
        surface.program->closure_instructions().size() == 1u,
        "layered Principled graph did not lower to one closure");
    const auto &closure =
        surface.program->closure_instructions().front();
    expect(
        closure.operation == ClosureOperation::principled &&
            closure.preserve_ggx_energy &&
            closure.alpha.valid() &&
            closure.sheen_weight.valid() &&
            closure.sheen_roughness.valid() &&
            closure.sheen_tint.valid() &&
            closure.transmission_weight.valid() &&
            closure.coat_weight.valid() &&
            closure.coat_roughness.valid() &&
            closure.coat_ior.valid() &&
            closure.coat_tint.valid() &&
            closure.coat_normal.valid() &&
            closure.coat_normal_linked &&
            estimate_surface_emission(
                *surface.program,
                parameters) == Vec3f{0.6f, 1.2f, 2.4f},
        "normalized Principled layer/emission sockets, linked-normal "
        "topology, or distribution changed during adaptation");
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
        test_image_texture_modes_are_structural();
        test_environment_texture_modes_are_structural();
        test_wave_texture_configuration_lowers_structurally();
        test_voronoi_texture_configuration_lowers_structurally();
        test_closure_tree_is_preserved();
        test_cycles_emission_evaluation_mode();
        test_cycles_principled_emission_metadata();
        test_volume_closure_tree_is_preserved();
        test_combine_color_lowers_to_surface_program();
        test_cycles_color_value_nodes_lower_to_surface_program();
        test_spectral_color_nodes_lower_to_surface_program();
        test_map_range_modes_lower_to_surface_program();
        test_vector_math_modes_lower_to_surface_program();
        test_cycles_normalized_graph_adapter();
        test_cycles_glass_closure_lowering();
        test_cycles_principled_emission_adapter();
        test_cycles_adapter_rejects_svm_lowered_graph();
        test_incremental_material_library();
        test_graph_errors_are_explicit();
        test_sampled_color_ramp_is_part_of_the_graph_contract();
        test_point_to_vector_conversion_lowers();
        test_scene_delta_is_atomic();
        std::cout << "All Psycles contract tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Psycles test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

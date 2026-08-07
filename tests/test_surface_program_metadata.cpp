#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

enum class NormalTopology {
    unlinked,
    geometry,
    normal_map
};

[[nodiscard]] ShaderGraph make_principled_graph(
    float subsurface_weight,
    NormalTopology normal_topology = NormalTopology::unlinked,
    bool thin_wall = false,
    bool automatic_displacement_bump = false) {
    ShaderGraph graph;
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Principled BSSRDF metadata");
    require(
        graph.set_input(
            principled,
            "SubsurfaceWeight",
            SocketValue::floating(subsurface_weight)) &&
            graph.set_input(
                principled,
                "ThinWall",
                SocketValue::boolean(thin_wall)),
        "failed to configure Principled BSSRDF metadata");
    if (normal_topology != NormalTopology::unlinked) {
        const auto source = graph.add_node(
            normal_topology == NormalTopology::geometry
                ? node_type::geometry
                : node_type::normal_map,
            "Principled normal source");
        require(
            graph.connect(
                {.node = source, .socket = "Normal"},
                principled,
                "Normal"),
            "failed to connect Principled normal source");
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    if (automatic_displacement_bump) {
        const auto bump = graph.add_node(
            node_type::bump,
            "Normalized automatic displacement bump");
        graph.set_root(
            ShaderDomain::surface_normal,
            OutputRef{.node = bump, .socket = "Normal"});
    }
    return graph;
}

[[nodiscard]] ShaderGraph make_standalone_bssrdf_graph(
    NormalTopology normal_topology) {
    ShaderGraph graph;
    const auto subsurface = graph.add_node(
        node_type::subsurface_scattering,
        "Standalone BSSRDF metadata");
    if (normal_topology != NormalTopology::unlinked) {
        const auto source = graph.add_node(
            normal_topology == NormalTopology::geometry
                ? node_type::geometry
                : node_type::normal_map,
            "Standalone normal source");
        require(
            graph.connect(
                {.node = source, .socket = "Normal"},
                subsurface,
                "Normal"),
            "failed to connect standalone BSSRDF normal source");
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = subsurface, .socket = "Closure"});
    return graph;
}

void test_cycles_surface_bssrdf_metadata() {
    ShaderCompiler compiler{make_core_node_registry()};
    const auto lower = [&](ShaderGraph graph) {
        const auto shader = compiler.compile(graph);
        require(shader.ok(), "BSSRDF metadata graph failed to compile");
        const auto surface = compile_surface_program(*shader.program);
        require(surface.ok(), "BSSRDF metadata graph failed to lower");
        return std::pair{
            surface.program,
            SurfaceParameterBlock{*surface.program}};
    };

    const auto [zero_program, zero_parameters] =
        lower(make_principled_graph(0.0f));
    require(
        !cycles_surface_has_bssrdf(*zero_program, zero_parameters),
        "direct zero Principled subsurface blocked static transforms");
    require(
        !cycles_surface_has_bssrdf_bump(
            *zero_program,
            zero_parameters,
            DisplacementMethod::bump),
        "zero Principled subsurface acquired BSSRDF bump metadata");

    const auto [positive_program, positive_parameters] =
        lower(make_principled_graph(0.25f));
    require(
        cycles_surface_has_bssrdf(*positive_program, positive_parameters),
        "positive Principled subsurface lost Cycles geometry metadata");
    require(
        !cycles_surface_has_bssrdf_bump(
            *positive_program,
            positive_parameters,
            DisplacementMethod::bump),
        "unlinked Principled Normal incorrectly enabled BSSRDF bump");

    const auto [geometry_program, geometry_parameters] = lower(
        make_principled_graph(
            0.25f, NormalTopology::geometry));
    require(
        cycles_surface_has_bssrdf(*geometry_program, geometry_parameters) &&
            !cycles_surface_has_bssrdf_bump(
                *geometry_program,
                geometry_parameters,
                DisplacementMethod::bump),
        "direct Geometry Normal did not preserve Cycles' no-bump proof");

    const auto [normal_map_program, normal_map_parameters] = lower(
        make_principled_graph(
            0.25f, NormalTopology::normal_map));
    require(
        cycles_surface_has_bssrdf_bump(
            *normal_map_program,
            normal_map_parameters,
            DisplacementMethod::bump),
        "linked Principled Normal lost Cycles BSSRDF bump metadata");

    ShaderGraph linked_zero_graph;
    const auto linked_zero = linked_zero_graph.add_node(
        node_type::constant_float,
        "Linked zero weight");
    const auto linked_principled = linked_zero_graph.add_node(
        node_type::principled_bsdf,
        "Linked Principled subsurface");
    require(
        linked_zero_graph.set_input(
            linked_zero,
            "Value",
            SocketValue::floating(0.0f)) &&
            linked_zero_graph.connect(
                {.node = linked_zero, .socket = "Value"},
                linked_principled,
                "SubsurfaceWeight"),
        "failed to link Principled subsurface weight");
    linked_zero_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = linked_principled, .socket = "Closure"});
    const auto [linked_program, linked_parameters] =
        lower(linked_zero_graph);
    require(
        cycles_surface_has_bssrdf(
            *linked_program, linked_parameters),
        "linked zero subsurface was incorrectly host-evaluated");

    const auto [explicit_program, explicit_parameters] =
        lower(make_standalone_bssrdf_graph(
            NormalTopology::unlinked));
    require(
        cycles_surface_has_bssrdf(
            *explicit_program, explicit_parameters),
        "explicit Subsurface closure lost Cycles geometry metadata");
    require(
        !cycles_surface_has_bssrdf_bump(
            *explicit_program,
            explicit_parameters,
            DisplacementMethod::bump),
        "unlinked standalone BSSRDF Normal incorrectly enabled bump");

    const auto [explicit_bump_program, explicit_bump_parameters] =
        lower(make_standalone_bssrdf_graph(
            NormalTopology::normal_map));
    require(
        cycles_surface_has_bssrdf_bump(
            *explicit_bump_program,
            explicit_bump_parameters,
            DisplacementMethod::bump),
        "standalone BSSRDF linked Normal lost bump metadata");

    const auto [displacement_program, displacement_parameters] = lower(
        make_principled_graph(
            0.25f,
            NormalTopology::unlinked,
            false,
            true));
    require(
        displacement_program->surface_normal_root().valid() &&
            cycles_surface_has_bssrdf_bump(
                *displacement_program,
                displacement_parameters,
                DisplacementMethod::bump) &&
            cycles_surface_has_bssrdf_bump(
                *displacement_program,
                displacement_parameters,
                DisplacementMethod::both) &&
            !cycles_surface_has_bssrdf_bump(
                *displacement_program,
                displacement_parameters,
                DisplacementMethod::displacement),
        "automatic displacement bump policy diverged from Cycles");

    const auto thin_graph = make_principled_graph(
        0.25f, NormalTopology::normal_map, true);
    const auto thin_shader = compiler.compile(thin_graph);
    require(thin_shader.ok(), "Thin Wall metadata graph failed to compile");
    require(
        thin_shader.program->analysis().structure_signature ==
            normal_map_program->structure_signature(),
        "direct Thin Wall parameter changed reusable topology");
    const auto thin_parameters = bind_surface_parameters(
        *normal_map_program, *thin_shader.program);
    require(
        thin_parameters.ok() &&
            !cycles_surface_has_bssrdf(
                *normal_map_program,
                *thin_parameters.parameters) &&
            !cycles_surface_has_bssrdf_bump(
                *normal_map_program,
                *thin_parameters.parameters,
                DisplacementMethod::bump),
        "direct Thin Wall did not disable real Principled BSSRDF metadata");

    ShaderGraph mixed_graph;
    const auto mixed_normal = mixed_graph.add_node(
        node_type::normal_map, "Zero-weight bumped normal");
    const auto mixed_principled = mixed_graph.add_node(
        node_type::principled_bsdf, "Zero-weight bumped Principled");
    const auto mixed_subsurface = mixed_graph.add_node(
        node_type::subsurface_scattering, "Unbumped standalone BSSRDF");
    const auto mixed_root = mixed_graph.add_node(
        node_type::add_closure, "Mixed BSSRDF root");
    require(
        mixed_graph.connect(
            {.node = mixed_normal, .socket = "Normal"},
            mixed_principled,
            "Normal") &&
            mixed_graph.connect(
                {.node = mixed_principled, .socket = "Closure"},
                mixed_root,
                "A") &&
            mixed_graph.connect(
                {.node = mixed_subsurface, .socket = "Closure"},
                mixed_root,
                "B"),
        "failed to configure mixed BSSRDF bump proof graph");
    mixed_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = mixed_root, .socket = "Closure"});
    const auto [mixed_program, mixed_parameters] = lower(mixed_graph);
    require(
        cycles_surface_has_bssrdf(*mixed_program, mixed_parameters) &&
            !cycles_surface_has_bssrdf_bump(
                *mixed_program,
                mixed_parameters,
                DisplacementMethod::bump),
        "zero-weight bumped closure contaminated another BSSRDF's proof");
}

}// namespace

int main() {
    try {
        test_cycles_surface_bssrdf_metadata();
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Surface-program metadata test failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

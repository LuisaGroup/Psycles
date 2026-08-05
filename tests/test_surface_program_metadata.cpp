#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>

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

    ShaderGraph zero_graph;
    const auto zero_principled = zero_graph.add_node(
        node_type::principled_bsdf,
        "Zero Principled subsurface");
    zero_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = zero_principled, .socket = "Closure"});
    const auto [zero_program, zero_parameters] = lower(zero_graph);
    require(
        !cycles_surface_has_bssrdf(*zero_program, zero_parameters),
        "direct zero Principled subsurface blocked static transforms");

    ShaderGraph positive_graph;
    const auto positive_principled = positive_graph.add_node(
        node_type::principled_bsdf,
        "Positive Principled subsurface");
    require(
        positive_graph.set_input(
            positive_principled,
            "SubsurfaceWeight",
            SocketValue::floating(0.25f)),
        "failed to configure positive Principled subsurface");
    positive_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = positive_principled, .socket = "Closure"});
    const auto [positive_program, positive_parameters] =
        lower(positive_graph);
    require(
        cycles_surface_has_bssrdf(
            *positive_program, positive_parameters),
        "positive Principled subsurface lost Cycles geometry metadata");

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

    ShaderGraph explicit_graph;
    const auto explicit_subsurface = explicit_graph.add_node(
        node_type::subsurface_scattering,
        "Explicit subsurface");
    explicit_graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = explicit_subsurface, .socket = "Closure"});
    const auto [explicit_program, explicit_parameters] =
        lower(explicit_graph);
    require(
        cycles_surface_has_bssrdf(
            *explicit_program, explicit_parameters),
        "explicit Subsurface closure lost Cycles geometry metadata");
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

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

void test_hair_info_lowers_as_typed_curve_state() {
    ShaderCompiler compiler{make_core_node_registry()};
    constexpr std::array outputs{
        std::pair{"IsStrand", ValueOperation::curve_is_strand},
        std::pair{"Intercept", ValueOperation::curve_intercept},
        std::pair{"Length", ValueOperation::curve_length},
        std::pair{"Thickness", ValueOperation::curve_thickness},
        std::pair{"TangentNormal", ValueOperation::curve_tangent_normal},
        std::pair{"Random", ValueOperation::curve_random}};

    for (const auto &[socket, operation] : outputs) {
        ShaderGraph graph;
        const auto hair = graph.add_node(node_type::hair_info, "Hair Info");
        if (std::string_view{socket} == "TangentNormal") {
            const auto diffuse =
                graph.add_node(node_type::diffuse_bsdf, "Diffuse");
            expect(
                graph.connect(
                    {.node = hair, .socket = socket}, diffuse, "Normal"),
                "failed to connect Hair Info Tangent Normal");
            graph.set_root(
                ShaderDomain::surface,
                OutputRef{.node = diffuse, .socket = "Closure"});
        } else {
            const auto emission =
                graph.add_node(node_type::emission, "Emission");
            expect(
                graph.connect(
                    {.node = hair, .socket = socket}, emission, "Strength"),
                "failed to connect Hair Info scalar output");
            graph.set_root(
                ShaderDomain::surface,
                OutputRef{.node = emission, .socket = "Closure"});
        }

        const auto compiled = compiler.compile(graph);
        expect(compiled.ok(), "Hair Info graph failed validation");
        const auto surface = compile_surface_program(*compiled.program);
        expect(surface.ok(), "Hair Info graph failed typed lowering");
        expect(
            std::ranges::any_of(
                surface.program->value_instructions(),
                [operation](const ValueInstruction &instruction) {
                    return instruction.operation == operation;
                }),
            "Hair Info output lost its typed curve operation");
    }
}

} // namespace

int main() {
    try {
        test_hair_info_lowers_as_typed_curve_state();
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "Hair Info graph test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

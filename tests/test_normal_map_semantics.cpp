#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

[[nodiscard]] ValueInstruction lower_normal_map(
    std::string_view base,
    std::string_view convention,
    bool named,
    std::uint64_t tangent_id) {
    ShaderGraph graph;
    const auto normal = graph.add_node(
        node_type::normal_map, "Normal Map semantic contract");
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf, "Normal Map consumer");
    const auto configured =
        graph.set_property(
            normal, "Space", SocketValue::string("TANGENT")) &&
        graph.set_property(
            normal, "Base", SocketValue::string(std::string{base})) &&
        graph.set_property(
            normal,
            "Convention",
            SocketValue::string(std::string{convention})) &&
        graph.set_property(
            normal, "UvMapNamed", SocketValue::boolean(named)) &&
        graph.set_property(
            normal,
            "UvMapId",
            SocketValue::unsigned_integer(tangent_id)) &&
        graph.connect(
            {.node = normal, .socket = "Normal"},
            diffuse,
            "Normal");
    require(configured, "could not construct Normal Map semantic graph");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});

    ShaderCompiler compiler{make_core_node_registry()};
    const auto shader = compiler.compile(graph);
    require(shader.ok(), "Normal Map semantic graph did not compile");
    const auto surface = compile_surface_program(*shader.program);
    require(surface.ok(), "Normal Map semantic graph did not lower");
    for (const auto &instruction :
         surface.program->value_instructions()) {
        if (instruction.operation == ValueOperation::normal_map) {
            return instruction;
        }
    }
    throw std::runtime_error{"Normal Map emitted no value instruction"};
}

void test_original_and_displaced_are_distinct_static_stages() {
    constexpr std::string_view uv_name = "DisplacedUV";
    const auto current_id = uv_tangent_attribute_id(uv_name);
    const auto original_id = uv_undisplaced_tangent_attribute_id(uv_name);
    require(current_id != original_id,
            "current and original named tangent identities alias");

    const auto original = lower_normal_map(
        "ORIGINAL", "OPENGL", true, original_id);
    require(
        decode_normal_map_space(original.static_u0) ==
                NormalMapSpace::tangent &&
            normal_map_has_named_tangent(original.static_u0) &&
            decode_normal_map_base(original.static_u0) ==
                NormalMapBase::original &&
            decode_normal_map_convention(original.static_u0) ==
                NormalMapConvention::open_gl &&
            original.static_u1 == original_id,
        "ORIGINAL/OpenGL Normal Map lost its immutable tangent contract");

    const auto displaced = lower_normal_map(
        "DISPLACED", "DIRECTX", true, current_id);
    require(
        decode_normal_map_space(displaced.static_u0) ==
                NormalMapSpace::tangent &&
            normal_map_has_named_tangent(displaced.static_u0) &&
            decode_normal_map_base(displaced.static_u0) ==
                NormalMapBase::displaced &&
            decode_normal_map_convention(displaced.static_u0) ==
                NormalMapConvention::direct_x &&
            displaced.static_u1 == current_id,
        "DISPLACED/DirectX Normal Map lost its current tangent contract");

    require(original.static_u0 != displaced.static_u0,
            "Normal Map base/convention collapsed to one JIT specialization");
}

} // namespace

int main() {
    test_original_and_displaced_are_distinct_static_stages();
    return EXIT_SUCCESS;
}

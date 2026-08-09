#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

struct LoweredNormalMap {
    ValueInstruction instruction;
    std::uint64_t uv_map_id{};
};

[[nodiscard]] LoweredNormalMap lower_normal_map(
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
    const auto &values = surface.program->value_instructions();
    const auto &parameters = surface.program->parameters();
    for (const auto &instruction : values) {
        if (instruction.operation == ValueOperation::normal_map) {
            const auto uv_map = instruction.operand(
                value_operand::normal_map::uv_map);
            require(uv_map.valid() && uv_map.value < values.size(),
                    "Normal Map UV operand is not a valid value expression");
            const auto &parameter_value = values[uv_map.value];
            require(parameter_value.operation == ValueOperation::parameter &&
                        parameter_value.parameter.valid() &&
                        parameter_value.parameter.value < parameters.size(),
                    "Normal Map UV identity is not a typed parameter");
            const auto &parameter =
                parameters[parameter_value.parameter.value];
            require(parameter.source == ParameterSource::property &&
                        parameter.socket == "UvMapId" &&
                        parameter.type == SocketType::unsigned_integer,
                    "Normal Map UV identity lost its property contract");
            return {
                .instruction = instruction,
                .uv_map_id = std::get<std::uint64_t>(
                    parameter.default_value.value)};
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
        decode_normal_map_space(original.instruction.static_u0) ==
                NormalMapSpace::tangent &&
            normal_map_has_named_tangent(original.instruction.static_u0) &&
            decode_normal_map_base(original.instruction.static_u0) ==
                NormalMapBase::original &&
            decode_normal_map_convention(original.instruction.static_u0) ==
                NormalMapConvention::open_gl &&
            original.uv_map_id == original_id,
        "ORIGINAL/OpenGL Normal Map lost its immutable tangent contract");

    const auto displaced = lower_normal_map(
        "DISPLACED", "DIRECTX", true, current_id);
    require(
        decode_normal_map_space(displaced.instruction.static_u0) ==
                NormalMapSpace::tangent &&
            normal_map_has_named_tangent(displaced.instruction.static_u0) &&
            decode_normal_map_base(displaced.instruction.static_u0) ==
                NormalMapBase::displaced &&
            decode_normal_map_convention(displaced.instruction.static_u0) ==
                NormalMapConvention::direct_x &&
            displaced.uv_map_id == current_id,
        "DISPLACED/DirectX Normal Map lost its current tangent contract");

    require(original.instruction.static_u0 != displaced.instruction.static_u0,
            "Normal Map base/convention collapsed to one JIT specialization");
}

} // namespace

int main() {
    test_original_and_displaced_are_distinct_static_stages();
    return EXIT_SUCCESS;
}

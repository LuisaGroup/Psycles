#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/surface_program.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using namespace psycles::compiler;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void test_dynamic_vector_component_lowering() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto normal_to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto source_to_color =
      graph.add_node(node_type::vector_to_color, "Source to Color");
  const auto separate = graph.add_node(node_type::separate_xyz, "Separate XYZ");
  const auto combine = graph.add_node(node_type::combine_xyz, "Combine XYZ");
  const auto vector_to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({geometry, "Normal"}, normal_to_vector, "Normal") &&
              graph.connect({normal_to_vector, "Vector"}, source_to_color,
                            "Vector") &&
              graph.connect({source_to_color, "Color"}, separate, "Vector") &&
              graph.connect({separate, "Z"}, combine, "X") &&
              graph.connect({separate, "X"}, combine, "Y") &&
              graph.connect({separate, "Y"}, combine, "Z") &&
              graph.connect({combine, "Vector"}, vector_to_color, "Vector") &&
              graph.connect({vector_to_color, "Color"}, emission, "Color"),
          "failed to construct dynamic Separate/Combine XYZ graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "dynamic Separate/Combine XYZ graph did not validate");
  const auto surface = compile_surface_program(*shader.program);
  if (!surface.ok()) {
    for (const auto &diagnostic : surface.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(surface.ok(), "dynamic Separate/Combine XYZ graph did not lower");

  std::array<bool, 3u> separate_components{};
  auto combine_count = std::size_t{};
  for (const auto &instruction : surface.program->value_instructions()) {
    if (instruction.source_node == separate) {
      require(instruction.static_u0 == 0u && instruction.static_u1 == 0u &&
                  instruction.result_type == SocketType::floating &&
                  instruction.operands.size() ==
                      value_operand::separate_color::count,
              "Separate XYZ did not use the RGB component projection ABI");
      const auto operation = instruction.operation;
      const auto index = operation == ValueOperation::separate_r   ? 0u
                         : operation == ValueOperation::separate_g ? 1u
                         : operation == ValueOperation::separate_b ? 2u
                                                                   : 3u;
      require(index < separate_components.size(),
              "Separate XYZ emitted a non-component operation");
      separate_components[index] = true;
    }
    if (instruction.source_node == combine) {
      require(instruction.operation == ValueOperation::combine_color &&
                  instruction.static_u0 == 0u && instruction.static_u1 == 0u &&
                  instruction.result_type == SocketType::vector &&
                  instruction.operands.size() ==
                      value_operand::combine_color::count,
              "Combine XYZ did not preserve vector typing over RGB packing");
      ++combine_count;
    }
  }
  require(std::ranges::all_of(separate_components,
                              [](bool present) { return present; }) &&
              combine_count == 1u,
          "dynamic Separate/Combine XYZ lowering is incomplete");

  const auto parameters =
      bind_surface_parameters(*surface.program, *shader.program);
  require(parameters.ok(),
          "dynamic Separate/Combine XYZ parameter binding failed");
}

} // namespace

int main() {
  test_dynamic_vector_component_lowering();
  return EXIT_SUCCESS;
}

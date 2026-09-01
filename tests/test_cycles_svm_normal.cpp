#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_graph.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

enum class OutputMode : std::uint8_t { dot, normal, both };

struct OracleCase {
  OutputMode mode;
  Vec3f input;
  Vec3f direction;
  std::array<std::uint32_t, 8u> record;
};

constexpr auto oracle_cases = std::array{
    OracleCase{OutputMode::dot,
               {0.0f, 0.0f, 4.0f},
               {0.0f, 0.0f, 2.0f},
               {0x00000048u, 0x00000000u, 0x00000000u, 0x40800000u, 0x000000ffu,
                0x00000000u, 0x00000000u, 0x40000000u}},
    OracleCase{OutputMode::dot,
               {-3.0f, 0.0f, 0.0f},
               {2.0f, 0.0f, 0.0f},
               {0x00000048u, 0xc0400000u, 0x00000000u, 0x00000000u, 0x000000ffu,
                0x40000000u, 0x00000000u, 0x00000000u}},
    OracleCase{OutputMode::dot,
               {4.0f, -2.0f, 1.0f},
               {1.0f, 2.0f, 3.0f},
               {0x00000048u, 0x40800000u, 0xc0000000u, 0x3f800000u, 0x000000ffu,
                0x3f800000u, 0x40000000u, 0x40400000u}},
    OracleCase{OutputMode::dot,
               {1.0f, 2.0f, 3.0f},
               {0.0f, 0.0f, 0.0f},
               {0x00000048u, 0x3f800000u, 0x40000000u, 0x40400000u, 0x000000ffu,
                0x00000000u, 0x00000000u, 0x00000000u}},
    OracleCase{OutputMode::dot,
               {0.0f, 0.0f, 0.0f},
               {1.0f, 2.0f, 3.0f},
               {0x00000048u, 0x00000000u, 0x00000000u, 0x00000000u, 0x000000ffu,
                0x3f800000u, 0x40000000u, 0x40400000u}},
    OracleCase{OutputMode::normal,
               {0.0f, 3.0f, 0.0f},
               {2.0f, 0.0f, 0.0f},
               {0x00000048u, 0x00000000u, 0x40400000u, 0x00000000u, 0x0000ff00u,
                0x40000000u, 0x00000000u, 0x00000000u}},
    OracleCase{OutputMode::normal,
               {8.0f, 1.0f, -2.0f},
               {0.0f, -3.0f, 4.0f},
               {0x00000048u, 0x41000000u, 0x3f800000u, 0xc0000000u, 0x0000ff00u,
                0x00000000u, 0xc0400000u, 0x40800000u}},
    OracleCase{OutputMode::normal,
               {1.0f, 2.0f, 3.0f},
               {0.0f, 0.0f, 0.0f},
               {0x00000048u, 0x3f800000u, 0x40000000u, 0x40400000u, 0x0000ff00u,
                0x00000000u, 0x00000000u, 0x00000000u}},
    OracleCase{OutputMode::both,
               {-4.0f, 5.0f, 1.0f},
               {1.0f, -2.0f, 3.0f},
               {0x00000048u, 0xc0800000u, 0x40a00000u, 0x3f800000u, 0x00000300u,
                0x3f800000u, 0xc0000000u, 0x40400000u}},
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] ShaderGraph make_graph(const OracleCase &test) {
  ShaderGraph graph;
  const auto normal = graph.add_node(node_type::normal, "Normal");
  require(graph.set_input(normal, "Normal", SocketValue::normal(test.input)),
          "failed to set Normal input");
  require(graph.set_property(normal, "Direction",
                             SocketValue::vector(test.direction)),
          "failed to set Normal direction");
  const auto emission = graph.add_node(node_type::emission, "Emission");

  if (test.mode != OutputMode::dot) {
    const auto to_vector =
        graph.add_node(node_type::normal_to_vector, "Normal to Vector");
    const auto to_color =
        graph.add_node(node_type::vector_to_color, "Vector to Color");
    require(graph.connect({normal, "Normal"}, to_vector, "Normal") &&
                graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
                graph.connect({to_color, "Color"}, emission, "Color"),
            "failed to keep Normal output live");
  }
  if (test.mode != OutputMode::normal) {
    require(graph.connect({normal, "Dot"}, emission, "Strength"),
            "failed to keep Dot output live");
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderCompilation compile_frontend(const ShaderGraph &graph) {
  const ShaderCompiler compiler{make_core_node_registry()};
  auto result = compiler.compile(graph);
  if (!result.ok()) {
    for (const auto &diagnostic : result.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(result.ok(), "Normal graph failed frontend validation");
  return result;
}

[[nodiscard]] std::array<std::uint32_t, 8u>
normal_record(const ShaderImage &image) {
  for (auto index = std::size_t{}; index + 7u < image.words.size(); ++index) {
    if (image.words[index] == static_cast<std::uint32_t>(NODE_NORMAL)) {
      std::array<std::uint32_t, 8u> result{};
      std::copy_n(image.words.begin() + index, result.size(), result.begin());
      return result;
    }
  }
  require(false, "compiled shader has no NODE_NORMAL record");
  return {};
}

[[nodiscard]] ShaderImage compile_svm(const ShaderProgram &program) {
  AttributeIDMap attributes;
  ImageIDMap images;
  auto image = compile_shader(program, attributes, images,
                              ShaderCompileContext{.background = false});
  require(image.valid && image.node_types_used[NODE_NORMAL],
          "Normal SVM compilation failed");
  return image;
}

void test_schema() {
  const auto registry = make_core_node_registry();
  const auto *schema = registry.find(node_type::normal);
  require(schema != nullptr && schema->inputs.size() == 1u &&
              schema->outputs.size() == 2u && schema->properties.size() == 1u &&
              schema->inputs[0u].name == "Normal" &&
              schema->inputs[0u].type == SocketType::normal &&
              schema->outputs[0u].name == "Normal" &&
              schema->outputs[0u].type == SocketType::normal &&
              schema->outputs[1u].name == "Dot" &&
              schema->outputs[1u].type == SocketType::floating &&
              schema->properties[0u].name == "Direction" &&
              schema->properties[0u].type == SocketType::vector &&
              schema->properties[0u].role == PropertyRole::runtime_parameter,
          "Normal typed schema differs from Cycles");

  const auto svm_node = make_graph_node(node_type::normal);
  require(svm_node != nullptr && svm_node->is_linear_operation() &&
              svm_node->shader_node_type() == NODE_NORMAL,
          "Normal host-node classification differs from Cycles");
}

void test_external_payloads() {
  for (const auto &test : oracle_cases) {
    const auto compiled = compile_frontend(make_graph(test));
    require(normal_record(compile_svm(*compiled.program)) == test.record,
            "NODE_NORMAL record differs from external Cycles 5.2.1 oracle");
  }
}

void test_direction_is_material_data() {
  auto first = oracle_cases.back();
  auto second = first;
  second.direction = {0.25f, -0.5f, 0.75f};
  const auto first_program = compile_frontend(make_graph(first));
  const auto second_program = compile_frontend(make_graph(second));
  require(
      first_program.program->analysis().structure_signature ==
              second_program.program->analysis().structure_signature &&
          first_program.program->analysis().parameter_signature !=
              second_program.program->analysis().parameter_signature &&
          first_program.program->analysis().runtime_properties.size() == 1u &&
          first_program.program->analysis().runtime_properties[0u].property ==
              "Direction",
      "authored Normal direction changed shader structure identity");
}

} // namespace

int main() {
  test_schema();
  test_external_payloads();
  test_direction_is_material_data();
  return EXIT_SUCCESS;
}

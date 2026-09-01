#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_graph.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

using Record = std::array<std::uint32_t, 5u>;

enum class OutputMode : std::uint8_t { quadratic, linear, constant, all };

struct OracleCase {
  OutputMode mode;
  float strength;
  float smooth;
  bool stack_inputs;
  std::vector<Record> records;
};

const auto oracle_cases = std::array{
    OracleCase{OutputMode::quadratic,
               8.0f,
               0.0f,
               false,
               {{0x00000049u, 0x00000000u, 0x41000000u, 0x00000000u,
                 0x00000000u}}},
    OracleCase{OutputMode::linear,
               3.5f,
               2.0f,
               false,
               {{0x00000049u, 0x00000001u, 0x40600000u, 0x40000000u,
                 0x00000000u}}},
    OracleCase{OutputMode::constant,
               0.25f,
               4.0f,
               false,
               {{0x00000049u, 0x00000002u, 0x3e800000u, 0x40800000u,
                 0x00000000u}}},
    OracleCase{OutputMode::all,
               6.0f,
               1.5f,
               false,
               {{0x00000049u, 0x00000000u, 0x40c00000u, 0x3fc00000u,
                 0x00000000u},
                {0x00000049u, 0x00000001u, 0x40c00000u, 0x3fc00000u,
                 0x00000001u},
                {0x00000049u, 0x00000002u, 0x40c00000u, 0x3fc00000u,
                 0x00000002u}}},
    OracleCase{OutputMode::all,
               0.0f,
               0.0f,
               true,
               {{0x00000049u, 0x00000000u, 0x7fc00000u, 0x7fc00001u,
                 0x00000002u},
                {0x00000049u, 0x00000001u, 0x7fc00000u, 0x7fc00001u,
                 0x00000003u},
                {0x00000049u, 0x00000002u, 0x7fc00000u, 0x7fc00001u,
                 0x00000004u}}},
    OracleCase{OutputMode::constant,
               2.0f,
               -1.0f,
               false,
               {{0x00000049u, 0x00000002u, 0x40000000u, 0xbf800000u,
                 0x00000000u}}},
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] std::vector<std::string_view> output_names(OutputMode mode) {
  switch (mode) {
  case OutputMode::quadratic:
    return {"Quadratic"};
  case OutputMode::linear:
    return {"Linear"};
  case OutputMode::constant:
    return {"Constant"};
  case OutputMode::all:
    return {"Quadratic", "Linear", "Constant"};
  }
  return {};
}

[[nodiscard]] ShaderGraph make_graph(const OracleCase &test) {
  ShaderGraph graph;
  const auto falloff =
      graph.add_node(node_type::light_falloff, "Light Falloff");
  if (test.stack_inputs) {
    const auto light_path =
        graph.add_node(node_type::light_path, "Light Path");
    require(graph.connect({light_path, "RayLength"}, falloff, "Strength") &&
                graph.connect({light_path, "RayDepth"}, falloff, "Smooth"),
            "failed to connect stack-backed Light Falloff inputs");
  } else {
    require(graph.set_input(falloff, "Strength",
                            SocketValue::floating(test.strength)) &&
                graph.set_input(falloff, "Smooth",
                                SocketValue::floating(test.smooth)),
            "failed to set immediate Light Falloff inputs");
  }

  std::vector<OutputRef> closures;
  for (const auto name : output_names(test.mode)) {
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(graph.connect({falloff, std::string{name}}, emission, "Strength"),
            "failed to keep Light Falloff output live");
    closures.push_back({emission, "Closure"});
  }
  while (closures.size() > 1u) {
    const auto left = closures.back();
    closures.pop_back();
    const auto right = closures.back();
    closures.pop_back();
    const auto add = graph.add_node(node_type::add_closure, "Add");
    require(graph.connect(left, add, "A") &&
                graph.connect(right, add, "B"),
            "failed to combine Light Falloff outputs");
    closures.push_back({add, "Closure"});
  }
  graph.set_root(ShaderDomain::surface, closures.front());
  return graph;
}

[[nodiscard]] ShaderProgram compile_frontend(const ShaderGraph &graph) {
  const ShaderCompiler compiler{make_core_node_registry()};
  auto result = compiler.compile(graph);
  if (!result.ok()) {
    for (const auto &diagnostic : result.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(result.ok(), "Light Falloff graph failed frontend validation");
  return *result.program;
}

[[nodiscard]] std::vector<Record> falloff_records(const ShaderImage &image) {
  std::vector<Record> records;
  for (auto index = std::size_t{}; index + 4u < image.words.size(); ++index) {
    if (image.words[index] != static_cast<std::uint32_t>(NODE_LIGHT_FALLOFF)) {
      continue;
    }
    Record record{};
    std::copy_n(image.words.begin() + index, record.size(), record.begin());
    records.push_back(record);
  }
  return records;
}

[[nodiscard]] ShaderImage compile_svm(const ShaderProgram &program) {
  AttributeIDMap attributes;
  ImageIDMap images;
  auto image = compile_shader(program, attributes, images,
                              ShaderCompileContext{.background = false});
  require(image.valid && image.node_types_used[NODE_LIGHT_FALLOFF],
          "Light Falloff SVM compilation failed");
  return image;
}

void test_schema() {
  const auto registry = make_core_node_registry();
  const auto *schema = registry.find(node_type::light_falloff);
  require(schema != nullptr && schema->inputs.size() == 2u &&
              schema->outputs.size() == 3u &&
              schema->inputs[0u].name == "Strength" &&
              schema->inputs[1u].name == "Smooth" &&
              schema->outputs[0u].name == "Quadratic" &&
              schema->outputs[1u].name == "Linear" &&
              schema->outputs[2u].name == "Constant",
          "Light Falloff typed schema differs from Cycles");

  const auto node = make_graph_node(node_type::light_falloff);
  require(node != nullptr && node->shader_node_type() == NODE_LIGHT_FALLOFF,
          "Light Falloff host node is not the Cycles opcode node");
}

void test_external_payloads() {
  for (const auto &test : oracle_cases) {
    const auto program = compile_frontend(make_graph(test));
    require(falloff_records(compile_svm(program)) == test.records,
            "NODE_LIGHT_FALLOFF records differ from external Cycles 5.2.1 oracle");
  }
}

} // namespace

int main() {
  test_schema();
  test_external_payloads();
  return EXIT_SUCCESS;
}

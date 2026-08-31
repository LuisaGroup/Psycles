#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;
using psycles::Vec3f;

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-6f) noexcept {
  return std::abs(actual - expected) <=
         tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] std::vector<packed_float4> linear_table() {
  std::vector<packed_float4> table;
  table.reserve(257u);
  for (auto index = std::uint32_t{}; index <= 256u; ++index) {
    const auto t = static_cast<float>(index) / 256.0f;
    table.emplace_back(packed_float4{
        t, 0.25f + 0.5f * t, 1.0f - t, 1.0f});
  }
  return table;
}

[[nodiscard]] std::string
table_string(const std::vector<packed_float4> &table) {
  std::ostringstream stream;
  stream << std::setprecision(9);
  for (auto index = std::size_t{}; index < table.size(); ++index) {
    if (index != 0u) {
      stream << ';';
    }
    const auto denominator = std::max<std::size_t>(table.size() - 1u, 1u);
    stream << static_cast<double>(index) / static_cast<double>(denominator)
           << ',' << table[index].x << ',' << table[index].y << ','
           << table[index].z;
  }
  return stream.str();
}

[[nodiscard]] ShaderGraph make_graph(
    const std::vector<packed_float4> &table, float factor, Vec3f color,
    bool dynamic_color, bool sampled = true, float min_x = 0.1f,
    float max_x = 0.9f, bool extrapolate = true,
    bool linked_constants = false) {
  ShaderGraph graph;
  const auto curve = graph.add_node(node_type::rgb_curve, "RGB Curves");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid =
      graph.set_input(curve, "Factor", SocketValue::floating(factor)) &&
      graph.set_input(curve, "Color", SocketValue::color(color)) &&
      graph.set_property(curve, "Sampled", SocketValue::boolean(sampled)) &&
      graph.set_property(curve, "MinX", SocketValue::floating(min_x)) &&
      graph.set_property(curve, "MaxX", SocketValue::floating(max_x)) &&
      graph.set_property(curve, "Extrapolate",
                         SocketValue::boolean(extrapolate)) &&
      graph.set_property(curve, "Table",
                         SocketValue::string(table_string(table)));
  if (linked_constants) {
    const auto constant_color =
        graph.add_node(node_type::constant_color, "Linked Color");
    const auto constant_factor =
        graph.add_node(node_type::constant_float, "Linked Factor");
    valid = valid &&
            graph.set_input(constant_color, "Color",
                            SocketValue::color(color)) &&
            graph.set_input(constant_factor, "Value",
                            SocketValue::floating(factor)) &&
            graph.connect({constant_color, "Color"}, curve, "Color") &&
            graph.connect({constant_factor, "Value"}, curve, "Factor");
  } else if (dynamic_color) {
    const auto coordinates = graph.add_node(node_type::texture_coordinate,
                                            "Generated Coordinates");
    const auto point_to_vector =
        graph.add_node(node_type::point_to_vector, "Point to Vector");
    const auto vector_to_color =
        graph.add_node(node_type::vector_to_color, "Vector to Color");
    valid = valid &&
            graph.connect({coordinates, "Generated"}, point_to_vector,
                          "Point") &&
            graph.connect({point_to_vector, "Vector"}, vector_to_color,
                          "Vector") &&
            graph.connect({vector_to_color, "Color"}, curve, "Color");
  }
  valid = valid && graph.connect({curve, "Color"}, emission, "Color");
  require(valid, "failed to construct RGB Curves graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderImage compile_graph(ShaderGraph &graph) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "RGB Curves graph failed frontend validation");
  AttributeIDMap attributes;
  ImageIDMap images;
  return compile_shader(*shader.program, attributes, images,
                        ShaderCompileContext{.background = false});
}

void test_cycles_reference_constant_fold() {
  auto table = linear_table();
  auto graph = make_graph(table, 0.5f, {0.3f, 0.5f, 0.7f}, false);
  const auto image = compile_graph(graph);
  require(image.valid, image.diagnostic.c_str());
  static constexpr std::array structure{
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
      0x00000005u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  require(image.words.size() == structure.size(),
          "RGB Curves constant-fold stream has the wrong size");
  for (auto index = std::size_t{}; index < structure.size(); ++index) {
    if (index < 5u || index > 7u) {
      require(image.words[index] == structure[index],
              "RGB Curves constant-fold structure differs from Cycles");
    }
  }
  require(near(std::bit_cast<float>(image.words[5u]), 0.275f) &&
              near(std::bit_cast<float>(image.words[6u]), 0.5f) &&
              near(std::bit_cast<float>(image.words[7u]), 0.475f),
          "RGB Curves constant fold differs from the Cycles source oracle");
}

void test_linked_cycles_value_and_color_fold() {
  auto table = linear_table();
  auto graph = make_graph(table, 0.5f, {0.3f, 0.5f, 0.7f}, false, true,
                          0.1f, 0.9f, true, true);
  const auto image = compile_graph(graph);
  require(image.valid, image.diagnostic.c_str());
  require(std::find(image.words.begin(), image.words.end(),
                    static_cast<std::uint32_t>(NODE_CURVES)) ==
              image.words.end() &&
              std::find(image.words.begin(), image.words.end(),
                        static_cast<std::uint32_t>(NODE_VALUE_F)) ==
                  image.words.end() &&
              std::find(image.words.begin(), image.words.end(),
                        static_cast<std::uint32_t>(NODE_VALUE_V)) ==
                  image.words.end(),
          "linked Cycles Value/Color nodes did not fold before RGB Curves");
  const auto weight = std::find(
      image.words.begin(), image.words.end(),
      static_cast<std::uint32_t>(NODE_CLOSURE_SET_WEIGHT));
  require(weight != image.words.end() && image.words.end() - weight >= 4u &&
              near(std::bit_cast<float>(weight[1u]), 0.275f) &&
              near(std::bit_cast<float>(weight[2u]), 0.5f) &&
              near(std::bit_cast<float>(weight[3u]), 0.475f),
          "linked Cycles Value/Color fold produced the wrong curve value");
}

void test_dynamic_payload() {
  const auto table = linear_table();
  auto graph = make_graph(table, 0.37f, {0.0f, 0.0f, 0.0f}, true, true,
                          -0.25f, 1.3f, false);
  const auto image = compile_graph(graph);
  require(image.valid, image.diagnostic.c_str());
  const auto opcode = std::find(image.words.begin(), image.words.end(),
                                static_cast<std::uint32_t>(NODE_CURVES));
  require(opcode != image.words.end(), "RGB Curves opcode is missing");
  const auto begin = static_cast<std::size_t>(opcode - image.words.begin());
  require(begin + 9u + table.size() * 4u <= image.words.size(),
          "RGB Curves payload or table is truncated");
  require((image.words[begin + 1u] >> 8u) ==
                  (SVM_INPUT_STACK_OFFSET_MASK >> 8u) &&
              image.words[begin + 2u] == 0u &&
              image.words[begin + 3u] == 0u,
          "RGB Curves Color is not a typed float3 stack input");
  require(image.words[begin + 4u] == std::bit_cast<std::uint32_t>(0.37f) &&
              image.words[begin + 5u] ==
                  std::bit_cast<std::uint32_t>(-0.25f) &&
              image.words[begin + 6u] ==
                  std::bit_cast<std::uint32_t>(1.3f) &&
              image.words[begin + 7u] == 257u,
          "RGB Curves scalar header differs from Cycles");
  const auto packed = image.words[begin + 8u];
  require((packed & 0xffu) == 0u &&
              ((packed >> 8u) & 0xffu) != SVM_STACK_INVALID &&
              (packed >> 16u) == 0u,
          "RGB Curves extrapolate/output bytes differ from Cycles");
  const auto table_word = [&](std::uint32_t sample,
                              std::uint32_t component) {
    return image.words[begin + 9u + sample * 4u + component];
  };
  require(table_word(0u, 0u) == std::bit_cast<std::uint32_t>(0.0f) &&
              table_word(128u, 0u) ==
                  std::bit_cast<std::uint32_t>(0.5f) &&
              table_word(128u, 1u) ==
                  std::bit_cast<std::uint32_t>(0.5f) &&
              table_word(256u, 2u) ==
                  std::bit_cast<std::uint32_t>(0.0f) &&
              table_word(0u, 3u) ==
                  std::bit_cast<std::uint32_t>(1.0f) &&
              table_word(256u, 3u) ==
                  std::bit_cast<std::uint32_t>(1.0f),
          "RGB Curves table is not Cycles float4 data");
}

void test_zero_factor_bypass() {
  auto table = linear_table();
  auto graph = make_graph(table, 0.0f, {0.0f, 0.0f, 0.0f}, true);
  const auto image = compile_graph(graph);
  require(image.valid, image.diagnostic.c_str());
  require(std::find(image.words.begin(), image.words.end(),
                    static_cast<std::uint32_t>(NODE_CURVES)) ==
              image.words.end(),
          "RGB Curves zero-Factor node did not bypass its linked Color");
}

void test_schema_and_invalid_tables() {
  const auto registry = make_core_node_registry();
  const auto *schema = registry.find(node_type::rgb_curve);
  require(schema != nullptr && schema->inputs.size() == 2u &&
              schema->inputs[0u].name == "Factor" &&
              schema->inputs[0u].type == SocketType::floating &&
              schema->inputs[1u].name == "Color" &&
              schema->inputs[1u].type == SocketType::color &&
              schema->outputs.size() == 1u &&
              schema->outputs[0u].name == "Color" &&
              schema->outputs[0u].type == SocketType::color,
          "RGB Curves schema lost its typed inputs or output");

  auto table = linear_table();
  auto unsampled = make_graph(table, 1.0f, {0.3f, 0.5f, 0.7f}, true,
                              false);
  require(!compile_graph(unsampled).valid,
          "unsampled control-point RGB Curves was accepted as Cycles bytecode");
  table.resize(1u);
  auto truncated = make_graph(table, 1.0f, {0.3f, 0.5f, 0.7f}, true);
  require(!compile_graph(truncated).valid,
          "one-entry RGB Curves table was accepted");
}

} // namespace

int main() {
  test_cycles_reference_constant_fold();
  test_linked_cycles_value_and_color_fold();
  test_dynamic_payload();
  test_zero_factor_bypass();
  test_schema_and_invalid_tables();
  return EXIT_SUCCESS;
}

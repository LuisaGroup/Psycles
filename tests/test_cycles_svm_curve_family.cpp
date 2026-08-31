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
#include <string_view>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;
using psycles::Vec3f;

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

void require(bool condition, std::string_view message) {
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

[[nodiscard]] std::vector<packed_float4> vector_table() {
  std::vector<packed_float4> table;
  table.reserve(257u);
  for (auto index = std::uint32_t{}; index <= 256u; ++index) {
    const auto t = static_cast<float>(index) / 256.0f;
    table.emplace_back(packed_float4{t, 0.25f + 0.5f * t, 1.0f - t, 1.0f});
  }
  return table;
}

[[nodiscard]] std::vector<float> float_table() {
  std::vector<float> table;
  table.reserve(257u);
  for (auto index = std::uint32_t{}; index <= 256u; ++index) {
    const auto t = static_cast<float>(index) / 256.0f;
    table.emplace_back(0.1f + 0.7f * t);
  }
  return table;
}

[[nodiscard]] std::string
vector_table_string(const std::vector<packed_float4> &table) {
  std::ostringstream stream;
  stream << std::setprecision(9);
  const auto denominator = std::max<std::size_t>(table.size() - 1u, 1u);
  for (auto index = std::size_t{}; index < table.size(); ++index) {
    if (index != 0u) {
      stream << ';';
    }
    stream << static_cast<double>(index) / static_cast<double>(denominator)
           << ',' << table[index].x << ',' << table[index].y << ','
           << table[index].z;
  }
  return stream.str();
}

[[nodiscard]] std::string float_table_string(const std::vector<float> &table) {
  std::ostringstream stream;
  stream << std::setprecision(9);
  const auto denominator = std::max<std::size_t>(table.size() - 1u, 1u);
  for (auto index = std::size_t{}; index < table.size(); ++index) {
    if (index != 0u) {
      stream << ';';
    }
    stream << static_cast<double>(index) / static_cast<double>(denominator)
           << ',' << table[index];
  }
  return stream.str();
}

[[nodiscard]] ShaderGraph
make_vector_graph(const std::vector<packed_float4> &table, float factor,
                  Vec3f value, bool dynamic_value, bool sampled = true,
                  float min_x = 0.1f, float max_x = 0.9f,
                  bool extrapolate = true) {
  ShaderGraph graph;
  const auto curve = graph.add_node(node_type::vector_curve, "Vector Curves");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid =
      graph.set_input(curve, "Factor", SocketValue::floating(factor)) &&
      graph.set_input(curve, "Vector", SocketValue::vector(value)) &&
      graph.set_property(curve, "Sampled", SocketValue::boolean(sampled)) &&
      graph.set_property(curve, "MinX", SocketValue::floating(min_x)) &&
      graph.set_property(curve, "MaxX", SocketValue::floating(max_x)) &&
      graph.set_property(curve, "Extrapolate",
                         SocketValue::boolean(extrapolate)) &&
      graph.set_property(curve, "Table",
                         SocketValue::string(vector_table_string(table)));
  if (dynamic_value) {
    const auto coordinates =
        graph.add_node(node_type::texture_coordinate, "Generated Coordinates");
    const auto point_to_vector =
        graph.add_node(node_type::point_to_vector, "Point to Vector");
    valid =
        valid &&
        graph.connect({coordinates, "Generated"}, point_to_vector, "Point") &&
        graph.connect({point_to_vector, "Vector"}, curve, "Vector");
  }
  valid = valid && graph.connect({curve, "Vector"}, to_color, "Vector") &&
          graph.connect({to_color, "Color"}, emission, "Color");
  require(valid, "failed to construct Vector Curves graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph
make_float_graph(const std::vector<float> &table, float factor, float value,
                 bool dynamic_value, bool sampled = true, float min_x = -0.25f,
                 float max_x = 1.25f, bool extrapolate = true) {
  ShaderGraph graph;
  const auto curve = graph.add_node(node_type::float_curve, "Float Curve");
  const auto to_color =
      graph.add_node(node_type::scalar_to_color, "Value to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid =
      graph.set_input(curve, "Factor", SocketValue::floating(factor)) &&
      graph.set_input(curve, "Value", SocketValue::floating(value)) &&
      graph.set_property(curve, "Sampled", SocketValue::boolean(sampled)) &&
      graph.set_property(curve, "MinX", SocketValue::floating(min_x)) &&
      graph.set_property(curve, "MaxX", SocketValue::floating(max_x)) &&
      graph.set_property(curve, "Extrapolate",
                         SocketValue::boolean(extrapolate)) &&
      graph.set_property(curve, "Table",
                         SocketValue::string(float_table_string(table)));
  if (dynamic_value) {
    const auto coordinates =
        graph.add_node(node_type::texture_coordinate, "Generated Coordinates");
    const auto point_to_vector =
        graph.add_node(node_type::point_to_vector, "Point to Vector");
    const auto vector_to_color =
        graph.add_node(node_type::vector_to_color, "Vector to Color");
    const auto separate = graph.add_node(node_type::separate_xyz, "Separate X");
    valid =
        valid &&
        graph.connect({coordinates, "Generated"}, point_to_vector, "Point") &&
        graph.connect({point_to_vector, "Vector"}, vector_to_color, "Vector") &&
        graph.connect({vector_to_color, "Color"}, separate, "Vector") &&
        graph.connect({separate, "X"}, curve, "Value");
  }
  valid = valid && graph.connect({curve, "Value"}, to_color, "Value") &&
          graph.connect({to_color, "Color"}, emission, "Color");
  require(valid, "failed to construct Float Curve graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderImage compile_graph(ShaderGraph &graph) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "curve-family graph failed frontend validation");
  AttributeIDMap attributes;
  ImageIDMap images;
  return compile_shader(*shader.program, attributes, images,
                        ShaderCompileContext{.background = false});
}

[[nodiscard]] std::array<float, 3u> closure_weight(const ShaderImage &image) {
  const auto weight =
      std::find(image.words.begin(), image.words.end(),
                static_cast<std::uint32_t>(NODE_CLOSURE_SET_WEIGHT));
  require(weight != image.words.end() && image.words.end() - weight >= 4u,
          "constant curve graph has no closure weight");
  return {std::bit_cast<float>(weight[1u]), std::bit_cast<float>(weight[2u]),
          std::bit_cast<float>(weight[3u])};
}

void test_typed_schemas() {
  const auto registry = make_core_node_registry();
  const auto *vector = registry.find(node_type::vector_curve);
  require(vector != nullptr && vector->inputs.size() == 2u &&
              vector->inputs[0u].name == "Factor" &&
              vector->inputs[0u].type == SocketType::floating &&
              vector->inputs[1u].name == "Vector" &&
              vector->inputs[1u].type == SocketType::vector &&
              vector->outputs.size() == 1u &&
              vector->outputs[0u].name == "Vector" &&
              vector->outputs[0u].type == SocketType::vector,
          "Vector Curves schema lost its typed sockets");
  const auto *scalar = registry.find(node_type::float_curve);
  require(scalar != nullptr && scalar->inputs.size() == 2u &&
              scalar->inputs[0u].name == "Factor" &&
              scalar->inputs[0u].type == SocketType::floating &&
              scalar->inputs[1u].name == "Value" &&
              scalar->inputs[1u].type == SocketType::floating &&
              scalar->outputs.size() == 1u &&
              scalar->outputs[0u].name == "Value" &&
              scalar->outputs[0u].type == SocketType::floating,
          "Float Curve schema lost its typed sockets");
}

void test_constant_folding_and_zero_factor_bypass() {
  auto vectors = vector_table();
  auto vector_graph =
      make_vector_graph(vectors, 0.5f, {0.3f, 0.5f, 0.7f}, false);
  const auto vector_image = compile_graph(vector_graph);
  require(vector_image.valid, vector_image.diagnostic);
  const auto vector_weight = closure_weight(vector_image);
  require(near(vector_weight[0u], 0.275f) && near(vector_weight[1u], 0.5f) &&
              near(vector_weight[2u], 0.475f) &&
              !vector_image.node_types_used[NODE_CURVES],
          "Vector Curves constant fold differs from Cycles 5.2.1");

  auto scalars = float_table();
  auto float_graph =
      make_float_graph(scalars, 0.4f, 0.25f, false, true, 0.0f, 1.0f);
  const auto float_image = compile_graph(float_graph);
  require(float_image.valid, float_image.diagnostic);
  const auto scalar_weight = closure_weight(float_image);
  require(near(scalar_weight[0u], 0.26f) && near(scalar_weight[1u], 0.26f) &&
              near(scalar_weight[2u], 0.26f) &&
              !float_image.node_types_used[NODE_FLOAT_CURVE],
          "Float Curve constant fold differs from Cycles 5.2.1");

  auto vector_bypass = make_vector_graph(vectors, 0.0f, {}, true);
  const auto vector_bypass_image = compile_graph(vector_bypass);
  require(vector_bypass_image.valid &&
              !vector_bypass_image.node_types_used[NODE_CURVES],
          "Vector Curves zero-Factor node did not bypass its linked value");
  auto float_bypass = make_float_graph(scalars, 0.0f, 0.0f, true);
  const auto float_bypass_image = compile_graph(float_bypass);
  require(float_bypass_image.valid &&
              !float_bypass_image.node_types_used[NODE_FLOAT_CURVE],
          "Float Curve zero-Factor node did not bypass its linked value");
}

void test_dynamic_payloads_match_external_cycles_5_2_1() {
  auto vectors = vector_table();
  constexpr std::array vector_rows{0u, 64u, 128u, 192u, 256u};
  constexpr std::array vector_oracle{
      std::array{0x3df5c28fu, 0x3f6147aeu, 0x3e75c28fu},
      std::array{0x3f540e2cu, 0x3e3bb925u, 0x3f66d4a5u},
      std::array{0x3f17f9deu, 0x3f0aae90u, 0x3f4bfc0eu},
      std::array{0x3e6ec68eu, 0x3f4780f0u, 0x3de3b8ebu},
      std::array{0x3f68defau, 0x3e23d6feu, 0x3f383affu}};
  for (auto row = std::size_t{}; row < vector_rows.size(); ++row) {
    const auto sample = vector_rows[row];
    vectors[sample] =
        packed_float4{f32(vector_oracle[row][0u]), f32(vector_oracle[row][1u]),
                      f32(vector_oracle[row][2u]), 1.0f};
  }
  auto vector_graph =
      make_vector_graph(vectors, 1.0f, {}, true, true, 0.0f, 1.0f, true);
  const auto vector_image = compile_graph(vector_graph);
  require(vector_image.valid, vector_image.diagnostic);
  const auto vector_opcode =
      std::find(vector_image.words.begin(), vector_image.words.end(),
                static_cast<std::uint32_t>(NODE_CURVES));
  require(vector_opcode != vector_image.words.end(),
          "Vector Curves opcode is missing");
  const auto vector_begin =
      static_cast<std::size_t>(vector_opcode - vector_image.words.begin());
  require(vector_begin + 9u + 257u * 4u <= vector_image.words.size(),
          "Vector Curves payload is truncated");
  require((vector_image.words[vector_begin + 1u] >> 8u) ==
                  (SVM_INPUT_STACK_OFFSET_MASK >> 8u) &&
              vector_image.words[vector_begin + 2u] == 0u &&
              vector_image.words[vector_begin + 3u] == 0u &&
              vector_image.words[vector_begin + 4u] == 0x3f800000u &&
              vector_image.words[vector_begin + 5u] == 0u &&
              vector_image.words[vector_begin + 6u] == 0x3f800000u &&
              vector_image.words[vector_begin + 7u] == 257u &&
              (vector_image.words[vector_begin + 8u] & 0xffu) == 1u &&
              ((vector_image.words[vector_begin + 8u] >> 8u) & 0xffu) !=
                  SVM_STACK_INVALID,
          "Vector Curves header differs from Cycles 5.2.1");
  for (auto row = std::size_t{}; row < vector_rows.size(); ++row) {
    const auto base = vector_begin + 9u + vector_rows[row] * 4u;
    require(vector_image.words[base + 0u] == vector_oracle[row][0u] &&
                vector_image.words[base + 1u] == vector_oracle[row][1u] &&
                vector_image.words[base + 2u] == vector_oracle[row][2u] &&
                vector_image.words[base + 3u] == 0x3f800000u,
            "Vector Curves external Cycles table row changed");
  }

  auto scalars = float_table();
  constexpr std::array float_oracle{0x3da3d70au, 0x3f638f51u, 0x3e8c20f8u,
                                    0x3f1fd781u, 0x3e9eb850u};
  for (auto row = std::size_t{}; row < vector_rows.size(); ++row) {
    scalars[vector_rows[row]] = f32(float_oracle[row]);
  }
  auto float_graph =
      make_float_graph(scalars, 1.0f, 0.0f, true, true, 0.0f, 1.0f, true);
  const auto float_image = compile_graph(float_graph);
  require(float_image.valid, float_image.diagnostic);
  const auto float_opcode =
      std::find(float_image.words.begin(), float_image.words.end(),
                static_cast<std::uint32_t>(NODE_FLOAT_CURVE));
  require(float_opcode != float_image.words.end(),
          "Float Curve opcode is missing");
  const auto float_begin =
      static_cast<std::size_t>(float_opcode - float_image.words.begin());
  require(float_begin + 7u + 257u <= float_image.words.size(),
          "Float Curve payload is truncated");
  require(float_image.words[float_begin + 1u] == 0x3f800000u &&
              (float_image.words[float_begin + 2u] >> 8u) ==
                  (SVM_INPUT_STACK_OFFSET_MASK >> 8u) &&
              float_image.words[float_begin + 3u] == 0u &&
              float_image.words[float_begin + 4u] == 0x3f800000u &&
              float_image.words[float_begin + 5u] == 257u &&
              (float_image.words[float_begin + 6u] & 0xffu) == 1u &&
              ((float_image.words[float_begin + 6u] >> 8u) & 0xffu) !=
                  SVM_STACK_INVALID,
          "Float Curve header differs from Cycles 5.2.1");
  for (auto row = std::size_t{}; row < vector_rows.size(); ++row) {
    require(float_image.words[float_begin + 7u + vector_rows[row]] ==
                float_oracle[row],
            "Float Curve external Cycles table row changed");
  }
}

void test_invalid_sampled_tables_are_rejected() {
  auto vectors = vector_table();
  auto unsampled_vector = make_vector_graph(vectors, 1.0f, {}, true, false);
  require(!compile_graph(unsampled_vector).valid,
          "unsampled Vector Curves table was accepted");
  vectors.resize(1u);
  auto one_vector = make_vector_graph(vectors, 1.0f, {}, true);
  require(!compile_graph(one_vector).valid,
          "one-entry Vector Curves table was accepted");

  auto scalars = float_table();
  auto unsampled_float = make_float_graph(scalars, 1.0f, 0.0f, true, false);
  require(!compile_graph(unsampled_float).valid,
          "unsampled Float Curve table was accepted");
  scalars.resize(1u);
  auto one_float = make_float_graph(scalars, 1.0f, 0.0f, true);
  require(!compile_graph(one_float).valid,
          "one-entry Float Curve table was accepted");
}

} // namespace

int main() {
  test_typed_schemas();
  test_constant_folding_and_zero_factor_bypass();
  test_dynamic_payloads_match_external_cycles_5_2_1();
  test_invalid_sampled_tables_are_rejected();
  return EXIT_SUCCESS;
}

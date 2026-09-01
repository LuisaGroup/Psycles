#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_graph.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] ShaderImage compile_graph(ShaderGraph &graph) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    std::exit(EXIT_FAILURE);
  }
  AttributeIDMap attributes;
  ImageIDMap images;
  return compile_shader(*shader.program, attributes, images,
                        ShaderCompileContext{.background = false});
}

[[nodiscard]] NodeId add_generated_coordinates(ShaderGraph &graph,
                                               NodeId texture) {
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Texture Coordinate");
  const auto convert =
      graph.add_node(node_type::point_to_vector, "Generated Point to Vector");
  require(graph.connect({coordinates, "Generated"}, convert, "Point") &&
              graph.connect({convert, "Vector"}, texture, "Vector"),
          "failed to bind generated texture coordinates");
  return coordinates;
}

void keep_texture_outputs_live(ShaderGraph &graph, NodeId texture,
                               bool factor) {
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid = graph.connect({texture, "Color"}, emission, "Color");
  if (factor) {
    valid = valid && graph.connect({texture, "Factor"}, emission, "Strength");
  }
  require(valid, "failed to keep procedural texture outputs live");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
}

template <std::size_t N>
void require_record(const ShaderImage &image,
                    const std::array<std::uint32_t, N> &expected,
                    std::string_view label) {
  const auto begin = std::search(image.words.begin(), image.words.end(),
                                 expected.begin(), expected.end());
  if (begin != image.words.end()) {
    return;
  }
  std::cerr << label << " differs from the external Cycles 5.2.1 record\n"
            << "  expected";
  for (const auto word : expected) {
    std::cerr << " 0x" << std::hex << word;
  }
  std::cerr << std::dec << '\n';
  std::exit(EXIT_FAILURE);
}

void test_wave_record() {
  ShaderGraph graph;
  const auto wave = graph.add_node(node_type::wave_texture, "Wave Texture");
  static_cast<void>(add_generated_coordinates(graph, wave));
  require(
      graph.set_property(wave, "WaveType", SocketValue::string("BANDS")) &&
          graph.set_property(wave, "BandsDirection",
                             SocketValue::string("X")) &&
          graph.set_property(wave, "RingsDirection",
                             SocketValue::string("X")) &&
          graph.set_property(wave, "Profile", SocketValue::string("SIN")) &&
          graph.set_input(wave, "Scale", SocketValue::floating(-1.7f)) &&
          graph.set_input(wave, "Distortion", SocketValue::floating(0.0f)) &&
          graph.set_input(wave, "Detail", SocketValue::floating(2.0f)) &&
          graph.set_input(wave, "DetailScale", SocketValue::floating(1.0f)) &&
          graph.set_input(wave, "DetailRoughness",
                          SocketValue::floating(0.5f)) &&
          graph.set_input(wave, "PhaseOffset", SocketValue::floating(-0.83f)),
      "failed to author Wave Texture");
  keep_texture_outputs_live(graph, wave, false);
  const auto image = compile_graph(graph);
  require(image.valid && image.node_types_used[NODE_TEX_WAVE],
          "Wave Texture did not compile to NODE_TEX_WAVE");

  // NODE_TEX_WAVE from Cycles 5.2.1 shader "Wave 00 BANDS X SIN".
  // The source scene uses a stack coordinate at 0, live Color at 3, and a
  // dead Fac. Its scalar words and enum words are copied without repacking.
  static constexpr std::array expected{0x00000043u, 0x00000000u, 0x00000000u,
                                       0x00000000u, 0x00000000u, 0xbfd9999au,
                                       0x00000000u, 0x40000000u, 0x3f800000u,
                                       0x3f000000u, 0xbf547ae1u, 0x00ff0300u};
  require_record(image, expected, "NODE_TEX_WAVE");
}

void test_magic_record() {
  ShaderGraph graph;
  const auto magic = graph.add_node(node_type::magic_texture, "Magic Texture");
  static_cast<void>(add_generated_coordinates(graph, magic));
  require(
      graph.set_property(magic, "Depth", SocketValue::unsigned_integer(6u)) &&
          graph.set_input(magic, "Scale", SocketValue::floating(4.25f)) &&
          graph.set_input(magic, "Distortion", SocketValue::floating(0.37f)),
      "failed to author Magic Texture");
  keep_texture_outputs_live(graph, magic, false);
  const auto image = compile_graph(graph);
  require(image.valid && image.node_types_used[NODE_TEX_MAGIC],
          "Magic Texture did not compile to NODE_TEX_MAGIC");

  // NODE_TEX_MAGIC from Cycles 5.2.1 shader "Magic 15 Depth 6 Color".
  static constexpr std::array expected{0x00000044u, 0x40880000u, 0x3ebd70a4u,
                                       0xff030006u};
  require_record(image, expected, "NODE_TEX_MAGIC");
}

void test_checker_record() {
  ShaderGraph graph;
  const auto checker =
      graph.add_node(node_type::checker_texture, "Checker Texture");
  static_cast<void>(add_generated_coordinates(graph, checker));
  require(graph.set_input(checker, "Color1",
                          SocketValue::color({0.13f, 0.37f, 0.79f})) &&
              graph.set_input(checker, "Color2",
                              SocketValue::color({0.83f, 0.61f, 0.17f})) &&
              graph.set_input(checker, "Scale", SocketValue::floating(1.0f)),
          "failed to author Checker Texture");
  keep_texture_outputs_live(graph, checker, true);
  const auto image = compile_graph(graph);
  require(image.valid && image.node_types_used[NODE_TEX_CHECKER],
          "Checker Texture did not compile to NODE_TEX_CHECKER");

  // NODE_TEX_CHECKER from Cycles 5.2.1 shader "Checker 00". Both outputs
  // are live: coordinate=0, Color=3, Fac=6.
  static constexpr std::array expected{0x00000045u, 0x3e051eb8u, 0x3ebd70a4u,
                                       0x3f4a3d71u, 0x3f547ae1u, 0x3f1c28f6u,
                                       0x3e2e147bu, 0x3f800000u, 0x00060300u};
  require_record(image, expected, "NODE_TEX_CHECKER");
}

void test_brick_record() {
  ShaderGraph graph;
  const auto brick = graph.add_node(node_type::brick_texture, "Brick Texture");
  static_cast<void>(add_generated_coordinates(graph, brick));
  require(
      graph.set_input(brick, "Color1",
                      SocketValue::color({0.78f, 0.08f, 0.03f})) &&
          graph.set_input(brick, "Color2",
                          SocketValue::color({0.12f, 0.43f, 0.91f})) &&
          graph.set_input(brick, "Mortar",
                          SocketValue::color({0.07f, 0.21f, 0.13f})) &&
          graph.set_input(brick, "Scale", SocketValue::floating(5.3f)) &&
          graph.set_input(brick, "MortarSize", SocketValue::floating(0.036f)) &&
          graph.set_input(brick, "MortarSmooth",
                          SocketValue::floating(0.017f)) &&
          graph.set_input(brick, "Bias", SocketValue::floating(-0.14f)) &&
          graph.set_input(brick, "BrickWidth", SocketValue::floating(0.53f)) &&
          graph.set_input(brick, "RowHeight", SocketValue::floating(0.19f)) &&
          graph.set_property(brick, "OffsetAmount",
                             SocketValue::floating(0.37f)) &&
          graph.set_property(brick, "OffsetFrequency",
                             SocketValue::unsigned_integer(3u)) &&
          graph.set_property(brick, "SquashAmount",
                             SocketValue::floating(0.72f)) &&
          graph.set_property(brick, "SquashFrequency",
                             SocketValue::unsigned_integer(2u)),
      "failed to author Brick Texture");
  keep_texture_outputs_live(graph, brick, true);
  const auto image = compile_graph(graph);
  require(image.valid && image.node_types_used[NODE_TEX_BRICK],
          "Brick Texture did not compile to NODE_TEX_BRICK");

  // NODE_TEX_BRICK from Cycles 5.2.1 shader "Brick Texture Probe".
  static constexpr std::array expected{
      0x00000046u, 0x3f47ae14u, 0x3da3d70au, 0x3cf5c28fu, 0x3df5c28fu,
      0x3edc28f6u, 0x3f68f5c3u, 0x3d8f5c29u, 0x3e570a3du, 0x3e051eb8u,
      0x40a9999au, 0x3d1374bcu, 0xbe0f5c29u, 0x3f07ae14u, 0x3e428f5cu,
      0x3c8b4396u, 0x3ebd70a4u, 0x3f3851ecu, 0x03000203u, 0x00000006u};
  require_record(image, expected, "NODE_TEX_BRICK");
}

void test_schema_and_point_projection() {
  const auto registry = make_core_node_registry();
  for (const auto type :
       {node_type::wave_texture, node_type::magic_texture,
        node_type::checker_texture, node_type::brick_texture}) {
    const auto *schema = registry.find(type);
    require(schema != nullptr && !schema->inputs.empty() &&
                schema->inputs.front().name == "Vector" &&
                schema->outputs.size() == 2u &&
                schema->outputs[0u].name == "Color" &&
                schema->outputs[1u].name == "Factor",
            "procedural texture schema differs from Blender/Cycles");
    const auto node = make_graph_node(type);
    require(node != nullptr, "procedural texture has no Cycles SVM host node");
  }

  ShaderGraph graph;
  const auto checker = graph.add_node(node_type::checker_texture, "Checker");
  static_cast<void>(add_generated_coordinates(graph, checker));
  keep_texture_outputs_live(graph, checker, false);
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "point-projection fixture failed validation");
  const auto projected = CyclesGraph::project(*shader.program);
  const auto node = std::find_if(
      projected.nodes().begin(), projected.nodes().end(), [](const auto &item) {
        return item->type == node_type::checker_texture;
      });
  const auto *vector =
      node == projected.nodes().end() ? nullptr : (*node)->input("Vector");
  require(projected.valid() && vector != nullptr &&
              vector->type == GraphSocketType::point &&
              vector->link != nullptr &&
              vector->link->parent->type == node_type::texture_coordinate,
          "procedural texture Vector did not restore Cycles' POINT socket");
}

void test_invalid_properties_fail_closed() {
  ShaderGraph graph;
  const auto wave = graph.add_node(node_type::wave_texture, "Invalid Wave");
  require(
      graph.set_property(wave, "Profile", SocketValue::string("NOT_A_PROFILE")),
      "failed to author invalid Wave profile");
  keep_texture_outputs_live(graph, wave, false);
  const auto image = compile_graph(graph);
  require(!image.valid,
          "invalid Wave enum silently changed Cycles SVM semantics");
}

} // namespace

int main() {
  test_schema_and_point_projection();
  test_wave_record();
  test_magic_record();
  test_checker_record();
  test_brick_record();
  test_invalid_properties_fail_closed();
  return EXIT_SUCCESS;
}

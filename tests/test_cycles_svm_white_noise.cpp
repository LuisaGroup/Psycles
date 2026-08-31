#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <span>
#include <string_view>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void require_words(std::span<const std::uint32_t> actual,
                   std::span<const std::uint32_t> expected,
                   std::string_view label) {
  if (actual.size() != expected.size()) {
    std::cerr << label << " word count differs from Cycles 5.2.1: got "
              << actual.size() << ", expected " << expected.size() << '\n';
    std::exit(EXIT_FAILURE);
  }
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << label << " differs at word " << index << ": got 0x"
                << std::hex << actual[index] << ", expected 0x"
                << expected[index] << std::dec << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

[[nodiscard]] ShaderGraph make_white_noise_graph(std::uint32_t dimensions,
                                                  bool color) {
  ShaderGraph graph;
  const auto white =
      graph.add_node(node_type::white_noise_texture, "White Noise");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid =
      graph.set_property(white, "Dimensions",
                         SocketValue::unsigned_integer(dimensions));
  if (dimensions != 1u) {
    valid = valid &&
            graph.set_input(
                white, "Vector",
                SocketValue::vector({0.173f, -0.625f, 1.375f}));
  }
  if (dimensions == 1u || dimensions == 4u) {
    valid = valid &&
            graph.set_input(white, "W", SocketValue::floating(-0.437f));
  }
  if (color) {
    valid = valid && graph.connect({white, "Color"}, emission, "Color");
  } else {
    const auto scalar_to_color =
        graph.add_node(node_type::scalar_to_color, "Value to Color");
    valid = valid &&
            graph.connect({white, "Value"}, scalar_to_color, "Value") &&
            graph.connect({scalar_to_color, "Color"}, emission, "Color");
  }
  require(valid, "failed to construct Cycles White Noise SVM graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_linked_white_noise_graph() {
  ShaderGraph graph;
  const auto combine =
      graph.add_node(node_type::combine_xyz, "White Noise Constant Vector");
  const auto white = graph.add_node(node_type::white_noise_texture,
                                    "White Noise 4D Linked Constant");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid =
      graph.set_input(combine, "X", SocketValue::floating(0.173f)) &&
      graph.set_input(combine, "Y", SocketValue::floating(-0.625f)) &&
      graph.set_input(combine, "Z", SocketValue::floating(1.375f)) &&
      graph.set_property(white, "Dimensions",
                         SocketValue::unsigned_integer(4u)) &&
      graph.set_input(white, "W", SocketValue::floating(-0.437f)) &&
      graph.connect({combine, "Vector"}, white, "Vector") &&
      graph.connect({white, "Color"}, emission, "Color");
  require(valid, "failed to construct linked White Noise fold graph");
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
    std::exit(EXIT_FAILURE);
  }
  AttributeIDMap attributes;
  ImageIDMap images;
  return compile_shader(*shader.program, attributes, images,
                        ShaderCompileContext{.background = false});
}

void test_external_streams_match_cycles_5_2_1() {
  // Shader-local streams from Cycles 5.2.1 at 9e2066aef7ef, with only the
  // global jump offsets rebased. The diagnostic build does not alter SVM
  // compilation or evaluation.
  static constexpr std::array value_2d{
      0x00000001u, 0x00000004u, 0x00000016u, 0x00000017u,
      0x00000047u, 0x00000002u, 0x3e3126e9u, 0xbf200000u,
      0x3fb00000u, 0x00000000u, 0x0000ff00u, 0x0000000du,
      0x00000000u, 0x00000100u, 0x00000007u, 0x7fc00001u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x00000003u,
      0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};
  static constexpr std::array value_1d{
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
      0x00000005u, 0x3f61eb51u, 0x3f61eb51u, 0x3f61eb51u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  static constexpr std::array color_1d{
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
      0x00000005u, 0x3f61eb51u, 0x3c430e8bu, 0x3e5761e7u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  static constexpr std::array color_4d_folded{
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
      0x00000005u, 0x3ea4e93cu, 0x3dea2fcbu, 0x3db5df56u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  static constexpr std::array color_4d{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u,
      0x00000047u, 0x00000004u, 0x3e3126e9u, 0xbf200000u,
      0x3fb00000u, 0xbedfbe77u, 0x000000ffu, 0x00000007u,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};

  auto value_1d_graph = make_white_noise_graph(1u, false);
  const auto value_1d_image = compile_graph(value_1d_graph);
  require(value_1d_image.valid, value_1d_image.diagnostic.c_str());
  require_words(std::span{value_1d_image.words}, value_1d,
                "White Noise Value 1D fold");

  auto color_1d_graph = make_white_noise_graph(1u, true);
  const auto color_1d_image = compile_graph(color_1d_graph);
  require(color_1d_image.valid, color_1d_image.diagnostic.c_str());
  require_words(std::span{color_1d_image.words}, color_1d,
                "White Noise Color 1D fold");

  auto color_4d_fold_graph = make_linked_white_noise_graph();
  const auto color_4d_fold_image = compile_graph(color_4d_fold_graph);
  require(color_4d_fold_image.valid,
          color_4d_fold_image.diagnostic.c_str());
  require_words(std::span{color_4d_fold_image.words}, color_4d_folded,
                "White Noise Color 4D linked fold");

  auto value_graph = make_white_noise_graph(2u, false);
  const auto value_image = compile_graph(value_graph);
  require(value_image.valid, value_image.diagnostic.c_str());
  require_words(std::span{value_image.words}, value_2d,
                "White Noise Value 2D");

  auto color_graph = make_white_noise_graph(4u, true);
  const auto color_image = compile_graph(color_graph);
  require(color_image.valid, color_image.diagnostic.c_str());
  require_words(std::span{color_image.words}, color_4d,
                "White Noise Color 4D");
}

void test_payload_cross_product() {
  for (auto dimensions = std::uint32_t{1u}; dimensions <= 4u;
       ++dimensions) {
    for (const auto color : {false, true}) {
      auto graph = make_white_noise_graph(dimensions, color);
      const auto image = compile_graph(graph);
      require(image.valid, image.diagnostic.c_str());
      const auto opcode = std::find(image.words.begin() + 4,
                                    image.words.end(),
                                    NODE_TEX_WHITE_NOISE);
      if (dimensions == 1u) {
        require(opcode == image.words.end(),
                "constant White Noise 1D escaped Blender-equivalent fold");
        continue;
      }
      require(opcode != image.words.end(),
              "White Noise opcode is missing from the SVM stream");
      const auto offset = static_cast<std::size_t>(
          std::distance(image.words.begin(), opcode)) + 1u;
      require(offset + 6u <= image.words.size(),
              "White Noise payload extends beyond the SVM stream");
      const std::array expected{
          dimensions,
          0x3e3126e9u,
          0xbf200000u,
          0x3fb00000u,
          dimensions == 4u ? 0xbedfbe77u : 0u,
          color ? 0x000000ffu : 0x0000ff00u};
      require_words(std::span{image.words}.subspan(offset, expected.size()),
                    expected, "White Noise payload");
    }
  }
}

void test_invalid_dimensions_are_rejected() {
  for (const auto dimensions : {0u, 5u}) {
    auto graph = make_white_noise_graph(dimensions, true);
    const auto image = compile_graph(graph);
    require(!image.valid, "invalid White Noise dimensions were accepted");
  }
}

} // namespace

int main() {
  test_external_streams_match_cycles_5_2_1();
  test_payload_cross_product();
  test_invalid_dimensions_are_rejected();
  return EXIT_SUCCESS;
}

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_graph.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <variant>

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
    std::cerr << label << " word count differs: got " << actual.size()
              << ", expected " << expected.size() << "; actual:";
    for (const auto word : actual) {
      std::cerr << " 0x" << std::hex << word;
    }
    std::cerr << std::dec << '\n';
    std::exit(EXIT_FAILURE);
  }
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << label << " differs at word " << index << ": got 0x"
                << std::hex << actual[index] << ", expected 0x"
                << expected[index] << "; actual:";
      for (const auto word : actual) {
        std::cerr << " 0x" << word;
      }
      std::cerr << std::dec << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

[[nodiscard]] ShaderImage compile_vertex_color(
    std::string_view layer_name, std::string_view output_name,
    AttributeIDMap &attribute_ids) {
  ShaderGraph graph;
  const auto vertex_color =
      graph.add_node(node_type::vertex_color, "Vertex Color");
  require(graph.set_property(
              vertex_color, "Layer Name",
              SocketValue::string(std::string{layer_name})),
          "failed to set Vertex Color layer name");

  const auto emission = graph.add_node(node_type::emission, "Emission");
  if (output_name == "Color") {
    require(graph.connect({vertex_color, "Color"}, emission, "Color"),
            "failed to connect Vertex Color Color");
  } else {
    require(output_name == "Alpha", "invalid Vertex Color output in test");
    require(graph.set_input(
                emission, "Color",
                SocketValue::color({0.31f, 0.57f, 0.83f})) &&
                graph.connect({vertex_color, "Alpha"}, emission, "Strength"),
            "failed to connect Vertex Color Alpha");
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "Vertex Color graph did not validate");
  const auto image = compile_shader(
      *shader.program, attribute_ids,
      ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic.c_str());
  return image;
}

[[nodiscard]] ShaderImage compile_attribute(
    std::string_view attribute_name, std::string_view output_name,
    AttributeIDMap &attribute_ids) {
  ShaderGraph graph;
  const auto attribute = graph.add_node(node_type::attribute, "Attribute");
  require(graph.set_property(
              attribute, "Attribute",
              SocketValue::string(std::string{attribute_name})),
          "failed to set Attribute name");

  const auto emission = graph.add_node(node_type::emission, "Emission");
  if (output_name == "Color") {
    require(graph.connect({attribute, "Color"}, emission, "Color"),
            "failed to connect Attribute Color");
  } else {
    require(output_name == "Fac" || output_name == "Alpha",
            "invalid Attribute output in test");
    require(graph.set_input(
                emission, "Color",
                SocketValue::color({0.31f, 0.57f, 0.83f})) &&
                graph.connect(
                    OutputRef{attribute, std::string{output_name}}, emission,
                    "Strength"),
            "failed to connect scalar Attribute output");
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "Attribute graph did not validate");
  const auto image = compile_shader(
      *shader.program, attribute_ids,
      ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic.c_str());
  return image;
}

[[nodiscard]] ShaderImage compile_attribute_bump(
    bool use_vertex_color, AttributeIDMap &attribute_ids) {
  ShaderGraph graph;
  const auto attribute = graph.add_node(
      use_vertex_color ? node_type::vertex_color : node_type::attribute,
      use_vertex_color ? "Vertex Color" : "Attribute");
  require(
      graph.set_property(
          attribute, use_vertex_color ? "Layer Name" : "Attribute",
          SocketValue::string("ProbeColor")),
      "failed to set bump attribute name");

  const auto bump = graph.add_node(node_type::bump, "Attribute Bump");
  const auto normal_to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto vector_to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.connect({attribute, "Alpha"}, bump, "Height") &&
          graph.set_input(bump, "Strength", SocketValue::floating(0.73f)) &&
          graph.set_input(bump, "Distance", SocketValue::floating(0.41f)) &&
          graph.set_input(bump, "FilterWidth",
                          SocketValue::floating(0.29f)) &&
          graph.set_property(bump, "Invert", SocketValue::boolean(false)) &&
          graph.set_property(bump, "UseObjectSpace",
                             SocketValue::boolean(false)) &&
          graph.connect({bump, "Normal"}, normal_to_vector, "Normal") &&
          graph.connect({normal_to_vector, "Vector"}, vector_to_color,
                        "Vector") &&
          graph.connect({vector_to_color, "Color"}, emission, "Color"),
      "failed to construct Attribute Bump graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "Attribute Bump graph did not validate");
  const auto image = compile_shader(
      *shader.program, attribute_ids,
      ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic.c_str());
  return image;
}

void test_scene_wide_attribute_ids_match_cycles_5_2_1() {
  AttributeIDMap attribute_ids;
  require(attribute_ids.get_attribute_id("ProbeColor") == ATTR_STD_NUM,
          "first named attribute did not receive ATTR_STD_NUM");
  require(attribute_ids.get_attribute_id("ProbeColor") == ATTR_STD_NUM,
          "repeated named attribute did not reuse its Cycles ID");
  require(attribute_ids.get_attribute_id("ProbeColorSecond") ==
              ATTR_STD_NUM + 1u,
          "second named attribute did not receive the next Cycles ID");
  require(AttributeIDMap::get_attribute_id(ATTR_STD_VERTEX_COLOR) ==
              ATTR_STD_VERTEX_COLOR,
          "standard attribute ID was remapped");
}

void test_vertex_color_streams_match_cycles_5_2_1() {
  AttributeIDMap attribute_ids;
  const auto named_color =
      compile_vertex_color("ProbeColor", "Color", attribute_ids);
  const auto named_alpha =
      compile_vertex_color("ProbeColor", "Alpha", attribute_ids);
  const auto named_second =
      compile_vertex_color("ProbeColorSecond", "Color", attribute_ids);
  const auto default_color = compile_vertex_color("", "Color", attribute_ids);

  // Frozen from shaders 5--8 of the Cycles 5.2.1 `svm_vertex_color`
  // diagnostic dump. Only the global jump-table relocation is normalized.
  static constexpr std::array<std::uint32_t, 17u> expected_named_color{
      0x00000001u, 0x00000004u, 0x0000000fu, 0x00000010u, 0x00000017u,
      0x00ff0023u, 0x00000000u, 0x00000007u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u};
  static constexpr std::array<std::uint32_t, 17u> expected_named_alpha{
      0x00000001u, 0x00000004u, 0x0000000fu, 0x00000010u, 0x00000017u,
      0x0000ff23u, 0x00000000u, 0x00000007u, 0x3e9eb852u, 0x3f11eb85u,
      0x3f547ae1u, 0x7fc00000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u};
  static constexpr auto expected_named_second = [] {
    auto words = expected_named_color;
    words[5] = 0x00ff0024u;
    return words;
  }();
  static constexpr auto expected_default_color = [] {
    auto words = expected_named_color;
    words[5] = 0x00ff000au;
    return words;
  }();

  require_words(named_color.words, expected_named_color,
                "named Vertex Color Color Cycles oracle");
  require_words(named_alpha.words, expected_named_alpha,
                "named Vertex Color Alpha Cycles oracle");
  require_words(named_second.words, expected_named_second,
                "second named Vertex Color Cycles oracle");
  require_words(default_color.words, expected_default_color,
                "default Vertex Color Cycles oracle");
  require(named_color.peak_stack_usage == 3u &&
              named_alpha.peak_stack_usage == 1u &&
              named_second.peak_stack_usage == 3u &&
              default_color.peak_stack_usage == 3u,
          "Vertex Color stack lifetime differs from Cycles");
  require(named_color.node_types_used[NODE_VERTEX_COLOR] &&
              named_alpha.node_types_used[NODE_VERTEX_COLOR] &&
              named_second.node_types_used[NODE_VERTEX_COLOR] &&
              default_color.node_types_used[NODE_VERTEX_COLOR],
          "Vertex Color opcode usage differs from Cycles");
}

void test_attribute_streams_match_cycles_5_2_1() {
  AttributeIDMap attribute_ids;
  static_cast<void>(
      compile_vertex_color("ProbeColor", "Color", attribute_ids));
  static_cast<void>(
      compile_vertex_color("ProbeColorSecond", "Color", attribute_ids));
  const auto named_color =
      compile_attribute("ProbeColor", "Color", attribute_ids);
  const auto named_alpha =
      compile_attribute("ProbeColorSecond", "Alpha", attribute_ids);
  const auto standard_fac =
      compile_attribute("pointiness", "Fac", attribute_ids);
  const auto empty_color = compile_attribute("", "Color", attribute_ids);

  // Frozen from shaders 9--11 of the same Cycles 5.2.1 dump.
  static constexpr std::array<std::uint32_t, 18u> expected_named_color{
      0x00000001u, 0x00000004u, 0x00000010u, 0x00000011u, 0x00000015u,
      0x00000023u, 0x00000000u, 0x00000000u, 0x00000007u, 0x7fc00000u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  static constexpr std::array<std::uint32_t, 18u> expected_named_alpha{
      0x00000001u, 0x00000004u, 0x00000010u, 0x00000011u, 0x00000015u,
      0x00000024u, 0x00000200u, 0x00000000u, 0x00000007u, 0x3e9eb852u,
      0x3f11eb85u, 0x3f547ae1u, 0x7fc00000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  static constexpr std::array<std::uint32_t, 18u> expected_standard_fac{
      0x00000001u, 0x00000004u, 0x00000010u, 0x00000011u, 0x00000015u,
      0x00000020u, 0x00000100u, 0x00000000u, 0x00000007u, 0x3e9eb852u,
      0x3f11eb85u, 0x3f547ae1u, 0x7fc00000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  static constexpr auto expected_empty_color = [] {
    auto words = expected_named_color;
    words[5] = 0x00000025u;
    return words;
  }();

  require_words(named_color.words, expected_named_color,
                "named Attribute Color Cycles oracle");
  require_words(named_alpha.words, expected_named_alpha,
                "named Attribute Alpha Cycles oracle");
  require_words(standard_fac.words, expected_standard_fac,
                "standard Attribute Fac Cycles oracle");
  require_words(empty_color.words, expected_empty_color,
                "empty Attribute Color Cycles oracle");
  require(named_color.peak_stack_usage == 3u &&
              named_alpha.peak_stack_usage == 1u &&
              standard_fac.peak_stack_usage == 1u &&
              empty_color.peak_stack_usage == 3u,
          "Attribute stack lifetime differs from Cycles");
  require(named_color.node_types_used[NODE_ATTR] &&
              named_alpha.node_types_used[NODE_ATTR] &&
              standard_fac.node_types_used[NODE_ATTR] &&
              empty_color.node_types_used[NODE_ATTR],
          "Attribute opcode usage differs from Cycles");
}

void test_attribute_bump_streams_match_cycles_5_2_1() {
  AttributeIDMap attribute_ids;
  const auto vertex_color = compile_attribute_bump(true, attribute_ids);
  const auto attribute = compile_attribute_bump(false, attribute_ids);

  // Frozen from shaders 15 and 16 of the Cycles 5.2.1
  // `svm_vertex_color` diagnostic dump. VertexColorNode intentionally leaves
  // the three refined samples as NODE_VERTEX_COLOR, while AttributeNode
  // explicitly selects NODE_ATTR_DERIVATIVE for CENTER/DX/DY.
  static constexpr std::array<std::uint32_t, 32u> expected_vertex_color{
      0x00000001u, 0x00000004u, 0x0000001eu, 0x0000001fu, 0x00000017u,
      0x0000ff23u, 0x3e947ae1u, 0x0000000bu, 0x01000001u, 0x00000000u,
      0x00000017u, 0x0104ff23u, 0x3e947ae1u, 0x00000017u, 0x0205ff23u,
      0x3e947ae1u, 0x00000021u, 0x3ed1eb85u, 0x3f3ae148u, 0x3e947ae1u,
      0x00000001u, 0xff060504u, 0x00000007u, 0x7fc00006u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u};
  static constexpr std::array<std::uint32_t, 35u> expected_attribute{
      0x00000001u, 0x00000004u, 0x00000021u, 0x00000022u, 0x00000016u,
      0x00000023u, 0x00000200u, 0x3e947ae1u, 0x0000000bu, 0x01000001u,
      0x00000000u, 0x00000016u, 0x00000023u, 0x00010204u, 0x3e947ae1u,
      0x00000016u, 0x00000023u, 0x00020205u, 0x3e947ae1u, 0x00000021u,
      0x3ed1eb85u, 0x3f3ae148u, 0x3e947ae1u, 0x00000001u, 0xff060504u,
      0x00000007u, 0x7fc00006u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};

  require_words(vertex_color.words, expected_vertex_color,
                "Vertex Color Bump Cycles oracle");
  require_words(attribute.words, expected_attribute,
                "Attribute Bump Cycles oracle");
  require(vertex_color.peak_stack_usage == 9u &&
              attribute.peak_stack_usage == 9u,
          "Attribute Bump stack lifetime differs from Cycles");
  require(vertex_color.node_types_used[NODE_VERTEX_COLOR] &&
              !vertex_color.node_types_used[NODE_VERTEX_COLOR_DERIVATIVE] &&
              attribute.node_types_used[NODE_ATTR_DERIVATIVE],
          "Attribute Bump derivative opcode selection differs from Cycles");
}

[[nodiscard]] bool projected_attribute_stochastic(bool nonlinear) {
  ShaderGraph graph;
  const auto attribute = graph.add_node(node_type::attribute, "Density");
  require(graph.set_property(
              attribute, "Attribute", SocketValue::string("density")),
          "failed to set volume Attribute name");

  OutputRef density{attribute, "Fac"};
  if (nonlinear) {
    const auto math = graph.add_node(node_type::math, "Nonlinear Density");
    require(graph.set_property(math, "Operation",
                               SocketValue::string("SINE")) &&
                graph.connect(density, math, "A"),
            "failed to construct nonlinear volume path");
    density = OutputRef{math, "Value"};
  }

  const auto volume =
      graph.add_node(node_type::volume_absorption, "Volume Absorption");
  require(graph.connect(density, volume, "Density"),
          "failed to connect volume Attribute");
  graph.set_root(ShaderDomain::volume,
                 OutputRef{.node = volume, .socket = "Volume"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "volume Attribute graph did not validate");
  const auto projected = CyclesGraph::project(*shader.program);
  require(projected.valid(), projected.diagnostic().c_str());
  for (const auto &node : projected.nodes()) {
    if (node->type != node_type::attribute) {
      continue;
    }
    const auto iter = node->properties.find("Stochastic");
    require(iter != node->properties.end(),
            "projected Attribute lost Stochastic property");
    const auto *value = std::get_if<bool>(&iter->second.value);
    require(value != nullptr,
            "projected Attribute Stochastic property is not boolean");
    return *value;
  }
  require(false, "projected volume graph lost Attribute node");
  return false;
}

void test_volume_stochastic_path_matches_cycles_5_2_1() {
  require(projected_attribute_stochastic(false),
          "linear volume Attribute lost stochastic sampling");
  require(!projected_attribute_stochastic(true),
          "nonlinear volume Attribute retained stochastic sampling");
}

} // namespace

int main() {
  test_scene_wide_attribute_ids_match_cycles_5_2_1();
  test_vertex_color_streams_match_cycles_5_2_1();
  test_attribute_streams_match_cycles_5_2_1();
  test_attribute_bump_streams_match_cycles_5_2_1();
  test_volume_stochastic_path_matches_cycles_5_2_1();
  return EXIT_SUCCESS;
}

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
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

struct LegacyMappingCase {
  std::string_view vector_type;
  Vec3f translation;
  Vec3f rotation;
  Vec3f scale;
  std::array<std::string_view, 3u> axes;
  std::span<const std::uint32_t> cycles_surface_words;
};

[[nodiscard]] ShaderGraph make_legacy_mapping_graph(
    const LegacyMappingCase &item) {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Object Coordinates");
  const auto point_to_vector = graph.add_node(
      node_type::point_to_vector, "Object Point to Vector");
  const auto mapping =
      graph.add_node(node_type::mapping, "Texture Mapping Projection");
  const auto image = graph.add_node(node_type::image_texture, "Image Texture");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.set_property(mapping, "LegacyTextureMapping",
                         SocketValue::boolean(true)) &&
          graph.set_property(mapping, "VectorType",
                             SocketValue::string(
                                 std::string{item.vector_type})) &&
          graph.set_property(mapping, "XMapping",
                             SocketValue::string(
                                 std::string{item.axes[0u]})) &&
          graph.set_property(mapping, "YMapping",
                             SocketValue::string(
                                 std::string{item.axes[1u]})) &&
          graph.set_property(mapping, "ZMapping",
                             SocketValue::string(
                                 std::string{item.axes[2u]})) &&
          graph.set_input(mapping, "Location",
                          SocketValue::vector(item.translation)) &&
          graph.set_input(mapping, "Rotation",
                          SocketValue::vector(item.rotation)) &&
          graph.set_input(mapping, "Scale", SocketValue::vector(item.scale)) &&
          graph.connect({coordinates, "Object"}, point_to_vector, "Point") &&
          graph.connect({point_to_vector, "Vector"}, mapping, "Vector") &&
          graph.connect({mapping, "Vector"}, image, "Vector") &&
          graph.set_property(image, "Image",
                             SocketValue::unsigned_integer(41u)) &&
          graph.set_property(image, "Interpolation",
                             SocketValue::string("Closest")) &&
          graph.set_property(image, "Extension",
                             SocketValue::string("REPEAT")) &&
          graph.set_property(image, "Projection",
                             SocketValue::string("FLAT")) &&
          graph.set_property(image, "ColorSpace",
                             SocketValue::string("Non-Color")) &&
          graph.connect({image, "Color"}, emission, "Color"),
      "failed to construct legacy TextureMapping graph");
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
  const auto result = compile_shader(
      *shader.program, attributes, images,
      ShaderCompileContext{.background = false});
  require(result.valid, result.diagnostic.c_str());
  return result;
}

[[nodiscard]] ShaderGraph make_authored_mapping_graph(
    std::string_view vector_type, bool link_vector, Vec3f scale) {
  ShaderGraph graph;
  const auto mapping = graph.add_node(node_type::mapping, "Authored Mapping");
  const auto vector_to_color = graph.add_node(
      node_type::vector_to_color, "Mapping Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid =
      graph.set_property(mapping, "VectorType",
                         SocketValue::string(std::string{vector_type})) &&
      graph.set_input(mapping, "Scale", SocketValue::vector(scale)) &&
      graph.connect({mapping, "Vector"}, vector_to_color, "Vector") &&
      graph.connect({vector_to_color, "Color"}, emission, "Color");
  if (link_vector) {
    const auto coordinates =
        graph.add_node(node_type::texture_coordinate, "Coordinates");
    const auto point_to_vector = graph.add_node(
        node_type::point_to_vector, "Object Point to Vector");
    valid = valid &&
            graph.connect({coordinates, "Object"}, point_to_vector,
                          "Point") &&
            graph.connect({point_to_vector, "Vector"}, mapping, "Vector");
  }
  require(valid, "failed to construct authored Mapping fold graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

void test_texture_mapping_streams_match_cycles_5_2_1() {
  // Exact final shader-local streams observed from the diagnostic Cycles
  // 5.2.1 build at 9e2066aef7ef. The words begin at each shader's surface
  // jump target and include the trailing surface/volume/displacement ENDs.
  static constexpr std::array point{
      0x0000000fu, 0x00000001u, 0x00000000u, 0x00000037u,
      0x00000300u, 0x3f1c1575u, 0xbe916e85u, 0x3d5d8cb2u,
      0x3e051eb8u, 0x3e122f18u, 0x3f8f1488u, 0x3e1d1728u,
      0xbe570a3du, 0x3d8da3eeu, 0x3e49780bu, 0xbf4b22bbu,
      0x3ebd70a4u, 0x0000001cu, 0x00000000u, 0x00000000u,
      0xff060300u, 0x00000007u, 0x7fc00006u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  static constexpr std::array texture{
      0x0000000fu, 0x00000001u, 0x00000000u, 0x00000037u,
      0x00000300u, 0x3dd5bcedu, 0xbea489b9u, 0x3f940737u,
      0x3e28d690u, 0xbf40122au, 0x3dd24099u, 0x3dc51759u,
      0x3e842010u, 0x3e9228f8u, 0x3fd64ee8u, 0x3ee10530u,
      0x3de5cd5bu, 0x0000001cu, 0x00000000u, 0x00000000u,
      0xff060300u, 0x00000007u, 0x7fc00006u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  static constexpr std::array vector{
      0x0000000fu, 0x00000001u, 0x00000000u, 0x00000037u,
      0x00000300u, 0x3c960ef1u, 0xbf31eb11u, 0x00000000u,
      0x00000000u, 0xbddc3df8u, 0x3e08de6bu, 0x00000000u,
      0x00000000u, 0x3ef4956au, 0x3d68679au, 0x00000000u,
      0x00000000u, 0x0000001cu, 0x00000000u, 0x00000000u,
      0xff060300u, 0x00000007u, 0x7fc00006u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  static constexpr std::array normal{
      0x0000000fu, 0x00000001u, 0x00000000u, 0x00000037u,
      0x00000300u, 0x3f9bc097u, 0x3dced0c4u, 0x3f0a7a4du,
      0x00000000u, 0x3ec79127u, 0x3dcb5473u, 0xbfccb4f4u,
      0x00000000u, 0xbe6e160cu, 0x3f31e76du, 0x3e1978ebu,
      0x00000000u, 0x0000002du, 0x0000000bu, 0x7fc00003u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x000003ffu, 0x0000001cu, 0x00000000u,
      0x00000000u, 0xff060300u, 0x00000007u, 0x7fc00006u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x00000003u,
      0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};
  const std::array cases{
      LegacyMappingCase{"POINT", {0.13f, -0.21f, 0.37f},
                        {0.17f, -0.11f, 0.23f},
                        {0.63f, 1.17f, -0.81f}, {"X", "Y", "Z"}, point},
      LegacyMappingCase{"TEXTURE", {-0.19f, 0.31f, -0.07f},
                        {-0.14f, 0.27f, 0.09f},
                        {0.83f, -1.31f, 0.57f}, {"Z", "X", "Y"}, texture},
      LegacyMappingCase{"VECTOR", {}, {0.21f, 0.08f, -0.19f},
                        {-0.71f, 1.23f, 0.49f}, {"Y", "NONE", "X"}, vector},
      LegacyMappingCase{"NORMAL", {}, {-0.09f, 0.18f, 0.31f},
                        {0.77f, -0.59f, 1.41f}, {"X", "Z", "Y"}, normal}};

  for (const auto &item : cases) {
    auto graph = make_legacy_mapping_graph(item);
    const auto image = compile_graph(graph);
    const auto jump_size = std::size_t{4u};
    require(image.words.size() == jump_size + item.cycles_surface_words.size(),
            "TextureMapping local stream size differs from Cycles");
    require_words(std::span{image.words}.subspan(jump_size),
                  item.cycles_surface_words, item.vector_type);
    require(image.node_types_used[NODE_TEXTURE_MAPPING] &&
                !image.node_types_used[NODE_MAPPING],
            "legacy TextureMapping compiled as an authored Mapping node");
    require(image.node_types_used[NODE_VECTOR_MATH] ==
                (item.vector_type == "NORMAL"),
            "TextureMapping NORMAL normalize opcode differs from Cycles");
  }
}

void test_authored_mapping_keeps_node_mapping() {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Coordinates");
  const auto point_to_vector = graph.add_node(
      node_type::point_to_vector, "Object Point to Vector");
  const auto mapping = graph.add_node(node_type::mapping, "Authored Mapping");
  const auto vector_to_color = graph.add_node(
      node_type::vector_to_color, "Mapping Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_property(mapping, "VectorType",
                             SocketValue::string("TEXTURE")) &&
              graph.set_input(mapping, "Location",
                              SocketValue::vector({0.2f, -0.3f, 0.4f})) &&
              graph.set_input(mapping, "Rotation",
                              SocketValue::vector({0.1f, 0.2f, -0.4f})) &&
              graph.set_input(mapping, "Scale",
                              SocketValue::vector({1.2f, 0.7f, -0.8f})) &&
              graph.connect({coordinates, "Object"}, point_to_vector,
                            "Point") &&
              graph.connect({point_to_vector, "Vector"}, mapping,
                            "Vector") &&
              graph.connect({mapping, "Vector"}, vector_to_color,
                            "Vector") &&
              graph.connect({vector_to_color, "Color"}, emission, "Color"),
          "failed to construct authored Mapping graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  const auto image = compile_graph(graph);
  require(image.node_types_used[NODE_MAPPING] &&
              !image.node_types_used[NODE_TEXTURE_MAPPING],
          "authored Mapping did not preserve NODE_MAPPING");
}

void test_mapping_constant_fold_matches_cycles() {
  {
    auto graph = make_authored_mapping_graph(
        "POINT", false, Vec3f{1.2f, 0.7f, -0.8f});
    const auto image = compile_graph(graph);
    require(!image.node_types_used[NODE_MAPPING],
            "constant authored Mapping was not folded");
  }
  {
    auto graph = make_authored_mapping_graph(
        "POINT", true, Vec3f{1.0f, 1.0f, 1.0f});
    const auto image = compile_graph(graph);
    require(!image.node_types_used[NODE_MAPPING],
            "identity POINT Mapping was not bypassed");
  }
  {
    auto graph = make_authored_mapping_graph(
        "VECTOR", true, Vec3f{0.0f, 0.0f, 0.0f});
    const auto image = compile_graph(graph);
    require(!image.node_types_used[NODE_MAPPING],
            "zero-scale Mapping was not folded to zero");
  }
  {
    auto graph = make_authored_mapping_graph(
        "NORMAL", true, Vec3f{1.0f, 1.0f, 1.0f});
    const auto image = compile_graph(graph);
    require(image.node_types_used[NODE_MAPPING],
            "identity NORMAL Mapping incorrectly bypassed normalization");
  }
}

void test_legacy_mapping_projection_rejects_non_texture_consumer() {
  ShaderGraph graph;
  const auto mapping = graph.add_node(
      node_type::mapping, "Malformed Texture Mapping Projection");
  const auto vector_to_color = graph.add_node(
      node_type::vector_to_color, "Non-texture Consumer");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_property(mapping, "LegacyTextureMapping",
                             SocketValue::boolean(true)) &&
              graph.connect({mapping, "Vector"}, vector_to_color,
                            "Vector") &&
              graph.connect({vector_to_color, "Color"}, emission, "Color"),
          "failed to construct malformed legacy Mapping graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "frontend rejected malformed projection fixture");
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto result = compile_shader(
      *shader.program, attributes, images,
      ShaderCompileContext{.background = false});
  require(!result.valid,
          "legacy TextureMapping projected into a non-TextureNode");
}

} // namespace

int main() {
  test_texture_mapping_streams_match_cycles_5_2_1();
  test_authored_mapping_keeps_node_mapping();
  test_mapping_constant_fold_matches_cycles();
  test_legacy_mapping_projection_rejects_non_texture_consumer();
  return EXIT_SUCCESS;
}

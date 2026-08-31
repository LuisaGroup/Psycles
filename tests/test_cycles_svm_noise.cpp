#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

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

struct NoiseCase {
  std::uint32_t dimensions;
  std::string_view type;
  bool normalize;
  bool color;
  float w;
  float scale;
  float detail;
  float roughness;
  float lacunarity;
  float offset;
  float gain;
  float distortion;
};

[[nodiscard]] ShaderGraph make_noise_graph(const NoiseCase &item,
                                           bool embedded_mapping = false) {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Texture Coordinate");
  const auto point_to_vector =
      graph.add_node(node_type::point_to_vector, "Object Point to Vector");
  const auto noise = graph.add_node(node_type::noise_texture, "Noise Texture");
  const auto emission = graph.add_node(node_type::emission, "Emission");

  auto valid =
      graph.connect({coordinates, "Object"}, point_to_vector, "Point") &&
      graph.set_property(noise, "Dimensions",
                         SocketValue::unsigned_integer(item.dimensions)) &&
      graph.set_property(noise, "NoiseType",
                         SocketValue::string(std::string{item.type})) &&
      graph.set_property(noise, "Normalize",
                         SocketValue::boolean(item.normalize)) &&
      graph.set_input(noise, "W", SocketValue::floating(item.w)) &&
      graph.set_input(noise, "Scale", SocketValue::floating(item.scale)) &&
      graph.set_input(noise, "Detail", SocketValue::floating(item.detail)) &&
      graph.set_input(noise, "Roughness",
                      SocketValue::floating(item.roughness)) &&
      graph.set_input(noise, "Lacunarity",
                      SocketValue::floating(item.lacunarity)) &&
      graph.set_input(noise, "Offset", SocketValue::floating(item.offset)) &&
      graph.set_input(noise, "Gain", SocketValue::floating(item.gain)) &&
      graph.set_input(noise, "Distortion",
                      SocketValue::floating(item.distortion));

  if (embedded_mapping) {
    const auto mapping =
        graph.add_node(node_type::mapping, "Legacy TextureMapping");
    valid = valid &&
            graph.set_property(mapping, "LegacyTextureMapping",
                               SocketValue::boolean(true)) &&
            graph.set_property(mapping, "VectorType",
                               SocketValue::string("POINT")) &&
            graph.set_property(mapping, "XMapping",
                               SocketValue::string("X")) &&
            graph.set_property(mapping, "YMapping",
                               SocketValue::string("Y")) &&
            graph.set_property(mapping, "ZMapping",
                               SocketValue::string("Z")) &&
            graph.set_input(mapping, "Location",
                            SocketValue::vector({0.1f, -0.2f, 0.3f})) &&
            graph.connect({point_to_vector, "Vector"}, mapping, "Vector") &&
            graph.connect({mapping, "Vector"}, noise, "Vector");
  } else {
    valid = valid &&
            graph.connect({point_to_vector, "Vector"}, noise, "Vector");
  }

  if (item.color) {
    valid = valid && graph.connect({noise, "Color"}, emission, "Color");
  } else {
    const auto scalar_to_color =
        graph.add_node(node_type::scalar_to_color, "Factor to Color");
    valid = valid &&
            graph.connect({noise, "Factor"}, scalar_to_color, "Value") &&
            graph.connect({scalar_to_color, "Color"}, emission, "Color");
  }
  require(valid, "failed to construct Cycles Noise SVM graph");
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

void test_linked_streams_match_cycles_5_2_1() {
  // Final shader-local streams from Cycles 5.2.1 at 9e2066aef7ef. The
  // diagnostic build only copied the already-compiled global SVM words.
  static constexpr std::array factor_2d{
      0x0000000fu, 0x00000001u, 0x00000000u, 0x00000020u,
      0x00000002u, 0x00000001u, 0x00000000u, 0x00000000u,
      0x40133333u, 0x3fe00000u, 0x3edc28f6u, 0x3fe66666u,
      0x00000000u, 0x3f800000u, 0x00000000u, 0x00ff0300u,
      0x0000000du, 0x00000000u, 0x00000003u, 0x00000007u,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  static constexpr std::array color_3d{
      0x0000000fu, 0x00000001u, 0x00000000u, 0x00000020u,
      0x00000003u, 0x00000001u, 0x00000001u, 0x00000000u,
      0x3fd9999au, 0x40166666u, 0x3f1c28f6u, 0x400ccccdu,
      0x00000000u, 0x3f800000u, 0x3ebd70a4u, 0x0003ff00u,
      0x00000007u, 0x7fc00003u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u};

  auto factor_graph = make_noise_graph(
      {.dimensions = 2u,
       .type = "FBM",
       .normalize = false,
       .color = false,
       .w = 0.0f,
       .scale = 2.3f,
       .detail = 1.75f,
       .roughness = 0.43f,
       .lacunarity = 1.8f,
       .offset = 0.0f,
       .gain = 1.0f,
       .distortion = 0.0f});
  const auto factor_image = compile_graph(factor_graph);
  require(factor_image.valid, factor_image.diagnostic.c_str());
  require_words(std::span{factor_image.words}.subspan(4u), factor_2d,
                "Noise Factor 2D");

  auto color_graph = make_noise_graph(
      {.dimensions = 3u,
       .type = "FBM",
       .normalize = true,
       .color = true,
       .w = 0.0f,
       .scale = 1.7f,
       .detail = 2.35f,
       .roughness = 0.61f,
       .lacunarity = 2.2f,
       .offset = 0.0f,
       .gain = 1.0f,
       .distortion = 0.37f});
  const auto color_image = compile_graph(color_graph);
  require(color_image.valid, color_image.diagnostic.c_str());
  require_words(std::span{color_image.words}.subspan(4u), color_3d,
                "Noise Color 3D");
}

void test_payload_cross_product_and_mapping() {
  struct TypeCase {
    std::string_view name;
    NodeNoiseType value;
  };
  static constexpr std::array types{
      TypeCase{"MULTIFRACTAL", NODE_NOISE_MULTIFRACTAL},
      TypeCase{"FBM", NODE_NOISE_FBM},
      TypeCase{"HYBRID_MULTIFRACTAL", NODE_NOISE_HYBRID_MULTIFRACTAL},
      TypeCase{"RIDGED_MULTIFRACTAL", NODE_NOISE_RIDGED_MULTIFRACTAL},
      TypeCase{"HETERO_TERRAIN", NODE_NOISE_HETERO_TERRAIN}};
  for (const auto &type : types) {
    for (auto dimensions = std::uint32_t{1u}; dimensions <= 4u;
         ++dimensions) {
      for (const auto normalize : {false, true}) {
        auto graph = make_noise_graph(
            {.dimensions = dimensions,
             .type = type.name,
             .normalize = normalize,
             .color = true,
             .w = -0.437f,
             .scale = 2.35f,
             .detail = 2.375f,
             .roughness = 0.63f,
             .lacunarity = 2.17f,
             .offset = 0.37f,
             .gain = 1.11f,
             .distortion = 0.42f});
        const auto image = compile_graph(graph);
        require(image.valid, image.diagnostic.c_str());
        // NODE_TEX_COORD occupies words 4..6. The exact Noise payload starts
        // at word 8 and matches the Cycles matrix oracle byte-for-byte.
        require(image.words.size() >= 20u &&
                    image.words[7u] == NODE_TEX_NOISE,
                "Noise cross-product lost the Cycles opcode position");
        const std::array expected{
            dimensions,
            static_cast<std::uint32_t>(type.value),
            static_cast<std::uint32_t>(normalize),
            0xbedfbe77u,
            0x40166666u,
            0x40180000u,
            0x3f2147aeu,
            0x400ae148u,
            0x3ebd70a4u,
            0x3f8e147bu,
            0x3ed70a3du,
            0x0003ff00u};
        require_words(std::span{image.words}.subspan(8u, expected.size()),
                      expected, type.name);
      }
    }
  }

  auto mapped = make_noise_graph(
      {.dimensions = 3u,
       .type = "FBM",
       .normalize = true,
       .color = true,
       .w = -0.437f,
       .scale = 1.0f,
       .detail = 2.0f,
       .roughness = 0.5f,
       .lacunarity = 2.0f,
       .offset = 0.0f,
       .gain = 1.0f,
       .distortion = 0.0f},
      true);
  const auto mapped_image = compile_graph(mapped);
  require(mapped_image.valid, mapped_image.diagnostic.c_str());
  require(mapped_image.node_types_used[NODE_TEXTURE_MAPPING] &&
              mapped_image.node_types_used[NODE_TEX_NOISE] &&
              !mapped_image.node_types_used[NODE_MAPPING],
          "Noise did not use Cycles embedded TextureMapping");
}

void test_invalid_static_properties_are_rejected() {
  for (const auto dimensions : {0u, 5u}) {
    auto graph = make_noise_graph(
        {.dimensions = dimensions,
         .type = "FBM",
         .normalize = true,
         .color = true,
         .w = 0.0f,
         .scale = 1.0f,
         .detail = 2.0f,
         .roughness = 0.5f,
         .lacunarity = 2.0f,
         .offset = 0.0f,
         .gain = 1.0f,
         .distortion = 0.0f});
    const auto image = compile_graph(graph);
    require(!image.valid, "invalid Noise dimensions were accepted");
  }
  auto graph = make_noise_graph(
      {.dimensions = 3u,
       .type = "NOT_A_CYCLES_NOISE_TYPE",
       .normalize = true,
       .color = true,
       .w = 0.0f,
       .scale = 1.0f,
       .detail = 2.0f,
       .roughness = 0.5f,
       .lacunarity = 2.0f,
       .offset = 0.0f,
       .gain = 1.0f,
       .distortion = 0.0f});
  const auto image = compile_graph(graph);
  require(!image.valid, "invalid Noise type was accepted");
}

} // namespace

int main() {
  test_linked_streams_match_cycles_5_2_1();
  test_payload_cross_product_and_mapping();
  test_invalid_static_properties_are_rejected();
  return EXIT_SUCCESS;
}

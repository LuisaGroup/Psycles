#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <array>
#include <bit>
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
              << actual.size() << ", expected " << expected.size()
              << "; actual:";
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
                << expected[index] << std::dec << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

[[nodiscard]] ShaderImage compile_graph(ShaderGraph &graph,
                                        AttributeIDMap &attribute_ids,
                                        ImageIDMap &image_ids) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    std::exit(EXIT_FAILURE);
  }
  const auto image = compile_shader(
      *shader.program, attribute_ids, image_ids,
      ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic.c_str());
  return image;
}

[[nodiscard]] ShaderGraph make_image_graph(
    Vec3f coordinate, std::uint64_t resource_id,
    std::string_view interpolation, std::string_view extension,
    std::string_view projection, float blend,
    std::string_view color_space = "Non-Color",
    bool unassociate_alpha = false) {
  ShaderGraph graph;
  const auto image = graph.add_node(node_type::image_texture, "Image Texture");
  const auto separate =
      graph.add_node(node_type::separate_color, "Image Channels");
  const auto combine =
      graph.add_node(node_type::combine_color, "RGB Alpha");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.set_input(image, "Vector", SocketValue::vector(coordinate)) &&
          graph.set_property(image, "Image",
                             SocketValue::unsigned_integer(resource_id)) &&
          graph.set_property(image, "Interpolation",
                             SocketValue::string(std::string{interpolation})) &&
          graph.set_property(image, "Extension",
                             SocketValue::string(std::string{extension})) &&
          graph.set_property(image, "Projection",
                             SocketValue::string(std::string{projection})) &&
          graph.set_property(image, "ProjectionBlend",
                             SocketValue::floating(blend)) &&
          graph.set_property(image, "ColorSpace",
                             SocketValue::string(std::string{color_space})) &&
          graph.set_property(image, "UnassociateAlpha",
                             SocketValue::boolean(unassociate_alpha)) &&
          graph.set_property(separate, "Mode", SocketValue::string("RGB")) &&
          graph.connect({image, "Color"}, separate, "Color") &&
          graph.set_property(combine, "Mode", SocketValue::string("RGB")) &&
          graph.connect({separate, "R"}, combine, "R") &&
          graph.connect({separate, "G"}, combine, "G") &&
          graph.connect({image, "Alpha"}, combine, "B") &&
          graph.connect({combine, "Color"}, emission, "Color"),
      "failed to construct Image Texture graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_environment_graph(
    Vec3f direction, std::uint64_t resource_id,
    std::string_view interpolation, std::string_view projection) {
  ShaderGraph graph;
  const auto image =
      graph.add_node(node_type::environment_texture, "Environment Texture");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.set_input(image, "Vector", SocketValue::vector(direction)) &&
          graph.set_property(image, "Image",
                             SocketValue::unsigned_integer(resource_id)) &&
          graph.set_property(image, "Interpolation",
                             SocketValue::string(std::string{interpolation})) &&
          graph.set_property(image, "Projection",
                             SocketValue::string(std::string{projection})) &&
          graph.set_property(image, "ColorSpace",
                             SocketValue::string("sRGB")) &&
          graph.connect({image, "Color"}, emission, "Color"),
      "failed to construct Environment Texture graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

void test_sampling_handles_and_streams_match_cycles_5_2_1() {
  struct InterpolationCase {
    std::string_view name;
    ImageInterpolation value;
  };
  struct ExtensionCase {
    std::string_view name;
    ImageExtension value;
    std::array<std::uint32_t, 3u> coordinate;
  };
  static constexpr std::array interpolations{
      InterpolationCase{"Closest", ImageInterpolation::closest},
      InterpolationCase{"Linear", ImageInterpolation::linear},
      InterpolationCase{"Cubic", ImageInterpolation::cubic},
      InterpolationCase{"Smart", ImageInterpolation::smart}};
  static constexpr std::array extensions{
      ExtensionCase{"REPEAT", ImageExtension::repeat,
                    {0xbe3126e9u, 0x3f9bc6a8u, 0x3ebd70a4u}},
      ExtensionCase{"EXTEND", ImageExtension::extend,
                    {0x3f904189u, 0xbda9fbe7u, 0x3ebd70a4u}},
      ExtensionCase{"CLIP", ImageExtension::clip,
                    {0xbd178d50u, 0x3f1df3b6u, 0x3ebd70a4u}},
      ExtensionCase{"MIRROR", ImageExtension::mirror,
                    {0xbe90e560u, 0x3f98b439u, 0x3ebd70a4u}}};

  AttributeIDMap attribute_ids;
  ImageIDMap image_ids;
  auto logical_id = std::uint32_t{};
  for (const auto &interpolation : interpolations) {
    for (const auto &extension : extensions) {
      auto graph = make_image_graph(
          {f32(extension.coordinate[0]), f32(extension.coordinate[1]),
           f32(extension.coordinate[2])},
          41u, interpolation.name, extension.name, "FLAT", 0.0f);
      const auto image = compile_graph(graph, attribute_ids, image_ids);
      const std::array<std::uint32_t, 35u> expected{
          0x00000001u, 0x00000004u, 0x00000021u, 0x00000022u,
          0x00000013u, 0x00000000u, extension.coordinate[0],
          extension.coordinate[1], extension.coordinate[2], 0x0000001cu,
          logical_id, 0x00000000u, 0x06030000u, 0x00000052u,
          0x00000000u, 0x7fc00003u, 0x00000000u, 0x00000000u,
          0x00ff0100u, 0x00000053u, 0x00000000u, 0x7fc00000u,
          0x7fc00001u, 0x7fc00006u, 0x00000002u, 0x00000007u,
          0x7fc00002u, 0x00000000u, 0x00000000u, 0x3f800000u,
          0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
          0x00000000u};
      require_words(image.words, expected, interpolation.name);
      require(image.peak_stack_usage == 7u &&
                  image.node_types_used[NODE_TEX_IMAGE],
              "Image Texture opcode or stack lifetime differs from Cycles");
      ++logical_id;
    }
  }

  const auto bindings = image_ids.bindings();
  require(bindings.size() == interpolations.size() * extensions.size(),
          "Image handle table did not preserve the sampler cross-product");
  auto index = std::size_t{};
  for (const auto &interpolation : interpolations) {
    for (const auto &extension : extensions) {
      require(bindings[index] ==
                  ImageBinding{.resource_id = 41u,
                               .interpolation = interpolation.value,
                               .extension = extension.value},
              "Image handle binding differs from Cycles ImageParams identity");
      ++index;
    }
  }

  auto duplicate = make_image_graph(
      {f32(extensions[0].coordinate[0]), f32(extensions[0].coordinate[1]),
       f32(extensions[0].coordinate[2])},
      41u, "Closest", "REPEAT", "FLAT", 0.0f);
  const auto duplicate_image =
      compile_graph(duplicate, attribute_ids, image_ids);
  require(duplicate_image.words[10u] == 0u &&
              image_ids.bindings().size() == bindings.size(),
          "equal Cycles ImageParams did not reuse one image handle");
}

void test_projection_payloads_match_cycles_5_2_1() {
  struct Case {
    std::string_view projection;
    std::array<std::uint32_t, 3u> coordinate;
    std::uint32_t projection_or_blend;
    std::uint32_t opcode;
  };
  static constexpr std::array cases{
      Case{"FLAT", {0x3e3126e9u, 0x3f1df3b6u, 0x3f547ae1u},
           0x00000000u, 0x0000001cu},
      Case{"SPHERE", {0x3f000000u, 0x3f000000u, 0x3f000000u},
           0x00000002u, 0x0000001cu},
      Case{"TUBE", {0x3f028f5cu, 0x3efae148u, 0x3f6e147bu},
           0x00000003u, 0x0000001cu},
      Case{"BOX", {0x3ebd70a4u, 0x3f51eb85u, 0x3e0f5c29u},
           0x3f0ccccdu, 0x0000001eu}};

  AttributeIDMap attribute_ids;
  ImageIDMap image_ids;
  for (const auto &item : cases) {
    auto graph = make_image_graph(
        {f32(item.coordinate[0]), f32(item.coordinate[1]),
         f32(item.coordinate[2])},
        52u, "Linear", "REPEAT", item.projection,
        item.projection == "BOX" ? f32(item.projection_or_blend) : 0.0f);
    const auto image = compile_graph(graph, attribute_ids, image_ids);
    const std::array<std::uint32_t, 35u> expected{
        0x00000001u, 0x00000004u, 0x00000021u, 0x00000022u,
        0x00000013u, 0x00000000u, item.coordinate[0], item.coordinate[1],
        item.coordinate[2], item.opcode, 0x00000000u,
        item.projection_or_blend, 0x06030000u, 0x00000052u,
        0x00000000u, 0x7fc00003u, 0x00000000u, 0x00000000u,
        0x00ff0100u, 0x00000053u, 0x00000000u, 0x7fc00000u,
        0x7fc00001u, 0x7fc00006u, 0x00000002u, 0x00000007u,
        0x7fc00002u, 0x00000000u, 0x00000000u, 0x3f800000u,
        0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
        0x00000000u};
    require_words(image.words, expected, item.projection);
    require(image.node_types_used[item.opcode == 0x0000001eu
                                      ? NODE_TEX_IMAGE_BOX
                                      : NODE_TEX_IMAGE],
            "Image projection selected the wrong Cycles opcode");
  }
  require(image_ids.bindings().size() == 1u,
          "projection incorrectly participated in image handle identity");
}

void test_environment_handles_and_payloads_match_cycles_5_2_1() {
  struct InterpolationCase {
    std::string_view name;
    ImageInterpolation value;
  };
  static constexpr std::array interpolations{
      InterpolationCase{"Closest", ImageInterpolation::closest},
      InterpolationCase{"Linear", ImageInterpolation::linear},
      InterpolationCase{"Cubic", ImageInterpolation::cubic},
      InterpolationCase{"Smart", ImageInterpolation::smart}};
  struct ProjectionCase {
    std::string_view name;
    std::array<std::uint32_t, 3u> direction;
    std::uint32_t value;
  };
  static constexpr std::array projections{
      ProjectionCase{"EQUIRECTANGULAR",
                     {0x3ebd70a4u, 0xbf1c28f6u, 0x3f35c28fu}, 0u},
      ProjectionCase{"MIRROR_BALL",
                     {0xbedc28f6u, 0x3e947ae1u, 0x3f547ae1u}, 1u}};

  AttributeIDMap attribute_ids;
  ImageIDMap image_ids;
  for (const auto &projection : projections) {
    auto logical_id = std::uint32_t{};
    for (const auto &interpolation : interpolations) {
      auto graph = make_environment_graph(
          {f32(projection.direction[0]), f32(projection.direction[1]),
           f32(projection.direction[2])},
          73u, interpolation.name, projection.name);
      const auto image = compile_graph(graph, attribute_ids, image_ids);
      const std::array<std::uint32_t, 23u> expected{
          0x00000001u, 0x00000004u, 0x00000015u, 0x00000016u,
          0x00000013u, 0x00000000u, projection.direction[0],
          projection.direction[1], projection.direction[2], 0x0000003du,
          logical_id, projection.value, 0xff030001u, 0x00000007u,
          0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
          0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
          0x00000000u};
      require_words(image.words, expected, projection.name);
      require(image.peak_stack_usage == 6u &&
                  image.node_types_used[NODE_TEX_ENVIRONMENT],
              "Environment Texture opcode or lifetime differs from Cycles");
      ++logical_id;
    }
  }
  const auto bindings = image_ids.bindings();
  require(bindings.size() == interpolations.size(),
          "Environment projection incorrectly changed image handle identity");
  for (auto index = std::size_t{}; index < bindings.size(); ++index) {
    require(bindings[index] ==
                ImageBinding{.resource_id = 73u,
                             .interpolation = interpolations[index].value,
                             .extension = ImageExtension::repeat},
            "Environment handle does not use Cycles repeat extension");
  }
}

void test_srgb_alpha_and_default_uv_match_cycles_5_2_1() {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Default Image Coordinates");
  const auto to_vector =
      graph.add_node(node_type::point_to_vector, "Default UV to Vector");
  const auto image = graph.add_node(node_type::image_texture, "Image Texture");
  const auto separate =
      graph.add_node(node_type::separate_color, "Separate Image Color");
  const auto combine =
      graph.add_node(node_type::combine_color, "Pack Color and Alpha");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.connect({coordinates, "UV"}, to_vector, "Point") &&
          graph.connect({to_vector, "Vector"}, image, "Vector") &&
          graph.set_property(image, "Image",
                             SocketValue::unsigned_integer(91u)) &&
          graph.set_property(image, "Interpolation",
                             SocketValue::string("Linear")) &&
          graph.set_property(image, "Extension",
                             SocketValue::string("EXTEND")) &&
          graph.set_property(image, "Projection",
                             SocketValue::string("FLAT")) &&
          graph.set_property(image, "ColorSpace",
                             SocketValue::string("sRGB")) &&
          graph.set_property(image, "UnassociateAlpha",
                             SocketValue::boolean(true)) &&
          graph.set_property(separate, "Mode", SocketValue::string("RGB")) &&
          graph.connect({image, "Color"}, separate, "Color") &&
          graph.set_property(combine, "Mode", SocketValue::string("RGB")) &&
          graph.connect({separate, "R"}, combine, "R") &&
          graph.connect({separate, "G"}, combine, "G") &&
          graph.connect({image, "Alpha"}, combine, "B") &&
          graph.connect({combine, "Color"}, emission, "Color"),
      "failed to construct sRGB Image Texture graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  AttributeIDMap attribute_ids;
  ImageIDMap image_ids;
  const auto compiled = compile_graph(graph, attribute_ids, image_ids);
  static constexpr std::array<std::uint32_t, 34u> expected{
      0x00000001u, 0x00000004u, 0x00000020u, 0x00000021u,
      0x00000015u, 0x00000005u, 0x00000000u, 0x00000000u,
      0x0000001cu, 0x00000000u, 0x00000000u, 0x06030003u,
      0x00000052u, 0x00000000u, 0x7fc00003u, 0x00000000u,
      0x00000000u, 0x00ff0100u, 0x00000053u, 0x00000000u,
      0x7fc00000u, 0x7fc00001u, 0x7fc00006u, 0x00000002u,
      0x00000007u, 0x7fc00002u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u};
  require_words(compiled.words, expected, "sRGB straight-alpha Image Texture");
  require(compiled.node_types_used[NODE_ATTR] &&
              compiled.node_types_used[NODE_TEX_IMAGE],
          "default Image Texture UV did not select the Cycles attribute path");
}

} // namespace

int main() {
  test_sampling_handles_and_streams_match_cycles_5_2_1();
  test_projection_payloads_match_cycles_5_2_1();
  test_environment_handles_and_payloads_match_cycles_5_2_1();
  test_srgb_alpha_and_default_uv_match_cycles_5_2_1();
  return 0;
}

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_test_compile.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;
using psycles::Mat4f;

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
                << expected[index] << "; actual:";
      for (const auto word : actual) {
        std::cerr << " 0x" << word;
      }
      std::cerr << std::dec << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

[[nodiscard]] constexpr std::uint32_t
ordered_float_bits(std::uint32_t bits) noexcept {
  return (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
}

void require_transform_words(std::span<const std::uint32_t> actual,
                             std::span<const std::uint32_t> expected,
                             std::string_view label) {
  require(actual.size() == expected.size(),
          "Texture Coordinate transform word count differs from Cycles 5.2.1");
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    if (index >= 7u && index < 19u) {
      const auto actual_ordered = ordered_float_bits(actual[index]);
      const auto expected_ordered = ordered_float_bits(expected[index]);
      const auto distance = actual_ordered > expected_ordered
                                ? actual_ordered - expected_ordered
                                : expected_ordered - actual_ordered;
      if (distance <= 1u) {
        continue;
      }
    } else if (actual[index] == expected[index]) {
      continue;
    }
    std::cerr << label << " differs structurally at word " << index
              << ": got 0x" << std::hex << actual[index]
              << ", expected 0x" << expected[index] << std::dec << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] ShaderImage compile_surface(std::string_view socket,
                                          bool from_dupli = false,
                                          bool use_transform = false,
                                          Mat4f object_transform = {}) {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Texture Coordinate");
  const auto to_vector = graph.add_node(
      socket == "Normal" || socket == "Reflection"
          ? node_type::normal_to_vector
          : node_type::point_to_vector,
      "Coordinate to Vector");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_property(coordinates, "FromDupli",
                             SocketValue::boolean(from_dupli)) &&
              graph.set_property(coordinates, "UseTransform",
                                 SocketValue::boolean(use_transform)) &&
              graph.set_property(coordinates, "ObjectTransform",
                                 SocketValue::transform(object_transform)) &&
              graph.connect({coordinates, std::string{socket}}, to_vector,
                            socket == "Normal" || socket == "Reflection"
                                ? "Normal"
                                : "Point") &&
              graph.connect({to_vector, "Vector"}, to_color,
                            "Vector") &&
              graph.connect({to_color, "Color"}, emission, "Color"),
          "failed to construct Texture Coordinate graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "Texture Coordinate graph did not validate");
  return compile_shader(*shader.program);
}

[[nodiscard]] constexpr float f32(std::uint32_t bits) noexcept {
  return std::bit_cast<float>(bits);
}

template<std::size_t N>
void require_surface_mode(std::string_view socket, bool from_dupli,
                          const std::array<std::uint32_t, N> &expected,
                          ShaderNodeType opcode) {
  const auto image = compile_surface(socket, from_dupli);
  require(image.valid, image.diagnostic.c_str());
  require_words(image.words, expected, socket);
  require(image.peak_stack_usage == 3u && image.node_types_used[opcode],
          "Texture Coordinate opcode or stack use differs from Cycles 5.2.1");
}

void test_attribute_backed_modes_match_cycles_5_2_1() {
  static constexpr std::array<std::uint32_t, 18u> generated{
      0x00000001u, 0x00000004u, 0x00000010u, 0x00000011u,
      0x00000015u, 0x0000000bu, 0x00000000u, 0x00000000u,
      0x00000007u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u};
  static constexpr std::array<std::uint32_t, 18u> uv{
      0x00000001u, 0x00000004u, 0x00000010u, 0x00000011u,
      0x00000015u, 0x00000005u, 0x00000000u, 0x00000000u,
      0x00000007u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u};
  require_surface_mode("Generated", false, generated, NODE_ATTR);
  require_surface_mode("UV", false, uv, NODE_ATTR);
}

void test_direct_modes_match_cycles_5_2_1() {
  struct Case {
    std::string_view socket;
    bool from_dupli;
    std::uint32_t mode;
  };
  static constexpr std::array cases{
      Case{"Normal", false, 0u},
      Case{"Object", false, 1u},
      Case{"Camera", false, 3u},
      Case{"Window", false, 4u},
      Case{"Reflection", false, 5u},
      Case{"Generated", true, 6u},
      Case{"UV", true, 7u},
  };
  for (const auto &item : cases) {
    const std::array<std::uint32_t, 17u> expected{
        0x00000001u, 0x00000004u, 0x0000000fu, 0x00000010u,
        0x0000000fu, item.mode,     0x00000000u, 0x00000007u,
        0x7fc00000u, 0x00000000u, 0x00000000u, 0x3f800000u,
        0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
        0x00000000u};
    require_surface_mode(item.socket, item.from_dupli, expected,
                         NODE_TEX_COORD);
  }
}

void test_explicit_object_transform_matches_cycles_5_2_1() {
  Mat4f transform;
  // Column-major storage of the object-to-world Transform observed in the
  // clean Cycles 5.2.1 probe. Cycles inverts this host value before appending
  // the 12-word PackedTransform payload.
  transform.elements = {
      f32(0x3f405467u), f32(0x3e12da3au), f32(0x00000000u),
      f32(0x00000000u), f32(0xbe56a9ddu), f32(0x3f8c91fcu),
      f32(0x00000000u), f32(0x00000000u), f32(0x00000000u),
      f32(0x00000000u), f32(0x3f800000u), f32(0x00000000u),
      f32(0x3eb33333u), f32(0xbe800000u), f32(0x3dcccccdu),
      f32(0x3f800000u)};
  const auto image = compile_surface("Object", false, true, transform);
  require(image.valid, image.diagnostic.c_str());
  static constexpr std::array<std::uint32_t, 29u> expected{
      0x00000001u, 0x00000004u, 0x0000001bu, 0x0000001cu,
      0x0000000fu, 0x00000002u, 0x00000000u, 0x3fa46264u,
      0x3e7b07a1u, 0x80000000u, 0xbec6c265u, 0xbe2bbb16u,
      0x3f60e992u, 0x00000000u, 0x3e8e8253u, 0x00000000u,
      0x80000000u, 0x3f800000u, 0xbdcccccdu, 0x00000007u,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  // The 12 transform words are host-derived arithmetic. Cycles may select an
  // AVX/FMA implementation and differ from the portable scalar formula by one
  // ULP; that is not an SVM structural difference. Every other word remains
  // exact, including opcode, mode, stack offsets and payload extent.
  require_transform_words(image.words, expected, "Object Transform");
  require(image.peak_stack_usage == 3u &&
              image.node_types_used[NODE_TEX_COORD],
          "Object Transform opcode or stack use differs from Cycles 5.2.1");
}

void test_volume_generated_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Texture Coordinate");
  const auto to_vector =
      graph.add_node(node_type::point_to_vector, "Point to Vector");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({coordinates, "Generated"}, to_vector, "Point") &&
              graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
              graph.connect({to_color, "Color"}, emission, "Color"),
          "failed to construct volume Texture Coordinate graph");
  graph.set_root(ShaderDomain::volume,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "volume Texture Coordinate graph did not validate");
  AttributeIDMap attribute_ids;
  const auto image = compile_shader(
      *shader.program, attribute_ids,
      ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic.c_str());
  static constexpr std::array<std::uint32_t, 17u> expected{
      0x00000001u, 0x00000004u, 0x00000005u, 0x00000010u,
      0x00000000u, 0x0000000fu, 0x00000008u, 0x00000000u,
      0x00000007u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u};
  require_words(image.words, expected, "Volume Generated");
  require(image.peak_stack_usage == 3u &&
              image.node_types_used[NODE_TEX_COORD],
          "Volume Generated opcode or stack use differs from Cycles 5.2.1");
}

void test_background_modes_match_cycles_5_2_1() {
  struct Case {
    std::string_view socket;
    std::uint32_t geometry_type;
    std::string_view conversion;
    std::string_view conversion_input;
  };
  static constexpr std::array cases{
      Case{"Generated", 0u, node_type::point_to_vector, "Point"},
      Case{"Reflection", 3u, node_type::normal_to_vector, "Normal"},
  };
  for (const auto &item : cases) {
    ShaderGraph graph;
    const auto coordinates =
        graph.add_node(node_type::texture_coordinate, "Texture Coordinate");
    const auto to_vector = graph.add_node(
        std::string{item.conversion}, "Coordinate to Vector");
    const auto to_color =
        graph.add_node(node_type::vector_to_color, "Vector to Color");
    const auto background =
        graph.add_node(node_type::background, "Background");
    require(graph.connect({coordinates, std::string{item.socket}}, to_vector,
                          std::string{item.conversion_input}) &&
                graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
                graph.connect({to_color, "Color"}, background, "Color"),
            "failed to construct background Texture Coordinate graph");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = background, .socket = "Closure"});

    const ShaderCompiler frontend{make_core_node_registry()};
    const auto shader = frontend.compile(graph);
    require(shader.ok(),
            "background Texture Coordinate graph did not validate");
    AttributeIDMap attribute_ids;
    const auto image = compile_shader(
        *shader.program, attribute_ids,
        ShaderCompileContext{.background = true});
    require(image.valid, image.diagnostic.c_str());
    const std::array<std::uint32_t, 17u> expected{
        0x00000001u, 0x00000004u, 0x0000000fu, 0x00000010u,
        0x0000000bu, item.geometry_type, 0x00000000u, 0x00000007u,
        0x7fc00000u, 0x00000000u, 0x00000000u, 0x3f800000u,
        0x00000004u, 0x000000ffu, 0x00000000u, 0x00000000u,
        0x00000000u};
    require_words(image.words, expected, item.socket);
    require(image.peak_stack_usage == 3u &&
                image.node_types_used[NODE_GEOMETRY] &&
                !image.node_types_used[NODE_TEX_COORD],
            "background coordinate lowering differs from Cycles 5.2.1");
  }
}

[[nodiscard]] ShaderImage compile_bump(std::string_view socket,
                                       bool from_dupli) {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Texture Coordinate");
  const auto to_vector = graph.add_node(
      socket == "Normal" || socket == "Reflection"
          ? node_type::normal_to_vector
          : node_type::point_to_vector,
      "Coordinate to Vector");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto separate = graph.add_node(node_type::separate_xyz, "Separate XYZ");
  const auto bump = graph.add_node(node_type::bump, "Bump");
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Diffuse");
  require(graph.set_property(coordinates, "FromDupli",
                             SocketValue::boolean(from_dupli)) &&
              // Match the Blender 5.2 node socket default used by the
              // external Cycles oracle. Cycles' standalone graph node default
              // is 0.1, while Blender authors a 0.001 distance here.
              graph.set_input(bump, "Distance",
                              SocketValue::floating(0.001f)) &&
              graph.connect({coordinates, std::string{socket}}, to_vector,
                            socket == "Normal" || socket == "Reflection"
                                ? "Normal"
                                : "Point") &&
              graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
              graph.connect({to_color, "Color"}, separate, "Vector") &&
              graph.connect({separate, "X"}, bump, "Height") &&
              graph.connect({bump, "Normal"}, diffuse, "Normal"),
          "failed to construct bump Texture Coordinate graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "bump Texture Coordinate graph did not validate");
  return compile_shader(*shader.program);
}

void test_bump_derivative_modes_match_cycles_5_2_1() {
  static constexpr std::array<std::uint32_t, 82u> normal_oracle{
      0x00000001u, 0x00000004u, 0x00000050u, 0x00000051u,
      0x00000010u, 0x00000000u, 0x3dcccccdu, 0x00000054u,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x00000300u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x0000ff01u, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x0000ff02u, 0x00000010u, 0x00000100u,
      0x3dcccccdu, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x00000400u, 0x00000054u, 0x7fc00000u,
      0x00000000u, 0x00000000u, 0x0000ff01u, 0x00000054u,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x0000ff02u,
      0x00000010u, 0x00000200u, 0x3dcccccdu, 0x00000054u,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x00000500u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x0000ff01u, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x0000ff02u, 0x0000000bu, 0x00000001u,
      0x00000000u, 0x00000021u, 0x3a83126fu, 0x3f800000u,
      0x3dcccccdu, 0x03000000u, 0xff060504u, 0x00000005u,
      0x3f4ccccdu, 0x3f4ccccdu, 0x3f4ccccdu, 0x00000002u,
      0x00000002u, 0x000000ffu, 0x3f4ccccdu, 0x3f4ccccdu,
      0x3f4ccccdu, 0x00000000u, 0x00000006u, 0x00000000u,
      0x00000000u, 0x00000000u};
  struct Case {
    std::string_view socket;
    bool from_dupli;
    std::uint32_t center;
    std::uint32_t dx;
    std::uint32_t dy;
  };
  static constexpr std::array cases{
      Case{"Normal", false, 0x00000000u, 0x00000100u, 0x00000200u},
      Case{"Reflection", false, 0x00000005u, 0x00000005u, 0x00000005u},
      Case{"Generated", true, 0x00000006u, 0x00000006u, 0x00000006u},
      Case{"UV", true, 0x00000007u, 0x00000007u, 0x00000007u},
  };
  for (const auto &item : cases) {
    auto expected = normal_oracle;
    expected[5u] = item.center;
    expected[23u] = item.dx;
    expected[41u] = item.dy;
    const auto image = compile_bump(item.socket, item.from_dupli);
    require(image.valid, image.diagnostic.c_str());
    require_words(image.words, expected, item.socket);
    if (image.peak_stack_usage != 9u ||
        !image.node_types_used[NODE_TEX_COORD_DERIVATIVE] ||
        image.node_types_used[NODE_TEX_COORD]) {
      std::cerr << "Texture Coordinate bump derivative metadata differs from "
                   "Cycles 5.2.1: peak="
                << image.peak_stack_usage
                << ", derivative="
                << image.node_types_used[NODE_TEX_COORD_DERIVATIVE]
                << ", plain=" << image.node_types_used[NODE_TEX_COORD]
                << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

} // namespace

int main() {
  test_attribute_backed_modes_match_cycles_5_2_1();
  test_direct_modes_match_cycles_5_2_1();
  test_explicit_object_transform_matches_cycles_5_2_1();
  test_volume_generated_matches_cycles_5_2_1();
  test_background_modes_match_cycles_5_2_1();
  test_bump_derivative_modes_match_cycles_5_2_1();
  return EXIT_SUCCESS;
}

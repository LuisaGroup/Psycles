#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_test_compile.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>

namespace {

using namespace psycles;
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
                   const char *message) {
  if (actual.size() != expected.size()) {
    std::cerr << message << ": got " << actual.size() << " words, expected "
              << expected.size() << '\n';
    for (auto index = std::size_t{}; index < actual.size(); ++index) {
      std::cerr << "  [" << std::dec << index << "] = 0x" << std::hex
                << actual[index] << '\n';
    }
    std::exit(EXIT_FAILURE);
  }
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << message << " at word " << index << ": got 0x" << std::hex
                << actual[index] << ", expected 0x" << expected[index] << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

[[nodiscard]] ShaderImage compile(ShaderGraph &graph) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "raw Hair graph did not validate");
  auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());
  return image;
}

void test_cycles_internal_defaults() {
  ShaderGraph graph;
  const auto hair = graph.add_node(node_type::hair_bsdf, "Cycles Default Hair");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = hair, .socket = "Closure"});

  const auto image = compile(graph);
  /* Cycles 5.2.1 HairBsdfNode declares 0.2 for both roughness axes. Blender's
   * UI node deliberately has different 0.1/1.0 socket defaults; that adapter
   * boundary is frozen independently below. */
  static constexpr std::array<std::uint32_t, 18u> expected{
      0x00000001u, 0x00000004u, 0x00000010u, 0x00000011u, 0x00000005u,
      0x3f4ccccdu, 0x3f4ccccdu, 0x3f4ccccdu, 0x00000002u, 0x00000013u,
      0x000000ffu, 0x3e4ccccdu, 0x3e4ccccdu, 0x00000000u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  require_words(image.words, expected,
                "Cycles-internal Hair defaults differ from Cycles 5.2.1");
  require(image.peak_stack_usage == 0u &&
              image.node_types_used[NODE_CLOSURE_SET_WEIGHT] &&
              image.node_types_used[NODE_CLOSURE_BSDF],
          "Cycles-internal Hair topology differs from Cycles 5.2.1");
}

void test_blender_default_reflection_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto hair = graph.add_node(node_type::hair_bsdf,
                                   "Standalone Hair Reflection SVM Oracle");
  require(graph.set_input(hair, "RoughnessU", SocketValue::floating(0.1f)) &&
              graph.set_input(hair, "RoughnessV", SocketValue::floating(1.0f)),
          "failed to project Blender Hair socket defaults");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = hair, .socket = "Closure"});

  const auto image = compile(graph);
  /* Exact compact image from the unmodified Cycles 5.2.1 Blender oracle. */
  static constexpr std::array<std::uint32_t, 18u> expected{
      0x00000001u, 0x00000004u, 0x00000010u, 0x00000011u, 0x00000005u,
      0x3f4ccccdu, 0x3f4ccccdu, 0x3f4ccccdu, 0x00000002u, 0x00000013u,
      0x000000ffu, 0x3dcccccdu, 0x3f800000u, 0x00000000u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  require_words(image.words, expected,
                "Blender-default Hair Reflection differs from Cycles 5.2.1");
}

void test_authored_transmission_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto tangent =
      graph.add_node(node_type::combine_xyz, "Authored Hair Tangent");
  require(graph.set_input(tangent, "X", SocketValue::floating(0.3f)) &&
              graph.set_input(tangent, "Y", SocketValue::floating(0.4f)) &&
              graph.set_input(tangent, "Z", SocketValue::floating(0.0f)),
          "failed to set authored Hair Tangent");
  const auto hair = graph.add_node(node_type::hair_bsdf,
                                   "Standalone Hair Transmission SVM Oracle");
  require(
      graph.set_property(hair, "Component",
                         SocketValue::string("TRANSMISSION")) &&
          graph.set_input(hair, "Color",
                          SocketValue::color({0.83f, 0.17f, 0.52f})) &&
          graph.set_input(hair, "Offset", SocketValue::floating(0.27f)) &&
          graph.set_input(hair, "RoughnessU", SocketValue::floating(0.0002f)) &&
          graph.set_input(hair, "RoughnessV", SocketValue::floating(1.4f)) &&
          graph.connect(OutputRef{.node = tangent, .socket = "Vector"}, hair,
                        "Tangent"),
      "failed to configure authored Hair Transmission");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = hair, .socket = "Closure"});

  const auto image = compile(graph);
  static constexpr std::array<std::uint32_t, 23u> expected{
      0x00000001u, 0x00000004u, 0x00000015u, 0x00000016u, 0x00000013u,
      0x00000000u, 0x3e99999au, 0x3ecccccdu, 0x00000000u, 0x00000005u,
      0x3f547ae1u, 0x3e2e147bu, 0x3f051eb8u, 0x00000002u, 0x00000017u,
      0x000000ffu, 0x3951b717u, 0x3fb33333u, 0x3e8a3d71u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u};
  require_words(image.words, expected,
                "authored Hair Transmission differs from Cycles 5.2.1");
  require(image.peak_stack_usage == 3u && image.node_types_used[NODE_VALUE_V] &&
              image.node_types_used[NODE_CLOSURE_BSDF],
          "authored Hair topology differs from Cycles 5.2.1");
}

} // namespace

int main() {
  test_cycles_internal_defaults();
  test_blender_default_reflection_matches_cycles_5_2_1();
  test_authored_transmission_matches_cycles_5_2_1();
  return EXIT_SUCCESS;
}

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
  require(actual.size() == expected.size(), message);
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
  require(shader.ok(), "raw Ray Portal graph did not validate");
  auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());
  return image;
}

void test_default_ray_portal_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto portal = graph.add_node(
      node_type::ray_portal_bsdf, "Standalone Default Ray Portal SVM Oracle");
  require(graph.set_input(portal, "Color",
                          SocketValue::color({0.26f, 0.71f, 0.43f})),
          "failed to set default Ray Portal Color");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = portal, .socket = "Closure"});

  const auto image = compile(graph);
  static constexpr std::array<std::uint32_t, 21u> expected{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x0000000bu,
      0x00000000u, 0x00000000u, 0x00000005u, 0x3e851eb8u, 0x3f35c28fu,
      0x3edc28f6u, 0x00000002u, 0x0000001du, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u};
  require_words(image.words, expected,
                "default Ray Portal differs from Cycles 5.2.1");
  require(image.peak_stack_usage == 3u &&
              image.node_types_used[NODE_GEOMETRY] &&
              image.node_types_used[NODE_CLOSURE_SET_WEIGHT] &&
              image.node_types_used[NODE_CLOSURE_BSDF],
          "default Ray Portal topology differs from Cycles 5.2.1");
}

void test_authored_ray_portal_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto position =
      graph.add_node(node_type::combine_xyz, "Authored Ray Portal Position");
  require(graph.set_input(position, "X", SocketValue::floating(1.25f)) &&
              graph.set_input(position, "Y", SocketValue::floating(-0.75f)) &&
              graph.set_input(position, "Z", SocketValue::floating(2.5f)),
          "failed to set authored Ray Portal Position");
  const auto portal = graph.add_node(
      node_type::ray_portal_bsdf, "Standalone Authored Ray Portal SVM Oracle");
  require(graph.set_input(portal, "Color",
                          SocketValue::color({0.83f, 0.17f, 0.52f})) &&
              graph.set_input(portal, "Direction",
                              SocketValue::vector({0.3f, -0.4f, 1.2f})) &&
              graph.connect(OutputRef{.node = position, .socket = "Vector"},
                            portal, "Position"),
          "failed to configure authored Ray Portal");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = portal, .socket = "Closure"});

  const auto image = compile(graph);
  static constexpr std::array<std::uint32_t, 23u> expected{
      0x00000001u, 0x00000004u, 0x00000015u, 0x00000016u, 0x00000013u,
      0x00000000u, 0x3fa00000u, 0xbf400000u, 0x40200000u, 0x00000005u,
      0x3f547ae1u, 0x3e2e147bu, 0x3f051eb8u, 0x00000002u, 0x0000001du,
      0x000000ffu, 0x3e99999au, 0xbecccccdu, 0x3f99999au, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000000u};
  require_words(image.words, expected,
                "authored Ray Portal differs from Cycles 5.2.1");
  require(image.peak_stack_usage == 3u && image.node_types_used[NODE_VALUE_V] &&
              image.node_types_used[NODE_CLOSURE_BSDF],
          "authored Ray Portal topology differs from Cycles 5.2.1");
}

} // namespace

int main() {
  test_default_ray_portal_matches_cycles_5_2_1();
  test_authored_ray_portal_matches_cycles_5_2_1();
  return EXIT_SUCCESS;
}

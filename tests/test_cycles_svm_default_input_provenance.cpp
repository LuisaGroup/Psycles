#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_test_compile.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>

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
                   std::span<const std::uint32_t> expected) {
  require(actual.size() == expected.size(),
          "linked Metallic SVM word count differs from Cycles 5.2.1");
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << "linked Metallic SVM differs from Cycles 5.2.1 at word "
                << index << ": got 0x" << std::hex << actual[index]
                << ", expected 0x" << expected[index] << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

void set_vector(ShaderGraph &graph, NodeId node, float x, float y, float z) {
  require(graph.set_input(node, "X", SocketValue::floating(x)) &&
              graph.set_input(node, "Y", SocketValue::floating(y)) &&
              graph.set_input(node, "Z", SocketValue::floating(z)),
          "failed to configure linked constant vector");
}

void test_folded_authored_links_precede_default_inputs() {
  ShaderGraph graph;
  const auto normal = graph.add_node(node_type::combine_xyz, "Linked Normal");
  const auto normal_conversion =
      graph.add_node(node_type::vector_to_normal, "Vector to Normal");
  const auto tangent =
      graph.add_node(node_type::combine_xyz, "Linked Tangent");
  const auto metallic =
      graph.add_node(node_type::metallic_bsdf, "Metallic BSDF Matrix 06");

  set_vector(graph, normal, 0.0f, 0.0f, 1.0f);
  set_vector(graph, tangent, 0.6f, 0.8f, 0.0f);
  require(
      graph.connect({normal, "Vector"}, normal_conversion, "Vector") &&
          graph.connect({normal_conversion, "Normal"}, metallic, "Normal") &&
          graph.connect({tangent, "Vector"}, metallic, "Tangent") &&
          graph.set_property(metallic, "FresnelType",
                             SocketValue::string("F82")) &&
          graph.set_property(metallic, "Distribution",
                             SocketValue::string("MULTI_GGX")) &&
          graph.set_input(metallic, "BaseColor",
                          SocketValue::color({0.18f, 0.54f, 0.88f})) &&
          graph.set_input(metallic, "EdgeTint",
                          SocketValue::color({0.64f, 0.90f, 0.99f})) &&
          graph.set_input(metallic, "Roughness",
                          SocketValue::floating(0.63f)) &&
          graph.set_input(metallic, "Anisotropy",
                          SocketValue::floating(0.90f)) &&
          graph.set_input(metallic, "Rotation",
                          SocketValue::floating(0.125f)) &&
          graph.set_input(metallic, "ThinFilmThickness",
                          SocketValue::floating(0.0f)) &&
          graph.set_input(metallic, "ThinFilmIOR",
                          SocketValue::floating(1.33f)),
      "failed to construct linked Metallic compiler oracle");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = metallic, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "linked Metallic graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Complete shader-local image from the unmodified Cycles 5.2.1 linked
  // `Metallic BSDF Matrix 06` oracle. Blender's inliner materializes the two
  // hidden-value Normal/Tangent primitives as links. Cycles then folds them
  // only after default-input insertion, so neither authored value may be
  // replaced by Geometry.Normal/Tangent.
  static constexpr std::array expected{
      0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
      0x00000013u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000013u, 0x00000003u, 0x3f19999au,
      0x3f4ccccdu, 0x00000000u, 0x00000002u, 0x0000000bu,
      0x000000ffu, 0x0000000eu, 0x3e3851ecu, 0x3f0a3d71u,
      0x3f6147aeu, 0x3f23d70au, 0x3f666666u, 0x3f7d70a4u,
      0x3f2147aeu, 0x3f666666u, 0x3e000000u, 0x00000000u,
      0x3faa3d71u, 0x00000300u, 0x00000000u, 0x00000000u,
      0x00000000u};
  require_words(image.words, expected);
  require(image.peak_stack_usage == 6u &&
              image.node_types_used[NODE_VALUE_V] &&
              !image.node_types_used[NODE_GEOMETRY],
          "authored Normal/Tangent did not retain Cycles stack provenance");
}

} // namespace

int main() {
  test_folded_authored_links_precede_default_inputs();
  return EXIT_SUCCESS;
}

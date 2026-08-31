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
    std::exit(1);
  }
}

void require_words(std::span<const std::uint32_t> actual,
                   std::span<const std::uint32_t> expected,
                   const char *label) {
  if (actual.size() != expected.size()) {
    std::cerr << label << " word count differs: got " << actual.size()
              << ", expected " << expected.size() << '\n';
    std::exit(1);
  }
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << label << " differs at word " << index << ": got 0x"
                << std::hex << actual[index] << ", expected 0x"
                << expected[index] << std::dec << '\n';
      std::exit(1);
    }
  }
}

[[nodiscard]] ShaderImage compile(ShaderGraph &graph) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    std::exit(1);
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    std::cerr << image.diagnostic << '\n';
    std::exit(1);
  }
  return image;
}

void test_dynamic_separate_combine_xyz() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto normal_to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto vector_to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto separate = graph.add_node(node_type::separate_xyz, "Separate XYZ");
  const auto combine = graph.add_node(node_type::combine_xyz, "Combine XYZ");
  const auto result_to_color =
      graph.add_node(node_type::vector_to_color, "Result to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");

  require(
      graph.connect({geometry, "Normal"}, normal_to_vector, "Normal") &&
          graph.connect({normal_to_vector, "Vector"}, vector_to_color,
                        "Vector") &&
          graph.connect({vector_to_color, "Color"}, separate, "Vector") &&
          graph.connect({separate, "Z"}, combine, "X") &&
          graph.connect({separate, "X"}, combine, "Y") &&
          graph.connect({separate, "Y"}, combine, "Z") &&
          graph.connect({combine, "Vector"}, result_to_color, "Vector") &&
          graph.connect({result_to_color, "Color"}, emission, "Color"),
      "failed to construct dynamic Separate/Combine XYZ graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const auto image = compile(graph);
  static constexpr std::array cycles_5_2_1_oracle{
      0x00000001u, 0x00000004u, 0x00000027u, 0x00000028u,
      0x0000000bu, 0x00000001u, 0x00000000u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x00000300u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x00000401u,
      0x00000054u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x00000502u,
      0x00000056u, 0x7fc00005u, 0x00000000u,
      0x00000056u, 0x7fc00003u, 0x00000001u,
      0x00000056u, 0x7fc00004u, 0x00000002u,
      0x00000007u, 0x7fc00000u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u,
  };
  require_words(image.words, cycles_5_2_1_oracle,
                "dynamic Separate/Combine XYZ");
  require(image.peak_stack_usage == 6u &&
              image.node_types_used[NODE_SEPARATE_VECTOR] &&
              image.node_types_used[NODE_COMBINE_VECTOR] &&
              !image.node_types_used[NODE_SEPARATE_COLOR] &&
              !image.node_types_used[NODE_COMBINE_COLOR],
          "dynamic vector split/pack opcode or lifetime differs from Cycles");
}

void test_constant_separate_combine_xyz() {
  ShaderGraph graph;
  const auto combine = graph.add_node(node_type::combine_xyz, "Combine XYZ");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto separate = graph.add_node(node_type::separate_xyz, "Separate XYZ");
  const auto packed = graph.add_node(node_type::combine_color, "Pack Result");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.set_input(combine, "X", SocketValue::floating(-0.7f)) &&
          graph.set_input(combine, "Y", SocketValue::floating(0.25f)) &&
          graph.set_input(combine, "Z", SocketValue::floating(1.3f)) &&
          graph.connect({combine, "Vector"}, to_color, "Vector") &&
          graph.connect({to_color, "Color"}, separate, "Vector") &&
          graph.set_property(packed, "Mode", SocketValue::string("RGB")) &&
          graph.connect({separate, "Z"}, packed, "R") &&
          graph.connect({separate, "X"}, packed, "G") &&
          graph.connect({separate, "Y"}, packed, "B") &&
          graph.connect({packed, "Color"}, emission, "Color"),
      "failed to construct constant Separate/Combine XYZ graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const auto image = compile(graph);
  static constexpr std::array cycles_5_2_1_oracle{
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
      0x00000005u, 0x3fa66666u, 0xbf333333u, 0x3e800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u,
  };
  require_words(image.words, cycles_5_2_1_oracle,
                "constant Separate/Combine XYZ");
  require(!image.node_types_used[NODE_SEPARATE_VECTOR] &&
              !image.node_types_used[NODE_COMBINE_VECTOR],
          "constant vector split/pack retained an opcode");
}

} // namespace

int main() {
  test_dynamic_separate_combine_xyz();
  test_constant_separate_combine_xyz();
  return EXIT_SUCCESS;
}

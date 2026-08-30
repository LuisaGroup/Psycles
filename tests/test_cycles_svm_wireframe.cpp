#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_graph.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
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
                << expected[index] << std::dec << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

[[nodiscard]] ShaderImage compile_wireframe(bool use_pixel_size,
                                            bool linked_size) {
  ShaderGraph graph;
  const auto wireframe = graph.add_node(node_type::wireframe, "Wireframe");
  require(
      graph.set_input(wireframe, "Size",
                      SocketValue::floating(use_pixel_size ? 2.5f : 0.09f)) &&
          graph.set_property(wireframe, "Use Pixel Size",
                             SocketValue::boolean(use_pixel_size)),
      "failed to configure Wireframe");

  if (linked_size) {
    const auto geometry = graph.add_node(node_type::geometry, "Geometry");
    const auto to_vector =
        graph.add_node(node_type::normal_to_vector, "Normal to Vector");
    const auto to_color =
        graph.add_node(node_type::vector_to_color, "Vector to Color");
    const auto separate =
        graph.add_node(node_type::separate_xyz, "Separate Normal");
    const auto scale = graph.add_node(node_type::math, "Scale Size");
    require(graph.connect({geometry, "Normal"}, to_vector, "Normal") &&
                graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
                graph.connect({to_color, "Color"}, separate, "Vector") &&
                graph.connect({separate, "Z"}, scale, "A") &&
                graph.set_input(
                    scale, "B",
                    SocketValue::floating(use_pixel_size ? 2.5f : 0.09f)) &&
                graph.set_property(scale, "Operation",
                                   SocketValue::string("MULTIPLY")) &&
                graph.connect({scale, "Value"}, wireframe, "Size"),
            "failed to connect linked Wireframe size");
  }

  const auto to_color =
      graph.add_node(node_type::scalar_to_color, "Float to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({wireframe, "Fac"}, to_color, "Value") &&
              graph.connect({to_color, "Color"}, emission, "Color"),
          "failed to connect Wireframe emission");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    std::exit(EXIT_FAILURE);
  }
  auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());
  return image;
}

[[nodiscard]] std::shared_ptr<const ShaderProgram>
make_wireframe_bump_program() {
  ShaderGraph graph;
  const auto wireframe = graph.add_node(node_type::wireframe, "Wireframe");
  const auto bump = graph.add_node(node_type::bump, "Wireframe Bump");
  const auto to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.set_input(wireframe, "Size", SocketValue::floating(0.13f)) &&
          graph.set_property(wireframe, "Use Pixel Size",
                             SocketValue::boolean(false)) &&
          graph.set_input(bump, "Strength", SocketValue::floating(0.8f)) &&
          graph.set_input(bump, "Distance", SocketValue::floating(0.2f)) &&
          graph.set_input(bump, "FilterWidth", SocketValue::floating(0.37f)) &&
          graph.set_property(bump, "Invert", SocketValue::boolean(true)) &&
          graph.set_property(bump, "UseObjectSpace",
                             SocketValue::boolean(false)) &&
          graph.connect({wireframe, "Fac"}, bump, "Height") &&
          graph.connect({bump, "Normal"}, to_vector, "Normal") &&
          graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
          graph.connect({to_color, "Color"}, emission, "Color"),
      "failed to construct Wireframe Bump graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    std::exit(EXIT_FAILURE);
  }
  return shader.program;
}

[[nodiscard]] ShaderImage compile_wireframe_bump() {
  const auto program = make_wireframe_bump_program();
  auto image = compile_shader(*program);
  require(image.valid, image.diagnostic.c_str());
  return image;
}

[[nodiscard]] ShaderImage compile_unlinked_bump() {
  ShaderGraph graph;
  const auto bump = graph.add_node(node_type::bump, "Unlinked Height Bump");
  const auto to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.set_input(bump, "Strength", SocketValue::floating(0.37f)) &&
          graph.set_input(bump, "Distance", SocketValue::floating(0.19f)) &&
          graph.set_input(bump, "FilterWidth", SocketValue::floating(0.23f)) &&
          graph.set_property(bump, "Invert", SocketValue::boolean(true)) &&
          graph.connect({bump, "Normal"}, to_vector, "Normal") &&
          graph.connect({to_vector, "Vector"}, to_color, "Vector") &&
          graph.connect({to_color, "Color"}, emission, "Color"),
      "failed to construct unlinked-Height Bump graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    std::exit(EXIT_FAILURE);
  }
  auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());
  return image;
}

void test_immediate_wireframe_payloads() {
  static constexpr std::array world{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x0000005au,
      0x3db851ecu, 0x00000000u, 0x00000000u, 0x0000000du, 0x00000000u,
      0x00000100u, 0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u,
  };
  static constexpr std::array pixel{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x0000005au,
      0x40200000u, 0x00000000u, 0x00000001u, 0x0000000du, 0x00000000u,
      0x00000100u, 0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u,
  };
  const auto world_image = compile_wireframe(false, false);
  const auto pixel_image = compile_wireframe(true, false);
  require(world_image.valid && pixel_image.valid,
          "immediate Wireframe graph did not compile");
  require_words(world_image.words, world,
                "Cycles 5.2.1 world Wireframe oracle");
  require_words(pixel_image.words, pixel,
                "Cycles 5.2.1 pixel Wireframe oracle");
  require(world_image.peak_stack_usage == 4u &&
              pixel_image.peak_stack_usage == 4u,
          "immediate Wireframe stack lifetime differs from Cycles");
}

void test_linked_wireframe_payloads() {
  static constexpr std::array world{
      0x00000001u, 0x00000004u, 0x0000002bu, 0x0000002cu, 0x0000000bu,
      0x00000001u, 0x00000000u, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x0000ff00u, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x0000ff01u, 0x00000054u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x00000302u, 0x0000002cu, 0x00000002u, 0x7fc00003u,
      0x3db851ecu, 0x00000000u, 0x00000000u, 0x0000005au, 0x7fc00000u,
      0x00000000u, 0x00010000u, 0x0000000du, 0x00000000u, 0x00000201u,
      0x00000007u, 0x7fc00002u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u,
  };
  auto pixel = world;
  pixel[25] = 0x40200000u;
  pixel[31] = 0x00010001u;

  const auto world_image = compile_wireframe(false, true);
  const auto pixel_image = compile_wireframe(true, true);
  require(world_image.valid && pixel_image.valid,
          "linked Wireframe graph did not compile");
  require_words(world_image.words, world,
                "Cycles 5.2.1 linked world Wireframe oracle");
  require_words(pixel_image.words, pixel,
                "Cycles 5.2.1 linked pixel Wireframe oracle");
}

void test_bump_refinement_stream() {
  static constexpr std::array expected{
      0x00000001u, 0x00000004u, 0x00000021u, 0x00000022u, 0x0000005au,
      0x3e051eb8u, 0x3ebd70a4u, 0x00000000u, 0x0000000bu, 0x01000001u,
      0x00000000u, 0x0000005au, 0x3e051eb8u, 0x3ebd70a4u, 0x00040100u,
      0x0000005au, 0x3e051eb8u, 0x3ebd70a4u, 0x00050200u, 0x00000021u,
      0x3e4ccccdu, 0x3f4ccccdu, 0x3ebd70a4u, 0x00000101u, 0xff060504u,
      0x00000007u, 0x7fc00006u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u,
  };
  const auto image = compile_wireframe_bump();
  require(image.valid, "Wireframe Bump graph did not compile");
  require_words(image.words, expected,
                "Cycles 5.2.1 Wireframe Bump refinement oracle");
  if (image.peak_stack_usage != 9u || !image.node_types_used[NODE_WIREFRAME] ||
      !image.node_types_used[NODE_SET_BUMP]) {
    std::cerr << "Wireframe Bump stack/opcode set differs from Cycles: stack="
              << image.peak_stack_usage
              << ", wireframe=" << image.node_types_used[NODE_WIREFRAME]
              << ", set_bump=" << image.node_types_used[NODE_SET_BUMP] << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void test_bump_refinement_features() {
  const auto program = make_wireframe_bump_program();
  const auto projected = CyclesGraph::project(*program);
  require(projected.valid(), projected.diagnostic().c_str());

  std::array<std::uint32_t, 4u> states{};
  std::size_t wireframe_count{};
  for (const auto &node : projected.nodes()) {
    if (node->type != node_type::wireframe) {
      continue;
    }
    ++wireframe_count;
    require(node->bump == SHADER_BUMP_CENTER || node->bump == SHADER_BUMP_DX ||
                node->bump == SHADER_BUMP_DY,
            "refined Wireframe has an invalid Cycles bump state");
    ++states[static_cast<std::size_t>(node->bump)];
    require(node->bump_filter_width == 0.37f,
            "refined Wireframe filter width differs from Cycles");
    require(node->get_feature() == kernel_feature_node_bump,
            "refined Wireframe did not inherit Cycles' Bump feature");
  }
  require(wireframe_count == 3u && states[SHADER_BUMP_CENTER] == 1u &&
              states[SHADER_BUMP_DX] == 1u && states[SHADER_BUMP_DY] == 1u,
          "Cycles Bump refinement did not create one CENTER/DX/DY subgraph");
}

void test_unlinked_height_bump_constant_fold() {
  // Exact program body at global words 89..101 in the Cycles 5.2.1
  // svm_bump_constant_fold external oracle, rebased behind the shader-local
  // four-word jump table. The Bump opcode and its authored values are absent
  // because BumpNode::constant_fold bypasses Geometry.Normal.
  static constexpr std::array expected{
      0x00000001u, 0x00000004u, 0x0000000fu, 0x00000010u, 0x0000000bu,
      0x00000001u, 0x00000000u, 0x00000007u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u,
  };
  const auto image = compile_unlinked_bump();
  require_words(image.words, expected,
                "Cycles 5.2.1 unlinked-Height Bump fold oracle");
  require(image.peak_stack_usage == 3u && !image.node_types_used[NODE_SET_BUMP],
          "unlinked-Height Bump was not folded exactly like Cycles");
}

} // namespace

int main() {
  test_immediate_wireframe_payloads();
  test_linked_wireframe_payloads();
  test_bump_refinement_stream();
  test_bump_refinement_features();
  test_unlinked_height_bump_constant_fold();
  return EXIT_SUCCESS;
}

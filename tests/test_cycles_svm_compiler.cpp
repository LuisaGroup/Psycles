#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <variant>

namespace {

using namespace psycles;
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
                   const char *message) {
  require(actual.size() == expected.size(), message);
  for (auto index = std::size_t{}; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << message << " at word " << index << ": got 0x" << std::hex
                << actual[index] << ", expected 0x" << expected[index] << '\n';
      std::exit(1);
    }
  }
}

void test_math_third_input_default_matches_cycles_5_2_1() {
  const auto registry = make_core_node_registry();
  const auto *schema = registry.find(node_type::math);
  require(schema != nullptr, "Cycles Math schema is absent");
  const auto input = std::find_if(
      schema->inputs.begin(), schema->inputs.end(),
      [](const auto &socket) { return socket.name == "C"; });
  require(input != schema->inputs.end() && input->default_value,
          "Cycles Math third input default is absent");
  const auto *value = std::get_if<float>(&input->default_value->value);
  require(value != nullptr && std::bit_cast<std::uint32_t>(*value) == 0u,
          "Cycles Math Value3 default must be positive zero");
}

void test_diffuse_surface_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Diffuse Probe");
  require(graph.set_input(diffuse, "Color",
                          SocketValue::color({0.68f, 0.24f, 0.09f})),
          "failed to set Diffuse Color");
  require(graph.set_input(diffuse, "Roughness", SocketValue::floating(0.43f)),
          "failed to set Diffuse Roughness");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "raw Diffuse graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from `Diffuse Probe` in a Cycles 5.2.1 SVM dump. Cycles' global
  // offsets (95,111,112) are normalized back to its per-shader local stream
  // (4,20,21), exactly as SVMShaderManager does before aggregation.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x00000014u, 0x00000015u,
      0x0000000bu, 0x00000001u, 0x00000000u,
      0x00000005u, 0x3f2e147bu, 0x3e75c28fu, 0x3db851ecu,
      0x00000002u, 0x00000002u, 0x000000ffu,
      0x3f2e147bu, 0x3e75c28fu, 0x3db851ecu, 0x3edc28f6u,
      0x00000000u, 0x00000000u,
      0x00000000u,
      0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles Diffuse SVM differs from the Cycles word oracle");
  require(image.peak_stack_usage == 3u,
          "Diffuse SVM peak stack usage differs from Cycles");
  require(image.node_types_used[NODE_SHADER_JUMP] &&
              image.node_types_used[NODE_GEOMETRY] &&
              image.node_types_used[NODE_CLOSURE_SET_WEIGHT] &&
              image.node_types_used[NODE_CLOSURE_BSDF] &&
              image.node_types_used[NODE_END],
          "Diffuse SVM node usage mask differs from Cycles");
}

void test_constant_mix_closure_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto transparent =
      graph.add_node(node_type::transparent_bsdf, "Transparent BSDF");
  require(graph.set_input(
              transparent, "Color",
              SocketValue::color({0.75f, 0.9f, 0.6f})),
          "failed to set Transparent Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_input(
              emission, "Color",
              SocketValue::color({0.85f, 0.08f, 0.03f})),
          "failed to set Emission Color");
  require(graph.set_input(emission, "Strength",
                          SocketValue::floating(1.2f)),
          "failed to set Emission Strength");
  const auto mix = graph.add_node(node_type::mix_closure, "Mix Shader");
  require(graph.set_input(mix, "Factor", SocketValue::floating(0.62f)),
          "failed to set Mix factor");
  require(graph.connect(OutputRef{transparent, "Closure"}, mix, "A"),
          "failed to connect Transparent branch");
  require(graph.connect(OutputRef{emission, "Closure"}, mix, "B"),
          "failed to connect Emission branch");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "constant Mix Shader graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from Cycles 5.2.1 shader `Transparent Probe`. The global stream
  // jump (95,114,115) is normalized to this local 25-word image. This locks
  // ShaderGraph::transform_multi_closure, NODE_MIX_CLOSURE stack placement,
  // branch order, and both leaf closure compilers as one oracle.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x00000017u, 0x00000018u,
      0x00000008u, 0x3f1eb852u, 0x000100ffu,
      0x00000005u, 0x3f400000u, 0x3f666666u, 0x3f19999au,
      0x00000002u, 0x0000001eu, 0x00000000u, 0x00000000u,
      0x00000000u,
      0x00000005u, 0x3f828f5du, 0x3dc49ba6u, 0x3d1374bdu,
      0x00000003u, 0x00000001u,
      0x00000000u,
      0x00000000u,
      0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles constant Mix SVM differs from Cycles");
  require(image.peak_stack_usage == 2u,
          "constant Mix SVM peak stack usage differs from Cycles");
}

void test_linked_mix_closure_jumps_match_cycles_5_2_1() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto transparent =
      graph.add_node(node_type::transparent_bsdf, "Transparent BSDF");
  require(graph.set_input(
              transparent, "Color",
              SocketValue::color({0.75f, 0.9f, 0.6f})),
          "failed to set dynamic-mix Transparent Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_input(
              emission, "Color",
              SocketValue::color({0.85f, 0.08f, 0.03f})),
          "failed to set dynamic-mix Emission Color");
  require(graph.set_input(emission, "Strength",
                          SocketValue::floating(1.2f)),
          "failed to set dynamic-mix Emission Strength");
  const auto mix = graph.add_node(node_type::mix_closure,
                                  "Dynamic Mix Shader");
  require(graph.connect(OutputRef{geometry, "Backfacing"}, mix, "Factor"),
          "failed to connect dynamic Mix factor");
  require(graph.connect(OutputRef{transparent, "Closure"}, mix, "A"),
          "failed to connect dynamic Transparent branch");
  require(graph.connect(OutputRef{emission, "Closure"}, mix, "B"),
          "failed to connect dynamic Emission branch");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "linked Mix Shader graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from the Cycles 5.2.1 `Dynamic Mix Probe` added beside this test.
  // Unlike the constant-factor oracle, this requires the exact linked-factor
  // stack lane and both forward jump distances from generate_multi_closure.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x00000020u, 0x00000021u,
      0x00000032u, 0x00000008u, 0x00000000u,
      0x00000008u, 0x7fc00000u, 0x000201ffu,
      0x0000000au, 0x00000009u, 0x00000000u,
      0x00000005u, 0x3f400000u, 0x3f666666u, 0x3f19999au,
      0x00000002u, 0x0000001eu, 0x00000001u, 0x00000000u,
      0x00000000u,
      0x00000009u, 0x00000006u, 0x00000000u,
      0x00000005u, 0x3f828f5du, 0x3dc49ba6u, 0x3d1374bdu,
      0x00000003u, 0x00000002u,
      0x00000000u,
      0x00000000u,
      0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles linked Mix SVM differs from Cycles");
  require(image.peak_stack_usage == 3u,
          "linked Mix SVM peak stack usage differs from Cycles");
}

void test_unsupported_node_rejects_without_old_fallback() {
  ShaderGraph graph;
  const auto principled =
      graph.add_node(node_type::principled_bsdf, "Not migrated yet");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = principled, .socket = "Closure"});
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "raw Principled graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(!image.valid &&
              image.diagnostic.find("not migrated") != std::string::npos,
          "unsupported Cycles SVM node silently selected another path");
}

} // namespace

int main() {
  test_math_third_input_default_matches_cycles_5_2_1();
  test_diffuse_surface_matches_cycles_5_2_1();
  test_constant_mix_closure_matches_cycles_5_2_1();
  test_linked_mix_closure_jumps_match_cycles_5_2_1();
  test_unsupported_node_rejects_without_old_fallback();
  return 0;
}

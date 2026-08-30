#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

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
  test_diffuse_surface_matches_cycles_5_2_1();
  test_unsupported_node_rejects_without_old_fallback();
  return 0;
}

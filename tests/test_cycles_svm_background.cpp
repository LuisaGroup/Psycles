#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

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
  if (actual.size() != expected.size()) {
    std::cerr << "Background SVM word count differs from Cycles 5.2.1: got "
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
      std::cerr << "Background SVM differs at word " << index << ": got 0x"
                << std::hex << actual[index] << ", expected 0x"
                << expected[index] << std::dec << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

void test_background_stream_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto background =
      graph.add_node(node_type::background, "Background");
  require(graph.set_input(
              background, "Color",
              SocketValue::color({0.16f, 0.48f, 0.77f})) &&
              graph.set_input(background, "Strength",
                              SocketValue::floating(2.3f)),
          "failed to construct Background graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = background, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "Background graph did not validate");
  AttributeIDMap attribute_ids;
  const auto image = compile_shader(*shader.program, attribute_ids);
  require(image.valid, image.diagnostic.c_str());

  // Shader 3 (`default_background`) from the Cycles 5.2.1
  // `background_world` diagnostic dump, with only the global jump-table
  // relocation normalized.
  static constexpr std::array<std::uint32_t, 13u> expected{
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu, 0x00000005u,
      0x3ebc6a7eu, 0x3f8d4fdfu, 0x3fe2b020u, 0x00000004u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  require_words(image.words, expected);
  require(image.peak_stack_usage == 0u &&
              image.node_types_used[NODE_CLOSURE_SET_WEIGHT] &&
              image.node_types_used[NODE_CLOSURE_BACKGROUND] &&
              !image.node_types_used[NODE_CLOSURE_EMISSION],
          "Background stack or opcode set differs from Cycles 5.2.1");
}

void test_zero_background_fold_matches_cycles_5_2_1() {
  const auto compile = [](psycles::Vec3f color, float strength) {
    ShaderGraph graph;
    const auto background =
        graph.add_node(node_type::background, "Zero Background");
    require(graph.set_input(background, "Color", SocketValue::color(color)) &&
                graph.set_input(background, "Strength",
                                SocketValue::floating(strength)),
            "failed to construct zero Background graph");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = background, .socket = "Closure"});
    const ShaderCompiler frontend{make_core_node_registry()};
    const auto shader = frontend.compile(graph);
    require(shader.ok(), "zero Background graph did not validate");
    AttributeIDMap attribute_ids;
    return compile_shader(*shader.program, attribute_ids);
  };

  static constexpr std::array<std::uint32_t, 7u> expected{
      0x00000001u, 0x00000004u, 0x00000005u, 0x00000006u,
      0x00000000u, 0x00000000u, 0x00000000u};
  for (const auto image : {compile({0.16f, 0.48f, 0.77f}, 0.0f),
                           compile({}, 2.3f)}) {
    require(image.valid, image.diagnostic.c_str());
    require_words(image.words, expected);
    require(image.peak_stack_usage == 0u &&
                image.node_types_used[NODE_END] &&
                !image.node_types_used[NODE_CLOSURE_SET_WEIGHT] &&
                !image.node_types_used[NODE_CLOSURE_BACKGROUND],
            "zero Background fold differs from Cycles 5.2.1");
  }
}

void test_linked_background_stream_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto normal_to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto vector_to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto background =
      graph.add_node(node_type::background, "Background");
  require(graph.connect({geometry, "Normal"}, normal_to_vector, "Normal") &&
              graph.connect({normal_to_vector, "Vector"}, vector_to_color,
                            "Vector") &&
              graph.connect({vector_to_color, "Color"}, background,
                            "Color") &&
              graph.connect({geometry, "Backfacing"}, background,
                            "Strength"),
          "failed to construct linked Background graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = background, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "linked Background graph did not validate");
  AttributeIDMap attribute_ids;
  const auto image = compile_shader(*shader.program, attribute_ids);
  require(image.valid, image.diagnostic.c_str());

  // Cycles 5.2.1 `background_world_linked`: Geometry Normal drives Color
  // and Geometry Backfacing drives Strength. Only the global jump-table
  // relocation is normalized.
  static constexpr std::array<std::uint32_t, 20u> expected{
      0x00000001u, 0x00000004u, 0x00000012u, 0x00000013u,
      0x0000000bu, 0x00000001u, 0x00000000u, 0x00000032u,
      0x00000008u, 0x00000003u, 0x00000007u, 0x7fc00000u,
      0x00000000u, 0x00000000u, 0x7fc00003u, 0x00000004u,
      0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};
  require_words(image.words, expected);
  require(image.peak_stack_usage == 4u &&
              image.node_types_used[NODE_GEOMETRY] &&
              image.node_types_used[NODE_LIGHT_PATH] &&
              image.node_types_used[NODE_EMISSION_WEIGHT] &&
              image.node_types_used[NODE_CLOSURE_BACKGROUND] &&
              !image.node_types_used[NODE_CLOSURE_SET_WEIGHT],
          "linked Background stack or opcode set differs from Cycles 5.2.1");
}

void test_mixed_background_stream_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto background_a =
      graph.add_node(node_type::background, "Background A");
  const auto background_b =
      graph.add_node(node_type::background, "Background B");
  const auto mix = graph.add_node(node_type::mix_closure, "Mix Shader");
  require(graph.set_input(background_a, "Color",
                          SocketValue::color({0.2f, 0.4f, 0.6f})) &&
              graph.set_input(background_a, "Strength",
                              SocketValue::floating(1.5f)) &&
              graph.set_input(background_b, "Color",
                              SocketValue::color({0.7f, 0.3f, 0.1f})) &&
              graph.set_input(background_b, "Strength",
                              SocketValue::floating(0.8f)) &&
              graph.connect({geometry, "Backfacing"}, mix, "Factor") &&
              graph.connect({background_a, "Closure"}, mix, "A") &&
              graph.connect({background_b, "Closure"}, mix, "B"),
          "failed to construct mixed Background graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "mixed Background graph did not validate");
  AttributeIDMap attribute_ids;
  const auto image = compile_shader(*shader.program, attribute_ids);
  require(image.valid, image.diagnostic.c_str());

  // Cycles 5.2.1 `background_world_mix`. This locks the two distinct
  // mix-weight lanes written by NODE_MIX_CLOSURE and consumed by the two
  // NODE_CLOSURE_BACKGROUND records, as well as closure branch order.
  static constexpr std::array<std::uint32_t, 31u> expected{
      0x00000001u, 0x00000004u, 0x0000001du, 0x0000001eu,
      0x00000032u, 0x00000008u, 0x00000000u, 0x00000008u,
      0x7fc00000u, 0x000201ffu, 0x0000000au, 0x00000006u,
      0x00000000u, 0x00000005u, 0x3e99999au, 0x3f19999au,
      0x3f666667u, 0x00000004u, 0x00000001u, 0x00000009u,
      0x00000006u, 0x00000000u, 0x00000005u, 0x3f0f5c29u,
      0x3e75c290u, 0x3da3d70bu, 0x00000004u, 0x00000002u,
      0x00000000u, 0x00000000u, 0x00000000u};
  require_words(image.words, expected);
  require(image.peak_stack_usage == 3u &&
              image.node_types_used[NODE_LIGHT_PATH] &&
              image.node_types_used[NODE_MIX_CLOSURE] &&
              image.node_types_used[NODE_JUMP_IF_ZERO] &&
              image.node_types_used[NODE_JUMP_IF_ONE] &&
              image.node_types_used[NODE_CLOSURE_BACKGROUND],
          "mixed Background stack or opcode set differs from Cycles 5.2.1");
}

} // namespace

int main() {
  test_background_stream_matches_cycles_5_2_1();
  test_zero_background_fold_matches_cycles_5_2_1();
  test_linked_background_stream_matches_cycles_5_2_1();
  test_mixed_background_stream_matches_cycles_5_2_1();
  return EXIT_SUCCESS;
}

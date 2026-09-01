#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_test_compile.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <utility>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

constexpr std::array mix_modes{
    std::pair{"MIX", NODE_MIX_BLEND},
    std::pair{"DARKEN", NODE_MIX_DARK},
    std::pair{"MULTIPLY", NODE_MIX_MUL},
    std::pair{"BURN", NODE_MIX_BURN},
    std::pair{"LIGHTEN", NODE_MIX_LIGHT},
    std::pair{"SCREEN", NODE_MIX_SCREEN},
    std::pair{"DODGE", NODE_MIX_DODGE},
    std::pair{"ADD", NODE_MIX_ADD},
    std::pair{"OVERLAY", NODE_MIX_OVERLAY},
    std::pair{"SOFT_LIGHT", NODE_MIX_SOFT},
    std::pair{"LINEAR_LIGHT", NODE_MIX_LINEAR},
    std::pair{"DIFFERENCE", NODE_MIX_DIFF},
    std::pair{"EXCLUSION", NODE_MIX_EXCLUSION},
    std::pair{"SUBTRACT", NODE_MIX_SUB},
    std::pair{"DIVIDE", NODE_MIX_DIV},
    std::pair{"HUE", NODE_MIX_HUE},
    std::pair{"SATURATION", NODE_MIX_SAT},
    std::pair{"COLOR", NODE_MIX_COL},
    std::pair{"VALUE", NODE_MIX_VAL},
};

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void require_words(std::span<const std::uint32_t> actual,
                   std::span<const std::uint32_t> expected,
                   std::string_view label) {
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
    for (const auto &node : graph.nodes()) {
      std::cerr << "node " << node.id.value << " " << node.type << '\n';
      for (const auto &[name, binding] : node.inputs) {
        if (binding.source) {
          std::cerr << "  " << name << " <- "
                    << binding.source->node.value << "."
                    << binding.source->socket << '\n';
        }
      }
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

[[nodiscard]] NodeId add_dynamic_factor(ShaderGraph &graph) {
  const auto geometry = graph.add_node(node_type::geometry, "Backfacing");
  const auto factor = graph.add_node(node_type::math, "Dynamic Factor");
  require(graph.set_property(factor, "Operation",
                             SocketValue::string("MULTIPLY_ADD")) &&
              graph.connect({geometry, "Backfacing"}, factor, "A") &&
              graph.set_input(factor, "B", SocketValue::floating(1.5f)) &&
              graph.set_input(factor, "C", SocketValue::floating(-0.25f)),
          "failed to create Cycles dynamic Mix factor");
  return factor;
}

[[nodiscard]] NodeId add_dynamic_color(ShaderGraph &graph) {
  const auto geometry = graph.add_node(node_type::geometry, "Backfacing");
  const auto combine = graph.add_node(node_type::combine_color, "Combine RGB");
  require(graph.set_property(combine, "Mode", SocketValue::string("RGB")) &&
              graph.connect({geometry, "Backfacing"}, combine, "R") &&
              graph.set_input(combine, "G", SocketValue::floating(-0.4f)) &&
              graph.set_input(combine, "B", SocketValue::floating(1.2f)),
          "failed to create Cycles dynamic color source");
  return combine;
}

void test_dynamic_mix_color() {
  static constexpr std::array base{
      0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
      0x00000032u, 0x00000008u, 0x00000000u,
      0x0000002cu, 0x00000024u, 0x7fc00000u, 0x3fc00000u,
      0xbe800000u, 0x00000001u,
      0x00000067u, 0x00000000u,
      0x3e2e147bu, 0x3f2147aeu, 0x3f63d70au,
      0x3f51eb85u, 0x3e75c28fu, 0x3f028f5cu,
      0x7fc00001u, 0x00020101u,
      0x00000007u, 0x7fc00002u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u,
  };

  for (auto index = std::size_t{}; index < mix_modes.size(); ++index) {
    const auto &[name, type] = mix_modes[index];
    ShaderGraph graph;
    const auto factor = add_dynamic_factor(graph);
    const auto mix = graph.add_node(node_type::mix_color, name);
    const auto emission = graph.add_node(node_type::emission, "Emission");
    const auto clamp_factor = index % 2u == 0u;
    const auto clamp_result = index % 3u == 0u;
    require(graph.set_property(mix, "BlendMode", SocketValue::string(name)) &&
                graph.set_property(mix, "ClampFactor",
                                   SocketValue::boolean(clamp_factor)) &&
                graph.set_property(mix, "ClampResult",
                                   SocketValue::boolean(clamp_result)) &&
                graph.connect({factor, "Value"}, mix, "Factor") &&
                graph.set_input(
                    mix, "A", SocketValue::color({0.17f, 0.63f, 0.89f})) &&
                graph.set_input(
                    mix, "B", SocketValue::color({0.82f, 0.24f, 0.51f})) &&
                graph.connect({mix, "Color"}, emission, "Color"),
            "failed to create dynamic Cycles MixColor graph");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = emission, .socket = "Closure"});

    const auto image = compile(graph);
    auto expected = base;
    expected[14] = static_cast<std::uint32_t>(type);
    expected[22] = 0x00020000u |
                   (clamp_result ? 0x00000100u : 0u) |
                   (clamp_factor ? 0x00000001u : 0u);
    require_words(image.words, expected, name);
    require(image.peak_stack_usage == 5u &&
                image.node_types_used[NODE_MIX_COLOR] &&
                !image.node_types_used[NODE_MIX],
            "dynamic MixColor stack or opcode mask differs from Cycles");
  }
}

void test_multiply_color_projects_to_cycles_mix_color() {
  ShaderGraph graph;
  const auto factor = add_dynamic_factor(graph);
  const auto multiply =
      graph.add_node(node_type::multiply_color, "Multiply Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({factor, "Value"}, multiply, "Factor") &&
              graph.set_input(
                  multiply, "A",
                  SocketValue::color({0.17f, 0.63f, 0.89f})) &&
              graph.set_input(
                  multiply, "B",
                  SocketValue::color({0.82f, 0.24f, 0.51f})) &&
              graph.connect({multiply, "Color"}, emission, "Color"),
          "failed to create canonical Multiply Color graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const auto image = compile(graph);
  // This is the same external Cycles 5.2.1 dynamic Mix Color oracle used
  // above, specialized to MULTIPLY, Clamp Factor=true, Clamp Result=false.
  // It proves that the canonical Psycles helper is normalized to the exact
  // Cycles node rather than acquiring a separate SVM encoding.
  static constexpr std::array expected{
      0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
      0x00000032u, 0x00000008u, 0x00000000u,
      0x0000002cu, 0x00000024u, 0x7fc00000u, 0x3fc00000u,
      0xbe800000u, 0x00000001u,
      0x00000067u, 0x00000002u,
      0x3e2e147bu, 0x3f2147aeu, 0x3f63d70au,
      0x3f51eb85u, 0x3e75c28fu, 0x3f028f5cu,
      0x7fc00001u, 0x00020001u,
      0x00000007u, 0x7fc00002u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u,
  };
  require_words(image.words, expected,
                "canonical Multiply Color projection");
  require(image.peak_stack_usage == 5u &&
              image.node_types_used[NODE_MIX_COLOR] &&
              !image.node_types_used[NODE_MIX],
          "canonical Multiply Color opcode or stack differs from Cycles");
}

void test_dynamic_mix_float() {
  static constexpr std::array base{
      0x00000001u, 0x00000004u, 0x0000001au, 0x0000001bu,
      0x00000032u, 0x00000008u, 0x00000000u,
      0x0000002cu, 0x00000024u, 0x7fc00000u, 0x3fc00000u,
      0xbe800000u, 0x00000001u,
      0x00000068u, 0x7fc00001u, 0x3e4ccccdu, 0x3f4ccccdu,
      0x00000000u,
      0x00000007u, 0x3e9eb852u, 0x3f11eb85u, 0x3f547ae1u,
      0x7fc00000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u,
  };
  for (const auto clamp : {false, true}) {
    ShaderGraph graph;
    const auto factor = add_dynamic_factor(graph);
    const auto mix = graph.add_node(node_type::mix_float, "Mix Float");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(graph.set_property(mix, "ClampFactor",
                               SocketValue::boolean(clamp)) &&
                graph.connect({factor, "Value"}, mix, "Factor") &&
                graph.set_input(mix, "A", SocketValue::floating(0.2f)) &&
                graph.set_input(mix, "B", SocketValue::floating(0.8f)) &&
                graph.set_input(
                    emission, "Color",
                    SocketValue::color({0.31f, 0.57f, 0.83f})) &&
                graph.connect({mix, "Value"}, emission, "Strength"),
            "failed to create dynamic Cycles MixFloat graph");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = emission, .socket = "Closure"});
    const auto image = compile(graph);
    auto expected = base;
    expected[17] = clamp ? 1u : 0u;
    require_words(image.words, expected,
                  clamp ? "clamped MixFloat" : "unclamped MixFloat");
    require(image.peak_stack_usage == 2u &&
                image.node_types_used[NODE_MIX_FLOAT],
            "dynamic MixFloat stack or opcode mask differs from Cycles");
  }
}

void test_dynamic_mix_vectors() {
  static constexpr std::array uniform_base{
      0x00000001u, 0x00000004u, 0x0000001eu, 0x0000001fu,
      0x00000032u, 0x00000008u, 0x00000000u,
      0x0000002cu, 0x00000024u, 0x7fc00000u, 0x3fc00000u,
      0xbe800000u, 0x00000001u,
      0x00000069u,
      0x3dcccccdu, 0x3f333333u, 0xbe4ccccdu,
      0x3f666666u, 0xbdcccccdu, 0x3f19999au,
      0x7fc00001u, 0x00000200u,
      0x00000007u, 0x7fc00002u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u,
  };
  static constexpr std::array nonuniform_base{
      0x00000001u, 0x00000004u, 0x0000001au, 0x0000001bu,
      0x0000000bu, 0x00000001u, 0x00000000u,
      0x0000006au,
      0x3dcccccdu, 0x3f333333u, 0xbe4ccccdu,
      0x3f666666u, 0xbdcccccdu, 0x3f19999au,
      0x7fc00000u, 0x00000000u, 0x00000000u, 0x00000300u,
      0x00000007u, 0x7fc00003u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u,
  };

  for (const auto nonuniform : {false, true}) {
    for (const auto clamp : {false, true}) {
      ShaderGraph graph;
      const auto mix = graph.add_node(
          nonuniform ? node_type::mix_vector_nonuniform
                     : node_type::mix_vector,
          nonuniform ? "Mix Vector Non Uniform" : "Mix Vector");
      if (nonuniform) {
        const auto geometry = graph.add_node(node_type::geometry, "Normal");
        const auto convert =
            graph.add_node(node_type::normal_to_vector, "Normal to Vector");
        require(graph.connect({geometry, "Normal"}, convert, "Normal") &&
                    graph.connect({convert, "Vector"}, mix, "Factor"),
                "failed to connect non-uniform MixVector factor");
      } else {
        const auto factor = add_dynamic_factor(graph);
        require(graph.connect({factor, "Value"}, mix, "Factor"),
                "failed to connect uniform MixVector factor");
      }
      const auto convert =
          graph.add_node(node_type::vector_to_color, "Vector to Color");
      const auto emission = graph.add_node(node_type::emission, "Emission");
      require(graph.set_property(mix, "ClampFactor",
                                 SocketValue::boolean(clamp)) &&
                  graph.set_input(
                      mix, "A", SocketValue::vector({0.1f, 0.7f, -0.2f})) &&
                  graph.set_input(
                      mix, "B", SocketValue::vector({0.9f, -0.1f, 0.6f})) &&
                  graph.connect({mix, "Vector"}, convert, "Vector") &&
                  graph.connect({convert, "Color"}, emission, "Color"),
              "failed to create dynamic Cycles MixVector graph");
      graph.set_root(ShaderDomain::surface,
                     OutputRef{.node = emission, .socket = "Closure"});
      const auto image = compile(graph);
      if (nonuniform) {
        auto expected = nonuniform_base;
        expected[17] |= clamp ? 1u : 0u;
        require_words(image.words, expected, "MixVector Non Uniform");
        require(image.peak_stack_usage == 6u &&
                    image.node_types_used[NODE_MIX_VECTOR_NON_UNIFORM],
                "non-uniform MixVector stack/opcode differs from Cycles");
      } else {
        auto expected = uniform_base;
        expected[21] |= clamp ? 1u : 0u;
        require_words(image.words, expected, "MixVector Uniform");
        require(image.peak_stack_usage == 5u &&
                    image.node_types_used[NODE_MIX_VECTOR],
                "uniform MixVector stack/opcode differs from Cycles");
      }
    }
  }
}

void test_constant_modern_mix() {
  static constexpr std::array<std::array<std::uint32_t, 3u>, 19u>
      color_oracle{{
          {0x3f51eb85u, 0x3e75c290u, 0x3f028f5cu},
          {0x3e2e147bu, 0x3dac0838u, 0x3eb74bc6u},
          {0x3e0ebee0u, 0x3e1ad42cu, 0x3ee86594u},
          {0x00000000u, 0x00000000u, 0x3f265187u},
          {0x3f51eb85u, 0x3f2147aeu, 0x3f63d70au},
          {0x3f8fb938u, 0x3f411b1du, 0x3f77f23du},
          {0x3f71c71cu, 0x3f5435e5u, 0x3f800000u},
          {0x3fa8b439u, 0x3f774bc6u, 0x3fcd4fdfu},
          {0x3e8ebee0u, 0x3ee00d1cu, 0x3f646738u},
          {0x3e97c519u, 0x3eebacdcu, 0x3f648ab0u},
          {0x3f4f5c29u, 0x3de147b0u, 0x3f68f5c2u},
          {0x3f578d4fu, 0x3e96872bu, 0x3e343958u},
          {0x3f361134u, 0x3f114e3cu, 0x3efc01a4u},
          {0xbf7a5e35u, 0x3e96872bu, 0x3e343958u},
          {0x3e544aeeu, 0x40280000u, 0x3fdf5f5fu},
          {0x3f800000u, 0x00000000u, 0x3eb3d5e2u},
          {0x3e855ea9u, 0x3f29a524u, 0x3f63d70au},
          {0x3f96c8b3u, 0x3de6c650u, 0x3ed680c1u},
          {0x3e206369u, 0x3f149855u, 0x3f51eb85u},
      }};

  for (auto index = std::size_t{}; index < mix_modes.size(); ++index) {
    ShaderGraph graph;
    const auto &[name, unused] = mix_modes[index];
    const auto mix = graph.add_node(node_type::mix_color, name);
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(graph.set_property(mix, "BlendMode", SocketValue::string(name)) &&
                graph.set_property(
                    mix, "ClampFactor",
                    SocketValue::boolean(index % 2u == 0u)) &&
                graph.set_property(
                    mix, "ClampResult",
                    SocketValue::boolean(index % 3u == 0u)) &&
                graph.set_input(mix, "Factor", SocketValue::floating(1.4f)) &&
                graph.set_input(
                    mix, "A", SocketValue::color({0.17f, 0.63f, 0.89f})) &&
                graph.set_input(
                    mix, "B", SocketValue::color({0.82f, 0.24f, 0.51f})) &&
                graph.connect({mix, "Color"}, emission, "Color"),
            "failed to create constant Cycles MixColor graph");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = emission, .socket = "Closure"});
    const auto image = compile(graph);
    const std::array expected{
        0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
        0x00000005u, color_oracle[index][0], color_oracle[index][1],
        color_oracle[index][2], 0x00000003u, 0x000000ffu,
        0x00000000u, 0x00000000u, 0x00000000u,
    };
    require_words(image.words, expected, name);
    require(!image.node_types_used[NODE_MIX_COLOR],
            "constant MixColor retained NODE_MIX_COLOR");
    static_cast<void>(unused);
  }

  static constexpr std::array typed_oracle{
      std::array{0x3ea5119du, 0x3f17c1bdu, 0x3f5cfaacu},
      std::array{0x3e7df3b7u, 0x3ee978d5u, 0x3f29fbe7u},
      std::array{0x3f9c28f6u, 0xbed70a3cu, 0x3f6b851fu},
      std::array{0x3f666666u, 0xbdcccccdu, 0x3f19999au},
      std::array{0xbd75c28cu, 0x3e999999u, 0x3f6b851fu},
      std::array{0x3dcccccdu, 0x3e999999u, 0x3f19999au},
  };
  for (auto case_index = std::size_t{}; case_index < typed_oracle.size();
       ++case_index) {
    const auto kind = case_index / 2u;
    const auto clamp = case_index % 2u != 0u;
    ShaderGraph graph;
    const auto type = kind == 0u
                          ? node_type::mix_float
                          : kind == 1u ? node_type::mix_vector
                                       : node_type::mix_vector_nonuniform;
    const auto mix = graph.add_node(type, "Constant Typed Mix");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(graph.set_property(mix, "ClampFactor",
                               SocketValue::boolean(clamp)),
            "failed to set typed Mix clamp");
    if (kind == 0u) {
      require(graph.set_input(mix, "Factor", SocketValue::floating(1.4f)) &&
                  graph.set_input(mix, "A", SocketValue::floating(0.2f)) &&
                  graph.set_input(mix, "B", SocketValue::floating(0.8f)) &&
                  graph.set_input(
                      emission, "Color",
                      SocketValue::color({0.31f, 0.57f, 0.83f})) &&
                  graph.connect({mix, "Value"}, emission, "Strength"),
              "failed to create constant MixFloat graph");
    } else {
      if (kind == 1u) {
        require(graph.set_input(mix, "Factor",
                                SocketValue::floating(1.4f)),
                "failed to set uniform MixVector factor");
      } else {
        require(graph.set_input(
                    mix, "Factor",
                    SocketValue::vector({-0.2f, 0.5f, 1.4f})),
                "failed to set non-uniform MixVector factor");
      }
      require(graph.set_input(
                  mix, "A", SocketValue::vector({0.1f, 0.7f, -0.2f})) &&
                  graph.set_input(
                      mix, "B", SocketValue::vector({0.9f, -0.1f, 0.6f})),
              "failed to create constant MixVector graph");
      const auto convert =
          graph.add_node(node_type::vector_to_color, "Vector to Color");
      require(graph.connect({mix, "Vector"}, convert, "Vector") &&
                  graph.connect({convert, "Color"}, emission, "Color"),
              "failed to convert constant MixVector result");
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = emission, .socket = "Closure"});
    const auto image = compile(graph);
    const std::array expected{
        0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
        0x00000005u, typed_oracle[case_index][0],
        typed_oracle[case_index][1], typed_oracle[case_index][2],
        0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
        0x00000000u,
    };
    require_words(image.words, expected, "constant typed Mix");
    require(!image.node_types_used[NODE_MIX_FLOAT] &&
                !image.node_types_used[NODE_MIX_VECTOR] &&
                !image.node_types_used[NODE_MIX_VECTOR_NON_UNIFORM],
            "constant typed Mix retained an opcode");
  }
}

void test_partial_fold_edges() {
  {
    ShaderGraph graph;
    const auto color = add_dynamic_color(graph);
    const auto mix = graph.add_node(node_type::mix_color, "Same Link Clamp");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(graph.set_property(mix, "BlendMode", SocketValue::string("MIX")) &&
                graph.set_property(mix, "ClampFactor",
                                   SocketValue::boolean(true)) &&
                graph.set_property(mix, "ClampResult",
                                   SocketValue::boolean(true)) &&
                graph.set_input(mix, "Factor", SocketValue::floating(0.37f)) &&
                graph.set_input(
                    mix, "A", SocketValue::color({0.5f, 0.5f, 0.5f})) &&
                graph.set_input(
                    mix, "B", SocketValue::color({0.5f, 0.5f, 0.5f})) &&
                graph.connect({color, "Color"}, mix, "A") &&
                graph.connect({color, "Color"}, mix, "B") &&
                graph.connect({mix, "Color"}, emission, "Color"),
            "failed to create same-link clamped MixColor graph");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = emission, .socket = "Closure"});
    const auto image = compile(graph);
    static constexpr std::array expected{
        0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
        0x00000032u, 0x00000008u, 0x00000000u,
        0x00000053u, 0x00000000u, 0x7fc00000u, 0xbecccccdu,
        0x3f99999au, 0x00000001u,
        0x00000067u, 0x00000000u,
        0x7fc00001u, 0x00000000u, 0x00000000u,
        0x3f000000u, 0x3f000000u, 0x3f000000u,
        0x00000000u, 0x00040101u,
        0x00000007u, 0x7fc00004u, 0x00000000u, 0x00000000u,
        0x3f800000u, 0x00000003u, 0x000000ffu,
        0x00000000u, 0x00000000u, 0x00000000u,
    };
    require_words(image.words, expected, "same-link clamped MixColor fold");
    require(image.peak_stack_usage == 7u &&
                image.node_types_used[NODE_MIX_COLOR],
            "same-link clamped MixColor lifetime differs from Cycles");
  }

  {
    ShaderGraph graph;
    const auto color = add_dynamic_color(graph);
    const auto mix = graph.add_node(node_type::mix_color, "Factor Zero");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(graph.set_property(mix, "BlendMode", SocketValue::string("MIX")) &&
                graph.set_property(mix, "ClampFactor",
                                   SocketValue::boolean(true)) &&
                graph.set_property(mix, "ClampResult",
                                   SocketValue::boolean(false)) &&
                graph.set_input(mix, "Factor", SocketValue::floating(0.0f)) &&
                graph.set_input(
                    mix, "B", SocketValue::color({0.8f, 0.2f, 0.6f})) &&
                graph.connect({color, "Color"}, mix, "A") &&
                graph.connect({mix, "Color"}, emission, "Color"),
            "failed to create zero-factor MixColor graph");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = emission, .socket = "Closure"});
    const auto image = compile(graph);
    static constexpr std::array expected{
        0x00000001u, 0x00000004u, 0x00000015u, 0x00000016u,
        0x00000032u, 0x00000008u, 0x00000000u,
        0x00000053u, 0x00000000u, 0x7fc00000u, 0xbecccccdu,
        0x3f99999au, 0x00000001u,
        0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
        0x3f800000u, 0x00000003u, 0x000000ffu,
        0x00000000u, 0x00000000u, 0x00000000u,
    };
    require_words(image.words, expected, "zero-factor MixColor fold");
    require(image.peak_stack_usage == 4u &&
                !image.node_types_used[NODE_MIX_COLOR],
            "zero-factor MixColor was not bypassed like Cycles");
  }

  {
    ShaderGraph graph;
    const auto geometry = graph.add_node(node_type::geometry, "Backfacing");
    const auto mix = graph.add_node(node_type::mix_float, "Factor One");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(graph.set_property(mix, "ClampFactor",
                               SocketValue::boolean(true)) &&
                graph.set_input(mix, "Factor", SocketValue::floating(1.0f)) &&
                graph.set_input(mix, "A", SocketValue::floating(0.2f)) &&
                graph.connect({geometry, "Backfacing"}, mix, "B") &&
                graph.set_input(
                    emission, "Color",
                    SocketValue::color({0.31f, 0.57f, 0.83f})) &&
                graph.connect({mix, "Value"}, emission, "Strength"),
            "failed to create one-factor MixFloat graph");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = emission, .socket = "Closure"});
    const auto image = compile(graph);
    static constexpr std::array expected{
        0x00000001u, 0x00000004u, 0x0000000fu, 0x00000010u,
        0x00000032u, 0x00000008u, 0x00000000u,
        0x00000007u, 0x3e9eb852u, 0x3f11eb85u, 0x3f547ae1u,
        0x7fc00000u, 0x00000003u, 0x000000ffu,
        0x00000000u, 0x00000000u, 0x00000000u,
    };
    require_words(image.words, expected, "one-factor MixFloat fold");
    require(image.peak_stack_usage == 1u &&
                !image.node_types_used[NODE_MIX_FLOAT],
            "one-factor MixFloat was not bypassed like Cycles");
  }
}

} // namespace

int main() {
  test_dynamic_mix_color();
  test_multiply_color_projects_to_cycles_mix_color();
  test_dynamic_mix_float();
  test_dynamic_mix_vectors();
  test_constant_modern_mix();
  test_partial_fold_edges();
  return EXIT_SUCCESS;
}

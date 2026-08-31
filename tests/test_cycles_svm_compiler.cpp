#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_test_compile.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

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

void test_dynamic_math_and_dedup_match_cycles_5_2_1() {
  ShaderGraph graph;
  const auto geometry_a = graph.add_node(node_type::geometry, "Geometry A");
  const auto geometry_b = graph.add_node(node_type::geometry, "Geometry B");
  const auto math = graph.add_node(node_type::math, "Dynamic Add");
  require(graph.set_property(math, "Operation", SocketValue::string("ADD")),
          "failed to set dynamic Math operation");
  require(graph.connect(OutputRef{geometry_a, "Backfacing"}, math, "A"),
          "failed to connect first duplicate Geometry");
  require(graph.connect(OutputRef{geometry_b, "Backfacing"}, math, "B"),
          "failed to connect second duplicate Geometry");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_input(emission, "Color",
                          SocketValue::color({0.21f, 0.47f, 0.83f})),
          "failed to set dynamic-Math Emission Color");
  require(graph.connect(OutputRef{math, "Value"}, emission, "Strength"),
          "failed to connect dynamic Math to Emission Strength");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "dynamic Math/dedup graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from Cycles 5.2.1 `SVM Math Dedup Probe`. The two separately
  // authored Geometry nodes are deduplicated before scheduling: NODE_MATH
  // therefore reads lane 0 for both Value1 and Value2, and only one
  // NODE_LIGHT_PATH instruction exists.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x00000015u, 0x00000016u,
      0x00000032u, 0x00000008u, 0x00000000u,
      0x0000002cu, 0x00000000u, 0x7fc00000u, 0x7fc00000u,
      0x00000000u, 0x00000001u,
      0x00000007u, 0x3e570a3du, 0x3ef0a3d7u, 0x3f547ae1u,
      0x7fc00001u,
      0x00000003u, 0x000000ffu,
      0x00000000u,
      0x00000000u,
      0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles dynamic Math/dedup SVM differs from Cycles");
  require(image.peak_stack_usage == 2u,
          "dynamic Math/dedup peak stack usage differs from Cycles");
}

void test_constant_math_fold_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto math = graph.add_node(node_type::math, "Constant Add");
  require(graph.set_property(math, "Operation", SocketValue::string("ADD")),
          "failed to set constant Math operation");
  require(graph.set_input(math, "A", SocketValue::floating(0.12f)) &&
              graph.set_input(math, "B", SocketValue::floating(0.23f)) &&
              graph.set_input(math, "C", SocketValue::floating(0.0f)),
          "failed to set constant Math inputs");
  const auto diffuse = graph.add_node(node_type::diffuse_bsdf, "Diffuse BSDF");
  require(graph.set_input(diffuse, "Color",
                          SocketValue::color({0.68f, 0.24f, 0.09f})),
          "failed to set folded-Math Diffuse Color");
  require(graph.connect(OutputRef{math, "Value"}, diffuse, "Roughness"),
          "failed to connect constant Math to Diffuse Roughness");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "constant Math graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from Cycles 5.2.1 `SVM Math Constant Fold Probe`. NODE_MATH is
  // absent and its exact float result, 0.35f (0x3eb33333), is embedded in the
  // Diffuse payload after ShaderGraph::constant_fold.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x00000014u, 0x00000015u,
      0x0000000bu, 0x00000001u, 0x00000000u,
      0x00000005u, 0x3f2e147bu, 0x3e75c28fu, 0x3db851ecu,
      0x00000002u, 0x00000002u, 0x000000ffu,
      0x3f2e147bu, 0x3e75c28fu, 0x3db851ecu, 0x3eb33333u,
      0x00000000u, 0x00000000u,
      0x00000000u,
      0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles constant-folded Math SVM differs from Cycles");
  require(!image.node_types_used[NODE_MATH],
          "constant-folded Math still emitted NODE_MATH");
}

void test_zero_mix_closure_fold_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto transparent =
      graph.add_node(node_type::transparent_bsdf, "Transparent BSDF");
  require(graph.set_input(transparent, "Color",
                          SocketValue::color({0.75f, 0.9f, 0.6f})),
          "failed to set folded Mix Transparent Color");
  const auto emission =
      graph.add_node(node_type::emission, "Discarded Emission");
  require(graph.set_input(emission, "Color",
                          SocketValue::color({0.85f, 0.08f, 0.03f})) &&
              graph.set_input(emission, "Strength",
                              SocketValue::floating(1.2f)),
          "failed to set folded Mix Emission");
  const auto mix = graph.add_node(node_type::mix_closure, "Folded Mix Shader");
  require(graph.set_input(mix, "Factor", SocketValue::floating(0.0f)),
          "failed to set folded Mix factor");
  require(graph.connect(OutputRef{transparent, "Closure"}, mix, "A") &&
              graph.connect(OutputRef{emission, "Closure"}, mix, "B"),
          "failed to connect folded Mix branches");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "zero-factor Mix graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from Cycles 5.2.1 `SVM Mix Closure Fold Probe`. The Mix and its
  // Emission branch disappear; the surviving Transparent closure has the
  // invalid parent-weight offset because no closure-weight transform remains.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x0000000eu, 0x0000000fu,
      0x00000005u, 0x3f400000u, 0x3f666666u, 0x3f19999au,
      0x00000002u, 0x0000001eu, 0x000000ffu, 0x00000000u,
      0x00000000u,
      0x00000000u,
      0x00000000u,
      0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles zero-factor Mix fold differs from Cycles");
  require(!image.node_types_used[NODE_MIX_CLOSURE] &&
              !image.node_types_used[NODE_CLOSURE_EMISSION],
          "zero-factor Mix retained its discarded branch");
}

void test_dynamic_color_pipeline_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Dynamic Backfacing");
  const auto invert = graph.add_node(node_type::invert_color, "Dynamic Invert");
  require(graph.set_input(invert, "Color",
                          SocketValue::color({0.12f, 0.47f, 0.81f})) &&
              graph.connect(OutputRef{geometry, "Backfacing"}, invert,
                            "Factor"),
          "failed to configure dynamic Invert");

  const auto gamma = graph.add_node(node_type::gamma_color, "Gamma");
  require(graph.set_input(gamma, "Gamma", SocketValue::floating(2.2f)) &&
              graph.connect(OutputRef{invert, "Color"}, gamma, "Color"),
          "failed to configure Gamma");

  const auto brightness = graph.add_node(
      node_type::brightness_contrast, "Dynamic Brightness Contrast");
  require(graph.set_input(brightness, "Bright",
                          SocketValue::floating(0.17f)) &&
              graph.connect(OutputRef{gamma, "Color"}, brightness, "Color") &&
              graph.connect(OutputRef{geometry, "Backfacing"}, brightness,
                            "Contrast"),
          "failed to configure dynamic Brightness Contrast");

  const auto hsv = graph.add_node(node_type::hue_saturation, "Dynamic HSV");
  require(graph.set_input(hsv, "Hue", SocketValue::floating(0.3f)) &&
              graph.set_input(hsv, "Saturation",
                              SocketValue::floating(1.4f)) &&
              graph.set_input(hsv, "Value", SocketValue::floating(0.75f)) &&
              graph.connect(OutputRef{brightness, "Color"}, hsv, "Color") &&
              graph.connect(OutputRef{geometry, "Backfacing"}, hsv,
                            "Factor"),
          "failed to configure dynamic HSV");

  const auto clamp = graph.add_node(node_type::clamp_range,
                                    "Dynamic Range Clamp");
  require(graph.set_property(clamp, "Mode", SocketValue::string("RANGE")) &&
              graph.set_input(clamp, "Min", SocketValue::floating(0.8f)) &&
              graph.set_input(clamp, "Max", SocketValue::floating(0.2f)) &&
              graph.connect(OutputRef{geometry, "Backfacing"}, clamp,
                            "Value"),
          "failed to configure dynamic RANGE Clamp");

  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect(OutputRef{hsv, "Color"}, emission, "Color") &&
              graph.connect(OutputRef{clamp, "Result"}, emission, "Strength"),
          "failed to connect color pipeline to Emission");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "dynamic color pipeline graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from Cycles 5.2.1 `SVM Color Pipeline Probe`. The global shader-5
  // jump (89,134,135) is normalized to this local 51-word stream. Every color
  // node remains live because Geometry::Backfacing drives its dynamic path.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x00000031u, 0x00000032u,
      0x00000032u, 0x00000008u, 0x00000000u,
      0x00000050u, 0x3df5c28fu, 0x3ef0a3d7u, 0x3f4f5c29u,
      0x7fc00000u, 0x00000001u,
      0x00000030u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x400ccccdu, 0x00000004u,
      0x00000031u, 0x7fc00004u, 0x00000000u, 0x00000000u,
      0x3e2e147bu, 0x7fc00000u, 0x00000001u,
      0x00000025u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3e99999au, 0x3fb33333u, 0x3f400000u, 0x7fc00000u,
      0x00000004u,
      0x0000005fu, 0x00000001u, 0x3f4ccccdu, 0x3e4ccccdu,
      0x7fc00000u, 0x00000001u,
      0x00000007u, 0x7fc00004u, 0x00000000u, 0x00000000u,
      0x7fc00001u,
      0x00000003u, 0x000000ffu,
      0x00000000u,
      0x00000000u,
      0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles color pipeline SVM differs from Cycles");
  require(image.peak_stack_usage == 7u,
          "color pipeline peak stack usage differs from Cycles");
  require(image.node_types_used[NODE_INVERT] &&
              image.node_types_used[NODE_GAMMA] &&
              image.node_types_used[NODE_BRIGHTCONTRAST] &&
              image.node_types_used[NODE_HSV] &&
              image.node_types_used[NODE_CLAMP],
          "color pipeline opcode usage differs from Cycles");
}

void test_color_constant_fold_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto invert = graph.add_node(node_type::invert_color,
                                     "Constant Invert");
  require(graph.set_input(invert, "Color",
                          SocketValue::color({0.12f, 0.47f, 0.81f})) &&
              graph.set_input(invert, "Factor",
                              SocketValue::floating(0.37f)),
          "failed to configure constant Invert");
  const auto gamma = graph.add_node(node_type::gamma_color,
                                    "Constant Gamma");
  require(graph.set_input(gamma, "Gamma", SocketValue::floating(2.2f)) &&
              graph.connect(OutputRef{invert, "Color"}, gamma, "Color"),
          "failed to configure constant Gamma");
  const auto brightness = graph.add_node(
      node_type::brightness_contrast, "Constant Brightness Contrast");
  require(graph.set_input(brightness, "Bright",
                          SocketValue::floating(0.17f)) &&
              graph.set_input(brightness, "Contrast",
                              SocketValue::floating(-0.35f)) &&
              graph.connect(OutputRef{gamma, "Color"}, brightness, "Color"),
          "failed to configure constant Brightness Contrast");
  const auto clamp = graph.add_node(node_type::clamp_range,
                                    "Constant Range Clamp");
  require(graph.set_property(clamp, "Mode", SocketValue::string("RANGE")) &&
              graph.set_input(clamp, "Value",
                              SocketValue::floating(0.1f)) &&
              graph.set_input(clamp, "Min", SocketValue::floating(0.8f)) &&
              graph.set_input(clamp, "Max", SocketValue::floating(0.2f)),
          "failed to configure constant RANGE Clamp");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect(OutputRef{brightness, "Color"}, emission, "Color") &&
              graph.connect(OutputRef{clamp, "Result"}, emission, "Strength"),
          "failed to connect constant color pipeline");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "constant color pipeline graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from Cycles 5.2.1 `SVM Color Constant Fold Probe`. All four value
  // nodes disappear and their composed result is embedded in closure weight.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
      0x00000005u, 0x3db1030eu, 0x3dc5492bu, 0x3dddd033u,
      0x00000003u, 0x000000ffu,
      0x00000000u,
      0x00000000u,
      0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles constant color folds differ from Cycles");
  require(!image.node_types_used[NODE_INVERT] &&
              !image.node_types_used[NODE_GAMMA] &&
              !image.node_types_used[NODE_BRIGHTCONTRAST] &&
              !image.node_types_used[NODE_CLAMP],
          "constant color pipeline retained folded opcodes");
}

void test_dynamic_combsep_color_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto geometry =
      graph.add_node(node_type::geometry, "Dynamic Backfacing");
  const auto combine_hsl =
      graph.add_node(node_type::combine_color, "Dynamic Combine HSL");
  require(
      graph.set_property(combine_hsl, "Mode", SocketValue::string("HSL")) &&
          graph.set_input(combine_hsl, "R", SocketValue::floating(0.13f)) &&
          graph.set_input(combine_hsl, "B", SocketValue::floating(0.36f)) &&
          graph.connect(OutputRef{geometry, "Backfacing"}, combine_hsl, "G"),
      "failed to configure dynamic HSL Combine Color");

  const auto separate_hsv =
      graph.add_node(node_type::separate_color, "Dynamic Separate HSV");
  require(
      graph.set_property(separate_hsv, "Mode", SocketValue::string("HSV")) &&
          graph.connect(OutputRef{combine_hsl, "Color"}, separate_hsv, "Color"),
      "failed to configure dynamic HSV Separate Color");

  const auto combine_rgb =
      graph.add_node(node_type::combine_color, "Dynamic Combine RGB");
  require(graph.set_property(combine_rgb, "Mode", SocketValue::string("RGB")) &&
              graph.connect(OutputRef{separate_hsv, "B"}, combine_rgb, "R") &&
              graph.connect(OutputRef{separate_hsv, "R"}, combine_rgb, "G") &&
              graph.connect(OutputRef{separate_hsv, "G"}, combine_rgb, "B"),
          "failed to configure dynamic RGB Combine Color");

  const auto separate_hsl =
      graph.add_node(node_type::separate_color, "Dynamic Separate HSL");
  require(
      graph.set_property(separate_hsl, "Mode", SocketValue::string("HSL")) &&
          graph.connect(OutputRef{combine_rgb, "Color"}, separate_hsl, "Color"),
      "failed to configure dynamic HSL Separate Color");
  const auto combine_hsv =
      graph.add_node(node_type::combine_color, "Dynamic Combine HSV");
  require(graph.set_property(combine_hsv, "Mode", SocketValue::string("HSV")) &&
              graph.connect(OutputRef{separate_hsl, "R"}, combine_hsv, "R") &&
              graph.connect(OutputRef{separate_hsl, "G"}, combine_hsv, "G") &&
              graph.connect(OutputRef{separate_hsl, "B"}, combine_hsv, "B"),
          "failed to configure dynamic HSV Combine Color");
  const auto separate_rgb =
      graph.add_node(node_type::separate_color, "Dynamic Separate RGB");
  require(
      graph.set_property(separate_rgb, "Mode", SocketValue::string("RGB")) &&
          graph.connect(OutputRef{combine_hsv, "Color"}, separate_rgb, "Color"),
      "failed to configure dynamic RGB Separate Color");
  const auto combine_rgb_final =
      graph.add_node(node_type::combine_color, "Dynamic Final Combine RGB");
  require(
      graph.set_property(combine_rgb_final, "Mode",
                         SocketValue::string("RGB")) &&
          graph.connect(OutputRef{separate_rgb, "B"}, combine_rgb_final, "R") &&
          graph.connect(OutputRef{separate_rgb, "R"}, combine_rgb_final, "G") &&
          graph.connect(OutputRef{separate_rgb, "G"}, combine_rgb_final, "B"),
      "failed to configure dynamic final RGB Combine Color");

  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.connect(OutputRef{combine_rgb_final, "Color"}, emission, "Color"),
      "failed to connect dynamic Combine/Separate Color pipeline");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "dynamic Combine/Separate Color graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from Cycles 5.2.1 `SVM Combine Separate Color Probe`. Shader 5's
  // global jump (89,142,143) is normalized to this local 59-word stream. The
  // chain exercises RGB, HSV, and HSL in both combine and separate directions.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x00000039u, 0x0000003au, 0x00000032u,
      0x00000008u, 0x00000000u, 0x00000053u, 0x00000002u, 0x3e051eb8u,
      0x7fc00000u, 0x3eb851ecu, 0x00000001u, 0x00000052u, 0x00000001u,
      0x7fc00001u, 0x00000000u, 0x00000000u, 0x00050400u, 0x00000053u,
      0x00000000u, 0x7fc00005u, 0x7fc00000u, 0x7fc00004u, 0x00000001u,
      0x00000052u, 0x00000002u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x00050400u, 0x00000053u, 0x00000001u, 0x7fc00000u, 0x7fc00004u,
      0x7fc00005u, 0x00000001u, 0x00000052u, 0x00000000u, 0x7fc00001u,
      0x00000000u, 0x00000000u, 0x00050400u, 0x00000053u, 0x00000000u,
      0x7fc00005u, 0x7fc00000u, 0x7fc00004u, 0x00000001u, 0x00000007u,
      0x7fc00001u, 0x00000000u, 0x00000000u, 0x3f800000u, 0x00000003u,
      0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles Combine/Separate Color SVM differs from Cycles");
  require(image.peak_stack_usage == 6u,
          "Combine/Separate Color peak stack usage differs from Cycles");
  require(image.node_types_used[NODE_SEPARATE_COLOR] &&
              image.node_types_used[NODE_COMBINE_COLOR],
          "Combine/Separate Color opcode usage differs from Cycles");
}

void test_combsep_color_constant_fold_matches_cycles_5_2_1() {
  ShaderGraph graph;
  const auto combine_hsl =
      graph.add_node(node_type::combine_color, "Constant Combine HSL");
  require(graph.set_property(combine_hsl, "Mode", SocketValue::string("HSL")) &&
              graph.set_input(combine_hsl, "R", SocketValue::floating(0.13f)) &&
              graph.set_input(combine_hsl, "G", SocketValue::floating(0.55f)) &&
              graph.set_input(combine_hsl, "B", SocketValue::floating(0.36f)),
          "failed to configure constant HSL Combine Color");
  const auto separate_hsv =
      graph.add_node(node_type::separate_color, "Constant Separate HSV");
  require(
      graph.set_property(separate_hsv, "Mode", SocketValue::string("HSV")) &&
          graph.connect(OutputRef{combine_hsl, "Color"}, separate_hsv, "Color"),
      "failed to configure constant HSV Separate Color");
  const auto combine_rgb =
      graph.add_node(node_type::combine_color, "Constant Combine RGB");
  require(graph.set_property(combine_rgb, "Mode", SocketValue::string("RGB")) &&
              graph.connect(OutputRef{separate_hsv, "B"}, combine_rgb, "R") &&
              graph.connect(OutputRef{separate_hsv, "R"}, combine_rgb, "G") &&
              graph.connect(OutputRef{separate_hsv, "G"}, combine_rgb, "B"),
          "failed to configure constant RGB Combine Color");
  const auto separate_hsl =
      graph.add_node(node_type::separate_color, "Constant Separate HSL");
  require(
      graph.set_property(separate_hsl, "Mode", SocketValue::string("HSL")) &&
          graph.connect(OutputRef{combine_rgb, "Color"}, separate_hsl, "Color"),
      "failed to configure constant HSL Separate Color");
  const auto combine_hsv =
      graph.add_node(node_type::combine_color, "Constant Combine HSV");
  require(graph.set_property(combine_hsv, "Mode", SocketValue::string("HSV")) &&
              graph.connect(OutputRef{separate_hsl, "R"}, combine_hsv, "R") &&
              graph.connect(OutputRef{separate_hsl, "G"}, combine_hsv, "G") &&
              graph.connect(OutputRef{separate_hsl, "B"}, combine_hsv, "B"),
          "failed to configure constant HSV Combine Color");
  const auto separate_rgb =
      graph.add_node(node_type::separate_color, "Constant Separate RGB");
  require(
      graph.set_property(separate_rgb, "Mode", SocketValue::string("RGB")) &&
          graph.connect(OutputRef{combine_hsv, "Color"}, separate_rgb, "Color"),
      "failed to configure constant RGB Separate Color");
  const auto combine_rgb_final =
      graph.add_node(node_type::combine_color, "Constant Final Combine RGB");
  require(
      graph.set_property(combine_rgb_final, "Mode",
                         SocketValue::string("RGB")) &&
          graph.connect(OutputRef{separate_rgb, "B"}, combine_rgb_final, "R") &&
          graph.connect(OutputRef{separate_rgb, "R"}, combine_rgb_final, "G") &&
          graph.connect(OutputRef{separate_rgb, "G"}, combine_rgb_final, "B"),
      "failed to configure constant final RGB Combine Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.connect(OutputRef{combine_rgb_final, "Color"}, emission, "Color"),
      "failed to connect constant Combine/Separate Color pipeline");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(),
          "constant Combine/Separate Color graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());

  // Frozen from Cycles 5.2.1 `SVM Combine Separate Color Fold Probe`.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu, 0x00000005u,
      0x3ed6f51au, 0x3eb020c6u, 0x3e051eb7u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles Combine/Separate Color folds differ from Cycles");
  require(!image.node_types_used[NODE_SEPARATE_COLOR] &&
              !image.node_types_used[NODE_COMBINE_COLOR],
          "constant Combine/Separate Color retained folded opcodes");
}

constexpr std::array legacy_mix_modes{
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

void test_dynamic_legacy_mix_matches_cycles_5_2_1() {
  static constexpr std::array normal_oracle{
      0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u, 0x00000032u,
      0x00000008u, 0x00000000u, 0x0000002cu, 0x00000024u, 0x7fc00000u,
      0x3ed1eb85u, 0x3e6b851fu, 0x00000001u, 0x00000051u, 0x00000000u,
      0x3e2e147bu, 0x3f2147aeu, 0x3f63d70au, 0x3f51eb85u, 0x3e75c28fu,
      0x3f028f5cu, 0x7fc00001u, 0x00000002u, 0x00000007u, 0x7fc00002u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u,
  };
  static constexpr std::array clamped_add_oracle{
      0x00000001u, 0x00000004u, 0x00000029u, 0x0000002au, 0x00000032u,
      0x00000008u, 0x00000000u, 0x0000002cu, 0x00000024u, 0x7fc00000u,
      0x3ed1eb85u, 0x3e6b851fu, 0x00000001u, 0x00000051u, 0x00000001u,
      0x3e2e147bu, 0x3f2147aeu, 0x3f63d70au, 0x3f51eb85u, 0x3e75c28fu,
      0x3f028f5cu, 0x7fc00001u, 0x00000002u, 0x00000051u, 0x00000013u,
      0x7fc00002u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000002u, 0x00000007u, 0x7fc00002u,
      0x00000000u, 0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u,
  };

  for (const auto &[name, type] : legacy_mix_modes) {
    ShaderGraph graph;
    const auto geometry = graph.add_node(node_type::geometry, "Backfacing");
    const auto factor = graph.add_node(node_type::math, "Dynamic Factor");
    require(graph.set_property(factor, "Operation",
                               SocketValue::string("MULTIPLY_ADD")) &&
                graph.connect(OutputRef{geometry, "Backfacing"}, factor,
                              "A") &&
                graph.set_input(factor, "B", SocketValue::floating(0.41f)) &&
                graph.set_input(factor, "C", SocketValue::floating(0.23f)),
            "failed to configure dynamic legacy Mix factor");
    const auto mix = graph.add_node(node_type::legacy_mix_color, name);
    const auto use_clamp = type == NODE_MIX_ADD;
    require(graph.set_property(mix, "BlendMode", SocketValue::string(name)) &&
                graph.set_property(mix, "ClampResult",
                                   SocketValue::boolean(use_clamp)) &&
                graph.connect(OutputRef{factor, "Value"}, mix, "Factor") &&
                graph.set_input(
                    mix, "A", SocketValue::color({0.17f, 0.63f, 0.89f})) &&
                graph.set_input(
                    mix, "B", SocketValue::color({0.82f, 0.24f, 0.51f})),
            "failed to configure dynamic legacy Mix node");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(graph.connect(OutputRef{mix, "Color"}, emission, "Color"),
            "failed to connect dynamic legacy Mix output");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = emission, .socket = "Closure"});

    const ShaderCompiler frontend{make_core_node_registry()};
    const auto shader = frontend.compile(graph);
    require(shader.ok(), "dynamic legacy Mix graph did not validate");
    const auto image = compile_shader(*shader.program);
    require(image.valid, image.diagnostic.c_str());
    if (use_clamp) {
      require_words(image.words, clamped_add_oracle,
                    "Psycles clamped legacy Mix differs from Cycles");
    } else {
      auto expected = normal_oracle;
      expected[14] = static_cast<std::uint32_t>(type);
      require_words(image.words, expected,
                    "Psycles legacy Mix differs from Cycles");
    }
    if (image.peak_stack_usage != 5u || !image.node_types_used[NODE_MIX] ||
        image.node_types_used[NODE_MIX_COLOR]) {
      std::cerr << "legacy Mix stack or opcode mask differs from Cycles for "
                << name << ": peak=" << image.peak_stack_usage
                << ", NODE_MIX=" << image.node_types_used[NODE_MIX]
                << ", NODE_MIX_COLOR="
                << image.node_types_used[NODE_MIX_COLOR] << '\n';
      std::exit(1);
    }
  }
}

void test_legacy_mix_constant_modes_match_cycles_5_2_1() {
  // Frozen from the 19 shaders in Cycles 5.2.1
  // `SVM Legacy Mix Constant Matrix`. The inputs are linked from Cycles
  // Value/Color nodes so that Cycles' MixNode::constant_fold, rather than
  // Blender's pre-Cycles node folding, is the oracle. Each row is the exact
  // folded closure weight for Fac=0.37, Color1=(0.17,0.63,0.89), and
  // Color2=(0.82,0.24,0.51).
  static constexpr std::array<std::array<std::uint32_t, 3u>, 19u> oracle{{
      {0x3ed22d0eu, 0x3ef8adacu, 0x3f3fd8aeu},
      {0x3e2e147bu, 0x3ef8adacu, 0x3f3fd8aeu},
      {0x3e227c7cu, 0x3ee7db2bu, 0x3f3a8859u},
      {0x3de2df80u, 0x3ef8731cu, 0x3f5d9aa0u},
      {0x3ed22d0eu, 0x3f2147aeu, 0x3f63d70au},
      {0x3ed7f90eu, 0x3f29b0eeu, 0x3f69275fu},
      {0x3e79e648u, 0x3f30ff53u, 0x3f800000u},
      {0x3ef2617cu, 0x3f380347u, 0x3f8a12d7u},
      {0x3e574d59u, 0x3f0f0e4eu, 0x3f640c63u},
      {0x3e504b5du, 0x3f15cc7eu, 0x3f640684u},
      {0x3ed04817u, 0x3ee00d1cu, 0x3f65bc01u},
      {0x3eb1f8a0u, 0x3f0a8c15u, 0x3f338865u},
      {0x3ebd909fu, 0x3f1b5e96u, 0x3f3e290fu},
      {0xbe089a03u, 0x3f0a8c15u, 0x3f338865u},
      {0x3e3c37fcu, 0x3faf1f8au, 0x3f9a6adeu},
      {0x3edf6fd2u, 0x3eeb6ae8u, 0x3f3f639du},
      {0x3e505d31u, 0x3f246005u, 0x3f63d70au},
      {0x3edf6fd2u, 0x3efc8f42u, 0x3f43f85du},
      {0x3e29039bu, 0x3f1c9629u, 0x3f5d35a8u},
  }};

  auto mismatch_count = std::size_t{};
  for (auto index = std::size_t{}; index < legacy_mix_modes.size(); ++index) {
    const auto &[name, type] = legacy_mix_modes[index];
    ShaderGraph graph;
    const auto mix = graph.add_node(node_type::legacy_mix_color, name);
    require(graph.set_property(mix, "BlendMode", SocketValue::string(name)) &&
                graph.set_property(mix, "ClampResult",
                                   SocketValue::boolean(false)) &&
                graph.set_input(mix, "Factor",
                                SocketValue::floating(0.37f)) &&
                graph.set_input(
                    mix, "A", SocketValue::color({0.17f, 0.63f, 0.89f})) &&
                graph.set_input(
                    mix, "B", SocketValue::color({0.82f, 0.24f, 0.51f})),
            "failed to configure constant legacy Mix mode");
    const auto emission = graph.add_node(node_type::emission, "Emission");
    require(graph.connect(OutputRef{mix, "Color"}, emission, "Color"),
            "failed to connect constant legacy Mix mode");
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = emission, .socket = "Closure"});

    const ShaderCompiler frontend{make_core_node_registry()};
    const auto shader = frontend.compile(graph);
    require(shader.ok(), "constant legacy Mix mode did not validate");
    const auto image = compile_shader(*shader.program);
    require(image.valid, image.diagnostic.c_str());
    const std::array expected{
        0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu,
        0x00000005u, oracle[index][0], oracle[index][1], oracle[index][2],
        0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
        0x00000000u,
    };
    require(image.words.size() == expected.size(),
            "constant legacy Mix mode word count differs from Cycles");
    for (auto word = std::size_t{}; word < expected.size(); ++word) {
      if (image.words[word] != expected[word]) {
        std::cerr << "constant legacy Mix " << name << " differs at word "
                  << word << ": got 0x" << std::hex << image.words[word]
                  << ", expected 0x" << expected[word] << std::dec << '\n';
        ++mismatch_count;
      }
    }
    require(!image.node_types_used[NODE_MIX],
            "constant legacy Mix mode retained NODE_MIX");
    static_cast<void>(type);
  }
  require(mismatch_count == 0u,
          "Psycles constant legacy Mix modes differ from Cycles");
}

void test_legacy_mix_constant_fold_matches_cycles_5_2_1() {
  ShaderGraph graph;
  std::array<std::vector<OutputRef>, 3u> channels;
  for (const auto &[name, type] : legacy_mix_modes) {
    const auto mix = graph.add_node(node_type::legacy_mix_color, name);
    require(graph.set_property(mix, "BlendMode", SocketValue::string(name)) &&
                graph.set_property(mix, "ClampResult",
                                   SocketValue::boolean(false)) &&
                graph.set_input(mix, "Factor", SocketValue::floating(0.37f)) &&
                graph.set_input(
                    mix, "A", SocketValue::color({0.17f, 0.63f, 0.89f})) &&
                graph.set_input(
                    mix, "B", SocketValue::color({0.82f, 0.24f, 0.51f})),
            "failed to configure constant legacy Mix node");
    const auto separate = graph.add_node(node_type::separate_color, name);
    require(graph.set_property(separate, "Mode", SocketValue::string("RGB")) &&
                graph.connect(OutputRef{mix, "Color"}, separate, "Color"),
            "failed to connect constant legacy Mix separation");
    channels[0].push_back(OutputRef{separate, "R"});
    channels[1].push_back(OutputRef{separate, "G"});
    channels[2].push_back(OutputRef{separate, "B"});
    static_cast<void>(type);
  }

  std::array<OutputRef, 3u> averages;
  for (auto channel = std::size_t{}; channel < channels.size(); ++channel) {
    auto value = channels[channel].front();
    for (auto index = std::size_t{1u}; index < channels[channel].size();
         ++index) {
      const auto add = graph.add_node(node_type::math, "Legacy Sum");
      require(graph.set_property(add, "Operation", SocketValue::string("ADD")) &&
                  graph.connect(value, add, "A") &&
                  graph.connect(channels[channel][index], add, "B"),
              "failed to construct legacy Mix sum");
      value = OutputRef{add, "Value"};
    }
    const auto scale = graph.add_node(node_type::math, "Legacy Average");
    require(graph.set_property(scale, "Operation",
                               SocketValue::string("MULTIPLY")) &&
                graph.connect(value, scale, "A") &&
                graph.set_input(scale, "B",
                                SocketValue::floating(1.0f / 19.0f)),
            "failed to construct legacy Mix average");
    averages[channel] = OutputRef{scale, "Value"};
  }

  const auto combine = graph.add_node(node_type::combine_color, "Pack Modes");
  require(graph.set_property(combine, "Mode", SocketValue::string("RGB")) &&
              graph.connect(averages[0], combine, "R") &&
              graph.connect(averages[1], combine, "G") &&
              graph.connect(averages[2], combine, "B"),
          "failed to pack constant legacy Mix modes");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect(OutputRef{combine, "Color"}, emission, "Color"),
          "failed to connect constant legacy Mix result");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "constant legacy Mix graph did not validate");
  const auto image = compile_shader(*shader.program);
  require(image.valid, image.diagnostic.c_str());
  // Frozen from `mix_rgb_legacy_modes` with Value/Color nodes linked into all
  // MixRGB inputs. That forces this fold to occur in Cycles' SVM graph rather
  // than in Blender's pre-Cycles node evaluator.
  static constexpr std::uint32_t expected[] = {
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu, 0x00000005u,
      0x3e8cedbcu, 0x3f1a6314u, 0x3f5b393eu, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u,
  };
  require_words(image.words, expected,
                "Psycles legacy Mix folding differs from Cycles");
  require(!image.node_types_used[NODE_MIX],
          "constant legacy Mix retained NODE_MIX");
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
  test_dynamic_math_and_dedup_match_cycles_5_2_1();
  test_constant_math_fold_matches_cycles_5_2_1();
  test_zero_mix_closure_fold_matches_cycles_5_2_1();
  test_dynamic_color_pipeline_matches_cycles_5_2_1();
  test_color_constant_fold_matches_cycles_5_2_1();
  test_dynamic_combsep_color_matches_cycles_5_2_1();
  test_combsep_color_constant_fold_matches_cycles_5_2_1();
  test_dynamic_legacy_mix_matches_cycles_5_2_1();
  test_legacy_mix_constant_modes_match_cycles_5_2_1();
  test_legacy_mix_constant_fold_matches_cycles_5_2_1();
  test_unsupported_node_rejects_without_old_fallback();
  return 0;
}

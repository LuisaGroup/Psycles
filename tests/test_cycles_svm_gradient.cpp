#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;
using psycles::Vec3f;

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
    std::cerr << label << " word count differs from Cycles 5.2.1: got "
              << actual.size() << ", expected " << expected.size() << '\n';
    for (const auto word : actual) {
      std::cerr << "0x" << std::hex << word << "u, ";
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

[[nodiscard]] ShaderGraph
make_gradient_graph(std::string_view type, bool color,
                    const Vec3f *constant_vector = nullptr) {
  ShaderGraph graph;
  NodeId vector{};
  if (constant_vector != nullptr) {
    vector = graph.add_node(node_type::combine_xyz, "Gradient Constant Vector");
    require(graph.set_input(vector, "X",
                            SocketValue::floating(constant_vector->x)) &&
                graph.set_input(vector, "Y",
                                SocketValue::floating(constant_vector->y)) &&
                graph.set_input(vector, "Z",
                                SocketValue::floating(constant_vector->z)),
            "failed to author Gradient constant vector");
  }
  const auto gradient =
      graph.add_node(node_type::gradient_texture, "Gradient Texture");
  if (constant_vector == nullptr) {
    const auto coordinates = graph.add_node(node_type::texture_coordinate,
                                            "Default Generated Coordinates");
    const auto point_to_vector =
        graph.add_node(node_type::point_to_vector, "Generated Point to Vector");
    require(
        graph.connect({coordinates, "Generated"}, point_to_vector, "Point") &&
            graph.connect({point_to_vector, "Vector"}, gradient, "Vector"),
        "failed to bind Gradient generated coordinates");
  }
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid = graph.set_property(gradient, "GradientType",
                                  SocketValue::string(std::string{type}));
  if (constant_vector != nullptr) {
    valid = valid && graph.connect({vector, "Vector"}, gradient, "Vector");
  }
  if (color) {
    valid = valid && graph.connect({gradient, "Color"}, emission, "Color");
  } else {
    const auto scalar_to_color =
        graph.add_node(node_type::scalar_to_color, "Factor to Color");
    valid = valid &&
            graph.connect({gradient, "Factor"}, scalar_to_color, "Value") &&
            graph.connect({scalar_to_color, "Color"}, emission, "Color");
  }
  require(valid, "failed to construct Cycles Gradient SVM graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_mapped_gradient_graph() {
  ShaderGraph graph;
  const auto vector =
      graph.add_node(node_type::combine_xyz, "Gradient Constant Vector");
  const auto mapping =
      graph.add_node(node_type::mapping, "Gradient Point Mapping");
  const auto gradient =
      graph.add_node(node_type::gradient_texture, "Mapped Diagonal Gradient");
  const auto scalar_to_color =
      graph.add_node(node_type::scalar_to_color, "Factor to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto valid =
      graph.set_input(vector, "X", SocketValue::floating(0.2f)) &&
      graph.set_input(vector, "Y", SocketValue::floating(-0.4f)) &&
      graph.set_input(vector, "Z", SocketValue::floating(0.3f)) &&
      graph.set_property(mapping, "VectorType", SocketValue::string("POINT")) &&
      graph.set_input(mapping, "Location",
                      SocketValue::vector({0.1f, 0.2f, -0.1f})) &&
      graph.set_input(mapping, "Rotation",
                      SocketValue::vector({0.17f, -0.11f, 0.3f})) &&
      graph.set_input(mapping, "Scale",
                      SocketValue::vector({2.0f, 0.5f, 1.3f})) &&
      graph.set_property(gradient, "GradientType",
                         SocketValue::string("DIAGONAL")) &&
      graph.connect({vector, "Vector"}, mapping, "Vector") &&
      graph.connect({mapping, "Vector"}, gradient, "Vector") &&
      graph.connect({gradient, "Factor"}, scalar_to_color, "Value") &&
      graph.connect({scalar_to_color, "Color"}, emission, "Color");
  require(valid, "failed to construct mapped Gradient graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderImage compile_graph(ShaderGraph &graph) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
    std::exit(EXIT_FAILURE);
  }
  AttributeIDMap attributes;
  ImageIDMap images;
  return compile_shader(*shader.program, attributes, images,
                        ShaderCompileContext{.background = false});
}

void test_external_opcode_streams_match_cycles_5_2_1() {
  static constexpr std::array types{"LINEAR",   "QUADRATIC", "EASING",
                                    "DIAGONAL", "RADIAL",    "QUADRATIC_SPHERE",
                                    "SPHERICAL"};
  static constexpr std::array enum_values{
      NODE_BLEND_LINEAR,   NODE_BLEND_QUADRATIC, NODE_BLEND_EASING,
      NODE_BLEND_DIAGONAL, NODE_BLEND_RADIAL,    NODE_BLEND_QUADRATIC_SPHERE,
      NODE_BLEND_SPHERICAL};
  for (auto index = std::size_t{}; index < types.size(); ++index) {
    auto graph = make_gradient_graph(types[index], false);
    const auto image = compile_graph(graph);
    require(image.valid, image.diagnostic.c_str());
    const std::array expected{
        0x00000001u, 0x00000004u,
        0x00000016u, 0x00000017u,
        0x00000015u, 0x0000000bu,
        0x00000000u, 0x00000000u,
        0x00000040u, static_cast<std::uint32_t>(enum_values[index]),
        0x00ff0300u, 0x0000000du,
        0x00000000u, 0x00000003u,
        0x00000007u, 0x7fc00000u,
        0x00000000u, 0x00000000u,
        0x3f800000u, 0x00000003u,
        0x000000ffu, 0x00000000u,
        0x00000000u, 0x00000000u};
    require_words(std::span{image.words}, expected,
                  "Gradient Factor opcode stream");
  }

  auto color_graph = make_gradient_graph("SPHERICAL", true);
  const auto color_image = compile_graph(color_graph);
  require(color_image.valid, color_image.diagnostic.c_str());
  static constexpr std::array color_expected{
      0x00000001u, 0x00000004u, 0x00000013u, 0x00000014u, 0x00000015u,
      0x0000000bu, 0x00000000u, 0x00000000u, 0x00000040u, 0x00000006u,
      0x0003ff00u, 0x00000007u, 0x7fc00003u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
      0x00000000u};
  require_words(std::span{color_image.words}, color_expected,
                "Gradient Color opcode stream");
}

void test_external_constant_fold_matrix_matches_cycles_5_2_1() {
  struct Case {
    const char *type;
    Vec3f vector;
    std::uint32_t result_bits;
  };
  static constexpr std::array cases{
      Case{"LINEAR", {-2.0f, 0.0f, 0.0f}, 0x00000000u},
      Case{"LINEAR", {0.37f, 2.0f, -1.0f}, 0x3ebd70a4u},
      Case{"LINEAR", {2.0f, 0.0f, 0.0f}, 0x3f800000u},
      Case{"QUADRATIC", {-1.0f, 0.0f, 0.0f}, 0x00000000u},
      Case{"QUADRATIC", {0.5f, 0.0f, 0.0f}, 0x3e800000u},
      Case{"EASING", {-0.5f, 0.0f, 0.0f}, 0x00000000u},
      Case{"EASING", {0.3f, 0.0f, 0.0f}, 0x3e5d2f1bu},
      Case{"EASING", {1.5f, 0.0f, 0.0f}, 0x3f800000u},
      Case{"DIAGONAL", {-1.0f, 0.2f, 0.0f}, 0x00000000u},
      Case{"DIAGONAL", {1.4f, 0.8f, 0.0f}, 0x3f800000u},
      Case{"RADIAL", {1.0f, 0.0f, 0.0f}, 0x3f000000u},
      Case{"RADIAL", {0.0f, -1.0f, 0.0f}, 0x3e800000u},
      Case{"SPHERICAL", {0.0f, 0.0f, 0.0f}, 0x3f7fffefu},
      Case{"SPHERICAL", {1.0f, 0.0f, 0.0f}, 0x00000000u},
      Case{"QUADRATIC_SPHERE", {0.0f, 0.0f, 0.0f}, 0x3f7fffdeu},
      Case{"QUADRATIC_SPHERE", {0.5f, 0.5f, 0.5f}, 0x3c9309a0u}};
  for (const auto &test : cases) {
    auto graph = make_gradient_graph(test.type, false, &test.vector);
    const auto image = compile_graph(graph);
    require(image.valid, image.diagnostic.c_str());
    if (test.result_bits == 0u) {
      static constexpr std::array expected{
          0x00000001u, 0x00000004u, 0x00000005u, 0x00000006u,
          0x00000000u, 0x00000000u, 0x00000000u};
      require_words(std::span{image.words}, expected,
                    "Gradient zero constant fold");
    } else {
      const std::array expected{
          0x00000001u, 0x00000004u,      0x0000000bu,      0x0000000cu,
          0x00000005u, test.result_bits, test.result_bits, test.result_bits,
          0x00000003u, 0x000000ffu,      0x00000000u,      0x00000000u,
          0x00000000u};
      require_words(std::span{image.words}, expected,
                    "Gradient nonzero constant fold");
    }
  }

  const auto color_value = Vec3f{0.3f, 0.0f, 0.0f};
  auto color_graph = make_gradient_graph("EASING", true, &color_value);
  const auto color_image = compile_graph(color_graph);
  require(color_image.valid, color_image.diagnostic.c_str());
  static constexpr std::array color_expected{
      0x00000001u, 0x00000004u, 0x0000000bu, 0x0000000cu, 0x00000005u,
      0x3e5d2f1bu, 0x3e5d2f1bu, 0x3e5d2f1bu, 0x00000003u, 0x000000ffu,
      0x00000000u, 0x00000000u, 0x00000000u};
  require_words(std::span{color_image.words}, color_expected,
                "Gradient Color constant fold");
}

void test_cycles_fold_does_not_reenter_blender_inline_stage() {
  auto graph = make_mapped_gradient_graph();
  const auto image = compile_graph(graph);
  require(image.valid, image.diagnostic.c_str());

  // Complete shader-local stream from the external
  // gradient_mapping_constant_fold probe. Blender first keeps Mapping and
  // Gradient because Mapping has no multi-function. Cycles then folds Mapping
  // to NODE_VALUE_V, but must retain NODE_TEX_GRADIENT: its graph fold cannot
  // travel backwards into the already-completed Blender inline stage.
  static constexpr std::array expected{
      0x00000001u, 0x00000004u, 0x00000017u, 0x00000018u, 0x00000013u,
      0x00000000u, 0x3f0553fbu, 0x3d605b04u, 0x3e95acd8u, 0x00000040u,
      0x00000003u, 0x00ff0300u, 0x0000000du, 0x00000000u, 0x00000003u,
      0x00000007u, 0x7fc00000u, 0x00000000u, 0x00000000u, 0x3f800000u,
      0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u, 0x00000000u};
  require_words(std::span{image.words}, expected,
                "Mapping to Gradient fold-stage boundary");
}

void test_schema_and_invalid_type() {
  const auto registry = make_core_node_registry();
  const auto *schema = registry.find(node_type::gradient_texture);
  require(schema != nullptr && schema->outputs.size() == 2u &&
              schema->outputs[0].name == "Color" &&
              schema->outputs[0].type == SocketType::color &&
              schema->outputs[1].name == "Factor" &&
              schema->outputs[1].type == SocketType::floating,
          "Gradient schema does not preserve Cycles Color and Factor outputs");

  auto graph = make_gradient_graph("NOT_A_GRADIENT", true);
  const auto image = compile_graph(graph);
  require(!image.valid, "invalid Gradient type was accepted");
}

} // namespace

int main() {
  test_external_opcode_streams_match_cycles_5_2_1();
  test_external_constant_fold_matrix_matches_cycles_5_2_1();
  test_cycles_fold_does_not_reenter_blender_inline_stage();
  test_schema_and_invalid_type();
  return EXIT_SUCCESS;
}

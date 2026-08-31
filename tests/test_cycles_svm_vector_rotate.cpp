#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_test_compile.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>

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

struct RotateCase {
  std::string_view type;
  bool invert;
  psycles::Vec3f axis;
  psycles::Vec3f rotation;
  float angle;
};

static constexpr std::array rotate_cases{
    RotateCase{"AXIS_ANGLE", false, {0.29f, 0.73f, -0.41f}, {}, 0.71f},
    RotateCase{"AXIS_ANGLE", true, {0.29f, 0.73f, -0.41f}, {}, 0.71f},
    RotateCase{"X_AXIS", false, {0.0f, 0.0f, 1.0f}, {}, 0.71f},
    RotateCase{"X_AXIS", true, {0.0f, 0.0f, 1.0f}, {}, 0.71f},
    RotateCase{"Y_AXIS", false, {0.0f, 0.0f, 1.0f}, {}, 0.71f},
    RotateCase{"Y_AXIS", true, {0.0f, 0.0f, 1.0f}, {}, 0.71f},
    RotateCase{"Z_AXIS", false, {0.0f, 0.0f, 1.0f}, {}, 0.71f},
    RotateCase{"Z_AXIS", true, {0.0f, 0.0f, 1.0f}, {}, 0.71f},
    RotateCase{"EULER_XYZ", false, {0.0f, 0.0f, 1.0f},
               {0.31f, -0.52f, 0.27f}, 0.0f},
    RotateCase{"EULER_XYZ", true, {0.0f, 0.0f, 1.0f},
               {0.31f, -0.52f, 0.27f}, 0.0f},
    RotateCase{"AXIS_ANGLE", false, {}, {}, 0.71f},
    RotateCase{"AXIS_ANGLE", true, {}, {}, 0.71f},
};

static constexpr std::array<std::array<std::uint32_t, 33u>, 12u>
    cycles_5_2_1_oracles{{
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000000u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x3e947ae1u,
         0x3f3ae148u, 0xbed1eb85u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000300u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000000u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x3e947ae1u,
         0x3f3ae148u, 0xbed1eb85u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000301u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000001u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x3f800000u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000300u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000001u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x3f800000u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000301u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000002u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x3f800000u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000300u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000002u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x3f800000u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000301u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000003u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x3f800000u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000300u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000003u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x3f800000u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000301u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000004u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x3f800000u, 0x3e9eb852u, 0xbf051eb8u,
         0x3e8a3d71u, 0x00000000u, 0x00000300u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000004u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x3f800000u, 0x3e9eb852u, 0xbf051eb8u,
         0x3e8a3d71u, 0x00000000u, 0x00000301u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000000u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000300u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
        {0x00000001u, 0x00000004u, 0x0000001fu, 0x00000020u,
         0x0000000bu, 0x00000001u, 0x00000000u, 0x00000058u,
         0x00000000u, 0x7fc00000u, 0x00000000u, 0x00000000u,
         0x3e2e147bu, 0xbe6b851fu, 0x3e9eb852u, 0x00000000u,
         0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
         0x00000000u, 0x3f35c28fu, 0x00000301u, 0x00000007u,
         0x7fc00003u, 0x00000000u, 0x00000000u, 0x3f800000u,
         0x00000003u, 0x000000ffu, 0x00000000u, 0x00000000u,
         0x00000000u},
    }};

[[nodiscard]] ShaderImage compile_case(const RotateCase &item) {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto rotate = graph.add_node(node_type::vector_rotate, "Vector Rotate");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(
      graph.connect({geometry, "Normal"}, to_vector, "Normal") &&
          graph.connect({to_vector, "Vector"}, rotate, "Vector") &&
          graph.set_input(rotate, "Center",
                          SocketValue::point({0.17f, -0.23f, 0.31f})) &&
          graph.set_input(rotate, "Axis", SocketValue::vector(item.axis)) &&
          graph.set_input(rotate, "Rotation",
                          SocketValue::point(item.rotation)) &&
          graph.set_input(rotate, "Angle", SocketValue::floating(item.angle)) &&
          graph.set_property(rotate, "Type",
                             SocketValue::string(std::string{item.type})) &&
          graph.set_property(rotate, "Invert",
                             SocketValue::boolean(item.invert)) &&
          graph.connect({rotate, "Vector"}, to_color, "Vector") &&
          graph.connect({to_color, "Color"}, emission, "Color"),
      "failed to construct Vector Rotate graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "Vector Rotate graph did not validate");
  return compile_shader(*shader.program);
}

void test_vector_rotate_matrix() {
  for (auto index = std::size_t{}; index < rotate_cases.size(); ++index) {
    const auto image = compile_case(rotate_cases[index]);
    require(image.valid, "Vector Rotate graph did not compile to SVM");
    require_words(image.words, cycles_5_2_1_oracles[index],
                  rotate_cases[index].type);
    require(image.peak_stack_usage == 6u &&
                image.node_types_used[NODE_VECTOR_ROTATE],
            "Vector Rotate opcode or stack lifetime differs from Cycles");
  }
}

void test_invalid_vector_rotate_type_rejected() {
  auto invalid = rotate_cases.front();
  invalid.type = "NOT_A_CYCLES_VECTOR_ROTATE_TYPE";
  const auto image = compile_case(invalid);
  require(!image.valid, "invalid Vector Rotate type silently selected a mode");
}

} // namespace

int main() {
  test_vector_rotate_matrix();
  test_invalid_vector_rotate_type_rejected();
  return EXIT_SUCCESS;
}

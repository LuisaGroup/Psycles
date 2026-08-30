#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>

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

struct TransformCase {
  std::string_view type;
  std::uint32_t type_value;
  std::string_view convert_from;
  std::uint32_t convert_from_value;
  std::string_view convert_to;
  std::uint32_t convert_to_value;
};

static constexpr std::array types{
    std::pair{"VECTOR", 0u},
    std::pair{"POINT", 1u},
    std::pair{"NORMAL", 2u},
};

static constexpr std::array spaces{
    std::pair{"WORLD", 0u},
    std::pair{"OBJECT", 1u},
    std::pair{"CAMERA", 2u},
};

[[nodiscard]] constexpr auto transform_cases() {
  std::array<TransformCase, 27u> result{};
  auto index = std::size_t{};
  for (const auto &[type, type_value] : types) {
    for (const auto &[convert_from, convert_from_value] : spaces) {
      for (const auto &[convert_to, convert_to_value] : spaces) {
        result[index++] = TransformCase{type,         type_value,
                                        convert_from, convert_from_value,
                                        convert_to,   convert_to_value};
      }
    }
  }
  return result;
}

static constexpr auto cases = transform_cases();

[[nodiscard]] ShaderImage compile_case(const TransformCase &item,
                                       psycles::Vec3f vector = {0.37f, -0.21f,
                                                                0.63f}) {
  ShaderGraph graph;
  const auto transform =
      graph.add_node(node_type::vector_transform, "Vector Transform");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_input(transform, "Vector", SocketValue::vector(vector)) &&
              graph.set_property(transform, "Type",
                                 SocketValue::string(std::string{item.type})) &&
              graph.set_property(
                  transform, "Convert From",
                  SocketValue::string(std::string{item.convert_from})) &&
              graph.set_property(
                  transform, "Convert To",
                  SocketValue::string(std::string{item.convert_to})) &&
              graph.connect({transform, "Vector"}, to_color, "Vector") &&
              graph.connect({to_color, "Color"}, emission, "Color"),
          "failed to construct Vector Transform graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  require(shader.ok(), "Vector Transform graph did not validate");
  return compile_shader(*shader.program);
}

[[nodiscard]] constexpr std::array<std::uint32_t, 22u>
cycles_5_2_1_oracle(const TransformCase &item) {
  return {
      0x00000001u,
      0x00000004u,
      0x00000014u,
      0x00000015u,
      0x00000059u,
      item.type_value,
      item.convert_from_value,
      item.convert_to_value,
      0x3ebd70a4u,
      0xbe570a3du,
      0x3f2147aeu,
      0x00000000u,
      0x00000007u,
      0x7fc00000u,
      0x00000000u,
      0x00000000u,
      0x3f800000u,
      0x00000003u,
      0x000000ffu,
      0x00000000u,
      0x00000000u,
      0x00000000u,
  };
}

void test_vector_transform_matrix() {
  for (const auto &item : cases) {
    const auto image = compile_case(item);
    require(image.valid, "Vector Transform graph did not compile to SVM");
    const auto expected = cycles_5_2_1_oracle(item);
    require_words(image.words, expected, item.type);
    require(image.peak_stack_usage == 3u &&
                image.node_types_used[NODE_VECTOR_TRANSFORM],
            "Vector Transform opcode or stack lifetime differs from Cycles");
  }
}

void test_world_to_object_zero_normal() {
  static constexpr TransformCase item{"NORMAL", 2u, "WORLD", 0u, "OBJECT", 1u};
  static constexpr std::array<std::uint32_t, 22u> expected{
      0x00000001u, 0x00000004u, 0x00000014u, 0x00000015u, 0x00000059u,
      0x00000002u, 0x00000000u, 0x00000001u, 0x00000000u, 0x00000000u,
      0x00000000u, 0x00000000u, 0x00000007u, 0x7fc00000u, 0x00000000u,
      0x00000000u, 0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u,
      0x00000000u, 0x00000000u};
  const auto image = compile_case(item, {});
  require(image.valid,
          "zero-normal Vector Transform graph did not compile to SVM");
  require_words(image.words, expected,
                "WORLD to OBJECT zero-normal Cycles 5.2.1 oracle");
}

void test_invalid_vector_transform_properties_rejected() {
  auto invalid = cases.front();
  invalid.type = "NOT_A_CYCLES_VECTOR_TRANSFORM_TYPE";
  require(!compile_case(invalid).valid,
          "invalid Vector Transform type silently selected a mode");

  invalid = cases.front();
  invalid.convert_from = "NOT_A_CYCLES_VECTOR_SPACE";
  require(!compile_case(invalid).valid,
          "invalid Vector Transform source silently selected a space");

  invalid = cases.front();
  invalid.convert_to = "NOT_A_CYCLES_VECTOR_SPACE";
  require(!compile_case(invalid).valid,
          "invalid Vector Transform destination silently selected a space");
}

} // namespace

int main() {
  test_vector_transform_matrix();
  test_world_to_object_zero_normal();
  test_invalid_vector_transform_properties_rejected();
  return EXIT_SUCCESS;
}

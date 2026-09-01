#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_graph.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

constexpr std::string_view type_c = R"IES(IESNA:LM-63-2002
[TEST] PSYCLES TYPE C
TILT=NONE
1 1000 1 5 5 1 1 0 0 0 1 1 100
0 20 75 120 180
0 45 130 250 360
1 2 5 9 12
2 4 8 13 17
4 7 11 16 22
3 6 10 15 20
1 2 5 9 12
)IES";

constexpr std::string_view type_b = R"IES(IESNA:LM-63-2002
[TEST] PSYCLES TYPE B
TILT=NONE
1 1000 0.75 4 3 2 1 0 0 0 1 1 100
0 30 60 90
0 45 90
2 3 5 8
4 7 11 16
6 10 15 21
)IES";

constexpr std::string_view type_a = R"IES(IESNA:LM-63-2002
[TEST] PSYCLES TYPE A
TILT=NONE
1 1000 0.5 4 3 3 1 0 0 0 1 1 100
-90 -30 30 90
0 45 90
3 5 8 13
4 7 11 17
6 10 15 22
)IES";

// Exact packed profile slices dumped from Cycles 5.2.1 commit 9e2066aef.
constexpr auto type_c_oracle = std::array{
    0x00000005u, 0x00000005u, 0x00000000u, 0x3f490fdbu,
    0x4011361eu, 0x408ba058u, 0x40c90fdbu, 0x00000000u,
    0x3eb2b8c2u, 0x3fa78d36u, 0x40060a92u, 0x40490fdbu,
    0x3d90b8dau, 0x3e10b8dau, 0x3eb4e711u, 0x3f22cff5u,
    0x3f591547u, 0x3e10b8dau, 0x3e90b8dau, 0x3f10b8dau,
    0x3f6b2c62u, 0x3f99c468u, 0x3e90b8dau, 0x3efd437eu,
    0x3f46fe2cu, 0x3f90b8dau, 0x3fc6fe2cu, 0x3e591547u,
    0x3ed91547u, 0x3f34e711u, 0x3f87ad4cu, 0x3fb4e711u,
    0x3d90b8dau, 0x3e10b8dau, 0x3eb4e711u, 0x3f22cff5u,
    0x3f591547u};

constexpr auto type_a_oracle = std::array{
    0x00000005u, 0x00000004u, 0x3fc90fdbu, 0x4016cbe4u,
    0x40490fdbu, 0x407b53d1u, 0x4096cbe4u, 0x00000000u,
    0x3f860a92u, 0x40060a92u, 0x40490fdbu, 0x3e591547u,
    0x3eb4e711u, 0x3f07ad4cu, 0x3f46fe2cu, 0x3e10b8dau,
    0x3e7d437eu, 0x3ec6fe2cu, 0x3f19c468u, 0x3dd91547u,
    0x3e34e711u, 0x3e90b8dau, 0x3eeb2c62u, 0x3e10b8dau,
    0x3e7d437eu, 0x3ec6fe2cu, 0x3f19c468u, 0x3e591547u,
    0x3eb4e711u, 0x3f07ad4cu, 0x3f46fe2cu};

constexpr auto type_b_oracle = std::array{
    0x00000007u, 0x00000005u, 0x00000000u, 0x3f060a92u,
    0x3f860a92u, 0x3fc90fdbu, 0x40060a92u, 0x40278d36u,
    0x40490fdbu, 0x00000000u, 0x3f490fdbu, 0x3fc90fdbu,
    0x4016cbe4u, 0x40490fdbu, 0x3f8e75f7u, 0x3f591547u,
    0x3ed91547u, 0x3f591547u, 0x3f8e75f7u, 0x3f4b83f3u,
    0x3f153ea1u, 0x3e87ad4cu, 0x3f153ea1u, 0x3f4b83f3u,
    0x3f07ad4cu, 0x3ebdf29eu, 0x3e22cff5u, 0x3ebdf29eu,
    0x3f07ad4cu, 0x3ea2cff5u, 0x3e591547u, 0x3dd91547u,
    0x3e591547u, 0x3ea2cff5u, 0x3f07ad4cu, 0x3ebdf29eu,
    0x3e22cff5u, 0x3ebdf29eu, 0x3f07ad4cu, 0x3f4b83f3u,
    0x3f153ea1u, 0x3e87ad4cu, 0x3f153ea1u, 0x3f4b83f3u,
    0x3f8e75f7u, 0x3f591547u, 0x3ed91547u, 0x3f591547u,
    0x3f8e75f7u};

using Record = std::array<std::uint32_t, 4u>;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] std::vector<std::uint32_t>
bits(std::span<const float> values) {
  std::vector<std::uint32_t> result;
  result.reserve(values.size());
  for (const auto value : values) {
    result.emplace_back(std::bit_cast<std::uint32_t>(value));
  }
  return result;
}

void require_slice(std::span<const std::uint32_t> actual,
                   std::span<const std::uint32_t> expected,
                   std::string_view label) {
  require(actual.size() == expected.size(),
          std::string{label} + " packed size differs from Cycles");
  for (auto index = std::size_t{}; index < expected.size(); ++index) {
    if (actual[index] != expected[index]) {
      std::cerr << label << " packed word " << index << " is 0x"
                << std::hex << actual[index] << ", expected 0x"
                << expected[index] << std::dec << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
}

[[nodiscard]] ShaderProgram make_program(std::string_view content,
                                         bool stack_strength) {
  ShaderGraph graph;
  const auto ies = graph.add_node(node_type::ies_light, "IES Light");
  require(graph.set_property(ies, "IES",
                             SocketValue::string(std::string{content})) &&
              graph.set_input(ies, "Strength",
                              SocketValue::floating(0.4f)),
          "failed to initialize IES graph");
  if (stack_strength) {
    const auto path = graph.add_node(node_type::light_path, "Light Path");
    require(graph.connect({path, "RayLength"}, ies, "Strength"),
            "failed to connect stack-backed IES Strength");
  }
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({ies, "Factor"}, emission, "Strength"),
          "failed to construct IES graph");
  graph.set_root(
      ShaderDomain::surface,
      OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "IES graph failed frontend validation");
  return *shader.program;
}

[[nodiscard]] ShaderProgram make_point_link_program(std::string_view content) {
  ShaderGraph graph;
  const auto coordinates =
      graph.add_node(node_type::texture_coordinate, "Texture Coordinate");
  const auto convert =
      graph.add_node(node_type::point_to_vector, "Point to Vector");
  const auto ies = graph.add_node(node_type::ies_light, "IES Light");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({coordinates, "Generated"}, convert, "Point") &&
              graph.connect({convert, "Vector"}, ies, "Vector") &&
              graph.connect({ies, "Factor"}, emission, "Strength") &&
              graph.set_property(ies, "IES",
                                 SocketValue::string(std::string{content})),
          "failed to construct explicit point-linked IES graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "point-linked IES graph failed frontend validation");
  return *shader.program;
}

[[nodiscard]] ShaderProgram make_normal_link_program(
    std::string_view content) {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto convert =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto ies = graph.add_node(node_type::ies_light, "IES Light");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({geometry, "Normal"}, convert, "Normal") &&
              graph.connect({convert, "Vector"}, ies, "Vector") &&
              graph.connect({ies, "Factor"}, emission, "Strength") &&
              graph.set_property(ies, "IES",
                                 SocketValue::string(std::string{content})),
          "failed to construct explicit normal-linked IES graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "normal-linked IES graph failed frontend validation");
  return *shader.program;
}

[[nodiscard]] const GraphNode *find_node(const CyclesGraph &graph,
                                         std::string_view type) {
  const auto iter = std::find_if(
      graph.nodes().begin(), graph.nodes().end(), [&](const auto &node) {
        return node->type == type;
      });
  return iter == graph.nodes().end() ? nullptr : iter->get();
}

void require_record(const ShaderImage &image, const Record &expected,
                    std::string_view label) {
  if (std::search(image.words.begin(), image.words.end(), expected.begin(),
                  expected.end()) != image.words.end()) {
    return;
  }
  Record actual{};
  auto found_opcode = false;
  for (auto index = std::size_t{}; index + actual.size() <= image.words.size();
       ++index) {
    if (image.words[index] == static_cast<std::uint32_t>(NODE_IES)) {
      std::copy_n(image.words.begin() + static_cast<std::ptrdiff_t>(index),
                  actual.size(), actual.begin());
      found_opcode = true;
      break;
    }
  }
  std::cerr << label << " differs from external Cycles oracle:\n  actual";
  if (found_opcode) {
    for (const auto word : actual) {
      std::cerr << " 0x" << std::hex << word;
    }
  } else {
    std::cerr << " <no NODE_IES candidate>";
  }
  std::cerr << "\n  expected";
  for (const auto word : expected) {
    std::cerr << " 0x" << std::hex << word;
  }
  std::cerr << std::dec << '\n';
  std::exit(EXIT_FAILURE);
}

void test_schema_and_external_records() {
  const auto registry = make_core_node_registry();
  const auto *schema = registry.find(node_type::ies_light);
  require(schema != nullptr && schema->inputs.size() == 2u &&
              schema->outputs.size() == 1u &&
              schema->properties.size() == 1u &&
              schema->inputs[0u].name == "Strength" &&
              schema->inputs[1u].name == "Vector" &&
              schema->outputs[0u].name == "Factor" &&
              schema->properties[0u].name == "IES",
          "IES typed schema differs from Cycles");
  const auto node = make_graph_node(node_type::ies_light);
  require(node != nullptr && node->shader_node_type() == NODE_IES,
          "IES host node is not the Cycles opcode node");

  const auto immediate_program = make_program(type_c, false);
  const auto projected = CyclesGraph::project(immediate_program);
  require(projected.valid(), "unlinked IES graph projection failed");
  const auto *projected_ies = find_node(projected, node_type::ies_light);
  const auto *vector = projected_ies == nullptr
                           ? nullptr
                           : projected_ies->input("Vector");
  const auto *autoconvert =
      vector != nullptr && vector->link != nullptr
          ? vector->link->parent
          : nullptr;
  require(vector != nullptr && vector->type == GraphSocketType::point &&
              vector->value &&
              vector->value->type == SocketType::point &&
              autoconvert != nullptr &&
              autoconvert->type == cycles_synthetic_float3_autoconvert &&
              autoconvert->special_type ==
                  GraphNodeSpecialType::autoconvert &&
              autoconvert->shader_node_type() == NODE_CONVERT &&
              autoconvert->inputs.size() == 1u &&
              autoconvert->inputs[0u].type == GraphSocketType::vector &&
              autoconvert->outputs.size() == 1u &&
              autoconvert->outputs[0u].type == GraphSocketType::point,
          "IES POINT socket or Cycles float3 autoconvert is not isomorphic");
  const auto *incoming_transform =
      autoconvert->inputs[0u].link == nullptr
          ? nullptr
          : autoconvert->inputs[0u].link->parent;
  require(incoming_transform != nullptr &&
              incoming_transform->type == node_type::vector_transform &&
              incoming_transform->input("Vector") != nullptr &&
              incoming_transform->input("Vector")->link != nullptr &&
              incoming_transform->input("Vector")->link->parent->type ==
                  node_type::geometry,
          "IES default Incoming transform topology differs from Cycles");

  const auto point_program = make_point_link_program(type_c);
  const auto point_projected = CyclesGraph::project(point_program);
  const auto *point_ies = find_node(point_projected, node_type::ies_light);
  const auto *point_vector =
      point_ies == nullptr ? nullptr : point_ies->input("Vector");
  require(point_projected.valid() && point_vector != nullptr &&
              point_vector->type == GraphSocketType::point &&
              point_vector->link != nullptr &&
              point_vector->link->parent->type ==
                  node_type::texture_coordinate &&
              find_node(point_projected, node_type::point_to_vector) ==
                  nullptr &&
              find_node(point_projected,
                        cycles_synthetic_float3_autoconvert) == nullptr,
          "inverse POINT/VECTOR conversions were not removed like Cycles");

  const auto normal_program = make_normal_link_program(type_c);
  const auto normal_projected = CyclesGraph::project(normal_program);
  const auto *normal_ies = find_node(normal_projected, node_type::ies_light);
  const auto *normal_vector =
      normal_ies == nullptr ? nullptr : normal_ies->input("Vector");
  const auto *normal_convert =
      normal_vector != nullptr && normal_vector->link != nullptr
          ? normal_vector->link->parent
          : nullptr;
  require(normal_projected.valid() && normal_convert != nullptr &&
              normal_convert->type ==
                  cycles_synthetic_float3_autoconvert &&
              normal_convert->inputs.size() == 1u &&
              normal_convert->inputs[0u].type == GraphSocketType::normal &&
              normal_convert->inputs[0u].link != nullptr &&
              normal_convert->inputs[0u].link->parent->type ==
                  node_type::geometry &&
              normal_convert->outputs.size() == 1u &&
              normal_convert->outputs[0u].type == GraphSocketType::point &&
              find_node(normal_projected, node_type::normal_to_vector) ==
                  nullptr,
          "NORMAL/VECTOR/POINT identity chain was not composed formally");

  AttributeIDMap attributes;
  ImageIDMap images;
  IESIDMap profiles;
  auto immediate = compile_shader(
      immediate_program, attributes, images, profiles,
      ShaderCompileContext{.background = false});
  require(immediate.valid && immediate.node_types_used[NODE_IES] &&
              !immediate.node_types_used[NODE_CONVERT],
          "immediate IES graph did not compile");
  require_record(
      immediate,
      Record{0x0000004au, 0x3ecccccd, 0x00000000u, 0x00000003u},
      "immediate NODE_IES record");

  auto stack = compile_shader(
      make_program(type_c, true), attributes, images, profiles,
      ShaderCompileContext{.background = false});
  require(stack.valid, "stack-backed IES graph did not compile");
  require_record(
      stack,
      Record{0x0000004au, 0x7fc00000u, 0x00000000u, 0x00000103u},
      "stack-backed NODE_IES record");
  require(profiles.slot_count() == 1u,
          "two shaders with identical IES bytes did not share a slot");
}

void test_external_packed_profiles() {
  IESIDMap profiles;
  require(profiles.get_ies_slot(type_c) == 0u &&
              profiles.get_ies_slot(type_c) == 0u &&
              profiles.get_ies_slot(type_a) == 1u &&
              profiles.get_ies_slot("") == 2u &&
              profiles.get_ies_slot(type_b) == 3u,
          "IES raw-content slot interning is not stable");

  const auto packed_float = profiles.packed_data();
  const auto packed = bits(packed_float);
  constexpr auto table_size = std::size_t{4u};
  constexpr auto type_c_offset = table_size;
  constexpr auto type_a_offset = type_c_offset + type_c_oracle.size();
  constexpr auto type_b_offset = type_a_offset + type_a_oracle.size();
  require(packed.size() == type_b_offset + type_b_oracle.size() &&
              packed[0u] == type_c_offset &&
              packed[1u] == type_a_offset &&
              packed[2u] == 0xffffffffu &&
              packed[3u] == type_b_offset,
          "IES offset table differs from Cycles LightManager layout");
  require_slice(std::span{packed}.subspan(type_c_offset,
                                          type_c_oracle.size()),
                type_c_oracle, "Type C");
  require_slice(std::span{packed}.subspan(type_a_offset,
                                          type_a_oracle.size()),
                type_a_oracle, "Type A");
  require_slice(std::span{packed}.subspan(type_b_offset,
                                          type_b_oracle.size()),
                type_b_oracle, "Type B");

  auto no_newline = std::string{type_c};
  require(!no_newline.empty() && no_newline.back() == '\n',
          "test profile unexpectedly lacks its final newline");
  no_newline.pop_back();
  require(profiles.get_ies_slot(no_newline) == 4u,
          "IES interning normalized two distinct raw byte strings");
  auto extra_newline = std::string{type_c};
  extra_newline.push_back('\n');
  require(profiles.get_ies_slot(extra_newline) == 5u,
          "IES interning dropped Cycles' internal TextLine terminator");
}

void test_malformed_profile_is_fail_closed() {
  constexpr std::string_view one_vertical = R"IES(IESNA:LM-63-2002
TILT=NONE
1 1000 1 1 1 1 1 0 0 0 1 1 100
0
0
7
)IES";
  constexpr std::string_view nonmonotone_vertical =
      R"IES(IESNA:LM-63-2002
TILT=NONE
1 1000 1 3 2 1 1 0 0 0 1 1 100
0 90 45
0 360
1 2 3
1 2 3
)IES";
  constexpr std::string_view nonfinite_vertical =
      R"IES(IESNA:LM-63-2002
TILT=NONE
1 1000 1 3 2 1 1 0 0 0 1 1 100
0 nan 180
0 360
1 2 3
1 2 3
)IES";
  IESIDMap profiles;
  require(profiles.get_ies_slot(one_vertical) == 0u &&
              profiles.get_ies_slot(nonmonotone_vertical) == 1u &&
              profiles.get_ies_slot(nonfinite_vertical) == 2u,
          "malformed IES profile did not receive a stable invalid slot");
  const auto packed = bits(profiles.packed_data());
  require(packed == std::vector<std::uint32_t>(3u, 0xffffffffu),
          "unsafe IES profile did not become Cycles invalid slot");
}

} // namespace

int main() {
  test_schema_and_external_records();
  test_external_packed_profiles();
  test_malformed_profile_is_fail_closed();
  return EXIT_SUCCESS;
}

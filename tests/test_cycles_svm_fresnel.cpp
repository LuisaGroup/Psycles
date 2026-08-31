#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include "cycles_svm_graph.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;
using psycles::Vec3f;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

void connect_normal(ShaderGraph &graph, NodeId destination,
                    const Vec3f &value) {
  const auto vector = graph.add_node(node_type::combine_xyz, "Linked Normal");
  const auto normal =
      graph.add_node(node_type::vector_to_normal, "Vector to Normal");
  require(graph.set_input(vector, "X", SocketValue::floating(value.x)) &&
              graph.set_input(vector, "Y", SocketValue::floating(value.y)) &&
              graph.set_input(vector, "Z", SocketValue::floating(value.z)) &&
              graph.connect({vector, "Vector"}, normal, "Vector") &&
              graph.connect({normal, "Normal"}, destination, "Normal"),
          "failed to connect an explicit Fresnel-family Normal");
}

[[nodiscard]] ShaderGraph make_fresnel_graph(float ior,
                                             const Vec3f *normal = nullptr) {
  ShaderGraph graph;
  const auto fresnel = graph.add_node(node_type::fresnel, "Fresnel");
  const auto to_color =
      graph.add_node(node_type::scalar_to_color, "Fresnel to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_input(fresnel, "IOR", SocketValue::floating(ior)) &&
              graph.connect({fresnel, "Factor"}, to_color, "Value") &&
              graph.connect({to_color, "Color"}, emission, "Color"),
          "failed to construct Fresnel graph");
  if (normal != nullptr) {
    connect_normal(graph, fresnel, *normal);
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_layer_graph(float blend, bool fresnel_output,
                                           bool facing_output,
                                           const Vec3f *normal = nullptr) {
  ShaderGraph graph;
  const auto layer = graph.add_node(node_type::layer_weight, "Layer Weight");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto valid = graph.set_input(layer, "Blend", SocketValue::floating(blend));
  if (fresnel_output && facing_output) {
    const auto combine =
        graph.add_node(node_type::combine_xyz, "Layer Weight Outputs");
    const auto to_color =
        graph.add_node(node_type::vector_to_color, "Layer Vector to Color");
    valid = valid && graph.connect({layer, "Fresnel"}, combine, "X") &&
            graph.connect({layer, "Facing"}, combine, "Y") &&
            graph.connect({combine, "Vector"}, to_color, "Vector") &&
            graph.connect({to_color, "Color"}, emission, "Color");
  } else {
    const auto to_color =
        graph.add_node(node_type::scalar_to_color, "Layer Weight to Color");
    valid = valid &&
            graph.connect({layer, fresnel_output ? "Fresnel" : "Facing"},
                          to_color, "Value") &&
            graph.connect({to_color, "Color"}, emission, "Color");
  }
  require(valid, "failed to construct Layer Weight graph");
  if (normal != nullptr) {
    connect_normal(graph, layer, *normal);
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] std::shared_ptr<const ShaderProgram>
compile_frontend(ShaderGraph &graph) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "Fresnel-family graph failed frontend validation");
  return shader.program;
}

[[nodiscard]] ShaderImage compile_graph(ShaderGraph &graph) {
  const auto program = compile_frontend(graph);
  AttributeIDMap attributes;
  ImageIDMap images;
  const auto image = compile_shader(*program, attributes, images,
                                    ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);
  return image;
}

[[nodiscard]] std::vector<std::size_t> opcode_offsets(const ShaderImage &image,
                                                      ShaderNodeType opcode) {
  std::vector<std::size_t> result;
  for (auto index = std::size_t{}; index < image.words.size(); ++index) {
    if (image.words[index] == static_cast<std::uint32_t>(opcode)) {
      result.emplace_back(index);
    }
  }
  return result;
}

void test_osl_internal_default_input_boundary() {
  auto unlinked_graph = make_fresnel_graph(1.5f);
  const auto unlinked_program = compile_frontend(unlinked_graph);
  const auto unlinked = CyclesGraph::project(*unlinked_program);
  require(unlinked.valid(), unlinked.diagnostic());

  const GraphNode *fresnel = nullptr;
  for (const auto &node : unlinked.nodes()) {
    if (node->type == node_type::fresnel) {
      fresnel = node.get();
      break;
    }
  }
  const auto *normal = fresnel != nullptr ? fresnel->input("Normal") : nullptr;
  require(normal != nullptr &&
              (normal->flags & graph_socket_link_normal) != 0u &&
              (normal->flags & graph_socket_osl_internal) != 0u &&
              normal->link == nullptr,
          "SVM projection injected Geometry.Normal into OSL_INTERNAL Normal");

  const Vec3f explicit_normal{0.0f, 0.0f, 1.0f};
  auto linked_graph = make_fresnel_graph(1.5f, &explicit_normal);
  const auto linked_program = compile_frontend(linked_graph);
  const auto linked = CyclesGraph::project(*linked_program);
  require(linked.valid(), linked.diagnostic());
  for (const auto &node : linked.nodes()) {
    if (node->type != node_type::fresnel) {
      continue;
    }
    const auto *linked_normal = node->input("Normal");
    require(linked_normal != nullptr && (linked_normal->link != nullptr ||
                                         linked_normal->constant_folded_in),
            "explicit Fresnel Normal was suppressed with OSL_INTERNAL");
    return;
  }
  require(false, "linked Fresnel disappeared during SVM projection");
}

void test_external_fresnel_payloads() {
  auto unlinked_graph = make_fresnel_graph(1.5f);
  const auto unlinked = compile_graph(unlinked_graph);
  auto offsets = opcode_offsets(unlinked, NODE_FRESNEL);
  require(offsets.size() == 1u && offsets[0u] + 2u < unlinked.words.size() &&
              unlinked.words[offsets[0u] + 1u] == 0x3fc00000u &&
              unlinked.words[offsets[0u] + 2u] == 0x000000ffu,
          "unlinked Fresnel payload differs from external Cycles word 149");

  const Vec3f normal{0.0f, 0.0f, 1.0f};
  auto linked_graph = make_fresnel_graph(1.0f, &normal);
  const auto linked = compile_graph(linked_graph);
  offsets = opcode_offsets(linked, NODE_FRESNEL);
  require(offsets.size() == 1u && offsets[0u] + 2u < linked.words.size() &&
              linked.words[offsets[0u] + 1u] == 0x3f800000u &&
              linked.words[offsets[0u] + 2u] == 0x00000300u,
          "linked Fresnel payload differs from external Cycles word 170");
}

void test_external_layer_weight_payloads() {
  auto fresnel_graph = make_layer_graph(-2.0f, true, false);
  const auto fresnel = compile_graph(fresnel_graph);
  auto offsets = opcode_offsets(fresnel, NODE_LAYER_WEIGHT);
  require(offsets.size() == 1u && offsets[0u] + 3u < fresnel.words.size() &&
              fresnel.words[offsets[0u] + 1u] ==
                  static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FRESNEL) &&
              fresnel.words[offsets[0u] + 2u] == 0xc0000000u &&
              fresnel.words[offsets[0u] + 3u] == 0x000000ffu,
          "unlinked Layer Weight Fresnel payload differs from Cycles word 213");

  auto facing_graph = make_layer_graph(-2.0f, false, true);
  const auto facing = compile_graph(facing_graph);
  offsets = opcode_offsets(facing, NODE_LAYER_WEIGHT);
  require(offsets.size() == 1u && offsets[0u] + 3u < facing.words.size() &&
              facing.words[offsets[0u] + 1u] ==
                  static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FACING) &&
              facing.words[offsets[0u] + 2u] == 0xc0000000u &&
              facing.words[offsets[0u] + 3u] == 0x000000ffu,
          "unlinked Layer Weight Facing payload differs from Cycles word 230");

  const Vec3f normal{0.0f, 0.0f, 1.0f};
  auto both_graph = make_layer_graph(-0.5f, true, true, &normal);
  const auto both = compile_graph(both_graph);
  offsets = opcode_offsets(both, NODE_LAYER_WEIGHT);
  require(offsets.size() == 2u && offsets[1u] + 3u < both.words.size() &&
              both.words[offsets[0u] + 1u] ==
                  static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FRESNEL) &&
              both.words[offsets[1u] + 1u] ==
                  static_cast<std::uint32_t>(NODE_LAYER_WEIGHT_FACING) &&
              both.words[offsets[0u] + 2u] == 0xbf000000u &&
              both.words[offsets[1u] + 2u] == 0xbf000000u &&
              (both.words[offsets[0u] + 3u] & 0xffu) == 0u &&
              (both.words[offsets[1u] + 3u] & 0xffu) == 0u,
          "shared Layer Weight did not emit Cycles' ordered two-opcode form");
}

void test_schema() {
  const auto registry = make_core_node_registry();
  const auto *fresnel = registry.find(node_type::fresnel);
  const auto *layer = registry.find(node_type::layer_weight);
  require(fresnel != nullptr && fresnel->inputs.size() == 2u &&
              fresnel->outputs.size() == 1u &&
              fresnel->inputs[0u].name == "IOR" &&
              fresnel->inputs[1u].name == "Normal" &&
              fresnel->outputs[0u].name == "Factor" && layer != nullptr &&
              layer->inputs.size() == 2u && layer->outputs.size() == 2u &&
              layer->outputs[0u].name == "Fresnel" &&
              layer->outputs[1u].name == "Facing",
          "Fresnel-family typed schema changed");
}

} // namespace

int main() {
  test_schema();
  test_osl_internal_default_input_boundary();
  test_external_fresnel_payloads();
  test_external_layer_weight_payloads();
  return EXIT_SUCCESS;
}

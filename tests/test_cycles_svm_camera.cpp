#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
using namespace psycles::contract;

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] ShaderImage compile_camera_graph(bool view_vector, bool zdepth,
                                               bool distance) {
  ShaderGraph graph;
  const auto camera = graph.add_node(node_type::camera_data, "Camera Data");
  const auto combine = graph.add_node(node_type::combine_xyz, "Combine XYZ");
  const auto vector_to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");

  auto valid = true;
  if (view_vector) {
    const auto view_to_color =
        graph.add_node(node_type::vector_to_color, "View Vector to Color");
    const auto separate =
        graph.add_node(node_type::separate_xyz, "Separate View Vector");
    valid = graph.connect({camera, "View Vector"}, view_to_color, "Vector") &&
            graph.connect({view_to_color, "Color"}, separate, "Vector") &&
            graph.connect({separate, "X"}, combine, "X") && valid;
  }
  if (zdepth) {
    valid = graph.connect({camera, "View Z Depth"}, combine, "Y") && valid;
  }
  if (distance) {
    valid = graph.connect({camera, "View Distance"}, combine, "Z") && valid;
  }
  valid = graph.connect({combine, "Vector"}, vector_to_color, "Vector") &&
          graph.connect({vector_to_color, "Color"}, emission, "Color") && valid;
  require(valid, "failed to construct Camera Data graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "Camera Data graph failed frontend validation");
  AttributeIDMap attributes;
  ImageIDMap images;
  return compile_shader(*shader.program, attributes, images,
                        ShaderCompileContext{.background = false});
}

[[nodiscard]] std::uint32_t camera_payload(const ShaderImage &image) {
  const auto opcode = std::find(image.words.begin(), image.words.end(),
                                static_cast<std::uint32_t>(NODE_CAMERA));
  require(opcode != image.words.end() && image.words.end() - opcode >= 2,
          "Camera Data opcode or payload is missing");
  require(image.node_types_used[NODE_CAMERA],
          "Camera Data opcode usage bit is missing");
  return opcode[1u];
}

void test_typed_schema() {
  const auto registry = make_core_node_registry();
  const auto *camera = registry.find(node_type::camera_data);
  require(camera != nullptr && camera->inputs.empty() &&
              camera->outputs.size() == 3u &&
              camera->outputs[0u].name == "View Vector" &&
              camera->outputs[0u].type == SocketType::vector &&
              camera->outputs[1u].name == "View Z Depth" &&
              camera->outputs[1u].type == SocketType::floating &&
              camera->outputs[2u].name == "View Distance" &&
              camera->outputs[2u].type == SocketType::floating &&
              camera->required_features == 0u,
          "Camera Data schema differs from Cycles 5.2.1");
}

void test_cycles_5_2_1_payloads() {
  // External Cycles 5.2.1 oracle camera_data.svm52: the all-live Camera node
  // at global word 89 has vector/z/distance offsets 0/3/4.
  require(camera_payload(compile_camera_graph(true, true, true)) == 0x00040300u,
          "all-live Camera Data payload differs from Cycles 5.2.1");

  // Each output validity mask is independently observable in SVM bytecode.
  require(camera_payload(compile_camera_graph(true, false, false)) ==
              0x00ffff00u,
          "Camera Data View Vector validity mask changed");
  require(camera_payload(compile_camera_graph(false, true, false)) ==
              0x00ff00ffu,
          "Camera Data View Z Depth validity mask changed");
  require(camera_payload(compile_camera_graph(false, false, true)) ==
              0x0000ffffu,
          "Camera Data View Distance validity mask changed");
}

} // namespace

int main() {
  test_typed_schema();
  test_cycles_5_2_1_payloads();
  return EXIT_SUCCESS;
}

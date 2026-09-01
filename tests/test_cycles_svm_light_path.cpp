#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
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

struct LightPathOutput {
  std::string_view projected_name;
  NodeLightPath path_type;
};

constexpr auto outputs = std::array{
    LightPathOutput{"IsCameraRay", NODE_LP_camera},
    LightPathOutput{"IsShadowRay", NODE_LP_shadow},
    LightPathOutput{"IsDiffuseRay", NODE_LP_diffuse},
    LightPathOutput{"IsGlossyRay", NODE_LP_glossy},
    LightPathOutput{"IsSingularRay", NODE_LP_singular},
    LightPathOutput{"IsReflectionRay", NODE_LP_reflection},
    LightPathOutput{"IsTransmissionRay", NODE_LP_transmission},
    LightPathOutput{"IsVolumeScatterRay", NODE_LP_volume_scatter},
    LightPathOutput{"RayLength", NODE_LP_ray_length},
    LightPathOutput{"RayDepth", NODE_LP_ray_depth},
    LightPathOutput{"DiffuseDepth", NODE_LP_ray_diffuse},
    LightPathOutput{"GlossyDepth", NODE_LP_ray_glossy},
    LightPathOutput{"TransparentDepth", NODE_LP_ray_transparent},
    LightPathOutput{"TransmissionDepth", NODE_LP_ray_transmission},
    LightPathOutput{"PortalDepth", NODE_LP_ray_portal},
};

void require(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(EXIT_FAILURE);
  }
}

[[nodiscard]] ShaderGraph make_graph(std::span<const std::size_t> live) {
  ShaderGraph graph;
  const auto light_path = graph.add_node(node_type::light_path, "Light Path");
  const auto combine = graph.add_node(node_type::combine_xyz, "Pack outputs");
  const auto to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(!live.empty() && live.size() <= 3u,
          "Light Path test requested an invalid output set");
  static constexpr auto inputs = std::array{
      std::string_view{"X"}, std::string_view{"Y"}, std::string_view{"Z"}};
  auto valid = true;
  for (auto index = std::size_t{}; index < live.size(); ++index) {
    valid = valid &&
            graph.connect(
                {light_path, std::string{outputs[live[index]].projected_name}},
                combine, std::string{inputs[index]});
  }
  valid = valid && graph.connect({combine, "Vector"}, to_color, "Vector") &&
          graph.connect({to_color, "Color"}, emission, "Color");
  require(valid, "failed to construct Light Path graph");
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
  }
  require(shader.ok(), "Light Path graph failed frontend validation");
  AttributeIDMap attributes;
  ImageIDMap images;
  auto image = compile_shader(*shader.program, attributes, images,
                              ShaderCompileContext{.background = false});
  require(image.valid, image.diagnostic);
  return image;
}

[[nodiscard]] std::vector<NodeLightPath>
light_path_records(const ShaderImage &image) {
  std::vector<NodeLightPath> result;
  for (auto index = std::size_t{}; index + 2u < image.words.size(); ++index) {
    if (image.words[index] != static_cast<std::uint32_t>(NODE_LIGHT_PATH)) {
      continue;
    }
    const auto path_type = image.words[index + 1u];
    const auto packed_output = image.words[index + 2u];
    require(path_type <= static_cast<std::uint32_t>(NODE_LP_ray_portal) &&
                (packed_output & 0xffffff00u) == 0u,
            "Light Path record does not have the Cycles two-word payload");
    result.emplace_back(static_cast<NodeLightPath>(path_type));
    index += 2u;
  }
  return result;
}

void test_schema() {
  const auto registry = make_core_node_registry();
  const auto *schema = registry.find(node_type::light_path);
  require(schema != nullptr && schema->inputs.empty() &&
              schema->outputs.size() == outputs.size(),
          "Light Path typed schema changed");
  for (auto index = std::size_t{}; index < outputs.size(); ++index) {
    require(schema->outputs[index].name == outputs[index].projected_name &&
                schema->outputs[index].type == SocketType::floating,
            "Light Path typed output order changed");
  }
}

void test_each_output_payload() {
  for (auto index = std::size_t{}; index < outputs.size(); ++index) {
    const std::array live{index};
    auto graph = make_graph(live);
    const auto image = compile_graph(graph);
    const auto records = light_path_records(image);
    require(records.size() == 1u &&
                records.front() == outputs[index].path_type &&
                image.node_types_used[NODE_LIGHT_PATH],
            "Light Path output did not emit its exact Cycles path type");
  }
}

void test_live_output_order() {
  // Connect in deliberately non-Cycles order. LightPathNode::compile must
  // still emit in the fixed Cycles socket order.
  static constexpr std::array live{std::size_t{14u}, std::size_t{8u},
                                   std::size_t{0u}};
  auto graph = make_graph(live);
  const auto image = compile_graph(graph);
  const auto records = light_path_records(image);
  static constexpr std::array expected{NODE_LP_camera, NODE_LP_ray_length,
                                       NODE_LP_ray_portal};
  require(std::ranges::equal(records, expected),
          "Light Path records do not follow Cycles' output order");
}

} // namespace

int main() {
  test_schema();
  test_each_output_payload();
  test_live_output_order();
  return EXIT_SUCCESS;
}

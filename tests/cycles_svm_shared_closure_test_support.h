#pragma once

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <array>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace psycles::test_support::shared_closure {

inline constexpr std::array names{"SurfaceAdd", "SurfaceMix", "VolumeAdd",
                                  "VolumeMix"};

inline void require(bool value, const std::string &message) {
  if (!value) {
    throw std::runtime_error{message};
  }
}

// These are the same four source graphs created by the original Blender
// probe, not replacement word streams or a reference shading evaluator.
inline contract::ShaderGraph make_graph(unsigned index) {
  using namespace contract;
  using namespace compiler;
  const bool volume = index >= 2u;
  const bool nested = (index & 1u) != 0u;
  const std::string output = volume ? "Volume" : "Closure";
  ShaderGraph graph;
  NodeId other;
  if (nested) {
    other = graph.add_node(volume ? node_type::volume_scatter
                                  : node_type::diffuse_bsdf,
                           "00 Other");
    require(graph.set_input(other, "Color",
                            SocketValue::color({0.5f, 0.25f, 0.125f})),
            "cannot set other closure color");
    if (volume) {
      require(graph.set_input(other, "Density", SocketValue::floating(0.75f)),
              "cannot set scatter density");
    }
  }
  const auto shared = graph.add_node(volume ? node_type::volume_absorption
                                            : node_type::transparent_bsdf,
                                     "01 Shared");
  require(graph.set_input(shared, "Color",
                          SocketValue::color({0.25f, 0.5f, 0.75f})),
          "cannot set shared closure color");
  if (volume) {
    require(graph.set_input(shared, "Density", SocketValue::floating(2.0f)),
            "cannot set absorption density");
  }
  NodeId root;
  if (nested) {
    const auto light_path =
        graph.add_node(node_type::light_path, "02 Light Path");
    const auto mix_type =
        volume ? node_type::mix_volume : node_type::mix_closure;
    const auto inner = graph.add_node(mix_type, "03 Inner");
    root = graph.add_node(mix_type, "04 Root");
    require(graph.connect({light_path, "IsCameraRay"}, inner, "Factor") &&
                graph.connect({light_path, "IsCameraRay"}, root, "Factor") &&
                graph.connect({other, output}, inner, "A") &&
                graph.connect({shared, output}, inner, "B") &&
                graph.connect({shared, output}, root, "A") &&
                graph.connect({inner, output}, root, "B"),
            "cannot construct nested shared-closure Mix");
  } else {
    root = graph.add_node(
        volume ? node_type::add_volume : node_type::add_closure, "04 Root");
    require(graph.connect({shared, output}, root, "A") &&
                graph.connect({shared, output}, root, "B"),
            "cannot construct shared-closure Add");
  }
  graph.set_root(volume ? ShaderDomain::volume : ShaderDomain::surface,
                 OutputRef{root, output});
  return graph;
}

inline compiler::cycles_svm::ShaderImage compile(unsigned index) {
  using namespace compiler;
  const auto shader =
      ShaderCompiler{make_core_node_registry()}.compile(make_graph(index));
  for (const auto &diagnostic : shader.diagnostics) {
    require(shader.ok(), diagnostic.message);
  }
  require(shader.ok(), "shared-closure graph did not normalize");
  cycles_svm::AttributeIDMap attributes;
  cycles_svm::ImageIDMap images;
  auto result = cycles_svm::compile_shader(*shader.program, attributes, images,
                                           {.background = false});
  require(result.valid, result.diagnostic);
  return result;
}

inline std::array<std::vector<std::uint32_t>, 4u> read_words(const char *path) {
  std::ifstream input{path};
  std::array<std::vector<std::uint32_t>, 4u> result;
  for (auto index = 0u; index < names.size(); ++index) {
    std::string name;
    unsigned count;
    require(bool(input >> name >> std::dec >> count) && name == names[index],
            "invalid original shared-closure word fixture");
    result[index].resize(count);
    for (auto &word : result[index]) {
      require(bool(input >> std::hex >> word), "truncated original word image");
    }
  }
  std::string trailing;
  require(!(input >> trailing), "trailing original word fixture data");
  return result;
}

} // namespace psycles::test_support::shared_closure

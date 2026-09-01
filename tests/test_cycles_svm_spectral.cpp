#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
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

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 2.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] ShaderImage compile_graph(
    ShaderGraph &graph, ShaderColorSpace color_space = {}) {
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    for (const auto &diagnostic : shader.diagnostics) {
      std::cerr << diagnostic.message << '\n';
    }
  }
  require(shader.ok(), "spectral graph failed frontend validation");
  AttributeIDMap attributes;
  ImageIDMap images;
  auto image = compile_shader(
      *shader.program, attributes, images,
      ShaderCompileContext{.background = false, .color_space = color_space});
  require(image.valid, image.diagnostic);
  return image;
}

[[nodiscard]] ShaderGraph dynamic_graph(std::string_view spectral_type,
                                        std::string_view input_name) {
  ShaderGraph graph;
  const auto path = graph.add_node(node_type::light_path, "Light Path");
  const auto spectral =
      graph.add_node(std::string{spectral_type}, "Dynamic Spectral");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.connect({path, "RayDepth"}, spectral,
                        std::string{input_name}) &&
              graph.connect({spectral, "Color"}, emission, "Color"),
          "failed to author dynamic spectral graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

template<std::size_t N>
void require_record(const ShaderImage &image,
                    const std::array<std::uint32_t, N> &expected,
                    std::string_view label) {
  const auto begin = std::search(image.words.begin(), image.words.end(),
                                 expected.begin(), expected.end());
  if (begin != image.words.end()) {
    return;
  }
  std::cerr << label << " differs from the external Cycles 5.2.1 stream\n"
            << "  expected";
  for (const auto word : expected) {
    std::cerr << " 0x" << std::hex << word;
  }
  std::cerr << std::dec << '\n';
  std::exit(EXIT_FAILURE);
}

void test_schema() {
  const auto registry = make_core_node_registry();
  const auto *blackbody = registry.find(node_type::blackbody);
  const auto *wavelength = registry.find(node_type::wavelength);
  require(blackbody != nullptr && blackbody->inputs.size() == 1u &&
              blackbody->inputs.front().name == "Temperature" &&
              blackbody->inputs.front().type == SocketType::floating &&
              blackbody->outputs.size() == 1u &&
              blackbody->outputs.front().name == "Color" &&
              blackbody->outputs.front().type == SocketType::color,
          "Blackbody typed schema changed");
  require(wavelength != nullptr && wavelength->inputs.size() == 1u &&
              wavelength->inputs.front().name == "Wavelength" &&
              wavelength->inputs.front().type == SocketType::floating &&
              wavelength->outputs.size() == 1u &&
              wavelength->outputs.front().name == "Color" &&
              wavelength->outputs.front().type == SocketType::color,
          "Wavelength typed schema changed");
}

void test_dynamic_records() {
  auto blackbody_graph =
      dynamic_graph(node_type::blackbody, "Temperature");
  const auto blackbody = compile_graph(blackbody_graph);
  require(blackbody.node_types_used[NODE_LIGHT_PATH] &&
              blackbody.node_types_used[NODE_BLACKBODY],
          "dynamic Blackbody did not retain its Cycles SVM nodes");

  // Exact shader-local surface routine copied from the pinned Cycles 5.2.1
  // diagnostic stream for Dynamic Blackbody. Only global jump offsets were
  // omitted. Ray Depth occupies stack lane 0 and Color occupies lanes 1..3.
  static constexpr std::array blackbody_surface{
      0x00000032u, 0x0000000au, 0x00000000u,
      0x0000005cu, 0x7fc00000u, 0x00000001u,
      0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u};
  require_record(blackbody, blackbody_surface, "NODE_BLACKBODY surface");

  auto wavelength_graph =
      dynamic_graph(node_type::wavelength, "Wavelength");
  const auto wavelength = compile_graph(wavelength_graph);
  require(wavelength.node_types_used[NODE_LIGHT_PATH] &&
              wavelength.node_types_used[NODE_WAVELENGTH],
          "dynamic Wavelength did not retain its Cycles SVM nodes");
  static constexpr std::array wavelength_surface{
      0x00000032u, 0x0000000au, 0x00000000u,
      0x0000005bu, 0x7fc00000u, 0x00000001u,
      0x00000007u, 0x7fc00001u, 0x00000000u, 0x00000000u,
      0x3f800000u, 0x00000003u, 0x000000ffu, 0x00000000u};
  require_record(wavelength, wavelength_surface, "NODE_WAVELENGTH surface");
}

void test_blackbody_scene_linear_constant_fold() {
  ShaderGraph graph;
  const auto blackbody = graph.add_node(node_type::blackbody, "Blackbody");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_input(blackbody, "Temperature",
                          SocketValue::floating(6500.0f)) &&
              graph.connect({blackbody, "Color"}, emission, "Color"),
          "failed to author constant Blackbody graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  // A cyclic channel projection makes accidental Rec.709 identity folding
  // observable. The input triplet below is from the external Cycles 5.2.1
  // svm_math_blackbody_color_rec709 oracle at 6500 K.
  ShaderColorSpace color_space;
  color_space.rec709_to_r = {0.0f, 1.0f, 0.0f};
  color_space.rec709_to_g = {0.0f, 0.0f, 1.0f};
  color_space.rec709_to_b = {1.0f, 0.0f, 0.0f};
  const auto image = compile_graph(graph, color_space);
  require(!image.node_types_used[NODE_BLACKBODY] &&
              image.node_types_used[NODE_CLOSURE_SET_WEIGHT],
          "constant Blackbody did not follow Cycles host folding");

  for (auto index = std::size_t{};
       index + 1u + sizeof(SVMNodeClosureSetWeight) / sizeof(std::uint32_t) <=
       image.words.size(); ++index) {
    if (image.words[index] != NODE_CLOSURE_SET_WEIGHT) {
      continue;
    }
    SVMNodeClosureSetWeight payload{};
    std::memcpy(&payload, image.words.data() + index + 1u, sizeof(payload));
    require(near(payload.rgb.x, 0.984052539f) &&
                near(payload.rgb.y, 1.03527153f) &&
                near(payload.rgb.z, 1.04255211f),
            "Blackbody constant fold ignored the active scene color space");
    return;
  }
  require(false, "folded Blackbody closure weight is absent");
}

void test_constant_wavelength_is_not_host_folded() {
  ShaderGraph graph;
  const auto wavelength = graph.add_node(node_type::wavelength, "Wavelength");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  require(graph.set_input(wavelength, "Wavelength",
                          SocketValue::floating(500.0f)) &&
              graph.connect({wavelength, "Color"}, emission, "Color"),
          "failed to author constant Wavelength graph");
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  const auto image = compile_graph(graph);
  require(image.node_types_used[NODE_WAVELENGTH],
          "Wavelength acquired a host fold absent from Cycles 5.2.1");
  static constexpr std::array expected{0x0000005bu, 0x43fa0000u,
                                       0x00000000u};
  require_record(image, expected, "constant NODE_WAVELENGTH");
}

} // namespace

int main() {
  test_schema();
  test_dynamic_records();
  test_blackbody_scene_linear_constant_fold();
  test_constant_wavelength_is_not_host_folded();
  return EXIT_SUCCESS;
}

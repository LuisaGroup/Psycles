#include "compact_surface_program_test_support.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/surface_execution_plan.h>

#include "path_tracer_internal.h"
#include "path_tracer_surface_value_family.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace psycles::test_support {
namespace {

using namespace compiler;
using namespace contract;
using namespace luisa_backend::detail;

static_assert(
    surface_value_family_has_direct_evaluator(SurfaceSvmValueOpcode::noise));

struct NoiseFixtureConfiguration {
  ValueOperation operation{};
  std::uint32_t dimensions{};
  NoiseType type{};
  bool normalize{};
};

constexpr std::array noise_configurations{
    NoiseFixtureConfiguration{ValueOperation::noise_factor, 1u,
                              NoiseType::multifractal, false},
    NoiseFixtureConfiguration{ValueOperation::noise_factor, 3u, NoiseType::fbm,
                              false},
    NoiseFixtureConfiguration{ValueOperation::noise_factor, 3u, NoiseType::fbm,
                              true},
    NoiseFixtureConfiguration{ValueOperation::noise_factor, 4u,
                              NoiseType::ridged_multifractal, true},
    NoiseFixtureConfiguration{ValueOperation::noise_color, 2u,
                              NoiseType::hybrid_multifractal, false},
    NoiseFixtureConfiguration{ValueOperation::noise_color, 2u,
                              NoiseType::hybrid_multifractal, true},
    NoiseFixtureConfiguration{ValueOperation::noise_color, 4u,
                              NoiseType::hetero_terrain, true}};

[[nodiscard]] constexpr std::string_view noise_type_name(NoiseType type) {
  switch (type) {
  case NoiseType::multifractal:
    return "MULTIFRACTAL";
  case NoiseType::hybrid_multifractal:
    return "HYBRID_MULTIFRACTAL";
  case NoiseType::ridged_multifractal:
    return "RIDGED_MULTIFRACTAL";
  case NoiseType::hetero_terrain:
    return "HETERO_TERRAIN";
  case NoiseType::fbm:
    return "FBM";
  }
  return {};
}

[[nodiscard]] constexpr std::uint16_t
noise_immediate(const NoiseFixtureConfiguration &configuration) {
  return static_cast<std::uint16_t>(make_surface_value_svm_immediate(
      configuration.operation, configuration.dimensions,
      (configuration.normalize ? std::uint64_t{1u} : std::uint64_t{0u}) |
          (static_cast<std::uint64_t>(configuration.type) << 8u)));
}

[[nodiscard]] ShaderGraph
make_direct_noise_graph(const NoiseFixtureConfiguration &configuration,
                        std::size_t index) {
  ShaderGraph graph;
  const auto geometry =
      graph.add_node(node_type::geometry, "Direct Noise geometry");
  const auto point_to_vector =
      graph.add_node(node_type::point_to_vector, "Direct Noise coordinate");
  const auto vector_to_scalar =
      graph.add_node(node_type::vector_to_scalar, "Direct Noise W coordinate");
  const auto noise =
      graph.add_node(node_type::noise_texture, "Direct Noise texture");
  const auto diffuse =
      graph.add_node(node_type::diffuse_bsdf, "Direct Noise diffuse");

  const auto color_needed =
      configuration.operation == ValueOperation::noise_color;
  const auto fixture_index = static_cast<float>(index);
  auto configured =
      graph.connect({.node = geometry, .socket = "Position"}, point_to_vector,
                    "Point") &&
      graph.connect({.node = point_to_vector, .socket = "Vector"},
                    vector_to_scalar, "Vector") &&
      graph.connect({.node = point_to_vector, .socket = "Vector"}, noise,
                    "Vector") &&
      graph.connect({.node = vector_to_scalar, .socket = "Value"}, noise,
                    "W") &&
      graph.set_input(noise, "Scale",
                      SocketValue::floating(1.37f + 0.11f * fixture_index)) &&
      graph.set_input(noise, "Detail",
                      SocketValue::floating(2.25f + 0.17f * fixture_index)) &&
      graph.set_input(noise, "Roughness",
                      SocketValue::floating(0.43f + 0.01f * fixture_index)) &&
      graph.set_input(noise, "Lacunarity",
                      SocketValue::floating(1.83f + 0.03f * fixture_index)) &&
      graph.set_input(noise, "Offset",
                      SocketValue::floating(0.13f + 0.02f * fixture_index)) &&
      graph.set_input(noise, "Gain",
                      SocketValue::floating(0.79f + 0.01f * fixture_index)) &&
      graph.set_input(noise, "Distortion",
                      SocketValue::floating(0.31f + 0.04f * fixture_index)) &&
      graph.set_property(
          noise, "Dimensions",
          SocketValue::unsigned_integer(configuration.dimensions)) &&
      graph.set_property(noise, "Normalize",
                         SocketValue::boolean(configuration.normalize)) &&
      graph.set_property(noise, "NoiseType",
                         SocketValue::string(std::string{
                             noise_type_name(configuration.type)})) &&
      graph.set_property(noise, "NeedsColor",
                         SocketValue::boolean(color_needed)) &&
      graph.set_input(diffuse, "Color",
                      SocketValue::color({0.27f, 0.51f, 0.73f})) &&
      graph.set_input(diffuse, "Roughness", SocketValue::floating(0.35f));
  configured =
      configured &&
      (color_needed
           ? graph.connect({.node = noise, .socket = "Color"}, diffuse, "Color")
           : graph.connect({.node = noise, .socket = "Factor"}, diffuse,
                           "Roughness"));
  if (!configured) {
    throw std::runtime_error{"failed to configure direct Noise SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});
  return graph;
}

[[nodiscard]] std::vector<std::uint16_t>
expected_noise_immediates(ValueOperation operation) {
  std::vector<std::uint16_t> result;
  for (const auto &configuration : noise_configurations) {
    if (configuration.operation == operation) {
      result.emplace_back(noise_immediate(configuration));
    }
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

} // namespace

std::vector<ShaderGraph> make_direct_noise_graphs() {
  std::vector<ShaderGraph> graphs;
  graphs.reserve(noise_configurations.size());
  for (auto index = std::size_t{}; index < noise_configurations.size();
       ++index) {
    graphs.emplace_back(
        make_direct_noise_graph(noise_configurations[index], index));
  }
  return graphs;
}

std::string
validate_direct_noise_surface_runtime(const SurfaceValueRuntime &runtime) {
  for (const auto operation :
       {ValueOperation::noise_factor, ValueOperation::noise_color}) {
    const auto expected = expected_noise_immediates(operation);
    const auto first = std::find_if(
        runtime.value_variants.begin(), runtime.value_variants.end(),
        [operation](const auto &variant) noexcept {
          return variant.instruction.operation == operation;
        });
    const auto count = std::count_if(
        runtime.value_variants.begin(), runtime.value_variants.end(),
        [operation](const auto &variant) noexcept {
          return variant.instruction.operation == operation;
        });
    if (first == runtime.value_variants.end() || count != 1u ||
        first->operand_types.size() != value_operand::noise::count ||
        first->svm_immediates != expected) {
      std::ostringstream message;
      message << "compact runtime violated the direct Noise evaluator/"
                 "immediate-domain quotient (operation="
              << static_cast<std::uint32_t>(operation) << ", variants=" << count
              << ')';
      return message.str();
    }
  }
  return {};
}

} // namespace psycles::test_support

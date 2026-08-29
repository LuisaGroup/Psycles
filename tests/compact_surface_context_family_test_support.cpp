#include "compact_surface_program_test_support.h"

#include "path_tracer_surface_value_family.h"
#include "path_tracer_surface_values.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <array>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace psycles::test_support {
namespace {

using namespace compiler;
using namespace contract;
using namespace luisa_backend::detail;

constexpr std::array direct_context_families{
    SurfaceSvmValueOpcode::tangent,      SurfaceSvmValueOpcode::fresnel,
    SurfaceSvmValueOpcode::layer_weight, SurfaceSvmValueOpcode::light_path,
    SurfaceSvmValueOpcode::object_info,  SurfaceSvmValueOpcode::particle_info,
    SurfaceSvmValueOpcode::hair_info};

static_assert(std::ranges::all_of(direct_context_families, [](auto family) {
  return surface_value_family_has_direct_evaluator(family);
}));

constexpr std::array direct_context_operations{
    ValueOperation::tangent,
    ValueOperation::object_location,
    ValueOperation::object_random,
    ValueOperation::random_per_island,
    ValueOperation::particle_index,
    ValueOperation::particle_random,
    ValueOperation::curve_is_strand,
    ValueOperation::curve_intercept,
    ValueOperation::curve_length,
    ValueOperation::curve_thickness,
    ValueOperation::curve_tangent_normal,
    ValueOperation::curve_random,
    ValueOperation::back_facing,
    ValueOperation::path_is_camera,
    ValueOperation::path_is_shadow,
    ValueOperation::path_is_diffuse,
    ValueOperation::path_is_glossy,
    ValueOperation::path_is_singular,
    ValueOperation::path_is_reflection,
    ValueOperation::path_is_transmission,
    ValueOperation::path_is_volume_scatter,
    ValueOperation::path_ray_length,
    ValueOperation::path_ray_depth,
    ValueOperation::path_diffuse_depth,
    ValueOperation::path_glossy_depth,
    ValueOperation::path_transparent_depth,
    ValueOperation::path_transmission_depth,
    ValueOperation::path_portal_depth,
    ValueOperation::fresnel,
    ValueOperation::layer_weight_fresnel,
    ValueOperation::layer_weight_facing};

[[nodiscard]] OutputRef scalarize(ShaderGraph &graph, OutputRef vector,
                                  std::string label, bool &configured) {
  const auto convert =
      graph.add_node(node_type::vector_to_scalar, std::move(label));
  configured &= graph.connect(std::move(vector), convert, "Vector");
  return {.node = convert, .socket = "Value"};
}

[[nodiscard]] OutputRef scalarize_normal(ShaderGraph &graph, OutputRef normal,
                                         std::string label, bool &configured) {
  const auto convert =
      graph.add_node(node_type::normal_to_vector, label + " vector");
  configured &= graph.connect(std::move(normal), convert, "Normal");
  return scalarize(graph, {.node = convert, .socket = "Vector"},
                   std::move(label), configured);
}

[[nodiscard]] OutputRef sum_scalars(ShaderGraph &graph,
                                    std::vector<OutputRef> values,
                                    bool &configured) {
  if (values.empty()) {
    throw std::runtime_error{"direct context fixture has no values"};
  }
  auto sum = std::move(values.front());
  for (auto index = std::size_t{1u}; index < values.size(); ++index) {
    const auto add = graph.add_node(
        node_type::add_float, "Direct context sum " + std::to_string(index));
    configured &= graph.connect(std::move(sum), add, "A");
    configured &= graph.connect(std::move(values[index]), add, "B");
    sum = OutputRef{.node = add, .socket = "Value"};
  }
  return sum;
}

void finish_emission_graph(ShaderGraph &graph, std::vector<OutputRef> values,
                           bool configured) {
  const auto emission =
      graph.add_node(node_type::emission, "Direct context emission");
  auto sum = sum_scalars(graph, std::move(values), configured);
  configured &= graph.connect(std::move(sum), emission, "Strength");
  configured &= graph.set_input(emission, "Color",
                                SocketValue::color({0.23f, 0.47f, 0.79f}));
  if (!configured) {
    throw std::runtime_error{"failed to configure direct context SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
}

[[nodiscard]] ShaderGraph make_projection_graph() {
  ShaderGraph graph;
  const auto geometry =
      graph.add_node(node_type::geometry, "Direct context Geometry");
  const auto object =
      graph.add_node(node_type::object_info, "Direct context Object Info");
  const auto particle =
      graph.add_node(node_type::particle_info, "Direct context Particle Info");
  const auto hair =
      graph.add_node(node_type::hair_info, "Direct context Hair Info");
  const auto light =
      graph.add_node(node_type::light_path, "Direct context Light Path");
  auto configured = true;
  std::vector<OutputRef> values;
  values.reserve(28u);
  values.emplace_back(
      scalarize(graph, OutputRef{.node = geometry, .socket = "Tangent"},
                "Direct tangent scalar", configured));
  values.emplace_back(OutputRef{.node = geometry, .socket = "Backfacing"});
  values.emplace_back(OutputRef{.node = geometry, .socket = "RandomPerIsland"});
  values.emplace_back(scalarize(graph,
                                OutputRef{.node = object, .socket = "Location"},
                                "Direct object location scalar", configured));
  values.emplace_back(OutputRef{.node = object, .socket = "Random"});
  values.emplace_back(OutputRef{.node = particle, .socket = "Index"});
  values.emplace_back(OutputRef{.node = particle, .socket = "Random"});
  for (const auto *socket :
       {"IsStrand", "Intercept", "Length", "Thickness", "Random"}) {
    values.emplace_back(OutputRef{.node = hair, .socket = socket});
  }
  values.emplace_back(scalarize_normal(
      graph, OutputRef{.node = hair, .socket = "TangentNormal"},
      "Direct hair tangent scalar", configured));
  for (const auto *socket :
       {"IsCameraRay", "IsShadowRay", "IsDiffuseRay", "IsGlossyRay",
        "IsSingularRay", "IsReflectionRay", "IsTransmissionRay",
        "IsVolumeScatterRay", "RayLength", "RayDepth", "DiffuseDepth",
        "GlossyDepth", "TransparentDepth", "TransmissionDepth",
        "PortalDepth"}) {
    values.emplace_back(OutputRef{.node = light, .socket = socket});
  }
  finish_emission_graph(graph, std::move(values), configured);
  return graph;
}

[[nodiscard]] ShaderGraph make_optics_graph(bool normal_linked) {
  ShaderGraph graph;
  const auto geometry =
      graph.add_node(node_type::geometry, "Direct optics Geometry");
  const auto object =
      graph.add_node(node_type::object_info, "Direct optics Object Info");
  const auto particle =
      graph.add_node(node_type::particle_info, "Direct optics Particle Info");
  const auto hair =
      graph.add_node(node_type::hair_info, "Direct optics Hair Info");
  const auto fresnel =
      graph.add_node(node_type::fresnel, "Direct optics Fresnel");
  const auto layer =
      graph.add_node(node_type::layer_weight, "Direct optics Layer Weight");
  const auto scalar_source = normal_linked ? particle : object;
  auto configured =
      graph.connect({.node = scalar_source, .socket = "Random"}, fresnel,
                    "IOR") &&
      graph.connect({.node = scalar_source, .socket = "Random"}, layer,
                    "Blend") &&
      // Both records carry a live normal operand. The immediate alone decides
      // whether Cycles observes it or the transaction shading normal.
      graph.connect({.node = hair, .socket = "TangentNormal"}, fresnel,
                    "Normal") &&
      graph.connect({.node = hair, .socket = "TangentNormal"}, layer,
                    "Normal") &&
      graph.set_property(fresnel, "NormalLinked",
                         SocketValue::boolean(normal_linked)) &&
      graph.set_property(layer, "NormalLinked",
                         SocketValue::boolean(normal_linked));
  std::vector<OutputRef> values{{.node = fresnel, .socket = "Factor"},
                                {.node = layer, .socket = "Fresnel"},
                                {.node = layer, .socket = "Facing"},
                                {.node = geometry, .socket = "Backfacing"}};
  finish_emission_graph(graph, std::move(values), configured);
  return graph;
}

[[nodiscard]] bool vector_result(ValueOperation operation) noexcept {
  return operation == ValueOperation::tangent ||
         operation == ValueOperation::object_location ||
         operation == ValueOperation::curve_tangent_normal;
}

} // namespace

std::vector<ShaderGraph> make_direct_context_graphs() {
  std::vector<ShaderGraph> graphs;
  graphs.reserve(3u);
  graphs.emplace_back(make_projection_graph());
  graphs.emplace_back(make_optics_graph(false));
  graphs.emplace_back(make_optics_graph(true));
  return graphs;
}

std::string
validate_direct_context_surface_runtime(const SurfaceValueRuntime &runtime) {
  for (const auto operation : direct_context_operations) {
    const auto variants = std::count_if(
        runtime.value_variants.begin(), runtime.value_variants.end(),
        [operation](const auto &variant) noexcept {
          return variant.instruction.operation == operation;
        });
    const auto variant = std::find_if(
        runtime.value_variants.begin(), runtime.value_variants.end(),
        [operation](const auto &candidate) noexcept {
          return candidate.instruction.operation == operation;
        });
    const auto optics = operation == ValueOperation::fresnel ||
                        operation == ValueOperation::layer_weight_fresnel ||
                        operation == ValueOperation::layer_weight_facing;
    const std::vector<std::uint16_t> expected_immediates =
        optics ? std::vector<std::uint16_t>{0u, 1u}
               : std::vector<std::uint16_t>{0u};
    const auto expected_operands =
        operation == ValueOperation::fresnel
            ? value_operand::fresnel::count
            : (operation == ValueOperation::layer_weight_fresnel ||
                       operation == ValueOperation::layer_weight_facing
                   ? value_operand::layer_weight::count
                   : 0u);
    auto bank = SurfaceValueBank::scalar;
    const auto valid =
        variants == 1u && variant != runtime.value_variants.end() &&
        variant->operand_types.size() == expected_operands &&
        variant->svm_immediates == expected_immediates &&
        classify_surface_value_type(variant->instruction.result_type, bank) &&
        bank == (vector_result(operation) ? SurfaceValueBank::vector
                                          : SurfaceValueBank::scalar) &&
        std::ranges::all_of(
            variant->svm_immediates, [&](std::uint16_t immediate) noexcept {
              const auto family =
                  surface_svm_value_opcode(operation, immediate);
              return surface_value_family_has_direct_evaluator(family);
            });
    if (!valid) {
      std::ostringstream message;
      message << "compact runtime violated the direct context family "
                 "projection (operation="
              << static_cast<std::uint32_t>(operation)
              << ", variants=" << variants << ')';
      return message.str();
    }
  }
  return {};
}

} // namespace psycles::test_support

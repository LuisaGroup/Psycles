#include "compact_surface_program_test_support.h"

#include "path_tracer_surface_value_family.h"
#include "path_tracer_surface_values.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <array>
#include <ranges>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace psycles::test_support {
namespace {

using namespace compiler;
using namespace contract;
using namespace luisa_backend::detail;

constexpr std::array direct_color_algebra_families{
    SurfaceSvmValueOpcode::hsv,
    SurfaceSvmValueOpcode::invert,
    SurfaceSvmValueOpcode::gamma,
    SurfaceSvmValueOpcode::brightness_contrast,
    SurfaceSvmValueOpcode::blackbody,
    SurfaceSvmValueOpcode::wavelength,
    SurfaceSvmValueOpcode::rgb_curve,
    SurfaceSvmValueOpcode::separate_color,
    SurfaceSvmValueOpcode::combine_color,
    SurfaceSvmValueOpcode::map_range,
    SurfaceSvmValueOpcode::vector_map_range,
    SurfaceSvmValueOpcode::mix_float,
    SurfaceSvmValueOpcode::mix_vector,
    SurfaceSvmValueOpcode::mix_vector_non_uniform};

static_assert(
    std::ranges::all_of(direct_color_algebra_families, [](auto family) {
      return surface_value_family_has_direct_evaluator(family);
    }));

constexpr std::array direct_color_algebra_operations{
    ValueOperation::hue_saturation,
    ValueOperation::invert,
    ValueOperation::gamma,
    ValueOperation::brightness_contrast,
    ValueOperation::blackbody,
    ValueOperation::wavelength,
    ValueOperation::rgb_curve,
    ValueOperation::separate_r,
    ValueOperation::separate_g,
    ValueOperation::separate_b,
    ValueOperation::combine_color,
    ValueOperation::map_range_float,
    ValueOperation::map_range_vector,
    ValueOperation::mix_float};

[[nodiscard]] OutputRef mix_color(ShaderGraph &graph, OutputRef a, OutputRef b,
                                  float factor, std::string label,
                                  bool &configured) {
  const auto mix = graph.add_node(node_type::mix_color, std::move(label));
  configured &= graph.connect(std::move(a), mix, "A");
  configured &= graph.connect(std::move(b), mix, "B");
  configured &= graph.set_input(mix, "Factor", SocketValue::floating(factor));
  configured &=
      graph.set_property(mix, "BlendMode", SocketValue::string("MIX"));
  configured &=
      graph.set_property(mix, "ClampFactor", SocketValue::boolean(false));
  configured &=
      graph.set_property(mix, "ClampResult", SocketValue::boolean(false));
  return {.node = mix, .socket = "Color"};
}

[[nodiscard]] ShaderGraph
make_color_algebra_graph(std::uint32_t configuration) {
  constexpr std::array gamma_exponents{0.0f, 2.2f, -0.75f, 0.5f};
  constexpr std::array hue_factors{-0.25f, 0.4f, 1.3f, 0.75f};
  constexpr std::array modes{"RGB", "HSV", "HSL"};
  const auto clamp_factor = (configuration & 1u) != 0u;
  const auto non_uniform = (configuration & 2u) != 0u;
  const auto sampled_curve = (configuration & 1u) != 0u;
  const auto color_mode = configuration % modes.size();

  ShaderGraph graph;
  const auto geometry =
      graph.add_node(node_type::geometry, "Direct color Geometry");
  const auto normal_to_vector = graph.add_node(node_type::normal_to_vector,
                                               "Direct color dynamic vector");
  const auto vector_to_color =
      graph.add_node(node_type::vector_to_color, "Direct color dynamic color");
  const auto vector_to_scalar = graph.add_node(node_type::vector_to_scalar,
                                               "Direct color dynamic scalar");
  const auto gamma =
      graph.add_node(node_type::gamma_color, "Direct Cycles Gamma");
  const auto hsv =
      graph.add_node(node_type::hue_saturation, "Direct Cycles Hue Saturation");
  const auto invert =
      graph.add_node(node_type::invert_color, "Direct Cycles Invert");
  const auto brightness = graph.add_node(node_type::brightness_contrast,
                                         "Direct Cycles Brightness Contrast");
  const auto curve =
      graph.add_node(node_type::rgb_curve, "Direct Cycles RGB Curve");
  const auto separate =
      graph.add_node(node_type::separate_color, "Direct Cycles Separate");
  const auto combine =
      graph.add_node(node_type::combine_color, "Direct Cycles Combine");
  const auto temperature_mix = graph.add_node(
      node_type::mix_float, "Direct Cycles Blackbody temperature");
  const auto wavelength_mix = graph.add_node(
      node_type::mix_float, "Direct Cycles wavelength coordinate");
  const auto blackbody =
      graph.add_node(node_type::blackbody, "Direct Cycles Blackbody");
  const auto wavelength =
      graph.add_node(node_type::wavelength, "Direct Cycles Wavelength");
  const auto vector_mix = graph.add_node(
      non_uniform ? node_type::mix_vector_nonuniform : node_type::mix_vector,
      "Direct Cycles Vector Mix");
  const auto mixed_vector_to_color =
      graph.add_node(node_type::vector_to_color, "Direct mixed vector color");
  const auto emission =
      graph.add_node(node_type::emission, "Direct color emission");

  auto configured =
      graph.connect({.node = geometry, .socket = "Normal"}, normal_to_vector,
                    "Normal") &&
      graph.connect({.node = normal_to_vector, .socket = "Vector"},
                    vector_to_color, "Vector") &&
      graph.connect({.node = normal_to_vector, .socket = "Vector"},
                    vector_to_scalar, "Vector") &&
      graph.connect({.node = vector_to_color, .socket = "Color"}, gamma,
                    "Color") &&
      graph.set_input(gamma, "Gamma",
                      SocketValue::floating(gamma_exponents[configuration])) &&
      graph.connect({.node = gamma, .socket = "Color"}, hsv, "Color") &&
      graph.set_input(hsv, "Hue", SocketValue::floating(0.17f)) &&
      graph.set_input(hsv, "Saturation", SocketValue::floating(1.8f)) &&
      graph.set_input(hsv, "Value", SocketValue::floating(0.73f)) &&
      graph.set_input(hsv, "Factor",
                      SocketValue::floating(hue_factors[configuration])) &&
      graph.connect({.node = hsv, .socket = "Color"}, invert, "Color") &&
      graph.set_input(invert, "Factor", SocketValue::floating(1.27f)) &&
      graph.connect({.node = invert, .socket = "Color"}, brightness, "Color") &&
      graph.set_input(brightness, "Bright", SocketValue::floating(-0.13f)) &&
      graph.set_input(brightness, "Contrast", SocketValue::floating(0.42f)) &&
      graph.connect({.node = brightness, .socket = "Color"}, curve, "Color") &&
      graph.set_input(curve, "Factor", SocketValue::floating(0.81f)) &&
      graph.set_property(curve, "Sampled",
                         SocketValue::boolean(sampled_curve)) &&
      graph.set_property(curve, "MinX", SocketValue::floating(-0.25f)) &&
      graph.set_property(curve, "MaxX", SocketValue::floating(1.25f)) &&
      graph.set_property(curve, "Extrapolate",
                         SocketValue::boolean(configuration != 3u)) &&
      graph.set_property(
          curve, "Table",
          SocketValue::string(
              configuration < 2u
                  ? "0,0.05,0.15,0.25;0.45,0.55,0.35,0.75;1,0.95,0.8,0.6"
                  : "0,0.9,0.2,0.1;0.6,0.4,0.7,0.3;1,0.1,0.8,0.95")) &&
      graph.connect({.node = curve, .socket = "Color"}, separate, "Color") &&
      graph.set_property(separate, "Mode",
                         SocketValue::string(modes[color_mode])) &&
      graph.connect({.node = separate, .socket = "R"}, combine, "R") &&
      graph.connect({.node = separate, .socket = "G"}, combine, "G") &&
      graph.connect({.node = separate, .socket = "B"}, combine, "B") &&
      graph.set_property(combine, "Mode",
                         SocketValue::string(modes[color_mode])) &&
      graph.connect({.node = vector_to_scalar, .socket = "Value"},
                    temperature_mix, "Factor") &&
      graph.set_input(temperature_mix, "A", SocketValue::floating(700.0f)) &&
      graph.set_input(temperature_mix, "B", SocketValue::floating(13000.0f)) &&
      graph.set_property(temperature_mix, "ClampFactor",
                         SocketValue::boolean(clamp_factor)) &&
      graph.connect({.node = temperature_mix, .socket = "Value"}, blackbody,
                    "Temperature") &&
      graph.connect({.node = vector_to_scalar, .socket = "Value"},
                    wavelength_mix, "Factor") &&
      graph.set_input(wavelength_mix, "A", SocketValue::floating(360.0f)) &&
      graph.set_input(wavelength_mix, "B", SocketValue::floating(830.0f)) &&
      graph.set_property(wavelength_mix, "ClampFactor",
                         SocketValue::boolean(clamp_factor)) &&
      graph.connect({.node = wavelength_mix, .socket = "Value"}, wavelength,
                    "Wavelength") &&
      graph.connect({.node = normal_to_vector, .socket = "Vector"}, vector_mix,
                    "A") &&
      graph.set_input(vector_mix, "B",
                      SocketValue::vector({0.91f, -0.37f, 0.23f})) &&
      graph.set_property(vector_mix, "ClampFactor",
                         SocketValue::boolean(clamp_factor));
  if (non_uniform) {
    configured &= graph.connect({.node = normal_to_vector, .socket = "Vector"},
                                vector_mix, "Factor");
  } else {
    configured &= graph.connect({.node = vector_to_scalar, .socket = "Value"},
                                vector_mix, "Factor");
  }
  configured &= graph.connect({.node = vector_mix, .socket = "Vector"},
                              mixed_vector_to_color, "Vector");

  auto spectral = mix_color(graph, {.node = blackbody, .socket = "Color"},
                            {.node = wavelength, .socket = "Color"}, 0.37f,
                            "Direct spectral color join", configured);
  auto algebra = mix_color(graph, {.node = combine, .socket = "Color"},
                           std::move(spectral), 0.43f,
                           "Direct algebra color join", configured);
  auto final_color =
      mix_color(graph, std::move(algebra),
                {.node = mixed_vector_to_color, .socket = "Color"}, 0.61f,
                "Direct color final join", configured);
  configured &= graph.connect(std::move(final_color), emission, "Color");
  configured &=
      graph.set_input(emission, "Strength", SocketValue::floating(1.0f));
  if (!configured) {
    throw std::runtime_error{
        "failed to configure direct color/algebra SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  return graph;
}

[[nodiscard]] std::size_t
expected_operand_count(ValueOperation operation) noexcept {
  switch (operation) {
  case ValueOperation::hue_saturation:
    return value_operand::hue_saturation::count;
  case ValueOperation::invert:
    return value_operand::color_factor::count;
  case ValueOperation::gamma:
    return value_operand::gamma::count;
  case ValueOperation::brightness_contrast:
    return value_operand::brightness_contrast::count;
  case ValueOperation::blackbody:
    return value_operand::blackbody::count;
  case ValueOperation::wavelength:
    return value_operand::wavelength::count;
  case ValueOperation::rgb_curve:
    return value_operand::rgb_curve::count;
  case ValueOperation::separate_r:
  case ValueOperation::separate_g:
  case ValueOperation::separate_b:
    return value_operand::separate_color::count;
  case ValueOperation::combine_color:
    return value_operand::combine_color::count;
  case ValueOperation::map_range_float:
  case ValueOperation::map_range_vector:
    return value_operand::map_range::count;
  case ValueOperation::mix_float:
  case ValueOperation::mix_vector:
    return value_operand::mix::count;
  default:
    return 0u;
  }
}

[[nodiscard]] SurfaceValueBank
expected_result_bank(ValueOperation operation) noexcept {
  switch (operation) {
  case ValueOperation::separate_r:
  case ValueOperation::separate_g:
  case ValueOperation::separate_b:
  case ValueOperation::map_range_float:
  case ValueOperation::mix_float:
    return SurfaceValueBank::scalar;
  default:
    return SurfaceValueBank::vector;
  }
}

[[nodiscard]] std::vector<std::uint16_t>
expected_immediates(ValueOperation operation) {
  switch (operation) {
  case ValueOperation::rgb_curve:
  case ValueOperation::mix_float:
    return {0u, 1u};
  case ValueOperation::separate_r:
  case ValueOperation::separate_g:
  case ValueOperation::separate_b:
  case ValueOperation::combine_color:
    return {0u, 1u, 2u};
  case ValueOperation::map_range_float:
  case ValueOperation::map_range_vector:
    return {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
  default:
    return {0u};
  }
}

[[nodiscard]] bool direct_variant_contract_holds(
    const SurfaceValueStaticVariant &variant,
    std::span<const std::uint16_t> expected_domain) noexcept {
  auto bank = SurfaceValueBank::scalar;
  return variant.operand_types.size() ==
             expected_operand_count(variant.instruction.operation) &&
         classify_surface_value_type(variant.instruction.result_type, bank) &&
         bank == expected_result_bank(variant.instruction.operation) &&
         std::ranges::equal(variant.svm_immediates, expected_domain) &&
         std::ranges::all_of(
             variant.svm_immediates, [&](std::uint16_t immediate) noexcept {
               return surface_value_family_has_direct_evaluator(
                   surface_svm_value_opcode(variant.instruction.operation,
                                            immediate));
             });
}

} // namespace

std::vector<ShaderGraph> make_direct_color_algebra_graphs() {
  std::vector<ShaderGraph> graphs;
  graphs.reserve(4u);
  for (auto configuration = 0u; configuration < 4u; ++configuration) {
    graphs.emplace_back(make_color_algebra_graph(configuration));
  }
  return graphs;
}

std::string validate_direct_color_algebra_surface_runtime(
    const SurfaceValueRuntime &runtime) {
  for (const auto operation : direct_color_algebra_operations) {
    const auto count = std::ranges::count_if(
        runtime.value_variants, [operation](const auto &variant) noexcept {
          return variant.instruction.operation == operation;
        });
    const auto variant = std::ranges::find_if(
        runtime.value_variants, [operation](const auto &candidate) noexcept {
          return candidate.instruction.operation == operation;
        });
    const auto domain = expected_immediates(operation);
    if (count != 1u || variant == runtime.value_variants.end() ||
        !direct_variant_contract_holds(*variant, domain)) {
      std::ostringstream message;
      message << "compact runtime violated direct color/algebra "
                 "projection (operation="
              << static_cast<std::uint32_t>(operation) << ", variants=" << count
              << ')';
      return message.str();
    }
  }

  std::array<bool, 2u> mix_vector_shapes{};
  auto mix_vector_count = std::size_t{};
  for (const auto &variant : runtime.value_variants) {
    if (variant.instruction.operation != ValueOperation::mix_vector) {
      continue;
    }
    ++mix_vector_count;
    if (variant.operand_types.size() != value_operand::mix::count) {
      return "direct Mix Vector variant has an invalid operand arity";
    }
    auto factor_bank = SurfaceValueBank::scalar;
    auto result = SurfaceValueBank::scalar;
    if (!classify_surface_value_type(
            variant.operand_types[value_operand::mix::factor], factor_bank) ||
        !classify_surface_value_type(variant.instruction.result_type, result) ||
        result != SurfaceValueBank::vector ||
        (factor_bank != SurfaceValueBank::scalar &&
         factor_bank != SurfaceValueBank::vector)) {
      return "direct Mix Vector variant has an invalid typed bank";
    }
    const auto non_uniform = factor_bank == SurfaceValueBank::vector;
    const std::array<std::uint16_t, 2u> expected =
        non_uniform ? std::array<std::uint16_t, 2u>{1u, 3u}
                    : std::array<std::uint16_t, 2u>{0u, 2u};
    if (mix_vector_shapes[non_uniform] ||
        !std::ranges::equal(variant.svm_immediates, expected) ||
        !std::ranges::all_of(
            variant.svm_immediates, [&](std::uint16_t immediate) noexcept {
              const auto family = surface_svm_value_opcode(
                  ValueOperation::mix_vector, immediate);
              return surface_value_family_has_direct_evaluator(family) &&
                     (family ==
                      SurfaceSvmValueOpcode::mix_vector_non_uniform) ==
                         non_uniform;
            })) {
      return "direct Mix Vector lost its uniform/non-uniform product";
    }
    mix_vector_shapes[non_uniform] = true;
  }
  if (mix_vector_count != 2u ||
      !std::ranges::all_of(mix_vector_shapes,
                           [](bool present) { return present; })) {
    return "direct Mix Vector did not produce exactly two typed shapes";
  }
  return {};
}

} // namespace psycles::test_support

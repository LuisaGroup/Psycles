#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_compiler.h>
#include <psycles/luisa/cycles_svm.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::contract;
using namespace psycles::compiler;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;

constexpr std::array legacy_mix_modes{
    std::pair{"MIX", NODE_MIX_BLEND},
    std::pair{"DARKEN", NODE_MIX_DARK},
    std::pair{"MULTIPLY", NODE_MIX_MUL},
    std::pair{"BURN", NODE_MIX_BURN},
    std::pair{"LIGHTEN", NODE_MIX_LIGHT},
    std::pair{"SCREEN", NODE_MIX_SCREEN},
    std::pair{"DODGE", NODE_MIX_DODGE},
    std::pair{"ADD", NODE_MIX_ADD},
    std::pair{"OVERLAY", NODE_MIX_OVERLAY},
    std::pair{"SOFT_LIGHT", NODE_MIX_SOFT},
    std::pair{"LINEAR_LIGHT", NODE_MIX_LINEAR},
    std::pair{"DIFFERENCE", NODE_MIX_DIFF},
    std::pair{"EXCLUSION", NODE_MIX_EXCLUSION},
    std::pair{"SUBTRACT", NODE_MIX_SUB},
    std::pair{"DIVIDE", NODE_MIX_DIV},
    std::pair{"HUE", NODE_MIX_HUE},
    std::pair{"SATURATION", NODE_MIX_SAT},
    std::pair{"COLOR", NODE_MIX_COL},
    std::pair{"VALUE", NODE_MIX_VAL},
};

[[nodiscard]] ShaderImage compile_dynamic_math() {
  ShaderGraph graph;
  const auto geometry_a = graph.add_node(node_type::geometry, "Geometry A");
  const auto geometry_b = graph.add_node(node_type::geometry, "Geometry B");
  const auto math = graph.add_node(node_type::math, "Dynamic Add");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto configured =
      graph.set_property(math, "Operation",
                         contract::SocketValue::string("ADD")) &&
      graph.connect({geometry_a, "Backfacing"}, math, "A") &&
      graph.connect({geometry_b, "Backfacing"}, math, "B") &&
      graph.set_input(emission, "Color",
                      contract::SocketValue::color(
                          {0.21f, 0.47f, 0.83f})) &&
      graph.connect({math, "Value"}, emission, "Strength");
  if (!configured) {
    throw std::runtime_error{"failed to create dynamic Math SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"dynamic Math graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] ShaderImage compile_dynamic_mix() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto transparent =
      graph.add_node(node_type::transparent_bsdf, "Transparent BSDF");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto mix = graph.add_node(node_type::mix_closure, "Dynamic Mix");
  const auto configured =
      graph.set_input(transparent, "Color",
                      contract::SocketValue::color(
                          {0.75f, 0.9f, 0.6f})) &&
      graph.set_input(emission, "Color",
                      contract::SocketValue::color(
                          {0.85f, 0.08f, 0.03f})) &&
      graph.set_input(emission, "Strength",
                      contract::SocketValue::floating(1.2f)) &&
      graph.connect({geometry, "Backfacing"}, mix, "Factor") &&
      graph.connect({transparent, "Closure"}, mix, "A") &&
      graph.connect({emission, "Closure"}, mix, "B");
  if (!configured) {
    throw std::runtime_error{"failed to create dynamic Mix SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = mix, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"dynamic Mix graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] ShaderImage compile_dynamic_color_pipeline() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Dynamic Backfacing");
  const auto invert = graph.add_node(node_type::invert_color, "Dynamic Invert");
  const auto gamma = graph.add_node(node_type::gamma_color, "Gamma");
  const auto brightness = graph.add_node(
      node_type::brightness_contrast, "Dynamic Brightness Contrast");
  const auto hsv = graph.add_node(node_type::hue_saturation, "Dynamic HSV");
  const auto clamp = graph.add_node(node_type::clamp_range,
                                    "Dynamic Range Clamp");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto configured =
      graph.set_input(invert, "Color",
                      contract::SocketValue::color({0.12f, 0.47f, 0.81f})) &&
      graph.connect({geometry, "Backfacing"}, invert, "Factor") &&
      graph.set_input(gamma, "Gamma",
                      contract::SocketValue::floating(2.2f)) &&
      graph.connect({invert, "Color"}, gamma, "Color") &&
      graph.set_input(brightness, "Bright",
                      contract::SocketValue::floating(0.17f)) &&
      graph.connect({gamma, "Color"}, brightness, "Color") &&
      graph.connect({geometry, "Backfacing"}, brightness, "Contrast") &&
      graph.set_input(hsv, "Hue",
                      contract::SocketValue::floating(0.3f)) &&
      graph.set_input(hsv, "Saturation",
                      contract::SocketValue::floating(1.4f)) &&
      graph.set_input(hsv, "Value",
                      contract::SocketValue::floating(0.75f)) &&
      graph.connect({brightness, "Color"}, hsv, "Color") &&
      graph.connect({geometry, "Backfacing"}, hsv, "Factor") &&
      graph.set_property(clamp, "Mode",
                         contract::SocketValue::string("RANGE")) &&
      graph.set_input(clamp, "Min",
                      contract::SocketValue::floating(0.8f)) &&
      graph.set_input(clamp, "Max",
                      contract::SocketValue::floating(0.2f)) &&
      graph.connect({geometry, "Backfacing"}, clamp, "Value") &&
      graph.connect({hsv, "Color"}, emission, "Color") &&
      graph.connect({clamp, "Result"}, emission, "Strength");
  if (!configured) {
    throw std::runtime_error{"failed to create dynamic color SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"dynamic color graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] ShaderImage compile_dynamic_combsep_color_pipeline() {
  ShaderGraph graph;
  const auto geometry =
      graph.add_node(node_type::geometry, "Dynamic Backfacing");
  const auto combine_hsl =
      graph.add_node(node_type::combine_color, "Dynamic Combine HSL");
  const auto separate_hsv =
      graph.add_node(node_type::separate_color, "Dynamic Separate HSV");
  const auto combine_rgb =
      graph.add_node(node_type::combine_color, "Dynamic Combine RGB");
  const auto separate_hsl =
      graph.add_node(node_type::separate_color, "Dynamic Separate HSL");
  const auto combine_hsv =
      graph.add_node(node_type::combine_color, "Dynamic Combine HSV");
  const auto separate_rgb =
      graph.add_node(node_type::separate_color, "Dynamic Separate RGB");
  const auto combine_rgb_final =
      graph.add_node(node_type::combine_color, "Dynamic Final Combine RGB");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto configured =
      graph.set_property(combine_hsl, "Mode",
                         contract::SocketValue::string("HSL")) &&
      graph.set_input(combine_hsl, "R",
                      contract::SocketValue::floating(0.13f)) &&
      graph.set_input(combine_hsl, "B",
                      contract::SocketValue::floating(0.36f)) &&
      graph.connect({geometry, "Backfacing"}, combine_hsl, "G") &&
      graph.set_property(separate_hsv, "Mode",
                         contract::SocketValue::string("HSV")) &&
      graph.connect({combine_hsl, "Color"}, separate_hsv, "Color") &&
      graph.set_property(combine_rgb, "Mode",
                         contract::SocketValue::string("RGB")) &&
      graph.connect({separate_hsv, "B"}, combine_rgb, "R") &&
      graph.connect({separate_hsv, "R"}, combine_rgb, "G") &&
      graph.connect({separate_hsv, "G"}, combine_rgb, "B") &&
      graph.set_property(separate_hsl, "Mode",
                         contract::SocketValue::string("HSL")) &&
      graph.connect({combine_rgb, "Color"}, separate_hsl, "Color") &&
      graph.set_property(combine_hsv, "Mode",
                         contract::SocketValue::string("HSV")) &&
      graph.connect({separate_hsl, "R"}, combine_hsv, "R") &&
      graph.connect({separate_hsl, "G"}, combine_hsv, "G") &&
      graph.connect({separate_hsl, "B"}, combine_hsv, "B") &&
      graph.set_property(separate_rgb, "Mode",
                         contract::SocketValue::string("RGB")) &&
      graph.connect({combine_hsv, "Color"}, separate_rgb, "Color") &&
      graph.set_property(combine_rgb_final, "Mode",
                         contract::SocketValue::string("RGB")) &&
      graph.connect({separate_rgb, "B"}, combine_rgb_final, "R") &&
      graph.connect({separate_rgb, "R"}, combine_rgb_final, "G") &&
      graph.connect({separate_rgb, "G"}, combine_rgb_final, "B") &&
      graph.connect({combine_rgb_final, "Color"}, emission, "Color");
  if (!configured) {
    throw std::runtime_error{
        "failed to create dynamic Combine/Separate Color SVM graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{
        "dynamic Combine/Separate Color graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] ShaderImage compile_dynamic_sepcomb_vector_pipeline() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto normal_to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto vector_to_color =
      graph.add_node(node_type::vector_to_color, "Vector to Color");
  const auto separate =
      graph.add_node(node_type::separate_xyz, "Separate XYZ");
  const auto combine = graph.add_node(node_type::combine_xyz, "Combine XYZ");
  const auto result_to_color =
      graph.add_node(node_type::vector_to_color, "Result to Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto configured =
      graph.connect({geometry, "Normal"}, normal_to_vector, "Normal") &&
      graph.connect({normal_to_vector, "Vector"}, vector_to_color,
                    "Vector") &&
      graph.connect({vector_to_color, "Color"}, separate, "Vector") &&
      graph.connect({separate, "Z"}, combine, "X") &&
      graph.connect({separate, "X"}, combine, "Y") &&
      graph.connect({separate, "Y"}, combine, "Z") &&
      graph.connect({combine, "Vector"}, result_to_color, "Vector") &&
      graph.connect({result_to_color, "Color"}, emission, "Color");
  if (!configured) {
    throw std::runtime_error{
        "failed to create dynamic Separate/Combine XYZ graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{
        "dynamic Separate/Combine XYZ graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] ShaderImage compile_dynamic_legacy_mix(std::string_view mode,
                                                     NodeMix type) {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Backfacing");
  const auto factor = graph.add_node(node_type::math, "Dynamic Factor");
  const auto mix =
      graph.add_node(node_type::legacy_mix_color, std::string{mode});
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto configured =
      graph.set_property(factor, "Operation",
                         contract::SocketValue::string("MULTIPLY_ADD")) &&
      graph.connect({geometry, "Backfacing"}, factor, "A") &&
      graph.set_input(factor, "B", contract::SocketValue::floating(0.41f)) &&
      graph.set_input(factor, "C", contract::SocketValue::floating(0.23f)) &&
      graph.set_property(mix, "BlendMode",
                         contract::SocketValue::string(std::string{mode})) &&
      graph.set_property(mix, "ClampResult",
                         contract::SocketValue::boolean(type == NODE_MIX_ADD)) &&
      graph.connect({factor, "Value"}, mix, "Factor") &&
      graph.set_input(
          mix, "A", contract::SocketValue::color({0.17f, 0.63f, 0.89f})) &&
      graph.set_input(
          mix, "B", contract::SocketValue::color({0.82f, 0.24f, 0.51f})) &&
      graph.connect({mix, "Color"}, emission, "Color");
  if (!configured) {
    throw std::runtime_error{"failed to create dynamic legacy Mix graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});

  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"dynamic legacy Mix graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] ShaderImage compile_dynamic_modern_mix_color(
    std::size_t index) {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Backfacing");
  const auto factor = graph.add_node(node_type::math, "Dynamic Factor");
  const auto &[mode, unused_type] = legacy_mix_modes[index];
  const auto mix = graph.add_node(node_type::mix_color, mode);
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto configured =
      graph.set_property(factor, "Operation",
                         contract::SocketValue::string("MULTIPLY_ADD")) &&
      graph.connect({geometry, "Backfacing"}, factor, "A") &&
      graph.set_input(factor, "B", contract::SocketValue::floating(1.5f)) &&
      graph.set_input(factor, "C", contract::SocketValue::floating(-0.25f)) &&
      graph.set_property(
          mix, "BlendMode",
          contract::SocketValue::string(std::string{mode})) &&
      graph.set_property(
          mix, "ClampFactor",
          contract::SocketValue::boolean(index % 2u == 0u)) &&
      graph.set_property(
          mix, "ClampResult",
          contract::SocketValue::boolean(index % 3u == 0u)) &&
      graph.connect({factor, "Value"}, mix, "Factor") &&
      graph.set_input(
          mix, "A", contract::SocketValue::color({0.17f, 0.63f, 0.89f})) &&
      graph.set_input(
          mix, "B", contract::SocketValue::color({0.82f, 0.24f, 0.51f})) &&
      graph.connect({mix, "Color"}, emission, "Color");
  if (!configured) {
    throw std::runtime_error{"failed to create dynamic modern MixColor graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"dynamic modern MixColor graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  static_cast<void>(unused_type);
  return image;
}

[[nodiscard]] ShaderImage compile_dynamic_modern_mix_data(
    std::size_t kind, bool clamp) {
  ShaderGraph graph;
  const auto type = kind == 0u
                        ? node_type::mix_float
                        : kind == 1u ? node_type::mix_vector
                                     : node_type::mix_vector_nonuniform;
  const auto mix = graph.add_node(type, "Dynamic Modern Typed Mix");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  auto configured = graph.set_property(
      mix, "ClampFactor", contract::SocketValue::boolean(clamp));
  if (kind != 2u) {
    const auto geometry = graph.add_node(node_type::geometry, "Backfacing");
    const auto factor = graph.add_node(node_type::math, "Dynamic Factor");
    configured =
        configured &&
        graph.set_property(factor, "Operation",
                           contract::SocketValue::string("MULTIPLY_ADD")) &&
        graph.connect({geometry, "Backfacing"}, factor, "A") &&
        graph.set_input(factor, "B",
                        contract::SocketValue::floating(1.5f)) &&
        graph.set_input(factor, "C",
                        contract::SocketValue::floating(-0.25f)) &&
        graph.connect({factor, "Value"}, mix, "Factor");
  } else {
    const auto geometry = graph.add_node(node_type::geometry, "Normal");
    const auto convert =
        graph.add_node(node_type::normal_to_vector, "Normal to Vector");
    configured = configured &&
                 graph.connect({geometry, "Normal"}, convert, "Normal") &&
                 graph.connect({convert, "Vector"}, mix, "Factor");
  }
  if (kind == 0u) {
    configured =
        configured &&
        graph.set_input(mix, "A", contract::SocketValue::floating(0.2f)) &&
        graph.set_input(mix, "B", contract::SocketValue::floating(0.8f)) &&
        graph.set_input(
            emission, "Color",
            contract::SocketValue::color({0.31f, 0.57f, 0.83f})) &&
        graph.connect({mix, "Value"}, emission, "Strength");
  } else {
    const auto convert =
        graph.add_node(node_type::vector_to_color, "Vector to Color");
    configured =
        configured &&
        graph.set_input(
            mix, "A", contract::SocketValue::vector({0.1f, 0.7f, -0.2f})) &&
        graph.set_input(
            mix, "B", contract::SocketValue::vector({0.9f, -0.1f, 0.6f})) &&
        graph.connect({mix, "Vector"}, convert, "Vector") &&
        graph.connect({convert, "Color"}, emission, "Color");
  }
  if (!configured) {
    throw std::runtime_error{"failed to create dynamic modern typed Mix graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"dynamic modern typed Mix graph did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] ShaderImage compile_dynamic_modern_mix_chain() {
  ShaderGraph graph;
  const auto geometry = graph.add_node(node_type::geometry, "Geometry");
  const auto mix_float = graph.add_node(node_type::mix_float, "Mix Float");
  const auto mix_uniform =
      graph.add_node(node_type::mix_vector, "Mix Vector Uniform");
  const auto mix_nonuniform = graph.add_node(
      node_type::mix_vector_nonuniform, "Mix Vector Non Uniform");
  const auto normal_to_vector =
      graph.add_node(node_type::normal_to_vector, "Normal to Vector");
  const auto uniform_to_color =
      graph.add_node(node_type::vector_to_color, "Uniform to Color");
  const auto nonuniform_to_color =
      graph.add_node(node_type::vector_to_color, "Non Uniform to Color");
  const auto mix_color = graph.add_node(node_type::mix_color, "Mix Color");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto configured =
      graph.set_property(mix_float, "ClampFactor",
                         contract::SocketValue::boolean(false)) &&
      graph.connect({geometry, "Backfacing"}, mix_float, "Factor") &&
      graph.set_input(mix_float, "A",
                      contract::SocketValue::floating(0.2f)) &&
      graph.set_input(mix_float, "B",
                      contract::SocketValue::floating(0.8f)) &&
      graph.set_property(mix_uniform, "ClampFactor",
                         contract::SocketValue::boolean(true)) &&
      graph.connect({mix_float, "Value"}, mix_uniform, "Factor") &&
      graph.set_input(
          mix_uniform, "A",
          contract::SocketValue::vector({0.1f, 0.7f, -0.2f})) &&
      graph.set_input(
          mix_uniform, "B",
          contract::SocketValue::vector({0.9f, -0.1f, 0.6f})) &&
      graph.set_property(mix_nonuniform, "ClampFactor",
                         contract::SocketValue::boolean(false)) &&
      graph.connect({geometry, "Normal"}, normal_to_vector, "Normal") &&
      graph.connect({normal_to_vector, "Vector"}, mix_nonuniform,
                    "Factor") &&
      graph.set_input(
          mix_nonuniform, "A",
          contract::SocketValue::vector({0.3f, -0.4f, 0.8f})) &&
      graph.set_input(
          mix_nonuniform, "B",
          contract::SocketValue::vector({-0.2f, 0.6f, 0.1f})) &&
      graph.connect({mix_uniform, "Vector"}, uniform_to_color, "Vector") &&
      graph.connect({mix_nonuniform, "Vector"}, nonuniform_to_color,
                    "Vector") &&
      graph.set_property(mix_color, "BlendMode",
                         contract::SocketValue::string("OVERLAY")) &&
      graph.set_property(mix_color, "ClampFactor",
                         contract::SocketValue::boolean(false)) &&
      graph.set_property(mix_color, "ClampResult",
                         contract::SocketValue::boolean(true)) &&
      graph.connect({geometry, "Backfacing"}, mix_color, "Factor") &&
      graph.connect({uniform_to_color, "Color"}, mix_color, "A") &&
      graph.connect({nonuniform_to_color, "Color"}, mix_color, "B") &&
      graph.connect({mix_color, "Color"}, emission, "Color");
  if (!configured) {
    throw std::runtime_error{"failed to create dynamic modern Mix chain"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = emission, .socket = "Closure"});
  const ShaderCompiler frontend{make_core_node_registry()};
  const auto shader = frontend.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"dynamic modern Mix chain did not validate"};
  }
  auto image = compile_shader(*shader.program);
  if (!image.valid) {
    throw std::runtime_error{image.diagnostic};
  }
  return image;
}

[[nodiscard]] auto make_interpreter_kernel(
    std::array<bool, NODE_NUM> node_types_used,
    luisa::float3 shader_normal = {0.0f, 0.0f, 1.0f}) {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float4>,
                  Buffer<luisa::uint4>>{
      [node_types_used, shader_normal](BufferUInt words,
                                       BufferFloat4 floating_output,
                                       BufferUInt4 integer_output) noexcept {
        const UInt index = dispatch_x();
        const UInt shader_flags =
            select(0u, device_svm::shader_data_backfacing, index != 0u);
        device_svm::ShaderData shader_data{
            make_float3(1.0f, 2.0f, 3.0f),
            make_float3(shader_normal.x, shader_normal.y, shader_normal.z),
            make_float3(shader_normal.x, shader_normal.y, shader_normal.z),
            make_float3(0.0f, 0.0f, -1.0f),
            0u,
            shader_flags,
            0.2f,
            0.3f,
            4.0f};
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(
            words, SHADER_TYPE_SURFACE,
            device_svm::kernel_feature_node_emission |
                device_svm::kernel_feature_node_light_path,
            node_types_used, shader_data, path_state, result);
        floating_output.write(
            index,
            make_float4(shader_data.closure_emission_background,
                        result.closure_weight.x));
        integer_output.write(
            index,
            make_uint4(result.status, result.final_offset, shader_data.flag,
                       0u));
      }};
}

struct InterpreterShape final : StmtVisitor {
  std::uint32_t loop_depth{};
  std::uint32_t switch_depth{};
  std::uint32_t loops{};
  std::uint32_t primary_switches{};

  void visit(const BreakStmt *) override {}
  void visit(const ContinueStmt *) override {}
  void visit(const ReturnStmt *) override {}
  void visit(const ScopeStmt *stmt) override {
    for (const auto *statement : stmt->statements()) {
      statement->accept(*this);
    }
  }
  void visit(const IfStmt *stmt) override {
    stmt->true_branch()->accept(*this);
    stmt->false_branch()->accept(*this);
  }
  void visit(const LoopStmt *stmt) override {
    ++loops;
    ++loop_depth;
    stmt->body()->accept(*this);
    --loop_depth;
  }
  void visit(const ExprStmt *) override {}
  void visit(const SwitchStmt *stmt) override {
    if (loop_depth == 1u && switch_depth == 0u) {
      ++primary_switches;
    }
    ++switch_depth;
    stmt->body()->accept(*this);
    --switch_depth;
  }
  void visit(const SwitchCaseStmt *stmt) override {
    stmt->body()->accept(*this);
  }
  void visit(const SwitchDefaultStmt *stmt) override {
    stmt->body()->accept(*this);
  }
  void visit(const AssignStmt *) override {}
  void visit(const ForStmt *stmt) override { stmt->body()->accept(*this); }
  void visit(const CommentStmt *) override {}
  void visit(const RayQueryStmt *stmt) override {
    stmt->on_triangle_candidate()->accept(*this);
    stmt->on_procedural_candidate()->accept(*this);
  }
  void visit(const SuspendStmt *) override {}
  void visit(const AutoDiffStmt *stmt) override {
    stmt->body()->accept(*this);
  }
  void visit(const PrintStmt *) override {}
  void visit(const DebugBreakStmt *) override {}
};

[[nodiscard]] bool approximately_equal(float actual, float expected) {
  return std::abs(actual - expected) <= 2.0e-6f;
}

[[nodiscard]] bool require_float3(const luisa::float4 &actual,
                                  const luisa::float3 &expected,
                                  std::string_view label) {
  if (approximately_equal(actual.x, expected.x) &&
      approximately_equal(actual.y, expected.y) &&
      approximately_equal(actual.z, expected.z)) {
    return true;
  }
  std::cerr << label << " mismatch: (" << actual.x << ", " << actual.y
            << ", " << actual.z << ") != (" << expected.x << ", "
            << expected.y << ", " << expected.z << ")\n";
  return false;
}

void run_image(Device &device, Stream &stream, const ShaderImage &image,
               std::array<luisa::float4, 2u> &floating,
               std::array<luisa::uint4, 2u> &integer,
               luisa::float3 shader_normal = {0.0f, 0.0f, 1.0f}) {
  const auto kernel =
      make_interpreter_kernel(image.node_types_used, shader_normal);
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto word_buffer = device.create_buffer<std::uint32_t>(image.words.size());
  auto floating_buffer = device.create_buffer<luisa::float4>(floating.size());
  auto integer_buffer = device.create_buffer<luisa::uint4>(integer.size());
  stream << word_buffer.copy_from(luisa::span{image.words})
         << shader(word_buffer, floating_buffer, integer_buffer).dispatch(2u)
         << floating_buffer.copy_to(luisa::span{floating})
         << integer_buffer.copy_to(luisa::span{integer}) << synchronize();
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  const auto math_image = compile_dynamic_math();
  const auto mix_image = compile_dynamic_mix();
  const auto color_image = compile_dynamic_color_pipeline();
  const auto combsep_color_image = compile_dynamic_combsep_color_pipeline();
  const auto sepcomb_vector_image = compile_dynamic_sepcomb_vector_pipeline();
  std::vector<ShaderImage> legacy_mix_images;
  legacy_mix_images.reserve(legacy_mix_modes.size());
  for (const auto &[mode, type] : legacy_mix_modes) {
    legacy_mix_images.emplace_back(compile_dynamic_legacy_mix(mode, type));
  }
  std::vector<ShaderImage> modern_color_images;
  modern_color_images.reserve(legacy_mix_modes.size());
  for (auto index = std::size_t{}; index < legacy_mix_modes.size(); ++index) {
    modern_color_images.emplace_back(
        compile_dynamic_modern_mix_color(index));
  }
  std::vector<ShaderImage> modern_data_images;
  modern_data_images.reserve(6u);
  for (auto kind = std::size_t{}; kind < 3u; ++kind) {
    modern_data_images.emplace_back(
        compile_dynamic_modern_mix_data(kind, false));
    modern_data_images.emplace_back(
        compile_dynamic_modern_mix_data(kind, true));
  }
  const auto modern_chain_image = compile_dynamic_modern_mix_chain();

  const auto shape_kernel = make_interpreter_kernel(math_image.node_types_used);
  InterpreterShape shape;
  shape_kernel.function()->function().body()->accept(shape);
  if (shape.loops != 1u || shape.primary_switches != 1u) {
    std::cerr << "Cycles SVM AST must contain exactly one PC loop and one "
                 "primary opcode switch; loops="
              << shape.loops << ", primary switches=" << shape.primary_switches
              << '\n';
    return EXIT_FAILURE;
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();

  std::array<luisa::float4, 2u> floating{};
  std::array<luisa::uint4, 2u> integer{};
  run_image(device, stream, math_image, floating, integer);
  const auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);
  if (!require_float3(floating[0], {0.0f, 0.0f, 0.0f},
                      "front-facing Math emission") ||
      !require_float3(floating[1], {0.42f, 0.94f, 1.66f},
                      "back-facing Math emission") ||
      integer[0].x != ended || integer[1].x != ended ||
      integer[0].y != 21u || integer[1].y != 21u ||
      integer[0].z != device_svm::shader_data_emission ||
      integer[1].z != (device_svm::shader_data_backfacing |
                       device_svm::shader_data_emission)) {
    std::cerr << "dynamic Math Cycles SVM state mismatch on " << backend
              << '\n';
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run_image(device, stream, mix_image, floating, integer);
  if (!require_float3(floating[0], {0.0f, 0.0f, 0.0f},
                      "front-facing Mix emission") ||
      !require_float3(floating[1], {1.02f, 0.096f, 0.036f},
                      "back-facing Mix emission") ||
      !approximately_equal(floating[0].w, 0.75f) ||
      !approximately_equal(floating[1].w, 1.02f) ||
      integer[0].x != ended || integer[1].x != ended ||
      integer[0].y != 32u || integer[1].y != 32u ||
      integer[0].z != 0u ||
      integer[1].z != (device_svm::shader_data_backfacing |
                       device_svm::shader_data_emission)) {
    std::cerr << "dynamic closure jump Cycles SVM state mismatch on "
              << backend << '\n';
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run_image(device, stream, color_image, floating, integer);
  // Frozen from the 4x4 Cycles CPU Emission pass of
  // `svm_color_pipeline`: the left half is front-facing and the right half
  // has its polygon winding reversed. The shader stream itself is separately
  // frozen word-for-word by test_cycles_svm_compiler.cpp.
  if (!require_float3(floating[0],
                      {0.035884645f, 0.071987897f, 0.159804747f},
                      "front-facing color pipeline emission") ||
      !require_float3(floating[1],
                      {0.665142953f, 0.0f, 0.707822502f},
                      "back-facing color pipeline emission") ||
      !approximately_equal(floating[0].w, 0.035884645f) ||
      !approximately_equal(floating[1].w, 0.665142953f) ||
      integer[0].x != ended || integer[1].x != ended ||
      integer[0].y != 49u || integer[1].y != 49u ||
      integer[0].z != device_svm::shader_data_emission ||
      integer[1].z != (device_svm::shader_data_backfacing |
                       device_svm::shader_data_emission)) {
    std::cerr << "dynamic color Cycles SVM state mismatch on " << backend
              << '\n';
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  run_image(device, stream, combsep_color_image, floating, integer);
  // Frozen from the Cycles CPU Emission pass of
  // `svm_combsep_color_pipeline`. The stream is frozen independently in the
  // host compiler regression.
  if (!require_float3(floating[0], {0.0f, 0.180000007f, 0.0f},
                      "front-facing Combine/Separate Color emission") ||
      !require_float3(floating[1], {0.564999998f, 0.383161038f, 0.0f},
                      "back-facing Combine/Separate Color emission") ||
      !approximately_equal(floating[0].w, 0.0f) ||
      !approximately_equal(floating[1].w, 0.564999998f) ||
      integer[0].x != ended || integer[1].x != ended || integer[0].y != 57u ||
      integer[1].y != 57u || integer[0].z != device_svm::shader_data_emission ||
      integer[1].z != (device_svm::shader_data_backfacing |
                       device_svm::shader_data_emission)) {
    std::cerr << "dynamic Combine/Separate Color Cycles SVM state mismatch on "
              << backend << '\n';
    return EXIT_FAILURE;
  }

  floating = {};
  integer = {};
  // Frozen from the Cycles CPU Emission pass of
  // `svm_sepcomb_vector_pipeline`. The object rotation makes all three normal
  // components non-trivial; the host stream is frozen independently by the
  // vector compiler and Blender-import regressions.
  static constexpr auto cycles_probe_normal =
      luisa::float3{-0.191833034f, -0.347542346f, 0.917831361f};
  run_image(device, stream, sepcomb_vector_image, floating, integer,
            cycles_probe_normal);
  static constexpr auto cycles_vector_emission =
      luisa::float3{0.917831361f, -0.191833034f, -0.347542346f};
  if (!require_float3(floating[0], cycles_vector_emission,
                      "front-facing Separate/Combine XYZ emission") ||
      !require_float3(floating[1], cycles_vector_emission,
                      "back-facing Separate/Combine XYZ emission") ||
      !approximately_equal(floating[0].w, cycles_vector_emission.x) ||
      !approximately_equal(floating[1].w, cycles_vector_emission.x) ||
      integer[0].x != ended || integer[1].x != ended ||
      integer[0].y != 39u || integer[1].y != 39u ||
      integer[0].z != device_svm::shader_data_emission ||
      integer[1].z != (device_svm::shader_data_backfacing |
                       device_svm::shader_data_emission)) {
    std::cerr << "dynamic Separate/Combine XYZ Cycles SVM state mismatch on "
              << backend << '\n';
    return EXIT_FAILURE;
  }

  static constexpr std::array legacy_front{
      luisa::float3{0.319499999f, 0.540300012f, 0.802599967f},
      luisa::float3{0.170000002f, 0.540300012f, 0.802599967f},
      luisa::float3{0.162962005f, 0.519876003f, 0.789696991f},
      luisa::float3{0.134153962f, 0.551623821f, 0.876028359f},
      luisa::float3{0.319499999f, 0.629999995f, 0.889999986f},
      luisa::float3{0.326538026f, 0.650424004f, 0.902902961f},
      luisa::float3{0.209514424f, 0.666807771f, 1.000000000f},
      luisa::float3{0.358600020f, 0.685199976f, 1.000000000f},
      luisa::float3{0.195023999f, 0.585748017f, 0.890505970f},
      luisa::float3{0.190769911f, 0.602121234f, 0.890450358f},
      luisa::float3{0.317200005f, 0.510399997f, 0.894599974f},
      luisa::float3{0.280399978f, 0.574800014f, 0.772700012f},
      luisa::float3{0.294476002f, 0.615647972f, 0.798506021f},
      luisa::float3{-0.018600002f, 0.574800014f, 0.772700012f},
      luisa::float3{0.178582922f, 1.088850021f, 1.086672544f},
      luisa::float3{0.335600019f, 0.524200022f, 0.801489592f},
      luisa::float3{0.190812230f, 0.637515485f, 0.889999986f},
      luisa::float3{0.335600019f, 0.545012176f, 0.812613368f},
      luisa::float3{0.166924730f, 0.618603349f, 0.873899996f},
  };
  static constexpr std::array legacy_back{
      luisa::float3{0.585999966f, 0.380400002f, 0.646799982f},
      luisa::float3{0.170000002f, 0.380400002f, 0.646799982f},
      luisa::float3{0.150416002f, 0.323568016f, 0.610895991f},
      luisa::float3{0.061934948f, 0.279595017f, 0.839743555f},
      luisa::float3{0.585999966f, 0.629999995f, 0.889999986f},
      luisa::float3{0.605584025f, 0.686831951f, 0.925903976f},
      luisa::float3{0.357744098f, 0.744328916f, 1.000000000f},
      luisa::float3{0.694800019f, 0.783599973f, 1.000000000f},
      luisa::float3{0.239632010f, 0.506864011f, 0.891407967f},
      luisa::float3{0.227794573f, 0.552424312f, 0.891253114f},
      luisa::float3{0.579599977f, 0.297200024f, 0.902799964f},
      luisa::float3{0.477199972f, 0.476399988f, 0.563600004f},
      luisa::float3{0.516367972f, 0.590063989f, 0.635408044f},
      luisa::float3{-0.354799986f, 0.476399988f, 0.563600004f},
      luisa::float3{0.193882942f, 1.906800032f, 1.437262774f},
      luisa::float3{0.630799949f, 0.335600019f, 0.643710256f},
      luisa::float3{0.227912173f, 0.650912702f, 0.889999986f},
      luisa::float3{0.630799949f, 0.393512189f, 0.674663365f},
      luisa::float3{0.161442712f, 0.598287582f, 0.845200002f},
  };
  const auto legacy_kernel =
      make_interpreter_kernel(legacy_mix_images.front().node_types_used);
  auto legacy_shader =
      device.compile(legacy_kernel, ShaderOption{.enable_cache = false});
  for (auto index = std::size_t{}; index < legacy_mix_images.size(); ++index) {
    const auto &image = legacy_mix_images[index];
    auto word_buffer = device.create_buffer<std::uint32_t>(image.words.size());
    auto floating_buffer = device.create_buffer<luisa::float4>(floating.size());
    auto integer_buffer = device.create_buffer<luisa::uint4>(integer.size());
    floating = {};
    integer = {};
    stream << word_buffer.copy_from(luisa::span{image.words})
           << legacy_shader(word_buffer, floating_buffer, integer_buffer)
                  .dispatch(2u)
           << floating_buffer.copy_to(luisa::span{floating})
           << integer_buffer.copy_to(luisa::span{integer}) << synchronize();
    const auto front_label =
        std::string{"front-facing legacy Mix "} + legacy_mix_modes[index].first;
    const auto back_label =
        std::string{"back-facing legacy Mix "} + legacy_mix_modes[index].first;
    const auto final_offset =
        legacy_mix_modes[index].second == NODE_MIX_ADD ? 41u : 31u;
    if (!require_float3(floating[0], legacy_front[index], front_label) ||
        !require_float3(floating[1], legacy_back[index], back_label) ||
        !approximately_equal(floating[0].w, legacy_front[index].x) ||
        !approximately_equal(floating[1].w, legacy_back[index].x) ||
        integer[0].x != ended || integer[1].x != ended ||
        integer[0].y != final_offset || integer[1].y != final_offset ||
        integer[0].z != device_svm::shader_data_emission ||
        integer[1].z != (device_svm::shader_data_backfacing |
                         device_svm::shader_data_emission)) {
      std::cerr << "dynamic legacy Mix Cycles SVM state mismatch for "
                << legacy_mix_modes[index].first << " on " << backend << '\n';
      return EXIT_FAILURE;
    }
  }

  static constexpr std::array modern_front{
      luisa::float3{0.170000002f, 0.629999995f, 0.889999986f},
      luisa::float3{0.170000002f, 0.727499962f, 0.985000014f},
      luisa::float3{0.170000002f, 0.629999995f, 0.889999986f},
      luisa::float3{0.205741584f, 0.689075649f, 0.902004421f},
      luisa::float3{0.170000002f, 0.629999995f, 0.889999986f},
      luisa::float3{-0.000149965286f, 0.607800007f, 0.875974953f},
      luisa::float3{0.170000002f, 0.629999995f, 0.889999986f},
      luisa::float3{-0.0349999964f, 0.569999993f, 0.762499988f},
      luisa::float3{0.170000002f, 0.629999995f, 0.889999986f},
      luisa::float3{0.147424012f, 0.660302997f, 0.889510453f},
      luisa::float3{0.170000002f, 0.629999995f, 0.889999986f},
      luisa::float3{0.0500000119f, 0.689999998f, 1.01749992f},
      luisa::float3{0.170000002f, 0.629999995f, 0.889999986f},
      luisa::float3{0.375f, 0.689999998f, 1.01749992f},
      luisa::float3{0.170000002f, 0.629999995f, 0.889999986f},
      luisa::float3{0.0f, 0.745000005f, 0.986206889f},
      luisa::float3{0.170000017f, 0.629999936f, 0.889999986f},
      luisa::float3{-0.00999999046f, 0.722378016f, 0.974115849f},
      luisa::float3{0.170000017f, 0.629999936f, 0.889999986f},
  };
  static constexpr std::array modern_back{
      luisa::float3{0.819999993f, 0.240000010f, 0.509999990f},
      luisa::float3{0.170000002f, 0.142500013f, 0.414999992f},
      luisa::float3{0.139400005f, 0.151199996f, 0.453899980f},
      luisa::float3{0.0f, 0.0f, 0.716128945f},
      luisa::float3{0.819999993f, 0.629999995f, 0.889999986f},
      luisa::float3{1.02075005f, 0.740999997f, 0.960124969f},
      luisa::float3{0.944444418f, 0.828947365f, 1.0f},
      luisa::float3{1.19499993f, 0.930000007f, 1.52749991f},
      luisa::float3{0.278800011f, 0.437599987f, 0.892199993f},
      luisa::float3{0.282880008f, 0.478485018f, 0.892447591f},
      luisa::float3{0.810000002f, 0.110000014f, 0.909999967f},
      luisa::float3{0.769999981f, 0.329999983f, 0.252499998f},
      luisa::float3{0.711199999f, 0.567600012f, 0.492200017f},
      luisa::float3{-0.854999959f, 0.329999983f, 0.252499998f},
      luisa::float3{0.207317084f, 2.625f, 1.74509799f},
      luisa::float3{1.0f, 0.0550000072f, 0.408965319f},
      luisa::float3{0.260487825f, 0.662676096f, 0.889999986f},
      luisa::float3{1.06999993f, 0.168109775f, 0.469420612f},
      luisa::float3{0.156629220f, 0.580449402f, 0.819999993f},
  };
  static constexpr std::array data_front{
      luisa::float3{0.0155000007f, 0.0285f, 0.0414999984f},
      luisa::float3{0.0620000027f, 0.114f, 0.165999994f},
      luisa::float3{-0.099999994f, 0.899999976f, -0.400000006f},
      luisa::float3{0.100000001f, 0.699999988f, -0.200000003f},
      luisa::float3{0.100000001f, 0.699999988f, 0.600000024f},
      luisa::float3{0.100000001f, 0.699999988f, 0.600000024f},
  };
  static constexpr std::array data_back{
      luisa::float3{0.294499993f, 0.541499972f, 0.788499951f},
      luisa::float3{0.248000011f, 0.456f, 0.663999975f},
      luisa::float3{1.10000002f, -0.300000012f, 0.800000012f},
      luisa::float3{0.899999976f, -0.100000001f, 0.600000024f},
      luisa::float3{0.100000001f, 0.699999988f, 0.600000024f},
      luisa::float3{0.100000001f, 0.699999988f, 0.600000024f},
  };

  auto modern_mask = std::array<bool, NODE_NUM>{};
  for (const auto &image : modern_color_images) {
    for (auto type = std::size_t{}; type < modern_mask.size(); ++type) {
      modern_mask[type] = modern_mask[type] || image.node_types_used[type];
    }
  }
  for (const auto &image : modern_data_images) {
    for (auto type = std::size_t{}; type < modern_mask.size(); ++type) {
      modern_mask[type] = modern_mask[type] || image.node_types_used[type];
    }
  }
  for (auto type = std::size_t{}; type < modern_mask.size(); ++type) {
    modern_mask[type] =
        modern_mask[type] || modern_chain_image.node_types_used[type];
  }
  const auto modern_kernel = make_interpreter_kernel(modern_mask);
  auto modern_shader =
      device.compile(modern_kernel, ShaderOption{.enable_cache = false});
  auto run_modern = [&](const ShaderImage &image) {
    auto word_buffer = device.create_buffer<std::uint32_t>(image.words.size());
    auto floating_buffer = device.create_buffer<luisa::float4>(floating.size());
    auto integer_buffer = device.create_buffer<luisa::uint4>(integer.size());
    floating = {};
    integer = {};
    stream << word_buffer.copy_from(luisa::span{image.words})
           << modern_shader(word_buffer, floating_buffer, integer_buffer)
                  .dispatch(2u)
           << floating_buffer.copy_to(luisa::span{floating})
           << integer_buffer.copy_to(luisa::span{integer}) << synchronize();
  };

  for (auto index = std::size_t{}; index < modern_color_images.size();
       ++index) {
    run_modern(modern_color_images[index]);
    const auto front_label = std::string{"front-facing modern MixColor "} +
                             legacy_mix_modes[index].first;
    const auto back_label = std::string{"back-facing modern MixColor "} +
                            legacy_mix_modes[index].first;
    if (!require_float3(floating[0], modern_front[index], front_label) ||
        !require_float3(floating[1], modern_back[index], back_label) ||
        !approximately_equal(floating[0].w, modern_front[index].x) ||
        !approximately_equal(floating[1].w, modern_back[index].x) ||
        integer[0].x != ended || integer[1].x != ended ||
        integer[0].y != 31u || integer[1].y != 31u ||
        integer[0].z != device_svm::shader_data_emission ||
        integer[1].z != (device_svm::shader_data_backfacing |
                         device_svm::shader_data_emission)) {
      std::cerr << "dynamic modern MixColor mismatch for "
                << legacy_mix_modes[index].first << " on " << backend << '\n';
      return EXIT_FAILURE;
    }
  }
  for (auto index = std::size_t{}; index < modern_data_images.size(); ++index) {
    run_modern(modern_data_images[index]);
    const auto final_offset = index < 2u ? 26u : index < 4u ? 30u : 26u;
    const auto label = std::string{"modern typed Mix "} +
                       std::to_string(index);
    if (!require_float3(floating[0], data_front[index], label + " front") ||
        !require_float3(floating[1], data_back[index], label + " back") ||
        !approximately_equal(floating[0].w, data_front[index].x) ||
        !approximately_equal(floating[1].w, data_back[index].x) ||
        integer[0].x != ended || integer[1].x != ended ||
        integer[0].y != final_offset || integer[1].y != final_offset ||
        integer[0].z != device_svm::shader_data_emission ||
        integer[1].z != (device_svm::shader_data_backfacing |
                         device_svm::shader_data_emission)) {
      std::cerr << "dynamic modern typed Mix mismatch for case " << index
                << " on " << backend << '\n';
      return EXIT_FAILURE;
    }
  }

  run_modern(modern_chain_image);
  static constexpr auto chain_front =
      luisa::float3{0.259999990f, 0.540000021f, 0.0f};
  static constexpr auto chain_back =
      luisa::float3{0.635999918f, 0.0f, 0.088000007f};
  if (!require_float3(floating[0], chain_front,
                      "front-facing modern Mix chain") ||
      !require_float3(floating[1], chain_back,
                      "back-facing modern Mix chain") ||
      !approximately_equal(floating[0].w, chain_front.x) ||
      !approximately_equal(floating[1].w, chain_back.x) ||
      integer[0].x != ended || integer[1].x != ended ||
      integer[0].y != 53u || integer[1].y != 53u ||
      integer[0].z != device_svm::shader_data_emission ||
      integer[1].z != (device_svm::shader_data_backfacing |
                       device_svm::shader_data_emission)) {
    std::cerr << "dynamic modern Mix chain mismatch on " << backend << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

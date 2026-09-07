/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_volume_nodes.h"
#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <optional>
#include <string>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {
template <typename T>
[[nodiscard]] std::optional<T> literal(const GraphInput *input,
                                       contract::SocketType type) noexcept {
  if (input == nullptr || input->link != nullptr || !input->value ||
      input->value->type != type) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<T>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view>
string_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::string) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<std::string>(&iter->second.value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ClosureType>
volume_phase(const GraphNode *node) noexcept {
  const auto phase = string_property(node, "Phase");
  if (phase == "HENYEY_GREENSTEIN") {
    return CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID;
  }
  if (phase == "FOURNIER_FORAND") {
    return CLOSURE_VOLUME_FOURNIER_FORAND_ID;
  }
  if (phase == "DRAINE") {
    return CLOSURE_VOLUME_DRAINE_ID;
  }
  if (phase == "RAYLEIGH") {
    return CLOSURE_VOLUME_RAYLEIGH_ID;
  }
  if (phase == "MIE") {
    return CLOSURE_VOLUME_MIE_ID;
  }
  return std::nullopt;
}

class VolumeClosureNode : public GraphNode {
protected:
  [[nodiscard]] bool
  compile_weight(SVMCompiler &compiler, std::string_view name = "Color",
                 contract::SocketType type = contract::SocketType::color) {
    auto *weight = input(name);
    if (weight == nullptr) {
      compiler.fail("Cycles Volume weight input is absent");
      return false;
    }
    if (weight->link != nullptr) {
      compiler.add_node(
          this, NODE_CLOSURE_WEIGHT,
          SVMNodeClosureWeight{.weight_offset = compiler.input_link(name),
                               ._pad = {0u, 0u, 0u}});
    } else {
      const auto value = literal<Vec3f>(weight, type);
      if (!value) {
        compiler.fail("Cycles Volume weight input is ill typed");
        return false;
      }
      compiler.add_node(
          this, NODE_CLOSURE_SET_WEIGHT,
          SVMNodeClosureSetWeight{
              .rgb = packed_float3{value->x, value->y, value->z}});
    }
    return true;
  }

  void compile_volume(SVMCompiler &compiler, ClosureType closure,
                      std::string_view density_name,
                      std::string_view param1_name = {},
                      std::string_view param2_name = {}) {
    if (!compile_weight(compiler)) {
      return;
    }
    compiler.add_node(
        this, NODE_CLOSURE_VOLUME,
        SVMNodeClosureVolume{
            .closure_type = closure,
            .density = density_name.empty()
                           ? SVMInputFloat{0u}
                           : compiler.input_float(density_name),
            .param1 = param1_name.empty() ? SVMInputFloat{0u}
                                          : compiler.input_float(param1_name),
            .param_extra = param2_name.empty()
                               ? SVMInputFloat{0u}
                               : compiler.input_float(param2_name),
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}});
  }

public:
  [[nodiscard]] bool equals(const GraphNode &) const noexcept override {
    // Cycles VolumeNode::equals: these closures are not graph-deduplicated.
    return false;
  }

  [[nodiscard]] bool has_volume_support() const noexcept override {
    return true;
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }

  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return GraphNode::get_feature() | kernel_feature_node_volume;
  }

  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_CLOSURE_VOLUME;
  }
};

class AbsorptionVolumeNode final : public VolumeClosureNode {
public:
  [[nodiscard]] ClosureType get_closure_type() const noexcept override {
    return CLOSURE_VOLUME_ABSORPTION_ID;
  }

  void compile(SVMCompiler &compiler) override {
    compile_volume(compiler, CLOSURE_VOLUME_ABSORPTION_ID, "Density");
  }
};

class ScatterVolumeNode final : public VolumeClosureNode {
public:
  [[nodiscard]] ClosureType get_closure_type() const noexcept override {
    return volume_phase(this).value_or(CLOSURE_NONE_ID);
  }

  void compile(SVMCompiler &compiler) override {
    const auto phase = volume_phase(this);
    if (!phase) {
      compiler.fail("Cycles Scatter Volume phase is not migrated exactly");
      return;
    }
    switch (*phase) {
    case CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID:
      compile_volume(compiler, *phase, "Density", "Anisotropy");
      break;
    case CLOSURE_VOLUME_FOURNIER_FORAND_ID:
      compile_volume(compiler, *phase, "Density", "IOR", "Backscatter");
      break;
    case CLOSURE_VOLUME_DRAINE_ID:
      compile_volume(compiler, *phase, "Density", "Anisotropy", "Alpha");
      break;
    case CLOSURE_VOLUME_RAYLEIGH_ID:
      compile_volume(compiler, *phase, "Density");
      break;
    case CLOSURE_VOLUME_MIE_ID:
      compile_volume(compiler, *phase, "Density", "Diameter");
      break;
    default:
      compiler.fail("Cycles Scatter Volume phase is not a physical phase");
      break;
    }
  }
};

class VolumeCoefficientsNode final : public VolumeClosureNode {
public:
  [[nodiscard]] ClosureType get_closure_type() const noexcept override {
    return volume_phase(this).value_or(CLOSURE_NONE_ID);
  }

  void compile(SVMCompiler &compiler) override {
    const auto phase = volume_phase(this);
    if (!phase) {
      compiler.fail("Cycles Volume Coefficients phase is not migrated exactly");
      return;
    }
    std::string_view param1, param2;
    switch (*phase) {
    case CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID:
      param1 = "Anisotropy";
      break;
    case CLOSURE_VOLUME_FOURNIER_FORAND_ID:
      param1 = "IOR";
      param2 = "Backscatter";
      break;
    case CLOSURE_VOLUME_DRAINE_ID:
      param1 = "Anisotropy";
      param2 = "Alpha";
      break;
    case CLOSURE_VOLUME_MIE_ID:
      param1 = "Diameter";
      break;
    case CLOSURE_VOLUME_RAYLEIGH_ID:
      break;
    default:
      compiler.fail("Cycles Volume Coefficients phase is not a physical phase");
      return;
    }
    if (!compile_weight(compiler, "ScatterCoefficients",
                        contract::SocketType::vector)) {
      return;
    }
    compiler.add_node(
        this, NODE_VOLUME_COEFFICIENTS,
        SVMNodeVolumeCoefficients{
            .closure_type = *phase,
            .absorption_coeffs =
                compiler.input_float3("AbsorptionCoefficients"),
            .emission_coeffs = compiler.input_float3("EmissionCoefficients"),
            .param1 = param1.empty() ? SVMInputFloat{0u}
                                     : compiler.input_float(param1),
            .param_extra = param2.empty() ? SVMInputFloat{0u}
                                          : compiler.input_float(param2),
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}});
  }
};

class PrincipledVolumeNode final : public VolumeClosureNode {
public:
  [[nodiscard]] ClosureType get_closure_type() const noexcept override {
    return CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID;
  }

  [[nodiscard]] bool has_attribute_dependency() const noexcept override {
    return true;
  }

  void attributes(const GraphAttributeContext &context,
                  AttributeRequestSet &requests) const override {
    if (context.has_volume) {
      const auto positive_or_linked = [&](std::string_view name) {
        const auto *socket = input(name);
        const auto value =
            literal<float>(socket, contract::SocketType::floating);
        return socket != nullptr &&
               (socket->link != nullptr || (value && *value > 0.0f));
      };
      if (positive_or_linked("Density")) {
        requests.add_standard(
            string_property(this, "DensityAttribute").value_or("density"));
        requests.add_standard(
            string_property(this, "ColorAttribute").value_or(""));
      }
      if (positive_or_linked("BlackbodyIntensity")) {
        requests.add_standard(string_property(this, "TemperatureAttribute")
                                  .value_or("temperature"));
      }
      requests.add(ATTR_STD_GENERATED_TRANSFORM);
    }
    GraphNode::attributes(context, requests);
  }

  void compile(SVMCompiler &compiler) override {
    if (!compile_weight(compiler)) {
      return;
    }
    compiler.add_node(
        this, NODE_PRINCIPLED_VOLUME,
        SVMNodePrincipledVolume{
            .absorption_color = compiler.input_float3("AbsorptionColor"),
            .emission_color = compiler.input_float3("EmissionColor"),
            .blackbody_tint = compiler.input_float3("BlackbodyTint"),
            .density = compiler.input_float("Density"),
            .anisotropy = compiler.input_float("Anisotropy"),
            .emission = compiler.input_float("EmissionStrength"),
            .blackbody = compiler.input_float("BlackbodyIntensity"),
            .temperature = compiler.input_float("Temperature"),
            .attr_density = static_cast<int>(compiler.attribute_standard(
                string_property(this, "DensityAttribute").value_or("density"))),
            .attr_color = static_cast<int>(compiler.attribute_standard(
                string_property(this, "ColorAttribute").value_or(""))),
            .attr_temperature = static_cast<int>(compiler.attribute_standard(
                string_property(this, "TemperatureAttribute")
                    .value_or("temperature"))),
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}});
  }
};
} // namespace

std::unique_ptr<GraphNode> make_volume_graph_node(std::string_view type) {
  if (type == node_type::volume_absorption) {
    return std::make_unique<AbsorptionVolumeNode>();
  }
  if (type == node_type::volume_scatter) {
    return std::make_unique<ScatterVolumeNode>();
  }
  if (type == node_type::volume_coefficients) {
    return std::make_unique<VolumeCoefficientsNode>();
  }
  if (type == node_type::principled_volume) {
    return std::make_unique<PrincipledVolumeNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm

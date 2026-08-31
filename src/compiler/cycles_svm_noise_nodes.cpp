/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_noise_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"
#include "cycles_svm_host_hash.h"
#include "cycles_svm_mapping_nodes.h"

#include <psycles/compiler/core_nodes.h>

#include <optional>
#include <string>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

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

[[nodiscard]] std::optional<bool>
boolean_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::boolean) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<bool>(&iter->second.value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::uint32_t>
dimensions_property(const GraphNode *node) noexcept {
  const auto iter = node->properties.find("Dimensions");
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::unsigned_integer) {
    return std::nullopt;
  }
  const auto *value = std::get_if<std::uint64_t>(&iter->second.value);
  if (value == nullptr || *value < 1u || *value > 4u) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(*value);
}

[[nodiscard]] std::optional<NodeNoiseType>
noise_type(std::string_view name) noexcept {
  if (name == "MULTIFRACTAL") {
    return NODE_NOISE_MULTIFRACTAL;
  }
  if (name == "FBM") {
    return NODE_NOISE_FBM;
  }
  if (name == "HYBRID_MULTIFRACTAL") {
    return NODE_NOISE_HYBRID_MULTIFRACTAL;
  }
  if (name == "RIDGED_MULTIFRACTAL") {
    return NODE_NOISE_RIDGED_MULTIFRACTAL;
  }
  if (name == "HETERO_TERRAIN") {
    return NODE_NOISE_HETERO_TERRAIN;
  }
  return std::nullopt;
}

class NoiseTextureNode final : public TextureNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_NOISE;
  }

  void compile(SVMCompiler &compiler) override {
    const auto dimensions = dimensions_property(this);
    const auto type_name = string_property(this, "NoiseType");
    const auto type = type_name ? noise_type(*type_name) : std::nullopt;
    const auto normalize = boolean_property(this, "Normalize");
    if (!dimensions || !type || !normalize) {
      compiler.fail("Cycles Noise Texture properties are invalid");
      return;
    }

    auto *vector_in = input("Vector");
    const auto vector_offset =
        tex_mapping.compile_begin(compiler, vector_in, nullptr);
    compiler.add_node(
        this, NODE_TEX_NOISE,
        SVMNodeTexNoise{
            .dimensions = *dimensions,
            .noise_type = *type,
            .normalize = static_cast<std::uint32_t>(*normalize),
            .w = compiler.input_float("W"),
            .scale = compiler.input_float("Scale"),
            .detail = compiler.input_float("Detail"),
            .roughness = compiler.input_float("Roughness"),
            .lacunarity = compiler.input_float("Lacunarity"),
            .offset = compiler.input_float("Offset"),
            .gain = compiler.input_float("Gain"),
            .distortion = compiler.input_float("Distortion"),
            .vector = vector_offset,
            .value_offset = compiler.output("Factor"),
            .color_offset = compiler.output("Color"),
            ._pad = {0u}});
    tex_mapping.compile_end(compiler, vector_in, vector_offset);
  }
};

class WhiteNoiseTextureNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_WHITE_NOISE;
  }

  void constant_fold(const ConstantFolder &folder) override {
    const auto dimensions = dimensions_property(this);
    auto *vector_input = input("Vector");
    auto *w_input = input("W");
    if (!dimensions || vector_input == nullptr || w_input == nullptr) {
      return;
    }

    auto vector = Vec3f{};
    auto w = 0.0f;

    // Blender 5.2.1 inline_shader_node_tree constant-folds a function node
    // exactly when every available input is primitive. White Noise's Vector
    // socket is unavailable in 1D. In 2D/3D/4D its unlinked declaration uses
    // NODE_DEFAULT_INPUT_POSITION_FIELD, which is deliberately non-primitive;
    // only an upstream fold marks it constant_folded_in. This predicate is
    // therefore the source condition, not a dimension-specific shortcut.
    if (*dimensions == 1u) {
      if (w_input->link != nullptr || !w_input->value) {
        return;
      }
      const auto *value = std::get_if<float>(&w_input->value->value);
      if (value == nullptr) {
        return;
      }
      w = *value;
    } else {
      if (vector_input->link != nullptr || !vector_input->value ||
          !vector_input->constant_folded_in) {
        return;
      }
      const auto *value =
          std::get_if<Vec3f>(&vector_input->value->value);
      if (value == nullptr) {
        return;
      }
      vector = *value;
      if (*dimensions == 4u) {
        if (w_input->link != nullptr || !w_input->value) {
          return;
        }
        const auto *w_value =
            std::get_if<float>(&w_input->value->value);
        if (w_value == nullptr) {
          return;
        }
        w = *w_value;
      }
    }

    if (folder.output->name == "Value") {
      folder.make_constant(
          host_hash::evaluate_value(vector, w, *dimensions));
    } else if (folder.output->name == "Color") {
      folder.make_constant(
          host_hash::evaluate_color(vector, w, *dimensions));
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto dimensions = dimensions_property(this);
    if (!dimensions) {
      compiler.fail("Cycles White Noise Texture dimensions are invalid");
      return;
    }
    compiler.add_node(
        this, NODE_TEX_WHITE_NOISE,
        SVMNodeTexWhiteNoise{
            .dimensions = *dimensions,
            .vector = compiler.input_float3("Vector"),
            .w = compiler.input_float("W"),
            .value_offset = compiler.output("Value"),
            .color_offset = compiler.output("Color"),
            ._pad = {0u, 0u}});
  }
};

} // namespace

std::unique_ptr<GraphNode> make_noise_graph_node(std::string_view type) {
  if (type == node_type::noise_texture) {
    return std::make_unique<NoiseTextureNode>();
  }
  if (type == node_type::white_noise_texture) {
    return std::make_unique<WhiteNoiseTextureNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm

/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_procedural_texture_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_mapping_nodes.h"

#include <psycles/compiler/core_nodes.h>

#include <cstdint>
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

[[nodiscard]] std::optional<float>
floating_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::floating) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<float>(&iter->second.value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::uint64_t>
unsigned_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::unsigned_integer) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<std::uint64_t>(&iter->second.value)) {
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
  const auto dimensions = unsigned_property(node, "Dimensions");
  if (!dimensions || *dimensions < 1u || *dimensions > 4u) {
    return std::nullopt;
  }
  return static_cast<std::uint32_t>(*dimensions);
}

[[nodiscard]] std::optional<NodeWaveType>
wave_type(std::string_view value) noexcept {
  if (value == "BANDS") {
    return NODE_WAVE_BANDS;
  }
  if (value == "RINGS") {
    return NODE_WAVE_RINGS;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeWaveBandsDirection>
bands_direction(std::string_view value) noexcept {
  if (value == "X") {
    return NODE_WAVE_BANDS_DIRECTION_X;
  }
  if (value == "Y") {
    return NODE_WAVE_BANDS_DIRECTION_Y;
  }
  if (value == "Z") {
    return NODE_WAVE_BANDS_DIRECTION_Z;
  }
  if (value == "DIAGONAL") {
    return NODE_WAVE_BANDS_DIRECTION_DIAGONAL;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeWaveRingsDirection>
rings_direction(std::string_view value) noexcept {
  if (value == "X") {
    return NODE_WAVE_RINGS_DIRECTION_X;
  }
  if (value == "Y") {
    return NODE_WAVE_RINGS_DIRECTION_Y;
  }
  if (value == "Z") {
    return NODE_WAVE_RINGS_DIRECTION_Z;
  }
  if (value == "SPHERICAL") {
    return NODE_WAVE_RINGS_DIRECTION_SPHERICAL;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeWaveProfile>
wave_profile(std::string_view value) noexcept {
  if (value == "SIN") {
    return NODE_WAVE_PROFILE_SIN;
  }
  if (value == "SAW") {
    return NODE_WAVE_PROFILE_SAW;
  }
  if (value == "TRI") {
    return NODE_WAVE_PROFILE_TRI;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeVoronoiFeature>
voronoi_feature(std::string_view value) noexcept {
  if (value == "F1") {
    return NODE_VORONOI_F1;
  }
  if (value == "F2") {
    return NODE_VORONOI_F2;
  }
  if (value == "SMOOTH_F1") {
    return NODE_VORONOI_SMOOTH_F1;
  }
  if (value == "DISTANCE_TO_EDGE") {
    return NODE_VORONOI_DISTANCE_TO_EDGE;
  }
  if (value == "N_SPHERE_RADIUS") {
    return NODE_VORONOI_N_SPHERE_RADIUS;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeVoronoiDistanceMetric>
voronoi_metric(std::string_view value) noexcept {
  if (value == "EUCLIDEAN") {
    return NODE_VORONOI_EUCLIDEAN;
  }
  if (value == "MANHATTAN") {
    return NODE_VORONOI_MANHATTAN;
  }
  if (value == "CHEBYCHEV") {
    return NODE_VORONOI_CHEBYCHEV;
  }
  if (value == "MINKOWSKI") {
    return NODE_VORONOI_MINKOWSKI;
  }
  return std::nullopt;
}

class WaveTextureNode final : public TextureNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_WAVE;
  }

  void compile(SVMCompiler &compiler) override {
    const auto type_name = string_property(this, "WaveType");
    const auto bands_name = string_property(this, "BandsDirection");
    const auto rings_name = string_property(this, "RingsDirection");
    const auto profile_name = string_property(this, "Profile");
    const auto type = type_name ? wave_type(*type_name) : std::nullopt;
    const auto bands = bands_name ? bands_direction(*bands_name) : std::nullopt;
    const auto rings = rings_name ? rings_direction(*rings_name) : std::nullopt;
    const auto profile =
        profile_name ? wave_profile(*profile_name) : std::nullopt;
    if (!type || !bands || !rings || !profile) {
      compiler.fail("Cycles Wave Texture properties are invalid");
      return;
    }

    auto *vector_in = input("Vector");
    const auto vector_offset =
        tex_mapping.compile_begin(compiler, vector_in, nullptr);
    compiler.add_node(
        this, NODE_TEX_WAVE,
        SVMNodeTexWave{.wave_type = *type,
                       .bands_direction = *bands,
                       .rings_direction = *rings,
                       .profile = *profile,
                       .scale = compiler.input_float("Scale"),
                       .distortion = compiler.input_float("Distortion"),
                       .detail = compiler.input_float("Detail"),
                       .dscale = compiler.input_float("DetailScale"),
                       .droughness = compiler.input_float("DetailRoughness"),
                       .phase = compiler.input_float("PhaseOffset"),
                       .co = vector_offset,
                       .color_offset = compiler.output("Color"),
                       .fac_offset = compiler.output("Factor"),
                       ._pad = {0u}});
    tex_mapping.compile_end(compiler, vector_in, vector_offset);
  }
};

class MagicTextureNode final : public TextureNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_MAGIC;
  }

  void compile(SVMCompiler &compiler) override {
    const auto depth = unsigned_property(this, "Depth");
    if (!depth) {
      compiler.fail("Cycles Magic Texture depth is invalid");
      return;
    }
    auto *vector_in = input("Vector");
    const auto vector_offset =
        tex_mapping.compile_begin(compiler, vector_in, nullptr);
    compiler.add_node(
        this, NODE_TEX_MAGIC,
        SVMNodeTexMagic{
            .scale = compiler.input_float("Scale"),
            .distortion = compiler.input_float("Distortion"),
            // Exact Cycles 5.2.1 host payload conversion. The kernel's
            // nested depth predicates naturally saturate behavior at ten.
            .depth = static_cast<std::uint8_t>(*depth),
            .co = vector_offset,
            .color_offset = compiler.output("Color"),
            .fac_offset = compiler.output("Factor")});
    tex_mapping.compile_end(compiler, vector_in, vector_offset);
  }
};

class CheckerTextureNode final : public TextureNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_CHECKER;
  }

  void compile(SVMCompiler &compiler) override {
    auto *vector_in = input("Vector");
    const auto vector_offset =
        tex_mapping.compile_begin(compiler, vector_in, nullptr);
    compiler.add_node(
        this, NODE_TEX_CHECKER,
        SVMNodeTexChecker{.color1 = compiler.input_float3("Color1"),
                          .color2 = compiler.input_float3("Color2"),
                          .scale = compiler.input_float("Scale"),
                          .co = vector_offset,
                          .color_offset = compiler.output("Color"),
                          .fac_offset = compiler.output("Factor"),
                          ._pad = {0u}});
    tex_mapping.compile_end(compiler, vector_in, vector_offset);
  }
};

class BrickTextureNode final : public TextureNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_BRICK;
  }

  void compile(SVMCompiler &compiler) override {
    const auto offset_amount = floating_property(this, "OffsetAmount");
    const auto squash_amount = floating_property(this, "SquashAmount");
    const auto offset_frequency = unsigned_property(this, "OffsetFrequency");
    const auto squash_frequency = unsigned_property(this, "SquashFrequency");
    if (!offset_amount || !squash_amount || !offset_frequency ||
        !squash_frequency) {
      compiler.fail("Cycles Brick Texture properties are invalid");
      return;
    }

    auto *vector_in = input("Vector");
    const auto vector_offset =
        tex_mapping.compile_begin(compiler, vector_in, nullptr);
    compiler.add_node(
        this, NODE_TEX_BRICK,
        SVMNodeTexBrick{
            .color1 = compiler.input_float3("Color1"),
            .color2 = compiler.input_float3("Color2"),
            .mortar = compiler.input_float3("Mortar"),
            .scale = compiler.input_float("Scale"),
            .mortar_size = compiler.input_float("MortarSize"),
            .bias = compiler.input_float("Bias"),
            .brick_width = compiler.input_float("BrickWidth"),
            .row_height = compiler.input_float("RowHeight"),
            .mortar_smooth = compiler.input_float("MortarSmooth"),
            .offset_amount = *offset_amount,
            .squash_amount = *squash_amount,
            .offset_frequency = static_cast<std::uint8_t>(*offset_frequency),
            .squash_frequency = static_cast<std::uint8_t>(*squash_frequency),
            .co = vector_offset,
            .color_offset = compiler.output("Color"),
            .fac_offset = compiler.output("Factor"),
            ._pad = {0u, 0u, 0u}});
    tex_mapping.compile_end(compiler, vector_in, vector_offset);
  }
};

class VoronoiTextureNode final : public TextureNode {
private:
  [[nodiscard]] std::optional<NodeVoronoiFeature>
  feature() const noexcept {
    const auto name = string_property(this, "Feature");
    return name ? voronoi_feature(*name) : std::nullopt;
  }

public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_VORONOI;
  }

  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    const auto dimensions = dimensions_property(this);
    const auto selected_feature = feature();
    auto result = GraphNode::get_feature();
    // Cycles 5.2.1 specializes the extra Voronoi implementation into the
    // kernel exactly for 4D and multidimensional Smooth F1 nodes.
    if (dimensions && selected_feature &&
        (*dimensions == 4u ||
         (*dimensions >= 2u &&
          *selected_feature == NODE_VORONOI_SMOOTH_F1))) {
      result |= kernel_feature_node_voronoi_extra;
    }
    return result;
  }

  void compile(SVMCompiler &compiler) override {
    const auto dimensions = dimensions_property(this);
    const auto selected_feature = feature();
    const auto metric_name = string_property(this, "DistanceMetric");
    const auto metric = metric_name ? voronoi_metric(*metric_name)
                                    : std::nullopt;
    const auto normalize = boolean_property(this, "Normalize");
    if (!dimensions || !selected_feature || !metric || !normalize) {
      compiler.fail("Cycles Voronoi Texture properties are invalid");
      return;
    }

    auto *vector_in = input("Vector");
    const auto vector_offset =
        tex_mapping.compile_begin(compiler, vector_in, nullptr);
    compiler.add_node(
        this, NODE_TEX_VORONOI,
        SVMNodeTexVoronoi{
            .dimensions = *dimensions,
            .feature = *selected_feature,
            .metric = *metric,
            .w = compiler.input_float("W"),
            .scale = compiler.input_float("Scale"),
            .detail = compiler.input_float("Detail"),
            .roughness = compiler.input_float("Roughness"),
            .lacunarity = compiler.input_float("Lacunarity"),
            .smoothness = compiler.input_float("Smoothness"),
            .exponent = compiler.input_float("Exponent"),
            .randomness = compiler.input_float("Randomness"),
            .normalize = static_cast<std::uint8_t>(*normalize),
            .coord = vector_offset,
            .distance_offset = compiler.output("Distance"),
            .color_offset = compiler.output("Color"),
            .position_offset = compiler.output("Position"),
            .w_out_offset = compiler.output("W"),
            .radius_offset = compiler.output("Radius"),
            ._pad = {0u}});
    tex_mapping.compile_end(compiler, vector_in, vector_offset);
  }
};

} // namespace

std::unique_ptr<GraphNode>
make_procedural_texture_graph_node(std::string_view type) {
  if (type == node_type::wave_texture) {
    return std::make_unique<WaveTextureNode>();
  }
  if (type == node_type::magic_texture) {
    return std::make_unique<MagicTextureNode>();
  }
  if (type == node_type::checker_texture) {
    return std::make_unique<CheckerTextureNode>();
  }
  if (type == node_type::brick_texture) {
    return std::make_unique<BrickTextureNode>();
  }
  if (type == node_type::voronoi_texture) {
    return std::make_unique<VoronoiTextureNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm

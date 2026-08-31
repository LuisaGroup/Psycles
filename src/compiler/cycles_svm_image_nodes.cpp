/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_image_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <limits>
#include <optional>
#include <string_view>
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

[[nodiscard]] std::optional<float>
float_property(const GraphNode *node, std::string_view name) noexcept {
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

[[nodiscard]] std::optional<ImageInterpolation>
image_interpolation(std::string_view name) noexcept {
  if (name == "Linear" || name == "LINEAR" || name == "linear") {
    return ImageInterpolation::linear;
  }
  if (name == "Closest" || name == "CLOSEST" || name == "closest") {
    return ImageInterpolation::closest;
  }
  if (name == "Cubic" || name == "CUBIC" || name == "cubic") {
    return ImageInterpolation::cubic;
  }
  if (name == "Smart" || name == "SMART" || name == "smart") {
    return ImageInterpolation::smart;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ImageExtension>
image_extension(std::string_view name) noexcept {
  if (name == "REPEAT" || name == "periodic" || name == "repeat") {
    return ImageExtension::repeat;
  }
  if (name == "EXTEND" || name == "clamp" || name == "extend") {
    return ImageExtension::extend;
  }
  if (name == "CLIP" || name == "black" || name == "clip") {
    return ImageExtension::clip;
  }
  if (name == "MIRROR" || name == "mirror") {
    return ImageExtension::mirror;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeImageProjection>
image_projection(std::string_view name) noexcept {
  if (name == "FLAT" || name == "flat") {
    return NODE_IMAGE_PROJ_FLAT;
  }
  if (name == "BOX" || name == "box") {
    return NODE_IMAGE_PROJ_BOX;
  }
  if (name == "SPHERE" || name == "sphere") {
    return NODE_IMAGE_PROJ_SPHERE;
  }
  if (name == "TUBE" || name == "tube") {
    return NODE_IMAGE_PROJ_TUBE;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeEnvironmentProjection>
environment_projection(std::string_view name) noexcept {
  if (name == "EQUIRECTANGULAR" || name == "equirectangular") {
    return NODE_ENVIRONMENT_EQUIRECTANGULAR;
  }
  if (name == "MIRROR_BALL" || name == "mirror_ball") {
    return NODE_ENVIRONMENT_MIRROR_BALL;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::uint8_t>
image_flags(const GraphNode *node, bool include_alpha) noexcept {
  const auto color_space = string_property(node, "ColorSpace");
  const auto unassociate_alpha =
      boolean_property(node, "UnassociateAlpha");
  if (!color_space || (include_alpha && !unassociate_alpha)) {
    return std::nullopt;
  }
  auto flags = std::uint8_t{};
  if (*color_space == "sRGB") {
    flags |= NODE_IMAGE_COMPRESS_AS_SRGB;
  } else if (*color_space != "Non-Color" && *color_space != "Linear") {
    return std::nullopt;
  }
  if (include_alpha && *unassociate_alpha) {
    const auto *alpha = node->output("Alpha");
    if (alpha != nullptr && !alpha->links.empty()) {
      flags |= NODE_IMAGE_ALPHA_UNASSOCIATE;
    }
  }
  return flags;
}

class ImageTextureNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    const auto projection = string_property(this, "Projection");
    return projection && image_projection(*projection) == NODE_IMAGE_PROJ_BOX
               ? NODE_TEX_IMAGE_BOX
               : NODE_TEX_IMAGE;
  }

  void compile(SVMCompiler &compiler) override {
    const auto resource = unsigned_property(this, "Image");
    const auto interpolation_name = string_property(this, "Interpolation");
    const auto extension_name = string_property(this, "Extension");
    const auto projection_name = string_property(this, "Projection");
    const auto projection_blend = float_property(this, "ProjectionBlend");
    const auto flags = image_flags(this, true);
    const auto interpolation = interpolation_name
                                   ? image_interpolation(*interpolation_name)
                                   : std::nullopt;
    const auto extension = extension_name
                               ? image_extension(*extension_name)
                               : std::nullopt;
    const auto projection = projection_name
                                ? image_projection(*projection_name)
                                : std::nullopt;
    if (!resource || !interpolation || !extension || !projection ||
        !projection_blend || !flags) {
      compiler.fail("Cycles Image Texture properties are invalid");
      return;
    }
    const auto image_id =
        compiler.image(*resource, *interpolation, *extension);
    if (image_id == std::numeric_limits<std::int32_t>::max()) {
      compiler.fail("Cycles SVM image handle table overflow");
      return;
    }
    if (*projection == NODE_IMAGE_PROJ_BOX) {
      compiler.add_node(
          this, NODE_TEX_IMAGE_BOX,
          SVMNodeTexImageBox{
              .id = image_id,
              .blend = *projection_blend,
              .flags = *flags,
              .co = compiler.input_stack("Vector"),
              .out_offset = compiler.output("Color"),
              .alpha_offset = compiler.output("Alpha")});
      return;
    }
    compiler.add_node(
        this, NODE_TEX_IMAGE,
        SVMNodeTexImage{
            .id = image_id,
            .projection = *projection,
            .flags = *flags,
            .co = compiler.input_stack("Vector"),
            .out_offset = compiler.output("Color"),
            .alpha_offset = compiler.output("Alpha")});
  }
};

class EnvironmentTextureNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_ENVIRONMENT;
  }

  void compile(SVMCompiler &compiler) override {
    const auto resource = unsigned_property(this, "Image");
    const auto interpolation_name = string_property(this, "Interpolation");
    const auto projection_name = string_property(this, "Projection");
    const auto flags = image_flags(this, false);
    const auto interpolation = interpolation_name
                                   ? image_interpolation(*interpolation_name)
                                   : std::nullopt;
    const auto projection = projection_name
                                ? environment_projection(*projection_name)
                                : std::nullopt;
    if (!resource || !interpolation || !projection || !flags) {
      compiler.fail("Cycles Environment Texture properties are invalid");
      return;
    }
    const auto image_id = compiler.image(
        *resource, *interpolation, ImageExtension::repeat);
    if (image_id == std::numeric_limits<std::int32_t>::max()) {
      compiler.fail("Cycles SVM image handle table overflow");
      return;
    }
    compiler.add_node(
        this, NODE_TEX_ENVIRONMENT,
        SVMNodeTexEnvironment{
            .id = image_id,
            .projection = *projection,
            .flags = *flags,
            .co = compiler.input_stack("Vector"),
            .out_offset = compiler.output("Color"),
            .alpha_offset = compiler.output("Alpha")});
  }
};

} // namespace

std::unique_ptr<GraphNode> make_image_graph_node(std::string_view type) {
  if (type == node_type::image_texture) {
    return std::make_unique<ImageTextureNode>();
  }
  if (type == node_type::environment_texture) {
    return std::make_unique<EnvironmentTextureNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm

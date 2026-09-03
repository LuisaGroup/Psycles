/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_sky_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_mapping_nodes.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_nishita.h>

#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] std::optional<float>
literal_float(const GraphNode *node, std::string_view name) noexcept {
  const auto *input = node->input(name);
  if (input == nullptr || input->link != nullptr || !input->value ||
      input->value->type != contract::SocketType::floating) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<float>(&input->value->value)) {
    return *value;
  }
  return std::nullopt;
}

template<typename T>
[[nodiscard]] std::optional<T>
property(const GraphNode *node, std::string_view name,
         contract::SocketType type) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() || iter->second.type != type) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<T>(&iter->second.value)) {
    return *value;
  }
  return std::nullopt;
}

class SkyTextureNode final : public TextureNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_SKY;
  }

  void compile(SVMCompiler &compiler) override {
    const auto sky_type_name = property<std::string>(
        this, "SkyType", contract::SocketType::string);
    const auto sun_disc = property<bool>(
        this, "SunDisc", contract::SocketType::boolean);
    const auto authored_sun_size = property<float>(
        this, "AuthoredSunSize", contract::SocketType::floating);
    const auto sun_elevation = literal_float(this, "SunElevation");
    const auto sun_rotation = literal_float(this, "SunRotation");
    const auto sun_intensity = literal_float(this, "SunIntensity");
    const auto altitude = literal_float(this, "Altitude");
    const auto air_density = literal_float(this, "AirDensity");
    const auto aerosol_density = literal_float(this, "DustDensity");
    const auto ozone_density = literal_float(this, "OzoneDensity");
    if (!sky_type_name || !sun_disc || !authored_sun_size ||
        !sun_elevation || !sun_rotation || !sun_intensity || !altitude ||
        !air_density || !aerosol_density || !ozone_density) {
      compiler.fail("Cycles Nishita Sky properties are invalid or dynamic");
      return;
    }

    const auto multiple_scattering =
        *sky_type_name == "MULTIPLE_SCATTERING";
    if (!multiple_scattering && *sky_type_name != "SINGLE_SCATTERING") {
      compiler.fail("Cycles Nishita Sky type is invalid");
      return;
    }
    const auto generated_image = NishitaImageBinding::encode(
        multiple_scattering, *sun_elevation, *altitude, *air_density,
        *aerosol_density, *ozone_density);
    const auto image_id = compiler.image(generated_image);
    if (image_id == std::numeric_limits<std::int32_t>::max()) {
      compiler.fail("Cycles SVM image handle table overflow");
      return;
    }
    const auto sun =
        precompute_nishita_sun(generated_image, *authored_sun_size);

    auto *vector_in = input("Vector");
    const auto vector_offset =
        tex_mapping.compile_begin(compiler, vector_in, this);
    compiler.add_node(
        this, NODE_TEX_SKY,
        SVMNodeTexSky{
            .sky_type = multiple_scattering
                            ? NODE_SKY_MULTIPLE_SCATTERING
                            : NODE_SKY_SINGLE_SCATTERING,
            .dir_offset = vector_offset,
            .out_offset = compiler.output("Color"),
            ._pad = {0u, 0u}});
    compiler.add_node_data(SVMNodeTexSkyNishitaData{
        .pixel_bottom_x = sun.pixel_bottom.x,
        .pixel_bottom_y = sun.pixel_bottom.y,
        .pixel_bottom_z = sun.pixel_bottom.z,
        .pixel_top_x = sun.pixel_top.x,
        .pixel_top_y = sun.pixel_top.y,
        .pixel_top_z = sun.pixel_top.z,
        .sun_elevation = *sun_elevation,
        .sun_rotation = *sun_rotation,
        .angular_diameter = *sun_disc ? *authored_sun_size : -1.0f,
        .sun_intensity = *sun_intensity,
        .earth_intersection_angle = sun.earth_intersection_angle,
        .texture_id = static_cast<std::uint32_t>(image_id)});
    tex_mapping.compile_end(compiler, vector_in, vector_offset);
  }
};

} // namespace

std::unique_ptr<GraphNode> make_sky_graph_node(std::string_view type) {
  if (type == node_type::nishita_sky) {
    return std::make_unique<SkyTextureNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm

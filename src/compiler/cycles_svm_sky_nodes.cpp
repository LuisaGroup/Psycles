/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_sky_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_mapping_nodes.h"

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_nishita.h>

#include <sky_hosek.h>

#include <algorithm>
#include <cmath>
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

template <typename T>
[[nodiscard]] std::optional<T> property(const GraphNode *node,
                                        std::string_view name,
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

void precompute_preetham(SVMNodeTexSkyPreethamData &data, float T) noexcept {
  const auto theta = data.theta;
  const auto theta2 = theta * theta;
  const auto theta3 = theta2 * theta;
  const auto T2 = T * T;
  constexpr auto pi = 3.14159265358979323846f;
  const auto chi = (4.0f / 9.0f - T / 120.0f) * (pi - 2.0f * theta);
  data.radiance_x =
      (4.0453f * T - 4.9710f) * std::tan(chi) - 0.2155f * T + 2.4192f;
  data.radiance_x *= 0.06f;
  data.radiance_y =
      (0.00166f * theta3 - 0.00375f * theta2 + 0.00209f * theta) * T2 +
      (-0.02903f * theta3 + 0.06377f * theta2 - 0.03202f * theta + 0.00394f) *
          T +
      (0.11693f * theta3 - 0.21196f * theta2 + 0.06052f * theta + 0.25886f);
  data.radiance_z =
      (0.00275f * theta3 - 0.00610f * theta2 + 0.00317f * theta) * T2 +
      (-0.04214f * theta3 + 0.08970f * theta2 - 0.04153f * theta + 0.00516f) *
          T +
      (0.15346f * theta3 - 0.26756f * theta2 + 0.06670f * theta + 0.26688f);
  data.config_x[0] = 0.1787f * T - 1.4630f;
  data.config_x[1] = -0.3554f * T + 0.4275f;
  data.config_x[2] = -0.0227f * T + 5.3251f;
  data.config_x[3] = 0.1206f * T - 2.5771f;
  data.config_x[4] = -0.0670f * T + 0.3703f;
  data.config_y[0] = -0.0193f * T - 0.2592f;
  data.config_y[1] = -0.0665f * T + 0.0008f;
  data.config_y[2] = -0.0004f * T + 0.2125f;
  data.config_y[3] = -0.0641f * T - 0.8989f;
  data.config_y[4] = -0.0033f * T + 0.0452f;
  data.config_z[0] = -0.0167f * T - 0.2608f;
  data.config_z[1] = -0.0950f * T + 0.0092f;
  data.config_z[2] = -0.0079f * T + 0.2102f;
  data.config_z[3] = -0.0441f * T - 1.6537f;
  data.config_z[4] = -0.0109f * T + 0.0529f;
  const auto perez = [theta](const float *lam) noexcept {
    return (1.0f + lam[0] * std::exp(lam[1])) *
           (1.0f + lam[2] * std::exp(lam[3] * theta) +
            lam[4] * std::cos(theta) * std::cos(theta));
  };
  data.radiance_x /= perez(data.config_x);
  data.radiance_y /= perez(data.config_y);
  data.radiance_z /= perez(data.config_z);
}

class AnalyticSkyTextureNode final : public TextureNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_SKY;
  }

  void compile(SVMCompiler &compiler) override {
    const auto x =
        property<float>(this, "SunDirectionX", contract::SocketType::floating);
    const auto y =
        property<float>(this, "SunDirectionY", contract::SocketType::floating);
    const auto z =
        property<float>(this, "SunDirectionZ", contract::SocketType::floating);
    const auto turbidity =
        property<float>(this, "Turbidity", contract::SocketType::floating);
    const auto albedo =
        property<float>(this, "GroundAlbedo", contract::SocketType::floating);
    if (!x || !y || !z || !turbidity || !albedo) {
      compiler.fail("Cycles analytic Sky properties are invalid");
      return;
    }
    // Cycles sky_texture_precompute_hosek: the authored direction is already
    // the ShaderNode value. Do not normalize it again or widen its spherical
    // conversion to double as the legacy compact-sky representation does.
    constexpr auto half_pi = 1.57079632679489661923f;
    SVMNodeTexSkyPreethamData data{};
    data.phi = std::atan2(*x, *y);
    data.theta = std::acos(*z);
    const auto preetham = type == node_type::preetham_sky;
    if (preetham) {
      precompute_preetham(data, *turbidity);
    } else {
      data.theta = std::clamp(data.theta, 0.0f, half_pi);
      const float elevation = half_pi - data.theta;
      auto *state = SKY_arhosek_xyz_skymodelstate_alloc_init(
          static_cast<double>(std::clamp(*turbidity, 0.0f, 10.0f)),
          static_cast<double>(*albedo), static_cast<double>(elevation));
      for (auto i = 0u; i < 9u; ++i) {
        data.config_x[i] = static_cast<float>(state->configs[0][i]);
        data.config_y[i] = static_cast<float>(state->configs[1][i]);
        data.config_z[i] = static_cast<float>(state->configs[2][i]);
      }
      data.radiance_x = static_cast<float>(state->radiances[0]);
      data.radiance_y = static_cast<float>(state->radiances[1]);
      data.radiance_z = static_cast<float>(state->radiances[2]);
      SKY_arhosekskymodelstate_free(state);
    }

    auto *vector_in = input("Vector");
    const auto vector_offset =
        tex_mapping.compile_begin(compiler, vector_in, this);
    compiler.add_node(
        this, NODE_TEX_SKY,
        SVMNodeTexSky{.sky_type = preetham ? NODE_SKY_PREETHAM : NODE_SKY_HOSEK,
                      .dir_offset = vector_offset,
                      .out_offset = compiler.output("Color"),
                      ._pad = {0u, 0u}});
    compiler.add_node_data(data);
    tex_mapping.compile_end(compiler, vector_in, vector_offset);
  }
};

class SkyTextureNode final : public TextureNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_TEX_SKY;
  }

  void compile(SVMCompiler &compiler) override {
    const auto sky_type_name =
        property<std::string>(this, "SkyType", contract::SocketType::string);
    const auto sun_disc =
        property<bool>(this, "SunDisc", contract::SocketType::boolean);
    const auto authored_sun_size = property<float>(
        this, "AuthoredSunSize", contract::SocketType::floating);
    const auto sun_elevation = literal_float(this, "SunElevation");
    const auto sun_rotation = literal_float(this, "SunRotation");
    const auto sun_intensity = literal_float(this, "SunIntensity");
    const auto altitude = literal_float(this, "Altitude");
    const auto air_density = literal_float(this, "AirDensity");
    const auto aerosol_density = literal_float(this, "DustDensity");
    const auto ozone_density = literal_float(this, "OzoneDensity");
    if (!sky_type_name || !sun_disc || !authored_sun_size || !sun_elevation ||
        !sun_rotation || !sun_intensity || !altitude || !air_density ||
        !aerosol_density || !ozone_density) {
      compiler.fail("Cycles Nishita Sky properties are invalid or dynamic");
      return;
    }

    const auto multiple_scattering = *sky_type_name == "MULTIPLE_SCATTERING";
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
        SVMNodeTexSky{.sky_type = multiple_scattering
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
  if (type == node_type::hosek_wilkie_sky || type == node_type::preetham_sky) {
    return std::make_unique<AnalyticSkyTextureNode>();
  }
  if (type == node_type::nishita_sky) {
    return std::make_unique<SkyTextureNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm

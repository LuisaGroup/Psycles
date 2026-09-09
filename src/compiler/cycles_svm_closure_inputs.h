/* SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "cycles_svm_graph.h"
#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <span>

namespace psycles::compiler::cycles_svm {

// Cycles 5.2.1 scene/shader_nodes.cpp NODE_DEFINE order, using the contract's
// spelling of socket names. This is not Blender UI order: generate_closure_node
// traverses each declared input separately, so order controls dependencies,
// emitted words and stack lifetimes even when final BSDF values agree.
inline std::span<const std::string_view> closure_input_order(std::string_view type) {
  static constexpr std::string_view diffuse[]{"Color", "Normal", "SurfaceMixWeight", "Roughness"};
  static constexpr std::string_view translucent[]{"Color", "Normal", "SurfaceMixWeight"};
  static constexpr std::string_view transparent[]{"Color", "SurfaceMixWeight"};
  static constexpr std::string_view glossy[]{"Color", "Normal", "SurfaceMixWeight", "Tangent",
                                           "Roughness", "Anisotropy", "Rotation"};
  static constexpr std::string_view metallic[]{"BaseColor", "Normal", "SurfaceMixWeight", "EdgeTint",
      "IOR", "Extinction", "Tangent", "Roughness", "Anisotropy", "Rotation",
      "ThinFilmThickness", "ThinFilmIOR"};
  static constexpr std::string_view glass[]{"Color", "Normal", "SurfaceMixWeight", "Roughness",
                                          "IOR", "ThinFilmThickness", "ThinFilmIOR"};
  static constexpr std::string_view refraction[]{"Color", "Normal", "SurfaceMixWeight", "Roughness", "IOR"};
  static constexpr std::string_view toon[]{"Color", "Normal", "SurfaceMixWeight", "Size", "Smooth"};
  static constexpr std::string_view subsurface[]{"Color", "Normal", "SurfaceMixWeight", "Scale",
                                               "Radius", "IOR", "Roughness", "Anisotropy"};
  static constexpr std::string_view hair[]{"Color", "SurfaceMixWeight", "Offset", "RoughnessU", "RoughnessV", "Tangent"};
  static constexpr std::string_view portal[]{"Color", "SurfaceMixWeight", "Position", "Direction"};
  static constexpr std::string_view principled[]{"BaseColor", "Metallic", "Roughness", "IOR", "Alpha",
      "ThinWall", "Normal", "DiffuseRoughness", "SubsurfaceWeight", "SubsurfaceScale",
      "SubsurfaceRadius", "SubsurfaceIOR", "SubsurfaceAnisotropy", "SpecularIORLevel",
      "SpecularTint", "Anisotropic", "AnisotropicRotation", "Tangent", "TransmissionWeight",
      "SheenWeight", "SheenRoughness", "SheenTint", "CoatWeight", "CoatRoughness", "CoatIOR",
      "CoatTint", "CoatNormal", "EmissionColor", "EmissionStrength", "ThinFilmThickness",
      "ThinFilmIOR", "SurfaceMixWeight"};
  static constexpr std::string_view emission[]{"Color", "Strength", "SurfaceMixWeight", "VolumeMixWeight"};
  static constexpr std::string_view background[]{"Color", "Strength", "SurfaceMixWeight"};
  static constexpr std::string_view holdout[]{"SurfaceMixWeight", "VolumeMixWeight"};
  if (type == node_type::diffuse_bsdf || type == node_type::sheen_bsdf) { return diffuse; }
  if (type == node_type::translucent_bsdf) { return translucent; }
  if (type == node_type::transparent_bsdf) { return transparent; }
  if (type == node_type::glossy_bsdf) { return glossy; }
  if (type == node_type::metallic_bsdf) { return metallic; }
  if (type == node_type::glass_bsdf) { return glass; }
  if (type == node_type::refraction_bsdf) { return refraction; }
  if (type == node_type::toon_bsdf) { return toon; }
  if (type == node_type::subsurface_scattering) { return subsurface; }
  if (type == node_type::hair_bsdf) { return hair; }
  if (type == node_type::ray_portal_bsdf) { return portal; }
  if (type == node_type::principled_bsdf) { return principled; }
  if (type == node_type::emission) { return emission; }
  if (type == node_type::background) { return background; }
  if (type == node_type::holdout) { return holdout; }
  return {};
}

inline bool project_closure_input_order(std::string_view type, std::vector<GraphInput> &inputs) {
  const auto order = closure_input_order(type);
  if (order.size() != inputs.size()) { return false; }
  for (const auto name : order) {
    if (std::count_if(inputs.begin(), inputs.end(), [name](const auto &input) {
          return input.name == name;
        }) != 1) { return false; }
  }
  const auto rank = [&](const auto &input) {
    return std::find(order.begin(), order.end(), input.name) - order.begin();
  };
  std::sort(inputs.begin(), inputs.end(), [&](const auto &a, const auto &b) {
    return rank(a) < rank(b);
  });
  return true;
}

} // namespace psycles::compiler::cycles_svm

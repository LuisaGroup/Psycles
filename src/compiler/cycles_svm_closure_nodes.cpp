/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_closure_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <variant>

namespace psycles::compiler::cycles_svm {
namespace {

template<typename T>
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

[[nodiscard]] std::optional<std::string_view> string_property(
    const GraphNode *node, std::string_view name) noexcept {
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

[[nodiscard]] bool
enum_token_is(std::string_view value,
              std::string_view canonical_uppercase) noexcept {
  if (value.size() != canonical_uppercase.size()) {
    return false;
  }
  for (auto index = std::size_t{}; index < value.size(); ++index) {
    auto character = value[index];
    if (character >= 'a' && character <= 'z') {
      character = static_cast<char>(character - 'a' + 'A');
    }
    if (character != canonical_uppercase[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::optional<ClosureType>
principled_distribution(const GraphNode *node) noexcept {
  const auto distribution = string_property(node, "Distribution");
  if (distribution == "GGX") {
    return CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID;
  }
  if (distribution == "MULTI_GGX") {
    return CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ClosureType>
glass_distribution(const GraphNode *node) noexcept {
  const auto distribution = string_property(node, "Distribution");
  if (distribution == "BECKMANN") {
    return CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID;
  }
  if (distribution == "GGX") {
    return CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID;
  }
  if (distribution == "MULTI_GGX") {
    return CLOSURE_BSDF_MICROFACET_MULTI_GGX_GLASS_ID;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ClosureType>
glossy_distribution(const GraphNode *node) noexcept {
  const auto distribution = string_property(node, "Distribution");
  if (distribution == "BECKMANN") {
    return CLOSURE_BSDF_MICROFACET_BECKMANN_ID;
  }
  if (distribution == "GGX") {
    return CLOSURE_BSDF_MICROFACET_GGX_ID;
  }
  if (distribution == "ASHIKHMIN_SHIRLEY") {
    return CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID;
  }
  if (distribution == "MULTI_GGX") {
    return CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ClosureType>
refraction_distribution(const GraphNode *node) noexcept {
  const auto distribution = string_property(node, "Distribution");
  if (distribution == "BECKMANN") {
    return CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID;
  }
  if (distribution == "GGX") {
    return CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ClosureType>
metallic_distribution(const GraphNode *node) noexcept {
  const auto distribution = string_property(node, "Distribution");
  if (distribution == "BECKMANN") {
    return CLOSURE_BSDF_MICROFACET_BECKMANN_ID;
  }
  if (distribution == "GGX") {
    return CLOSURE_BSDF_MICROFACET_GGX_ID;
  }
  if (distribution == "MULTI_GGX") {
    return CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ClosureType>
metallic_fresnel_type(const GraphNode *node) noexcept {
  const auto fresnel_type = string_property(node, "FresnelType");
  if (fresnel_type == "F82") {
    return CLOSURE_BSDF_F82_CONDUCTOR;
  }
  if (fresnel_type == "PHYSICAL_CONDUCTOR") {
    return CLOSURE_BSDF_PHYSICAL_CONDUCTOR;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ClosureType>
sheen_distribution(const GraphNode *node) noexcept {
  const auto distribution = string_property(node, "Distribution");
  if (distribution == "MICROFIBER") {
    return CLOSURE_BSDF_SHEEN_ID;
  }
  if (distribution == "ASHIKHMIN") {
    return CLOSURE_BSDF_ASHIKHMIN_VELVET_ID;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ClosureType>
toon_component(const GraphNode *node) noexcept {
  const auto component = string_property(node, "Component");
  if (component == "DIFFUSE") {
    return CLOSURE_BSDF_DIFFUSE_TOON_ID;
  }
  if (component == "GLOSSY") {
    return CLOSURE_BSDF_GLOSSY_TOON_ID;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ClosureType>
hair_component(const GraphNode *node) noexcept {
  const auto component = string_property(node, "Component");
  if (component && enum_token_is(*component, "REFLECTION")) {
    return CLOSURE_BSDF_HAIR_REFLECTION_ID;
  }
  if (component && enum_token_is(*component, "TRANSMISSION")) {
    return CLOSURE_BSDF_HAIR_TRANSMISSION_ID;
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

[[nodiscard]] std::optional<ClosureType>
subsurface_method(const GraphNode *node,
                  std::string_view property_name) noexcept {
  const auto method = string_property(node, property_name);
  if (method == "BURLEY") {
    return CLOSURE_BSSRDF_BURLEY_ID;
  }
  if (method == "RANDOM_WALK") {
    return CLOSURE_BSSRDF_RANDOM_WALK_ID;
  }
  if (method == "RANDOM_WALK_SKIN") {
    return CLOSURE_BSSRDF_RANDOM_WALK_SKIN_ID;
  }
  if (method == "RANDOM_WALK_LEGACY") {
    return CLOSURE_BSSRDF_RANDOM_WALK_LEGACY_ID;
  }
  return std::nullopt;
}

void disconnect_input(GraphInput *input) noexcept {
  if (input == nullptr || input->link == nullptr) {
    return;
  }
  auto &links = input->link->links;
  links.erase(std::remove(links.begin(), links.end(), input), links.end());
  input->link = nullptr;
}

class BsdfNode : public GraphNode {
private:
  ClosureType _closure;

protected:
  explicit BsdfNode(ClosureType closure) noexcept : _closure{closure} {}

  template<SvmPayload T>
  void compile_bsdf(SVMCompiler &compiler, ClosureType closure,
                    const T &data) {
    auto *color = input("Color");
    if (color == nullptr) {
      compiler.fail("Cycles BSDF Color input is absent");
      return;
    }
    if (color->link != nullptr) {
      compiler.add_node(
          this, NODE_CLOSURE_WEIGHT,
          SVMNodeClosureWeight{.weight_offset =
                                   compiler.input_link("Color"),
                               ._pad = {0u, 0u, 0u}});
    } else {
      const auto value = literal<Vec3f>(color, contract::SocketType::color);
      if (!value) {
        compiler.fail("Cycles BSDF Color input is ill typed");
        return;
      }
      compiler.add_node(
          this, NODE_CLOSURE_SET_WEIGHT,
          SVMNodeClosureSetWeight{
              .rgb = packed_float3{value->x, value->y, value->z}});
    }
    compiler.add_bsdf_node(
        SVMNodeClosureBsdf{
            .closure_type = closure,
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}},
        data);
  }

  template<SvmPayload T>
  void compile_bsdf(SVMCompiler &compiler, const T &data) {
    compile_bsdf(compiler, _closure, data);
  }

public:
  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return GraphNode::get_feature() | kernel_feature_node_bsdf;
  }
  [[nodiscard]] bool equals(const GraphNode &) const noexcept override {
    return false;
  }
  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_CLOSURE_BSDF;
  }
};

class DiffuseBsdfNode final : public BsdfNode {
public:
  DiffuseBsdfNode() noexcept : BsdfNode{CLOSURE_BSDF_DIFFUSE_ID} {}

  void compile(SVMCompiler &compiler) override {
    compile_bsdf(
        compiler,
        SVMNodeDiffuseBsdfData{.color = compiler.input_float3("Color"),
                               .roughness =
                                   compiler.input_float("Roughness"),
                               .normal_offset =
                                   compiler.input_link("Normal"),
                               ._pad = {0u, 0u, 0u}});
  }
};

class TranslucentBsdfNode final : public BsdfNode {
public:
  TranslucentBsdfNode() noexcept
      : BsdfNode{CLOSURE_BSDF_TRANSLUCENT_ID} {}

  void compile(SVMCompiler &compiler) override {
    compile_bsdf(
        compiler,
        SVMNodeSimpleBsdfData{.param1 = {},
                              .normal_offset =
                                  compiler.input_link("Normal"),
                              ._pad = {0u, 0u, 0u}});
  }
};

class TransparentBsdfNode final : public BsdfNode {
public:
  TransparentBsdfNode() noexcept
      : BsdfNode{CLOSURE_BSDF_TRANSPARENT_ID} {}

  void compile(SVMCompiler &compiler) override {
    compile_bsdf(compiler, SVMNodeSimpleBsdfData{});
  }
};

class SheenBsdfNode final : public BsdfNode {
public:
  SheenBsdfNode() noexcept : BsdfNode{CLOSURE_BSDF_SHEEN_ID} {}

  void compile(SVMCompiler &compiler) override {
    const auto distribution = sheen_distribution(this);
    if (!distribution) {
      compiler.fail("Cycles Sheen BSDF distribution is not migrated exactly");
      return;
    }
    compile_bsdf(
        compiler, *distribution,
        SVMNodeSimpleBsdfData{.param1 = compiler.input_float("Roughness"),
                              .normal_offset = compiler.input_link("Normal"),
                              ._pad = {0u, 0u, 0u}});
  }
};

class ToonBsdfNode final : public BsdfNode {
public:
  ToonBsdfNode() noexcept : BsdfNode{CLOSURE_BSDF_DIFFUSE_TOON_ID} {}

  void compile(SVMCompiler &compiler) override {
    const auto component = toon_component(this);
    if (!component) {
      compiler.fail("Cycles Toon BSDF component is not migrated exactly");
      return;
    }
    compile_bsdf(
        compiler, *component,
        SVMNodeToonBsdfData{.size = compiler.input_float("Size"),
                            .smooth = compiler.input_float("Smooth"),
                            .normal_offset = compiler.input_link("Normal"),
                            ._pad = {0u, 0u, 0u}});
  }
};

class RayPortalBsdfNode final : public BsdfNode {
public:
  RayPortalBsdfNode() noexcept : BsdfNode{CLOSURE_BSDF_RAY_PORTAL_ID} {}

  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return BsdfNode::get_feature() | kernel_feature_node_portal;
  }

  void compile(SVMCompiler &compiler) override {
    compile_bsdf(
        compiler, CLOSURE_BSDF_RAY_PORTAL_ID,
        SVMNodeRayPortalBsdfData{
            .direction = compiler.input_float3("Direction"),
            .position_offset = compiler.input_link("Position"),
            ._pad = {0u, 0u, 0u}});
  }
};

class HairBsdfNode final : public BsdfNode {
public:
  HairBsdfNode() noexcept : BsdfNode{CLOSURE_BSDF_HAIR_REFLECTION_ID} {}

  void compile(SVMCompiler &compiler) override {
    const auto component = hair_component(this);
    if (!component) {
      compiler.fail("Cycles Hair BSDF component is not migrated exactly");
      return;
    }
    compile_bsdf(
        compiler, *component,
        SVMNodeHairBsdfData{
            .roughness1 = compiler.input_float("RoughnessU"),
            .roughness2 = compiler.input_float("RoughnessV"),
            .offset = compiler.input_float("Offset"),
            .tangent_offset = compiler.input_link("Tangent"),
            ._pad = {0u, 0u, 0u}});
  }
};

class GlassBsdfNode final : public BsdfNode {
public:
  GlassBsdfNode() noexcept
      : BsdfNode{CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID} {}

  void compile(SVMCompiler &compiler) override {
    const auto distribution = glass_distribution(this);
    if (!distribution) {
      compiler.fail("Cycles Glass BSDF distribution is not migrated exactly");
      return;
    }
    compile_bsdf(
        compiler, *distribution,
        SVMNodeGlassBsdfData{
            .color = compiler.input_float3("Color"),
            .roughness = compiler.input_float("Roughness"),
            .ior = compiler.input_float("IOR"),
            .thin_film_thickness =
                compiler.input_float("ThinFilmThickness"),
            .thin_film_ior = compiler.input_float("ThinFilmIOR"),
            .normal_offset = compiler.input_link("Normal"),
            ._pad = {0u, 0u, 0u}});
  }
};

class GlossyBsdfNode final : public BsdfNode {
private:
  [[nodiscard]] bool is_isotropic() const noexcept {
    const auto anisotropy =
        literal<float>(input("Anisotropy"), contract::SocketType::floating);
    return anisotropy && std::abs(*anisotropy) <= 1.0e-4f;
  }

public:
  GlossyBsdfNode() noexcept
      : BsdfNode{CLOSURE_BSDF_MICROFACET_GGX_ID} {}

  void simplify_settings() override {
    // Direct copy of Cycles 5.2.1 GlossyBsdfNode::simplify_settings. The
    // literal helper also proves that Anisotropy has no authored link.
    if (is_isotropic()) {
      disconnect_input(input("Tangent"));
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto distribution = glossy_distribution(this);
    if (!distribution) {
      compiler.fail("Cycles Glossy BSDF distribution is not migrated exactly");
      return;
    }
    compile_bsdf(
        compiler, *distribution,
        SVMNodeGlossyBsdfData{
            .color = *distribution == CLOSURE_BSDF_MICROFACET_MULTI_GGX_ID
                         ? compiler.input_float3("Color")
                         : SVMInputFloat3{},
            .roughness = compiler.input_float("Roughness"),
            .anisotropy = compiler.input_float("Anisotropy"),
            .rotation = compiler.input_float("Rotation"),
            .normal_offset = compiler.input_link("Normal"),
            .tangent_offset = compiler.input_link("Tangent"),
            ._pad = {0u, 0u}});
  }
};

class RefractionBsdfNode final : public BsdfNode {
public:
  RefractionBsdfNode() noexcept
      : BsdfNode{CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID} {}

  void compile(SVMCompiler &compiler) override {
    const auto distribution = refraction_distribution(this);
    if (!distribution) {
      compiler.fail(
          "Cycles Refraction BSDF distribution is not migrated exactly");
      return;
    }
    compile_bsdf(
        compiler, *distribution,
        SVMNodeRefractionBsdfData{
            .roughness = compiler.input_float("Roughness"),
            .ior = compiler.input_float("IOR"),
            .normal_offset = compiler.input_link("Normal"),
            ._pad = {0u, 0u, 0u}});
  }
};

class MetallicBsdfNode final : public BsdfNode {
private:
  [[nodiscard]] bool is_isotropic() const noexcept {
    const auto anisotropy =
        literal<float>(input("Anisotropy"), contract::SocketType::floating);
    return anisotropy && std::abs(*anisotropy) <= 1.0e-4f;
  }

public:
  MetallicBsdfNode() noexcept : BsdfNode{CLOSURE_BSDF_PHYSICAL_CONDUCTOR} {}

  void simplify_settings() override {
    // Direct copy of Cycles 5.2.1 MetallicBsdfNode::simplify_settings.
    if (is_isotropic()) {
      disconnect_input(input("Tangent"));
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto distribution = metallic_distribution(this);
    const auto fresnel_type = metallic_fresnel_type(this);
    if (!distribution || !fresnel_type) {
      compiler.fail("Cycles Metallic BSDF enum property is not migrated exactly");
      return;
    }
    compiler.add_bsdf_node(
        SVMNodeClosureBsdf{
            .closure_type = *fresnel_type,
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}},
        SVMNodeMetallicBsdfData{
            .distribution = *distribution,
            .base_ior = *fresnel_type == CLOSURE_BSDF_PHYSICAL_CONDUCTOR
                            ? compiler.input_float3("IOR")
                            : compiler.input_float3("BaseColor"),
            .edge_tint_k = *fresnel_type == CLOSURE_BSDF_PHYSICAL_CONDUCTOR
                               ? compiler.input_float3("Extinction")
                               : compiler.input_float3("EdgeTint"),
            .roughness = compiler.input_float("Roughness"),
            .anisotropy = compiler.input_float("Anisotropy"),
            .rotation = compiler.input_float("Rotation"),
            .thin_film_thickness =
                compiler.input_float("ThinFilmThickness"),
            .thin_film_ior = compiler.input_float("ThinFilmIOR"),
            .normal_offset = compiler.input_link("Normal"),
            .tangent_offset = compiler.input_link("Tangent"),
            ._pad = {0u, 0u}});
  }
};

class SubsurfaceScatteringNode final : public BsdfNode {
public:
  SubsurfaceScatteringNode() noexcept
      : BsdfNode{CLOSURE_BSSRDF_RANDOM_WALK_ID} {}

  void compile(SVMCompiler &compiler) override {
    const auto method = subsurface_method(this, "Method");
    if (!method) {
      compiler.fail("Cycles BSSRDF method is not migrated exactly");
      return;
    }
    compile_bsdf(
        compiler, *method,
        SVMNodeBssrdfData{
            .radius = compiler.input_float3("Radius"),
            .scale = compiler.input_float("Scale"),
            .ior = compiler.input_float("IOR"),
            .anisotropy = compiler.input_float("Anisotropy"),
            .roughness = compiler.input_float("Roughness"),
            .normal_offset = compiler.input_link("Normal"),
            ._pad = {0u, 0u, 0u}});
  }
};

class VolumeClosureNode : public GraphNode {
protected:
  void compile_volume(SVMCompiler &compiler, ClosureType closure,
                      std::string_view density_name,
                      std::string_view param1_name = {},
                      std::string_view param2_name = {}) {
    auto *color = input("Color");
    if (color == nullptr) {
      compiler.fail("Cycles Volume Color input is absent");
      return;
    }
    if (color->link != nullptr) {
      compiler.add_node(
          this, NODE_CLOSURE_WEIGHT,
          SVMNodeClosureWeight{.weight_offset =
                                   compiler.input_link("Color"),
                               ._pad = {0u, 0u, 0u}});
    } else {
      const auto value = literal<Vec3f>(color, contract::SocketType::color);
      if (!value) {
        compiler.fail("Cycles Volume Color input is ill typed");
        return;
      }
      compiler.add_node(
          this, NODE_CLOSURE_SET_WEIGHT,
          SVMNodeClosureSetWeight{
              .rgb = packed_float3{value->x, value->y, value->z}});
    }
    compiler.add_node(
        this, NODE_CLOSURE_VOLUME,
        SVMNodeClosureVolume{
            .closure_type = closure,
            .density = density_name.empty()
                           ? SVMInputFloat{0u}
                           : compiler.input_float(density_name),
            .param1 = param1_name.empty()
                          ? SVMInputFloat{0u}
                          : compiler.input_float(param1_name),
            .param_extra = param2_name.empty()
                               ? SVMInputFloat{0u}
                               : compiler.input_float(param2_name),
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}});
  }

public:
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
  void compile(SVMCompiler &compiler) override {
    compile_volume(compiler, CLOSURE_VOLUME_ABSORPTION_ID, "Density");
  }
};

class ScatterVolumeNode final : public VolumeClosureNode {
public:
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

class PrincipledBsdfNode final : public GraphNode {
private:
  [[nodiscard]] bool has_nonzero_weight(std::string_view name) const noexcept {
    const auto *weight = input(name);
    if (weight == nullptr || weight->link != nullptr) {
      return true;
    }
    const auto value = literal<float>(weight, contract::SocketType::floating);
    return !value || *value >= CLOSURE_WEIGHT_CUTOFF;
  }

  [[nodiscard]] bool thin_wall() const noexcept {
    const auto *socket = input("ThinWall");
    if (socket == nullptr || socket->link != nullptr || !socket->value) {
      return false;
    }
    if (const auto *value = std::get_if<bool>(&socket->value->value)) {
      return *value;
    }
    if (const auto *value =
            std::get_if<std::int64_t>(&socket->value->value)) {
      return *value != 0;
    }
    if (const auto *value =
            std::get_if<std::uint64_t>(&socket->value->value)) {
      return *value != 0u;
    }
    return false;
  }

  [[nodiscard]] bool has_surface_emission() const noexcept {
    const auto *color = input("EmissionColor");
    const auto *strength = input("EmissionStrength");
    const auto color_value =
        literal<Vec3f>(color, contract::SocketType::color);
    const auto strength_value =
        literal<float>(strength, contract::SocketType::floating);
    const auto color_nonzero =
        color != nullptr &&
        (color->link != nullptr ||
         (color_value && std::max({color_value->x, color_value->y,
                                  color_value->z}) > CLOSURE_WEIGHT_CUTOFF));
    const auto strength_nonzero =
        strength != nullptr &&
        (strength->link != nullptr ||
         (strength_value && *strength_value > CLOSURE_WEIGHT_CUTOFF));
    return color_nonzero && strength_nonzero;
  }

  [[nodiscard]] bool has_surface_bssrdf() const noexcept {
    if (thin_wall() || !has_nonzero_weight("SubsurfaceWeight")) {
      return false;
    }
    const auto *scale = input("SubsurfaceScale");
    const auto value =
        literal<float>(scale, contract::SocketType::floating);
    return scale == nullptr || scale->link != nullptr || !value || *value != 0.0f;
  }

public:
  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return GraphNode::get_feature() | kernel_feature_node_bsdf;
  }

  [[nodiscard]] bool equals(const GraphNode &) const noexcept override {
    return false;
  }

  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_CLOSURE_BSDF;
  }

  void simplify_settings() override {
    if (!has_surface_emission()) {
      disconnect_input(input("EmissionColor"));
      disconnect_input(input("EmissionStrength"));
    }

    if (thin_wall()) {
      disconnect_input(input("SubsurfaceRadius"));
      disconnect_input(input("SubsurfaceScale"));
      disconnect_input(input("SubsurfaceIOR"));
    } else if (!has_surface_bssrdf()) {
      disconnect_input(input("SubsurfaceWeight"));
      disconnect_input(input("SubsurfaceRadius"));
      disconnect_input(input("SubsurfaceScale"));
      disconnect_input(input("SubsurfaceIOR"));
      disconnect_input(input("SubsurfaceAnisotropy"));
    }

    if (!has_nonzero_weight("CoatWeight")) {
      disconnect_input(input("CoatWeight"));
      disconnect_input(input("CoatIOR"));
      disconnect_input(input("CoatRoughness"));
      disconnect_input(input("CoatTint"));
    }
    if (!has_nonzero_weight("SheenWeight")) {
      disconnect_input(input("SheenWeight"));
      disconnect_input(input("SheenRoughness"));
      disconnect_input(input("SheenTint"));
    }
    if (!has_nonzero_weight("Anisotropic")) {
      disconnect_input(input("Anisotropic"));
      disconnect_input(input("AnisotropicRotation"));
      disconnect_input(input("Tangent"));
    }
    if (!has_nonzero_weight("ThinFilmThickness")) {
      disconnect_input(input("ThinFilmThickness"));
      disconnect_input(input("ThinFilmIOR"));
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto distribution = principled_distribution(this);
    const auto method = subsurface_method(this, "SubsurfaceMethod");
    if (!distribution || !method) {
      compiler.fail("Cycles Principled BSDF enum property is not migrated exactly");
      return;
    }
    const auto tangent_offset =
        has_nonzero_weight("Anisotropic")
            ? compiler.input_link("Tangent")
            : SVM_STACK_INVALID;
    compiler.add_bsdf_node(
        SVMNodeClosureBsdf{
            .closure_type = CLOSURE_BSDF_PRINCIPLED_ID,
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}},
        SVMNodePrincipledBsdfData{
            .distribution = *distribution,
            .ior = compiler.input_float("IOR"),
            .roughness = compiler.input_float("Roughness"),
            .sheen_weight = compiler.input_float("SheenWeight"),
            .coat_weight = compiler.input_float("CoatWeight"),
            .metallic = compiler.input_float("Metallic"),
            .transmission_weight = compiler.input_float("TransmissionWeight"),
            .subsurface_weight = compiler.input_float("SubsurfaceWeight"),
            .base_color = compiler.input_float3("BaseColor"),
            .alpha = compiler.input_float("Alpha"),
            .diffuse_roughness = compiler.input_float("DiffuseRoughness"),
            .normal_offset = compiler.input_link("Normal"),
            .tangent_offset = tangent_offset,
            .coat_normal_offset = compiler.input_link("CoatNormal"),
            ._pad = {0u},
            .specular_tint = compiler.input_float3("SpecularTint"),
            .specular_ior_level = compiler.input_float("SpecularIORLevel"),
            .anisotropic = compiler.input_float("Anisotropic"),
            .anisotropic_rotation =
                compiler.input_float("AnisotropicRotation"),
            .emission_color = compiler.input_float3("EmissionColor"),
            .emission_strength = compiler.input_float("EmissionStrength"),
            .sheen_tint = compiler.input_float3("SheenTint"),
            .sheen_roughness = compiler.input_float("SheenRoughness"),
            .coat_tint = compiler.input_float3("CoatTint"),
            .coat_roughness = compiler.input_float("CoatRoughness"),
            .coat_ior = compiler.input_float("CoatIOR"),
            .subsurface_method = *method,
            .subsurface_radius = compiler.input_float3("SubsurfaceRadius"),
            .subsurface_scale = compiler.input_float("SubsurfaceScale"),
            .subsurface_ior = compiler.input_float("SubsurfaceIOR"),
            .subsurface_anisotropy =
                compiler.input_float("SubsurfaceAnisotropy"),
            .thin_film_thickness =
                compiler.input_float("ThinFilmThickness"),
            .thin_film_ior = compiler.input_float("ThinFilmIOR"),
            .thin_wall = compiler.input_int("ThinWall")});
  }
};

class EmissionNode final : public GraphNode {
public:
  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return GraphNode::get_feature() | kernel_feature_node_emission;
  }

  [[nodiscard]] bool has_volume_support() const noexcept override {
    return true;
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }

  void constant_fold(const ConstantFolder &folder) override {
    auto *color = input("Color");
    auto *strength = input("Strength");
    const auto color_value =
        literal<Vec3f>(color, contract::SocketType::color);
    const auto strength_value =
        literal<float>(strength, contract::SocketType::floating);
    if ((color_value && *color_value == Vec3f{}) ||
        (strength_value && *strength_value == 0.0f)) {
      folder.discard();
    }
  }

  void compile(SVMCompiler &compiler) override {
    auto *color = input("Color");
    auto *strength = input("Strength");
    if (color == nullptr || strength == nullptr) {
      compiler.fail("Cycles Emission inputs are absent");
      return;
    }
    if (color->link != nullptr || strength->link != nullptr) {
      compiler.add_node(
          this, NODE_EMISSION_WEIGHT,
          SVMNodeEmissionWeight{.color = compiler.input_float3("Color"),
                                .strength =
                                    compiler.input_float("Strength")});
    } else {
      const auto c = literal<Vec3f>(color, contract::SocketType::color);
      const auto s =
          literal<float>(strength, contract::SocketType::floating);
      if (!c || !s) {
        compiler.fail("Cycles Emission inputs are ill typed");
        return;
      }
      compiler.add_node(
          this, NODE_CLOSURE_SET_WEIGHT,
          SVMNodeClosureSetWeight{.rgb = packed_float3{
                                      c->x * *s, c->y * *s, c->z * *s}});
    }
    compiler.add_node(
        this, NODE_CLOSURE_EMISSION,
        SVMNodeClosureEmission{
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}});
  }
};

class BackgroundNode final : public GraphNode {
public:
  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return GraphNode::get_feature() | kernel_feature_node_emission;
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }

  void constant_fold(const ConstantFolder &folder) override {
    auto *color = input("Color");
    auto *strength = input("Strength");
    const auto color_value =
        literal<Vec3f>(color, contract::SocketType::color);
    const auto strength_value =
        literal<float>(strength, contract::SocketType::floating);
    if ((color_value && *color_value == Vec3f{}) ||
        (strength_value && *strength_value == 0.0f)) {
      folder.discard();
    }
  }

  void compile(SVMCompiler &compiler) override {
    auto *color = input("Color");
    auto *strength = input("Strength");
    if (color == nullptr || strength == nullptr) {
      compiler.fail("Cycles Background inputs are absent");
      return;
    }
    if (color->link != nullptr || strength->link != nullptr) {
      compiler.add_node(
          this, NODE_EMISSION_WEIGHT,
          SVMNodeEmissionWeight{.color = compiler.input_float3("Color"),
                                .strength =
                                    compiler.input_float("Strength")});
    } else {
      const auto c = literal<Vec3f>(color, contract::SocketType::color);
      const auto s =
          literal<float>(strength, contract::SocketType::floating);
      if (!c || !s) {
        compiler.fail("Cycles Background inputs are ill typed");
        return;
      }
      compiler.add_node(
          this, NODE_CLOSURE_SET_WEIGHT,
          SVMNodeClosureSetWeight{.rgb = packed_float3{
                                      c->x * *s, c->y * *s, c->z * *s}});
    }
    compiler.add_node(
        this, NODE_CLOSURE_BACKGROUND,
        SVMNodeClosureBackground{
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}});
  }
};

} // namespace

std::unique_ptr<GraphNode>
make_closure_graph_node(std::string_view type) {
  if (type == node_type::background) {
    return std::make_unique<BackgroundNode>();
  }
  if (type == node_type::diffuse_bsdf) {
    return std::make_unique<DiffuseBsdfNode>();
  }
  if (type == node_type::translucent_bsdf) {
    return std::make_unique<TranslucentBsdfNode>();
  }
  if (type == node_type::transparent_bsdf) {
    return std::make_unique<TransparentBsdfNode>();
  }
  if (type == node_type::sheen_bsdf) {
    return std::make_unique<SheenBsdfNode>();
  }
  if (type == node_type::toon_bsdf) {
    return std::make_unique<ToonBsdfNode>();
  }
  if (type == node_type::ray_portal_bsdf) {
    return std::make_unique<RayPortalBsdfNode>();
  }
  if (type == node_type::hair_bsdf) {
    return std::make_unique<HairBsdfNode>();
  }
  if (type == node_type::emission) {
    return std::make_unique<EmissionNode>();
  }
  if (type == node_type::principled_bsdf) {
    return std::make_unique<PrincipledBsdfNode>();
  }
  if (type == node_type::glass_bsdf) {
    return std::make_unique<GlassBsdfNode>();
  }
  if (type == node_type::glossy_bsdf) {
    return std::make_unique<GlossyBsdfNode>();
  }
  if (type == node_type::refraction_bsdf) {
    return std::make_unique<RefractionBsdfNode>();
  }
  if (type == node_type::metallic_bsdf) {
    return std::make_unique<MetallicBsdfNode>();
  }
  if (type == node_type::subsurface_scattering) {
    return std::make_unique<SubsurfaceScatteringNode>();
  }
  if (type == node_type::volume_absorption) {
    return std::make_unique<AbsorptionVolumeNode>();
  }
  if (type == node_type::volume_scatter) {
    return std::make_unique<ScatterVolumeNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm

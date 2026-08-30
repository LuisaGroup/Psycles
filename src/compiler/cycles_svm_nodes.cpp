/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
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

[[nodiscard]] std::optional<NodeMathType>
math_type(const GraphNode *node) noexcept {
  if (node->type == cycles_synthetic_math) {
    return NODE_MATH_MULTIPLY;
  }
  const auto operation = string_property(node, "Operation");
  if (!operation) {
    return std::nullopt;
  }
#define PSYCLES_CYCLES_MATH_TYPE(name, value) \
  if (*operation == name) {                   \
    return value;                             \
  }
  PSYCLES_CYCLES_MATH_TYPE("ADD", NODE_MATH_ADD)
  PSYCLES_CYCLES_MATH_TYPE("SUBTRACT", NODE_MATH_SUBTRACT)
  PSYCLES_CYCLES_MATH_TYPE("MULTIPLY", NODE_MATH_MULTIPLY)
  PSYCLES_CYCLES_MATH_TYPE("DIVIDE", NODE_MATH_DIVIDE)
  PSYCLES_CYCLES_MATH_TYPE("MULTIPLY_ADD", NODE_MATH_MULTIPLY_ADD)
  PSYCLES_CYCLES_MATH_TYPE("POWER", NODE_MATH_POWER)
  PSYCLES_CYCLES_MATH_TYPE("LOGARITHM", NODE_MATH_LOGARITHM)
  PSYCLES_CYCLES_MATH_TYPE("SQRT", NODE_MATH_SQRT)
  PSYCLES_CYCLES_MATH_TYPE("INVERSE_SQRT", NODE_MATH_INV_SQRT)
  PSYCLES_CYCLES_MATH_TYPE("ABSOLUTE", NODE_MATH_ABSOLUTE)
  PSYCLES_CYCLES_MATH_TYPE("EXPONENT", NODE_MATH_EXPONENT)
  PSYCLES_CYCLES_MATH_TYPE("MINIMUM", NODE_MATH_MINIMUM)
  PSYCLES_CYCLES_MATH_TYPE("MAXIMUM", NODE_MATH_MAXIMUM)
  PSYCLES_CYCLES_MATH_TYPE("LESS_THAN", NODE_MATH_LESS_THAN)
  PSYCLES_CYCLES_MATH_TYPE("GREATER_THAN", NODE_MATH_GREATER_THAN)
  PSYCLES_CYCLES_MATH_TYPE("SIGN", NODE_MATH_SIGN)
  PSYCLES_CYCLES_MATH_TYPE("COMPARE", NODE_MATH_COMPARE)
  PSYCLES_CYCLES_MATH_TYPE("SMOOTH_MIN", NODE_MATH_SMOOTH_MIN)
  PSYCLES_CYCLES_MATH_TYPE("SMOOTH_MAX", NODE_MATH_SMOOTH_MAX)
  PSYCLES_CYCLES_MATH_TYPE("ROUND", NODE_MATH_ROUND)
  PSYCLES_CYCLES_MATH_TYPE("FLOOR", NODE_MATH_FLOOR)
  PSYCLES_CYCLES_MATH_TYPE("CEIL", NODE_MATH_CEIL)
  PSYCLES_CYCLES_MATH_TYPE("TRUNC", NODE_MATH_TRUNC)
  PSYCLES_CYCLES_MATH_TYPE("FRACT", NODE_MATH_FRACTION)
  PSYCLES_CYCLES_MATH_TYPE("MODULO", NODE_MATH_MODULO)
  PSYCLES_CYCLES_MATH_TYPE("FLOORED_MODULO", NODE_MATH_FLOORED_MODULO)
  PSYCLES_CYCLES_MATH_TYPE("WRAP", NODE_MATH_WRAP)
  PSYCLES_CYCLES_MATH_TYPE("SNAP", NODE_MATH_SNAP)
  PSYCLES_CYCLES_MATH_TYPE("PINGPONG", NODE_MATH_PINGPONG)
  PSYCLES_CYCLES_MATH_TYPE("SINE", NODE_MATH_SINE)
  PSYCLES_CYCLES_MATH_TYPE("COSINE", NODE_MATH_COSINE)
  PSYCLES_CYCLES_MATH_TYPE("TANGENT", NODE_MATH_TANGENT)
  PSYCLES_CYCLES_MATH_TYPE("ARCSINE", NODE_MATH_ARCSINE)
  PSYCLES_CYCLES_MATH_TYPE("ARCCOSINE", NODE_MATH_ARCCOSINE)
  PSYCLES_CYCLES_MATH_TYPE("ARCTANGENT", NODE_MATH_ARCTANGENT)
  PSYCLES_CYCLES_MATH_TYPE("ARCTAN2", NODE_MATH_ARCTAN2)
  PSYCLES_CYCLES_MATH_TYPE("SINH", NODE_MATH_SINH)
  PSYCLES_CYCLES_MATH_TYPE("COSH", NODE_MATH_COSH)
  PSYCLES_CYCLES_MATH_TYPE("TANH", NODE_MATH_TANH)
  PSYCLES_CYCLES_MATH_TYPE("RADIANS", NODE_MATH_RADIANS)
  PSYCLES_CYCLES_MATH_TYPE("DEGREES", NODE_MATH_DEGREES)
#undef PSYCLES_CYCLES_MATH_TYPE
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeClampType>
clamp_type(const GraphNode *node) noexcept {
  const auto mode = string_property(node, "Mode");
  if (mode == "MINMAX") {
    return NODE_CLAMP_MINMAX;
  }
  if (mode == "RANGE") {
    return NODE_CLAMP_RANGE;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<NodeCombSepColorType>
combsep_color_type(const GraphNode *node) noexcept {
  const auto mode = string_property(node, "Mode");
  if (mode == "RGB") {
    return NODE_COMBSEP_COLOR_RGB;
  }
  if (mode == "HSV") {
    return NODE_COMBSEP_COLOR_HSV;
  }
  if (mode == "HSL") {
    return NODE_COMBSEP_COLOR_HSL;
  }
  return std::nullopt;
}

[[nodiscard]] float cycles_max(float a, float b) noexcept {
  return a > b ? a : b;
}

[[nodiscard]] float cycles_min(float a, float b) noexcept {
  return a < b ? a : b;
}

[[nodiscard]] float cycles_clamp(float value, float minimum,
                                 float maximum) noexcept {
  return cycles_min(cycles_max(value, minimum), maximum);
}

[[nodiscard]] Vec3f cycles_gamma(Vec3f color, float gamma) noexcept {
  if (gamma == 0.0f) {
    return {1.0f, 1.0f, 1.0f};
  }
  if (color.x > 0.0f) {
    color.x = std::pow(color.x, gamma);
  }
  if (color.y > 0.0f) {
    color.y = std::pow(color.y, gamma);
  }
  if (color.z > 0.0f) {
    color.z = std::pow(color.z, gamma);
  }
  return color;
}

[[nodiscard]] Vec3f cycles_brightness_contrast(Vec3f color,
                                                float brightness,
                                                float contrast) noexcept {
  const auto a = 1.0f + contrast;
  const auto b = brightness - contrast * 0.5f;
  color.x = cycles_max(a * color.x + b, 0.0f);
  color.y = cycles_max(a * color.y + b, 0.0f);
  color.z = cycles_max(a * color.z + b, 0.0f);
  return color;
}

[[nodiscard]] Vec3f cycles_invert(Vec3f color, float factor) noexcept {
  const Vec3f inverse{1.0f - color.x, 1.0f - color.y, 1.0f - color.z};
  return {color.x + factor * (inverse.x - color.x),
          color.y + factor * (inverse.y - color.y),
          color.z + factor * (inverse.z - color.z)};
}

[[nodiscard]] Vec3f cycles_rgb_to_hsv(Vec3f rgb) noexcept {
  const auto cmax = std::fmax(rgb.x, std::fmax(rgb.y, rgb.z));
  const auto cmin = cycles_min(rgb.x, cycles_min(rgb.y, rgb.z));
  const auto cdelta = cmax - cmin;
  auto h = 0.0f;
  auto s = 0.0f;
  const auto v = cmax;

  if (cmax != 0.0f) {
    s = cdelta / cmax;
  }
  if (s != 0.0f) {
    const Vec3f c{(cmax - rgb.x) / cdelta, (cmax - rgb.y) / cdelta,
                  (cmax - rgb.z) / cdelta};
    if (rgb.x == cmax) {
      h = c.z - c.y;
    } else if (rgb.y == cmax) {
      h = 2.0f + c.x - c.z;
    } else {
      h = 4.0f + c.y - c.x;
    }
    h /= 6.0f;
    if (h < 0.0f) {
      h += 1.0f;
    }
  }
  return {h, s, v};
}

[[nodiscard]] Vec3f cycles_hsv_to_rgb(Vec3f hsv) noexcept {
  auto h = hsv.x;
  const auto s = hsv.y;
  const auto v = hsv.z;
  if (s == 0.0f) {
    return {v, v, v};
  }
  if (h == 1.0f) {
    h = 0.0f;
  }
  h *= 6.0f;
  const auto i = std::floor(h);
  const auto f = h - i;
  const auto p = v * (1.0f - s);
  const auto q = v * (1.0f - s * f);
  const auto t = v * (1.0f - s * (1.0f - f));
  if (i == 0.0f) {
    return {v, t, p};
  }
  if (i == 1.0f) {
    return {q, v, p};
  }
  if (i == 2.0f) {
    return {p, v, t};
  }
  if (i == 3.0f) {
    return {p, q, v};
  }
  if (i == 4.0f) {
    return {t, p, v};
  }
  return {v, p, q};
}

[[nodiscard]] Vec3f cycles_rgb_to_hsl(Vec3f rgb) noexcept {
  const auto cmax = std::fmax(rgb.x, std::fmax(rgb.y, rgb.z));
  const auto cmin = cycles_min(rgb.x, cycles_min(rgb.y, rgb.z));
  auto h = 0.0f;
  auto s = 0.0f;
  const auto l = cycles_min(1.0f, (cmax + cmin) / 2.0f);
  if (cmax != cmin) {
    const auto cdelta = cmax - cmin;
    s = l > 0.5f ? cdelta / (2.0f - cmax - cmin) : cdelta / (cmax + cmin);
    if (cmax == rgb.x) {
      h = (rgb.y - rgb.z) / cdelta + (rgb.y < rgb.z ? 6.0f : 0.0f);
    } else if (cmax == rgb.y) {
      h = (rgb.z - rgb.x) / cdelta + 2.0f;
    } else {
      h = (rgb.x - rgb.y) / cdelta + 4.0f;
    }
  }
  h /= 6.0f;
  return {h, s, l};
}

[[nodiscard]] Vec3f cycles_hsl_to_rgb(Vec3f hsl) noexcept {
  auto nr = std::fabs(hsl.x * 6.0f - 3.0f) - 1.0f;
  auto ng = 2.0f - std::fabs(hsl.x * 6.0f - 2.0f);
  auto nb = 2.0f - std::fabs(hsl.x * 6.0f - 4.0f);
  nr = cycles_clamp(nr, 0.0f, 1.0f);
  nb = cycles_clamp(nb, 0.0f, 1.0f);
  ng = cycles_clamp(ng, 0.0f, 1.0f);
  const auto chroma = (1.0f - std::fabs(2.0f * hsl.z - 1.0f)) * hsl.y;
  return {(nr - 0.5f) * chroma + hsl.z, (ng - 0.5f) * chroma + hsl.z,
          (nb - 0.5f) * chroma + hsl.z};
}

[[nodiscard]] Vec3f cycles_combine_color(NodeCombSepColorType type,
                                         Vec3f color) noexcept {
  switch (type) {
  case NODE_COMBSEP_COLOR_HSV:
    return cycles_hsv_to_rgb(color);
  case NODE_COMBSEP_COLOR_HSL:
    return cycles_hsl_to_rgb(color);
  case NODE_COMBSEP_COLOR_RGB:
  default:
    return color;
  }
}

[[nodiscard]] Vec3f cycles_separate_color(NodeCombSepColorType type,
                                          Vec3f color) noexcept {
  switch (type) {
  case NODE_COMBSEP_COLOR_HSV:
    return cycles_rgb_to_hsv(color);
  case NODE_COMBSEP_COLOR_HSL:
    return cycles_rgb_to_hsl(color);
  case NODE_COMBSEP_COLOR_RGB:
  default:
    return color;
  }
}

class UnsupportedNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    compiler.fail("Cycles SVM node family is not migrated: " + type);
  }
};

class UnsupportedVolumeNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    compiler.fail("Cycles SVM node family is not migrated: " + type);
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

class UnsupportedBsdfNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    compiler.fail("Cycles SVM node family is not migrated: " + type);
  }

  [[nodiscard]] bool equals(const GraphNode &) const noexcept override {
    return false;
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }

  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    return GraphNode::get_feature() | kernel_feature_node_bsdf;
  }

  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_CLOSURE_BSDF;
  }
};

class NullNode final : public GraphNode {
public:
  void compile(SVMCompiler &) override {}
};

class AddClosureNode final : public GraphNode {
public:
  void compile(SVMCompiler &) override {}

  void constant_fold(const ConstantFolder &folder) override {
    auto *closure1 = input("Closure1");
    auto *closure2 = input("Closure2");
    if (closure1->link == nullptr) {
      folder.bypass_or_discard(closure2);
    } else if (closure2->link == nullptr) {
      folder.bypass_or_discard(closure1);
    }
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }
};

class MixClosureNode final : public GraphNode {
public:
  void compile(SVMCompiler &) override {}

  void constant_fold(const ConstantFolder &folder) override {
    auto *factor = input("Fac");
    auto *closure1 = input("Closure1");
    auto *closure2 = input("Closure2");
    if (closure1->link == closure2->link) {
      folder.bypass_or_discard(closure1);
    } else if (factor->link == nullptr) {
      const auto value =
          literal<float>(factor, contract::SocketType::floating);
      if (!value) {
        return;
      }
      if (*value <= 0.0f) {
        folder.bypass_or_discard(closure1);
      } else if (*value >= 1.0f) {
        folder.bypass_or_discard(closure2);
      }
    }
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }
};

class OutputNode final : public GraphNode {
public:
  [[nodiscard]] bool equals(const GraphNode &) const noexcept override {
    return false;
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    return true;
  }

  void compile(SVMCompiler &compiler) override {
    if (compiler.output_type() == SHADER_TYPE_DISPLACEMENT) {
      auto *displacement = input("Displacement");
      if (displacement != nullptr && displacement->link != nullptr) {
        compiler.add_node(
            this, NODE_SET_DISPLACEMENT,
            SVMNodeSetDisplacement{
                .fac_offset = compiler.input_link("Displacement"),
                ._pad = {0u, 0u, 0u}});
      }
    }
  }
};

class GeometryNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_GEOMETRY;
  }

  void compile(SVMCompiler &compiler) override {
    struct Output {
      std::string_view name;
      NodeGeometry geometry;
    };
    static constexpr Output outputs[] = {
        {"Position", NODE_GEOM_P},
        {"Normal", NODE_GEOM_N},
        {"Tangent", NODE_GEOM_T},
        {"True Normal", NODE_GEOM_Ng},
        {"Incoming", NODE_GEOM_I},
        {"Parametric", NODE_GEOM_uv},
    };
    for (const auto &item : outputs) {
      auto *socket = output(item.name);
      if (socket == nullptr || socket->links.empty()) {
        continue;
      }
      compiler.add_node(
          this, NODE_GEOMETRY,
          SVMNodeGeometry{
              .geom_type = item.geometry,
              .bump_offset = NODE_BUMP_OFFSET_CENTER,
              .store_derivatives =
                  static_cast<std::uint8_t>(need_derivatives),
              .out_offset = compiler.output(item.name),
              .bump_filter_width = 0.0f},
          item.geometry != NODE_GEOM_N && item.geometry != NODE_GEOM_Ng);
    }
    if (auto *socket = output("Backfacing");
        socket != nullptr && !socket->links.empty()) {
      compiler.add_node(
          this, NODE_LIGHT_PATH,
          SVMNodeLightPath{.path_type = NODE_LP_backfacing,
                           .out_offset = compiler.output("Backfacing"),
                           ._pad = {0u, 0u, 0u}});
    }
    for (const auto name : {"Pointiness", "Random Per Island"}) {
      if (const auto *socket = output(name);
          socket != nullptr && !socket->links.empty()) {
        compiler.fail("Cycles Geometry output is not migrated: " +
                      std::string{name});
        return;
      }
    }
  }
};

class MixClosureWeightNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    compiler.add_node(
        this, NODE_MIX_CLOSURE,
        SVMNodeMixClosure{.fac = compiler.input_float("Fac"),
                          .in_weight_offset = compiler.input_link("Weight"),
                          .weight1_offset = compiler.output("Weight1"),
                          .weight2_offset = compiler.output("Weight2"),
                          ._pad = {0u}});
  }
};

class MathNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    const auto operation = math_type(this);
    if (!operation) {
      compiler.fail("Cycles Math operation is not migrated exactly");
      return;
    }
    compiler.add_node(
        this, NODE_MATH,
        SVMNodeMath{.math_type = *operation,
                    .value1 = compiler.input_float("Value1"),
                    .value2 = compiler.input_float("Value2"),
                    .value3 = compiler.input_float("Value3"),
                    .result_offset = compiler.output("Value"),
                    ._pad = {0u, 0u, 0u}});
  }

  void constant_fold(const ConstantFolder &folder) override {
    const auto operation = math_type(this);
    if (!operation) {
      return;
    }
    if (folder.all_inputs_constant()) {
      const auto value1 =
          literal<float>(input("Value1"), contract::SocketType::floating);
      const auto value2 =
          literal<float>(input("Value2"), contract::SocketType::floating);
      const auto value3 =
          literal<float>(input("Value3"), contract::SocketType::floating);
      if (value1 && value2 && value3) {
        folder.make_constant(
            svm_math(*operation, *value1, *value2, *value3));
      }
    } else {
      folder.fold_math(*operation);
    }
  }

  [[nodiscard]] bool is_linear_operation() const noexcept override {
    const auto operation = math_type(this);
    if (!operation) {
      return false;
    }
    switch (*operation) {
      case NODE_MATH_ADD:
      case NODE_MATH_SUBTRACT:
      case NODE_MATH_MULTIPLY:
      case NODE_MATH_MULTIPLY_ADD:
        break;
      case NODE_MATH_DIVIDE:
        return input("Value2")->link == nullptr;
      default:
        return false;
    }
    auto variable_inputs = 0u;
    for (const auto &input_socket : inputs) {
      variable_inputs += input_socket.link != nullptr ? 1u : 0u;
    }
    return variable_inputs <= 1u;
  }
};

class InvertNode final : public GraphNode {
public:
  void constant_fold(const ConstantFolder &folder) override {
    auto *color = input("Color");
    auto *factor = input("Fac");
    if (factor == nullptr || factor->link != nullptr) {
      return;
    }
    const auto factor_value =
        literal<float>(factor, contract::SocketType::floating);
    if (!factor_value) {
      return;
    }
    if (color != nullptr && color->link == nullptr) {
      const auto color_value =
          literal<Vec3f>(color, contract::SocketType::color);
      if (color_value) {
        folder.make_constant(cycles_invert(*color_value, *factor_value));
      }
    } else if (*factor_value == 0.0f) {
      folder.bypass(color->link);
    }
  }

  void compile(SVMCompiler &compiler) override {
    compiler.add_node(
        this, NODE_INVERT,
        SVMNodeInvert{.color = compiler.input_float3("Color"),
                      .fac = compiler.input_float("Fac"),
                      .out_offset = compiler.output("Color"),
                      ._pad = {0u, 0u, 0u}});
  }
};

class GammaNode final : public GraphNode {
public:
  void constant_fold(const ConstantFolder &folder) override {
    auto *color = input("Color");
    auto *gamma = input("Gamma");
    if (folder.all_inputs_constant()) {
      const auto color_value =
          literal<Vec3f>(color, contract::SocketType::color);
      const auto gamma_value =
          literal<float>(gamma, contract::SocketType::floating);
      if (color_value && gamma_value) {
        folder.make_constant(cycles_gamma(*color_value, *gamma_value));
      }
    } else if (folder.is_one(color) || folder.is_zero(gamma)) {
      folder.make_one();
    } else if (folder.is_one(gamma)) {
      static_cast<void>(folder.try_bypass_or_make_constant(color, false));
    }
  }

  void compile(SVMCompiler &compiler) override {
    compiler.add_node(
        this, NODE_GAMMA,
        SVMNodeGamma{.color = compiler.input_float3("Color"),
                     .gamma = compiler.input_float("Gamma"),
                     .out_offset = compiler.output("Color"),
                     ._pad = {0u, 0u, 0u}});
  }
};

class BrightContrastNode final : public GraphNode {
public:
  void constant_fold(const ConstantFolder &folder) override {
    if (!folder.all_inputs_constant()) {
      return;
    }
    const auto color =
        literal<Vec3f>(input("Color"), contract::SocketType::color);
    const auto bright =
        literal<float>(input("Bright"), contract::SocketType::floating);
    const auto contrast =
        literal<float>(input("Contrast"), contract::SocketType::floating);
    if (color && bright && contrast) {
      folder.make_constant(
          cycles_brightness_contrast(*color, *bright, *contrast));
    }
  }

  void compile(SVMCompiler &compiler) override {
    compiler.add_node(
        this, NODE_BRIGHTCONTRAST,
        SVMNodeBrightContrast{.color = compiler.input_float3("Color"),
                              .bright = compiler.input_float("Bright"),
                              .contrast = compiler.input_float("Contrast"),
                              .out_offset = compiler.output("Color"),
                              ._pad = {0u, 0u, 0u}});
  }
};

class HSVNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    compiler.add_node(
        this, NODE_HSV,
        SVMNodeHSV{.color = compiler.input_float3("Color"),
                   .hue = compiler.input_float("Hue"),
                   .sat = compiler.input_float("Saturation"),
                   .val = compiler.input_float("Value"),
                   .fac = compiler.input_float("Fac"),
                   .out_color_offset = compiler.output("Color"),
                   ._pad = {0u, 0u, 0u}});
  }
};

class ClampNode final : public GraphNode {
public:
  void constant_fold(const ConstantFolder &folder) override {
    if (!folder.all_inputs_constant()) {
      return;
    }
    const auto type = clamp_type(this);
    const auto value =
        literal<float>(input("Value"), contract::SocketType::floating);
    const auto minimum =
        literal<float>(input("Min"), contract::SocketType::floating);
    const auto maximum =
        literal<float>(input("Max"), contract::SocketType::floating);
    if (!type || !value || !minimum || !maximum) {
      return;
    }
    if (*type == NODE_CLAMP_RANGE && *minimum > *maximum) {
      folder.make_constant(cycles_clamp(*value, *maximum, *minimum));
    } else {
      folder.make_constant(cycles_clamp(*value, *minimum, *maximum));
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto type = clamp_type(this);
    if (!type) {
      compiler.fail("Cycles Clamp mode is not migrated exactly");
      return;
    }
    compiler.add_node(
        this, NODE_CLAMP,
        SVMNodeClamp{.clamp_type = *type,
                     .min = compiler.input_float("Min"),
                     .max = compiler.input_float("Max"),
                     .value = compiler.input_float("Value"),
                     .result_offset = compiler.output("Result"),
                     ._pad = {0u, 0u, 0u}});
  }
};

class SeparateColorNode final : public GraphNode {
public:
  void constant_fold(const ConstantFolder &folder) override {
    if (!folder.all_inputs_constant()) {
      return;
    }
    const auto type = combsep_color_type(this);
    const auto color =
        literal<Vec3f>(input("Color"), contract::SocketType::color);
    if (!type || !color) {
      return;
    }
    const auto separated = cycles_separate_color(*type, *color);
    if (folder.output == output("Red")) {
      folder.make_constant(separated.x);
    } else if (folder.output == output("Green")) {
      folder.make_constant(separated.y);
    } else if (folder.output == output("Blue")) {
      folder.make_constant(separated.z);
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto type = combsep_color_type(this);
    if (!type) {
      compiler.fail("Cycles Separate Color mode is not migrated exactly");
      return;
    }
    compiler.add_node(
        this, NODE_SEPARATE_COLOR,
        SVMNodeSeparateColor{.color_type = *type,
                             .color = compiler.input_float3("Color"),
                             .red_offset = compiler.output("Red"),
                             .green_offset = compiler.output("Green"),
                             .blue_offset = compiler.output("Blue"),
                             ._pad = {0u}});
  }
};

class CombineColorNode final : public GraphNode {
public:
  void constant_fold(const ConstantFolder &folder) override {
    if (!folder.all_inputs_constant()) {
      return;
    }
    const auto type = combsep_color_type(this);
    const auto red =
        literal<float>(input("Red"), contract::SocketType::floating);
    const auto green =
        literal<float>(input("Green"), contract::SocketType::floating);
    const auto blue =
        literal<float>(input("Blue"), contract::SocketType::floating);
    if (type && red && green && blue) {
      folder.make_constant(cycles_combine_color(*type, {*red, *green, *blue}));
    }
  }

  void compile(SVMCompiler &compiler) override {
    const auto type = combsep_color_type(this);
    if (!type) {
      compiler.fail("Cycles Combine Color mode is not migrated exactly");
      return;
    }
    compiler.add_node(
        this, NODE_COMBINE_COLOR,
        SVMNodeCombineColor{.color_type = *type,
                            .red = compiler.input_float("Red"),
                            .green = compiler.input_float("Green"),
                            .blue = compiler.input_float("Blue"),
                            .color_offset = compiler.output("Color"),
                            ._pad = {0u, 0u, 0u}});
  }
};

class BsdfNode : public GraphNode {
private:
  ClosureType _closure;

protected:
  explicit BsdfNode(ClosureType closure) noexcept : _closure{closure} {}

  template<SvmPayload T>
  void compile_bsdf(SVMCompiler &compiler, const T &data) {
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
            .closure_type = _closure,
            .mix_weight_offset = compiler.closure_mix_weight_offset(),
            ._pad = {0u, 0u, 0u}},
        data);
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

} // namespace

void GraphNode::expand(CyclesGraph &) {}

void GraphNode::constant_fold(const ConstantFolder &) {}

void GraphNode::simplify_settings() {}

std::uint32_t GraphNode::get_feature() const noexcept { return 0u; }

ShaderNodeType GraphNode::shader_node_type() const noexcept {
  return NODE_NONE;
}

bool GraphNode::equals(const GraphNode &other) const noexcept {
  if (type != other.type || properties != other.properties ||
      inputs.size() != other.inputs.size()) {
    return false;
  }
  for (auto index = std::size_t{}; index < inputs.size(); ++index) {
    const auto &lhs = inputs[index];
    const auto &rhs = other.inputs[index];
    if (lhs.name != rhs.name || lhs.type != rhs.type) {
      return false;
    }
    if (lhs.link == nullptr && rhs.link == nullptr) {
      if (lhs.value != rhs.value) {
        return false;
      }
    } else if (lhs.link == nullptr || rhs.link == nullptr ||
               lhs.link != rhs.link) {
      return false;
    }
  }
  return true;
}

bool GraphNode::has_volume_support() const noexcept { return false; }

bool GraphNode::is_linear_operation() const noexcept { return false; }

std::unique_ptr<GraphNode> make_graph_node(std::string_view type) {
  if (type == "cycles.synthetic.output") {
    return std::make_unique<OutputNode>();
  }
  if (type == cycles_synthetic_geometry || type == node_type::geometry) {
    return std::make_unique<GeometryNode>();
  }
  if (type == cycles_synthetic_mix_closure_weight) {
    return std::make_unique<MixClosureWeightNode>();
  }
  if (type == cycles_synthetic_math || type == node_type::math) {
    return std::make_unique<MathNode>();
  }
  if (type == node_type::invert_color) {
    return std::make_unique<InvertNode>();
  }
  if (type == node_type::gamma_color) {
    return std::make_unique<GammaNode>();
  }
  if (type == node_type::brightness_contrast) {
    return std::make_unique<BrightContrastNode>();
  }
  if (type == node_type::hue_saturation) {
    return std::make_unique<HSVNode>();
  }
  if (type == node_type::clamp_range) {
    return std::make_unique<ClampNode>();
  }
  if (type == node_type::separate_color) {
    return std::make_unique<SeparateColorNode>();
  }
  if (type == node_type::combine_color) {
    return std::make_unique<CombineColorNode>();
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
  if (type == node_type::emission) {
    return std::make_unique<EmissionNode>();
  }
  if (type == node_type::principled_bsdf ||
      type == node_type::subsurface_scattering ||
      type == node_type::glossy_bsdf || type == node_type::metallic_bsdf ||
      type == node_type::sheen_bsdf || type == node_type::hair_bsdf ||
      type == node_type::glass_bsdf || type == node_type::refraction_bsdf) {
    return std::make_unique<UnsupportedBsdfNode>();
  }
  if (type == node_type::null_closure || type == node_type::null_volume) {
    return std::make_unique<NullNode>();
  }
  if (type == node_type::add_closure || type == node_type::add_volume) {
    return std::make_unique<AddClosureNode>();
  }
  if (type == node_type::mix_closure || type == node_type::mix_volume) {
    return std::make_unique<MixClosureNode>();
  }
  if (type == node_type::volume_absorption ||
      type == node_type::volume_scatter ||
      type == node_type::volume_coefficients ||
      type == node_type::volume_emission ||
      type == node_type::principled_volume) {
    return std::make_unique<UnsupportedVolumeNode>();
  }
  return std::make_unique<UnsupportedNode>();
}

} // namespace psycles::compiler::cycles_svm

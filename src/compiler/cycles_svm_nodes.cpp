/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

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

class UnsupportedNode final : public GraphNode {
public:
  void compile(SVMCompiler &compiler) override {
    compiler.fail("Cycles SVM node family is not migrated: " + type);
  }
};

class EmptyNode final : public GraphNode {
public:
  void compile(SVMCompiler &) override {}
};

class OutputNode final : public GraphNode {
public:
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
        {"GeometricNormal", NODE_GEOM_Ng},
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
    for (const auto name : {"Pointiness", "RandomPerIsland"}) {
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
    if (type != cycles_synthetic_math) {
      compiler.fail("Cycles Math operation is not migrated exactly");
      return;
    }
    compiler.add_node(
        this, NODE_MATH,
        SVMNodeMath{.math_type = NODE_MATH_MULTIPLY,
                    .value1 = compiler.input_float("Value1"),
                    .value2 = compiler.input_float("Value2"),
                    .value3 = compiler.input_float("Value3"),
                    .result_offset = compiler.output("Value"),
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

std::uint32_t GraphNode::get_feature() const noexcept { return 0u; }

ShaderNodeType GraphNode::shader_node_type() const noexcept {
  return NODE_NONE;
}

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
  if (type == node_type::null_closure || type == node_type::null_volume ||
      type == node_type::add_closure || type == node_type::mix_closure ||
      type == node_type::add_volume || type == node_type::mix_volume) {
    return std::make_unique<EmptyNode>();
  }
  return std::make_unique<UnsupportedNode>();
}

} // namespace psycles::compiler::cycles_svm

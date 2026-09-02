/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_info_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <array>

namespace psycles::compiler::cycles_svm {
namespace {

template<typename InfoType>
struct InfoOutput {
  std::string_view name;
  InfoType info_type;
};

constexpr auto object_outputs = std::array{
    InfoOutput{"Location", NODE_INFO_OB_LOCATION},
    InfoOutput{"Color", NODE_INFO_OB_COLOR},
    InfoOutput{"Alpha", NODE_INFO_OB_ALPHA},
    InfoOutput{"Object Index", NODE_INFO_OB_INDEX},
    InfoOutput{"Material Index", NODE_INFO_MAT_INDEX},
    InfoOutput{"Random", NODE_INFO_OB_RANDOM},
};

constexpr auto particle_outputs = std::array{
    InfoOutput{"Index", NODE_INFO_PAR_INDEX},
    InfoOutput{"Random", NODE_INFO_PAR_RANDOM},
    InfoOutput{"Age", NODE_INFO_PAR_AGE},
    InfoOutput{"Lifetime", NODE_INFO_PAR_LIFETIME},
    InfoOutput{"Location", NODE_INFO_PAR_LOCATION},
    InfoOutput{"Size", NODE_INFO_PAR_SIZE},
    InfoOutput{"Velocity", NODE_INFO_PAR_VELOCITY},
    InfoOutput{"Angular Velocity", NODE_INFO_PAR_ANGULAR_VELOCITY},
};

[[nodiscard]] bool output_is_live(const GraphNode *node,
                                  std::string_view name) noexcept {
  const auto *out = node->output(name);
  return out != nullptr && !out->links.empty();
}

class ObjectInfoNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_OBJECT_INFO;
  }

  void compile(SVMCompiler &compiler) override {
    for (const auto &[name, info_type] : object_outputs) {
      if (output_is_live(this, name)) {
        compiler.add_node(
            this, NODE_OBJECT_INFO,
            SVMNodeObjectInfo{.info_type = info_type,
                              .out_offset = compiler.output(name),
                              ._pad = {0u, 0u, 0u}});
      }
    }
  }
};

class ParticleInfoNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_PARTICLE_INFO;
  }

  void attributes(const GraphAttributeContext &context,
                  AttributeRequestSet &requests) const override {
    if (std::ranges::any_of(particle_outputs, [this](const auto &item) {
          return output_is_live(this, item.name);
        })) {
      requests.add(ATTR_STD_PARTICLE);
    }
    GraphNode::attributes(context, requests);
  }

  void compile(SVMCompiler &compiler) override {
    for (const auto &[name, info_type] : particle_outputs) {
      if (output_is_live(this, name)) {
        compiler.add_node(
            this, NODE_PARTICLE_INFO,
            SVMNodeParticleInfo{.info_type = info_type,
                                .out_offset = compiler.output(name),
                                ._pad = {0u, 0u, 0u}});
      }
    }
  }
};

class HairInfoNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_HAIR_INFO;
  }

  void attributes(const GraphAttributeContext &context,
                  AttributeRequestSet &requests) const override {
    if (context.has_surface_link()) {
      if (output_is_live(this, "Intercept")) {
        requests.add(ATTR_STD_CURVE_INTERCEPT);
      }
      if (output_is_live(this, "Length")) {
        requests.add(ATTR_STD_CURVE_LENGTH);
      }
      if (output_is_live(this, "Random")) {
        requests.add(ATTR_STD_CURVE_RANDOM);
      }
    }
    GraphNode::attributes(context, requests);
  }

  void compile(SVMCompiler &compiler) override {
    if (output_is_live(this, "Is Strand")) {
      compiler.add_node(
          this, NODE_HAIR_INFO,
          SVMNodeHairInfo{.info_type = NODE_INFO_CURVE_IS_STRAND,
                          .out_offset = compiler.output("Is Strand"),
                          ._pad = {0u, 0u, 0u}});
    }
    if (output_is_live(this, "Intercept")) {
      compiler.add_node(
          this, NODE_ATTR,
          SVMNodeAttr{
              .attr = static_cast<std::int32_t>(
                  compiler.attribute(ATTR_STD_CURVE_INTERCEPT)),
              .out_offset = compiler.output("Intercept"),
              .output_type = NODE_ATTR_OUTPUT_FLOAT,
              .bump_offset = NODE_BUMP_OFFSET_CENTER,
              .store_derivatives = 0u,
              .bump_filter_width = 0.0f});
    }
    if (output_is_live(this, "Length")) {
      compiler.add_node(
          this, NODE_ATTR,
          SVMNodeAttr{
              .attr = static_cast<std::int32_t>(
                  compiler.attribute(ATTR_STD_CURVE_LENGTH)),
              .out_offset = compiler.output("Length"),
              .output_type = NODE_ATTR_OUTPUT_FLOAT,
              .bump_offset = NODE_BUMP_OFFSET_CENTER,
              .store_derivatives = 0u,
              .bump_filter_width = 0.0f});
    }
    if (output_is_live(this, "Thickness")) {
      compiler.add_node(
          this, NODE_HAIR_INFO,
          SVMNodeHairInfo{.info_type = NODE_INFO_CURVE_THICKNESS,
                          .out_offset = compiler.output("Thickness"),
                          ._pad = {0u, 0u, 0u}});
    }
    if (output_is_live(this, "Tangent Normal")) {
      compiler.add_node(
          this, NODE_HAIR_INFO,
          SVMNodeHairInfo{.info_type = NODE_INFO_CURVE_TANGENT_NORMAL,
                          .out_offset = compiler.output("Tangent Normal"),
                          ._pad = {0u, 0u, 0u}});
    }
    if (output_is_live(this, "Random")) {
      compiler.add_node(
          this, NODE_ATTR,
          SVMNodeAttr{
              .attr = static_cast<std::int32_t>(
                  compiler.attribute(ATTR_STD_CURVE_RANDOM)),
              .out_offset = compiler.output("Random"),
              .output_type = NODE_ATTR_OUTPUT_FLOAT,
              .bump_offset = NODE_BUMP_OFFSET_CENTER,
              .store_derivatives = 0u,
              .bump_filter_width = 0.0f});
    }
  }
};

class PointInfoNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_POINT_INFO;
  }

  void attributes(const GraphAttributeContext &context,
                  AttributeRequestSet &requests) const override {
    if (context.has_surface_link() && output_is_live(this, "Random")) {
      requests.add(ATTR_STD_POINT_RANDOM);
    }
    GraphNode::attributes(context, requests);
  }

  void compile(SVMCompiler &compiler) override {
    if (output_is_live(this, "Position")) {
      compiler.add_node(
          this, NODE_POINT_INFO,
          SVMNodePointInfo{.info_type = NODE_INFO_POINT_POSITION,
                           .out_offset = compiler.output("Position"),
                           ._pad = {0u, 0u, 0u}});
    }
    if (output_is_live(this, "Radius")) {
      compiler.add_node(
          this, NODE_POINT_INFO,
          SVMNodePointInfo{.info_type = NODE_INFO_POINT_RADIUS,
                           .out_offset = compiler.output("Radius"),
                           ._pad = {0u, 0u, 0u}});
    }
    if (output_is_live(this, "Random")) {
      compiler.add_node(
          this, NODE_ATTR,
          SVMNodeAttr{
              .attr = static_cast<std::int32_t>(
                  compiler.attribute(ATTR_STD_POINT_RANDOM)),
              .out_offset = compiler.output("Random"),
              .output_type = NODE_ATTR_OUTPUT_FLOAT,
              .bump_offset = NODE_BUMP_OFFSET_CENTER,
              .store_derivatives = 0u,
              .bump_filter_width = 0.0f});
    }
  }
};

} // namespace

std::unique_ptr<GraphNode> make_info_graph_node(std::string_view type) {
  if (type == node_type::object_info) {
    return std::make_unique<ObjectInfoNode>();
  }
  if (type == node_type::particle_info) {
    return std::make_unique<ParticleInfoNode>();
  }
  if (type == node_type::hair_info) {
    return std::make_unique<HairInfoNode>();
  }
  if (type == node_type::point_info) {
    return std::make_unique<PointInfoNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm

/* SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_spectral_nodes.h"

#include "cycles_svm_compiler_internal.h"
#include "cycles_svm_constant_fold.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <array>
#include <optional>
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

// Exact Cycles 5.2.1 kernel/tables.h coefficients used by
// BlackbodyNode::constant_fold. This is compiler evaluation, not a reference
// renderer; dynamic inputs remain NODE_BLACKBODY bytecode.
constexpr std::array blackbody_r{
    Vec3f{1.61919106e+03f, -2.05010916e-03f, 5.02995757e+00f},
    Vec3f{2.48845471e+03f, -1.11330907e-03f, 3.22621544e+00f},
    Vec3f{3.34143193e+03f, -4.86551192e-04f, 1.76486769e+00f},
    Vec3f{4.09461742e+03f, -1.27446582e-04f, 7.25731635e-01f},
    Vec3f{4.67028036e+03f, 2.91258199e-05f, 1.26703442e-01f},
    Vec3f{4.59509185e+03f, 2.87495649e-05f, 1.50345020e-01f},
    Vec3f{3.78717450e+03f, 9.35907826e-06f, 3.99075871e-01f},
};

constexpr std::array blackbody_g{
    Vec3f{-4.88999748e+02f, 6.04330754e-04f, -7.55807526e-02f},
    Vec3f{-7.55994277e+02f, 3.16730098e-04f, 4.78306139e-01f},
    Vec3f{-1.02363977e+03f, 1.20223470e-04f, 9.36662319e-01f},
    Vec3f{-1.26571316e+03f, 4.87340896e-06f, 1.27054498e+00f},
    Vec3f{-1.42529332e+03f, -4.01150431e-05f, 1.43972784e+00f},
    Vec3f{-1.17554822e+03f, -2.16378048e-05f, 1.30408023e+00f},
    Vec3f{-5.00799571e+02f, -4.59832026e-06f, 1.09098763e+00f},
};

constexpr std::array<std::array<float, 4u>, 7u> blackbody_b{{
    {5.96945309e-11f, -4.85742887e-08f, -9.70622247e-05f,
     -4.07936148e-03f},
    {2.40430366e-11f, 5.55021075e-08f, -1.98503712e-04f,
     2.89312858e-02f},
    {-1.40949732e-11f, 1.89878968e-07f, -3.56632824e-04f,
     9.10767778e-02f},
    {-3.61460868e-11f, 2.84822009e-07f, -4.93211319e-04f,
     1.56723440e-01f},
    {-1.97075738e-11f, 1.75359352e-07f, -2.50542825e-04f,
     -2.22783266e-02f},
    {-1.61997957e-13f, -1.64216008e-08f, 3.86216271e-04f,
     -7.38077418e-01f},
    {6.72650283e-13f, -2.73078809e-08f, 4.24098264e-04f,
     -7.52335691e-01f},
}};

[[nodiscard]] Vec3f blackbody_color_rec709(float temperature) noexcept {
  if (temperature >= 12000.0f) {
    return {0.8262954810464208f, 0.9945080501520986f,
            1.566307710274283f};
  }
  if (temperature < 800.0f) {
    return {5.413294490189271f, -0.20319390035873933f,
            -0.0822535242887164f};
  }
  const auto index = temperature >= 6365.0f ? 6u
                     : temperature >= 3315.0f ? 5u
                     : temperature >= 1902.0f ? 4u
                     : temperature >= 1449.0f ? 3u
                     : temperature >= 1167.0f ? 2u
                     : temperature >= 965.0f  ? 1u
                                              : 0u;
  const auto &r = blackbody_r[index];
  const auto &g = blackbody_g[index];
  const auto &b = blackbody_b[index];
  const auto inverse_temperature = 1.0f / temperature;
  return {
      r.x * inverse_temperature + r.y * temperature + r.z,
      g.x * inverse_temperature + g.y * temperature + g.z,
      ((b[0u] * temperature + b[1u]) * temperature + b[2u]) * temperature +
          b[3u],
  };
}

class WavelengthNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_WAVELENGTH;
  }

  void compile(SVMCompiler &compiler) override {
    compiler.add_node(
        this, NODE_WAVELENGTH,
        SVMNodeWavelength{.wavelength = compiler.input_float("Wavelength"),
                          .color_offset = compiler.output("Color"),
                          ._pad = {0u, 0u, 0u}});
  }
};

class BlackbodyNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_BLACKBODY;
  }

  void constant_fold(const ConstantFolder &folder) override {
    if (!folder.all_inputs_constant()) {
      return;
    }
    const auto temperature =
        literal<float>(input("Temperature"), contract::SocketType::floating);
    if (!temperature) {
      return;
    }
    auto color = folder.graph->rec709_to_scene_linear(
        blackbody_color_rec709(*temperature));
    color.x = std::max(color.x, 0.0f);
    color.y = std::max(color.y, 0.0f);
    color.z = std::max(color.z, 0.0f);
    folder.make_constant(color);
  }

  void compile(SVMCompiler &compiler) override {
    compiler.add_node(
        this, NODE_BLACKBODY,
        SVMNodeBlackbody{.temperature = compiler.input_float("Temperature"),
                         .color_offset = compiler.output("Color"),
                         ._pad = {0u, 0u, 0u}});
  }
};

} // namespace

std::unique_ptr<GraphNode> make_spectral_graph_node(std::string_view type) {
  if (type == node_type::wavelength) {
    return std::make_unique<WavelengthNode>();
  }
  if (type == node_type::blackbody) {
    return std::make_unique<BlackbodyNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm

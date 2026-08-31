#include "blender_texture_coordinate_import_expectations.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>

namespace psycles::tests {
namespace {

[[nodiscard]] bool transform_near(
    const Mat4f &actual,
    const std::array<float, 16u> &expected,
    float tolerance) noexcept {
  for (auto index = std::size_t{}; index < expected.size(); ++index) {
    if (std::abs(actual.elements[index] - expected[index]) > tolerance) {
      return false;
    }
  }
  return true;
}

}// namespace

void expect_texture_coordinate_import(
    const contract::MaterialDesc &material,
    const compiler::SurfaceProgram &surface_program) {
  bool has_raw_reflection_coordinates = false;
  bool has_raw_projector_coordinates = false;
  bool rewrote_reflection_coordinates = false;
  static constexpr std::array raw_projector_transform{
      0.46153846f, 0.15384615f, -0.02884615f, 0.0f,
      -0.07692308f, 0.30769231f, 0.00480769f, 0.0f,
      0.0f, 0.0f, 0.25f, 0.0f,
      -0.61538462f, 0.46153846f, -0.08653846f, 1.0f};

  for (const auto &node : material.shader.nodes()) {
    if (node.type == compiler::node_type::texture_coordinate &&
        node.label == "Texture Coordinate") {
      const auto from_dupli = node.properties.find("FromDupli");
      has_raw_reflection_coordinates =
          from_dupli != node.properties.end() &&
          std::get<bool>(from_dupli->second.value);
    }
    if (node.type == compiler::node_type::texture_coordinate &&
        node.label == "Projector Coordinates") {
      const auto use_transform = node.properties.find("UseTransform");
      const auto object_transform = node.properties.find("ObjectTransform");
      has_raw_projector_coordinates =
          use_transform != node.properties.end() &&
          std::get<bool>(use_transform->second.value) &&
          object_transform != node.properties.end() &&
          transform_near(
              std::get<Mat4f>(object_transform->second.value),
              raw_projector_transform, 1.0e-7f);
    }
    rewrote_reflection_coordinates |=
        node.type == compiler::node_type::vector_math &&
        node.label.find("/ Reflection") != std::string::npos;
  }
  if (!has_raw_reflection_coordinates || !has_raw_projector_coordinates ||
      rewrote_reflection_coordinates) {
    throw std::runtime_error{
        "Texture Coordinate import did not preserve raw Cycles properties"};
  }

  bool has_projector_coordinates = false;
  static constexpr std::array expected_projector_coordinates{
      2.0f, -1.0f, 0.25f, 0.0f, 0.5f, 3.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 4.0f, 0.0f, 1.0f, -2.0f, 0.5f, 1.0f};
  for (const auto &instruction : surface_program.value_instructions()) {
    if (instruction.operation !=
            compiler::ValueOperation::object_position_with_transform ||
        instruction.static_table.size() !=
            expected_projector_coordinates.size()) {
      continue;
    }
    has_projector_coordinates = true;
    for (auto index = std::size_t{};
         index < expected_projector_coordinates.size(); ++index) {
      has_projector_coordinates &=
          std::abs(instruction.static_table[index] -
                   expected_projector_coordinates[index]) <= 1.0e-5f;
    }
  }
  if (!has_projector_coordinates) {
    throw std::runtime_error{
        "explicit Texture Coordinate object transform was not lowered"};
  }
}

}// namespace psycles::tests

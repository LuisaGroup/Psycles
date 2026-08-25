#include "graph_surface_internal.h"

#include <luisa/dsl/sugar.h>
#include <psycles/compiler/surface_execution_plan.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/cycles_color_nodes.h>
#include <psycles/luisa/cycles_noise.h>

#include <array>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] bool
supports_context_value(compiler::ValueOperation operation) noexcept {
  switch (operation) {
  case compiler::ValueOperation::multiply_color:
  case compiler::ValueOperation::hue_saturation:
  case compiler::ValueOperation::invert:
  case compiler::ValueOperation::gamma:
  case compiler::ValueOperation::brightness_contrast:
  case compiler::ValueOperation::blackbody:
  case compiler::ValueOperation::wavelength:
  case compiler::ValueOperation::surface_position:
  case compiler::ValueOperation::shading_normal:
  case compiler::ValueOperation::geometric_normal:
  case compiler::ValueOperation::incoming:
  case compiler::ValueOperation::tangent:
  case compiler::ValueOperation::uv:
  case compiler::ValueOperation::generated:
  case compiler::ValueOperation::object_position:
  case compiler::ValueOperation::object_position_with_transform:
  case compiler::ValueOperation::object_location:
  case compiler::ValueOperation::object_random:
  case compiler::ValueOperation::particle_index:
  case compiler::ValueOperation::particle_random:
  case compiler::ValueOperation::back_facing:
  case compiler::ValueOperation::pointiness:
  case compiler::ValueOperation::random_per_island:
  case compiler::ValueOperation::curve_is_strand:
  case compiler::ValueOperation::curve_intercept:
  case compiler::ValueOperation::curve_length:
  case compiler::ValueOperation::curve_thickness:
  case compiler::ValueOperation::curve_tangent_normal:
  case compiler::ValueOperation::curve_random:
  case compiler::ValueOperation::path_is_camera:
  case compiler::ValueOperation::path_is_shadow:
  case compiler::ValueOperation::path_is_diffuse:
  case compiler::ValueOperation::path_is_glossy:
  case compiler::ValueOperation::path_is_singular:
  case compiler::ValueOperation::path_is_reflection:
  case compiler::ValueOperation::path_is_transmission:
  case compiler::ValueOperation::path_is_volume_scatter:
  case compiler::ValueOperation::path_ray_length:
  case compiler::ValueOperation::path_ray_depth:
  case compiler::ValueOperation::path_diffuse_depth:
  case compiler::ValueOperation::path_glossy_depth:
  case compiler::ValueOperation::path_transparent_depth:
  case compiler::ValueOperation::path_transmission_depth:
  case compiler::ValueOperation::fresnel:
  case compiler::ValueOperation::layer_weight_fresnel:
  case compiler::ValueOperation::layer_weight_facing:
  case compiler::ValueOperation::mapping:
    return true;
  default:
    return false;
  }
}

class ContextValueNode final : public ValueNode {

public:
  using ValueNode::ValueNode;

  [[nodiscard]] SurfaceValueExpression
  evaluate(ValueEvaluationContext &context) const noexcept override {
    [[maybe_unused]] const auto &services = context.services;
    [[maybe_unused]] const auto &point = context.point;
    [[maybe_unused]] auto &result = context.result;
    const auto &instruction = this->instruction();
    Float4 value = make_float4(0.0f);
    switch (instruction.operation) {
    case compiler::ValueOperation::multiply_color: {
      auto t = clamp(
          scalar(instruction.operand(operand::mix::factor), result),
          0.0f,
          1.0f);
      auto a = vector(instruction.operand(operand::mix::a), result);
      value = make_float4(
          lerp(
              a,
              a * vector(instruction.operand(operand::mix::b), result),
              t),
          1.0f);
      break;
    }
    case compiler::ValueOperation::hue_saturation: {
      // Cycles' NODE_HSV contract: adjust in HSV space,
      // wrap hue with fract(), clamp only saturation, blend
      // with the unmodified input, and clamp the final RGB
      // against negative oversaturation artifacts. Fac is
      // intentionally not clamped.
      auto color = vector(
          instruction.operand(operand::hue_saturation::color), result);
      auto adjusted = rgb_to_hsv(services, color);
      adjusted.x = fract(
          adjusted.x +
          scalar(instruction.operand(operand::hue_saturation::hue), result) +
          0.5f);
      adjusted.y =
          clamp(
              adjusted.y *
                  scalar(
                      instruction.operand(
                          operand::hue_saturation::saturation),
                      result),
              0.0f,
              1.0f);
      adjusted.z *= scalar(
          instruction.operand(operand::hue_saturation::value), result);
      adjusted = hsv_to_rgb(services, adjusted);
      auto factor = scalar(
          instruction.operand(operand::hue_saturation::factor), result);
      value = make_float4(max(lerp(color, adjusted, factor), make_float3(0.0f)),
                          1.0f);
      break;
    }
    case compiler::ValueOperation::invert: {
      auto color = vector(
          instruction.operand(operand::color_factor::color), result);
      auto factor = scalar(
          instruction.operand(operand::color_factor::factor), result);
      value = make_float4(lerp(color, make_float3(1.0f) - color, factor), 1.0f);
      break;
    }
    case compiler::ValueOperation::gamma: {
      auto color = vector(instruction.operand(operand::gamma::color), result);
      auto exponent = scalar(
          instruction.operand(operand::gamma::exponent), result);
      auto adjusted = make_float3(
          select(color.x, pow(max(color.x, 0.0f), exponent), color.x > 0.0f),
          select(color.y, pow(max(color.y, 0.0f), exponent), color.y > 0.0f),
          select(color.z, pow(max(color.z, 0.0f), exponent), color.z > 0.0f));
      adjusted = select(adjusted, make_float3(1.0f), exponent == 0.0f);
      value = make_float4(adjusted, 1.0f);
      break;
    }
    case compiler::ValueOperation::brightness_contrast: {
      auto color = vector(
          instruction.operand(operand::brightness_contrast::color), result);
      auto brightness = scalar(
          instruction.operand(operand::brightness_contrast::brightness),
          result);
      auto contrast = scalar(
          instruction.operand(operand::brightness_contrast::contrast),
          result);
      auto a = 1.0f + contrast;
      auto b = brightness - contrast * 0.5f;
      value =
          make_float4(max(a * color + make_float3(b), make_float3(0.0f)), 1.0f);
      break;
    }
    case compiler::ValueOperation::blackbody:
      value = make_float4(
          max(services.rec709_to_rgb(cycles_color_nodes::blackbody_rec709(
                  scalar(
                      instruction.operand(operand::blackbody::temperature),
                      result))),
              make_float3(0.0f)),
          1.0f);
      break;
    case compiler::ValueOperation::wavelength:
      value = make_float4(
          max(services.xyz_to_rgb(cycles_color_nodes::wavelength_xyz(
                  scalar(
                      instruction.operand(operand::wavelength::nanometers),
                      result))) *
                  (1.0f / 2.52f),
              make_float3(0.0f)),
          1.0f);
      break;
    case compiler::ValueOperation::surface_position:
      value = make_float4(point.position, 1.0f);
      break;
    case compiler::ValueOperation::shading_normal:
      value = make_float4(point.shading_normal, 0.0f);
      break;
    case compiler::ValueOperation::geometric_normal:
      value = make_float4(point.geometric_normal, 0.0f);
      break;
    case compiler::ValueOperation::incoming:
      value = make_float4(point.incoming, 0.0f);
      break;
    case compiler::ValueOperation::tangent:
      value = make_float4(point.dpdu, 0.0f);
      break;
    case compiler::ValueOperation::uv:
      value = make_float4(point.uv.x, point.uv.y, 0.0f, 0.0f);
      if (context.svm_immediate_override != nullptr) {
        $if((*context.svm_immediate_override &
             compiler::surface_value_uv_named_immediate_bit) != 0u) {
          value = services.attribute(
              unsigned_integer(
                  instruction.operand(operand::uv::map), result),
              point).value;
        };
      } else if (instruction.static_u0 != 0u) {
        value = services.attribute(
            unsigned_integer(
                instruction.operand(operand::uv::map), result),
            point).value;
      }
      break;
    case compiler::ValueOperation::generated:
      value = make_float4(point.generated, 1.0f);
      break;
    case compiler::ValueOperation::object_position:
      value = make_float4(point.object_position, 1.0f);
      break;
    case compiler::ValueOperation::object_position_with_transform: {
      const auto m = [&](std::size_t index) noexcept {
        return value_static_table_entry(context, instruction, index);
      };
      const auto world_to_object =
          make_float4x4(make_float4(m(0u), m(1u), m(2u), m(3u)),
                        make_float4(m(4u), m(5u), m(6u), m(7u)),
                        make_float4(m(8u), m(9u), m(10u), m(11u)),
                        make_float4(m(12u), m(13u), m(14u), m(15u)));
      value = world_to_object * make_float4(point.position, 1.0f);
      break;
    }
    case compiler::ValueOperation::object_location:
      value = make_float4(point.object_location, 1.0f);
      break;
    case compiler::ValueOperation::object_random:
      value = make_float4(point.object_random);
      break;
    case compiler::ValueOperation::particle_index:
      value = make_float4(cast<float>(point.particle_index));
      break;
    case compiler::ValueOperation::particle_random:
      value = make_float4(cycles_noise::uint_to_float_inclusive(
          cycles_noise::hash_uint2(point.particle_index, 0u)));
      break;
    case compiler::ValueOperation::back_facing:
      value = make_float4(select(0.0f, 1.0f, point.back_facing));
      break;
    case compiler::ValueOperation::pointiness:
      value = make_float4(
          services.attribute(contract::cycles_pointiness_attribute_id, point)
              .value.x);
      break;
    case compiler::ValueOperation::random_per_island:
      value = make_float4(point.random_per_island);
      break;
    case compiler::ValueOperation::curve_is_strand:
      value = make_float4(select(0.0f, 1.0f, point.is_curve));
      break;
    case compiler::ValueOperation::curve_intercept:
      value = make_float4(point.curve_intercept);
      break;
    case compiler::ValueOperation::curve_length:
      value = make_float4(point.curve_length);
      break;
    case compiler::ValueOperation::curve_thickness:
      value = make_float4(point.curve_thickness);
      break;
    case compiler::ValueOperation::curve_tangent_normal:
      value = make_float4(point.curve_tangent_normal, 0.0f);
      break;
    case compiler::ValueOperation::curve_random:
      value = make_float4(point.curve_random);
      break;
    case compiler::ValueOperation::path_is_camera:
      value = make_float4(select(
          0.0f, 1.0f, (point.ray_visibility & camera_ray_visibility) != 0u));
      break;
    case compiler::ValueOperation::path_is_shadow:
      value = make_float4(select(
          0.0f, 1.0f, (point.ray_visibility & shadow_ray_visibility) != 0u));
      break;
    case compiler::ValueOperation::path_is_diffuse:
      value = make_float4(select(
          0.0f, 1.0f, (point.ray_visibility & diffuse_ray_visibility) != 0u));
      break;
    case compiler::ValueOperation::path_is_glossy:
      value = make_float4(select(
          0.0f, 1.0f, (point.ray_visibility & glossy_ray_visibility) != 0u));
      break;
    case compiler::ValueOperation::path_is_singular:
      value = make_float4(
          select(0.0f, 1.0f,
                 (point.ray_events &
                  static_cast<std::uint32_t>(contract::event_singular)) != 0u));
      break;
    case compiler::ValueOperation::path_is_reflection:
      value = make_float4(select(
          0.0f, 1.0f,
          (point.ray_events &
           static_cast<std::uint32_t>(contract::event_reflection)) != 0u));
      break;
    case compiler::ValueOperation::path_is_transmission:
      value = make_float4(
          select(0.0f, 1.0f,
                 (point.ray_visibility & transmission_ray_visibility) != 0u));
      break;
    case compiler::ValueOperation::path_is_volume_scatter:
      value = make_float4(select(
          0.0f, 1.0f, (point.ray_visibility & volume_ray_visibility) != 0u));
      break;
    case compiler::ValueOperation::path_ray_length:
      value = make_float4(point.ray_length);
      break;
    case compiler::ValueOperation::path_ray_depth:
      value = make_float4(cast<float>(point.ray_depth));
      break;
    case compiler::ValueOperation::path_diffuse_depth:
      value = make_float4(cast<float>(point.diffuse_depth));
      break;
    case compiler::ValueOperation::path_glossy_depth:
      value = make_float4(cast<float>(point.glossy_depth));
      break;
    case compiler::ValueOperation::path_transparent_depth:
      value = make_float4(cast<float>(point.transparent_depth));
      break;
    case compiler::ValueOperation::path_transmission_depth:
      value = make_float4(cast<float>(point.transmission_depth));
      break;
    case compiler::ValueOperation::fresnel: {
      auto eta = max(
          scalar(instruction.operand(operand::fresnel::ior), result),
          1.0e-5f);
      eta = select(eta, 1.0f / eta, point.back_facing);
      // Cycles' stack_load_float3_default substitutes sd->N only for an
      // unconnected socket. A connected vector is consumed verbatim: shader
      // vector sockets do not acquire an implicit normalization operation.
      auto normal = result.shading_normal;
      if (context.svm_immediate_override != nullptr) {
        normal = select(
            result.shading_normal,
            vector(instruction.operand(operand::fresnel::normal), result),
            (*context.svm_immediate_override & 1u) != 0u);
      } else if (instruction.static_u0 != 0u) {
        normal = vector(
            instruction.operand(operand::fresnel::normal), result);
      }
      value =
          make_float4(fresnel_dielectric_cos(dot(point.incoming, normal), eta));
      break;
    }
    case compiler::ValueOperation::layer_weight_fresnel: {
      auto blend = scalar(
          instruction.operand(operand::layer_weight::blend), result);
      auto normal = result.shading_normal;
      if (context.svm_immediate_override != nullptr) {
        normal = select(
            result.shading_normal,
            vector(
                instruction.operand(operand::layer_weight::normal), result),
            (*context.svm_immediate_override & 1u) != 0u);
      } else if (instruction.static_u0 != 0u) {
        normal = vector(
            instruction.operand(operand::layer_weight::normal), result);
      }
      auto eta = max(1.0f - blend, 1.0e-5f);
      eta = select(1.0f / eta, eta, point.back_facing);
      value =
          make_float4(fresnel_dielectric_cos(dot(point.incoming, normal), eta));
      break;
    }
    case compiler::ValueOperation::layer_weight_facing: {
      auto blend = clamp(
          scalar(
              instruction.operand(operand::layer_weight::blend), result),
          0.0f,
          1.0f - 1.0e-5f);
      auto normal = result.shading_normal;
      if (context.svm_immediate_override != nullptr) {
        normal = select(
            result.shading_normal,
            vector(
                instruction.operand(operand::layer_weight::normal), result),
            (*context.svm_immediate_override & 1u) != 0u);
      } else if (instruction.static_u0 != 0u) {
        normal = vector(
            instruction.operand(operand::layer_weight::normal), result);
      }
      auto facing = abs(dot(point.incoming, normal));
      auto exponent = select(0.5f / (1.0f - blend), 2.0f * blend, blend < 0.5f);
      value = make_float4(1.0f - pow(facing, exponent));
      break;
    }
    case compiler::ValueOperation::mapping: {
      const auto static_immediate =
          static_cast<std::uint16_t>(compiler::make_surface_value_svm_immediate(
              instruction.operation, instruction.static_u0,
              instruction.static_u1));
      const std::array static_domain{static_immediate};
      auto immediate_domain = std::span<const std::uint16_t>{static_domain};
      UInt immediate = static_immediate;
      if (context.svm_immediate_override != nullptr) {
        immediate = *context.svm_immediate_override;
        immediate_domain = context.svm_immediate_domain;
      }
      auto input =
          vector(instruction.operand(operand::mapping::vector), result);
      const auto axes =
          (immediate & compiler::surface_value_mapping_axes_mask) >>
          compiler::surface_value_mapping_axes_shift;
      $if(axes != 0u) {
        const auto component = [&input](UInt axis) noexcept -> Float {
          Float selected = 0.0f;
          selected = select(selected, input.x, axis == 1u);
          selected = select(selected, input.y, axis == 2u);
          selected = select(selected, input.z, axis == 3u);
          return selected;
        };
        input =
            make_float3(component(axes & 0x3u), component((axes >> 2u) & 0x3u),
                        component((axes >> 4u) & 0x3u));
      };
      auto location =
          vector(instruction.operand(operand::mapping::location), result);
      auto rotation =
          vector(instruction.operand(operand::mapping::rotation), result);
      auto scale = vector(instruction.operand(operand::mapping::scale), result);
      Float3 mapped = make_float3(0.0f);
      const auto mode = immediate & compiler::surface_value_mapping_type_mask;
      luisa::compute::detail::SwitchStmtBuilder{mode} % [&] {
        std::array<bool, 4u> emitted{};
        for (const auto encoded : immediate_domain) {
          const auto static_mode = static_cast<std::uint32_t>(encoded) &
                                   compiler::surface_value_mapping_type_mask;
          if (emitted[static_mode]) {
            continue;
          }
          emitted[static_mode] = true;
          luisa::compute::detail::SwitchCaseStmtBuilder{static_mode} %
              [&, static_mode] {
                switch (static_cast<compiler::MappingVectorType>(static_mode)) {
                case compiler::MappingVectorType::point:
                  mapped = map_vector_point(services, input, location, rotation,
                                            scale);
                  break;
                case compiler::MappingVectorType::texture:
                  mapped = map_vector_texture(services, input, location,
                                              rotation, scale);
                  break;
                case compiler::MappingVectorType::normal:
                  mapped = map_vector_normal(services, input, rotation, scale);
                  break;
                case compiler::MappingVectorType::vector:
                  mapped =
                      map_vector_direction(services, input, rotation, scale);
                  break;
                }
              };
        }
        luisa::compute::detail::SwitchDefaultStmtBuilder{} % [] {
          luisa::compute::dsl::unreachable("invalid Mapping SVM immediate");
        };
      };
      value = make_float4(mapped, 0.0f);
      break;
    }
    default:
      break;
    }
    return project_surface_value(instruction.result_type, value);
  }
};

} // namespace

std::unique_ptr<ValueNode> try_make_context_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
  if (!supports_context_value(instruction.operation)) {
    return nullptr;
  }
  return std::make_unique<ContextValueNode>(instruction);
}

} // namespace psycles::luisa_backend::detail

#include "graph_surface_internal.h"
#include "surface_displacement.h"

#include <cstdlib>
#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] SurfaceDisplacementInput
displacement_input(const SurfacePoint &point, const TracedValues &values,
                   const compiler::ValueInstruction &instruction,
                   Float3 normal) noexcept {
  return {.height = scalar(instruction.operand(operand::displacement::height),
                           values),
          .midlevel = scalar(
              instruction.operand(operand::displacement::midlevel), values),
          .scale =
              scalar(instruction.operand(operand::displacement::scale), values),
          .normal = std::move(normal),
          .normal_to_world_x = point.normal_to_world_x,
          .normal_to_world_y = point.normal_to_world_y,
          .normal_to_world_z = point.normal_to_world_z};
}

[[nodiscard]] Float3 vector_displacement(
    const ShaderServices &services, const SurfacePoint &point,
    const TracedValues &values,
    const compiler::ValueInstruction &instruction,
    UInt configuration) noexcept {
  const auto default_tangent = vector_displacement_default_tangent(
      point.undisplaced_object_tangent, point.undisplaced_tangent_sign,
      point.is_curve, point.geometry_index);
  Float3 tangent = default_tangent.object_tangent;
  Float tangent_sign = default_tangent.tangent_sign;
  Bool tangent_attribute_found = default_tangent.tangent_attribute_found;
  Bool tangent_sign_found = default_tangent.tangent_sign_found;
  const auto named =
      (configuration & static_cast<std::uint32_t>(
                           compiler::vector_displacement_named_tangent)) != 0u;
  $if(named) {
    const auto attribute = services.attribute(
        unsigned_integer(
            instruction.operand(operand::vector_displacement::attribute),
            values),
        point);
    tangent = attribute.value.xyz();
    tangent_sign = attribute.value.w;
    tangent_attribute_found = attribute.found;
    tangent_sign_found = attribute.found;
  };
  const auto input = SurfaceVectorDisplacementInput{
      .vector = vector(
          instruction.operand(operand::vector_displacement::vector), values),
      .midlevel = scalar(
          instruction.operand(operand::vector_displacement::midlevel), values),
      .scale = scalar(
          instruction.operand(operand::vector_displacement::scale), values),
      .shading_normal = values.shading_normal,
      .object_tangent = tangent,
      .tangent_sign = tangent_sign,
      .tangent_attribute_found = tangent_attribute_found,
      .tangent_sign_found = tangent_sign_found,
      .dpdu = point.dpdu,
      .normal_to_world_x = point.normal_to_world_x,
      .normal_to_world_y = point.normal_to_world_y,
      .normal_to_world_z = point.normal_to_world_z};
  return vector_displacement_inline(
      input, configuration & static_cast<std::uint32_t>(
                                 compiler::vector_displacement_space_mask));
}

class DisplacementValueNode final : public ValueNode {

public:
  using ValueNode::ValueNode;

  [[nodiscard]] SurfaceValueExpression
  evaluate(ValueEvaluationContext &context) const noexcept override {
    const auto &instruction = this->instruction();
    if (instruction.operation ==
        compiler::ValueOperation::vector_displacement) {
      const UInt configuration =
          context.svm_immediate_override != nullptr
              ? *context.svm_immediate_override
              : static_cast<std::uint32_t>(instruction.static_u0);
      const auto result = vector_displacement(
          context.services, context.point, context.result, instruction,
          configuration);
      return SurfaceValueExpression::from_vector(
          Expr<luisa::float3>{result.expression()});
    }
    if (context.svm_immediate_override != nullptr) {
      const auto configuration = *context.svm_immediate_override;
      const auto normal_linked =
          (configuration & static_cast<std::uint32_t>(
                               compiler::displacement_normal_linked)) != 0u;
      const auto object_space =
          (configuration & static_cast<std::uint32_t>(
                               compiler::displacement_object_space)) != 0u;
      const auto linked_normal = vector(
          instruction.operand(operand::displacement::normal), context.result);
      const auto input = displacement_input(
          context.point, context.result, instruction,
          select(context.result.shading_normal, linked_normal, normal_linked));
      Float3 result = make_float3(0.0f);
      $if(object_space) { result = displacement_object_inline(input); }
      $else { result = displacement_world_inline(input); };
      return SurfaceValueExpression::from_vector(
          Expr<luisa::float3>{result.expression()});
    }

    const auto normal_linked =
        (instruction.static_u0 & compiler::displacement_normal_linked) != 0u;
    const auto input = displacement_input(
        context.point, context.result, instruction,
        normal_linked
            ? vector(instruction.operand(operand::displacement::normal),
                     context.result)
            : context.result.shading_normal);
    const auto result =
        (instruction.static_u0 & compiler::displacement_object_space) != 0u
            ? displacement_object_inline(input)
            : displacement_world_inline(input);
    return SurfaceValueExpression::from_vector(
        Expr<luisa::float3>{result.expression()});
  }
};

} // namespace

std::unique_ptr<ValueNode> try_make_displacement_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
  if (instruction.operation != compiler::ValueOperation::displacement &&
      instruction.operation !=
          compiler::ValueOperation::vector_displacement) {
    return nullptr;
  }
  return std::make_unique<DisplacementValueNode>(instruction);
}

} // namespace psycles::luisa_backend::detail

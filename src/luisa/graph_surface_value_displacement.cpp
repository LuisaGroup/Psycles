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

class DisplacementValueNode final : public ValueNode {

public:
  using ValueNode::ValueNode;

  [[nodiscard]] SurfaceValueExpression
  evaluate(ValueEvaluationContext &context) const noexcept override {
    const auto &instruction = this->instruction();
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
  if (instruction.operation != compiler::ValueOperation::displacement) {
    return nullptr;
  }
  return std::make_unique<DisplacementValueNode>(instruction);
}

} // namespace psycles::luisa_backend::detail

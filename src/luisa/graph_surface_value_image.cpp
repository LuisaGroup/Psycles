#include "graph_surface_internal.h"
#include "surface_image_svm.h"

#include <array>
#include <psycles/compiler/surface_execution_plan.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] bool supports_image_value(
    compiler::ValueOperation operation) noexcept {
    switch (operation) {
        case compiler::ValueOperation::image_color:
        case compiler::ValueOperation::image_alpha:
        case compiler::ValueOperation::environment_color:
        case compiler::ValueOperation::environment_alpha:
        case compiler::ValueOperation::attribute_color:
        case compiler::ValueOperation::attribute_factor:
        case compiler::ValueOperation::attribute_alpha:
            return true;
        default:
            return false;
    }
}

class ImageValueNode final : public ValueNode {

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
    case compiler::ValueOperation::image_color:
    case compiler::ValueOperation::image_alpha:
    case compiler::ValueOperation::environment_color:
    case compiler::ValueOperation::environment_alpha: {
      const auto environment =
          instruction.operation ==
              compiler::ValueOperation::environment_color ||
          instruction.operation == compiler::ValueOperation::environment_alpha;
      const auto box_projection =
          !environment &&
          ((instruction.static_u1 &
            compiler::surface_value_image_projection_mask) >>
           compiler::surface_value_image_projection_shift) == 1u;
      const auto vector_operand = environment
                                      ? operand::environment_texture::vector
                                      : operand::image_texture::vector;
      const auto image_operand = environment
                                     ? operand::environment_texture::image
                                     : operand::image_texture::image;
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
      const auto texture_handle =
          cast<std::uint32_t>(unsigned_integer(
              instruction.operand(image_operand), result));
      Float projection_blend = 0.0f;
      if (box_projection) {
          projection_blend = scalar(
              instruction.operand(operand::image_texture::projection_blend),
              result);
      }
      const auto shape = environment      ? SurfaceImageSvmShape::environment
                         : box_projection ? SurfaceImageSvmShape::image_box
                                          : SurfaceImageSvmShape::image;
      const auto sampled = evaluate_surface_image_svm(
          services, point, shape, immediate, immediate_domain,
          vector(instruction.operand(vector_operand), result), texture_handle,
          projection_blend);
      const auto color_output =
          instruction.operation == compiler::ValueOperation::image_color ||
          instruction.operation == compiler::ValueOperation::environment_color;
      value = color_output ? sampled : make_float4(sampled.w);
      break;
    }
    case compiler::ValueOperation::attribute_color:
    case compiler::ValueOperation::attribute_factor:
    case compiler::ValueOperation::attribute_alpha: {
      auto attribute = services.attribute(
          unsigned_integer(instruction.operand(operand::attribute::id), result),
          point);
      if (instruction.operation == compiler::ValueOperation::attribute_factor) {
        value = make_float4(
            (attribute.value.x + attribute.value.y + attribute.value.z) / 3.0f);
      } else if (instruction.operation ==
                 compiler::ValueOperation::attribute_alpha) {
        // Cycles treats a missing attribute as a zero
        // FLOAT3 descriptor. Its Alpha projection is one,
        // while an existing RGBA attribute supplies w.
        value = make_float4(select(1.0f, attribute.value.w, attribute.found));
      } else {
        value = attribute.value;
      }
      break;
    }
    default:
      break;
    }
    return project_surface_value(instruction.result_type, value);
  }
};

}// namespace

std::unique_ptr<ValueNode> try_make_image_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    if (!supports_image_value(instruction.operation)) {
        return nullptr;
    }
    return std::make_unique<ImageValueNode>(instruction);
}

}// namespace psycles::luisa_backend::detail

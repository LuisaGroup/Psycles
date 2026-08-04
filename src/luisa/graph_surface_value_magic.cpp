#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_magic.h>

namespace psycles::luisa_backend::detail {
namespace {

class MagicValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &instruction = this->instruction();
        const auto color = cycles_magic::evaluate(
            static_cast<std::uint32_t>(instruction.static_u0),
            vector(instruction.a, context.result),
            scalar(instruction.b, context.result),
            scalar(instruction.c, context.result));
        const auto value =
            instruction.operation ==
                    compiler::ValueOperation::magic_factor
                ? make_float4(
                      (color.x + color.y + color.z) *
                      (1.0f / 3.0f))
                : make_float4(color, 1.0f);
        return project_surface_value(
            instruction.result_type,
            value);
    }
};

}// namespace

std::unique_ptr<ValueNode> try_make_magic_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    if (instruction.operation != compiler::ValueOperation::magic_color &&
        instruction.operation != compiler::ValueOperation::magic_factor) {
        return nullptr;
    }
    return std::make_unique<MagicValueNode>(instruction);
}

}// namespace psycles::luisa_backend::detail

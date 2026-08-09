#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_voronoi.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

class VoronoiValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression
    evaluate(ValueEvaluationContext &context) const noexcept override {
        const auto &instruction = this->instruction();
        const auto value = cycles_voronoi::evaluate(
            cycles_voronoi::decode_configuration(instruction),
            vector(
                instruction.operand(operand::voronoi::vector),
                context.result),
            scalar(
                instruction.operand(operand::voronoi::w),
                context.result),
            scalar(
                instruction.operand(operand::voronoi::scale),
                context.result),
            scalar(
                instruction.operand(operand::voronoi::detail),
                context.result),
            scalar(
                instruction.operand(operand::voronoi::roughness),
                context.result),
            scalar(
                instruction.operand(operand::voronoi::lacunarity),
                context.result),
            scalar(
                instruction.operand(operand::voronoi::smoothness),
                context.result),
            scalar(
                instruction.operand(operand::voronoi::exponent),
                context.result),
            scalar(
                instruction.operand(operand::voronoi::randomness),
                context.result));
        return project_surface_value(instruction.result_type, value);
    }
};

}// namespace

std::unique_ptr<ValueNode> try_make_voronoi_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    if (!cycles_voronoi::is_operation(instruction.operation)) {
        return nullptr;
    }
    return std::make_unique<VoronoiValueNode>(instruction);
}

}// namespace psycles::luisa_backend::detail

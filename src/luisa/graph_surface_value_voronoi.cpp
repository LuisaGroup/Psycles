#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_voronoi.h>

namespace psycles::luisa_backend::detail {
namespace {

class VoronoiValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression
    evaluate(ValueEvaluationContext &context) const noexcept override {
        const auto &instruction = this->instruction();
        const auto value = cycles_voronoi::evaluate(
            cycles_voronoi::decode_configuration(instruction),
            vector(instruction.a, context.result),
            scalar(instruction.b, context.result),
            scalar(instruction.c, context.result),
            scalar(instruction.d, context.result),
            scalar(instruction.e, context.result),
            scalar(instruction.f, context.result),
            scalar(instruction.g, context.result),
            scalar(instruction.h, context.result),
            scalar(instruction.i, context.result));
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

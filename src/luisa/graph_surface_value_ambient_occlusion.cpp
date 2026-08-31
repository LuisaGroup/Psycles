#include "graph_surface_internal.h"

#include <psycles/luisa/native_vector_math.h>

#include <cstdint>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] Float3 cycles_safe_normalize(Float3 value) noexcept {
    return native_vector_math::safe_normalize_nonzero(value);
}

class AmbientOcclusionValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &instruction = this->instruction();
        const auto configuration =
            context.svm_immediate_override != nullptr
                ? UInt{*context.svm_immediate_override}
                : UInt{static_cast<std::uint32_t>(instruction.static_u0)};
        const auto linked =
            (configuration & static_cast<std::uint32_t>(
                                 compiler::ambient_occlusion_normal_linked)) !=
            0u;
        auto normal = select(
            context.point.shading_normal,
            vector(instruction.operand(operand::ambient_occlusion::normal),
                   context.result),
            linked);
        normal = cycles_safe_normalize(normal);

        Float ao = 1.0f;
        if (const auto *provider =
                context.services.surface_ambient_occlusion_provider()) {
            ao = provider->evaluate(
                context.point,
                {.normal = std::move(normal),
                 .distance = scalar(
                     instruction.operand(operand::ambient_occlusion::distance),
                     context.result),
                 .samples = cast<luisa::uint>(unsigned_integer(
                     instruction.operand(operand::ambient_occlusion::samples),
                     context.result)),
                 .only_local =
                     (configuration &
                      static_cast<std::uint32_t>(
                          compiler::ambient_occlusion_only_local)) != 0u,
                 .inside =
                     (configuration & static_cast<std::uint32_t>(
                                          compiler::ambient_occlusion_inside)) !=
                     0u,
                 .global_radius =
                     (configuration &
                      static_cast<std::uint32_t>(
                          compiler::ambient_occlusion_global_radius)) != 0u});
        }
        return SurfaceValueExpression::from_scalar(
            Expr<float>{ao.expression()});
    }
};

}// namespace

std::unique_ptr<ValueNode> try_make_ambient_occlusion_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    if (instruction.operation !=
        compiler::ValueOperation::ambient_occlusion) {
        return nullptr;
    }
    return std::make_unique<AmbientOcclusionValueNode>(instruction);
}

}// namespace psycles::luisa_backend::detail

#include "graph_surface_internal.h"
#include "surface_bump.h"

#include <cstddef>
#include <cstdlib>
#include <memory>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

[[nodiscard]] compiler::ValueOperation
base_sampled_operation(compiler::ValueOperation operation) noexcept {
    using compiler::ValueOperation;
    switch (operation) {
    case ValueOperation::sampled_surface_position:
        return ValueOperation::surface_position;
    case ValueOperation::sampled_uv:
        return ValueOperation::uv;
    case ValueOperation::sampled_generated:
        return ValueOperation::generated;
    case ValueOperation::sampled_object_position:
        return ValueOperation::object_position;
    case ValueOperation::sampled_object_position_with_transform:
        return ValueOperation::object_position_with_transform;
    case ValueOperation::sampled_pointiness:
        return ValueOperation::pointiness;
    case ValueOperation::sampled_attribute_color:
        return ValueOperation::attribute_color;
    case ValueOperation::sampled_attribute_factor:
        return ValueOperation::attribute_factor;
    case ValueOperation::sampled_attribute_alpha:
        return ValueOperation::attribute_alpha;
    case ValueOperation::sampled_normal_map:
        return ValueOperation::normal_map;
    default:
        std::abort();
    }
}

[[nodiscard]] SurfacePoint differential_sample_point(
    const SurfacePoint &point,
    Float dx,
    Float dy) noexcept {
    auto sampled = point;
    sampled.position = point.position + point.dPdx * dx + point.dPdy * dy;
    sampled.object_position =
        point.object_position + point.object_dPdx * dx +
        point.object_dPdy * dy;
    sampled.generated =
        point.generated + point.generated_dx * dx + point.generated_dy * dy;
    sampled.uv = point.uv + point.uv_dx * dx + point.uv_dy * dy;
    sampled.barycentric =
        point.barycentric + point.barycentric_dx * dx +
        point.barycentric_dy * dy;
    return sampled;
}

class BumpOffsetZeroValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &) const noexcept override {
        return SurfaceValueExpression::from_scalar(0.0f);
    }
};

class BumpFilterWidthValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        return SurfaceValueExpression::from_scalar(max(
            scalar(instruction().operand(operand::unary::input),
                   context.result),
            0.0f));
    }
};

class BumpSamplesValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &bump = instruction();
        UInt encoded_configuration =
            static_cast<std::uint32_t>(bump.static_u0);
        if (context.svm_immediate_override != nullptr) {
            encoded_configuration = *context.svm_immediate_override;
        }
        const auto configuration = SurfaceBumpSvmConfiguration{
            .invert = (encoded_configuration & 1u) != 0u,
            .normal_linked = (encoded_configuration & 2u) != 0u,
            .object_space = (encoded_configuration & 4u) != 0u};
        const auto normal = select(
            context.result.shading_normal,
            vector(bump.operand(operand::bump_samples::normal),
                   context.result),
            configuration.normal_linked);
        const auto filter_width = scalar(
            bump.operand(operand::bump_samples::filter_width), context.result);
        const auto result = evaluate_surface_bump(
            context.services, context.point, configuration, normal,
            filter_width,
            scalar(bump.operand(operand::bump_samples::height_center),
                   context.result),
            scalar(bump.operand(operand::bump_samples::height_x),
                   context.result),
            scalar(bump.operand(operand::bump_samples::height_y),
                   context.result),
            scalar(bump.operand(operand::bump_samples::distance),
                   context.result),
            scalar(bump.operand(operand::bump_samples::strength),
                   context.result));
        return SurfaceValueExpression::from_vector(
            Expr<luisa::float3>{result.expression()});
    }
};

class SampledValueNode final : public ValueNode {

private:
    compiler::ValueInstruction _base_instruction;
    std::unique_ptr<ValueNode> _base_node;

public:
    explicit SampledValueNode(
        const compiler::ValueInstruction &sampled) noexcept
        : ValueNode{sampled}, _base_instruction{sampled} {
        _base_instruction.operation =
            base_sampled_operation(sampled.operation);
        if (_base_instruction.operands.size() <
            operand::sampled_nullary::count) {
            std::abort();
        }
        _base_instruction.operands.erase(
            _base_instruction.operands.begin(),
            _base_instruction.operands.begin() +
                static_cast<std::ptrdiff_t>(
                    operand::sampled_nullary::count));
        _base_node = make_value_node(_base_instruction);
    }

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &sampled = instruction();
        auto point = differential_sample_point(
            context.point,
            scalar(sampled.operand(operand::sampled_nullary::dx),
                   context.result),
            scalar(sampled.operand(operand::sampled_nullary::dy),
                   context.result));
        ValueEvaluationContext sampled_context{
            .services = context.services,
            .point = point,
            .result = context.result,
            .surface = context.surface,
            .parameter_override = context.parameter_override,
            .static_table_override = context.static_table_override,
            .svm_immediate_override = context.svm_immediate_override,
            .svm_immediate_domain = context.svm_immediate_domain};
        return _base_node->evaluate(sampled_context);
    }
};

} // namespace

std::unique_ptr<ValueNode> try_make_bump_expanded_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    using compiler::ValueOperation;
    switch (instruction.operation) {
    case ValueOperation::bump_offset_zero:
        return std::make_unique<BumpOffsetZeroValueNode>(instruction);
    case ValueOperation::bump_filter_width:
        return std::make_unique<BumpFilterWidthValueNode>(instruction);
    case ValueOperation::bump_samples:
        return std::make_unique<BumpSamplesValueNode>(instruction);
    case ValueOperation::sampled_surface_position:
    case ValueOperation::sampled_uv:
    case ValueOperation::sampled_generated:
    case ValueOperation::sampled_object_position:
    case ValueOperation::sampled_object_position_with_transform:
    case ValueOperation::sampled_pointiness:
    case ValueOperation::sampled_attribute_color:
    case ValueOperation::sampled_attribute_factor:
    case ValueOperation::sampled_attribute_alpha:
    case ValueOperation::sampled_normal_map:
        return std::make_unique<SampledValueNode>(instruction);
    default:
        return nullptr;
    }
}

} // namespace psycles::luisa_backend::detail

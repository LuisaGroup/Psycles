#include "graph_surface_internal.h"
#include "surface_bump.h"
#include "surface_normal_map.h"

#include <cstdlib>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

namespace operand = compiler::value_operand;

class NormalMapValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &services = context.services;
        const auto &point = context.point;
        const auto &result = context.result;
        const auto &instruction = this->instruction();

        auto mapped =
            vector(
                instruction.operand(operand::normal_map::color),
                result) *
                2.0f -
            1.0f;
        auto strength = scalar(
            instruction.operand(operand::normal_map::strength), result);
        const auto space = compiler::decode_normal_map_space(
            instruction.static_u0);
        const auto named = compiler::normal_map_has_named_tangent(
            instruction.static_u0);
        const auto displaced_base =
            compiler::decode_normal_map_base(instruction.static_u0) ==
            compiler::NormalMapBase::displaced;
        const auto invert_green =
            compiler::decode_normal_map_convention(
                instruction.static_u0) ==
            compiler::NormalMapConvention::direct_x;
        if (invert_green) {
            mapped.y = -mapped.y;
        }

        auto object_tangent = displaced_base
                                  ? point.object_tangent
                                  : point.undisplaced_object_tangent;
        auto tangent_sign = displaced_base
                                ? point.tangent_sign
                                : point.undisplaced_tangent_sign;
        Bool tangent_attribute_found = true;
        if (named) {
            auto named_tangent =
                services.attribute(
                    unsigned_integer(
                        instruction.operand(operand::normal_map::uv_map),
                        result),
                    point);
            object_tangent =
                named_tangent.value.xyz();
            tangent_sign =
                named_tangent.value.w;
            tangent_attribute_found = named_tangent.found;
        }
        SurfaceNormalMapInput input{
            .mapped = mapped,
            .strength = strength,
            .object_tangent = object_tangent,
            .tangent_sign = tangent_sign,
            .tangent_attribute_found = tangent_attribute_found,
            .object_shading_normal = point.object_shading_normal,
            .undisplaced_object_shading_normal =
                point.undisplaced_object_shading_normal,
            .triangle_smooth = point.triangle_smooth,
            .normal_to_world_x = point.normal_to_world_x,
            .normal_to_world_y = point.normal_to_world_y,
            .normal_to_world_z = point.normal_to_world_z,
            .shading_normal = point.shading_normal,
            .back_facing = point.back_facing,
            .geometry_index = point.geometry_index,
            .is_curve = point.is_curve};
        Float3 world;
        if (space ==
            compiler::NormalMapSpace::tangent) {
            // Cycles' DISPLACED path uses
            // triangle_smooth_normal_unnormalized_object_space(). Despite
            // the historical function name, the current implementation
            // normalizes the interpolated current normal before constructing
            // the tangent frame. ORIGINAL deliberately keeps the raw saved
            // attribute for smooth triangles because its strength is blended
            // in world space afterwards.
            world = displaced_base
                        ? normal_map_tangent_displaced(
                              services, input)
                        : normal_map_tangent_original(
                              services, input);
        } else {
            if (space ==
                    compiler::NormalMapSpace::
                        blender_object ||
                space ==
                    compiler::NormalMapSpace::
                        blender_world) {
                mapped.y = -mapped.y;
                mapped.z = -mapped.z;
            }
            input.mapped = mapped;
            world =
                space ==
                            compiler::NormalMapSpace::
                                object ||
                        space ==
                            compiler::NormalMapSpace::
                                blender_object ?
                    normal_map_object(services, input) :
                    normal_map_world(services, input);
        }
        return SurfaceValueExpression::from_vector(
            Expr<luisa::float3>{world.expression()});
    }
};

class BumpValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &services = context.services;
        const auto &point = context.point;
        auto &result = context.result;
        const auto &instruction = this->instruction();

        const auto configuration =
            decode_surface_bump_configuration(
                instruction.static_u0);
        auto normal_in =
            configuration.normal_linked
                ? vector(
                      instruction.operand(operand::bump::normal),
                      result)
                : result.shading_normal;
        const auto domain =
            make_surface_bump_evaluation_domain(
                point,
                scalar(
                    instruction.operand(
                        operand::bump::filter_width),
                    result));

        if (context.surface == nullptr) {
            std::abort();
        }
        const auto height_dependencies =
            context.surface->value_dependency_mask(
                instruction.operand(operand::bump::height));
        auto values_x =
            context.surface->trace_values(
                services,
                domain.point_x,
                &height_dependencies);
        auto values_y =
            context.surface->trace_values(
                services,
                domain.point_y,
                &height_dependencies);
        const auto height_center =
            scalar(
                instruction.operand(operand::bump::height), result);
        const auto height_x =
            scalar(
                instruction.operand(operand::bump::height), values_x);
        const auto height_y =
            scalar(
                instruction.operand(operand::bump::height), values_y);
        const auto distance =
            scalar(
                instruction.operand(operand::bump::distance), result);
        const auto strength =
            scalar(
                instruction.operand(operand::bump::strength), result);
        const auto normal_out = evaluate_surface_bump(
            services,
            point,
            configuration,
            normal_in,
            domain,
            height_center,
            height_x,
            height_y,
            distance,
            strength);
        return SurfaceValueExpression::from_vector(
            Expr<luisa::float3>{normal_out.expression()});
    }
};

}// namespace

std::unique_ptr<ValueNode> try_make_normal_value_node(
    const compiler::ValueInstruction &instruction) noexcept {
    switch (instruction.operation) {
        case compiler::ValueOperation::normal_map:
            return std::make_unique<
                NormalMapValueNode>(instruction);
        case compiler::ValueOperation::bump:
            return std::make_unique<
                BumpValueNode>(instruction);
        default:
            return nullptr;
    }
}

}// namespace psycles::luisa_backend::detail

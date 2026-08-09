#include "graph_surface_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

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
            vector(instruction.a, result) * 2.0f - 1.0f;
        auto strength = scalar(instruction.b, result);
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
                        instruction.c, result),
                    point);
            object_tangent =
                named_tangent.value.xyz();
            tangent_sign =
                named_tangent.value.w;
            tangent_attribute_found = named_tangent.found;
        }
        const auto transform_object_normal =
            [&](Float3 object_normal) noexcept {
                return safe_normalize(
                    point.normal_to_world_x *
                            object_normal.x +
                        point.normal_to_world_y *
                            object_normal.y +
                        point.normal_to_world_z *
                            object_normal.z,
                    point.shading_normal);
            };
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
            auto object_base = safe_normalize(
                point.object_shading_normal,
                make_float3(0.0f));
            Bool linear_interpolate_strength = false;
            if (!displaced_base) {
                // ATTR_STD_NORMAL_UNDISPLACED is used only for smooth
                // triangles. Cycles deliberately retains current Ng for a
                // flat triangle even when the tangent frame is ORIGINAL.
                object_base = select(
                    object_base,
                    point.undisplaced_object_shading_normal,
                    point.triangle_smooth);
                linear_interpolate_strength =
                    point.triangle_smooth;
            }
            const auto object_base_length_squared =
                length_squared(object_base);
            const auto object_base_available =
                object_base_length_squared >
                1.0e-20f;
            $if(!linear_interpolate_strength) {
                mapped.x *= strength;
                mapped.y *= strength;
                mapped.z =
                    1.0f +
                    (mapped.z - 1.0f) *
                        clamp(strength, 0.0f, 1.0f);
            };
            const auto object_bitangent =
                tangent_sign *
                cross(object_base, object_tangent);
            const auto object_normal =
                safe_normalize(
                    object_tangent * mapped.x +
                        object_bitangent * mapped.y +
                        object_base * mapped.z,
                    make_float3(0.0f));
            world =
                transform_object_normal(
                    object_normal);
            world = select(
                world, -world, point.back_facing);
            const auto linearly_blended =
                safe_normalize(
                    point.shading_normal +
                        (world - point.shading_normal) *
                            max(strength, 0.0f),
                    point.shading_normal);
            world = select(
                world,
                linearly_blended,
                linear_interpolate_strength);
            const auto tangent_available =
                (point.geometry_index != ~0u) &
                !point.is_curve &
                tangent_attribute_found &
                object_base_available &
                (length_squared(
                     object_tangent) >
                 1.0e-20f) &
                (abs(tangent_sign) >
                 1.0e-20f);
            world = select(
                point.shading_normal,
                world,
                tangent_available);
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
            world =
                space ==
                            compiler::NormalMapSpace::
                                object ||
                        space ==
                            compiler::NormalMapSpace::
                                blender_object ?
                    transform_object_normal(mapped) :
                    safe_normalize(
                        mapped,
                        point.shading_normal);
            world = select(
                world, -world, point.back_facing);
            const auto nonnegative_strength =
                max(strength, 0.0f);
            world = safe_normalize(
                point.shading_normal +
                    (world -
                     point.shading_normal) *
                        nonnegative_strength,
                point.shading_normal);
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

        auto normal_in =
            (instruction.static_u0 & 2u) != 0u
                ? vector(instruction.e, result)
                : result.shading_normal;
        const auto use_object_space =
            (instruction.static_u0 & 4u) != 0u;
        auto filter_width = max(
            scalar(instruction.d, result), 0.0f);

        auto point_x = point;
        point_x.position =
            point.position +
            point.dPdx * filter_width;
        point_x.object_position =
            point.object_position +
            point.object_dPdx * filter_width;
        point_x.generated =
            point.generated +
            point.generated_dx * filter_width;
        point_x.uv =
            point.uv + point.uv_dx * filter_width;
        point_x.barycentric =
            point.barycentric +
            point.barycentric_dx * filter_width;

        auto point_y = point;
        point_y.position =
            point.position +
            point.dPdy * filter_width;
        point_y.object_position =
            point.object_position +
            point.object_dPdy * filter_width;
        point_y.generated =
            point.generated +
            point.generated_dy * filter_width;
        point_y.uv =
            point.uv + point.uv_dy * filter_width;
        point_y.barycentric =
            point.barycentric +
            point.barycentric_dy * filter_width;

        const auto height_dependencies =
            context.surface.value_dependency_mask(
                instruction.a);
        auto values_x =
            context.surface.trace_values(
                services,
                point_x,
                &height_dependencies);
        auto values_y =
            context.surface.trace_values(
                services,
                point_y,
                &height_dependencies);
        const auto height_center =
            scalar(instruction.a, result);
        const auto height_x =
            scalar(instruction.a, values_x);
        const auto height_y =
            scalar(instruction.a, values_y);
        auto dPdx = point.dPdx;
        auto dPdy = point.dPdy;
        if (use_object_space) {
            // The stored columns form the object-to-world normal matrix.
            // Its inverse maps an arbitrary world normal back to object
            // space, including under non-uniform instance transforms.
            const auto column_x = point.normal_to_world_x;
            const auto column_y = point.normal_to_world_y;
            const auto column_z = point.normal_to_world_z;
            const auto transform_determinant = dot(
                column_x, cross(column_y, column_z));
            const auto inverse_determinant =
                1.0f / select(
                    1.0f,
                    transform_determinant,
                    abs(transform_determinant) > 1.0e-20f);
            normal_in = safe_normalize(
                make_float3(
                    dot(normal_in, cross(column_y, column_z)),
                    dot(normal_in, cross(column_z, column_x)),
                    dot(normal_in, cross(column_x, column_y))) *
                    inverse_determinant,
                point.object_shading_normal);
            dPdx = point.object_dPdx;
            dPdy = point.object_dPdy;
        }
        const auto rx =
            cross(dPdy, normal_in);
        const auto ry =
            cross(normal_in, dPdx);
        const auto determinant =
            dot(dPdx, rx);
        const auto surface_gradient =
            (height_x - height_center) * rx +
            (height_y - height_center) * ry;
        auto distance =
            scalar(instruction.c, result);
        if ((instruction.static_u0 & 1u) != 0u) {
            distance = -distance;
        }
        const auto determinant_sign =
            select(
                -1.0f,
                1.0f,
                determinant >= 0.0f);
        const auto perturbed_vector =
            filter_width * abs(determinant) *
                normal_in -
            distance * determinant_sign *
                surface_gradient;
        const auto perturbed_valid =
            length_squared(perturbed_vector) >
            0.0f;
        const auto perturbed =
            safe_normalize(
                perturbed_vector,
                make_float3(0.0f));
        const auto strength =
            max(
                scalar(instruction.b, result),
                0.0f);
        const auto blended =
            safe_normalize(
                strength * perturbed +
                    (1.0f - strength) *
                        normal_in,
                make_float3(0.0f));
        auto normal_out =
            select(
                normal_in,
                blended,
                perturbed_valid);
        if (use_object_space) {
            normal_out = safe_normalize(
                point.normal_to_world_x * normal_out.x +
                    point.normal_to_world_y * normal_out.y +
                    point.normal_to_world_z * normal_out.z,
                point.shading_normal);
        }
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

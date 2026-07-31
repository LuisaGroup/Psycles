#include "graph_surface_internal.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

class NormalMapValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] Float4 evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &services = context.services;
        const auto &point = context.point;
        const auto &result = context.result;
        const auto &instruction = this->instruction();

        auto mapped =
            vector(instruction.a, result) * 2.0f - 1.0f;
        auto strength =
            scalar(instruction.b, result);
        const auto space =
            static_cast<compiler::NormalMapSpace>(
                instruction.static_u0 & 0xffu);
        auto object_tangent =
            point.object_tangent;
        auto tangent_sign =
            point.tangent_sign;
        if ((instruction.static_u0 & 0x100u) != 0u) {
            auto named_tangent =
                services.attribute(
                    instruction.static_u1,
                    point);
            object_tangent =
                named_tangent.value.xyz();
            tangent_sign =
                named_tangent.value.w;
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
            // Current Cycles evaluates its special triangle normal-map
            // base with safe_normalize(interpolate(n0, n1, n2)).
            // Normalizing before constructing the MikkTSpace frame is
            // observable whenever smooth vertex normals do not interpolate
            // to unit length.
            const auto object_base_length_squared =
                length_squared(
                    point.object_shading_normal);
            const auto object_base_available =
                object_base_length_squared >
                1.0e-20f;
            const auto object_base_normal =
                safe_normalize(
                    point.object_shading_normal,
                    make_float3(0.0f));

            mapped.x *= strength;
            mapped.y *= strength;
            mapped.z =
                1.0f +
                (mapped.z - 1.0f) *
                    clamp(strength, 0.0f, 1.0f);
            const auto object_bitangent =
                tangent_sign *
                cross(
                    object_base_normal,
                    object_tangent);
            const auto object_normal =
                safe_normalize(
                    object_tangent * mapped.x +
                        object_bitangent * mapped.y +
                        object_base_normal *
                            mapped.z,
                    object_base_normal);
            world =
                transform_object_normal(
                    object_normal);
            world = select(
                world, -world, point.back_facing);
            const auto tangent_available =
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
        return make_float4(world, 0.0f);
    }
};

class BumpValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] Float4 evaluate(
        ValueEvaluationContext &context) const noexcept override {
        const auto &services = context.services;
        const auto &point = context.point;
        auto &result = context.result;
        const auto &instruction = this->instruction();

        auto normal_in =
            (instruction.static_u0 & 2u) != 0u
                ? vector(instruction.e, result)
                : result.shading_normal;
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
        const auto rx =
            cross(point.dPdy, normal_in);
        const auto ry =
            cross(normal_in, point.dPdx);
        const auto determinant =
            dot(point.dPdx, rx);
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
        const auto normal_out =
            select(
                normal_in,
                blended,
                perturbed_valid);
        result.shading_normal =
            normal_out;
        return make_float4(
            normal_out, 0.0f);
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

#include "graph_surface_internal.h"

#include <psycles/luisa/spherical_geometry.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

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

[[nodiscard]] Float3 cycles_safe_normalize_direction(
    Float3 direction) noexcept {
    const auto length = sqrt(dot(direction, direction));
    const auto nonzero = length != 0.0f;
    const auto safe_length = select(1.0f, length, nonzero);
    return select(
        direction,
        direction * (1.0f / safe_length),
        nonzero);
}

[[nodiscard]] Float2 cycles_direction_to_equirectangular(
    Float3 direction) noexcept {
    direction = cycles_safe_normalize_direction(direction);
    const auto length = sqrt(dot(direction, direction));
    Float2 uv = make_float2(0.0f);
    $if (length != 0.0f) {
        const auto azimuth =
            spherical_geometry::canonical_direction_azimuth(direction);
        uv = make_float2(
            (pi - azimuth) /
                (2.0f * pi),
            1.0f - acos(direction.z / length) / pi);
    };
    return uv;
}

[[nodiscard]] Float2 cycles_direction_to_mirrorball(
    Float3 direction) noexcept {
    direction = cycles_safe_normalize_direction(direction);
    direction.y -= 1.0f;
    const auto divisor =
        2.0f * sqrt(max(-0.5f * direction.y, 0.0f));
    $if (divisor > 0.0f) {
        direction /= divisor;
    };
    return 0.5f * (direction.xz() + 1.0f);
}

class ImageValueNode final : public ValueNode {

public:
    using ValueNode::ValueNode;

    [[nodiscard]] SurfaceValueExpression evaluate(
        ValueEvaluationContext &context) const noexcept override {
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
                        instruction.operation ==
                            compiler::ValueOperation::environment_alpha;
                    const auto extension =
                        static_cast<std::uint32_t>(
                            instruction.static_u1 & 0xffu);
                    const auto interpolation =
                        static_cast<std::uint32_t>(
                            (instruction.static_u1 >> 10u) &
                            0x03u);
                    const auto projection =
                        static_cast<std::uint32_t>(
                            (instruction.static_u1 >> 12u) &
                            0x03u);
                    const auto unassociate_alpha =
                        ((instruction.static_u1 >> 9u) & 1u) !=
                        0u;
                    const auto encoded_as_srgb =
                        ((instruction.static_u1 >> 8u) & 1u) !=
                        0u;
                    const auto decode_sample =
                        [&](Float4 sampled) noexcept {
                        if (unassociate_alpha) {
                            auto alpha = sampled.w;
                            auto should_unassociate =
                                (alpha != 0.0f) &
                                (alpha != 1.0f);
                            auto safe_alpha = select(
                                1.0f,
                                alpha,
                                should_unassociate);
                            sampled = make_float4(
                                select(
                                    sampled.xyz(),
                                    sampled.xyz() / safe_alpha,
                                    should_unassociate),
                                alpha);
                        }
                        if (encoded_as_srgb) {
                            // Match Cycles' svm_image_texture ordering:
                            // filter associated encoded texels, optionally
                            // unassociate, then decode sRGB.
                            sampled = make_float4(
                                srgb_to_linear(sampled.xyz()),
                                sampled.w);
                        }
                        return sampled;
                    };
                    const auto sample_uv =
                        [&](Float2 uv) noexcept {
                        // Blender UVs use a bottom-left origin while decoded
                        // host images are uploaded in top-to-bottom row
                        // order.
                        uv.y = 1.0f - uv.y;
                        return decode_sample(
                            services.texture_2d(
                                static_cast<std::uint32_t>(
                                    instruction.static_u0),
                                uv,
                                make_float2(0.0f),
                                make_float2(0.0f),
                                interpolation,
                                extension));
                    };

                    auto coordinate =
                        vector(instruction.a, result);
                    Float4 sampled;
                    if (environment) {
                        const auto uv = projection == 1u
                                            ? cycles_direction_to_mirrorball(
                                                  coordinate)
                                            : cycles_direction_to_equirectangular(
                                                  coordinate);
                        sampled = sample_uv(uv);
                    } else if (projection == 1u) {
                        // Cycles transforms the current sd->N back to object
                        // space. This is observably different from reading
                        // the mesh normal after the automatic bump region has
                        // executed its SetNormal stage.
                        const auto column_x = point.normal_to_world_x;
                        const auto column_y = point.normal_to_world_y;
                        const auto column_z = point.normal_to_world_z;
                        const auto determinant = dot(
                            column_x,
                            cross(column_y, column_z));
                        const auto safe_determinant = select(
                            1.0f,
                            determinant,
                            abs(determinant) > 1.0e-20f);
                        auto signed_normal = safe_normalize(
                            make_float3(
                                dot(point.shading_normal,
                                    cross(column_y, column_z)),
                                dot(point.shading_normal,
                                    cross(column_z, column_x)),
                                dot(point.shading_normal,
                                    cross(column_x, column_y))) /
                                safe_determinant,
                            point.object_shading_normal);
                        auto normal = abs(signed_normal);
                        auto normal_sum =
                            normal.x + normal.y + normal.z;
                        normal /= select(
                            1.0f,
                            normal_sum,
                            normal_sum != 0.0f);
                        Float3 weight =
                            make_float3(0.0f);
                        const auto blend =
                            instruction.static_f0;
                        const auto limit =
                            0.5f * (1.0f + blend);
                        $if ((normal.x >
                              limit *
                                  (normal.x + normal.y)) &
                             (normal.x >
                              limit *
                                  (normal.x + normal.z))) {
                            weight.x = 1.0f;
                        }
                        $elif ((normal.y >
                                limit *
                                    (normal.x + normal.y)) &
                               (normal.y >
                                limit *
                                    (normal.y + normal.z))) {
                            weight.y = 1.0f;
                        }
                        $elif ((normal.z >
                                limit *
                                    (normal.x + normal.z)) &
                               (normal.z >
                                limit *
                                    (normal.y + normal.z))) {
                            weight.z = 1.0f;
                        }
                        $elif (blend > 0.0f) {
                            $if (
                                normal.z <
                                (1.0f - limit) *
                                    (normal.y + normal.x)) {
                                weight.x =
                                    normal.x /
                                    (normal.x + normal.y);
                                weight.x = luisa::compute::clamp(
                                    (weight.x -
                                     0.5f *
                                         (1.0f - blend)) /
                                        blend,
                                    0.0f,
                                    1.0f);
                                weight.y = 1.0f - weight.x;
                            }
                            $elif (
                                normal.x <
                                (1.0f - limit) *
                                    (normal.y + normal.z)) {
                                weight.y =
                                    normal.y /
                                    (normal.y + normal.z);
                                weight.y = luisa::compute::clamp(
                                    (weight.y -
                                     0.5f *
                                         (1.0f - blend)) /
                                        blend,
                                    0.0f,
                                    1.0f);
                                weight.z = 1.0f - weight.y;
                            }
                            $elif (
                                normal.y <
                                (1.0f - limit) *
                                    (normal.x + normal.z)) {
                                weight.x =
                                    normal.x /
                                    (normal.x + normal.z);
                                weight.x = luisa::compute::clamp(
                                    (weight.x -
                                     0.5f *
                                         (1.0f - blend)) /
                                        blend,
                                    0.0f,
                                    1.0f);
                                weight.z = 1.0f - weight.x;
                            }
                            $else {
                                weight.x =
                                    ((2.0f - limit) *
                                         normal.x +
                                     (limit - 1.0f)) /
                                    (2.0f * limit - 1.0f);
                                weight.y =
                                    ((2.0f - limit) *
                                         normal.y +
                                     (limit - 1.0f)) /
                                    (2.0f * limit - 1.0f);
                                weight.z =
                                    ((2.0f - limit) *
                                         normal.z +
                                     (limit - 1.0f)) /
                                    (2.0f * limit - 1.0f);
                            };
                        }
                        $else {
                            weight.x = 1.0f;
                        };

                        auto uv_x = make_float2(
                            select(
                                coordinate.y,
                                1.0f - coordinate.y,
                                signed_normal.x < 0.0f),
                            coordinate.z);
                        auto uv_y = make_float2(
                            select(
                                coordinate.x,
                                1.0f - coordinate.x,
                                signed_normal.y > 0.0f),
                            coordinate.z);
                        auto uv_z = make_float2(
                            select(
                                coordinate.y,
                                1.0f - coordinate.y,
                                signed_normal.z > 0.0f),
                            coordinate.x);
                        sampled =
                            weight.x * sample_uv(uv_x) +
                            weight.y * sample_uv(uv_y) +
                            weight.z * sample_uv(uv_z);
                    } else {
                        Float2 uv = coordinate.xy();
                        if (projection == 2u) {
                            auto direction =
                                (coordinate - 0.5f) * 2.0f;
                            auto length_squared =
                                dot(direction, direction);
                            Float2 spherical =
                                make_float2(0.0f);
                            $if (length_squared > 0.0f) {
                                Float u = 0.0f;
                                $if ((direction.x != 0.0f) |
                                     (direction.y != 0.0f)) {
                                    u =
                                        0.5f -
                                        atan2(
                                            direction.x,
                                            direction.y) /
                                            (2.0f * pi);
                                };
                                auto z = luisa::compute::clamp(
                                    direction.z /
                                        sqrt(length_squared),
                                    -1.0f,
                                    1.0f);
                                spherical = make_float2(
                                    u,
                                    1.0f -
                                        acos(z) / pi);
                            };
                            uv = spherical;
                        } else if (projection == 3u) {
                            auto direction =
                                (coordinate - 0.5f) * 2.0f;
                            auto radial_length = sqrt(
                                direction.x * direction.x +
                                direction.y * direction.y);
                            Float2 tube =
                                make_float2(0.0f);
                            $if (radial_length > 0.0f) {
                                tube = make_float2(
                                    (1.0f -
                                     atan2(
                                         direction.x /
                                             radial_length,
                                         direction.y /
                                             radial_length) /
                                         pi) *
                                        0.5f,
                                    (direction.z + 1.0f) *
                                        0.5f);
                            };
                            uv = tube;
                        }
                        sampled = sample_uv(uv);
                    }
                    const auto color_output =
                        instruction.operation ==
                            compiler::ValueOperation::image_color ||
                        instruction.operation ==
                            compiler::ValueOperation::environment_color;
                    value = color_output
                                ? sampled
                                : make_float4(sampled.w);
                    break;
                }
                case compiler::ValueOperation::attribute_color:
                case compiler::ValueOperation::attribute_factor:
                case compiler::ValueOperation::attribute_alpha: {
                    auto attribute = services.attribute(
                        instruction.static_u0, point);
                    if (instruction.operation ==
                        compiler::ValueOperation::attribute_factor) {
                        value = make_float4(
                            (attribute.value.x +
                             attribute.value.y +
                             attribute.value.z) /
                            3.0f);
                    } else if (
                        instruction.operation ==
                        compiler::ValueOperation::attribute_alpha) {
                        // Cycles treats a missing attribute as a zero
                        // FLOAT3 descriptor. Its Alpha projection is one,
                        // while an existing RGBA attribute supplies w.
                        value = make_float4(select(
                            1.0f,
                            attribute.value.w,
                            attribute.found));
                    } else {
                        value = attribute.value;
                    }
                    break;
                }
            default:
                break;
        }
        return project_surface_value(
            instruction.result_type, value);
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

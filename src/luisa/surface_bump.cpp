#include "surface_bump.h"

#include "surface_math.h"

#include <luisa/dsl/sugar.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float3 bump_core(
    const SurfaceBumpInput &input) noexcept {
    const auto rx = cross(input.dPdy, input.normal);
    const auto ry = cross(input.normal, input.dPdx);
    const auto determinant = dot(input.dPdx, rx);
    const auto surface_gradient =
        (input.height_x - input.height_center) * rx +
        (input.height_y - input.height_center) * ry;
    const auto determinant_sign = select(
        -1.0f,
        1.0f,
        determinant >= 0.0f);
    const auto perturbed_vector =
        input.filter_width * abs(determinant) *
            input.normal -
        input.distance * determinant_sign *
            surface_gradient;
    const auto perturbed_valid =
        length_squared(perturbed_vector) > 0.0f;
    const auto perturbed = safe_normalize(
        perturbed_vector,
        make_float3(0.0f));
    const auto blended = safe_normalize(
        input.strength * perturbed +
            (1.0f - input.strength) *
                input.normal,
        make_float3(0.0f));
    return select(
        input.normal,
        blended,
        perturbed_valid);
}

}// namespace

SurfacePoint surface_differential_sample_point(
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

SurfaceBumpConfiguration decode_surface_bump_configuration(
    std::uint64_t encoded) noexcept {
    return {.invert = (encoded & 1u) != 0u,
            .normal_linked = (encoded & 2u) != 0u,
            .object_space = (encoded & 4u) != 0u};
}

SurfaceBumpEvaluationDomain make_surface_bump_evaluation_domain(
    const SurfacePoint &point,
    Float filter_width_expression) noexcept {
    auto filter_width = max(filter_width_expression, 0.0f);
    auto point_x = point;
    point_x.position = point.position + point.dPdx * filter_width;
    point_x.object_position =
        point.object_position + point.object_dPdx * filter_width;
    point_x.generated = point.generated + point.generated_dx * filter_width;
    point_x.uv = point.uv + point.uv_dx * filter_width;
    point_x.barycentric =
        point.barycentric + point.barycentric_dx * filter_width;

    auto point_y = point;
    point_y.position = point.position + point.dPdy * filter_width;
    point_y.object_position =
        point.object_position + point.object_dPdy * filter_width;
    point_y.generated = point.generated + point.generated_dy * filter_width;
    point_y.uv = point.uv + point.uv_dy * filter_width;
    point_y.barycentric =
        point.barycentric + point.barycentric_dy * filter_width;
    return {.filter_width = std::move(filter_width),
            .point_x = std::move(point_x),
            .point_y = std::move(point_y)};
}

Float3 evaluate_surface_bump(
    const ShaderServices &services,
    const SurfacePoint &point,
    SurfaceBumpConfiguration configuration,
    Float3 normal,
    const SurfaceBumpEvaluationDomain &domain,
    Float height_center,
    Float height_x,
    Float height_y,
    Float distance_expression,
    Float strength_expression) noexcept {
    auto distance = configuration.invert
                        ? -distance_expression
                        : distance_expression;
    const auto strength = max(strength_expression, 0.0f);
    const auto input = SurfaceBumpInput{
        .normal = std::move(normal),
        .filter_width = domain.filter_width,
        .dPdx = configuration.object_space
                    ? point.object_dPdx
                    : point.dPdx,
        .dPdy = configuration.object_space
                    ? point.object_dPdy
                    : point.dPdy,
        .height_center = std::move(height_center),
        .height_x = std::move(height_x),
        .height_y = std::move(height_y),
        .distance = std::move(distance),
        .strength = strength,
        .normal_to_world_x = point.normal_to_world_x,
        .normal_to_world_y = point.normal_to_world_y,
        .normal_to_world_z = point.normal_to_world_z,
        .object_shading_normal = point.object_shading_normal,
        .shading_normal = point.shading_normal};
    return configuration.object_space
               ? bump_object(services, input)
               : bump_world(services, input);
}

Float3 evaluate_surface_bump(
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceBumpSvmConfiguration &configuration,
    Float3 normal,
    Float filter_width,
    Float height_center,
    Float height_x,
    Float height_y,
    Float distance_expression,
    Float strength_expression) noexcept {
    const auto distance = select(
        distance_expression,
        -distance_expression,
        configuration.invert);
    const auto strength = max(strength_expression, 0.0f);
    const auto input = SurfaceBumpInput{
        .normal = std::move(normal),
        .filter_width = std::move(filter_width),
        .dPdx = select(point.dPdx,
                       point.object_dPdx,
                       configuration.object_space),
        .dPdy = select(point.dPdy,
                       point.object_dPdy,
                       configuration.object_space),
        .height_center = std::move(height_center),
        .height_x = std::move(height_x),
        .height_y = std::move(height_y),
        .distance = distance,
        .strength = strength,
        .normal_to_world_x = point.normal_to_world_x,
        .normal_to_world_y = point.normal_to_world_y,
        .normal_to_world_z = point.normal_to_world_z,
        .object_shading_normal = point.object_shading_normal,
        .shading_normal = point.shading_normal};
    Float3 result = point.shading_normal;
    $if(configuration.object_space) {
        result = bump_object(services, input);
    }
    $else {
        result = bump_world(services, input);
    };
    return result;
}

Float3 bump_world_inline(
    const SurfaceBumpInput &input) noexcept {
    return bump_core(input);
}

Float3 bump_object_inline(
    const SurfaceBumpInput &input) noexcept {
    const auto column_x = input.normal_to_world_x;
    const auto column_y = input.normal_to_world_y;
    const auto column_z = input.normal_to_world_z;
    const auto transform_determinant = dot(
        column_x, cross(column_y, column_z));
    const auto inverse_determinant =
        1.0f / select(
                   1.0f,
                   transform_determinant,
                   abs(transform_determinant) > 1.0e-20f);
    auto transformed = input;
    transformed.normal = safe_normalize(
        make_float3(
            dot(input.normal, cross(column_y, column_z)),
            dot(input.normal, cross(column_z, column_x)),
            dot(input.normal, cross(column_x, column_y))) *
            inverse_determinant,
        input.object_shading_normal);
    const auto object_normal = bump_core(transformed);
    return safe_normalize(
        column_x * object_normal.x +
            column_y * object_normal.y +
            column_z * object_normal.z,
        input.shading_normal);
}

Float3 bump_world(
    const ShaderServices &services,
    const SurfaceBumpInput &input) noexcept {
    if (const auto provider =
            services.surface_bump_provider()) {
        return provider->evaluate_world(input);
    }
    return bump_world_inline(input);
}

Float3 bump_object(
    const ShaderServices &services,
    const SurfaceBumpInput &input) noexcept {
    if (const auto provider =
            services.surface_bump_provider()) {
        return provider->evaluate_object(input);
    }
    return bump_object_inline(input);
}

}// namespace psycles::luisa_backend::detail

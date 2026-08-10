#include "surface_bump.h"

#include "surface_math.h"

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

#include "surface_normal_map.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float3 normal_map_safe_normalize(
    Float3 value,
    Float3 fallback) noexcept {
    const auto valid = dot(value, value) > 1.0e-20f;
    auto selected = select(fallback, value, valid);
    const auto fallback_valid =
        dot(selected, selected) > 1.0e-20f;
    selected = select(
        make_float3(0.0f, 0.0f, 1.0f),
        selected,
        fallback_valid);
    return normalize(selected);
}

[[nodiscard]] Float3 transform_object_normal(
    const SurfaceNormalMapInput &input,
    Float3 object_normal) noexcept {
    return normal_map_safe_normalize(
        input.normal_to_world_x * object_normal.x +
            input.normal_to_world_y * object_normal.y +
            input.normal_to_world_z * object_normal.z,
        input.shading_normal);
}

[[nodiscard]] Float3 tangent_world_normal(
    const SurfaceNormalMapInput &input,
    Float3 mapped,
    Float3 object_base) noexcept {
    const auto object_bitangent =
        input.tangent_sign *
        cross(object_base, input.object_tangent);
    const auto object_normal = normal_map_safe_normalize(
        input.object_tangent * mapped.x +
            object_bitangent * mapped.y +
            object_base * mapped.z,
        make_float3(0.0f));
    auto world = transform_object_normal(
        input, object_normal);
    return select(
        world, -world, input.back_facing);
}

[[nodiscard]] Bool tangent_available(
    const SurfaceNormalMapInput &input,
    Float3 object_base) noexcept {
    return (input.geometry_index != ~0u) &
           !input.is_curve &
           input.tangent_attribute_found &
           (length_squared(object_base) > 1.0e-20f) &
           (length_squared(input.object_tangent) > 1.0e-20f) &
           (abs(input.tangent_sign) > 1.0e-20f);
}

[[nodiscard]] Float3 blend_world_normal(
    const SurfaceNormalMapInput &input,
    Float3 world) noexcept {
    world = select(
        world, -world, input.back_facing);
    const auto nonnegative_strength =
        max(input.strength, 0.0f);
    return normal_map_safe_normalize(
        input.shading_normal +
            (world - input.shading_normal) *
                nonnegative_strength,
        input.shading_normal);
}

}// namespace

Float3 normal_map_tangent_displaced_inline(
    const SurfaceNormalMapInput &input) noexcept {
    auto mapped = input.mapped;
    mapped.x *= input.strength;
    mapped.y *= input.strength;
    mapped.z =
        1.0f +
        (mapped.z - 1.0f) *
            clamp(input.strength, 0.0f, 1.0f);
    const auto object_base = normal_map_safe_normalize(
        input.object_shading_normal,
        make_float3(0.0f));
    const auto world = tangent_world_normal(
        input, mapped, object_base);
    return select(
        input.shading_normal,
        world,
        tangent_available(input, object_base));
}

Float3 normal_map_tangent_original_inline(
    const SurfaceNormalMapInput &input) noexcept {
    auto object_base = normal_map_safe_normalize(
        input.object_shading_normal,
        make_float3(0.0f));
    object_base = select(
        object_base,
        input.undisplaced_object_shading_normal,
        input.triangle_smooth);
    auto mapped = input.mapped;
    $if(!input.triangle_smooth) {
        mapped.x *= input.strength;
        mapped.y *= input.strength;
        mapped.z =
            1.0f +
            (mapped.z - 1.0f) *
                clamp(input.strength, 0.0f, 1.0f);
    };
    auto world = tangent_world_normal(
        input, mapped, object_base);
    const auto linearly_blended = normal_map_safe_normalize(
        input.shading_normal +
            (world - input.shading_normal) *
                max(input.strength, 0.0f),
        input.shading_normal);
    world = select(
        world,
        linearly_blended,
        input.triangle_smooth);
    return select(
        input.shading_normal,
        world,
        tangent_available(input, object_base));
}

Float3 normal_map_object_inline(
    const SurfaceNormalMapInput &input) noexcept {
    return blend_world_normal(
        input,
        transform_object_normal(input, input.mapped));
}

Float3 normal_map_world_inline(
    const SurfaceNormalMapInput &input) noexcept {
    return blend_world_normal(
        input,
        normal_map_safe_normalize(
            input.mapped,
            input.shading_normal));
}

Float3 normal_map_tangent_displaced(
    const ShaderServices &services,
    const SurfaceNormalMapInput &input) noexcept {
    if (const auto provider =
            services.surface_normal_map_provider()) {
        return provider->evaluate_tangent_displaced(input);
    }
    return normal_map_tangent_displaced_inline(input);
}

Float3 normal_map_tangent_original(
    const ShaderServices &services,
    const SurfaceNormalMapInput &input) noexcept {
    if (const auto provider =
            services.surface_normal_map_provider()) {
        return provider->evaluate_tangent_original(input);
    }
    return normal_map_tangent_original_inline(input);
}

Float3 normal_map_object(
    const ShaderServices &services,
    const SurfaceNormalMapInput &input) noexcept {
    if (const auto provider =
            services.surface_normal_map_provider()) {
        return provider->evaluate_object(input);
    }
    return normal_map_object_inline(input);
}

Float3 normal_map_world(
    const ShaderServices &services,
    const SurfaceNormalMapInput &input) noexcept {
    if (const auto provider =
            services.surface_normal_map_provider()) {
        return provider->evaluate_world(input);
    }
    return normal_map_world_inline(input);
}

}// namespace psycles::luisa_backend::detail

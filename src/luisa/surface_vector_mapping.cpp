#include "surface_vector_mapping.h"

namespace psycles::luisa_backend::detail {

Float3 rotate_euler(
    Float3 value, Float3 rotation) noexcept {
    auto sx = sin(rotation.x);
    auto cx = cos(rotation.x);
    auto sy = sin(rotation.y);
    auto cy = cos(rotation.y);
    auto sz = sin(rotation.z);
    auto cz = cos(rotation.z);
    auto x_rotated = make_float3(
        value.x,
        cx * value.y - sx * value.z,
        sx * value.y + cx * value.z);
    auto y_rotated = make_float3(
        cy * x_rotated.x + sy * x_rotated.z,
        x_rotated.y,
        -sy * x_rotated.x + cy * x_rotated.z);
    return make_float3(
        cz * y_rotated.x - sz * y_rotated.y,
        sz * y_rotated.x + cz * y_rotated.y,
        y_rotated.z);
}

Float3 rotate_euler_transposed(
    Float3 value, Float3 rotation) noexcept {
    auto sx = sin(rotation.x);
    auto cx = cos(rotation.x);
    auto sy = sin(rotation.y);
    auto cy = cos(rotation.y);
    auto sz = sin(rotation.z);
    auto cz = cos(rotation.z);
    return make_float3(
        cy * cz * value.x +
            cy * sz * value.y -
            sy * value.z,
        (sy * sx * cz - cx * sz) * value.x +
            (sy * sx * sz + cx * cz) * value.y +
            cy * sx * value.z,
        (sy * cx * cz + sx * sz) * value.x +
            (sy * cx * sz - sx * cz) * value.y +
            cy * cx * value.z);
}

Float3 safe_divide_components(
    Float3 numerator,
    Float3 denominator) noexcept {
    return make_float3(
        select(
            0.0f,
            numerator.x / denominator.x,
            denominator.x != 0.0f),
        select(
            0.0f,
            numerator.y / denominator.y,
            denominator.y != 0.0f),
        select(
            0.0f,
            numerator.z / denominator.z,
            denominator.z != 0.0f));
}

Float3 map_vector_point_inline(
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) noexcept {
    return rotate_euler(input * scale, rotation) +
           location;
}

Float3 map_vector_texture_inline(
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) noexcept {
    return safe_divide_components(
        rotate_euler_transposed(
            input - location,
            rotation),
        scale);
}

Float3 map_vector_direction_inline(
    Float3 input,
    Float3 rotation,
    Float3 scale) noexcept {
    return rotate_euler(input * scale, rotation);
}

Float3 map_vector_normal_inline(
    Float3 input,
    Float3 rotation,
    Float3 scale) noexcept {
    auto mapped = rotate_euler(
        safe_divide_components(input, scale),
        rotation);
    auto mapped_length = length(mapped);
    return mapped /
           select(
               1.0f,
               mapped_length,
               mapped_length != 0.0f);
}

Float3 map_vector_point(
    const ShaderServices &services,
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) noexcept {
    if (const auto provider =
            services.surface_vector_mapping_provider()) {
        return provider->map_point(
            input, location, rotation, scale);
    }
    return map_vector_point_inline(
        input, location, rotation, scale);
}

Float3 map_vector_texture(
    const ShaderServices &services,
    Float3 input,
    Float3 location,
    Float3 rotation,
    Float3 scale) noexcept {
    if (const auto provider =
            services.surface_vector_mapping_provider()) {
        return provider->map_texture(
            input, location, rotation, scale);
    }
    return map_vector_texture_inline(
        input, location, rotation, scale);
}

Float3 map_vector_direction(
    const ShaderServices &services,
    Float3 input,
    Float3 rotation,
    Float3 scale) noexcept {
    if (const auto provider =
            services.surface_vector_mapping_provider()) {
        return provider->map_vector(
            input, rotation, scale);
    }
    return map_vector_direction_inline(
        input, rotation, scale);
}

Float3 map_vector_normal(
    const ShaderServices &services,
    Float3 input,
    Float3 rotation,
    Float3 scale) noexcept {
    if (const auto provider =
            services.surface_vector_mapping_provider()) {
        return provider->map_normal(
            input, rotation, scale);
    }
    return map_vector_normal_inline(
        input, rotation, scale);
}

}// namespace psycles::luisa_backend::detail

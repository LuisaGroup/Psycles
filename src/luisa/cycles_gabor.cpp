#include <psycles/luisa/cycles_gabor.h>

#include <psycles/compiler/cycles_svm_types.h>
#include <psycles/luisa/cycles_noise.h>

#include <luisa/dsl/func.h>
#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_gabor {

namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;

constexpr auto impulses_count = 8u;
constexpr auto pi = 3.14159265358979323846f;
constexpr auto sqrt_two = 1.41421356237309504880f;

[[nodiscard]] Float2 polar_to_cartesian(Float radius, Float angle) noexcept {
    return make_float2(radius * cos(angle), radius * sin(angle));
}

[[nodiscard]] Float3 spherical_to_direction(Float inclination,
                                             Float azimuth) noexcept {
    const auto sine = sin(inclination);
    return make_float3(sine * cos(azimuth), sine * sin(azimuth),
                       cos(inclination));
}

[[nodiscard]] Float2 hash_float3_to_float2(Float3 value) noexcept {
    return make_float2(
        cycles_noise::hash_float3(value),
        cycles_noise::hash_float3(make_float3(value.z, value.x, value.y)));
}

[[nodiscard]] Float2 hash_float4_to_float2(Float4 value) noexcept {
    return make_float2(
        cycles_noise::hash_float4(value),
        cycles_noise::hash_float4(
            make_float4(value.z, value.x, value.w, value.y)));
}

[[nodiscard]] Float2 compute_2d_gabor_kernel(
    Float2 position, Float frequency, Float orientation) noexcept {
    const auto distance_squared = dot(position, position);
    const auto hann_window = 0.5f + 0.5f * cos(pi * distance_squared);
    const auto gaussian_envelope = exp(-pi * distance_squared);
    const auto windowed_gaussian_envelope = gaussian_envelope * hann_window;
    const auto frequency_vector = polar_to_cartesian(frequency, orientation);
    const auto angle = 2.0f * pi * dot(position, frequency_vector);
    return polar_to_cartesian(windowed_gaussian_envelope, angle);
}

[[nodiscard]] Float compute_2d_gabor_standard_deviation() noexcept {
    constexpr auto integral_of_gabor_squared = 0.25f;
    constexpr auto second_moment = 0.5f;
    return sqrt(static_cast<float>(impulses_count) * second_moment *
                integral_of_gabor_squared);
}

[[nodiscard]] Float2 compute_2d_gabor_noise_cell(
    Float2 cell, Float2 position, Float frequency, Float isotropy,
    Float base_orientation) noexcept {
    Float2 noise = make_float2(0.0f);
    $for (i, impulses_count) {
        const auto seed_index = cast<float>(i * 3u);
        const auto seed_for_orientation = make_float3(cell, seed_index);
        const auto seed_for_kernel_center =
            make_float3(cell, seed_index + 1.0f);
        const auto seed_for_weight = make_float3(cell, seed_index + 2.0f);
        const auto random_orientation =
            (cycles_noise::hash_float3(seed_for_orientation) - 0.5f) * pi;
        const auto orientation =
            base_orientation + random_orientation * isotropy;
        const auto kernel_center =
            hash_float3_to_float2(seed_for_kernel_center);
        const auto position_in_kernel_space = position - kernel_center;
        $if (dot(position_in_kernel_space, position_in_kernel_space) >= 1.0f) {
            $continue;
        };
        const auto weight = select(
            1.0f, -1.0f,
            cycles_noise::hash_float3(seed_for_weight) < 0.5f);
        noise += weight * compute_2d_gabor_kernel(
                              position_in_kernel_space, frequency, orientation);
    };
    return noise;
}

[[nodiscard]] Float2 compute_2d_gabor_noise(
    Float2 coordinates, Float frequency, Float isotropy,
    Float base_orientation) noexcept {
    const auto cell_position = floor(coordinates);
    const auto local_position = coordinates - cell_position;
    Float2 sum = make_float2(0.0f);
    $for (j, -1, 2) {
        $for (i, -1, 2) {
            const auto cell_offset = make_float2(cast<float>(i), cast<float>(j));
            const auto current_cell_position = cell_position + cell_offset;
            const auto position_in_cell_space = local_position - cell_offset;
            sum += compute_2d_gabor_noise_cell(
                current_cell_position, position_in_cell_space, frequency,
                isotropy, base_orientation);
        };
    };
    return sum;
}

[[nodiscard]] Float2 compute_3d_gabor_kernel(
    Float3 position, Float frequency, Float3 orientation) noexcept {
    const auto distance_squared = dot(position, position);
    const auto hann_window = 0.5f + 0.5f * cos(pi * distance_squared);
    const auto gaussian_envelope = exp(-pi * distance_squared);
    const auto windowed_gaussian_envelope = gaussian_envelope * hann_window;
    const auto frequency_vector = frequency * orientation;
    const auto angle = 2.0f * pi * dot(position, frequency_vector);
    return polar_to_cartesian(windowed_gaussian_envelope, angle);
}

[[nodiscard]] Float compute_3d_gabor_standard_deviation() noexcept {
    constexpr auto integral_of_gabor_squared = 1.0f / (4.0f * sqrt_two);
    constexpr auto second_moment = 0.5f;
    return sqrt(static_cast<float>(impulses_count) * second_moment *
                integral_of_gabor_squared);
}

[[nodiscard]] Float3 compute_3d_orientation(
    Float3 orientation, Float isotropy, Float4 seed) noexcept {
    Float3 result;
    $if (isotropy == 0.0f) { result = orientation; }
    $else {
        auto inclination = acos(orientation.z);
        auto azimuth =
            select(-1.0f, 1.0f, orientation.y >= 0.0f) *
            acos(orientation.x /
                 length(make_float2(orientation.x, orientation.y)));
        const auto random_angles = hash_float4_to_float2(seed) * pi;
        inclination += random_angles.x * isotropy;
        azimuth += random_angles.y * isotropy;
        result = spherical_to_direction(inclination, azimuth);
    };
    return result;
}

[[nodiscard]] Float2 compute_3d_gabor_noise_cell(
    Float3 cell, Float3 position, Float frequency, Float isotropy,
    Float3 base_orientation) noexcept {
    Float2 noise = make_float2(0.0f);
    $for (i, impulses_count) {
        const auto seed_index = cast<float>(i * 3u);
        const auto seed_for_orientation = make_float4(cell, seed_index);
        const auto seed_for_kernel_center =
            make_float4(cell, seed_index + 1.0f);
        const auto seed_for_weight = make_float4(cell, seed_index + 2.0f);
        const auto orientation = compute_3d_orientation(
            base_orientation, isotropy, seed_for_orientation);
        const auto kernel_center =
            cycles_noise::hash_float4_to_color(seed_for_kernel_center);
        const auto position_in_kernel_space = position - kernel_center;
        $if (dot(position_in_kernel_space, position_in_kernel_space) >= 1.0f) {
            $continue;
        };
        const auto weight = select(
            1.0f, -1.0f,
            cycles_noise::hash_float4(seed_for_weight) < 0.5f);
        noise += weight * compute_3d_gabor_kernel(
                              position_in_kernel_space, frequency, orientation);
    };
    return noise;
}

[[nodiscard]] Float2 compute_3d_gabor_noise(
    Float3 coordinates, Float frequency, Float isotropy,
    Float3 base_orientation) noexcept {
    const auto cell_position = floor(coordinates);
    const auto local_position = coordinates - cell_position;
    Float2 sum = make_float2(0.0f);
    $for (k, -1, 2) {
        $for (j, -1, 2) {
            $for (i, -1, 2) {
                const auto cell_offset = make_float3(
                    cast<float>(i), cast<float>(j), cast<float>(k));
                const auto current_cell_position = cell_position + cell_offset;
                const auto position_in_cell_space = local_position - cell_offset;
                sum += compute_3d_gabor_noise_cell(
                    current_cell_position, position_in_cell_space, frequency,
                    isotropy, base_orientation);
            };
        };
    };
    return sum;
}

} // namespace

Float3 evaluate(luisa::compute::Expr<std::uint32_t> gabor_type_expression,
                Float3 coordinates, Float3 orientation_3d, Float scale,
                Float frequency, Float anisotropy,
                Float orientation_2d) noexcept {
    const UInt gabor_type{gabor_type_expression};
    const auto scaled_coordinates = coordinates * scale;
    const auto isotropy = 1.0f - clamp(anisotropy, 0.0f, 1.0f);
    frequency = max(0.001f, frequency);
    Float2 phasor = make_float2(0.0f);
    Float standard_deviation = 1.0f;
    $switch (gabor_type) {
        $case(static_cast<std::uint32_t>(NODE_GABOR_TYPE_2D)) {
            phasor = compute_2d_gabor_noise(
                scaled_coordinates.xy(), frequency, isotropy, orientation_2d);
            standard_deviation = compute_2d_gabor_standard_deviation();
        };
        $case(static_cast<std::uint32_t>(NODE_GABOR_TYPE_3D)) {
            const auto orientation = normalize(orientation_3d);
            phasor = compute_3d_gabor_noise(
                scaled_coordinates, frequency, isotropy, orientation);
            standard_deviation = compute_3d_gabor_standard_deviation();
        };
        $default {
            luisa::compute::dsl::unreachable("invalid Cycles Gabor type");
        };
    };
    const auto normalization_factor = 6.0f * standard_deviation;
    const auto value =
        (phasor.y / normalization_factor) * 0.5f + 0.5f;
    const auto phase = (atan2(phasor.y, phasor.x) + pi) / (2.0f * pi);
    const auto intensity = length(phasor) / normalization_factor;
    return make_float3(value, phase, intensity);
}

} // namespace psycles::luisa_backend::cycles_gabor

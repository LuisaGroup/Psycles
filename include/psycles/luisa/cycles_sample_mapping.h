#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error                                                                 \
    "Include <psycles/luisa/cycles_sample_mapping.h> through the Psycles::luisa target."
#endif

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_sample_mapping {

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float inverse_pi = 0.31830988618379067154f;
inline constexpr float inverse_two_pi = 0.15915494309189533577f;

struct OrthonormalBasis {
    luisa::compute::Float3 tangent;
    luisa::compute::Float3 bitangent;
};

struct CosineHemisphereSample {
    luisa::compute::Float3 direction;
    luisa::compute::Float pdf;
};

struct UniformConeSample {
    luisa::compute::Float3 direction;
    luisa::compute::Float cosine;
    luisa::compute::Float pdf;
};

// This is the Cycles square-to-disk measure-preserving map. The branch
// at |a| == |b| and the center case are part of the mapping definition:
// changing either changes the deterministic pairing between a
// two-dimensional RNG sample and the generated direction even though
// the resulting density stays uniform.
[[nodiscard]] inline luisa::compute::Float2 sample_uniform_disk(
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;
    const auto a = 2.0f * random.x - 1.0f;
    const auto b = 2.0f * random.y - 1.0f;
    const auto x_major = a * a > b * b;
    const auto safe_a = select(1.0f, a, a != 0.0f);
    const auto safe_b = select(1.0f, b, b != 0.0f);
    const auto radius = select(b, a, x_major);
    const auto angle = select(0.5f * pi - 0.25f * pi * (a / safe_b),
        0.25f * pi * (b / safe_a),
        x_major);
    const auto mapped = make_float2(cos(angle), sin(angle)) * radius;
    return select(mapped, make_float2(0.0f), (a == 0.0f) & (b == 0.0f));
}

// Cycles fixes the otherwise free rotation around a normal with this
// algebraic basis. Keeping that convention is required for
// bitwise-correlated sample dimensions: an arbitrary valid basis
// preserves the PDF but sends the same RNG pair to a different
// world-space direction.
[[nodiscard]] inline OrthonormalBasis make_orthonormals(
    luisa::compute::Float3 normal) noexcept {
    using namespace luisa::compute;
    const auto general = make_float3(
        normal.z - normal.y, normal.x - normal.z, normal.y - normal.x);
    const auto equal_components = make_float3(
        normal.z - normal.y, normal.x + normal.z, -normal.y - normal.x);
    const auto use_general =
        (normal.x != normal.y) | (normal.x != normal.z);
    const auto tangent =
        normalize(select(equal_components, general, use_general));
    return {.tangent = tangent, .bitangent = cross(normal, tangent)};
}

// Cycles' Sheen LTC fixes its azimuthal frame with the incoming direction
// when that tangent is usable, and falls back to make_orthonormals otherwise.
// safe_normalize here has the exact zero-only contract from Cycles.
[[nodiscard]] inline OrthonormalBasis make_orthonormals_safe_tangent(
    luisa::compute::Float3 normal,
    luisa::compute::Float3 candidate_tangent) noexcept {
    using namespace luisa::compute;
    const auto unnormalized_bitangent =
        cross(normal, candidate_tangent);
    const auto length = sqrt(dot(
        unnormalized_bitangent, unnormalized_bitangent));
    const auto bitangent = unnormalized_bitangent /
                           select(1.0f, length, length != 0.0f);
    const auto candidate = OrthonormalBasis{
        .tangent = cross(bitangent, normal),
        .bitangent = bitangent};
    const auto fallback = make_orthonormals(normal);
    const auto valid = dot(bitangent, bitangent) >= 0.99f;
    return {
        .tangent = select(fallback.tangent, candidate.tangent, valid),
        .bitangent = select(
            fallback.bitangent, candidate.bitangent, valid)};
}

// The disk map, basis orientation, and lack of a final renormalization
// are one deterministic sampling invariant. The input normal is
// expected to be unit length, matching the Cycles closure contract.
[[nodiscard]] inline CosineHemisphereSample sample_cosine_hemisphere(
    luisa::compute::Float3 normal,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;
    const auto disk = sample_uniform_disk(random);
    const auto cosine = sqrt(max(1.0f - dot(disk, disk), 0.0f));
    const auto basis = make_orthonormals(normal);
    return {.direction = disk.x * basis.tangent +
                         disk.y * basis.bitangent + cosine * normal,
        .pdf = cosine * inverse_pi};
}

// Isotropic GGX visible-normal sampling is also a deterministic sample
// mapping. Cycles fixes both the world-space basis around `normal` and
// the concentric square-to-disk map; changing either preserves the GGX
// density but breaks path-by-path correlation with its sampler. Inputs
// follow the Cycles closure contract: normal and incoming are unit
// length, incoming is in the upper hemisphere, and alpha is the squared
// perceptual roughness.
[[nodiscard]] inline luisa::compute::Float3
sample_ggx_visible_normal(luisa::compute::Float3 normal,
    luisa::compute::Float3 incoming,
    luisa::compute::Float alpha,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;

    const auto world_basis = make_orthonormals(normal);
    const auto local_incoming =
        make_float3(dot(world_basis.tangent, incoming),
            dot(world_basis.bitangent, incoming),
            dot(normal, incoming));

    // Heitz 2018, sections 3.2 and 4.1.
    const auto stretched_incoming =
        normalize(make_float3(alpha * local_incoming.x,
            alpha * local_incoming.y,
            local_incoming.z));
    const auto projected_length_squared =
        stretched_incoming.x * stretched_incoming.x +
        stretched_incoming.y * stretched_incoming.y;
    const auto projected_tangent =
        make_float3(-stretched_incoming.y, stretched_incoming.x, 0.0f) /
        sqrt(max(projected_length_squared, 1.0e-20f));
    const auto basis_x = select(make_float3(1.0f, 0.0f, 0.0f),
        projected_tangent,
        projected_length_squared > 1.0e-7f);
    const auto basis_y = cross(stretched_incoming, basis_x);

    // Heitz 2018, sections 4.2 and 4.3.
    auto disk = sample_uniform_disk(random);
    const auto projected_area = 0.5f * (1.0f + stretched_incoming.z);
    disk.y = lerp(sqrt(max(1.0f - disk.x * disk.x, 0.0f)),
        disk.y,
        projected_area);
    const auto hemisphere_z =
        sqrt(max(1.0f - disk.x * disk.x - disk.y * disk.y, 0.0f));
    const auto stretched_half = basis_x * disk.x + basis_y * disk.y +
                                stretched_incoming * hemisphere_z;

    // Transform the visible normal back to the GGX ellipsoid and then
    // to world space. Cycles does not add a final world-space
    // normalization.
    const auto local_half =
        normalize(make_float3(alpha * stretched_half.x,
            alpha * stretched_half.y,
            max(stretched_half.z, 0.0f)));
    const auto half_vector = world_basis.tangent * local_half.x +
                             world_basis.bitangent * local_half.y +
                             normal * local_half.z;
    return half_vector;
}

[[nodiscard]] inline luisa::compute::Float3
sample_ggx_visible_normal_reflection(luisa::compute::Float3 normal,
    luisa::compute::Float3 incoming,
    luisa::compute::Float alpha,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;
    const auto half_vector =
        sample_ggx_visible_normal(normal, incoming, alpha, random);
    return 2.0f * dot(incoming, half_vector) * half_vector - incoming;
}

[[nodiscard]] inline luisa::compute::Float cosine_hemisphere_pdf(
    luisa::compute::Float3 normal,
    luisa::compute::Float3 direction) noexcept {
    using namespace luisa::compute;
    const auto cosine = dot(normal, direction);
    return select(0.0f, cosine * inverse_pi, cosine > 0.0f);
}

// Cycles evaluates 1 - cos(theta) directly from sin(theta)^2 so tiny
// finite lights retain their solid angle in single precision. The
// branch point and second-order limit are part of the light-sampling
// measure.
[[nodiscard]] inline luisa::compute::Float
sin_squared_to_one_minus_cosine(
    luisa::compute::Float sine_squared) noexcept {
    using namespace luisa::compute;
    return select(0.5f * sine_squared,
        1.0f - sqrt(max(1.0f - sine_squared, 0.0f)),
        sine_squared > 0.0004f);
}

// Uniform solid-angle cone sampling in Cycles is not the usual polar
// (u, phi) map. It first applies the concentric square-to-disk map and
// then remaps the disk radius. This keeps the deterministic Sobol
// pairing as well as the target density.
[[nodiscard]] inline UniformConeSample sample_uniform_cone(
    luisa::compute::Float3 axis,
    luisa::compute::Float one_minus_cosine,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;

    const auto disk = sample_uniform_disk(random);
    const auto radius_squared = dot(disk, disk);
    const auto cosine = 1.0f - radius_squared * one_minus_cosine;
    const auto radial_scale = sqrt(max(
        one_minus_cosine * (2.0f - one_minus_cosine * radius_squared),
        0.0f));
    const auto mapped = disk * radial_scale;
    const auto basis = make_orthonormals(axis);
    const auto finite_direction = mapped.x * basis.tangent +
                                  mapped.y * basis.bitangent +
                                  cosine * axis;
    const auto finite_pdf = inverse_two_pi / one_minus_cosine;
    const auto finite = one_minus_cosine > 0.0f;
    return {.direction = select(axis, finite_direction, finite),
        .cosine = select(1.0f, cosine, finite),
        .pdf = select(1.0f, finite_pdf, finite)};
}

[[nodiscard]] inline luisa::compute::Float3 sample_uniform_sphere(
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;
    const auto z = 1.0f - 2.0f * random.x;
    const auto radius = sqrt(max(1.0f - z * z, 0.0f));
    const auto phi = 2.0f * pi * random.y;
    return make_float3(radius * cos(phi), radius * sin(phi), z);
}

} // namespace psycles::luisa_backend::cycles_sample_mapping

#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error                                                                 \
    "Include <psycles/luisa/cycles_sample_mapping.h> through the Psycles::luisa target."
#endif

#include <cmath>

#include <luisa/dsl/sugar.h>

#include <psycles/luisa/native_vector_math.h>

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

struct UniformHemisphereSample {
    luisa::compute::Float3 direction;
    luisa::compute::Float pdf;
};

struct UniformConeSample {
    luisa::compute::Float3 direction;
    luisa::compute::Float cosine;
    luisa::compute::Float pdf;
};

// Cycles deliberately switches to the second-order small-angle relation.
// This is part of the cone measure: sampling and PDF evaluation must use the
// same value or MIS is biased, and replacing it with an independently stable
// trigonometric identity changes path-by-path results for solar discs.
[[nodiscard]] inline float
one_minus_cosine_from_angle(float angle) noexcept {
    return angle > 0.02f ? 1.0f - std::cos(angle)
                         : 0.5f * angle * angle;
}

[[nodiscard]] inline luisa::compute::Float
one_minus_cosine_from_angle(luisa::compute::Float angle) noexcept {
    using namespace luisa::compute;
    return select(0.5f * angle * angle,
        1.0f - cos(angle),
        angle > 0.02f);
}

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
// safe_normalize here has the exact zero-only contract from Cycles; its
// accepted-domain arithmetic remains backend-native.
[[nodiscard]] inline OrthonormalBasis make_orthonormals_safe_tangent(
    luisa::compute::Float3 normal,
    luisa::compute::Float3 candidate_tangent) noexcept {
    using namespace luisa::compute;
    const auto unnormalized_bitangent =
        cross(normal, candidate_tangent);
    const auto bitangent =
        native_vector_math::safe_normalize_nonzero(
            unnormalized_bitangent);
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

// Cycles' anisotropic microfacet path deliberately uses the unsafe tangent
// basis: a linked degenerate tangent is material data, not an instruction to
// substitute an unrelated azimuth. Setup makes this basis observable only
// when alpha_x != alpha_y.
[[nodiscard]] inline OrthonormalBasis make_orthonormals_tangent(
    luisa::compute::Float3 normal,
    luisa::compute::Float3 tangent) noexcept {
    using namespace luisa::compute;
    const auto bitangent = native_vector_math::normalize_unchecked(
        cross(normal, tangent));
    return {
        .tangent = cross(bitangent, normal),
        .bitangent = bitangent};
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

// Cycles' uniform-hemisphere map is not the usual polar parameterization.
// It lifts the same concentric disk sample with z = 1 - r^2 and rescales the
// disk by sqrt(z + 1). This exact square-to-direction mapping is observable
// through the renderer's correlated RNG dimensions.
[[nodiscard]] inline UniformHemisphereSample sample_uniform_hemisphere(
    luisa::compute::Float3 normal,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;
    auto disk = sample_uniform_disk(random);
    const auto z = 1.0f - dot(disk, disk);
    disk *= sqrt(max(z + 1.0f, 0.0f));
    const auto basis = make_orthonormals(normal);
    return {.direction = disk.x * basis.tangent +
                         disk.y * basis.bitangent + z * normal,
        .pdf = inverse_two_pi};
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
    const OrthonormalBasis &world_basis,
    luisa::compute::Float3 incoming,
    luisa::compute::Float alpha_x,
    luisa::compute::Float alpha_y,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;

    const auto local_incoming =
        make_float3(dot(world_basis.tangent, incoming),
            dot(world_basis.bitangent, incoming),
            dot(normal, incoming));

    // Heitz 2018, sections 3.2 and 4.1.
    const auto stretched_incoming =
        normalize(make_float3(alpha_x * local_incoming.x,
            alpha_y * local_incoming.y,
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
        normalize(make_float3(alpha_x * stretched_half.x,
            alpha_y * stretched_half.y,
            max(stretched_half.z, 0.0f)));
    const auto half_vector = world_basis.tangent * local_half.x +
                             world_basis.bitangent * local_half.y +
                             normal * local_half.z;
    return half_vector;
}

[[nodiscard]] inline luisa::compute::Float3
sample_ggx_visible_normal(luisa::compute::Float3 normal,
    luisa::compute::Float3 incoming,
    luisa::compute::Float alpha,
    luisa::compute::Float2 random) noexcept {
    return sample_ggx_visible_normal(
        normal,
        make_orthonormals(normal),
        incoming,
        alpha,
        alpha,
        random);
}

// Cycles' Beckmann sampler draws the distribution of visible normals, not
// the underlying NDF. The error-function approximations and the bounded
// Newton/bisection solve below are part of its deterministic sample mapping;
// replacing them with a statistically equivalent inverse CDF breaks
// path-by-path correlation with Cycles.
[[nodiscard]] inline luisa::compute::Float fast_erf(
    luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    const auto magnitude = abs(value);
    const auto crushed = 1.0f - (1.0f - magnitude);
    auto polynomial = fma(0.0000430638f, crushed, 0.0002765672f);
    polynomial = fma(polynomial, crushed, 0.0001520143f);
    polynomial = fma(polynomial, crushed, 0.0092705272f);
    polynomial = fma(polynomial, crushed, 0.0422820123f);
    polynomial = fma(polynomial, crushed, 0.0705230784f);
    polynomial = fma(polynomial, crushed, 1.0f);
    const auto square = polynomial * polynomial;
    const auto fourth = square * square;
    const auto eighth = fourth * fourth;
    const auto sixteenth = eighth * eighth;
    const auto unsigned_result = 1.0f - 1.0f / sixteenth;
    const auto signed_result =
        select(-unsigned_result, unsigned_result, value >= 0.0f);
    const auto signed_limit = select(-1.0f, 1.0f, value >= 0.0f);
    return select(signed_result, signed_limit, magnitude >= 12.3f);
}

[[nodiscard]] inline luisa::compute::Float fast_inverse_erf(
    luisa::compute::Float value) noexcept {
    using namespace luisa::compute;
    const auto magnitude = min(abs(value), 0.99999994f);
    const auto base = -log((1.0f - magnitude) * (1.0f + magnitude));

    const auto central_x = base - 2.5f;
    auto central = fma(2.81022636e-08f, central_x, 3.43273939e-07f);
    central = fma(central, central_x, -3.5233877e-06f);
    central = fma(central, central_x, -4.39150654e-06f);
    central = fma(central, central_x, 0.00021858087f);
    central = fma(central, central_x, -0.00125372503f);
    central = fma(central, central_x, -0.00417768164f);
    central = fma(central, central_x, 0.246640727f);
    central = fma(central, central_x, 1.50140941f);

    const auto tail_x = sqrt(base) - 3.0f;
    auto tail = fma(-0.000200214257f, tail_x, 0.000100950558f);
    tail = fma(tail, tail_x, 0.00134934322f);
    tail = fma(tail, tail_x, -0.00367342844f);
    tail = fma(tail, tail_x, 0.00573950773f);
    tail = fma(tail, tail_x, -0.0076224613f);
    tail = fma(tail, tail_x, 0.00943887047f);
    tail = fma(tail, tail_x, 1.00167406f);
    tail = fma(tail, tail_x, 2.83297682f);
    return select(tail, central, base < 5.0f) * value;
}

[[nodiscard]] inline luisa::compute::Float3
sample_beckmann_visible_normal(luisa::compute::Float3 normal,
    const OrthonormalBasis &world_basis,
    luisa::compute::Float3 incoming,
    luisa::compute::Float alpha_x,
    luisa::compute::Float alpha_y,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;

    const auto local_incoming =
        make_float3(dot(world_basis.tangent, incoming),
            dot(world_basis.bitangent, incoming),
            dot(normal, incoming));
    const auto stretched_incoming =
        normalize(make_float3(alpha_x * local_incoming.x,
            alpha_y * local_incoming.y,
            local_incoming.z));

    const auto normal_radius = sqrt(-log(random.x));
    const auto normal_phi = 2.0f * pi * random.y;
    const auto normal_slope = normal_radius *
                              make_float2(cos(normal_phi), sin(normal_phi));

    const auto cosine = stretched_incoming.z;
    const auto sine = sqrt(max(1.0f - cosine * cosine, 0.0f));
    const auto safe_cosine = select(1.0f, cosine, cosine != 0.0f);
    const auto safe_sine = select(1.0f, sine, sine != 0.0f);
    const auto tangent = sine / safe_cosine;
    const auto safe_tangent = select(1.0f, tangent, tangent != 0.0f);
    const auto cotangent = 1.0f / safe_tangent;
    const auto erf_a = fast_erf(cotangent);
    const auto exponential = exp(-cotangent * cotangent);
    const auto cosine_phi = select(
        1.0f, stretched_incoming.x / safe_sine, sine != 0.0f);
    const auto sine_phi = select(
        0.0f, stretched_incoming.y / safe_sine, sine != 0.0f);
    const auto k = tangent * 0.56418958354f;
    const auto approximate_y =
        random.x * (1.0f + erf_a + k * (1.0f - erf_a * erf_a));
    const auto exact_y = random.x * (1.0f + erf_a + k * exponential);
    const auto safe_k = select(1.0f, k, k > 0.0f);
    const auto quadratic_b =
        (0.5f - sqrt(max(k * (k - approximate_y + 1.0f) + 0.25f,
                          0.0f))) /
        safe_k;
    const auto b = select(approximate_y - 1.0f, quadratic_b, k > 0.0f);

    Float inverse_erf = fast_inverse_erf(b);
    Float2 begin = make_float2(-1.0f, -exact_y);
    Float2 end =
        make_float2(erf_a, 1.0f + erf_a + k * exponential - exact_y);
    Float2 current = make_float2(
        b, 1.0f + b + k * exp(-inverse_erf * inverse_erf) - exact_y);
    UInt iteration = 0u;
    $while((abs(current.y) > 1.0e-6f) & (iteration < 3u)) {
        const auto same_sign = (begin.y < 0.0f) == (current.y < 0.0f);
        begin = make_float2(select(begin.x, current.x, same_sign),
                            select(begin.y, current.y, same_sign));
        end.x = select(current.x, end.x, same_sign);
        const auto newton =
            current.x - current.y / (1.0f - inverse_erf * tangent);
        const auto inside = (newton >= begin.x) & (newton <= end.x);
        current.x = select(0.5f * (begin.x + end.x), newton, inside);
        inverse_erf = fast_inverse_erf(current.x);
        current.y = 1.0f + current.x +
                    k * exp(-inverse_erf * inverse_erf) - exact_y;
        iteration += 1u;
    };

    const auto general_slope =
        make_float2(inverse_erf, fast_inverse_erf(2.0f * random.y - 1.0f));
    const auto slope = select(general_slope, normal_slope,
                              stretched_incoming.z >= 0.99999f);
    const auto rotated_slope =
        make_float2(cosine_phi * slope.x - sine_phi * slope.y,
                    sine_phi * slope.x + cosine_phi * slope.y) *
        make_float2(alpha_x, alpha_y);
    const auto local_half =
        normalize(make_float3(-rotated_slope.x, -rotated_slope.y, 1.0f));
    return world_basis.tangent * local_half.x +
           world_basis.bitangent * local_half.y + normal * local_half.z;
}

[[nodiscard]] inline luisa::compute::Float3
sample_beckmann_visible_normal(luisa::compute::Float3 normal,
    luisa::compute::Float3 incoming,
    luisa::compute::Float alpha,
    luisa::compute::Float2 random) noexcept {
    return sample_beckmann_visible_normal(
        normal,
        make_orthonormals(normal),
        incoming,
        alpha,
        alpha,
        random);
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

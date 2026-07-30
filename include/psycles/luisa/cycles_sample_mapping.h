#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_sample_mapping.h> through the Psycles::luisa target."
#endif

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_sample_mapping {

inline constexpr float pi = 3.14159265358979323846f;
inline constexpr float inverse_pi = 0.31830988618379067154f;

struct OrthonormalBasis {
    luisa::compute::Float3 tangent;
    luisa::compute::Float3 bitangent;
};

struct CosineHemisphereSample {
    luisa::compute::Float3 direction;
    luisa::compute::Float pdf;
};

// This is the Cycles square-to-disk measure-preserving map. The branch at
// |a| == |b| and the center case are part of the mapping definition: changing
// either changes the deterministic pairing between a two-dimensional RNG
// sample and the generated direction even though the resulting density stays
// uniform.
[[nodiscard]] inline luisa::compute::Float2
sample_uniform_disk(luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;
    const auto a = 2.0f * random.x - 1.0f;
    const auto b = 2.0f * random.y - 1.0f;
    const auto x_major = a * a > b * b;
    const auto safe_a = select(1.0f, a, a != 0.0f);
    const auto safe_b = select(1.0f, b, b != 0.0f);
    const auto radius = select(b, a, x_major);
    const auto angle = select(
        0.5f * pi - 0.25f * pi * (a / safe_b),
        0.25f * pi * (b / safe_a),
        x_major);
    const auto mapped =
        make_float2(cos(angle), sin(angle)) * radius;
    return select(
        mapped,
        make_float2(0.0f),
        (a == 0.0f) & (b == 0.0f));
}

// Cycles fixes the otherwise free rotation around a normal with this
// algebraic basis. Keeping that convention is required for bitwise-correlated
// sample dimensions: an arbitrary valid basis preserves the PDF but sends the
// same RNG pair to a different world-space direction.
[[nodiscard]] inline OrthonormalBasis
make_orthonormals(luisa::compute::Float3 normal) noexcept {
    using namespace luisa::compute;
    const auto general =
        make_float3(normal.z - normal.y,
                    normal.x - normal.z,
                    normal.y - normal.x);
    const auto equal_components =
        make_float3(normal.z - normal.y,
                    normal.x + normal.z,
                    -normal.y - normal.x);
    const auto use_general =
        (normal.x != normal.y) | (normal.x != normal.z);
    const auto tangent =
        normalize(select(equal_components, general, use_general));
    return {.tangent = tangent,
            .bitangent = cross(normal, tangent)};
}

// The disk map, basis orientation, and lack of a final renormalization are one
// deterministic sampling invariant. The input normal is expected to be unit
// length, matching the Cycles closure contract.
[[nodiscard]] inline CosineHemisphereSample
sample_cosine_hemisphere(
    luisa::compute::Float3 normal,
    luisa::compute::Float2 random) noexcept {
    using namespace luisa::compute;
    const auto disk = sample_uniform_disk(random);
    const auto cosine =
        sqrt(max(1.0f - dot(disk, disk), 0.0f));
    const auto basis = make_orthonormals(normal);
    return {
        .direction =
            disk.x * basis.tangent +
            disk.y * basis.bitangent +
            cosine * normal,
        .pdf = cosine * inverse_pi};
}

} // namespace psycles::luisa_backend::cycles_sample_mapping

#include <psycles/luisa/surface_closure_evaluation.h>
#include <psycles/luisa/surface_closure_physical_blocks.h>
#include <psycles/luisa/surface_closure_sampling.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using psycles::test_support::approximately_equal;
using psycles::test_support::compile_named_kernel;
using psycles::test_support::make_surface_point;
using psycles::test_support::ParameterShaderServices;

template<typename T>
concept HasAlbedo = requires(T value) { value.albedo; };

template<typename T>
concept HasReflectionAlbedo = requires(T value) {
    value.reflection_albedo;
};

template<typename T>
concept HasTransmissionAlbedo = requires(T value) {
    value.transmission_albedo;
};

template<typename T>
concept HasSpecularIorLevel = requires(T value) {
    value.specular_ior_level;
};

template<typename T>
concept HasGeneralMetallic = requires(T value) {
    value.payload.metallic;
};

template<typename T>
concept HasThinFilm = requires(T value) {
    value.payload.thin_film_thickness;
    value.payload.thin_film_ior;
};

template<typename T>
concept HasGeneralMicrofacetState = requires(T value) {
    value.payload.microfacet_tangent;
    value.payload.microfacet_alpha_x;
    value.payload.microfacet_alpha_y;
};

template<typename T>
concept HasDielectricFresnel = requires(T value) {
    value.payload.fresnel_f0;
};

template<typename T>
concept HasBssrdfRadius = requires(T value) {
    value.payload.radius;
};

template<typename T>
concept HasBssrdfIor = requires(T value) {
    value.payload.bssrdf_ior;
};

static_assert(surface_closure_physical_block_count == 2u);
static_assert(!HasAlbedo<SurfaceClosurePhysicalRecord>);
static_assert(!HasReflectionAlbedo<SurfaceClosurePhysicalRecord>);
static_assert(!HasTransmissionAlbedo<SurfaceClosurePhysicalRecord>);
static_assert(!HasSpecularIorLevel<SurfaceClosurePhysicalRecord>);
static_assert(!HasGeneralMetallic<SurfaceClosurePhysicalGeneralRecord>);
static_assert(HasThinFilm<SurfaceClosurePhysicalGeneralRecord>);
static_assert(HasGeneralMicrofacetState<SurfaceClosurePhysicalGeneralRecord>);
static_assert(!HasDielectricFresnel<SurfaceClosurePhysicalGeneralRecord>);
static_assert(!HasBssrdfRadius<SurfaceClosurePhysicalGeneralRecord>);
static_assert(!HasBssrdfIor<SurfaceClosurePhysicalGeneralRecord>);
static_assert(!HasGeneralMetallic<SurfaceClosurePhysicalDielectricRecord>);
static_assert(HasThinFilm<SurfaceClosurePhysicalDielectricRecord>);
static_assert(!HasGeneralMicrofacetState<SurfaceClosurePhysicalDielectricRecord>);
static_assert(HasDielectricFresnel<SurfaceClosurePhysicalDielectricRecord>);
static_assert(!HasBssrdfRadius<SurfaceClosurePhysicalDielectricRecord>);
static_assert(!HasBssrdfIor<SurfaceClosurePhysicalDielectricRecord>);
static_assert(!HasGeneralMetallic<SurfaceClosurePhysicalBssrdfRecord>);
static_assert(!HasThinFilm<SurfaceClosurePhysicalBssrdfRecord>);
static_assert(!HasGeneralMicrofacetState<SurfaceClosurePhysicalBssrdfRecord>);
static_assert(!HasDielectricFresnel<SurfaceClosurePhysicalBssrdfRecord>);
static_assert(HasBssrdfRadius<SurfaceClosurePhysicalBssrdfRecord>);
static_assert(HasBssrdfIor<SurfaceClosurePhysicalBssrdfRecord>);

struct ClosureCase {
    SurfaceClosureKind kind;
    SurfaceClosureLobe lobe{SurfaceClosureLobe::none};
};

constexpr std::array closure_cases{
    ClosureCase{SurfaceClosureKind::diffuse},
    ClosureCase{SurfaceClosureKind::translucent},
    ClosureCase{SurfaceClosureKind::principled, SurfaceClosureLobe::sheen},
    ClosureCase{SurfaceClosureKind::principled, SurfaceClosureLobe::metallic},
    ClosureCase{SurfaceClosureKind::principled, SurfaceClosureLobe::dielectric},
    ClosureCase{SurfaceClosureKind::glossy},
    ClosureCase{SurfaceClosureKind::glass},
    ClosureCase{SurfaceClosureKind::refraction},
    ClosureCase{SurfaceClosureKind::bssrdf},
    ClosureCase{SurfaceClosureKind::rough_translucent},
    ClosureCase{SurfaceClosureKind::thin_glass_transmission},
    ClosureCase{SurfaceClosureKind::transparent}};
constexpr auto case_count =
    static_cast<std::uint32_t>(closure_cases.size());
constexpr std::uint32_t logical_row_count = 14u;
namespace invariant_row {
constexpr std::uint32_t stable_representation = 0u;
constexpr std::uint32_t family_projection = 1u;
constexpr std::uint32_t evaluation = 2u;
constexpr std::uint32_t sample_components = 3u;
constexpr std::uint32_t sample_state = 4u;
constexpr std::uint32_t product_sample_numeric = 5u;
constexpr std::uint32_t tagged_sample_numeric = 6u;
constexpr std::uint32_t payload_reads = 7u;
constexpr std::uint32_t count = 8u;
}// namespace invariant_row
constexpr std::uint32_t invariant_row_count = invariant_row::count;
constexpr std::uint32_t rows_per_case =
    logical_row_count + invariant_row_count;

using PhysicalRoundTripCallable = Callable<luisa::float4x4(
    luisa::float4x4,
    luisa::float4x4,
    std::uint32_t)>;

[[nodiscard]] constexpr bool uses_glass_payload(
    SurfaceClosureKind kind) noexcept {
    return kind == SurfaceClosureKind::glass ||
           kind == SurfaceClosureKind::refraction;
}

[[nodiscard]] constexpr bool uses_bssrdf_payload(
    SurfaceClosureKind kind) noexcept {
    return kind == SurfaceClosureKind::bssrdf;
}

[[nodiscard]] constexpr bool uses_thin_film_payload(
    ClosureCase closure) noexcept {
    return closure.kind == SurfaceClosureKind::glass ||
           (closure.kind == SurfaceClosureKind::principled &&
            (closure.lobe == SurfaceClosureLobe::metallic ||
             closure.lobe == SurfaceClosureLobe::dielectric));
}

[[nodiscard]] luisa::float4 identity_row(
    std::uint32_t case_index) noexcept {
    const auto flags =
        ((case_index & 1u) != 0u ? 1u : 0u) |
        ((case_index & 2u) != 0u ? 2u : 0u) |
        ((case_index & 4u) != 0u ? 4u : 0u);
    const auto closure = closure_cases[case_index];
    const auto kind = static_cast<std::uint32_t>(closure.kind);
    const auto lobe = static_cast<std::uint32_t>(closure.lobe);
    return std::bit_cast<luisa::float4>(
        luisa::uint4{kind, lobe, flags, 17u + case_index});
}

[[nodiscard]] luisa::float4 expected_row(
    std::uint32_t case_index,
    std::uint32_t row) noexcept {
    const auto closure = closure_cases[case_index];
    const auto kind = closure.kind;
    const auto glass = uses_glass_payload(kind);
    const auto bssrdf = uses_bssrdf_payload(kind);
    const auto general = !glass && !bssrdf;
    const auto thin_film = uses_thin_film_payload(closure);
    const auto offset = 10.0f * static_cast<float>(case_index);
    switch (row) {
        case 0u:
            return identity_row(case_index);
        case 1u:
            return {1.1f + offset,
                    1.2f + offset,
                    1.3f + offset,
                    1.4f + offset};
        case 2u:
            return {glass ? 0.0f : 2.1f + offset,
                    glass ? 0.0f : 2.2f + offset,
                    glass ? 0.0f : 2.3f + offset,
                    1.5f + offset};
        case 3u:
            return {3.1f + offset,
                    3.2f + offset,
                    3.3f + offset,
                    3.4f + offset};
        case 4u:
            return general
                       ? luisa::float4{4.1f + offset,
                                      4.2f + offset,
                                      4.3f + offset,
                                      0.0f}
                       : luisa::float4{0.0f};
        case 5u:
            return general
                       ? luisa::float4{5.1f + offset,
                                      5.2f + offset,
                                      5.3f + offset,
                                      0.0f}
                       : glass
                             ? luisa::float4{5.1f + offset,
                                            5.2f + offset,
                                            5.3f + offset,
                                            0.0f}
                             : luisa::float4{1.0f, 1.0f, 1.0f, 0.0f};
        case 6u:
            return glass
                       ? luisa::float4{6.1f + offset,
                                      6.2f + offset,
                                      6.3f + offset,
                                      6.4f + offset}
                       : luisa::float4{0.0f,
                                      0.0f,
                                      0.0f,
                                      6.4f + offset};
        case 7u:
            return glass
                       ? luisa::float4{7.1f + offset,
                                      7.2f + offset,
                                      7.3f + offset,
                                      0.0f}
                       : luisa::float4{0.0f,
                                      0.0f,
                                      0.0f,
                                      general ? 7.4f + offset : 0.0f};
        case 8u:
            return glass
                       ? luisa::float4{8.1f + offset,
                                      8.2f + offset,
                                      8.3f + offset,
                                      0.0f}
                       : luisa::float4{0.0f,
                                      0.0f,
                                      0.0f,
                                      general ? 8.4f + offset : 0.0f};
        case 9u:
            return glass
                       ? luisa::float4{9.1f + offset,
                                      9.2f + offset,
                                      9.3f + offset,
                                      1.4f}
                       : luisa::float4{0.0f,
                                      0.0f,
                                      0.0f,
                                      bssrdf ? 9.4f + offset : 1.4f};
        case 10u:
            return bssrdf
                       ? luisa::float4{10.1f + offset,
                                      10.2f + offset,
                                      10.3f + offset,
                                      10.4f + offset}
                       : luisa::float4{0.0f};
        case 11u:
            return bssrdf
                       ? luisa::float4{11.1f + offset,
                                      11.2f + offset,
                                      11.3f + offset,
                                      11.4f + offset}
                       : luisa::float4{0.0f, 0.0f, 0.0f, 1.0f};
        case 12u:
            return general
                       ? luisa::float4{12.1f + offset,
                                      12.2f + offset,
                                      12.3f + offset,
                                      12.4f + offset}
                       : luisa::float4{0.0f};
        case 13u:
            return {general ? 13.1f + offset : 0.0f,
                    thin_film ? 14.4f + offset : 0.0f,
                    thin_film ? 14.5f + offset : 0.0f,
                    0.0f};
        default:
            return luisa::float4{1.0f};
    }
}

[[nodiscard]] bool same_identity_bits(
    luisa::float4 observed,
    luisa::float4 expected) noexcept {
    const auto observed_bits =
        std::bit_cast<luisa::uint4>(observed);
    const auto expected_bits =
        std::bit_cast<luisa::uint4>(expected);
    return all(observed_bits == expected_bits);
}

[[nodiscard]] Bool same_common_projection(
    const SurfaceClosurePhysicalRecord &lhs,
    const SurfaceClosurePhysicalCommonRecord &rhs,
    Float3 expected_color_or_evaluation_scale) noexcept {
    return (lhs.kind == rhs.kind) &
           (lhs.lobe == rhs.lobe) &
           all(lhs.weight == rhs.weight) &
           (lhs.allocation_weight == rhs.allocation_weight) &
           (lhs.sample_weight == rhs.sample_weight) &
           (lhs.setup_valid == rhs.setup_valid) &
           all(expected_color_or_evaluation_scale ==
               rhs.color_or_evaluation_scale) &
           all(lhs.normal == rhs.normal) &
           (lhs.roughness == rhs.roughness) &
           (lhs.preserve_ggx_energy == rhs.preserve_ggx_energy) &
           (lhs.beckmann == rhs.beckmann) &
           (lhs.bssrdf_method == rhs.bssrdf_method);
}

[[nodiscard]] Bool same_payload_projection(
    const SurfaceClosurePhysicalRecord &lhs,
    const SurfaceClosurePhysicalGeneralRecord &rhs) noexcept {
    return same_common_projection(
               lhs, rhs.common, lhs.color) &
           (lhs.thin_film_thickness ==
            rhs.payload.thin_film_thickness) &
           (lhs.thin_film_ior == rhs.payload.thin_film_ior) &
           (lhs.ior == rhs.payload.ior) &
           all(lhs.specular_tint == rhs.payload.specular_tint) &
           (lhs.sheen_transform_a == rhs.payload.sheen_transform_a) &
           (lhs.sheen_transform_b == rhs.payload.sheen_transform_b) &
           all(lhs.evaluation_scale == rhs.payload.evaluation_scale) &
           all(lhs.microfacet_tangent == rhs.payload.microfacet_tangent) &
           (lhs.microfacet_alpha_x == rhs.payload.microfacet_alpha_x) &
           (lhs.microfacet_alpha_y == rhs.payload.microfacet_alpha_y);
}

[[nodiscard]] Bool same_payload_projection(
    const SurfaceClosurePhysicalRecord &lhs,
    const SurfaceClosurePhysicalDielectricRecord &rhs) noexcept {
    return same_common_projection(
               lhs, rhs.common, lhs.evaluation_scale) &
           (lhs.ior == rhs.payload.ior) &
           (lhs.thin_film_thickness ==
            rhs.payload.thin_film_thickness) &
           (lhs.thin_film_ior == rhs.payload.thin_film_ior) &
           all(lhs.fresnel_f0 == rhs.payload.fresnel_f0) &
           all(lhs.fresnel_f90 == rhs.payload.fresnel_f90) &
           all(lhs.reflection_tint == rhs.payload.reflection_tint) &
           all(lhs.transmission_tint == rhs.payload.transmission_tint);
}

[[nodiscard]] Bool same_payload_projection(
    const SurfaceClosurePhysicalRecord &lhs,
    const SurfaceClosurePhysicalBssrdfRecord &rhs) noexcept {
    return same_common_projection(
               lhs, rhs.common, lhs.color) &
           all(lhs.bssrdf_radius == rhs.payload.radius) &
           all(lhs.bssrdf_albedo == rhs.payload.albedo) &
           (lhs.bssrdf_ior == rhs.payload.bssrdf_ior) &
           (lhs.bssrdf_roughness == rhs.payload.roughness) &
           (lhs.bssrdf_anisotropy == rhs.payload.anisotropy);
}

inline constexpr float fp_absolute_tolerance = 1.0e-5f;
inline constexpr float fp_relative_tolerance =
    8.0f * std::numeric_limits<float>::epsilon();

[[nodiscard]] Bool close(Float lhs, Float rhs) noexcept {
    const auto scale = max(abs(lhs), abs(rhs));
    return abs(lhs - rhs) <=
           fp_absolute_tolerance + fp_relative_tolerance * scale;
}

[[nodiscard]] Bool close(Float2 lhs, Float2 rhs) noexcept {
    const auto scale = max(abs(lhs), abs(rhs));
    return all(abs(lhs - rhs) <=
               make_float2(fp_absolute_tolerance) +
                   fp_relative_tolerance * scale);
}

[[nodiscard]] Bool close(Float3 lhs, Float3 rhs) noexcept {
    const auto scale = max(abs(lhs), abs(rhs));
    return all(abs(lhs - rhs) <=
               make_float3(fp_absolute_tolerance) +
                   fp_relative_tolerance * scale);
}

[[nodiscard]] Bool same_evaluation_contribution(
    const Var<SurfaceClosureEvaluationContributionCall> &lhs,
    const Var<SurfaceClosureEvaluationContributionCall> &rhs) noexcept {
    return close(lhs.f, rhs.f) & close(lhs.diffuse_f, rhs.diffuse_f) &
           close(lhs.glossy_f, rhs.glossy_f) &
           close(lhs.total_sample_weight, rhs.total_sample_weight) &
           close(lhs.weighted_pdf, rhs.weighted_pdf) &
           close(lhs.weighted_roughness_squared,
                 rhs.weighted_roughness_squared) &
           (lhs.events == rhs.events);
}

// The production shader option enables fast math. Two algebraically
// equivalent inlined projections may therefore receive different FMA/rsqrt
// schedules even when their decoded inputs compare bit-for-bit. Bound sample
// directions independently: 5e-4 per component is below 0.05 degrees for
// unit vectors, while all transported scalar state keeps the tighter bound.
[[nodiscard]] Bool close_sample_direction(
    Float3 lhs, Float3 rhs) noexcept {
    return all(abs(lhs - rhs) <= make_float3(5.0e-4f));
}

[[nodiscard]] Bool same_conditional_sample_transport_state(
    const Var<SurfaceClosureConditionalSampleCall> &lhs,
    const Var<SurfaceClosureConditionalSampleCall> &rhs) noexcept {
    return close(lhs.roughness, rhs.roughness) &
           close(lhs.singular_evaluation, rhs.singular_evaluation) &
           close(lhs.singular_pdf, rhs.singular_pdf) &
           close(lhs.eta, rhs.eta);
}

[[nodiscard]] Bool same_conditional_sample_identity_state(
    const Var<SurfaceClosureConditionalSampleCall> &lhs,
    const Var<SurfaceClosureConditionalSampleCall> &rhs) noexcept {
    return (lhs.properties == rhs.properties) &
           (lhs.valid == rhs.valid);
}

[[nodiscard]] Bool same_conditional_sample_bssrdf_state(
    const Var<SurfaceClosureConditionalSampleCall> &lhs,
    const Var<SurfaceClosureConditionalSampleCall> &rhs) noexcept {
    return (lhs.bssrdf_method == rhs.bssrdf_method) &
           close(lhs.bssrdf_radius, rhs.bssrdf_radius) &
           close(lhs.bssrdf_albedo, rhs.bssrdf_albedo) &
           close(lhs.bssrdf_normal, rhs.bssrdf_normal) &
           close(lhs.bssrdf_ior, rhs.bssrdf_ior) &
           close(lhs.bssrdf_roughness, rhs.bssrdf_roughness) &
           close(lhs.bssrdf_anisotropy, rhs.bssrdf_anisotropy);
}

[[nodiscard]] Bool same_conditional_sample_state(
    const Var<SurfaceClosureConditionalSampleCall> &lhs,
    const Var<SurfaceClosureConditionalSampleCall> &rhs) noexcept {
    return same_conditional_sample_transport_state(lhs, rhs) &
           same_conditional_sample_identity_state(lhs, rhs) &
           same_conditional_sample_bssrdf_state(lhs, rhs);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    PhysicalRoundTripCallable round_trip = [](
        Float4x4 block_0,
        Float4x4 block_1,
        UInt requested_block) noexcept {
        const auto closure = unpack_surface_closure_physical(
            Expr<luisa::float4x4>{block_0.expression()},
            Expr<luisa::float4x4>{block_1.expression()});
        const auto packed = pack_surface_closure_physical(closure);
        Float4x4 result{packed.block_0};
        $if (requested_block == 1u) {
            result = packed.block_1;
        };
        return result;
    };
    round_trip.set_name("surface_closure_physical_round_trip");

    Kernel1D test = [round_trip](
                        BufferFloat4 parameter_buffer,
                        BufferFloat4 output) noexcept {
        const auto case_index = dispatch_x();
        const auto offset = 10.0f * cast<float>(case_index);
        auto closure = SurfaceClosureRecord::zero();
        closure.kind = static_cast<std::uint32_t>(
            SurfaceClosureKind::diffuse);
        for (auto i = 1u; i < case_count; ++i) {
            closure.kind = select(
                closure.kind,
                static_cast<std::uint32_t>(closure_cases[i].kind),
                case_index == i);
            closure.lobe = select(
                closure.lobe,
                static_cast<std::uint32_t>(closure_cases[i].lobe),
                case_index == i);
        }
        closure.weight = make_float3(
            1.1f + offset, 1.2f + offset, 1.3f + offset);
        closure.allocation_weight = 1.4f + offset;
        closure.sample_weight = 1.5f + offset;
        closure.setup_valid = (case_index & 1u) != 0u;
        closure.color = make_float3(
            2.1f + offset, 2.2f + offset, 2.3f + offset);
        closure.normal = make_float3(
            3.1f + offset, 3.2f + offset, 3.3f + offset);
        closure.roughness = 3.4f + offset;
        closure.specular_tint = make_float3(
            4.1f + offset, 4.2f + offset, 4.3f + offset);
        closure.diffuse_roughness = 4.4f + offset;
        closure.evaluation_scale = make_float3(
            5.1f + offset, 5.2f + offset, 5.3f + offset);
        closure.metallic = 5.4f + offset;
        closure.fresnel_f0 = make_float3(
            6.1f + offset, 6.2f + offset, 6.3f + offset);
        closure.ior = 6.4f + offset;
        closure.fresnel_f90 = make_float3(
            7.1f + offset, 7.2f + offset, 7.3f + offset);
        closure.sheen_transform_a = 7.4f + offset;
        closure.reflection_tint = make_float3(
            8.1f + offset, 8.2f + offset, 8.3f + offset);
        closure.sheen_transform_b = 8.4f + offset;
        closure.transmission_tint = make_float3(
            9.1f + offset, 9.2f + offset, 9.3f + offset);
        closure.bssrdf_ior = 9.4f + offset;
        closure.bssrdf_radius = make_float3(
            10.1f + offset, 10.2f + offset, 10.3f + offset);
        closure.bssrdf_anisotropy = 10.4f + offset;
        closure.bssrdf_albedo = make_float3(
            11.1f + offset, 11.2f + offset, 11.3f + offset);
        closure.bssrdf_roughness = 11.4f + offset;
        closure.microfacet_tangent = make_float3(
            12.1f + offset, 12.2f + offset, 12.3f + offset);
        closure.microfacet_alpha_x = 12.4f + offset;
        closure.microfacet_alpha_y = 13.1f + offset;
        closure.thin_film_thickness = 14.4f + offset;
        closure.thin_film_ior = 14.5f + offset;
        closure.preserve_ggx_energy = (case_index & 2u) != 0u;
        closure.beckmann = (case_index & 4u) != 0u;
        closure.bssrdf_method = 17u + case_index;

        // Setup/AOV-only fields deliberately remain distinct poison values.
        closure.albedo = make_float3(101.0f + offset);
        closure.reflection_albedo = make_float3(211.0f + offset);
        closure.transmission_albedo = make_float3(307.0f + offset);
        closure.specular_ior_level = 401.0f + offset;

        const auto packed = pack_surface_closure_physical(closure);
        const auto round_trip_0 = round_trip(
            packed.block_0, packed.block_1, 0u);
        const auto round_trip_1 = round_trip(
            packed.block_0, packed.block_1, 1u);
        const auto canonical = unpack_surface_closure_physical(
            Expr<luisa::float4x4>{round_trip_0.expression()},
            Expr<luisa::float4x4>{round_trip_1.expression()});

        UInt flags = 0u;
        flags |= select(0u, 1u, canonical.setup_valid);
        flags |= select(0u, 2u, canonical.preserve_ggx_energy);
        flags |= select(0u, 4u, canonical.beckmann);
        std::array logical_rows{
            make_uint4(
                canonical.kind,
                canonical.lobe,
                flags,
                canonical.bssrdf_method)
                .bitcast<luisa::float4>(),
            make_float4(canonical.weight, canonical.allocation_weight),
            make_float4(canonical.color, canonical.sample_weight),
            make_float4(canonical.normal, canonical.roughness),
            make_float4(
                canonical.specular_tint,
                canonical.diffuse_roughness),
            make_float4(canonical.evaluation_scale, canonical.metallic),
            make_float4(canonical.fresnel_f0, canonical.ior),
            make_float4(canonical.fresnel_f90, canonical.sheen_transform_a),
            make_float4(
                canonical.reflection_tint,
                canonical.sheen_transform_b),
            make_float4(canonical.transmission_tint, canonical.bssrdf_ior),
            make_float4(
                canonical.bssrdf_radius,
                canonical.bssrdf_anisotropy),
            make_float4(
                canonical.bssrdf_albedo,
                canonical.bssrdf_roughness),
            make_float4(
                canonical.microfacet_tangent,
                canonical.microfacet_alpha_x),
            make_float4(canonical.microfacet_alpha_y,
                        canonical.thin_film_thickness,
                        canonical.thin_film_ior,
                        0.0f)};
        for (auto row = 0u; row < logical_row_count; ++row) {
            output->write(
                case_index * rows_per_case + row,
                logical_rows[row]);
        }

        // Formal round-trip property on the encoded quotient: irrelevant
        // fields may canonicalize, but the physical representation must be
        // idempotent bit-for-bit after one pack/unpack step.
        const auto repacked = pack_surface_closure_physical(canonical);
        Bool stable = true;
        for (auto row = 0u; row < 4u; ++row) {
            stable &= all(
                packed.block_0[row].bitcast<luisa::uint4>() ==
                repacked.block_0[row].bitcast<luisa::uint4>());
            stable &= all(
                packed.block_1[row].bitcast<luisa::uint4>() ==
                repacked.block_1[row].bitcast<luisa::uint4>());
        }
        output->write(
            case_index * rows_per_case + logical_row_count +
                invariant_row::stable_representation,
            make_float4(select(0.0f, 1.0f, stable)));

        // A family eliminator is correct precisely when it agrees with the
        // generic inverse after projection to fields observable for that
        // family. The comparison itself stays branch-local: merging the
        // distinct record types would recreate the forbidden product.
        const auto common = unpack_surface_closure_physical_common(
            Expr<luisa::float4x4>{round_trip_0.expression()});
        const auto glass =
            (canonical.kind == static_cast<std::uint32_t>(
                                   SurfaceClosureKind::glass)) |
            (canonical.kind == static_cast<std::uint32_t>(
                                   SurfaceClosureKind::refraction));
        const auto bssrdf =
            canonical.kind == static_cast<std::uint32_t>(
                                  SurfaceClosureKind::bssrdf);
        const auto general =
            (canonical.kind == static_cast<std::uint32_t>(
                                   SurfaceClosureKind::principled)) |
            (canonical.kind == static_cast<std::uint32_t>(
                                   SurfaceClosureKind::glossy)) |
            (canonical.kind == static_cast<std::uint32_t>(
                                   SurfaceClosureKind::thin_glass_transmission));
        Bool projection_equal = false;
        $if(glass) {
            const auto projected =
                unpack_surface_closure_physical_dielectric(
                common,
                Expr<luisa::float4x4>{round_trip_1.expression()});
            projection_equal =
                same_payload_projection(canonical, projected);
        }
        $elif(bssrdf) {
            const auto projected =
                unpack_surface_closure_physical_bssrdf(
                common,
                Expr<luisa::float4x4>{round_trip_1.expression()});
            projection_equal =
                same_payload_projection(canonical, projected);
        }
        $elif(general) {
            const auto projected =
                unpack_surface_closure_physical_general(
                common,
                Expr<luisa::float4x4>{round_trip_1.expression()});
            projection_equal =
                same_payload_projection(canonical, projected);
        }
        $else {
            const auto projected =
                unpack_surface_closure_physical_common_only(common);
            projection_equal = same_common_projection(
                canonical, projected.common, canonical.color);
        };
        output->write(
            case_index * rows_per_case +
                logical_row_count + invariant_row::family_projection,
            make_float4(select(
                0.0f, 1.0f, projection_equal)));

        // Differential semantic oracle for the consumer-directed
        // eliminators. Use a separate finite closure so every family executes
        // meaningful directional algebra while the ABI rows above retain
        // their poison values for projection testing.
        auto consumer_source = SurfaceClosureRecord::zero();
        consumer_source.kind = closure.kind;
        consumer_source.lobe = closure.lobe;
        consumer_source.weight = make_float3(0.37f, 0.29f, 0.41f);
        consumer_source.allocation_weight = 0.73f;
        consumer_source.sample_weight = 0.61f;
        consumer_source.setup_valid = true;
        consumer_source.color = make_float3(0.62f, 0.48f, 0.31f);
        consumer_source.normal = normalize(make_float3(0.13f, -0.09f, 0.98f));
        consumer_source.roughness = 0.36f;
        consumer_source.microfacet_tangent = normalize(
            make_float3(0.93f, 0.31f, 0.07f));
        consumer_source.microfacet_alpha_x = 0.08f;
        consumer_source.microfacet_alpha_y = 0.31f;
        consumer_source.diffuse_roughness = 0.27f;
        consumer_source.metallic = 0.22f;
        consumer_source.ior = 1.43f;
        consumer_source.specular_tint = make_float3(0.84f, 0.71f, 0.59f);
        consumer_source.sheen_transform_a = 0.83f;
        consumer_source.sheen_transform_b = 0.17f;
        consumer_source.evaluation_scale = make_float3(0.91f, 0.87f, 0.79f);
        consumer_source.fresnel_f0 = make_float3(0.04f, 0.05f, 0.06f);
        consumer_source.fresnel_f90 = make_float3(1.0f);
        consumer_source.reflection_tint = make_float3(0.92f, 0.81f, 0.73f);
        consumer_source.transmission_tint = make_float3(0.67f, 0.76f, 0.88f);
        consumer_source.preserve_ggx_energy = false;
        consumer_source.beckmann = false;
        consumer_source.bssrdf_method =
            static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk);
        consumer_source.bssrdf_radius = make_float3(0.8f, 0.5f, 0.3f);
        consumer_source.bssrdf_albedo = make_float3(0.7f, 0.4f, 0.2f);
        consumer_source.bssrdf_ior = 1.4f;
        consumer_source.bssrdf_roughness = 0.32f;
        consumer_source.bssrdf_anisotropy = 0.15f;
        const auto consumer_blocks =
            pack_surface_closure_physical(consumer_source);
        const auto consumer_common = unpack_surface_closure_physical_common(
            Expr<luisa::float4x4>{consumer_blocks.block_0.expression()});
        const auto consumer_product = unpack_surface_closure_physical(
            Expr<luisa::float4x4>{consumer_blocks.block_0.expression()},
            Expr<luisa::float4x4>{consumer_blocks.block_1.expression()});
        auto surface_point = make_surface_point();
        surface_point.incoming = normalize(make_float3(0.21f, -0.16f, 0.96f));
        const SurfaceClosurePoint point{surface_point};
        ParameterShaderServices services{parameter_buffer, 1.0f};
        const auto incoming = make_surface_closure_sampling_incoming(point);
        const auto outgoing = normalize(make_float3(-0.18f, 0.24f, 0.95f));
        const auto query = SurfaceQuery{
            .lobe_mask = static_cast<std::uint32_t>(
                event_diffuse | event_glossy |
                event_transmission | event_transparent),
            .transport_mode =
                static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f,
            .reflective_caustics = true,
            .refractive_caustics = true};
        const auto policy = SurfaceClosureEvaluationPolicy{
            .diffuse_included = true,
            .glossy_included = true,
            .glass_included = true,
            .transmission_included = true,
            .preserve_pdf = true};
        const auto product_evaluation =
            surface_closure_evaluation_contribution(
                services, point, point.shading_normal, consumer_product,
                incoming, outgoing, query, policy, false);
        const auto raw_tagged_evaluation =
            surface_closure_evaluation_contribution_from_physical_blocks(
                services, point, point.shading_normal,
                Expr<luisa::float4x4>{consumer_blocks.block_0.expression()},
                Expr<luisa::float4x4>{consumer_blocks.block_1.expression()},
                incoming, outgoing, query, policy, false);
        UInt payload_read_count = 0u;
        const auto load_payload = [&] {
            payload_read_count += 1u;
            return Float4x4{consumer_blocks.block_1};
        };
        const auto tagged_evaluation =
            surface_closure_evaluation_contribution_from_physical_common(
                services, point, point.shading_normal, consumer_common,
                load_payload, incoming, outgoing, query, policy, false);
        const auto random_direction = make_float2(0.37f, 0.64f);
        const auto product_sample = surface_closure_conditional_sample(
            services, point, point.shading_normal, consumer_product, incoming,
            consumer_product.normal, random_direction, 0.42f, query);
        const auto raw_tagged_sample =
            surface_closure_conditional_sample_from_physical_blocks(
                services, point, point.shading_normal,
                Expr<luisa::float4x4>{consumer_blocks.block_0.expression()},
                Expr<luisa::float4x4>{consumer_blocks.block_1.expression()},
                incoming, consumer_product.normal, random_direction,
                0.42f, query);
        const auto tagged_sample =
            surface_closure_conditional_sample_from_physical_common(
                services, point, point.shading_normal, consumer_common,
                load_payload, incoming, consumer_product.normal,
                random_direction, 0.42f, query);
        output->write(
            case_index * rows_per_case + logical_row_count +
                invariant_row::evaluation,
            make_float4(
                select(0.0f, 1.0f,
                    same_evaluation_contribution(
                        product_evaluation, tagged_evaluation)),
                select(0.0f, 1.0f,
                    same_evaluation_contribution(
                        raw_tagged_evaluation, tagged_evaluation)),
                select(0.0f, 1.0f,
                    close_sample_direction(
                        raw_tagged_sample.direction,
                        tagged_sample.direction)),
                select(0.0f, 1.0f,
                    same_conditional_sample_state(
                        raw_tagged_sample, tagged_sample))));
        output->write(
            case_index * rows_per_case + logical_row_count +
                invariant_row::sample_components,
            make_float4(
                select(0.0f, 1.0f,
                    close_sample_direction(
                        product_sample.direction,
                        tagged_sample.direction)),
                select(0.0f, 1.0f,
                    close(product_sample.roughness,
                          tagged_sample.roughness)),
                select(0.0f, 1.0f,
                    close(product_sample.singular_evaluation,
                          tagged_sample.singular_evaluation) &
                        close(product_sample.singular_pdf,
                              tagged_sample.singular_pdf)),
                select(0.0f, 1.0f,
                    close(product_sample.eta,
                          tagged_sample.eta))));

        output->write(
            case_index * rows_per_case + logical_row_count +
                invariant_row::sample_state,
            make_float4(
                select(0.0f, 1.0f,
                    same_conditional_sample_transport_state(
                        product_sample, tagged_sample)),
                select(0.0f, 1.0f,
                    same_conditional_sample_identity_state(
                        product_sample, tagged_sample)),
                select(0.0f, 1.0f,
                    same_conditional_sample_bssrdf_state(
                        product_sample, tagged_sample)),
                1.0f));

        output->write(
            case_index * rows_per_case + logical_row_count +
                invariant_row::product_sample_numeric,
            make_float4(product_sample.singular_evaluation,
                        product_sample.singular_pdf));
        output->write(
            case_index * rows_per_case + logical_row_count +
                invariant_row::tagged_sample_numeric,
            make_float4(tagged_sample.singular_evaluation,
                        tagged_sample.singular_pdf));

        const auto general_payload =
            (consumer_source.kind ==
             static_cast<std::uint32_t>(SurfaceClosureKind::principled)) |
            (consumer_source.kind ==
             static_cast<std::uint32_t>(SurfaceClosureKind::glossy)) |
            (consumer_source.kind ==
             static_cast<std::uint32_t>(
                 SurfaceClosureKind::thin_glass_transmission));
        const auto expected_payload_reads =
            select(0u, 2u, glass | general_payload) +
            select(0u, 1u, bssrdf);
        output->write(
            case_index * rows_per_case + logical_row_count +
                invariant_row::payload_reads,
            make_float4(select(
                0.0f, 1.0f,
                payload_read_count == expected_payload_reads)));
    };

    if (test.function()->function().custom_callables().size() != 1u) {
        std::cerr << "Physical closure callable reuse regression: expected "
                     "one definition, got "
                  << test.function()->function().custom_callables().size()
                  << '\n';
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output =
        device.create_buffer<luisa::float4>(case_count * rows_per_case);
    auto parameter_buffer = device.create_buffer<luisa::float4>(1u);
    auto shader = compile_named_kernel(
        device, "surface closure physical ABI", test);
    std::array<luisa::float4, case_count * rows_per_case> actual{};
    const std::array parameter_data{luisa::float4{1.0f}};
    stream << parameter_buffer.copy_from(luisa::span{parameter_data})
           << shader(parameter_buffer, output).dispatch(case_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (auto case_index = 0u; case_index < case_count; ++case_index) {
        for (auto row = 0u; row < rows_per_case; ++row) {
            if (row == logical_row_count +
                           invariant_row::product_sample_numeric ||
                row == logical_row_count +
                           invariant_row::tagged_sample_numeric) {
                continue;
            }
            const auto observed =
                actual[case_index * rows_per_case + row];
            const auto expected = expected_row(case_index, row);
            const auto matches = row == 0u
                                     ? same_identity_bits(observed, expected)
                                     : approximately_equal(
                                           observed, expected, 1.0e-6f);
            if (!matches) {
                std::cerr << "Physical closure ABI mismatch on " << backend
                          << std::setprecision(9)
                          << ", case " << case_index << ", row " << row
                          << ": got {" << observed.x << ", " << observed.y
                          << ", " << observed.z << ", " << observed.w
                          << "}, expected {" << expected.x << ", "
                          << expected.y << ", " << expected.z << ", "
                          << expected.w << "}";
                if (row == logical_row_count +
                               invariant_row::sample_components ||
                    row == logical_row_count +
                               invariant_row::sample_state) {
                    const auto product_numeric = actual[
                        case_index * rows_per_case + logical_row_count +
                        invariant_row::product_sample_numeric];
                    const auto tagged_numeric = actual[
                        case_index * rows_per_case + logical_row_count +
                        invariant_row::tagged_sample_numeric];
                    std::cerr << "; product singular={"
                              << product_numeric.x << ", "
                              << product_numeric.y << ", "
                              << product_numeric.z << "}, pdf="
                              << product_numeric.w
                              << "; tagged singular={"
                              << tagged_numeric.x << ", "
                              << tagged_numeric.y << ", "
                              << tagged_numeric.z << "}, pdf="
                              << tagged_numeric.w;
                }
                std::cerr << '\n';
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

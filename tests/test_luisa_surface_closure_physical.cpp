#include <psycles/luisa/surface_closure_physical_blocks.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using psycles::test_support::approximately_equal;
using psycles::test_support::compile_named_kernel;

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

static_assert(surface_closure_physical_block_count == 2u);
static_assert(!HasAlbedo<SurfaceClosurePhysicalRecord>);
static_assert(!HasReflectionAlbedo<SurfaceClosurePhysicalRecord>);
static_assert(!HasTransmissionAlbedo<SurfaceClosurePhysicalRecord>);
static_assert(!HasSpecularIorLevel<SurfaceClosurePhysicalRecord>);

constexpr std::array closure_kinds{
    SurfaceClosureKind::diffuse,
    SurfaceClosureKind::translucent,
    SurfaceClosureKind::principled,
    SurfaceClosureKind::glossy,
    SurfaceClosureKind::glass,
    SurfaceClosureKind::refraction,
    SurfaceClosureKind::bssrdf,
    SurfaceClosureKind::rough_translucent,
    SurfaceClosureKind::thin_glass_transmission,
    SurfaceClosureKind::transparent};
constexpr auto case_count =
    static_cast<std::uint32_t>(closure_kinds.size());
constexpr std::uint32_t logical_row_count = 12u;
constexpr std::uint32_t invariant_row_count = 2u;
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

[[nodiscard]] luisa::float4 identity_row(
    std::uint32_t case_index) noexcept {
    const auto flags =
        ((case_index & 1u) != 0u ? 1u : 0u) |
        ((case_index & 2u) != 0u ? 2u : 0u) |
        ((case_index & 4u) != 0u ? 4u : 0u);
    const auto kind = static_cast<std::uint32_t>(
        closure_kinds[case_index]);
    const auto lobe = case_index == 2u
                          ? static_cast<std::uint32_t>(
                                SurfaceClosureLobe::sheen)
                          : static_cast<std::uint32_t>(
                                SurfaceClosureLobe::none);
    return std::bit_cast<luisa::float4>(
        luisa::uint4{kind, lobe, flags, 17u + case_index});
}

[[nodiscard]] luisa::float4 expected_row(
    std::uint32_t case_index,
    std::uint32_t row) noexcept {
    const auto kind = closure_kinds[case_index];
    const auto glass = uses_glass_payload(kind);
    const auto bssrdf = uses_bssrdf_payload(kind);
    const auto general = !glass && !bssrdf;
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
            return {2.1f + offset,
                    2.2f + offset,
                    2.3f + offset,
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
                                      4.4f + offset}
                       : luisa::float4{0.0f};
        case 5u:
            return general
                       ? luisa::float4{5.1f + offset,
                                      5.2f + offset,
                                      5.3f + offset,
                                      5.4f + offset}
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
    const SurfaceClosurePhysicalRecord &rhs) noexcept {
    return (lhs.kind == rhs.kind) &
           (lhs.lobe == rhs.lobe) &
           all(lhs.weight == rhs.weight) &
           (lhs.allocation_weight == rhs.allocation_weight) &
           (lhs.sample_weight == rhs.sample_weight) &
           (lhs.setup_valid == rhs.setup_valid) &
           all(lhs.color == rhs.color) &
           all(lhs.normal == rhs.normal) &
           (lhs.roughness == rhs.roughness) &
           (lhs.preserve_ggx_energy == rhs.preserve_ggx_energy) &
           (lhs.beckmann == rhs.beckmann) &
           (lhs.bssrdf_method == rhs.bssrdf_method);
}

[[nodiscard]] Bool same_payload_projection(
    const SurfaceClosurePhysicalRecord &lhs,
    const SurfaceClosurePhysicalRecord &rhs) noexcept {
    return (lhs.diffuse_roughness == rhs.diffuse_roughness) &
           (lhs.metallic == rhs.metallic) &
           (lhs.ior == rhs.ior) &
           all(lhs.specular_tint == rhs.specular_tint) &
           (lhs.sheen_transform_a == rhs.sheen_transform_a) &
           (lhs.sheen_transform_b == rhs.sheen_transform_b) &
           all(lhs.evaluation_scale == rhs.evaluation_scale) &
           all(lhs.fresnel_f0 == rhs.fresnel_f0) &
           all(lhs.fresnel_f90 == rhs.fresnel_f90) &
           all(lhs.reflection_tint == rhs.reflection_tint) &
           all(lhs.transmission_tint == rhs.transmission_tint) &
           all(lhs.bssrdf_radius == rhs.bssrdf_radius) &
           all(lhs.bssrdf_albedo == rhs.bssrdf_albedo) &
           (lhs.bssrdf_ior == rhs.bssrdf_ior) &
           (lhs.bssrdf_roughness == rhs.bssrdf_roughness) &
           (lhs.bssrdf_anisotropy == rhs.bssrdf_anisotropy);
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

    Kernel1D test = [round_trip](BufferFloat4 output) noexcept {
        const auto case_index = dispatch_x();
        const auto offset = 10.0f * cast<float>(case_index);
        auto closure = SurfaceClosureRecord::zero();
        closure.kind = static_cast<std::uint32_t>(
            SurfaceClosureKind::diffuse);
        for (auto i = 1u; i < case_count; ++i) {
            closure.kind = select(
                closure.kind,
                static_cast<std::uint32_t>(closure_kinds[i]),
                case_index == i);
        }
        closure.lobe = select(
            static_cast<std::uint32_t>(SurfaceClosureLobe::none),
            static_cast<std::uint32_t>(SurfaceClosureLobe::sheen),
            case_index == 2u);
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
                canonical.bssrdf_roughness)};
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
            case_index * rows_per_case + logical_row_count,
            make_float4(select(0.0f, 1.0f, stable)));

        // A family eliminator is correct precisely when it agrees with the
        // generic inverse after projection to fields observable for that
        // family. General, dielectric, and BSSRDF decoders preserve their
        // complete canonical family records. Transparent observes only the
        // common record and therefore has no block_1 dependency at all.
        const auto common = unpack_surface_closure_physical_common(
            Expr<luisa::float4x4>{round_trip_0.expression()});
        auto projected =
            unpack_surface_closure_physical_common_only(common);
        const auto glass =
            (canonical.kind == static_cast<std::uint32_t>(
                                   SurfaceClosureKind::glass)) |
            (canonical.kind == static_cast<std::uint32_t>(
                                   SurfaceClosureKind::refraction));
        const auto bssrdf =
            canonical.kind == static_cast<std::uint32_t>(
                                  SurfaceClosureKind::bssrdf);
        const auto transparent =
            canonical.kind == static_cast<std::uint32_t>(
                                  SurfaceClosureKind::transparent);
        $if(glass) {
            projected = unpack_surface_closure_physical_dielectric(
                common,
                Expr<luisa::float4x4>{round_trip_1.expression()});
        }
        $elif(bssrdf) {
            projected = unpack_surface_closure_physical_bssrdf(
                common,
                Expr<luisa::float4x4>{round_trip_1.expression()});
        }
        $elif(!transparent) {
            projected = unpack_surface_closure_physical_general(
                common,
                Expr<luisa::float4x4>{round_trip_1.expression()});
        };
        auto projection_equal =
            same_common_projection(canonical, projected);
        projection_equal &= transparent |
                            same_payload_projection(
                                canonical, projected);
        output->write(
            case_index * rows_per_case +
                logical_row_count + 1u,
            make_float4(select(
                0.0f, 1.0f, projection_equal)));
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
    auto shader = compile_named_kernel(
        device, "surface closure physical ABI", test);
    std::array<luisa::float4, case_count * rows_per_case> actual{};
    stream << shader(output).dispatch(case_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (auto case_index = 0u; case_index < case_count; ++case_index) {
        for (auto row = 0u; row < rows_per_case; ++row) {
            const auto observed =
                actual[case_index * rows_per_case + row];
            const auto expected = expected_row(case_index, row);
            const auto matches = row == 0u
                                     ? same_identity_bits(observed, expected)
                                     : approximately_equal(
                                           observed, expected, 1.0e-6f);
            if (!matches) {
                std::cerr << "Physical closure ABI mismatch on " << backend
                          << ", case " << case_index << ", row " << row
                          << ": got {" << observed.x << ", " << observed.y
                          << ", " << observed.z << ", " << observed.w
                          << "}, expected {" << expected.x << ", "
                          << expected.y << ", " << expected.z << ", "
                          << expected.w << "}\n";
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

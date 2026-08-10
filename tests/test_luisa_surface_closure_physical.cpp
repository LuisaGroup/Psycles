#include <psycles/luisa/surface_closure_physical_blocks.h>

#include "luisa_surface_test_support.h"

#include <array>
#include <bit>
#include <cmath>
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

static_assert(surface_closure_physical_block_count == 3u);
static_assert(!HasAlbedo<SurfaceClosurePhysicalRecord>);
static_assert(!HasReflectionAlbedo<SurfaceClosurePhysicalRecord>);
static_assert(!HasTransmissionAlbedo<SurfaceClosurePhysicalRecord>);
static_assert(!HasSpecularIorLevel<SurfaceClosurePhysicalRecord>);

constexpr std::uint32_t case_count = 8u;
constexpr std::uint32_t rows_per_case =
    static_cast<std::uint32_t>(
        surface_closure_physical_block_count * 4u);

using PhysicalRoundTripCallable = Callable<luisa::float4x4(
    luisa::float4x4,
    luisa::float4x4,
    luisa::float4x4,
    std::uint32_t)>;

[[nodiscard]] luisa::float4 expected_row(
    std::uint32_t case_index,
    std::uint32_t row) noexcept {
    const auto offset = 10.0f * static_cast<float>(case_index);
    switch (row) {
        case 0u: {
            const auto flags =
                ((case_index & 1u) != 0u ? 1u : 0u) |
                ((case_index & 2u) != 0u ? 2u : 0u) |
                ((case_index & 4u) != 0u ? 4u : 0u);
            return {
                std::bit_cast<float>(11u + case_index),
                std::bit_cast<float>(13u + case_index),
                std::bit_cast<float>(flags),
                std::bit_cast<float>(17u + case_index)};
        }
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
            return {4.1f + offset,
                    4.2f + offset,
                    4.3f + offset,
                    4.4f + offset};
        case 5u:
            return {5.1f + offset,
                    5.2f + offset,
                    5.3f + offset,
                    5.4f + offset};
        case 6u:
            return {6.1f + offset,
                    6.2f + offset,
                    6.3f + offset,
                    6.4f + offset};
        case 7u:
            return {7.1f + offset,
                    7.2f + offset,
                    7.3f + offset,
                    7.4f + offset};
        case 8u:
            return {8.1f + offset,
                    8.2f + offset,
                    8.3f + offset,
                    8.4f + offset};
        case 9u:
            return {9.1f + offset,
                    9.2f + offset,
                    9.3f + offset,
                    9.4f + offset};
        case 10u:
            return {10.1f + offset,
                    10.2f + offset,
                    10.3f + offset,
                    10.4f + offset};
        default:
            return {11.1f + offset,
                    11.2f + offset,
                    11.3f + offset,
                    11.4f + offset};
    }
}

[[nodiscard]] bool same_identity_bits(
    luisa::float4 observed,
    luisa::float4 expected) noexcept {
    const auto observed_bits =
        std::bit_cast<luisa::uint4>(observed);
    const auto expected_bits =
        std::bit_cast<luisa::uint4>(expected);
    return observed_bits.x == expected_bits.x &&
           observed_bits.y == expected_bits.y &&
           observed_bits.z == expected_bits.z &&
           observed_bits.w == expected_bits.w;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    PhysicalRoundTripCallable round_trip = [](
        Float4x4 block_0,
        Float4x4 block_1,
        Float4x4 block_2,
        UInt requested_block) noexcept {
        const auto closure = unpack_surface_closure_physical(
            Expr<luisa::float4x4>{block_0.expression()},
            Expr<luisa::float4x4>{block_1.expression()},
            Expr<luisa::float4x4>{block_2.expression()});
        const auto packed = pack_surface_closure_physical(closure);
        Float4x4 result{packed.block_0};
        $if (requested_block == 1u) {
            result = packed.block_1;
        };
        $if (requested_block == 2u) {
            result = packed.block_2;
        };
        return result;
    };
    round_trip.set_name("surface_closure_physical_round_trip");

    Kernel1D test = [round_trip](BufferFloat4 output) noexcept {
        const auto case_index = dispatch_x();
        const auto offset = 10.0f * cast<float>(case_index);
        auto closure = SurfaceClosureRecord::zero();
        closure.kind = 11u + case_index;
        closure.lobe = 13u + case_index;
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

        // These deliberately distinct sentinels live on the setup/AOV side of
        // the dependency cut and therefore have no lane in the callable ABI.
        closure.albedo = make_float3(101.0f + offset);
        closure.reflection_albedo = make_float3(211.0f + offset);
        closure.transmission_albedo = make_float3(307.0f + offset);
        closure.specular_ior_level = 401.0f + offset;

        const auto packed = pack_surface_closure_physical(closure);
        for (auto block = 0u;
             block < surface_closure_physical_block_count;
             ++block) {
            const auto matrix = round_trip(
                packed.block_0,
                packed.block_1,
                packed.block_2,
                static_cast<std::uint32_t>(block));
            for (auto row = 0u; row < 4u; ++row) {
                output->write(
                    case_index * rows_per_case +
                        static_cast<std::uint32_t>(block * 4u + row),
                    matrix[row]);
            }
        }
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

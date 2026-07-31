#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/heterogeneous_volume_tracking.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;

inline constexpr std::size_t record_count = 16u;

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 4.0e-5f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

[[nodiscard]] bool approximately_equal(
    luisa::float4 actual,
    luisa::float4 expected) noexcept {
    return approximately_equal(actual.x, expected.x) &&
           approximately_equal(actual.y, expected.y) &&
           approximately_equal(actual.z, expected.z) &&
           approximately_equal(actual.w, expected.w);
}

}// namespace

int main(int argc, char **argv) {
    try {
        const auto backend =
            std::string_view{
                argc > 1 ? argv[1] : "fallback"};
        Context context{argv[0]};
        auto device = context.create_device(backend);
        auto stream = device.create_stream();
        auto output =
            device.create_buffer<luisa::float4>(
                record_count);
        auto hash_output =
            device.create_buffer<luisa::uint4>(1u);

        Kernel1D evaluate =
            [](BufferFloat4 records,
               BufferUInt4 hashes) noexcept {
                HeterogeneousVolumeTracking tracking;
                const auto flag =
                    [](Bool value) noexcept {
                        return select(
                            0.0f, 1.0f, value);
                    };

                const VolumeCoefficients scattering{
                    .sigma_t =
                        make_float3(
                            0.5f, 1.0f, 2.0f),
                    .sigma_s =
                        make_float3(
                            0.25f, 0.2f, 1.5f),
                    .emission =
                        make_float3(
                            0.1f, 0.2f, 0.3f),
                    .has_extinction = true,
                    .has_scatter = true,
                    .has_emission = true};
                const auto collision =
                    tracking.evaluate_collision(
                        scattering,
                        2.5f,
                        0.3f,
                        make_float3(
                            0.8f, 1.2f, 0.5f),
                        0.7f);
                records.write(
                    0u,
                    make_float4(
                        collision
                            .null_event.sigma_n,
                        collision
                            .null_event.majorant));
                records.write(
                    1u,
                    make_float4(
                        collision
                            .normalized_throughput,
                        flag(collision
                                 .null_event
                                 .majorant_exceeded)));
                records.write(
                    2u,
                    make_float4(
                        collision.channel_pdf,
                        collision
                            .scatter_probability));
                records.write(
                    3u,
                    make_float4(
                        collision
                            .scatter_throughput,
                        collision.null_probability));
                records.write(
                    4u,
                    make_float4(
                        collision.null_throughput,
                        collision
                            .null_transmittance));
                records.write(
                    5u,
                    make_float4(
                        collision.emission,
                        flag(collision
                                 .absorption_only)));

                const VolumeCoefficients absorption{
                    .sigma_t =
                        make_float3(
                            0.25f, 0.0f, 1.0f),
                    .sigma_s =
                        make_float3(0.0f),
                    .emission =
                        make_float3(
                            0.4f, 0.2f, 0.1f),
                    .has_extinction = true,
                    .has_scatter = false,
                    .has_emission = true};
                const auto absorption_collision =
                    tracking.evaluate_collision(
                        absorption,
                        1.2f,
                        0.4f,
                        make_float3(
                            2.0f, 1.0f, 0.5f),
                        0.35f);
                records.write(
                    6u,
                    make_float4(
                        absorption_collision
                            .null_throughput,
                        absorption_collision
                            .null_transmittance));
                records.write(
                    7u,
                    make_float4(
                        absorption_collision
                            .emission,
                        flag(
                            absorption_collision
                                .absorption_only)));
                records.write(
                    8u,
                    make_float4(
                        absorption_collision
                            .channel_pdf,
                        absorption_collision
                            .scatter_probability));

                const VolumeCoefficients violated{
                    .sigma_t =
                        make_float3(
                            0.4f, 1.4f, 0.2f),
                    .sigma_s =
                        make_float3(
                            0.1f, 0.2f, 0.05f),
                    .emission =
                        make_float3(0.0f),
                    .has_extinction = true,
                    .has_scatter = true,
                    .has_emission = false};
                const auto violated_collision =
                    tracking.evaluate_collision(
                        violated,
                        0.8f,
                        0.25f,
                        make_float3(
                            0.75f, 0.5f, 0.25f),
                        1.0f);
                records.write(
                    9u,
                    make_float4(
                        violated_collision
                            .normalized_throughput,
                        flag(
                            violated_collision
                                .null_event
                                .majorant_exceeded)));
                records.write(
                    10u,
                    make_float4(
                        violated_collision
                            .null_event.sigma_n,
                        violated_collision
                            .null_event.majorant));

                const auto scatter_selection =
                    tracking.select_event(
                        0.2f, 0.35f);
                const auto null_selection =
                    tracking.select_event(
                        0.8f, 0.35f);
                records.write(
                    11u,
                    make_float4(
                        scatter_selection.random,
                        flag(scatter_selection
                                 .scattered),
                        null_selection.random,
                        flag(null_selection
                                 .scattered)));
                records.write(
                    12u,
                    make_float4(
                        tracking.candidate_distance(
                            0.37f, 1.7f),
                        tracking.candidate_distance(
                            0.37f, 0.0f),
                        0.0f,
                        0.0f));

                records.write(
                    13u,
                    make_float4(0.0f));
                records.write(
                    14u,
                    make_float4(0.0f));
                hashes.write(
                    0u,
                    make_uint4(
                        cycles_sampler::
                            hash_hp_uint(0u),
                        cycles_sampler::
                            hash_hp_uint(
                                0x12345678u),
                        cycles_sampler::
                            hash_hp_seeded_uint(
                                16u,
                                0xe35fad82u),
                        cycles_sampler::
                            scramble_path_offset(
                                16u,
                                0xe35fad82u)));
                records.write(
                    15u,
                    make_float4(
                        flag(collision.active),
                        flag(
                            absorption_collision
                                .active),
                        flag(
                            violated_collision
                                .active),
                        0.0f));
            };
        auto shader = device.compile(
            evaluate,
            ShaderOption{
                .enable_cache = false,
                .enable_fast_math = false});
        std::array<luisa::float4, record_count>
            actual{};
        std::array<luisa::uint4, 1u>
            actual_hashes{};
        stream
            << shader(
                   output,
                   hash_output)
                   .dispatch(1u)
            << output.copy_to(luisa::span{actual})
            << hash_output.copy_to(
                   luisa::span{actual_hashes})
            << synchronize();

        // Pinned to the scalar-RGB volume kernel in Blender/Cycles
        // b82c3f0. These are collision-transition fixtures, not a second
        // renderer or a host reference integrator.
        const std::array<luisa::float4, record_count>
            expected{{
                {2.0f, 1.5f, 0.5f, 2.5f},
                {0.32f, 0.48f, 0.2f, 0.0f},
                {0.39408866f,
                 0.2364532f,
                 0.36945814f,
                 0.34869924f},
                {0.22942407f,
                 0.27530888f,
                 0.86034024f,
                 0.6513007f},
                {0.98264897f,
                 1.1054801f,
                 0.1535389f,
                 0.45591053f},
                {0.032f, 0.096f, 0.06f, 0.0f},
                {1.5833333f,
                 1.0f,
                 0.08333333f,
                 0.35f},
                {0.6666667f,
                 0.16666667f,
                 0.04166667f,
                 1.0f},
                {0.0f,
                 0.0f,
                 0.0f,
                 0.0f},
                {0.80691373f,
                 0.53794247f,
                 0.26897123f,
                 1.0f},
                {1.0f, 0.0f, 1.2f, 1.4f},
                {0.5714286f,
                 1.0f,
                 0.6923077f,
                 0.0f},
                {0.27178556f, 0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f, 0.0f},
                {0.0f, 0.0f, 0.0f, 0.0f},
                {1.0f, 1.0f, 1.0f, 0.0f},
            }};
        for (std::size_t i = 0u;
             i < actual.size();
             ++i) {
            if (!approximately_equal(
                    actual[i], expected[i])) {
                std::cerr
                    << "actual=("
                    << actual[i].x << ", "
                    << actual[i].y << ", "
                    << actual[i].z << ", "
                    << actual[i].w << "), expected=("
                    << expected[i].x << ", "
                    << expected[i].y << ", "
                    << expected[i].z << ", "
                    << expected[i].w << ")\n";
                throw std::runtime_error{
                    "Cycles heterogeneous collision fixture " +
                    std::to_string(i) +
                    " changed on backend " +
                    std::string{backend}};
            }
        }
        constexpr luisa::uint4 expected_hashes{
            0xe6fe3bebu,
            0x10188d56u,
            0x15982c29u,
            0x15982c28u};
        const auto hash_matches =
            actual_hashes[0].x == expected_hashes.x &&
            actual_hashes[0].y == expected_hashes.y &&
            actual_hashes[0].z == expected_hashes.z &&
            actual_hashes[0].w == expected_hashes.w;
        if (!hash_matches) {
            throw std::runtime_error{
                "Cycles Hash Prospector fixture changed on backend " +
                std::string{backend}};
        }

        std::cout
            << "All current-Cycles heterogeneous collision fixtures "
               "passed on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Heterogeneous volume fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

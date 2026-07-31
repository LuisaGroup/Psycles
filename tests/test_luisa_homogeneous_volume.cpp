#include <psycles/luisa/homogeneous_volume_transport.h>
#include <psycles/luisa/volume_scatter_probability.h>
#include <psycles/luisa/volume_direct_sampling.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;

inline constexpr std::size_t record_count = 46u;

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 3.0e-5f) noexcept {
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
    const auto backend =
        std::string_view{
            argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output =
        device.create_buffer<luisa::float4>(
            record_count);

    Kernel1D evaluate =
        [](BufferFloat4 records) noexcept {
            HomogeneousVolumeTransport transport;
            const auto flag =
                [](Bool value) noexcept {
                    return select(
                        0.0f, 1.0f, value);
                };
            const auto channel_as_float =
                [](UInt value) noexcept {
                    return cast<float>(value);
                };

            records.write(
                0u,
                make_float4(
                    transport.transmittance(
                        make_float3(
                            0.0f, 0.5f, 2.0f),
                        0.75f),
                    0.0f));
            const VolumeCoefficients emission{
                .sigma_t =
                    make_float3(
                        0.0f, 0.5f, 2.0f),
                .sigma_s = make_float3(0.0f),
                .emission =
                    make_float3(
                        1.0f, 2.0f, 3.0f),
                .has_extinction = true,
                .has_scatter = false,
                .has_emission = true};
            records.write(
                1u,
                make_float4(
                    transport.emission_integral(
                        emission, 0.75f),
                    0.0f));
            const VolumeCoefficients taylor{
                .sigma_t =
                    make_float3(
                        1.0e-7f,
                        2.0e-6f,
                        1.0e-5f),
                .sigma_s = make_float3(0.0f),
                .emission =
                    make_float3(
                        1.0f, 2.0f, 3.0f),
                .has_extinction = true,
                .has_scatter = false,
                .has_emission = true};
            records.write(
                2u,
                make_float4(
                    transport.emission_integral(
                        taylor, 0.5f),
                    0.0f));
            auto no_extinction = emission;
            no_extinction.has_extinction = false;
            records.write(
                3u,
                make_float4(
                    transport.emission_integral(
                        no_extinction, 0.75f),
                    0.0f));

            const auto standard_pdf =
                transport.channel_pdf(
                    make_float3(
                        0.2f, 0.5f, 0.9f),
                    make_float3(
                        2.0f, 0.5f, 0.1f));
            records.write(
                4u,
                make_float4(
                    standard_pdf,
                    standard_pdf.x +
                        standard_pdf.y +
                        standard_pdf.z));
            const auto uniform_pdf =
                transport.channel_pdf(
                    make_float3(0.0f),
                    make_float3(1.0f));
            records.write(
                5u,
                make_float4(
                    uniform_pdf,
                    uniform_pdf.x +
                        uniform_pdf.y +
                        uniform_pdf.z));
            const auto channel_0 =
                transport.sample_channel(
                    make_float3(
                        0.2f, 0.5f, 0.9f),
                    make_float3(
                        2.0f, 0.5f, 0.1f),
                    0.1f);
            records.write(
                6u,
                make_float4(
                    channel_as_float(
                        channel_0.channel),
                    channel_0.random,
                    flag(channel_0.matched),
                    channel_0.pdf.x));
            const auto channel_1 =
                transport.sample_channel(
                    make_float3(
                        0.2f, 0.5f, 0.9f),
                    make_float3(
                        2.0f, 0.5f, 0.1f),
                    0.7f);
            records.write(
                7u,
                make_float4(
                    channel_as_float(
                        channel_1.channel),
                    channel_1.random,
                    flag(channel_1.matched),
                    channel_1.pdf.x));
            const auto unmatched =
                transport.sample_channel(
                    make_float3(0.0f),
                    make_float3(1.0f),
                    1.0f);
            records.write(
                8u,
                make_float4(
                    channel_as_float(
                        unmatched.channel),
                    unmatched.random,
                    flag(unmatched.matched),
                    unmatched.pdf.z));

            const auto exponential =
                transport.bounded_exponential_sample(
                    0.37f,
                    1.7f,
                    0.2f,
                    2.3f);
            records.write(
                9u,
                make_float4(
                    exponential,
                    0.2f,
                    2.3f,
                    1.7f));
            records.write(
                10u,
                make_float4(
                    transport.bounded_exponential_pdf(
                        exponential,
                        make_float3(
                            0.2f, 1.7f, 4.0f),
                        0.2f,
                        2.3f),
                    0.0f));

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
            const auto path_throughput =
                make_float3(
                    0.8f, 1.2f, 0.5f);
            const auto scattered =
                transport.sample(
                    scattering,
                    1.3f,
                    path_throughput,
                    0.2f,
                    0.15f,
                    false);
            records.write(
                11u,
                make_float4(
                    scattered.transmittance,
                    flag(scattered.scattered)));
            records.write(
                12u,
                make_float4(
                    scattered.emission,
                    flag(scattered.active)));
            records.write(
                13u,
                make_float4(
                    scattered.scatter_probability,
                    channel_as_float(
                        scattered.channel)));
            records.write(
                14u,
                make_float4(
                    scattered.channel_pdf,
                    scattered.event_pdf));
            records.write(
                15u,
                make_float4(
                    scattered.throughput,
                    scattered.distance));
            records.write(
                16u,
                make_float4(
                    scattered.scatter_random,
                    flag(scattered.scattered),
                    scattered.event_pdf,
                    channel_as_float(
                        scattered.channel)));

            const auto transmitted =
                transport.sample(
                    scattering,
                    1.3f,
                    path_throughput,
                    0.95f,
                    0.82f,
                    false);
            records.write(
                17u,
                make_float4(
                    transmitted.scatter_probability,
                    channel_as_float(
                        transmitted.channel)));
            records.write(
                18u,
                make_float4(
                    transmitted.throughput,
                    transmitted.distance));
            records.write(
                19u,
                make_float4(
                    transmitted.event_pdf,
                    transmitted.scatter_random,
                    flag(transmitted.scattered),
                    channel_as_float(
                        transmitted.channel)));

            const auto terminated =
                transport.sample(
                    scattering,
                    1.3f,
                    path_throughput,
                    0.2f,
                    0.15f,
                    true);
            records.write(
                20u,
                make_float4(
                    terminated.throughput,
                    flag(terminated.scattered)));
            records.write(
                21u,
                make_float4(
                    terminated.scatter_probability,
                    terminated.event_pdf));

            const VolumeCoefficients absorption{
                .sigma_t =
                    make_float3(
                        0.25f, 0.0f, 1.0f),
                .sigma_s = make_float3(0.0f),
                .emission =
                    make_float3(
                        0.4f, 0.2f, 0.1f),
                .has_extinction = true,
                .has_scatter = false,
                .has_emission = true};
            const auto absorbed =
                transport.sample(
                    absorption,
                    2.0f,
                    make_float3(
                        2.0f, 1.0f, 0.5f),
                    0.3f,
                    0.6f,
                    false);
            records.write(
                22u,
                make_float4(
                    absorbed.transmittance,
                    flag(absorbed.scattered)));
            records.write(
                23u,
                make_float4(
                    absorbed.emission,
                    flag(absorbed.active)));
            records.write(
                24u,
                make_float4(
                    absorbed.throughput,
                    absorbed.distance));

            const auto guided =
                transport.sample_with_probability(
                    scattering,
                    1.3f,
                    path_throughput,
                    0.3f,
                    0.65f,
                    make_float3(
                        0.05f, 0.9f, 0.4f),
                    false);
            records.write(
                25u,
                make_float4(
                    guided.scatter_probability,
                    channel_as_float(
                        guided.channel)));
            records.write(
                26u,
                make_float4(
                    guided.channel_pdf,
                    guided.event_pdf));
            records.write(
                27u,
                make_float4(
                    guided.throughput,
                    guided.distance));
            records.write(
                28u,
                make_float4(
                    guided.scatter_random,
                    flag(guided.scattered),
                    guided.event_pdf,
                    channel_as_float(
                        guided.channel)));

            const auto zero_length =
                transport.sample(
                    scattering,
                    0.0f,
                    path_throughput,
                    0.2f,
                    0.15f,
                    false);
            records.write(
                29u,
                make_float4(
                    zero_length.throughput,
                    flag(zero_length.active)));
            records.write(
                30u,
                make_float4(
                    zero_length.distance,
                    flag(zero_length.scattered),
                    zero_length.event_pdf,
                    zero_length.scatter_random));

            HomogeneousVolumeScatterProbability
                probability;
            records.write(
                31u,
                make_float4(
                    probability.evaluate(
                        scattering,
                        1.3f,
                        false,
                        {.scattered_radiance =
                             make_float3(0.0f),
                         .transmitted_radiance =
                             make_float3(0.0f),
                         .majorant_optical_depth =
                             std::numeric_limits<
                                 float>::max(),
                         .enabled = true}),
                    0.0f));
            records.write(
                32u,
                make_float4(
                    probability.evaluate(
                        scattering,
                        1.3f,
                        false,
                        {.scattered_radiance =
                             make_float3(
                                 0.2f,
                                 0.6f,
                                 0.3f),
                         .transmitted_radiance =
                             make_float3(
                                 0.8f,
                                 0.4f,
                                 0.7f),
                         .majorant_optical_depth =
                             2.6f,
                         .enabled = true}),
                    0.0f));
            records.write(
                33u,
                make_float4(
                    probability.evaluate(
                        scattering,
                        1.3f,
                        false,
                        {.scattered_radiance =
                             make_float3(0.0f),
                         .transmitted_radiance =
                             make_float3(0.0f),
                         .majorant_optical_depth =
                             0.0f,
                         .enabled = false}),
                    0.0f));
            records.write(
                34u,
                make_float4(
                    probability.evaluate(
                        scattering,
                        1.3f,
                        true,
                        {.scattered_radiance =
                             make_float3(0.0f),
                         .transmitted_radiance =
                             make_float3(0.0f),
                         .majorant_optical_depth =
                             0.0f,
                         .enabled = true}),
                    0.0f));

            VolumeDirectSampling
                direct_sampling;
            const auto write_sampling_state =
                [&](std::uint32_t index,
                    const VolumeDirectSamplingState
                        &sampling) noexcept {
                    records.write(
                        index,
                        make_float4(
                            cast<float>(
                                sampling.method),
                            sampling.random,
                            flag(
                                sampling.use_mis),
                            flag(
                                sampling.enabled)));
                };
            write_sampling_state(
                35u,
                direct_sampling.prepare(
                    volume_sample_distance,
                    0.37f,
                    true));
            write_sampling_state(
                36u,
                direct_sampling.prepare(
                    volume_sample_equiangular,
                    0.37f,
                    true));
            write_sampling_state(
                37u,
                direct_sampling.prepare(
                    volume_sample_mis,
                    0.2f,
                    true));
            write_sampling_state(
                38u,
                direct_sampling.prepare(
                    volume_sample_mis,
                    0.8f,
                    true));
            write_sampling_state(
                39u,
                direct_sampling.prepare(
                    volume_sample_mis,
                    0.8f,
                    false));

            const VolumeEquiangularCoefficients
                symmetric{
                    .light_position =
                        make_float3(
                            1.0f, 0.0f, 0.0f),
                    .interval =
                        {.minimum = -1.0f,
                         .maximum = 1.0f}};
            const auto symmetric_middle =
                direct_sampling
                    .sample_equiangular(
                        make_float3(0.0f),
                        make_float3(
                            0.0f, 0.0f, 1.0f),
                        symmetric,
                        0.5f);
            records.write(
                40u,
                make_float4(
                    symmetric_middle.distance,
                    symmetric_middle.pdf,
                    flag(
                        symmetric_middle.valid),
                    0.0f));
            const auto symmetric_minimum =
                direct_sampling
                    .sample_equiangular(
                        make_float3(0.0f),
                        make_float3(
                            0.0f, 0.0f, 1.0f),
                        symmetric,
                        0.0f);
            records.write(
                41u,
                make_float4(
                    symmetric_minimum.distance,
                    symmetric_minimum.pdf,
                    flag(
                        symmetric_minimum.valid),
                    0.0f));
            records.write(
                42u,
                make_float4(
                    direct_sampling
                        .equiangular_pdf(
                            make_float3(0.0f),
                            make_float3(
                                0.0f,
                                0.0f,
                                1.0f),
                            symmetric,
                            0.5f),
                    0.0f,
                    0.0f,
                    0.0f));
            const auto parallel =
                direct_sampling
                    .sample_equiangular(
                        make_float3(0.0f),
                        make_float3(
                            0.0f, 0.0f, 1.0f),
                        {.light_position =
                             make_float3(
                                 0.0f,
                                 0.0f,
                                 2.0f),
                         .interval =
                             {.minimum = 0.0f,
                              .maximum = 3.0f}},
                        0.4f);
            records.write(
                43u,
                make_float4(
                    parallel.distance,
                    parallel.pdf,
                    flag(parallel.valid),
                    0.0f));
            records.write(
                44u,
                make_float4(
                    direct_sampling
                        .power_heuristic(
                            3.0f, 4.0f),
                    0.0f,
                    0.0f,
                    0.0f));
            const auto empty_interval =
                direct_sampling
                    .sample_equiangular(
                        make_float3(0.0f),
                        make_float3(
                            0.0f, 0.0f, 1.0f),
                        {.light_position =
                             make_float3(
                                 1.0f,
                                 0.0f,
                                 0.0f),
                         .interval =
                             {.minimum = 2.0f,
                              .maximum = 2.0f}},
                        0.5f);
            records.write(
                45u,
                make_float4(
                    direct_sampling
                        .power_heuristic(
                            0.0f, 0.0f),
                    empty_interval.distance,
                    empty_interval.pdf,
                    flag(
                        empty_interval.valid)));
        };

    auto shader = device.compile(evaluate);
    std::array<luisa::float4, record_count>
        actual{};
    stream
        << shader(output).dispatch(1u)
        << output.copy_to(luisa::span{actual})
        << synchronize();

    // Generated by compiling the official Cycles main b82c3f0 kernel
    // volume.h, mapping.h, and the homogeneous estimator from
    // shade_volume.h. This pins the estimator's probability measure, not a
    // separately maintained CPU reference implementation.
    constexpr std::array expected{
        luisa::float4{1.0f, 0.687289298f, 0.223130167f, 0.0f},
        luisa::float4{0.75f, 1.25084281f, 1.16530478f, 0.0f},
        luisa::float4{0.5f, 0.999999523f, 1.49999619f, 0.0f},
        luisa::float4{0.75f, 1.5f, 2.25f, 0.0f},
        luisa::float4{
            0.540540576f, 0.337837875f, 0.121621624f, 1.00000012f},
        luisa::float4{
            0.333333343f, 0.333333343f, 0.333333343f, 1.0f},
        luisa::float4{0.0f, 0.184999987f, 1.0f, 0.540540576f},
        luisa::float4{1.0f, 0.471999794f, 1.0f, 0.540540576f},
        luisa::float4{2.0f, 1.0f, 0.0f, 0.333333343f},
        luisa::float4{0.462138057f, 0.2f, 2.3f, 1.7f},
        luisa::float4{0.55338341f, 1.12025177f, 1.40209436f, 0.0f},
        luisa::float4{0.522045791f, 0.272531807f, 0.0742735863f, 1.0f},
        luisa::float4{0.0764726773f, 0.174592376f, 0.069429487f, 1.0f},
        luisa::float4{0.477954209f, 0.727468193f, 0.925726414f, 0.0f},
        luisa::float4{0.432967097f, 0.407142311f, 0.159890622f, 0.5647403f},
        luisa::float4{0.283316076f, 0.271983445f, 0.543966889f, 0.446287066f},
        luisa::float4{0.418450147f, 1.0f, 0.5647403f, 0.0f},
        luisa::float4{0.477954209f, 0.727468193f, 0.925726414f, 1.0f},
        luisa::float4{1.19713473f, 0.937438667f, 0.106450774f, 1.3f},
        luisa::float4{0.348863542f, 0.816535115f, 0.0f, 1.0f},
        luisa::float4{0.417636633f, 0.327038169f, 0.0371367931f, 0.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 1.0f},
        luisa::float4{0.606530666f, 1.0f, 0.135335281f, 0.0f},
        luisa::float4{1.25910187f, 0.400000006f, 0.0432332382f, 1.0f},
        luisa::float4{1.21306145f, 1.00000012f, 0.0676676482f, 2.0f},
        luisa::float4{0.05f, 0.9f, 0.4f, 1.0f},
        luisa::float4{0.432967097f, 0.407142311f, 0.159890622f, 0.480559349f},
        luisa::float4{0.362224072f, 0.378314435f, 0.895553827f, 0.277717739f},
        luisa::float4{0.333333343f, 1.0f, 0.480559349f, 1.0f},
        luisa::float4{0.8f, 1.2f, 0.5f, 1.0f},
        luisa::float4{0.0f, 0.0f, 1.0f, 0.2f},
        luisa::float4{
            0.494488537f, 0.556867063f, 0.606431603f, 0.0f},
        luisa::float4{
            0.269488543f, 0.631867051f, 0.456431597f, 0.0f},
        luisa::float4{
            0.477954209f, 0.727468193f, 0.925726414f, 0.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{1.0f, 0.37f, 0.0f, 1.0f},
        luisa::float4{2.0f, 0.37f, 0.0f, 1.0f},
        luisa::float4{1.0f, 0.4f, 1.0f, 1.0f},
        luisa::float4{2.0f, 0.6f, 1.0f, 1.0f},
        luisa::float4{0.0f, 0.6f, 0.0f, 0.0f},
        luisa::float4{
            0.0f, 0.636619747f, 1.0f, 0.0f},
        luisa::float4{
            -1.0f, 0.318309873f, 1.0f, 0.0f},
        luisa::float4{
            0.509295821f, 0.0f, 0.0f, 0.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{0.36f, 0.0f, 0.0f, 0.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 0.0f}};
    for (auto index = std::size_t{0u};
         index < expected.size();
         ++index) {
        if (!approximately_equal(
                actual[index],
                expected[index])) {
            std::cerr
                << "Cycles homogeneous volume fixture failed on "
                << backend << " at record " << index
                << ": got {" << actual[index].x
                << ", " << actual[index].y
                << ", " << actual[index].z
                << ", " << actual[index].w
                << "}, expected {"
                << expected[index].x << ", "
                << expected[index].y << ", "
                << expected[index].z << ", "
                << expected[index].w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

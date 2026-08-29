#include "../src/luisa/path_kernel_background_portal.h"

#include <psycles/luisa/background_sampling.h>
#include <psycles/sampling/background_distribution.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace background_sampling = psycles::luisa_backend::background_sampling;
using psycles::luisa_backend::detail::BackgroundPortalSampling;
using psycles::luisa_backend::detail::light_flag_full_spread;
using psycles::luisa_backend::detail::LightGpu;

[[nodiscard]] bool
near(float actual, float expected, float tolerance) noexcept {
    return std::abs(actual - expected) <= tolerance;
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    constexpr std::uint32_t width = 4u;
    constexpr std::uint32_t height = 2u;
    constexpr std::array radiance{psycles::Vec3f{1.0f, 1.0f, 1.0f},
                                  psycles::Vec3f{2.0f, 2.0f, 2.0f},
                                  psycles::Vec3f{4.0f, 4.0f, 4.0f},
                                  psycles::Vec3f{8.0f, 8.0f, 8.0f},
                                  psycles::Vec3f{8.0f, 8.0f, 8.0f},
                                  psycles::Vec3f{4.0f, 4.0f, 4.0f},
                                  psycles::Vec3f{2.0f, 2.0f, 2.0f},
                                  psycles::Vec3f{1.0f, 1.0f, 1.0f}};
    const auto host =
        psycles::sampling::build_cycles_background_map_distribution(
            radiance, width, height);
    std::vector<luisa::float2> conditional;
    conditional.reserve(host.conditional.size());
    for (const auto entry : host.conditional) {
        conditional.emplace_back(entry.function, entry.cumulative);
    }
    std::vector<luisa::float2> marginal;
    marginal.reserve(host.marginal.size());
    for (const auto entry : host.marginal) {
        marginal.emplace_back(entry.function, entry.cumulative);
    }

    constexpr std::array random{luisa::float2{0.1f, 0.2f},
                                luisa::float2{0.5f, 0.5f},
                                luisa::float2{0.9f, 0.8f},
                                luisa::float2{0.79f, 0.4f},
                                luisa::float2{0.81f, 0.6f}};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto conditional_buffer =
        device.create_buffer<luisa::float2>(conditional.size());
    auto marginal_buffer = device.create_buffer<luisa::float2>(marginal.size());
    auto random_buffer = device.create_buffer<luisa::float2>(random.size());
    auto direction_pdf_buffer =
        device.create_buffer<luisa::float4>(random.size());
    auto round_trip_buffer = device.create_buffer<luisa::float4>(random.size());
    auto pole_uv_buffer = device.create_buffer<luisa::float2>(2u);
    auto cycles_sun_oracle_buffer = device.create_buffer<luisa::float4>(1u);
    constexpr std::array portal_lights{
        LightGpu{.position = luisa::float3{0.0f, 0.0f, 0.0f},
                 .axis_x = luisa::float3{1.0f, 0.0f, 0.0f},
                 .axis_y = luisa::float3{0.0f, 1.0f, 0.0f},
                 .axis_z = luisa::float3{0.0f, 0.0f, -1.0f},
                 .size_u = 2.0f,
                 .size_v = 2.0f,
                 .flags = light_flag_full_spread},
        LightGpu{.position = luisa::float3{0.0f, 0.0f, 0.0f},
                 .axis_x = luisa::float3{1.0f, 0.0f, 0.0f},
                 .axis_y = luisa::float3{0.0f, 1.0f, 0.0f},
                 .axis_z = luisa::float3{0.0f, 0.0f, 1.0f},
                 .size_u = 2.0f,
                 .size_v = 2.0f,
                 .flags = light_flag_full_spread}};
    auto portal_light_buffer =
        device.create_buffer<LightGpu>(portal_lights.size());
    auto portal_result_buffer = device.create_buffer<luisa::float4>(3u);

    constexpr auto sun_radius = 0.01f;
    Kernel1D evaluate = [&portal_light_buffer](BufferFloat2 conditional_cdf,
                           BufferFloat2 marginal_cdf,
                           BufferFloat2 randoms,
                           BufferFloat4 direction_pdf,
                           BufferFloat4 round_trip,
                           BufferFloat2 pole_uv,
                           BufferFloat4 cycles_sun_oracle,
                           BufferFloat4 portal_results) noexcept {
        const auto index = dispatch_x();
        const auto sample = background_sampling::sample(conditional_cdf,
                                                        marginal_cdf,
                                                        width,
                                                        height,
                                                        1.0f,
                                                        4.0f,
                                                        make_float3(0.0f,
                                                                    0.0f,
                                                                    1.0f),
                                                        sun_radius,
                                                        randoms.read(index));
        const auto forward_pdf = background_sampling::pdf(conditional_cdf,
                                                          marginal_cdf,
                                                          width,
                                                          height,
                                                          1.0f,
                                                          4.0f,
                                                          make_float3(0.0f,
                                                                      0.0f,
                                                                      1.0f),
                                                          sun_radius,
                                                          sample.direction);
        const auto uv =
            background_sampling::direction_to_equirectangular(sample.direction);
        const auto reconstructed =
            background_sampling::equirectangular_to_direction(uv.x, uv.y);
        direction_pdf.write(index, make_float4(sample.direction, sample.pdf));
        round_trip.write(index, make_float4(reconstructed, forward_pdf));
        $if (index == 0u) {
            pole_uv.write(
                0u,
                background_sampling::direction_to_equirectangular(
                    make_float3(0.0f, 0.0f, 1.0f)));
            pole_uv.write(
                1u,
                background_sampling::direction_to_equirectangular(
                    make_float3(0.0f, 0.0f, -1.0f)));

            // Lone Monk, film (491, 221), sample 0, first NEE event. These
            // inputs and the expected direction below come from the latest
            // Cycles HIP path oracle. Together they lock mixture remapping,
            // concentric-disk sampling, cone-radius remapping, and Cycles'
            // orthonormal-frame rotation as one deterministic relation.
            constexpr auto lone_monk_radius =
                0.008726646192371845f;
            const auto lone_monk_axis = make_float3(
                0.3009074926376343f,
                -0.5211871266365051f,
                0.7986355423927307f);
            const auto oracle_sample =
                background_sampling::sample(
                    conditional_cdf,
                    marginal_cdf,
                    width,
                    height,
                    1.0f,
                    4.0f,
                    lone_monk_axis,
                    lone_monk_radius,
                    make_float2(
                        0.6973861455917358f,
                        0.4494679570198059f));
            cycles_sun_oracle.write(
                0u,
                make_float4(
                    oracle_sample.direction,
                    background_sampling::sun_pdf(
                        lone_monk_axis,
                        lone_monk_radius,
                        oracle_sample.direction)));

            const BackgroundPortalSampling portals;
            const auto reference = make_float3(0.0f, 0.0f, 1.0f);
            const auto portal = portals.sample(
                portal_light_buffer,
                0u,
                2u,
                reference,
                make_float2(0.25f, 0.75f));
            const auto portal_forward_pdf = portals.pdf(
                portal_light_buffer,
                0u,
                2u,
                reference,
                portal.direction);
            portal_results.write(
                0u, make_float4(portal.direction, portal.pdf));
            portal_results.write(
                1u,
                make_float4(
                    portal_forward_pdf,
                    cast<float>(portals.count_possible(
                        portal_light_buffer, 0u, 2u, reference)),
                    cast<float>(portals.count_possible(
                        portal_light_buffer,
                        0u,
                        2u,
                        make_float3(0.0f, 0.0f, -1.0f))),
                    cast<float>(portal.valid)));
            portal_results.write(
                2u,
                make_float4(portals.pdf(
                    portal_light_buffer,
                    0u,
                    2u,
                    reference,
                    make_float3(0.0f, 0.0f, 1.0f))));
        };
    };
    auto shader = device.compile(evaluate);
    std::array<luisa::float4, random.size()> direction_pdf{};
    std::array<luisa::float4, random.size()> round_trip{};
    std::array<luisa::float2, 2u> pole_uv{};
    std::array<luisa::float4, 1u> cycles_sun_oracle{};
    std::array<luisa::float4, 3u> portal_results{};
    stream << conditional_buffer.copy_from(luisa::span{conditional})
           << marginal_buffer.copy_from(luisa::span{marginal})
           << random_buffer.copy_from(luisa::span{random})
           << portal_light_buffer.copy_from(luisa::span{portal_lights})
           << shader(conditional_buffer,
                     marginal_buffer,
                     random_buffer,
                     direction_pdf_buffer,
                     round_trip_buffer,
                     pole_uv_buffer,
                     cycles_sun_oracle_buffer,
                     portal_result_buffer)
                  .dispatch(static_cast<std::uint32_t>(random.size()))
           << direction_pdf_buffer.copy_to(luisa::span{direction_pdf})
           << round_trip_buffer.copy_to(luisa::span{round_trip})
           << pole_uv_buffer.copy_to(luisa::span{pole_uv})
           << cycles_sun_oracle_buffer.copy_to(
                  luisa::span{cycles_sun_oracle})
           << portal_result_buffer.copy_to(luisa::span{portal_results})
           << synchronize();

    if (!near(pole_uv[0u].x, 0.5f, 1.0e-7f) ||
        !near(pole_uv[0u].y, 1.0f, 1.0e-7f) ||
        !near(pole_uv[1u].x, 0.5f, 1.0e-7f) ||
        !near(pole_uv[1u].y, 0.0f, 1.0e-7f)) {
        std::cerr << "canonical pole azimuth failed on " << backend
                  << ": north {" << pole_uv[0u].x << ", "
                  << pole_uv[0u].y << "}, south {" << pole_uv[1u].x
                  << ", " << pole_uv[1u].y << "}\n";
        return EXIT_FAILURE;
    }

    constexpr auto square_solid_angle_pdf = 0.477464829275686f;
    const auto portal_direction_length =
        std::sqrt(portal_results[0u].x * portal_results[0u].x +
                  portal_results[0u].y * portal_results[0u].y +
                  portal_results[0u].z * portal_results[0u].z);
    if (!near(portal_direction_length, 1.0f, 2.0e-5f) ||
        !near(portal_results[0u].w, square_solid_angle_pdf, 2.0e-5f) ||
        !near(portal_results[1u].x, portal_results[0u].w, 2.0e-5f) ||
        portal_results[1u].y != 1.0f ||
        portal_results[1u].z != 1.0f ||
        portal_results[1u].w != 1.0f ||
        portal_results[2u].x != 0.0f) {
        std::cerr << "Cycles portal proposal failed on " << backend
                  << ": sample {" << portal_results[0u].x << ", "
                  << portal_results[0u].y << ", " << portal_results[0u].z
                  << ", " << portal_results[0u].w << "}, pdf/counts {"
                  << portal_results[1u].x << ", " << portal_results[1u].y
                  << ", " << portal_results[1u].z << ", "
                  << portal_results[1u].w << "}, outside "
                  << portal_results[2u].x << '\n';
        return EXIT_FAILURE;
    }

    constexpr auto expected_lone_monk_direction = luisa::float3{
        0.30576658248901367f,
        -0.5236937999725342f,
        0.7951425313949585f};
    constexpr auto expected_lone_monk_sun_pdf = 4179.798828125f;
    const auto &oracle = cycles_sun_oracle.front();
    const auto oracle_error = std::max(
        {std::abs(oracle.x - expected_lone_monk_direction.x),
         std::abs(oracle.y - expected_lone_monk_direction.y),
         std::abs(oracle.z - expected_lone_monk_direction.z)});
    if (oracle_error > 3.0e-6f ||
        !near(oracle.w, expected_lone_monk_sun_pdf, 0.05f)) {
        std::cerr << "Cycles background-Sun oracle failed on " << backend
                  << ": direction/pdf {" << oracle.x << ", " << oracle.y
                  << ", " << oracle.z << ", " << oracle.w << "}\n";
        return EXIT_FAILURE;
    }

    for (std::size_t i = 0u; i < random.size(); ++i) {
        const auto direction_length =
            std::sqrt(direction_pdf[i].x * direction_pdf[i].x +
                      direction_pdf[i].y * direction_pdf[i].y +
                      direction_pdf[i].z * direction_pdf[i].z);
        const auto reconstruction_error =
            std::max({std::abs(direction_pdf[i].x - round_trip[i].x),
                      std::abs(direction_pdf[i].y - round_trip[i].y),
                      std::abs(direction_pdf[i].z - round_trip[i].z)});
        if (!near(direction_length, 1.0f, 2.0e-5f) ||
            reconstruction_error > 2.0e-4f || !(direction_pdf[i].w > 0.0f) ||
            !near(direction_pdf[i].w,
                  round_trip[i].w,
                  3.0e-4f * std::max(direction_pdf[i].w, 1.0f))) {
            std::cerr << "background sampling invariant failed on " << backend
                      << " at sample " << i << ": direction/pdf {"
                      << direction_pdf[i].x << ", " << direction_pdf[i].y
                      << ", " << direction_pdf[i].z << ", "
                      << direction_pdf[i].w << "}, reconstructed/pdf {"
                      << round_trip[i].x << ", " << round_trip[i].y << ", "
                      << round_trip[i].z << ", " << round_trip[i].w << "}\n";
            return EXIT_FAILURE;
        }
    }

    // With 1:4 method weights, x < 0.8 selects the guided sun and x > 0.8
    // selects the map. Check the branch contract directly.
    if (direction_pdf[3u].z < std::cos(sun_radius) ||
        direction_pdf[4u].z > std::cos(sun_radius)) {
        std::cerr << "background mixture method split failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

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

    constexpr auto sun_axis = luisa::float3{0.0f, 0.0f, 1.0f};
    constexpr auto sun_radius = 0.01f;
    Kernel1D evaluate = [](BufferFloat2 conditional_cdf,
                           BufferFloat2 marginal_cdf,
                           BufferFloat2 randoms,
                           BufferFloat4 direction_pdf,
                           BufferFloat4 round_trip) noexcept {
        const auto index = dispatch_x();
        const auto sample = background_sampling::sample(conditional_cdf,
                                                        marginal_cdf,
                                                        width,
                                                        height,
                                                        1.0f,
                                                        4.0f,
                                                        make_float3(sun_axis),
                                                        sun_radius,
                                                        randoms.read(index));
        const auto forward_pdf = background_sampling::pdf(conditional_cdf,
                                                          marginal_cdf,
                                                          width,
                                                          height,
                                                          1.0f,
                                                          4.0f,
                                                          make_float3(sun_axis),
                                                          sun_radius,
                                                          sample.direction);
        const auto uv =
            background_sampling::direction_to_equirectangular(sample.direction);
        const auto reconstructed =
            background_sampling::equirectangular_to_direction(uv.x, uv.y);
        direction_pdf.write(index, make_float4(sample.direction, sample.pdf));
        round_trip.write(index, make_float4(reconstructed, forward_pdf));
    };
    auto shader = device.compile(evaluate);
    std::array<luisa::float4, random.size()> direction_pdf{};
    std::array<luisa::float4, random.size()> round_trip{};
    stream << conditional_buffer.copy_from(luisa::span{conditional})
           << marginal_buffer.copy_from(luisa::span{marginal})
           << random_buffer.copy_from(luisa::span{random})
           << shader(conditional_buffer,
                     marginal_buffer,
                     random_buffer,
                     direction_pdf_buffer,
                     round_trip_buffer)
                  .dispatch(static_cast<std::uint32_t>(random.size()))
           << direction_pdf_buffer.copy_to(luisa::span{direction_pdf})
           << round_trip_buffer.copy_to(luisa::span{round_trip})
           << synchronize();

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

#include <psycles/luisa/cycles_sampler.h>
#include <psycles/sampling/tabulated_sobol.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace cycles_sampler =
    psycles::luisa_backend::cycles_sampler;
namespace tabulated_sobol =
    psycles::sampling::tabulated_sobol;

constexpr auto probe_count = std::size_t{4u};
constexpr auto metadata_count = std::size_t{11u};

void expect(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] std::array<std::uint32_t, 4u> lanes(
    luisa::uint4 value) noexcept {
    return {value.x, value.y, value.z, value.w};
}

}// namespace

int main(int argc, char **argv) {
    try {
        const auto generated = tabulated_sobol::generate_table(256u);
        std::vector<luisa::float4> table;
        table.reserve(generated.size());
        for (const auto sample : generated) {
            table.emplace_back(luisa::make_float4(
                sample.x, sample.y, sample.z, sample.w));
        }

        Context context{
            argc > 0 && argv != nullptr ? argv[0] : ""};
        auto device = context.create_device("fallback");
        auto stream = device.create_stream();
        auto table_buffer =
            device.create_buffer<luisa::float4>(table.size());
        auto sample_bits =
            device.create_buffer<luisa::uint4>(probe_count);
        auto metadata =
            device.create_buffer<std::uint32_t>(metadata_count);

        Kernel1D kernel = [](
                              BufferFloat4 samples,
                              UInt sequence_size,
                              BufferUInt4 output_bits,
                              BufferUInt output_metadata,
                              UInt pixel_x,
                              UInt pixel_y,
                              UInt seed,
                              UInt camera_sample,
                              UInt path_sample,
                              UInt first_path_step,
                              UInt next_path_step) noexcept {
            const auto rng_hash =
                cycles_sampler::pixel_hash(pixel_x, pixel_y, seed);
            const auto camera_dimension =
                UInt{tabulated_sobol::camera_filter_dimension};
            const auto first_terminate_dimension =
                cycles_sampler::path_dimension(
                    first_path_step,
                    tabulated_sobol::terminate_dimension);
            const auto first_light_dimension =
                cycles_sampler::path_dimension(
                    first_path_step,
                    tabulated_sobol::light_dimension);
            const auto first_bsdf_dimension =
                cycles_sampler::path_dimension(
                    first_path_step,
                    tabulated_sobol::surface_bsdf_dimension);
            const auto next_light_dimension =
                cycles_sampler::path_dimension(
                    next_path_step,
                    tabulated_sobol::light_dimension);

            output_bits.write(
                0u,
                as<luisa::uint4>(cycles_sampler::sample_4d(
                    samples,
                    sequence_size,
                    camera_sample,
                    rng_hash,
                    camera_dimension)));
            output_bits.write(
                1u,
                as<luisa::uint4>(cycles_sampler::sample_4d(
                    samples,
                    sequence_size,
                    path_sample,
                    rng_hash,
                    first_light_dimension)));
            output_bits.write(
                2u,
                as<luisa::uint4>(cycles_sampler::sample_4d(
                    samples,
                    sequence_size,
                    path_sample,
                    rng_hash,
                    first_bsdf_dimension)));
            output_bits.write(
                3u,
                as<luisa::uint4>(cycles_sampler::sample_4d(
                    samples,
                    sequence_size,
                    path_sample,
                    rng_hash,
                    next_light_dimension)));

            output_metadata.write(0u, rng_hash);
            output_metadata.write(1u, camera_dimension);
            output_metadata.write(
                2u, first_terminate_dimension);
            output_metadata.write(3u, first_light_dimension);
            output_metadata.write(4u, first_bsdf_dimension);
            output_metadata.write(5u, next_light_dimension);
            output_metadata.write(
                6u,
                next_light_dimension - first_light_dimension);
            output_metadata.write(
                7u,
                cycles_sampler::shuffled_sample_index(
                    camera_sample,
                    camera_dimension,
                    rng_hash,
                    sequence_size));
            output_metadata.write(
                8u,
                cycles_sampler::shuffled_sample_index(
                    path_sample,
                    first_light_dimension,
                    rng_hash,
                    sequence_size));
            output_metadata.write(
                9u,
                cycles_sampler::shuffled_sample_index(
                    path_sample,
                    first_bsdf_dimension,
                    rng_hash,
                    sequence_size));
            output_metadata.write(
                10u,
                cycles_sampler::shuffled_sample_index(
                    path_sample,
                    next_light_dimension,
                    rng_hash,
                    sequence_size));
        };
        auto shader = device.compile(
            kernel,
            ShaderOption{
                .enable_cache = false,
                .enable_fast_math = false});

        constexpr auto sentinel = std::uint32_t{0xdeadbeefu};
        std::array<luisa::uint4, probe_count> actual_bits;
        actual_bits.fill(luisa::uint4{sentinel});
        std::array<std::uint32_t, metadata_count>
            actual_metadata;
        actual_metadata.fill(sentinel);
        stream << table_buffer.copy_from(luisa::span{table})
               << sample_bits.copy_from(luisa::span{actual_bits})
               << metadata.copy_from(luisa::span{actual_metadata})
               << shader(
                      table_buffer,
                      256u,
                      sample_bits,
                      metadata,
                      17u,
                      29u,
                      0u,
                      0u,
                      63u,
                      0u,
                      1u)
                      .dispatch(1u)
               << sample_bits.copy_to(luisa::span{actual_bits})
               << metadata.copy_to(luisa::span{actual_metadata})
               << synchronize();

        constexpr std::array expected_bits{
            std::array{
                0x3f097b05u,
                0x3ef928f8u,
                0x3ef370ffu,
                0x3e8be141u},
            std::array{
                0x3f329a10u,
                0x3f464a1au,
                0x3f74ab7eu,
                0x3f156126u},
            std::array{
                0x3dcc86e3u,
                0x3c9bbc35u,
                0x3ee70124u,
                0x3cdb3660u},
            std::array{
                0x3f6819a5u,
                0x3d561a3au,
                0x3e81b40au,
                0x3f6b1363u}};
        for (auto i = std::size_t{0u}; i < probe_count; ++i) {
            expect(
                lanes(actual_bits[i]) == expected_bits[i],
                "fallback Sobol sample bits changed at probe " +
                    std::to_string(i));
        }

        constexpr std::array<std::uint32_t, metadata_count>
            expected_metadata{
                0x4bf378cbu,
                0u,
                16u,
                17u,
                19u,
                33u,
                16u,
                38201u,
                54017u,
                10898u,
                30843u};
        expect(
            actual_metadata == expected_metadata,
            "fallback Sobol dimension or shuffled-index metadata changed");

        std::cout
            << "All fallback-device tabulated-Sobol bit fixtures passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Fallback Sobol fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

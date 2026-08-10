#include "../src/luisa/cycles_texture_sampling.h"
#include "../src/luisa/path_tracer_texture_sampling.h"

#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;
using psycles::test_support::xir_instruction_count;

constexpr auto coordinate_count = std::uint32_t{7u};
constexpr auto source_interpolation_mode_count = std::uint32_t{4u};
constexpr auto source_extension_mode_count = std::uint32_t{4u};
constexpr auto comparison_variant_count =
    source_interpolation_mode_count * source_extension_mode_count;
constexpr auto result_stride = std::uint32_t{2u};

[[nodiscard]] Kernel1D<BindlessArray, Buffer<luisa::float4>>
make_repeated_sampling_kernel(bool shared) {
    constexpr auto repetition_count = std::uint32_t{8u};
    const auto callables = make_texture_2d_sampling_callables();
    return [callables, shared](
               BindlessVar textures,
               BufferFloat4 output) noexcept {
        CallableTexture2DSamplingProvider provider{
            textures,
            callables};
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            const auto uv = make_float2(
                0.17f + 0.031f * static_cast<float>(index),
                0.29f + 0.023f * static_cast<float>(index));
            output.write(
                index,
                shared
                    ? provider.sample(0u, uv, 1u, 0u)
                    : sample_cycles_texture_2d(
                          textures, 0u, uv, 1u, 0u));
        }
    };
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto inline_shape =
        make_repeated_sampling_kernel(false);
    const auto shared_shape =
        make_repeated_sampling_kernel(true);
    const auto inline_instructions =
        xir_instruction_count(inline_shape);
    const auto shared_instructions =
        xir_instruction_count(shared_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "texture sampler 8x XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
    }
    if (shared_instructions >= inline_instructions) {
        std::cerr
            << "Shared texture sampler did not reduce repeated XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
        return EXIT_FAILURE;
    }

    // Two independently recorded, structurally identical callables must
    // collapse by their complete AST hash. This is the construction pattern
    // produced by independent surface-operation factories.
    const auto first = make_texture_2d_sampling_callables();
    const auto second = make_texture_2d_sampling_callables();
    Kernel1D hash_reuse = [first, second](
                              BindlessVar textures,
                              BufferFloat4 output) noexcept {
        CallableTexture2DSamplingProvider a{textures, first};
        CallableTexture2DSamplingProvider b{textures, second};
        output.write(0u, a.sample(
                             0u,
                             make_float2(0.25f),
                             1u,
                             0u));
        output.write(1u, b.sample(
                             0u,
                             make_float2(0.75f),
                             1u,
                             0u));
    };
    if (hash_reuse.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Independently constructed texture callables were not "
               "deduplicated by complete hash\n";
        return EXIT_FAILURE;
    }

    const auto callables = make_texture_2d_sampling_callables();
    Kernel1D compare = [callables](
                           BindlessVar textures,
                           BufferFloat2 coordinates,
                           BufferFloat4 output) noexcept {
        const auto coordinate_index = dispatch_x();
        const auto uv = coordinates.read(coordinate_index);
        CallableTexture2DSamplingProvider provider{
            textures,
            callables};
        for (auto interpolation = std::uint32_t{0u};
             interpolation < source_interpolation_mode_count;
             ++interpolation) {
            for (auto extension = std::uint32_t{0u};
                 extension < source_extension_mode_count;
                 ++extension) {
                const auto variant =
                    interpolation * source_extension_mode_count +
                    extension;
                const auto base =
                    (variant * coordinate_count + coordinate_index) *
                    result_stride;
                output.write(
                    base,
                    sample_cycles_texture_2d(
                        textures,
                        0u,
                        uv,
                        interpolation,
                        extension));
                output.write(
                    base + 1u,
                    provider.sample(
                        0u,
                        uv,
                        interpolation,
                        extension));
            }
        }
    };
    if (compare.function()->function()
            .custom_callables()
            .size() !=
        texture_sampling_specialization_count) {
        std::cerr
            << "Texture sampler specialization regression: expected "
            << texture_sampling_specialization_count
            << " definitions, got "
            << compare.function()->function()
                   .custom_callables()
                   .size()
            << '\n';
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    constexpr auto width = std::uint32_t{3u};
    constexpr auto height = std::uint32_t{2u};
    constexpr std::array pixels{
        luisa::float4{0.10f, 0.20f, 0.30f, 0.40f},
        luisa::float4{0.50f, 0.60f, 0.70f, 0.80f},
        luisa::float4{0.90f, 1.00f, 0.10f, 0.20f},
        luisa::float4{0.30f, 0.40f, 0.50f, 0.60f},
        luisa::float4{0.70f, 0.80f, 0.90f, 1.00f},
        luisa::float4{0.20f, 0.30f, 0.40f, 0.50f}};
    constexpr std::array coordinates{
        luisa::float2{0.10f, 0.10f},
        luisa::float2{0.45f, 0.25f},
        luisa::float2{0.90f, 0.80f},
        luisa::float2{-0.10f, 0.35f},
        luisa::float2{1.10f, 0.65f},
        luisa::float2{-1.00f, 1.00f},
        luisa::float2{0.50f, 1.25f}};

    auto image = device.create_image<float>(
        PixelStorage::FLOAT4, width, height);
    auto textures = device.create_bindless_array(1u);
    textures.emplace_on_update(
        0u, image, Sampler::linear_point_repeat());
    auto coordinate_buffer =
        device.create_buffer<luisa::float2>(coordinate_count);
    constexpr auto result_count =
        comparison_variant_count * coordinate_count * result_stride;
    auto output = device.create_buffer<luisa::float4>(result_count);
    auto shader = device.compile(compare);
    std::vector<luisa::float4> actual(result_count);
    stream << image.copy_from(luisa::span{pixels})
           << coordinate_buffer.copy_from(luisa::span{coordinates})
           << textures.update()
           << shader(textures, coordinate_buffer, output)
                  .dispatch(coordinate_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (auto variant = std::uint32_t{0u};
         variant < comparison_variant_count;
         ++variant) {
        for (auto coordinate = std::uint32_t{0u};
             coordinate < coordinate_count;
             ++coordinate) {
            const auto base =
                (variant * coordinate_count + coordinate) *
                result_stride;
            if (!approximately_equal(
                    actual[base], actual[base + 1u], 2.0e-6f)) {
                const auto direct = actual[base];
                const auto shared = actual[base + 1u];
                std::cerr
                    << "Texture sampler mismatch on " << backend
                    << ", variant " << variant
                    << ", coordinate " << coordinate
                    << ": direct={" << direct.x << ", "
                    << direct.y << ", " << direct.z << ", "
                    << direct.w << "}, shared={" << shared.x
                    << ", " << shared.y << ", " << shared.z
                    << ", " << shared.w << "}\n";
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

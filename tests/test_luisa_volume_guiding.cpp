#include "../src/luisa/volume_guiding_filter.h"

#include <psycles/luisa/volume_guiding.h>

#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace guiding =
    psycles::luisa_backend::volume_guiding;
using psycles::luisa_backend::detail::
    VolumeGuidingFilter;

[[nodiscard]] bool same_float(
    float actual, float expected) noexcept {
    return std::bit_cast<std::uint32_t>(actual) ==
           std::bit_cast<std::uint32_t>(expected);
}

[[nodiscard]] bool same_float3(
    luisa::float4 actual,
    luisa::float3 expected) noexcept {
    return same_float(actual.x, expected.x) &&
           same_float(actual.y, expected.y) &&
           same_float(actual.z, expected.z);
}

void fail(std::string_view backend,
          std::string_view stage,
          std::size_t index,
          std::uint32_t actual,
          std::uint32_t expected) {
    std::cerr << "Cycles VSPG " << stage
              << " oracle failed on " << backend
              << " at record " << index
              << ": got 0x" << std::hex << actual
              << ", expected 0x" << expected
              << std::dec << '\n';
    std::exit(EXIT_FAILURE);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{
            argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();

    constexpr std::array codec_expected{
        0x00000000u,
        0x10204080u,
        0xb1c08040u,
        0x5f0180ffu,
        0x00000001u,
        0x00000000u,
        0xd0008000u,
        0x38204080u};
    constexpr std::array codec_decoded{
        luisa::float3{0.0f, 0.0f, 0.0f},
        luisa::float3{1.0f, 0.5f, 0.25f},
        luisa::float3{-1.0f, 2.0f, -3.0f},
        luisa::float3{65280.0f, -32768.0f, 256.0f},
        luisa::float3{0x1p-23f, 0.0f, 0.0f},
        luisa::float3{0.0f, 0.0f, 0.0f},
        luisa::float3{-0.0f, -1.0f, 0.0f},
        luisa::float3{256.0f, 128.0f, -64.0f}};

    auto codec_packed =
        device.create_buffer<luisa::uint>(
            codec_expected.size());
    auto codec_rgb =
        device.create_buffer<luisa::float4>(
            codec_expected.size());
    Kernel1D codec =
        [](BufferUInt packed,
           BufferFloat4 decoded) noexcept {
            const auto write =
                [&](std::uint32_t index,
                    Float3 rgb) noexcept {
                    const auto rgbe =
                        guiding::encode_rgbe(rgb);
                    packed.write(index, rgbe);
                    decoded.write(
                        index,
                        make_float4(
                            guiding::decode_rgbe(
                                rgbe),
                            0.0f));
                };
            write(0u, make_float3(0.0f));
            write(1u, make_float3(
                          1.0f, 0.5f, 0.25f));
            write(2u, make_float3(
                          -1.0f, 2.0f, -3.0f));
            write(3u, make_float3(
                          65280.0f,
                          -32640.0f,
                          255.0f));
            write(4u, make_float3(
                          0x1p-24f,
                          0.0f,
                          0.0f));
            write(5u, make_float3(
                          0x1p-25f,
                          0.0f,
                          0.0f));
            write(6u, make_float3(
                          -0.0f,
                          -1.0f,
                          0.0f));
            write(7u, make_float3(
                          255.5f,
                          127.75f,
                          -63.875f));
        };
    auto codec_shader =
        device.compile(
            codec,
            ShaderOption{
                .enable_cache = true,
                .enable_fast_math = false});
    std::array<luisa::uint, codec_expected.size()>
        actual_codec{};
    std::array<luisa::float4, codec_expected.size()>
        actual_decoded{};
    stream
        << codec_shader(codec_packed, codec_rgb)
               .dispatch(1u)
        << codec_packed.copy_to(
               luisa::span{actual_codec})
        << codec_rgb.copy_to(
               luisa::span{actual_decoded})
        << synchronize();
    for (std::size_t index = 0u;
         index < codec_expected.size();
         ++index) {
        if (actual_codec[index] !=
            codec_expected[index]) {
            fail(backend,
                 "RGBE encode",
                 index,
                 actual_codec[index],
                 codec_expected[index]);
        }
        if (!same_float3(
                actual_decoded[index],
                codec_decoded[index])) {
            std::cerr
                << "Cycles VSPG RGBE decode oracle "
                   "failed on "
                << backend << " at record " << index
                << '\n';
            return EXIT_FAILURE;
        }
    }

    constexpr auto width = 4u;
    constexpr auto height = 3u;
    constexpr auto pixels = width * height;
    luisa::vector<luisa::float4> raw(
        pixels * guiding::raw_pixel_stride);
    luisa::vector<luisa::uint> sample_count(
        pixels);
    for (auto y = 0u; y < height; ++y) {
        for (auto x = 0u; x < width; ++x) {
            const auto pixel = y * width + x;
            const auto base =
                pixel *
                guiding::raw_pixel_stride;
            raw[
                base +
                guiding::raw_scatter_slot] =
                luisa::make_float4(
                    static_cast<float>(x + 1u) *
                        0.25f,
                    static_cast<float>(y + 1u) *
                        -0.5f,
                    static_cast<float>(
                        (x + 1u) * (y + 1u)) *
                        0.125f,
                    0.0f);
            raw[
                base +
                guiding::raw_transmit_slot] =
                luisa::make_float4(
                    1.0f +
                        0.1f *
                            static_cast<float>(
                                pixel),
                    0.25f *
                        static_cast<float>(
                            static_cast<int>(
                                pixel % 3u) -
                            1),
                    -0.05f *
                        static_cast<float>(
                            pixel + 1u),
                    0.0f);
            sample_count[pixel] =
                pixel % 4u + 1u;
        }
    }

    auto raw_buffer =
        device.create_buffer<luisa::float4>(
            raw.size());
    auto count_buffer =
        device.create_buffer<luisa::uint>(
            sample_count.size());
    auto intermediate_buffer =
        device.create_buffer<luisa::uint>(
            pixels *
            guiding::denoised_pixel_stride);
    auto denoised_buffer =
        device.create_buffer<luisa::uint>(
            pixels *
            guiding::denoised_pixel_stride);
    VolumeGuidingFilter filter{device};
    stream
        << raw_buffer.copy_from(luisa::span{raw})
        << count_buffer.copy_from(
               luisa::span{sample_count});
    filter.dispatch(stream,
                    raw_buffer,
                    count_buffer,
                    intermediate_buffer,
                    denoised_buffer,
                    width,
                    height);

    constexpr std::array expected_x{
        0x4d4fd39fu, 0x6e101edeu,
        0x4d66e1cbu, 0x6e1416f3u,
        0x4d66b6cbu, 0x6e140acdu,
        0x4d4f759fu, 0x6e100489u,
        0x4e4fd34fu, 0x2f1d0499u,
        0x4e66e166u, 0x2f2104a7u,
        0x4e66b666u, 0x2f1c018bu,
        0x4d9feb9fu, 0x6e2702b8u,
        0x4f3c9f28u, 0x2f320bc4u,
        0x4f4ca933u, 0x2f3707d4u,
        0x4f4c8933u, 0x2f2f04afu,
        0x4e77b04fu, 0x2e3f06e7u};
    constexpr std::array expected_y{
        0x0d51d75du, 0x0e1b04a6u,
        0x0d67e577u, 0x0e1f03b5u,
        0x0d67b977u, 0x0e1b0197u,
        0x0ca1eeb9u, 0x0d2702c8u,
        0x0e369036u, 0x0e2800d1u,
        0x0e459a46u, 0x0e2d00e4u,
        0x0d8bf98bu, 0x0e2700beu,
        0x0d6ca06cu, 0x0d3600fbu,
        0x0e348b2eu, 0x0e2804c0u,
        0x0e43943bu, 0x0e2d03d0u,
        0x0d86f077u, 0x0e2701adu,
        0x0d689a5cu, 0x0d3501e5u};
    std::array<luisa::uint, expected_x.size()>
        actual_x{};
    std::array<luisa::uint, expected_y.size()>
        actual_y{};
    stream
        << intermediate_buffer.copy_to(
               luisa::span{actual_x})
        << denoised_buffer.copy_to(
               luisa::span{actual_y})
        << synchronize();
    for (std::size_t index = 0u;
         index < expected_x.size();
         ++index) {
        if (actual_x[index] != expected_x[index]) {
            fail(backend,
                 "filter X",
                 index,
                 actual_x[index],
                 expected_x[index]);
        }
        if (actual_y[index] != expected_y[index]) {
            fail(backend,
                 "filter Y",
                 index,
                 actual_y[index],
                 expected_y[index]);
        }
    }
    return EXIT_SUCCESS;
}

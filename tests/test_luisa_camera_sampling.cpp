#include <psycles/luisa/camera_sampling.h>
#include <psycles/luisa/pixel_filter.h>
#include <psycles/sampling/pixel_filter.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace camera_sampling = psycles::luisa_backend::camera_sampling;

constexpr std::array random_samples{luisa::float2{0.5f, 0.5f},
                                    luisa::float2{0.75f, 0.5f},
                                    luisa::float2{0.5f, 0.75f},
                                    luisa::float2{0.25f, 0.25f},
                                    luisa::float2{0.1f, 0.8f}};

constexpr std::array expected_disk{
    luisa::float2{0.0f, 0.0f},
    luisa::float2{0.5f, 0.0f},
    luisa::float2{0.0f, 0.5f},
    luisa::float2{-0.3535533906f, -0.3535533906f},
    luisa::float2{-0.6651756898f, 0.4444561864f}};

constexpr std::array raster_samples{luisa::float2{0.0f, 0.25f},
                                    luisa::float2{2.0f, 0.5f},
                                    luisa::float2{7.0f, 0.9f}};

constexpr std::array expected_raster{luisa::float2{7.0f, 0.75f},
                                     luisa::float2{5.0f, 0.5f},
                                     luisa::float2{0.0f, 0.1f}};

constexpr std::array filter_samples{0.0f, 0.125f, 0.5f, 0.75f, 0.999f, 1.0f};

[[nodiscard]] bool approximately_equal(float actual,
                                       float expected,
                                       float tolerance = 2.0e-5f) noexcept {
    return std::abs(actual - expected) <= tolerance;
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    constexpr auto landscape_horizontal_fit =
        camera_sampling::orthographic_viewplane_span(2.2f, 2.0f, true);
    constexpr auto landscape_vertical_fit =
        camera_sampling::orthographic_viewplane_span(2.2f, 2.0f, false);
    constexpr auto portrait_horizontal_fit =
        camera_sampling::orthographic_viewplane_span(2.2f, 0.5f, true);
    if (!approximately_equal(landscape_horizontal_fit.horizontal, 2.2f) ||
        !approximately_equal(landscape_horizontal_fit.vertical, 1.1f) ||
        !approximately_equal(landscape_vertical_fit.horizontal, 4.4f) ||
        !approximately_equal(landscape_vertical_fit.vertical, 2.2f) ||
        !approximately_equal(portrait_horizontal_fit.horizontal, 2.2f) ||
        !approximately_equal(portrait_horizontal_fit.vertical, 4.4f)) {
        std::cerr << "Cycles orthographic sensor-fit viewplane mapping failed\n";
        return EXIT_FAILURE;
    }
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto random_buffer =
        device.create_buffer<luisa::float2>(random_samples.size());
    auto disk_buffer =
        device.create_buffer<luisa::float2>(random_samples.size());
    auto polygon_buffer =
        device.create_buffer<luisa::float2>(random_samples.size());
    auto raster_sample_buffer =
        device.create_buffer<luisa::float2>(raster_samples.size());
    auto raster_result_buffer =
        device.create_buffer<luisa::float4>(raster_samples.size());
    const auto filter_table = psycles::sampling::make_pixel_filter_table(
        psycles::contract::PixelFilter::blackman_harris, 1.5f);
    auto filter_table_buffer = device.create_buffer<float>(filter_table.size());
    auto filter_sample_buffer =
        device.create_buffer<float>(filter_samples.size());
    auto filter_result_buffer =
        device.create_buffer<float>(filter_samples.size());
    auto clip_result_buffer =
        device.create_buffer<luisa::float4>(3u);
    auto camera_transform_result_buffer =
        device.create_buffer<luisa::uint4>(1u);

    Kernel1D evaluate = [](BufferFloat2 randoms,
                           BufferFloat2 disk,
                           BufferFloat2 polygon) noexcept {
        const auto index = dispatch_x();
        const auto random = randoms.read(index);
        disk.write(index, camera_sampling::sample_aperture(random, 0u, 0.0f));
        polygon.write(index,
                      camera_sampling::sample_aperture(random, 5u, 0.37f));
    };
    auto shader = device.compile(evaluate);
    Kernel1D evaluate_raster = [](BufferFloat2 samples,
                                  BufferFloat4 results) noexcept {
        const auto index = dispatch_x();
        const auto sample = samples.read(index);
        const auto output_y = cast<uint>(sample.x);
        constexpr auto height = 8u;
        const auto cycles_y = camera_sampling::cycles_pixel_y(output_y, height);
        const auto output_filter = camera_sampling::output_filter_y(sample.y);
        const auto cycles_raster = 2.0f * (cast<float>(cycles_y) + sample.y) /
                                       static_cast<float>(height) -
                                   1.0f;
        const auto output_raster = 1.0f - 2.0f * (sample.x + output_filter) /
                                              static_cast<float>(height);
        results.write(index,
                      make_float4(cast<float>(cycles_y),
                                  output_filter,
                                  cycles_raster,
                                  output_raster));
    };
    auto raster_shader = device.compile(evaluate_raster);
    Kernel1D evaluate_filter = [](BufferFloat table,
                                  BufferFloat samples,
                                  BufferFloat results) noexcept {
        const auto index = dispatch_x();
        results.write(index,
                      psycles::luisa_backend::pixel_filter::sample(
                          table, samples.read(index)));
    };
    auto filter_shader = device.compile(evaluate_filter);
    Kernel1D evaluate_clipping =
        [](BufferFloat4 results) noexcept {
            const auto index = dispatch_x();
            Float cosine = select(
                1.0f,
                0.8f,
                index == 1u);
            Float near_clip = select(
                0.1f,
                0.25f,
                index == 2u);
            Float far_clip = select(
                1000.0f,
                4.25f,
                index == 2u);
            auto range =
                camera_sampling::camera_clip_range(
                    near_clip,
                    far_clip,
                    cosine);
            const auto differential_position =
                0.03f + cast<float>(index) * 0.1f;
            const auto differential_direction =
                0.2f + cast<float>(index) * 0.1f;
            const auto clipped_differential_position =
                camera_sampling::advance_compact_differential_position(
                    differential_position,
                    differential_direction,
                    range.x);
            results.write(
                index,
                make_float4(
                    range.x,
                    clipped_differential_position,
                    range.y,
                    range.x + range.y));
        };
    auto clipping_shader = device.compile(evaluate_clipping);
    Kernel1D evaluate_camera_transform =
        [](BufferUInt4 results) noexcept {
            // This affine input is a real Cycles scene boundary case. Fusing
            // the local x product with translation yields 0x3f24212b;
            // backend-native matrix multiplication may round it to
            // 0x3f24212c and move a camera ray across coincident support.
            const auto transform = make_float4x4(
                make_float4(0.16329917311668396f, 0.0f, 0.0f, 0.0f),
                make_float4(0.0f, 0.16329915821552277f, 0.0f, 0.0f),
                make_float4(0.0f, 0.0f, 0.08164958655834198f, 0.0f),
                make_float4(0.4866490066051483f, 4.260610580444336f,
                            0.8810397982597351f, 1.0f));
            const auto camera_ray = camera_sampling::camera_to_world_ray(
                transform,
                make_float3(0.9460067749023438f, 0.0f, 0.0f),
                make_float3(-0.3722669184207916f,
                            -0.9224809408187866f,
                            -0.1022074967622757f));
            results.write(
                0u,
                make_uint4(as<uint>(camera_ray.origin.x),
                           as<uint>(camera_ray.origin.y),
                           as<uint>(camera_ray.origin.z),
                           as<uint>(camera_ray.direction.x)));
        };
    auto camera_transform_shader =
        device.compile(evaluate_camera_transform);
    std::array<luisa::float2, random_samples.size()> disk{};
    std::array<luisa::float2, random_samples.size()> polygon{};
    std::array<luisa::float4, raster_samples.size()> raster{};
    std::array<float, filter_samples.size()> filtered{};
    std::array<luisa::float4, 3u> clipping{};
    std::array<luisa::uint4, 1u> camera_transform_bits{};
    stream << random_buffer.copy_from(luisa::span{random_samples})
           << raster_sample_buffer.copy_from(luisa::span{raster_samples})
           << filter_table_buffer.copy_from(luisa::span{filter_table})
           << filter_sample_buffer.copy_from(luisa::span{filter_samples})
           << shader(random_buffer, disk_buffer, polygon_buffer)
                  .dispatch(static_cast<std::uint32_t>(random_samples.size()))
           << raster_shader(raster_sample_buffer, raster_result_buffer)
                  .dispatch(static_cast<std::uint32_t>(raster_samples.size()))
           << filter_shader(filter_table_buffer,
                            filter_sample_buffer,
                            filter_result_buffer)
                  .dispatch(static_cast<std::uint32_t>(filter_samples.size()))
           << clipping_shader(clip_result_buffer).dispatch(3u)
           << camera_transform_shader(camera_transform_result_buffer)
                  .dispatch(1u)
           << disk_buffer.copy_to(luisa::span{disk})
           << polygon_buffer.copy_to(luisa::span{polygon})
           << raster_result_buffer.copy_to(luisa::span{raster})
           << filter_result_buffer.copy_to(luisa::span{filtered})
           << clip_result_buffer.copy_to(luisa::span{clipping})
           << camera_transform_result_buffer.copy_to(
                  luisa::span{camera_transform_bits})
           << synchronize();

    for (std::size_t i = 0u; i < random_samples.size(); ++i) {
        if (!approximately_equal(disk[i].x, expected_disk[i].x) ||
            !approximately_equal(disk[i].y, expected_disk[i].y)) {
            std::cerr << "concentric aperture mapping failed on " << backend
                      << " for sample " << i << ": got {" << disk[i].x << ", "
                      << disk[i].y << "}, expected {" << expected_disk[i].x
                      << ", " << expected_disk[i].y << "}\n";
            return EXIT_FAILURE;
        }
        const auto polygon_radius_squared =
            polygon[i].x * polygon[i].x + polygon[i].y * polygon[i].y;
        if (!std::isfinite(polygon[i].x) || !std::isfinite(polygon[i].y) ||
            polygon_radius_squared > 1.0f + 2.0e-5f) {
            std::cerr << "polygon aperture invariant failed on " << backend
                      << " for sample " << i << ": {" << polygon[i].x << ", "
                      << polygon[i].y << "}\n";
            return EXIT_FAILURE;
        }
    }

    // Lock two non-trivial finite-blade values so this test checks the
    // corner selection, triangle map, and rotation—not only containment.
    constexpr auto expected_polygon_1 =
        luisa::float2{-0.3962322305f, -0.2903498468f};
    constexpr auto expected_polygon_4 =
        luisa::float2{0.7055815575f, 0.1016874003f};
    if (!approximately_equal(polygon[1u].x, expected_polygon_1.x) ||
        !approximately_equal(polygon[1u].y, expected_polygon_1.y) ||
        !approximately_equal(polygon[4u].x, expected_polygon_4.x) ||
        !approximately_equal(polygon[4u].y, expected_polygon_4.y)) {
        std::cerr << "finite-blade aperture mapping failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }

    for (std::size_t i = 0u; i < raster_samples.size(); ++i) {
        if (!approximately_equal(raster[i].x, expected_raster[i].x) ||
            !approximately_equal(raster[i].y, expected_raster[i].y) ||
            !approximately_equal(raster[i].z, raster[i].w)) {
            std::cerr << "top-left/output to lower-left/Cycles raster "
                         "mapping failed on "
                      << backend << " for sample " << i << ": got {"
                      << raster[i].x << ", " << raster[i].y << ", "
                      << raster[i].z << ", " << raster[i].w << "}\n";
            return EXIT_FAILURE;
        }
    }

    constexpr auto last_filter_index =
        psycles::sampling::pixel_filter_table_size - 1u;
    for (std::size_t i = 0u; i < filter_samples.size(); ++i) {
        const auto coordinate = std::clamp(filter_samples[i], 0.0f, 1.0f) *
                                static_cast<float>(last_filter_index);
        const auto index =
            std::min(static_cast<std::size_t>(coordinate), last_filter_index);
        const auto next = std::min(index + 1u, last_filter_index);
        const auto interpolation = coordinate - static_cast<float>(index);
        const auto expected = filter_table[index] * (1.0f - interpolation) +
                              filter_table[next] * interpolation;
        if (!approximately_equal(filtered[i], expected)) {
            std::cerr << "pixel-filter lookup failed on " << backend
                      << " for sample " << i << ": got " << filtered[i]
                      << ", expected " << expected << '\n';
            return EXIT_FAILURE;
        }
    }
    constexpr std::array expected_clipping{
        luisa::float4{0.1f, 0.05f, 999.9f, 1000.0f},
        luisa::float4{0.125f, 0.1675f, 1249.875f, 1250.0f},
        luisa::float4{0.25f, 0.33f, 4.0f, 4.25f}};
    for (auto i = std::size_t{0u};
         i < expected_clipping.size();
         ++i) {
        if (
            !approximately_equal(
                clipping[i].x,
                expected_clipping[i].x) ||
            !approximately_equal(
                clipping[i].y,
                expected_clipping[i].y) ||
            !approximately_equal(
                clipping[i].z,
                expected_clipping[i].z,
                5.0e-4f) ||
            !approximately_equal(
                clipping[i].w,
                expected_clipping[i].w,
                5.0e-4f)) {
            std::cerr
                << "Cycles camera clipping representation failed on "
                << backend << " for case " << i << '\n';
            return EXIT_FAILURE;
        }
    }
    constexpr auto cycles_camera_origin_x = std::uint32_t{0x3f24212bu};
    if (camera_transform_bits[0u].x != cycles_camera_origin_x) {
        std::cerr << "Cycles camera affine transform contraction failed on "
                  << backend << ": origin.x bits 0x" << std::hex
                  << camera_transform_bits[0u].x << ", expected 0x"
                  << cycles_camera_origin_x << std::dec << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

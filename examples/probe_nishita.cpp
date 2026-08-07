#include <psycles/luisa/cycles_nishita.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace nishita =
    psycles::luisa_backend::cycles_nishita;

constexpr float sun_elevation = 0.9250245094299316f;
constexpr float sun_rotation = 3.6651914755450647f;
constexpr float angular_diameter = 0.01745329238474369f;
constexpr float sun_intensity = 1.0f;
constexpr float altitude = 0.0f;
constexpr float air_density = 1.0f;
constexpr float dust_density = 1.0f;
constexpr float ozone_density = 1.0f;

void enforce_vulkan_native_xir_spirv(std::string_view backend) {
    if (backend != "vk" && backend != "vulkan") {
        return;
    }
    if (std::getenv("LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV") != nullptr) {
        return;
    }
#ifdef _WIN32
    _putenv_s("LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV", "1");
#else
    setenv(
        "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV",
        "1",
        1);
#endif
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    enforce_vulkan_native_xir_spirv(backend);
    auto device = context.create_device(backend);
    auto stream = device.create_stream();

    auto sky = device.create_image<float>(
        PixelStorage::FLOAT4,
        nishita::lut_width,
        nishita::lut_height);
    auto sun_pixels = device.create_buffer<luisa::float4>(2u);

    Kernel2D precompute = [](ImageFloat output,
                             BufferFloat4 sun_output,
                             Float elevation,
                             Float diameter,
                             Float height,
                             Float air,
                             Float dust,
                             Float ozone) noexcept {
        set_block_size(8u, 8u, 1u);
        UInt2 coordinate = dispatch_id().xy();
        const auto xyz = nishita::sky_lut_texel(
            coordinate.x,
            coordinate.y,
            elevation,
            height,
            air,
            dust,
            ozone);
        const auto pixel = make_float4(xyz, 1.0f);
        output.write(coordinate, pixel);
        output.write(
            make_uint2(
                nishita::lut_width - coordinate.x - 1u,
                coordinate.y),
            pixel);
        $if ((coordinate.x == 0u) &
             (coordinate.y == 0u)) {
            const auto values = nishita::sun_pixels(
                elevation,
                max(abs(diameter), 1.0e-7f),
                height,
                air,
                dust);
            sun_output.write(
                0u, make_float4(values.bottom_xyz, 1.0f));
            sun_output.write(
                1u, make_float4(values.top_xyz, 1.0f));
        };
    };
    auto precompute_shader = device.compile(precompute);
    stream
        << precompute_shader(
               sky,
               sun_pixels,
               sun_elevation,
               angular_diameter,
               altitude,
               air_density,
               dust_density,
               ozone_density)
               .dispatch(
                   nishita::half_lut_width,
                   nishita::lut_height)
        << synchronize();

    auto heap = device.create_bindless_array(1u);
    heap.emplace_on_update(
        0u, sky, Sampler::linear_point_edge());

    const auto cosine_elevation = std::cos(sun_elevation);
    const auto sun_direction = luisa::make_float3(
        -cosine_elevation * std::sin(sun_rotation),
        cosine_elevation * std::cos(sun_rotation),
        std::sin(sun_elevation));
    const std::array directions{
        normalize(luisa::make_float3(
            1.0e-3f, 2.0e-3f, 1.0f)),
        normalize(luisa::make_float3(
            0.0f, 1.0f, 1.0e-4f)),
        normalize(luisa::make_float3(
            1.0f, 0.0f, 1.0e-4f)),
        luisa::make_float3(0.0f, 0.0f, -1.0f),
        sun_direction};
    auto direction_buffer =
        device.create_buffer<luisa::float3>(
            directions.size());
    auto output_buffer =
        device.create_buffer<luisa::float4>(
            directions.size());

    Kernel1D evaluate = [](BindlessVar textures,
                           BufferVar<luisa::float3> rays,
                           BufferFloat4 precomputed_sun,
                           BufferFloat4 output,
                           Float elevation,
                           Float rotation,
                           Float diameter,
                           Float intensity) noexcept {
        const auto index = dispatch_x();
        const auto direction = rays.read(index);
        const auto sun_axis = make_float3(
            -cos(elevation) * sin(rotation),
            cos(elevation) * cos(rotation),
            sin(elevation));
        const auto base_xyz = nishita::sky_radiance_xyz(
            textures->tex2d(0u), direction, rotation);
        const auto sun_xyz =
            nishita::sun_disc_radiance_xyz(
                direction,
                sun_axis,
                precomputed_sun.read(0u).xyz(),
                precomputed_sun.read(1u).xyz(),
                elevation,
                diameter,
                intensity);
        const auto xyz = base_xyz + sun_xyz;
        const auto rgb = max(
            make_float3(
                3.2404542f * xyz.x -
                    1.5371385f * xyz.y -
                    0.4985314f * xyz.z,
                -0.9692660f * xyz.x +
                    1.8760108f * xyz.y +
                    0.0415560f * xyz.z,
                0.0556434f * xyz.x -
                    0.2040259f * xyz.y +
                    1.0572252f * xyz.z),
            make_float3(0.0f));
        output.write(
            index,
            make_float4(rgb, 1.0f));
    };
    auto evaluate_shader = device.compile(evaluate);
    std::array<luisa::float4, directions.size()> values{};
    stream << direction_buffer.copy_from(
                  luisa::span{directions})
           << heap.update()
           << evaluate_shader(
                  heap,
                  direction_buffer,
                  sun_pixels,
                  output_buffer,
                  sun_elevation,
                  sun_rotation,
                  angular_diameter,
                  sun_intensity)
                  .dispatch(
                      static_cast<std::uint32_t>(
                          directions.size()))
           << output_buffer.copy_to(luisa::span{values})
           << synchronize();

    constexpr std::array names{
        "near_zenith",
        "north_horizon",
        "east_horizon",
        "ground",
        "sun_0"};
    std::cout << std::setprecision(10)
              << "{\n  \"backend\": \"" << backend
              << "\",\n  \"probes\": {\n";
    for (std::size_t i = 0u; i < values.size(); ++i) {
        const auto value = values[i];
        std::cout << "    \"" << names[i] << "\": ["
                  << value.x << ", " << value.y << ", "
                  << value.z << "]"
                  << (i + 1u == values.size() ? "\n" : ",\n");
    }
    std::cout << "  }\n}\n";
    return EXIT_SUCCESS;
}

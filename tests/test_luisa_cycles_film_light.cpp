#include <psycles/luisa/cycles_film_light.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

[[nodiscard]] bool near(float actual,
                        float expected,
                        float tolerance = 2.0e-6f) noexcept {
    return std::isfinite(actual) &&
           std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool matches(luisa::float4 actual,
                           luisa::float4 expected) noexcept {
    return near(actual.x, expected.x) &&
           near(actual.y, expected.y) &&
           near(actual.z, expected.z) &&
           near(actual.w, expected.w);
}

}// namespace

int main(int argc, char **argv) {
    using namespace luisa::compute;
    namespace film =
        psycles::luisa_backend::cycles_film_light;

    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(6u);

    Kernel1D evaluate = [](BufferFloat4 records) noexcept {
        const Float nan = as<float>(UInt{0x7fc00000u});
        const Float positive_infinity =
            as<float>(UInt{0x7f800000u});
        const Float negative_infinity =
            as<float>(UInt{0xff800000u});
        const Float maximum =
            as<float>(UInt{0x7f7fffffu});

        records.write(
            0u,
            make_float4(
                film::ensure_finite(make_float3(
                    nan,
                    positive_infinity,
                    negative_infinity)),
                1.0f));
        records.write(
            1u,
            make_float4(
                film::clamp(
                    make_float3(nan, 4.0f, negative_infinity),
                    0u,
                    2.0f,
                    3.5f),
                1.0f));
        records.write(
            2u,
            make_float4(
                film::clamp(
                    make_float3(3.0f, -4.0f, 0.0f),
                    0u,
                    2.0f,
                    3.5f),
                1.0f));
        records.write(
            3u,
            make_float4(
                film::clamp(
                    make_float3(3.0f, -4.0f, 0.0f),
                    1u,
                    2.0f,
                    3.5f),
                1.0f));
        records.write(
            4u,
            make_float4(
                film::clamp(
                    make_float3(nan, 4.0f, -2.0f),
                    0u,
                    0.0f,
                    0.0f),
                1.0f));
        records.write(
            5u,
            make_float4(
                film::clamp(
                    make_float3(maximum, maximum, 0.0f),
                    0u,
                    1.0f,
                    1.0f),
                1.0f));
    };
    auto shader = device.compile(evaluate);
    std::array<luisa::float4, 6u> actual{};
    stream << shader(output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    constexpr std::array expected{
        luisa::float4{0.0f, 0.0f, 0.0f, 1.0f},
        luisa::float4{0.0f, 2.0f, 0.0f, 1.0f},
        luisa::float4{6.0f / 7.0f, -8.0f / 7.0f, 0.0f, 1.0f},
        luisa::float4{1.5f, -2.0f, 0.0f, 1.0f},
        luisa::float4{0.0f, 4.0f, -2.0f, 1.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 1.0f}};
    for (std::size_t i = 0u; i < expected.size(); ++i) {
        if (!matches(actual[i], expected[i])) {
            std::cerr
                << "Cycles film-light clamp failed on " << backend
                << " at record " << i << ": got {"
                << actual[i].x << ", " << actual[i].y << ", "
                << actual[i].z << ", " << actual[i].w
                << "}, expected {" << expected[i].x << ", "
                << expected[i].y << ", " << expected[i].z << ", "
                << expected[i].w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

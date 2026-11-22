#include <psycles/luisa/cycles_ray_differential.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

[[nodiscard]] bool approximately_equal(
    float actual, float expected) noexcept {
    return std::abs(actual - expected) <=
           2.0e-6f *
               (1.0f + std::max(
                           std::abs(actual),
                           std::abs(expected)));
}

} // namespace

int main(int argc, char **argv) {
    using namespace luisa::compute;
    using psycles::luisa_backend::cycles_ray_differential::
        after_surface_bounce;
    using psycles::luisa_backend::cycles_ray_differential::
        for_surface_shadow;

    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(3u);
    Kernel1D kernel = [&](BufferFloat4 result) noexcept {
        const auto diffuse = after_surface_bounce(
            0.125f, 0.25f, 0.5f, 1.0f, false);
        const auto narrower = after_surface_bounce(
            0.125f, 0.25f, 0.5f, 0.04f, false);
        const auto rough = after_surface_bounce(
            0.125f, 0.25f, 0.5f, 0.36f, false);
        const auto transparent = after_surface_bounce(
            0.125f, 0.25f, 0.5f, 1.0f, true);
        const auto shadow = for_surface_shadow(
            0.25f, 0.5f, 0.36f);
        result.write(0u,
            make_float4(
                diffuse.position,
                diffuse.direction,
                narrower.position,
                narrower.direction));
        result.write(1u,
            make_float4(
                rough.position,
                rough.direction,
                transparent.position,
                transparent.direction));
        result.write(2u,
            make_float4(
                shadow.position,
                shadow.direction,
                0.0f,
                0.0f));
    };
    auto compiled = device.compile(kernel);
    std::array<luisa::float4, 3u> actual{};
    stream << compiled(output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    constexpr std::array expected{
        luisa::float4{0.5f, 1.0f, 0.5f, 0.25f},
        luisa::float4{0.5f, 0.6f, 0.125f, 0.25f},
        luisa::float4{0.5f, 0.6f, 0.0f, 0.0f}};
    for (std::size_t record = 0u;
         record < actual.size();
         ++record) {
        const auto got = actual[record];
        const auto want = expected[record];
        if (!approximately_equal(got.x, want.x) ||
            !approximately_equal(got.y, want.y) ||
            !approximately_equal(got.z, want.z) ||
            !approximately_equal(got.w, want.w)) {
            std::cerr
                << "Cycles surface ray-differential regression on "
                << backend << " at record " << record
                << ": got {" << got.x << ", " << got.y
                << ", " << got.z << ", " << got.w << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

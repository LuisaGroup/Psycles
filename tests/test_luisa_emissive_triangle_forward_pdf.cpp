#include "../src/luisa/path_kernel_emissive_triangle.h"

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
using namespace psycles::luisa_backend::detail;

[[nodiscard]] bool near(float lhs, float rhs) noexcept {
    const auto scale = std::max({1.0f, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= 1.0e-5f * scale;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(5u);
    const auto component = make_emissive_triangle_component();

    // No scene or emitter buffer is captured by this kernel. That is the
    // structural regression: forward-hit PDF evaluation depends only on the
    // committed primitive's sampling policy, its geometry, and the global
    // legacy-distribution density, so its work cannot grow with emitter count.
    Kernel1D evaluate = [component](BufferFloat4 records) noexcept {
        const auto reference = make_float3(0.0f, 0.0f, 2.0f);
        const auto light_position = make_float3(0.0f, 0.0f, 0.0f);
        const auto p0 = make_float3(-1.0f, -1.0f, 0.0f);
        const auto p1 = make_float3(1.0f, -1.0f, 0.0f);
        const auto p2 = make_float3(0.0f, 1.0f, 0.0f);
        const auto front_normal = make_float3(0.0f, 0.0f, 1.0f);
        const auto back_normal = -front_normal;
        const auto density = 0.125f;
        const auto flag = [](Bool value) noexcept {
            return select(0.0f, 1.0f, value);
        };
        const auto pdf = [&](std::uint32_t sampling, Float triangle_density,
                             Float3 normal, Float3 vertex2) noexcept {
            return component->from_intersection(triangle_density, sampling,
                                                reference, light_position, p0,
                                                p1, vertex2, normal);
        };

        const auto front = pdf(static_cast<std::uint32_t>(
                                   psycles::contract::EmissionSampling::front),
                               density, front_normal, p2);
        const auto back = pdf(static_cast<std::uint32_t>(
                                  psycles::contract::EmissionSampling::back),
                              density, front_normal, p2);
        records.write(0u, make_float4(front.value, flag(front.valid),
                                      back.value, flag(back.valid)));

        const auto automatic =
            pdf(static_cast<std::uint32_t>(
                    psycles::contract::EmissionSampling::automatic),
                density, front_normal, p2);
        const auto front_back =
            pdf(static_cast<std::uint32_t>(
                    psycles::contract::EmissionSampling::front_back),
                density, front_normal, p2);
        records.write(1u,
                      make_float4(automatic.value, flag(automatic.valid),
                                  front_back.value, flag(front_back.valid)));

        const auto none = pdf(static_cast<std::uint32_t>(
                                  psycles::contract::EmissionSampling::none),
                              density, front_normal, p2);
        const auto doubled_density =
            pdf(static_cast<std::uint32_t>(
                    psycles::contract::EmissionSampling::front),
                density * 2.0f, front_normal, p2);
        records.write(2u, make_float4(none.value, flag(none.valid),
                                      doubled_density.value,
                                      flag(doubled_density.valid)));

        const auto flipped_front =
            pdf(static_cast<std::uint32_t>(
                    psycles::contract::EmissionSampling::front),
                density, back_normal, p2);
        const auto flipped_back =
            pdf(static_cast<std::uint32_t>(
                    psycles::contract::EmissionSampling::back),
                density, back_normal, p2);
        records.write(
            3u, make_float4(flipped_front.value, flag(flipped_front.valid),
                            flipped_back.value, flag(flipped_back.valid)));

        const auto degenerate =
            pdf(static_cast<std::uint32_t>(
                    psycles::contract::EmissionSampling::front_back),
                density, front_normal, p1);
        records.write(4u, make_float4(degenerate.value, flag(degenerate.valid),
                                      0.0f, 0.0f));
    };

    auto shader =
        device.compile(evaluate, ShaderOption{.enable_cache = false,
                                              .enable_fast_math = false});
    std::array<luisa::float4, 5u> actual{};
    stream << shader(output).dispatch(1u) << output.copy_to(luisa::span{actual})
           << synchronize();

    const auto front = actual[0u].x;
    const auto failed =
        !(front > 0.0f) || actual[0u].y != 1.0f || actual[0u].z != 0.0f ||
        actual[0u].w != 0.0f || !near(actual[1u].x, front) ||
        actual[1u].y != 1.0f || !near(actual[1u].z, front) ||
        actual[1u].w != 1.0f || actual[2u].x != 0.0f || actual[2u].y != 0.0f ||
        !near(actual[2u].z, 2.0f * front) || actual[2u].w != 1.0f ||
        actual[3u].x != 0.0f || actual[3u].y != 0.0f ||
        !near(actual[3u].z, front) || actual[3u].w != 1.0f ||
        actual[4u].x != 0.0f || actual[4u].y != 0.0f;
    if (failed) {
        std::cerr << "O(1) emissive-triangle forward PDF contract failed on "
                  << backend << '\n';
        for (auto index = std::size_t{0u}; index < actual.size(); ++index) {
            std::cerr << "  [" << index << "] = {" << actual[index].x << ", "
                      << actual[index].y << ", " << actual[index].z << ", "
                      << actual[index].w << "}\n";
        }
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

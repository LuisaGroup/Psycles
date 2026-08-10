#include "../src/luisa/path_tracer_color_transforms.h"
#include "../src/luisa/surface_color_transforms.h"

#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <array>
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

constexpr auto input_count = std::uint32_t{8u};
constexpr auto result_stride = std::uint32_t{8u};

[[nodiscard]] bool approximately_equal_float3(
    luisa::float3 lhs,
    luisa::float3 rhs,
    float tolerance = 1.0e-6f) noexcept {
    return approximately_equal(lhs.x, rhs.x, tolerance) &&
           approximately_equal(lhs.y, rhs.y, tolerance) &&
           approximately_equal(lhs.z, rhs.z, tolerance);
}

[[nodiscard]] Kernel1D<Buffer<luisa::float3>>
make_repeated_roundtrip_kernel(bool shared) {
    constexpr auto repetition_count = std::uint32_t{8u};
    return [shared](BufferFloat3 output) noexcept {
        CallableSurfaceColorTransformProvider provider;
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            auto rgb = make_float3(
                0.13f + 0.031f * static_cast<float>(index),
                0.29f + 0.017f * static_cast<float>(index),
                0.71f - 0.023f * static_cast<float>(index));
            auto hsv = shared
                           ? provider.rgb_to_hsv(rgb)
                           : rgb_to_hsv_inline(rgb);
            output.write(
                index,
                shared
                    ? provider.hsv_to_rgb(hsv)
                    : hsv_to_rgb_inline(hsv));
        }
    };
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto inline_shape =
        make_repeated_roundtrip_kernel(false);
    const auto shared_shape =
        make_repeated_roundtrip_kernel(true);
    const auto inline_instructions =
        xir_instruction_count(inline_shape);
    const auto shared_instructions =
        xir_instruction_count(shared_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "color transform roundtrip 8x XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
    }
    if (shared_instructions >= inline_instructions) {
        std::cerr
            << "Shared color transforms did not reduce repeated XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
        return EXIT_FAILURE;
    }

    // Recording one endpoint must not attach the other three members of the
    // finite family to the module.
    Kernel1D used_only = [](BufferFloat3 output) noexcept {
        CallableSurfaceColorTransformProvider provider;
        output.write(
            0u,
            provider.rgb_to_hsv(
                make_float3(0.25f, 0.5f, 0.75f)));
    };
    if (used_only.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Unused color-transform definitions reached the module\n";
        return EXIT_FAILURE;
    }

    // Independent providers model separately recorded surface-operation
    // factories. Their identical endpoint definitions must collapse by the
    // complete callable AST hash.
    Kernel1D hash_reuse = [](BufferFloat3 output) noexcept {
        CallableSurfaceColorTransformProvider first;
        CallableSurfaceColorTransformProvider second;
        output.write(
            0u,
            first.rgb_to_hsv(
                make_float3(0.25f, 0.5f, 0.75f)));
        output.write(
            1u,
            second.rgb_to_hsv(
                make_float3(0.75f, 0.5f, 0.25f)));
    };
    if (hash_reuse.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Independent color-transform callables were not deduplicated "
               "by complete hash\n";
        return EXIT_FAILURE;
    }

    Kernel1D compare = [](
                           BufferFloat3 input,
                           BufferFloat3 output) noexcept {
        const auto index = dispatch_x();
        const auto value = input.read(index);
        const auto base = index * result_stride;
        CallableSurfaceColorTransformProvider provider;
        output.write(base, rgb_to_hsv_inline(value));
        output.write(base + 1u, provider.rgb_to_hsv(value));
        output.write(base + 2u, hsv_to_rgb_inline(value));
        output.write(base + 3u, provider.hsv_to_rgb(value));
        output.write(base + 4u, rgb_to_hsl_inline(value));
        output.write(base + 5u, provider.rgb_to_hsl(value));
        output.write(base + 6u, hsl_to_rgb_inline(value));
        output.write(base + 7u, provider.hsl_to_rgb(value));
    };
    if (compare.function()->function()
            .custom_callables()
            .size() != 4u) {
        std::cerr
            << "Color-transform family regression: expected four reachable "
               "definitions, got "
            << compare.function()->function()
                   .custom_callables()
                   .size()
            << '\n';
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    constexpr std::array inputs{
        luisa::float3{0.0f, 0.0f, 0.0f},
        luisa::float3{1.0f, 1.0f, 1.0f},
        luisa::float3{1.0f, 0.0f, 0.0f},
        luisa::float3{0.0f, 1.0f, 0.0f},
        luisa::float3{0.0f, 0.0f, 1.0f},
        luisa::float3{0.13f, 0.47f, 0.89f},
        luisa::float3{0.75f, 0.25f, 0.60f},
        luisa::float3{0.999f, 0.001f, 0.50f}};
    auto input = device.create_buffer<luisa::float3>(input_count);
    auto output = device.create_buffer<luisa::float3>(
        input_count * result_stride);
    auto shader = device.compile(compare);
    std::vector<luisa::float3> actual(
        input_count * result_stride);
    stream << input.copy_from(luisa::span{inputs})
           << shader(input, output).dispatch(input_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (auto input_index = std::uint32_t{0u};
         input_index < input_count;
         ++input_index) {
        const auto base = input_index * result_stride;
        for (auto transform = std::uint32_t{0u};
             transform < 4u;
             ++transform) {
            const auto direct = actual[base + transform * 2u];
            const auto shared = actual[base + transform * 2u + 1u];
            if (!approximately_equal_float3(direct, shared)) {
                std::cerr
                    << "Color-transform mismatch on " << backend
                    << ", input " << input_index
                    << ", transform " << transform
                    << ": direct={" << direct.x << ", "
                    << direct.y << ", " << direct.z
                    << "}, shared={" << shared.x << ", "
                    << shared.y << ", " << shared.z << "}\n";
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

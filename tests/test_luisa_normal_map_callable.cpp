#include "../src/luisa/path_tracer_normal_maps.h"
#include "../src/luisa/surface_normal_map.h"

#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;
using psycles::test_support::xir_instruction_count;

constexpr auto case_count = std::uint32_t{8u};
constexpr auto result_stride = std::uint32_t{8u};

[[nodiscard]] SurfaceNormalMapInput make_input(
    UInt index) noexcept {
    const auto value = cast<float>(index);
    const auto odd = (index & 1u) != 0u;
    const auto tangent_valid = (index % 4u) != 0u;
    const auto sign_valid = (index % 3u) != 0u;
    const auto base_valid = (index % 5u) != 0u;
    const auto tangent_sign = select(
        0.0f,
        select(1.0f, -1.0f, odd),
        sign_valid);
    return {
        .mapped = make_float3(
            -0.75f + 0.23f * value,
            0.65f - 0.17f * value,
            0.15f + 0.11f * value),
        .strength = -0.25f + 0.30f * value,
        .object_tangent = select(
            make_float3(0.0f),
            make_float3(1.0f, 0.07f * value, 0.1f),
            tangent_valid),
        .tangent_sign = tangent_sign,
        .tangent_attribute_found = (index & 2u) == 0u,
        .object_shading_normal = select(
            make_float3(0.0f),
            make_float3(0.05f * value, 0.1f, 1.0f),
            base_valid),
        .undisplaced_object_shading_normal = make_float3(
            -0.03f * value, 0.2f, 0.85f),
        .triangle_smooth = odd,
        .normal_to_world_x = make_float3(1.4f, 0.1f, 0.0f),
        .normal_to_world_y = make_float3(0.0f, 0.8f, 0.2f),
        .normal_to_world_z = make_float3(0.1f, 0.0f, 1.7f),
        .shading_normal = make_float3(0.0f, 0.0f, 1.0f),
        .back_facing = (index & 4u) != 0u,
        .geometry_index = select(~0u, 7u, (index & 2u) == 0u),
        .is_curve = (index % 7u) == 0u};
}

[[nodiscard]] Kernel1D<Buffer<luisa::float3>>
make_repeated_tangent_displaced_kernel(bool shared) {
    // The host loop represents eight distinct immutable Normal Map graph
    // sites, not a path-depth/device loop.
    constexpr auto repetition_count = std::uint32_t{8u};
    return [shared](BufferFloat3 output) noexcept {
        CallableSurfaceNormalMapProvider provider;
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            const auto input = make_input(UInt{index});
            output.write(
                index,
                shared
                    ? provider.evaluate_tangent_displaced(input)
                    : normal_map_tangent_displaced_inline(input));
        }
    };
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto inline_shape =
        make_repeated_tangent_displaced_kernel(false);
    const auto shared_shape =
        make_repeated_tangent_displaced_kernel(true);
    const auto inline_instructions =
        xir_instruction_count(inline_shape);
    const auto shared_instructions =
        xir_instruction_count(shared_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "tangent-displaced Normal Map 8x XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
    }
    if (shared_instructions >= inline_instructions) {
        std::cerr
            << "Shared Normal Map evaluation did not reduce repeated XIR: "
               "inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
        return EXIT_FAILURE;
    }

    Kernel1D used_only = [](BufferFloat3 output) noexcept {
        CallableSurfaceNormalMapProvider provider;
        output.write(
            0u,
            provider.evaluate_tangent_displaced(
                make_input(1u)));
    };
    if (used_only.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Unused Normal Map definitions reached the module\n";
        return EXIT_FAILURE;
    }

    Kernel1D hash_reuse = [](BufferFloat3 output) noexcept {
        CallableSurfaceNormalMapProvider first;
        CallableSurfaceNormalMapProvider second;
        output.write(
            0u,
            first.evaluate_tangent_displaced(
                make_input(1u)));
        output.write(
            1u,
            second.evaluate_tangent_displaced(
                make_input(2u)));
    };
    if (hash_reuse.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Independent Normal Map callables were not deduplicated "
               "by complete hash\n";
        return EXIT_FAILURE;
    }

    Kernel1D compare = [](BufferFloat4 output) noexcept {
        const auto index = dispatch_x();
        const auto input = make_input(index);
        const auto base = index * result_stride;
        CallableSurfaceNormalMapProvider provider;
        output.write(
            base,
            make_float4(
                normal_map_tangent_displaced_inline(input),
                1.0f));
        output.write(
            base + 1u,
            make_float4(
                provider.evaluate_tangent_displaced(input),
                1.0f));
        output.write(
            base + 2u,
            make_float4(
                normal_map_tangent_original_inline(input),
                1.0f));
        output.write(
            base + 3u,
            make_float4(
                provider.evaluate_tangent_original(input),
                1.0f));
        output.write(
            base + 4u,
            make_float4(
                normal_map_object_inline(input),
                1.0f));
        output.write(
            base + 5u,
            make_float4(
                provider.evaluate_object(input),
                1.0f));
        output.write(
            base + 6u,
            make_float4(
                normal_map_world_inline(input),
                1.0f));
        output.write(
            base + 7u,
            make_float4(
                provider.evaluate_world(input),
                1.0f));
    };
    const auto reachable_definitions = compare.function()->function()
                                           .custom_callables()
                                           .size();
    if (reachable_definitions != 4u) {
        std::cerr
            << "Normal Map family regression: expected four reachable "
               "definitions, got "
            << reachable_definitions << '\n';
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(
        case_count * result_stride);
    auto shader = device.compile(compare);
    std::vector<luisa::float4> actual(case_count * result_stride);
    stream << shader(output).dispatch(case_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (auto case_index = std::uint32_t{0u};
         case_index < case_count;
         ++case_index) {
        const auto base = case_index * result_stride;
        for (auto endpoint = std::uint32_t{0u};
             endpoint < result_stride / 2u;
             ++endpoint) {
            const auto direct = actual[base + endpoint * 2u];
            const auto shared = actual[base + endpoint * 2u + 1u];
            if (!approximately_equal(direct, shared)) {
                std::cerr
                    << "Normal Map callable mismatch on " << backend
                    << ", case " << case_index
                    << ", endpoint " << endpoint
                    << ": direct={" << direct.x << ", "
                    << direct.y << ", " << direct.z << "}, shared={"
                    << shared.x << ", " << shared.y << ", "
                    << shared.z << "}\n";
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

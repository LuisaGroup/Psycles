#include "../src/luisa/path_tracer_bump.h"
#include "../src/luisa/surface_bump.h"

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
constexpr auto result_stride = std::uint32_t{4u};

[[nodiscard]] SurfaceBumpInput make_input(UInt index) noexcept {
    const auto value = cast<float>(index);
    const auto odd = (index & 1u) != 0u;
    const auto derivatives_valid = (index % 4u) != 0u;
    const auto transform_valid = (index % 5u) != 0u;
    const auto normal = select(
        make_float3(0.0f),
        make_float3(0.05f * value, 0.1f, 1.0f),
        (index % 6u) != 0u);
    const auto dPdx = select(
        make_float3(0.0f),
        make_float3(
            select(1.0f, -1.0f, odd),
            0.03f * value,
            0.0f),
        derivatives_valid);
    const auto dPdy = select(
        make_float3(0.0f),
        make_float3(0.0f, 0.8f, 0.07f * value),
        derivatives_valid);
    const auto column_z = select(
        make_float3(0.0f),
        make_float3(0.1f, 0.0f, 1.7f),
        transform_valid);
    return {
        .normal = normal,
        .filter_width = 0.125f * value,
        .dPdx = dPdx,
        .dPdy = dPdy,
        .height_center = 0.1f * value,
        .height_x = 0.1f * value + 0.03f * (value - 2.0f),
        .height_y = 0.1f * value - 0.02f * (value - 5.0f),
        .distance = -0.4f + 0.15f * value,
        .strength = 0.2f * value,
        .normal_to_world_x = make_float3(1.4f, 0.1f, 0.0f),
        .normal_to_world_y = make_float3(0.0f, 0.8f, 0.2f),
        .normal_to_world_z = column_z,
        .object_shading_normal = make_float3(0.0f, 0.0f, 1.0f),
        .shading_normal = make_float3(0.0f, 0.0f, 1.0f)};
}

[[nodiscard]] Kernel1D<Buffer<luisa::float3>>
make_repeated_world_bump_kernel(bool shared) {
    // These are eight distinct post-height Bump graph sites. Runtime height
    // evaluation is intentionally outside this focused finite-endpoint test.
    constexpr auto repetition_count = std::uint32_t{8u};
    return [shared](BufferFloat3 output) noexcept {
        CallableSurfaceBumpProvider provider;
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            const auto input = make_input(UInt{index});
            output.write(
                index,
                shared
                    ? provider.evaluate_world(input)
                    : bump_world_inline(input));
        }
    };
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto inline_shape =
        make_repeated_world_bump_kernel(false);
    const auto shared_shape =
        make_repeated_world_bump_kernel(true);
    const auto inline_instructions =
        xir_instruction_count(inline_shape);
    const auto shared_instructions =
        xir_instruction_count(shared_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "world Bump core 8x XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
    }
    if (shared_instructions >= inline_instructions) {
        std::cerr
            << "Shared Bump evaluation did not reduce repeated XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
        return EXIT_FAILURE;
    }

    Kernel1D used_only = [](BufferFloat3 output) noexcept {
        CallableSurfaceBumpProvider provider;
        output.write(
            0u,
            provider.evaluate_world(make_input(1u)));
    };
    if (used_only.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr << "Unused Bump definitions reached the module\n";
        return EXIT_FAILURE;
    }

    Kernel1D hash_reuse = [](BufferFloat3 output) noexcept {
        CallableSurfaceBumpProvider first;
        CallableSurfaceBumpProvider second;
        output.write(
            0u,
            first.evaluate_world(make_input(1u)));
        output.write(
            1u,
            second.evaluate_world(make_input(2u)));
    };
    if (hash_reuse.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Independent Bump callables were not deduplicated by "
               "complete hash\n";
        return EXIT_FAILURE;
    }

    Kernel1D compare = [](BufferFloat4 output) noexcept {
        const auto index = dispatch_x();
        const auto input = make_input(index);
        const auto base = index * result_stride;
        CallableSurfaceBumpProvider provider;
        output.write(
            base,
            make_float4(bump_world_inline(input), 1.0f));
        output.write(
            base + 1u,
            make_float4(provider.evaluate_world(input), 1.0f));
        output.write(
            base + 2u,
            make_float4(bump_object_inline(input), 1.0f));
        output.write(
            base + 3u,
            make_float4(provider.evaluate_object(input), 1.0f));
    };
    const auto reachable_definitions = compare.function()->function()
                                           .custom_callables()
                                           .size();
    if (reachable_definitions != 2u) {
        std::cerr
            << "Bump family regression: expected two reachable definitions, "
               "got "
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
                    << "Bump callable mismatch on " << backend
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

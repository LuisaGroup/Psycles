#include "../src/luisa/path_tracer_shader_tables.h"
#include "../src/luisa/surface_shader_table_evaluation.h"

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
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;
using psycles::test_support::xir_instruction_count;

constexpr auto case_count = std::uint32_t{8u};
constexpr auto result_stride = std::uint32_t{12u};

constexpr auto sampled_ramp_offset = std::uint32_t{0u};
constexpr auto sampled_ramp_count = std::uint32_t{5u};
constexpr auto sampled_ramp_width = std::uint32_t{4u};
constexpr auto control_ramp_offset = std::uint32_t{20u};
constexpr auto control_ramp_count = std::uint32_t{4u};
constexpr auto control_ramp_width = std::uint32_t{5u};
constexpr auto sampled_curve_offset = std::uint32_t{40u};
constexpr auto sampled_curve_count = std::uint32_t{5u};
constexpr auto sampled_curve_width = std::uint32_t{3u};
constexpr auto control_curve_offset = std::uint32_t{55u};
constexpr auto control_curve_count = std::uint32_t{4u};
constexpr auto control_curve_width = std::uint32_t{4u};

[[nodiscard]] SurfaceShaderTableView sampled_ramp_view() noexcept {
    return {
        .offset = sampled_ramp_offset,
        .count = sampled_ramp_count,
        .width = sampled_ramp_width};
}

[[nodiscard]] SurfaceShaderTableView control_ramp_view() noexcept {
    return {
        .offset = control_ramp_offset,
        .count = control_ramp_count,
        .width = control_ramp_width};
}

[[nodiscard]] SurfaceShaderTableView sampled_curve_view() noexcept {
    return {
        .offset = sampled_curve_offset,
        .count = sampled_curve_count,
        .width = sampled_curve_width};
}

[[nodiscard]] SurfaceShaderTableView control_curve_view() noexcept {
    return {
        .offset = control_curve_offset,
        .count = control_curve_count,
        .width = control_curve_width};
}

[[nodiscard]] Kernel1D<
    Buffer<float>,
    Buffer<float>,
    Buffer<luisa::float4>>
make_repeated_sampled_ramp_kernel(bool shared) {
    // This host loop deliberately models eight distinct Color Ramp nodes in a
    // graph. It is not a path-depth/device loop: the assertion below requires
    // those immutable sites to share one typed callable definition.
    constexpr auto repetition_count = std::uint32_t{8u};
    return [shared](
               BufferFloat data,
               BufferFloat factors,
               BufferFloat4 output) noexcept {
        const auto table = sampled_ramp_view();
        BufferSurfaceShaderTableReader reader{data, table};
        CallableSurfaceShaderTableProvider<BufferFloat> provider{data};
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            const auto factor = factors.read(index);
            output.write(
                index,
                shared
                    ? provider.color_ramp_sampled_linear(
                          table, factor)
                    : color_ramp_sampled_linear_inline(
                          reader, factor));
        }
    };
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto inline_shape =
        make_repeated_sampled_ramp_kernel(false);
    const auto shared_shape =
        make_repeated_sampled_ramp_kernel(true);
    const auto inline_instructions =
        xir_instruction_count(inline_shape);
    const auto shared_instructions =
        xir_instruction_count(shared_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "sampled Color Ramp 8x XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
    }
    if (shared_instructions >= inline_instructions) {
        std::cerr
            << "Shared Color Ramp evaluation did not reduce repeated XIR: "
               "inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
        return EXIT_FAILURE;
    }

    // Constructing the provider does not eagerly attach its finite family.
    Kernel1D used_only = [](
                             BufferFloat data,
                             BufferFloat4 output) noexcept {
        CallableSurfaceShaderTableProvider<BufferFloat> provider{data};
        output.write(
            0u,
            provider.color_ramp_sampled_linear(
                sampled_ramp_view(), 0.37f));
    };
    if (used_only.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Unused shader-table definitions reached the module\n";
        return EXIT_FAILURE;
    }

    // Two independently constructed providers must converge on the same
    // complete callable AST hash instead of cloning an equivalent definition.
    Kernel1D hash_reuse = [](
                              BufferFloat data,
                              BufferFloat4 output) noexcept {
        CallableSurfaceShaderTableProvider<BufferFloat> first{data};
        CallableSurfaceShaderTableProvider<BufferFloat> second{data};
        output.write(
            0u,
            first.color_ramp_sampled_linear(
                sampled_ramp_view(), 0.25f));
        output.write(
            1u,
            second.color_ramp_sampled_linear(
                sampled_ramp_view(), 0.75f));
    };
    if (hash_reuse.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Independent shader-table callables were not deduplicated "
               "by complete hash\n";
        return EXIT_FAILURE;
    }

    Kernel1D compare = [](
                           BufferFloat data,
                           BufferFloat factors,
                           BufferFloat3 inputs,
                           BufferFloat4 output) noexcept {
        const auto index = dispatch_x();
        const auto factor = factors.read(index);
        const auto input = inputs.read(index);
        const auto extrapolate = cast<float>(index & 1u);
        const auto base = index * result_stride;
        const auto sampled_ramp = sampled_ramp_view();
        const auto control_ramp = control_ramp_view();
        const auto sampled_curve = sampled_curve_view();
        const auto control_curve = control_curve_view();
        BufferSurfaceShaderTableReader sampled_ramp_reader{
            data, sampled_ramp};
        BufferSurfaceShaderTableReader control_ramp_reader{
            data, control_ramp};
        BufferSurfaceShaderTableReader sampled_curve_reader{
            data, sampled_curve};
        BufferSurfaceShaderTableReader control_curve_reader{
            data, control_curve};
        CallableSurfaceShaderTableProvider<BufferFloat> provider{data};

        output.write(
            base,
            color_ramp_sampled_linear_inline(
                sampled_ramp_reader, factor));
        output.write(
            base + 1u,
            provider.color_ramp_sampled_linear(
                sampled_ramp, factor));
        output.write(
            base + 2u,
            color_ramp_sampled_constant_inline(
                sampled_ramp_reader, factor));
        output.write(
            base + 3u,
            provider.color_ramp_sampled_constant(
                sampled_ramp, factor));
        output.write(
            base + 4u,
            color_ramp_control_linear_inline(
                control_ramp_reader, factor));
        output.write(
            base + 5u,
            provider.color_ramp_control_linear(
                control_ramp, factor));
        output.write(
            base + 6u,
            color_ramp_control_constant_inline(
                control_ramp_reader, factor));
        output.write(
            base + 7u,
            provider.color_ramp_control_constant(
                control_ramp, factor));
        output.write(
            base + 8u,
            make_float4(
                rgb_curve_sampled_inline(
                    sampled_curve_reader,
                    input,
                    factor,
                    -0.5f,
                    1.5f,
                    extrapolate),
                1.0f));
        output.write(
            base + 9u,
            make_float4(
                provider.rgb_curve_sampled(
                    sampled_curve,
                    input,
                    factor,
                    -0.5f,
                    1.5f,
                    extrapolate),
                1.0f));
        output.write(
            base + 10u,
            make_float4(
                rgb_curve_control_inline(
                    control_curve_reader,
                    input,
                    factor),
                1.0f));
        output.write(
            base + 11u,
            make_float4(
                provider.rgb_curve_control(
                    control_curve,
                    input,
                    factor),
                1.0f));
    };
    const auto reachable_definitions = compare.function()->function()
                                           .custom_callables()
                                           .size();
    if (reachable_definitions != 6u) {
        std::cerr
            << "Shader-table family regression: expected six reachable "
               "definitions, got "
            << reachable_definitions << '\n';
        return EXIT_FAILURE;
    }

    constexpr std::array table_data{
        // Five implicit-position RGBA samples.
        0.10f, 0.20f, 0.30f, 0.40f,
        0.30f, 0.10f, 0.50f, 0.60f,
        0.70f, 0.40f, 0.20f, 0.80f,
        0.20f, 0.90f, 0.60f, 0.50f,
        1.00f, 0.80f, 0.10f, 1.00f,
        // Four explicit position/RGBA control points.
        0.00f, 0.10f, 0.20f, 0.30f, 0.40f,
        0.20f, 0.90f, 0.10f, 0.20f, 0.80f,
        0.65f, 0.20f, 0.80f, 0.40f, 0.60f,
        1.00f, 0.70f, 0.60f, 0.90f, 1.00f,
        // Five implicit-position RGB curve samples.
        0.00f, 0.10f, 0.20f,
        0.15f, 0.30f, 0.25f,
        0.50f, 0.45f, 0.60f,
        0.80f, 0.75f, 0.70f,
        1.10f, 1.00f, 0.90f,
        // Four explicit position/RGB curve control points.
        -0.50f, -0.20f, -0.10f, 0.00f,
        0.00f, 0.10f, 0.20f, 0.30f,
        0.60f, 0.70f, 0.50f, 0.80f,
        1.50f, 1.20f, 1.10f, 1.00f};
    static_assert(table_data.size() == 71u);
    constexpr std::array factors{
        -0.25f, 0.0f, 0.125f, 0.25f,
        0.5f, 0.75f, 1.0f, 1.25f};
    constexpr std::array inputs{
        luisa::float3{-1.0f, -0.5f, 0.0f},
        luisa::float3{-0.5f, 0.0f, 0.5f},
        luisa::float3{0.0f, 0.25f, 0.75f},
        luisa::float3{0.1f, 0.5f, 1.0f},
        luisa::float3{0.25f, 0.75f, 1.25f},
        luisa::float3{0.5f, 1.0f, 1.5f},
        luisa::float3{0.75f, 1.5f, 2.0f},
        luisa::float3{2.0f, -1.0f, 0.33f}};

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto data = device.create_buffer<float>(table_data.size());
    auto factor_buffer = device.create_buffer<float>(case_count);
    auto input_buffer = device.create_buffer<luisa::float3>(case_count);
    auto output = device.create_buffer<luisa::float4>(
        case_count * result_stride);
    auto shader = device.compile(compare);
    std::vector<luisa::float4> actual(case_count * result_stride);
    stream << data.copy_from(luisa::span{table_data})
           << factor_buffer.copy_from(luisa::span{factors})
           << input_buffer.copy_from(luisa::span{inputs})
           << shader(
                  data,
                  factor_buffer,
                  input_buffer,
                  output)
                  .dispatch(case_count)
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
                    << "Shader-table callable mismatch on " << backend
                    << ", case " << case_index
                    << ", endpoint " << endpoint
                    << ": direct={" << direct.x << ", "
                    << direct.y << ", " << direct.z << ", "
                    << direct.w << "}, shared={" << shared.x << ", "
                    << shared.y << ", " << shared.z << ", "
                    << shared.w << "}\n";
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

#include "../src/luisa/path_tracer_vector_mapping.h"
#include "../src/luisa/surface_vector_mapping.h"

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

[[nodiscard]] Kernel1D<
    Buffer<luisa::float3>,
    Buffer<luisa::float3>>
make_repeated_point_mapping_kernel(bool shared) {
    constexpr auto repetition_count = std::uint32_t{8u};
    return [shared](
               BufferFloat3 input,
               BufferFloat3 output) noexcept {
        CallableSurfaceVectorMappingProvider provider;
        for (auto index = std::uint32_t{0u};
             index < repetition_count;
             ++index) {
            auto value = input.read(index);
            auto location = make_float3(
                0.11f * static_cast<float>(index),
                -0.07f * static_cast<float>(index),
                0.03f * static_cast<float>(index));
            auto rotation = make_float3(
                0.13f,
                0.29f + 0.01f * static_cast<float>(index),
                -0.47f);
            auto scale = make_float3(1.25f, 0.75f, 2.0f);
            output.write(
                index,
                shared
                    ? provider.map_point(
                          value, location, rotation, scale)
                    : map_vector_point_inline(
                          value, location, rotation, scale));
        }
    };
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    const auto inline_shape =
        make_repeated_point_mapping_kernel(false);
    const auto shared_shape =
        make_repeated_point_mapping_kernel(true);
    const auto inline_instructions =
        xir_instruction_count(inline_shape);
    const auto shared_instructions =
        xir_instruction_count(shared_shape);
    if (std::getenv("PSYCLES_REPORT_SHADER_SHAPES") != nullptr) {
        std::cout
            << "point mapping 8x XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
    }
    if (shared_instructions >= inline_instructions) {
        std::cerr
            << "Shared point mapping did not reduce repeated XIR: inline="
            << inline_instructions << ", shared="
            << shared_instructions << '\n';
        return EXIT_FAILURE;
    }

    // Recording one immutable mode must not attach the other three members
    // of the finite Mapping family to the module.
    Kernel1D used_only = [](BufferFloat3 output) noexcept {
        CallableSurfaceVectorMappingProvider provider;
        output.write(
            0u,
            provider.map_point(
                make_float3(0.25f, 0.5f, 0.75f),
                make_float3(0.1f, -0.2f, 0.3f),
                make_float3(0.2f, 0.4f, -0.1f),
                make_float3(1.0f, 2.0f, 0.5f)));
    };
    if (used_only.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Unused vector-mapping definitions reached the module\n";
        return EXIT_FAILURE;
    }

    // Independently recorded surface-operation factories must reuse the
    // identical endpoint through the complete callable AST hash.
    Kernel1D hash_reuse = [](BufferFloat3 output) noexcept {
        CallableSurfaceVectorMappingProvider first;
        CallableSurfaceVectorMappingProvider second;
        const auto location = make_float3(0.1f, -0.2f, 0.3f);
        const auto rotation = make_float3(0.2f, 0.4f, -0.1f);
        const auto scale = make_float3(1.0f, 2.0f, 0.5f);
        output.write(
            0u,
            first.map_point(
                make_float3(0.25f, 0.5f, 0.75f),
                location,
                rotation,
                scale));
        output.write(
            1u,
            second.map_point(
                make_float3(0.75f, 0.5f, 0.25f),
                location,
                rotation,
                scale));
    };
    if (hash_reuse.function()->function()
            .custom_callables()
            .size() != 1u) {
        std::cerr
            << "Independent vector-mapping callables were not "
               "deduplicated by complete hash\n";
        return EXIT_FAILURE;
    }

    Kernel1D compare = [](
                           BufferFloat3 input,
                           BufferFloat3 location,
                           BufferFloat3 rotation,
                           BufferFloat3 scale,
                           BufferFloat3 output) noexcept {
        const auto index = dispatch_x();
        const auto value = input.read(index);
        const auto translation = location.read(index);
        const auto angles = rotation.read(index);
        const auto factors = scale.read(index);
        const auto base = index * result_stride;
        CallableSurfaceVectorMappingProvider provider;
        output.write(
            base,
            map_vector_point_inline(
                value, translation, angles, factors));
        output.write(
            base + 1u,
            provider.map_point(
                value, translation, angles, factors));
        output.write(
            base + 2u,
            map_vector_texture_inline(
                value, translation, angles, factors));
        output.write(
            base + 3u,
            provider.map_texture(
                value, translation, angles, factors));
        output.write(
            base + 4u,
            map_vector_direction_inline(
                value, angles, factors));
        output.write(
            base + 5u,
            provider.map_vector(value, angles, factors));
        output.write(
            base + 6u,
            map_vector_normal_inline(
                value, angles, factors));
        output.write(
            base + 7u,
            provider.map_normal(value, angles, factors));
    };
    if (compare.function()->function()
            .custom_callables()
            .size() != 4u) {
        std::cerr
            << "Vector-mapping family regression: expected four reachable "
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
        luisa::float3{-1.0f, 2.0f, -3.0f},
        luisa::float3{0.13f, 0.47f, 0.89f},
        luisa::float3{10.0f, -0.25f, 4.0f},
        luisa::float3{-0.001f, 100.0f, 0.5f},
        luisa::float3{0.75f, 0.25f, 0.60f},
        luisa::float3{3.0f, -2.0f, 1.0f}};
    constexpr std::array locations{
        luisa::float3{0.0f, 0.0f, 0.0f},
        luisa::float3{0.1f, -0.2f, 0.3f},
        luisa::float3{-1.0f, 2.0f, -3.0f},
        luisa::float3{4.0f, -5.0f, 6.0f},
        luisa::float3{0.001f, 0.002f, -0.003f},
        luisa::float3{-10.0f, 0.25f, 4.0f},
        luisa::float3{2.0f, 2.0f, 2.0f},
        luisa::float3{-0.5f, 0.75f, -1.25f}};
    constexpr std::array rotations{
        luisa::float3{0.0f, 0.0f, 0.0f},
        luisa::float3{0.2f, 0.4f, -0.1f},
        luisa::float3{-1.0f, 0.5f, 2.0f},
        luisa::float3{3.14159265f, 0.0f, 0.0f},
        luisa::float3{0.0f, -1.57079633f, 0.0f},
        luisa::float3{0.3f, -0.7f, 1.1f},
        luisa::float3{-2.5f, 1.25f, 0.75f},
        luisa::float3{6.0f, -4.0f, 2.0f}};
    constexpr std::array scales{
        luisa::float3{1.0f, 1.0f, 1.0f},
        luisa::float3{1.0f, 2.0f, 0.5f},
        luisa::float3{-1.0f, 3.0f, -2.0f},
        luisa::float3{0.0f, 1.0f, 2.0f},
        luisa::float3{1.0f, 0.0f, 2.0f},
        luisa::float3{1.0f, 2.0f, 0.0f},
        luisa::float3{0.0f, 0.0f, 0.0f},
        luisa::float3{0.001f, 1000.0f, -0.25f}};
    auto input = device.create_buffer<luisa::float3>(input_count);
    auto location = device.create_buffer<luisa::float3>(input_count);
    auto rotation = device.create_buffer<luisa::float3>(input_count);
    auto scale = device.create_buffer<luisa::float3>(input_count);
    auto output = device.create_buffer<luisa::float3>(
        input_count * result_stride);
    auto shader = device.compile(compare);
    std::vector<luisa::float3> actual(
        input_count * result_stride);
    stream << input.copy_from(luisa::span{inputs})
           << location.copy_from(luisa::span{locations})
           << rotation.copy_from(luisa::span{rotations})
           << scale.copy_from(luisa::span{scales})
           << shader(input, location, rotation, scale, output)
                  .dispatch(input_count)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    for (auto input_index = std::uint32_t{0u};
         input_index < input_count;
         ++input_index) {
        const auto base = input_index * result_stride;
        for (auto mode = std::uint32_t{0u}; mode < 4u; ++mode) {
            const auto direct = actual[base + mode * 2u];
            const auto shared = actual[base + mode * 2u + 1u];
            if (!approximately_equal_float3(direct, shared)) {
                std::cerr
                    << "Vector-mapping mismatch on " << backend
                    << ", input " << input_index
                    << ", mode " << mode
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

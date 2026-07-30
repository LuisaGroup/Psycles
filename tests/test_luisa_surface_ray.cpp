#include <psycles/luisa/surface_ray.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace surface_ray = psycles::luisa_backend::surface_ray;

[[nodiscard]] bool equal_bits(float actual, float expected) noexcept {
    return std::bit_cast<std::uint32_t>(actual) ==
           std::bit_cast<std::uint32_t>(expected);
}

} // namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto results_buffer =
        device.create_buffer<luisa::float4>(4u);
    auto offsets_buffer =
        device.create_buffer<float>(3u);
    auto terminators_buffer =
        device.create_buffer<luisa::float4>(4u);
    auto neighboring_triangles_buffer =
        device.create_buffer<luisa::float4>(2u);

    Kernel1D evaluate =
        [](BufferFloat4 results,
           BufferFloat offsets,
           BufferFloat4 terminators,
           BufferFloat4 neighboring_triangles) noexcept {
            const auto index = dispatch_x();
            const auto world_origin =
                make_float3(32.0f, -17.0f, 0.25f);
            const auto world_normal =
                make_float3(0.6f, 0.0f, 0.8f);
            const auto object_direction =
                select(
                    make_float3(0.3f, -0.2f, 1.0f),
                    make_float3(-0.3f, 0.2f, -1.0f),
                    index == 1u);
            const auto object_origin =
                select(
                    make_float3(0.0f, 0.0f, 0.0f),
                    make_float3(1.25f, 0.0f, 0.0f),
                    index >= 2u);
            const auto p0 =
                make_float3(-1.0f, -1.0f, 0.0f);
            const auto p1 =
                make_float3(1.0f, -1.0f, 0.0f);
            const auto p2 =
                make_float3(0.0f, 1.0f, 0.0f);
            const auto covered =
                surface_ray::
                    source_triangle_covers_ray_origin(
                        object_origin,
                        object_direction,
                        p0,
                        p1,
                        p2);
            const auto origin =
                surface_ray::
                    origin_with_explicit_self_exclusion(
                        world_origin,
                        world_normal,
                        object_origin,
                        object_direction,
                        p0,
                        p1,
                        p2);
            const auto excluded =
                surface_ray::excluded_shadow_primitive(
                    7u,
                    11u,
                    select(7u, 3u, index == 3u),
                    11u,
                    surface_ray::invalid_primitive,
                    surface_ray::invalid_primitive);
            results.write(
                index,
                make_float4(
                    origin.x,
                    origin.y,
                    select(0.0f, 1.0f, covered),
                    select(0.0f, 1.0f, excluded)));
            $if (index < 3u) {
                const auto distance = select(
                    select(1.0f, 8.5f, index == 2u),
                    0.0f,
                    index == 0u);
                offsets.write(
                    index,
                    surface_ray::intersection_t_offset(
                        distance));
            };

            const auto curved_p0 =
                make_float3(-1.0f, -1.0f, 0.0f);
            const auto curved_p1 =
                make_float3(1.0f, -1.0f, 0.0f);
            const auto curved_p2 =
                make_float3(0.0f, 1.0f, 0.0f);
            const auto curved_n0 =
                make_float3(0.0f, 0.0f, 1.0f);
            const auto curved_n1 =
                make_float3(0.6f, 0.0f, 0.8f);
            const auto curved_n2 =
                make_float3(0.0f, 0.6f, 0.8f);
            const auto curved_normal =
                normalize(
                    (curved_n0 + curved_n1 + curved_n2) /
                    3.0f);
            const auto light_direction = normalize(select(
                make_float3(1.0f, 0.0f, 0.01f),
                make_float3(1.0f, 0.0f, -0.01f),
                index == 3u));
            const auto terminator =
                surface_ray::shadow_terminator_origin(
                    make_float3(
                        0.0f, -1.0f / 3.0f, 0.0f),
                    curved_normal,
                    make_float3(0.0f, 0.0f, 1.0f),
                    light_direction,
                    select(
                        0.1f,
                        0.0f,
                        index == 1u),
                    index != 0u,
                    make_float4x4(1.0f),
                    make_float2(
                        1.0f / 3.0f,
                        1.0f / 3.0f),
                    curved_p0,
                    curved_p1,
                    curved_p2,
                    curved_n0,
                    curved_n1,
                    curved_n2);
            terminators.write(
                index,
                make_float4(
                    terminator.position,
                    select(
                        0.0f,
                        1.0f,
                        terminator.skip_self)));

            $if (index < 2u) {
                const auto barycentric = select(
                    make_float2(
                        1.0f / 3.0f,
                        1.0f / 3.0f),
                    make_float2(0.5f, 0.0f),
                    index == 1u);
                const auto flat_position =
                    curved_p0 *
                        (1.0f -
                         barycentric.x -
                         barycentric.y) +
                    curved_p1 * barycentric.x +
                    curved_p2 * barycentric.y;
                const auto flat_normal =
                    make_float3(0.0f, 0.0f, 1.0f);
                const auto complete_origin =
                    surface_ray::surface_shadow_origin(
                        flat_position,
                        flat_normal,
                        flat_normal,
                        normalize(
                            make_float3(
                                0.3f, -0.2f, 1.0f)),
                        0.0f,
                        false,
                        make_float4x4(1.0f),
                        make_float4x4(1.0f),
                        barycentric,
                        curved_p0,
                        curved_p1,
                        curved_p2,
                        flat_normal,
                        flat_normal,
                        flat_normal);
                neighboring_triangles.write(
                    index,
                    make_float4(
                        complete_origin.position,
                        select(
                            0.0f,
                            1.0f,
                            complete_origin.skip_self)));
            };
        };
    auto shader = device.compile(evaluate);
    std::array<luisa::float4, 4u> results{};
    std::array<float, 3u> offsets{};
    std::array<luisa::float4, 4u> terminators{};
    std::array<luisa::float4, 2u>
        neighboring_triangles{};
    stream << shader(
                  results_buffer,
                  offsets_buffer,
                  terminators_buffer,
                  neighboring_triangles_buffer)
                  .dispatch(4u)
           << results_buffer.copy_to(luisa::span{results})
           << offsets_buffer.copy_to(luisa::span{offsets})
           << terminators_buffer.copy_to(
                  luisa::span{terminators})
           << neighboring_triangles_buffer.copy_to(
                  luisa::span{neighboring_triangles})
           << synchronize();

    for (auto index = 0u; index < 2u; ++index) {
        if (!equal_bits(results[index].x, 32.0f) ||
            !equal_bits(results[index].y, -17.0f) ||
            results[index].z != 1.0f ||
            results[index].w != 1.0f) {
            std::cerr
                << "certified source-triangle origin was displaced on "
                << backend << " for orientation " << index << ": {"
                << results[index].x << ", " << results[index].y
                << ", " << results[index].z << ", "
                << results[index].w << "}\n";
            return EXIT_FAILURE;
        }
    }
    for (auto index = 2u; index < 4u; ++index) {
        const auto displaced =
            !equal_bits(results[index].x, 32.0f) ||
            !equal_bits(results[index].y, -17.0f);
        const auto expected_excluded =
            index == 2u ? 1.0f : 0.0f;
        if (!displaced ||
            results[index].z != 0.0f ||
            results[index].w != expected_excluded) {
            std::cerr
                << "ambiguous/outside origin did not retain robust "
                   "offset or primitive exclusion on "
                << backend << " for case " << index << ": {"
                << results[index].x << ", " << results[index].y
                << ", " << results[index].z << ", "
                << results[index].w << "}\n";
            return EXIT_FAILURE;
        }
    }
    const std::array expected_offsets{
        std::numeric_limits<float>::min(),
        std::nextafter(
            1.0f,
            std::numeric_limits<float>::max()),
        std::nextafter(
            8.5f,
            std::numeric_limits<float>::max())};
    for (auto index = 0u;
         index < expected_offsets.size();
         ++index) {
        if (!equal_bits(
                offsets[index],
                expected_offsets[index])) {
            std::cerr
                << "transparent tmin offset did not match "
                   "Cycles bit-for-bit on "
                << backend << " for case " << index
                << ": got 0x" << std::hex
                << std::bit_cast<std::uint32_t>(
                       offsets[index])
                << ", expected 0x"
                << std::bit_cast<std::uint32_t>(
                       expected_offsets[index])
                << std::dec << '\n';
            return EXIT_FAILURE;
        }
    }

    const auto unchanged =
        luisa::make_float3(
            0.0f, -1.0f / 3.0f, 0.0f);
    for (auto index = 0u; index < 2u; ++index) {
        if (!equal_bits(terminators[index].x, unchanged.x) ||
            !equal_bits(terminators[index].y, unchanged.y) ||
            !equal_bits(terminators[index].z, unchanged.z) ||
            terminators[index].w != 1.0f) {
            std::cerr
                << "disabled shadow-terminator offset changed the "
                   "origin or self-exclusion state on "
                << backend << " for case " << index << '\n';
            return EXIT_FAILURE;
        }
    }
    const std::array expected_terminators{
        luisa::make_float4(
            0.04200023f,
            -0.29133311f,
            0.18200099f,
            1.0f),
        luisa::make_float4(
            0.04666667f,
            -0.28666666f,
            0.20222223f,
            0.0f)};
    for (auto expected_index = 0u;
         expected_index < expected_terminators.size();
         ++expected_index) {
        const auto actual =
            terminators[expected_index + 2u];
        const auto expected =
            expected_terminators[expected_index];
        const auto error =
            std::max(
                std::abs(actual.x - expected.x),
                std::max(
                    std::abs(actual.y - expected.y),
                    std::abs(actual.z - expected.z)));
        if (error > 2.0e-6f || actual.w != expected.w) {
            std::cerr
                << "Cycles shadow-terminator construction mismatch on "
                << backend << " for case " << expected_index
                << ": {" << actual.x << ", " << actual.y
                << ", " << actual.z << ", " << actual.w
                << "}\n";
            return EXIT_FAILURE;
        }
    }
    const auto interior = neighboring_triangles[0u];
    if (std::abs(interior.x) > 1.0e-6f ||
        std::abs(interior.y + 1.0f / 3.0f) >
            1.0e-6f ||
        !equal_bits(interior.z, 0.0f) ||
        interior.w != 1.0f) {
        std::cerr
            << "certified flat-triangle shadow origin changed on "
            << backend << ": {" << interior.x << ", "
            << interior.y << ", " << interior.z << ", "
            << interior.w << "}\n";
        return EXIT_FAILURE;
    }
    const auto shared_edge = neighboring_triangles[1u];
    if (std::abs(shared_edge.x) > 1.0e-6f ||
        std::abs(shared_edge.y + 1.0f) >
            1.0e-6f ||
        !(shared_edge.z > 0.0f) ||
        shared_edge.w != 1.0f) {
        std::cerr
            << "shared-edge shadow origin did not receive Cycles' "
               "neighboring-triangle offset on "
            << backend << ": {" << shared_edge.x << ", "
            << shared_edge.y << ", " << shared_edge.z << ", "
            << shared_edge.w << "}\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

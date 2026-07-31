#include <psycles/luisa/triangle_light_sampling.h>
#include <psycles/luisa/volume_light_interval.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 3.0e-5f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

[[nodiscard]] bool approximately_equal(
    luisa::float4 actual,
    luisa::float4 expected,
    float tolerance = 3.0e-5f) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance) &&
           approximately_equal(actual.w, expected.w, tolerance);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{
            argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    constexpr auto record_count = 10u;
    auto output =
        device.create_buffer<luisa::float4>(
            record_count);

    Kernel1D evaluate =
        [](BufferFloat4 records) noexcept {
            const auto flag =
                [](Bool value) noexcept {
                    return select(
                        0.0f, 1.0f, value);
                };
            TriangleLightSampling sampling;
            const TriangleLightSampleInput near{
                .reference = make_float3(0.0f),
                .p0 = make_float3(-2.0f, -2.0f, 1.0f),
                .p1 = make_float3(2.0f, -2.0f, 1.0f),
                .p2 = make_float3(0.0f, 2.0f, 1.0f),
                .random = make_float2(0.1f, 0.8f)};
            const auto segment =
                sampling.from_segment(near);
            const auto collision =
                sampling.from_position(near);
            const auto forward =
                sampling.from_intersection(
                    near,
                    collision.position);
            records.write(
                0u,
                make_float4(
                    segment.barycentric,
                    segment.conditional_pdf,
                    flag(segment.valid)));
            records.write(
                1u,
                make_float4(
                    segment.position,
                    flag(
                        segment.uses_solid_angle)));
            records.write(
                2u,
                make_float4(
                    collision.barycentric,
                    collision.conditional_pdf,
                    flag(
                        collision.uses_solid_angle)));
            records.write(
                3u,
                make_float4(
                    forward.value,
                    flag(
                        forward.uses_solid_angle),
                    flag(forward.valid),
                    collision.conditional_pdf));

            auto far = near;
            far.p0.z = 100.0f;
            far.p1.z = 100.0f;
            far.p2.z = 100.0f;
            const auto far_segment =
                sampling.from_segment(far);
            const auto far_collision =
                sampling.from_position(far);
            records.write(
                4u,
                make_float4(
                    far_segment.barycentric,
                    far_segment.conditional_pdf,
                    flag(
                        far_segment.uses_solid_angle)));
            records.write(
                5u,
                make_float4(
                    far_collision.barycentric,
                    far_collision.conditional_pdf,
                    flag(
                        far_collision.uses_solid_angle)));

            VolumeLightInterval interval;
            const auto write_interval =
                [&](std::uint32_t index,
                    Bool front,
                    Bool back) noexcept {
                    const auto result =
                        interval.triangle(
                            {.ray_origin =
                                 make_float3(
                                     0.0f,
                                     0.0f,
                                     -2.0f),
                             .ray_direction =
                                 make_float3(
                                     0.0f,
                                     0.0f,
                                     1.0f),
                             .interval =
                                 {.minimum = 0.0f,
                                  .maximum = 5.0f},
                             .plane_point =
                                 make_float3(
                                     0.0f,
                                     0.0f,
                                     1.0f),
                             .normal =
                                 make_float3(
                                     0.0f,
                                     0.0f,
                                     1.0f),
                             .sample_front =
                                 front,
                             .sample_back =
                                 back});
                    records.write(
                        index,
                        make_float4(
                            select(
                                0.0f,
                                result.interval.minimum,
                                result.valid),
                            select(
                                0.0f,
                                result.interval.maximum,
                                result.valid),
                            flag(result.valid),
                            0.0f));
                };
            write_interval(6u, true, false);
            write_interval(7u, false, true);
            write_interval(8u, true, true);
            write_interval(9u, false, false);
        };

    auto shader = device.compile(evaluate);
    std::array<luisa::float4, record_count>
        actual{};
    stream
        << shader(output).dispatch(1u)
        << output.copy_to(luisa::span{actual})
        << synchronize();

    // Exact invariants from Cycles main b82c3f0 triangle.h:
    // - triangle_light_sample<true> always uses the low-distortion area map;
    // - triangle_light_sample<false> uses solid angle only for the near case;
    // - triangle_light_valid_ray_segment keeps the emitting plane half-space.
    const auto segment_distance_squared =
        0.3f * 0.3f + 1.0f + 1.0f;
    const auto segment_pdf =
        segment_distance_squared *
        std::sqrt(segment_distance_squared) /
        8.0f;
    const auto far_distance_squared =
        0.3f * 0.3f + 1.0f +
        100.0f * 100.0f;
    const auto far_pdf =
        far_distance_squared *
        std::sqrt(far_distance_squared) /
        (8.0f * 100.0f);
    const std::array expected{
        luisa::float4{
            0.05f,
            0.75f,
            segment_pdf,
            1.0f},
        luisa::float4{
            -0.3f,
            1.0f,
            1.0f,
            0.0f},
        // The collision barycentrics and solid-angle PDF are checked through
        // their formal domain and forward-measure equality below.
        actual[2u],
        actual[3u],
        luisa::float4{
            0.05f,
            0.75f,
            far_pdf,
            0.0f},
        luisa::float4{
            0.05f,
            0.75f,
            far_pdf,
            0.0f},
        luisa::float4{
            3.0f, 5.0f, 1.0f, 0.0f},
        luisa::float4{
            0.0f, 3.0f, 1.0f, 0.0f},
        luisa::float4{
            0.0f, 5.0f, 1.0f, 0.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 0.0f}};

    auto passed = true;
    for (auto index = std::size_t{0u};
         index < expected.size();
         ++index) {
        if (!approximately_equal(
                actual[index],
                expected[index])) {
            std::cerr
                << "Cycles triangle-light fixture failed on "
                << backend << " at record " << index
                << ": got {" << actual[index].x
                << ", " << actual[index].y
                << ", " << actual[index].z
                << ", " << actual[index].w
                << "}, expected {"
                << expected[index].x << ", "
                << expected[index].y << ", "
                << expected[index].z << ", "
                << expected[index].w << "}\n";
            passed = false;
        }
    }
    const auto collision_barycentric =
        actual[2u];
    const auto collision_valid =
        collision_barycentric.x >= -1.0e-5f &&
        collision_barycentric.y >= -1.0e-5f &&
        collision_barycentric.x +
                collision_barycentric.y <=
            1.0f + 1.0e-5f &&
        collision_barycentric.z > 0.0f &&
        collision_barycentric.w > 0.5f;
    const auto forward_matches =
        actual[3u].y > 0.5f &&
        actual[3u].z > 0.5f &&
        approximately_equal(
            actual[3u].x,
            actual[3u].w);
    if (!collision_valid ||
        !forward_matches) {
        std::cerr
            << "Cycles triangle-light collision/forward measure "
               "invariant failed on "
            << backend << '\n';
        passed = false;
    }
    return passed
               ? EXIT_SUCCESS
               : EXIT_FAILURE;
}

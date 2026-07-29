#include <psycles/luisa/spherical_geometry.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
namespace spherical_geometry = psycles::luisa_backend::spherical_geometry;

constexpr std::array radii{1.0e-7f,
                           1.0e-4f,
                           0.5f * 0.01745329238474369f,
                           0.1f,
                           0.5f * spherical_geometry::pi};

constexpr std::array triangle_random{luisa::float2{0.1f, 0.8f},
                                     luisa::float2{0.75f, 0.2f},
                                     luisa::float2{0.25f, 0.75f},
                                     luisa::float2{0.8f, 0.3f}};

[[nodiscard]] bool approximately_equal(float actual,
                                       double expected,
                                       double relative_tolerance) noexcept {
    const auto absolute_error =
        std::abs(static_cast<double>(actual) - expected);
    return absolute_error <=
           relative_tolerance * std::max(std::abs(expected), 1.0e-30);
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto radius_buffer = device.create_buffer<float>(radii.size());
    auto result_buffer = device.create_buffer<luisa::float2>(radii.size());

    Kernel1D evaluate = [](BufferFloat radii, BufferFloat2 results) noexcept {
        const auto radius = radii.read(dispatch_x());
        results.write(dispatch_x(),
                      make_float2(spherical_geometry::unit_cap_height(radius),
                                  spherical_geometry::cap_solid_angle(radius)));
    };
    auto shader = device.compile(evaluate);
    std::array<luisa::float2, radii.size()> results{};
    stream << radius_buffer.copy_from(luisa::span{radii})
           << shader(radius_buffer, result_buffer)
                  .dispatch(static_cast<std::uint32_t>(radii.size()))
           << result_buffer.copy_to(luisa::span{results}) << synchronize();

    for (std::size_t i = 0u; i < radii.size(); ++i) {
        const auto radius = static_cast<double>(radii[i]);
        const auto sine_half = std::sin(0.5 * radius);
        const auto expected_height = 2.0 * sine_half * sine_half;
        const auto expected_solid_angle =
            2.0 * static_cast<double>(spherical_geometry::pi) * expected_height;
        if (!approximately_equal(results[i].x, expected_height, 2.0e-5) ||
            !approximately_equal(results[i].y, expected_solid_angle, 2.0e-5)) {
            std::cerr << "spherical-cap invariant failed on " << backend
                      << " at radius " << radii[i] << ": got {" << results[i].x
                      << ", " << results[i].y << "}, expected {"
                      << expected_height << ", " << expected_solid_angle
                      << "}\n";
            return EXIT_FAILURE;
        }
    }

    auto random_buffer =
        device.create_buffer<luisa::float2>(triangle_random.size());
    auto sample_a_buffer =
        device.create_buffer<luisa::float4>(triangle_random.size());
    auto sample_b_buffer =
        device.create_buffer<luisa::float4>(triangle_random.size());
    auto sample_c_buffer =
        device.create_buffer<luisa::float4>(triangle_random.size());
    auto sample_d_buffer =
        device.create_buffer<luisa::float4>(triangle_random.size());
    Kernel1D sample_triangle = [](BufferFloat2 randoms,
                                  BufferFloat4 sample_a,
                                  BufferFloat4 sample_b,
                                  BufferFloat4 sample_c,
                                  BufferFloat4 sample_d) noexcept {
        const auto index = dispatch_x();
        const auto near = index < 2u;
        const auto z = select(100.0f, 1.0f, near);
        const auto reference = make_float3(0.0f, 0.0f, 0.0f);
        const auto p0 = make_float3(-2.0f, -2.0f, z);
        const auto p1 = make_float3(2.0f, -2.0f, z);
        const auto p2 = make_float3(0.0f, 2.0f, z);
        const auto result = spherical_geometry::sample_triangle(
            reference, p0, p1, p2, randoms.read(index));
        const auto forward = spherical_geometry::triangle_directional_pdf(
            reference, result.position, p0, p1, p2);
        sample_a.write(index,
                       make_float4(result.barycentric,
                                   result.distance,
                                   result.conditional_pdf));
        sample_b.write(
            index,
            make_float4(result.position, select(0.0f, 1.0f, result.valid)));
        sample_c.write(
            index,
            make_float4(result.direction,
                        select(0.0f, 1.0f, result.uses_solid_angle)));
        sample_d.write(index,
                       make_float4(forward.value,
                                   select(0.0f, 1.0f, forward.uses_solid_angle),
                                   select(0.0f, 1.0f, forward.valid),
                                   spherical_geometry::triangle_solid_angle(
                                       reference, p0, p1, p2)));
    };
    auto triangle_shader = device.compile(sample_triangle);
    std::array<luisa::float4, triangle_random.size()> sample_a{};
    std::array<luisa::float4, triangle_random.size()> sample_b{};
    std::array<luisa::float4, triangle_random.size()> sample_c{};
    std::array<luisa::float4, triangle_random.size()> sample_d{};
    stream << random_buffer.copy_from(luisa::span{triangle_random})
           << triangle_shader(random_buffer,
                              sample_a_buffer,
                              sample_b_buffer,
                              sample_c_buffer,
                              sample_d_buffer)
                  .dispatch(static_cast<std::uint32_t>(triangle_random.size()))
           << sample_a_buffer.copy_to(luisa::span{sample_a})
           << sample_b_buffer.copy_to(luisa::span{sample_b})
           << sample_c_buffer.copy_to(luisa::span{sample_c})
           << sample_d_buffer.copy_to(luisa::span{sample_d}) << synchronize();

    for (std::size_t i = 0u; i < triangle_random.size(); ++i) {
        const auto expected_solid_angle_branch = i < 2u;
        const auto &a = sample_a[i];
        const auto &b = sample_b[i];
        const auto &c = sample_c[i];
        const auto &d = sample_d[i];
        const auto reconstructed =
            luisa::float3{(1.0f - a.x - a.y) * -2.0f + a.x * 2.0f,
                          (1.0f - a.x - a.y) * -2.0f + a.x * -2.0f + a.y * 2.0f,
                          i < 2u ? 1.0f : 100.0f};
        const auto direction_length =
            std::sqrt(c.x * c.x + c.y * c.y + c.z * c.z);
        const auto position_error = std::max({std::abs(reconstructed.x - b.x),
                                              std::abs(reconstructed.y - b.y),
                                              std::abs(reconstructed.z - b.z)});
        const auto branch = c.w > 0.5f && d.y > 0.5f;
        const auto valid = b.w > 0.5f && d.z > 0.5f;
        if (!valid || branch != expected_solid_angle_branch || a.x < -1.0e-4f ||
            a.y < -1.0e-4f || a.x + a.y > 1.0f + 1.0e-4f ||
            std::abs(direction_length - 1.0f) > 2.0e-5f ||
            position_error > 2.0e-4f ||
            !approximately_equal(a.w, d.x, 3.0e-5) || !(a.w > 0.0f) ||
            !(d.w > 0.0f)) {
            std::cerr << "spherical-triangle invariant failed on " << backend
                      << " for sample " << i << ": barycentric {" << a.x << ", "
                      << a.y << "}, distance/pdf {" << a.z << ", " << a.w
                      << "}, position {" << b.x << ", " << b.y << ", " << b.z
                      << "}, direction/branch {" << c.x << ", " << c.y << ", "
                      << c.z << ", " << c.w << "}, forward {" << d.x << ", "
                      << d.y << ", " << d.z << ", " << d.w << "}\n";
            return EXIT_FAILURE;
        }
    }

    // The far samples exercise the area branch and therefore must preserve
    // Cycles' low-distortion square-to-triangle map exactly.
    const auto expected_far_0 = luisa::float2{0.125f, 0.625f};
    const auto expected_far_1 = luisa::float2{0.65f, 0.15f};
    if (!approximately_equal(sample_a[2u].x, expected_far_0.x, 1.0e-6) ||
        !approximately_equal(sample_a[2u].y, expected_far_0.y, 1.0e-6) ||
        !approximately_equal(sample_a[3u].x, expected_far_1.x, 1.0e-6) ||
        !approximately_equal(sample_a[3u].y, expected_far_1.y, 1.0e-6)) {
        std::cerr << "low-distortion triangle map failed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

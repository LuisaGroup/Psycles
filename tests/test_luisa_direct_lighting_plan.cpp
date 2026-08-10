#include "path_kernel_builder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using psycles::luisa_backend::detail::DirectLightingStagePlan;
using psycles::luisa_backend::detail::make_direct_lighting_stage_plan;
using psycles::luisa_backend::detail::make_path_kernel_scene_stage_plan;

struct PlanCase {
    bool enabled{};
    bool environment{};
    std::uint32_t emissive_meshes{};
    std::uint32_t analytic_lights{};
    std::size_t triangle_geometries{};
    std::size_t curve_geometries{};
};

inline constexpr auto environment_bit = std::uint32_t{1u} << 0u;
inline constexpr auto emissive_mesh_bit = std::uint32_t{1u} << 1u;
inline constexpr auto analytic_nee_bit = std::uint32_t{1u} << 2u;
inline constexpr auto analytic_endpoint_bit = std::uint32_t{1u} << 3u;
inline constexpr auto triangle_primitive_bit = std::uint32_t{1u} << 4u;
inline constexpr auto curve_primitive_bit = std::uint32_t{1u} << 5u;
inline constexpr auto direct_lighting_bits =
    environment_bit |
    emissive_mesh_bit |
    analytic_nee_bit;

[[nodiscard]] constexpr std::uint32_t
plan_mask(DirectLightingStagePlan plan) noexcept {
    return (plan.environment ? environment_bit : 0u) |
           (plan.emissive_mesh ? emissive_mesh_bit : 0u) |
           (plan.analytic ? analytic_nee_bit : 0u);
}

[[nodiscard]] constexpr std::uint32_t
scene_stage_mask(const PlanCase &test) noexcept {
    const auto plan = make_path_kernel_scene_stage_plan(
        test.enabled,
        test.environment,
        test.emissive_meshes,
        test.analytic_lights,
        test.triangle_geometries,
        test.curve_geometries);
    return plan_mask(plan.direct_lighting) |
           (plan.analytic_light_endpoints
                ? analytic_endpoint_bit
                : 0u) |
           (plan.primitives.triangles
                ? triangle_primitive_bit
                : 0u) |
           (plan.primitives.curves
                ? curve_primitive_bit
                : 0u);
}

constexpr auto cases =
    std::array{PlanCase{},
               PlanCase{.enabled = false,
                        .environment = true,
                        .emissive_meshes = 3u,
                        .analytic_lights = 4u},
               PlanCase{.enabled = true},
               PlanCase{.enabled = true, .environment = true},
               PlanCase{.enabled = true, .emissive_meshes = 1u},
               PlanCase{.enabled = true, .analytic_lights = 1u},
               PlanCase{.enabled = true,
                        .environment = true,
                        .emissive_meshes = 7u,
                        .analytic_lights = 0u},
               PlanCase{.enabled = true,
                        .environment = true,
                        .emissive_meshes = 7u,
                        .analytic_lights = 9u},
               PlanCase{.triangle_geometries = 1u},
               PlanCase{.curve_geometries = 1u},
               PlanCase{.triangle_geometries = 5u,
                        .curve_geometries = 8u}};

[[nodiscard]] constexpr std::uint32_t
expected_mask(const PlanCase &test) noexcept {
    const auto direct =
        test.enabled
            ? (test.environment ? environment_bit : 0u) |
                  (test.emissive_meshes != 0u
                       ? emissive_mesh_bit
                       : 0u) |
                  (test.analytic_lights != 0u
                       ? analytic_nee_bit
                       : 0u)
            : 0u;
    return direct |
           (test.analytic_lights != 0u
                ? analytic_endpoint_bit
                : 0u) |
           (test.triangle_geometries != 0u
                ? triangle_primitive_bit
                : 0u) |
           (test.curve_geometries != 0u
                ? curve_primitive_bit
                : 0u);
}

static_assert(
    plan_mask(make_direct_lighting_stage_plan(true, true, 7u, 0u)) ==
        (environment_bit | emissive_mesh_bit),
    "an absent analytic-light population must not reach the path kernel");
static_assert(make_direct_lighting_stage_plan(false, true, 7u, 9u).size() == 0u,
              "disabled NEE must record no direct-light component");
static_assert(
    make_path_kernel_scene_stage_plan(false, false, 0u, 1u, 0u, 0u)
        .analytic_light_endpoints,
    "forward analytic-light endpoints do not depend on NEE");
static_assert(
    !make_path_kernel_scene_stage_plan(true, true, 7u, 0u, 0u, 0u)
         .analytic_light_endpoints,
    "a zero analytic population must prune endpoint intersection and shading");
static_assert(
    make_path_kernel_scene_stage_plan(false, false, 0u, 0u, 3u, 0u)
            .primitives.triangles &&
        !make_path_kernel_scene_stage_plan(false, false, 0u, 0u, 3u, 0u)
             .primitives.curves,
    "triangle-only scenes must not record curve stages");
static_assert(
    make_path_kernel_scene_stage_plan(false, false, 0u, 0u, 0u, 2u)
            .primitives.curves &&
        !make_path_kernel_scene_stage_plan(false, false, 0u, 0u, 0u, 2u)
             .primitives.triangles,
    "curve-only scenes must not record triangle stages");

}// namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};

    std::array<std::uint32_t, cases.size()> expected{};
    for (auto index = std::size_t{0u}; index < cases.size(); ++index) {
        const auto &test = cases[index];
        const auto plan = make_direct_lighting_stage_plan(
            test.enabled, test.environment, test.emissive_meshes,
            test.analytic_lights);
        expected[index] = expected_mask(test);
        if (scene_stage_mask(test) != expected[index] ||
            plan.size() !=
                static_cast<std::size_t>(
                    std::popcount(
                        expected[index] &
                        direct_lighting_bits))) {
            std::cerr << "Path-kernel scene plan mismatch at case " << index
                      << '\n';
            return EXIT_FAILURE;
        }
    }

    Kernel1D write_plan = [](BufferUInt output) noexcept {
        for (auto index = std::size_t{0u}; index < cases.size(); ++index) {
            const auto test = cases[index];
            output.write(
                static_cast<std::uint32_t>(index),
                scene_stage_mask(test));
        }
    };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<std::uint32_t>(cases.size());
    auto shader = device.compile(write_plan);
    std::vector<std::uint32_t> actual(cases.size());
    stream << shader(output).dispatch(1u) << output.copy_to(luisa::span{actual})
           << synchronize();
    if (!std::equal(actual.begin(), actual.end(), expected.begin())) {
        std::cerr << "Device-visible direct-light plan changed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

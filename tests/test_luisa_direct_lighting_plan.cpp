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

struct PlanCase {
    bool enabled{};
    bool environment{};
    std::uint32_t emissive_meshes{};
    std::uint32_t analytic_lights{};
};

[[nodiscard]] constexpr std::uint32_t
plan_mask(DirectLightingStagePlan plan) noexcept {
    constexpr auto environment_bit = std::uint32_t{1u} << 0u;
    constexpr auto emissive_mesh_bit = std::uint32_t{1u} << 1u;
    constexpr auto analytic_bit = std::uint32_t{1u} << 2u;
    return (plan.environment ? environment_bit : 0u) |
           (plan.emissive_mesh ? emissive_mesh_bit : 0u) |
           (plan.analytic ? analytic_bit : 0u);
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
                        .analytic_lights = 9u}};

[[nodiscard]] constexpr std::uint32_t
expected_mask(const PlanCase &test) noexcept {
    if (!test.enabled) {
        return 0u;
    }
    return (test.environment ? 1u : 0u) | (test.emissive_meshes != 0u ? 2u : 0u) |
           (test.analytic_lights != 0u ? 4u : 0u);
}

static_assert(
    plan_mask(make_direct_lighting_stage_plan(true, true, 7u, 0u)) == 3u,
    "an absent analytic-light population must not reach the path kernel");
static_assert(make_direct_lighting_stage_plan(false, true, 7u, 9u).size() == 0u,
              "disabled NEE must record no direct-light component");

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
        if (plan_mask(plan) != expected[index] ||
            plan.size() !=
                static_cast<std::size_t>(std::popcount(expected[index]))) {
            std::cerr << "Direct-light host plan mismatch at case " << index << '\n';
            return EXIT_FAILURE;
        }
    }

    Kernel1D write_plan = [](BufferUInt output) noexcept {
        for (auto index = std::size_t{0u}; index < cases.size(); ++index) {
            const auto test = cases[index];
            const auto plan = make_direct_lighting_stage_plan(
                test.enabled, test.environment, test.emissive_meshes,
                test.analytic_lights);
            output.write(static_cast<std::uint32_t>(index), plan_mask(plan));
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

#include "path_kernel_builder.h"
#include "path_kernel_direct_light_queue.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using psycles::luisa_backend::detail::can_stage_direct_light_queue;
using psycles::luisa_backend::detail::DirectLightSampleState;
using psycles::luisa_backend::detail::DirectLightingStagePlan;
using psycles::luisa_backend::detail::finalize_direct_light_sample;
using psycles::luisa_backend::detail::LightSampleRouletteCallable;
using psycles::luisa_backend::detail::make_direct_lighting_stage_plan;
using psycles::luisa_backend::detail::make_path_kernel_scene_stage_plan;
using psycles::luisa_backend::detail::make_runtime_direct_light_task_storage;
using psycles::luisa_backend::detail::SceneTraversalStagePlan;

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
inline constexpr auto direct_transport_bit = std::uint32_t{1u} << 7u;
inline constexpr auto direct_lighting_bits =
    environment_bit | emissive_mesh_bit | analytic_nee_bit;

[[nodiscard]] constexpr std::uint32_t
plan_mask(DirectLightingStagePlan plan) noexcept {
    return (plan.environment ? environment_bit : 0u) |
           (plan.emissive_mesh ? emissive_mesh_bit : 0u) |
           (plan.analytic ? analytic_nee_bit : 0u) |
         (plan.transport_stage_count() != 0u ? direct_transport_bit : 0u);
}

[[nodiscard]] constexpr std::uint32_t
scene_stage_mask(const PlanCase &test) noexcept {
    const auto plan = make_path_kernel_scene_stage_plan(
      test.enabled, test.environment, test.emissive_meshes,
      test.analytic_lights, test.triangle_geometries,
      test.curve_geometries);
    const auto traversal = plan.traversal;
    return plan_mask(plan.direct_lighting) |
         (plan.analytic_light_endpoints ? analytic_endpoint_bit : 0u) |
         (traversal.primitives.triangles ? triangle_primitive_bit : 0u) |
         (traversal.primitives.curves ? curve_primitive_bit : 0u);
}

constexpr auto cases = std::array{
    PlanCase{},
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
    PlanCase{.triangle_geometries = 5u, .curve_geometries = 8u}};

[[nodiscard]] constexpr std::uint32_t
expected_mask(const PlanCase &test) noexcept {
    const auto direct =
      test.enabled ? (test.environment ? environment_bit : 0u) |
                         (test.emissive_meshes != 0u ? emissive_mesh_bit : 0u) |
                         (test.analytic_lights != 0u ? analytic_nee_bit : 0u)
            : 0u;
  return direct | (direct != 0u ? direct_transport_bit : 0u) |
         (test.analytic_lights != 0u ? analytic_endpoint_bit : 0u) |
         (test.triangle_geometries != 0u ? triangle_primitive_bit : 0u) |
         (test.curve_geometries != 0u ? curve_primitive_bit : 0u);
}

static_assert(
    plan_mask(make_direct_lighting_stage_plan(true, true, 7u, 0u)) ==
        (environment_bit | emissive_mesh_bit | direct_transport_bit),
    "an absent analytic-light population must not reach the path kernel");
static_assert(make_direct_lighting_stage_plan(false, true, 7u, 9u).size() == 0u,
              "disabled NEE must record no direct-light component");
static_assert(
    make_direct_lighting_stage_plan(true, true, 7u, 9u)
            .transport_stage_count() == 1u,
    "all reachable emitter kinds must share one transport continuation");
static_assert(
    make_direct_lighting_stage_plan(false, true, 7u, 9u)
            .transport_stage_count() == 0u,
    "an empty proposal plan must not record a transport continuation");
static_assert(make_path_kernel_scene_stage_plan(false, false, 0u, 1u, 0u, 0u)
        .analytic_light_endpoints,
    "forward analytic-light endpoints do not depend on NEE");
static_assert(
    !make_path_kernel_scene_stage_plan(true, true, 7u, 0u, 0u, 0u)
         .analytic_light_endpoints,
    "a zero analytic population must prune endpoint intersection and shading");
static_assert(make_path_kernel_scene_stage_plan(false, false, 0u, 0u, 3u, 0u)
            .traversal.primitives.triangles &&
                  !make_path_kernel_scene_stage_plan(false, false, 0u, 0u, 3u,
                                                     0u)
             .traversal.primitives.curves,
    "triangle-only scenes must not record curve stages");
static_assert(make_path_kernel_scene_stage_plan(false, false, 0u, 0u, 0u, 2u)
            .traversal.primitives.curves &&
                  !make_path_kernel_scene_stage_plan(false, false, 0u, 0u, 0u,
                                                     2u)
             .traversal.primitives.triangles,
    "curve-only scenes must not record triangle stages");
constexpr auto stageable_direct_light =
    make_path_kernel_scene_stage_plan(true, false, 0u, 1u, 1u, 0u);
static_assert(can_stage_direct_light_queue(true, false,
                                           stageable_direct_light));
static_assert(!can_stage_direct_light_queue(false, false,
                                            stageable_direct_light));
static_assert(!can_stage_direct_light_queue(true, true,
                                            stageable_direct_light));
static_assert(!can_stage_direct_light_queue(
    true, false,
    make_path_kernel_scene_stage_plan(true, false, 0u, 1u, 0u, 0u)));
static_assert(!can_stage_direct_light_queue(
    true, false,
    make_path_kernel_scene_stage_plan(true, false, 0u, 0u, 1u, 0u)));

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
        plan.size() != static_cast<std::size_t>(std::popcount(
                           expected[index] & direct_lighting_bits))) {
      std::cerr << "Path-kernel scene plan mismatch at case " << index << '\n';
            return EXIT_FAILURE;
        }
    }

    Kernel1D write_plan = [](BufferUInt output) noexcept {
        for (auto index = std::size_t{0u}; index < cases.size(); ++index) {
            const auto test = cases[index];
      output.write(static_cast<std::uint32_t>(index), scene_stage_mask(test));
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

    // Three equality predicates over one emitter-kind discriminant are
    // pairwise disjoint. They may therefore populate one typed LightSample-
    // equivalent state before a single post-merge receiving evaluator. The
    // host counter is structural: it proves the evaluator body was recorded
    // once, rather than once in every proposal arm.
    std::uint32_t recorded_receiving_evaluators = 0u;
    Kernel1D write_merged_light_sample =
        [&recorded_receiving_evaluators](BufferFloat4 values) noexcept {
            const auto kind = dispatch_x();
            auto light = DirectLightSampleState::empty();
            const auto accept = [&](Float3 direction,
                                    std::uint32_t shader_flags) noexcept {
                light.accept(
                    {.direction = direction,
                     .target_position = direction,
                     .light_normal = -direction,
                     .light_uv = make_float2(0.0f),
                     .barycentric = make_float2(0.0f),
                     .radiometric_weight = make_float3(1.0f),
                     .light_shader = make_float3(1.0f),
                     .pdf = 1.0f,
                     .normalization_pdf = 1.0f,
                     .distance = 1.0f,
                     .emitter_kind = kind,
                     .emitter_index = shader_flags,
                     .light_object = shader_flags,
                     .light_primitive = shader_flags + 1u,
                     .shader_flags = shader_flags,
                     .apply_mis = true,
                     .constant_light_shader = true,
                     .distant = false,
                     .valid = true});
            };
            $if(kind == 0u) {
                accept(make_float3(1.0f, 2.0f, 3.0f), 10u);
            };
            $if(kind == 1u) {
                accept(make_float3(4.0f, 5.0f, 6.0f), 20u);
            };
            $if(kind == 2u) {
                accept(make_float3(7.0f, 8.0f, 9.0f), 30u);
            };
            Float3 evaluated = make_float3(0.0f);
            $if(light.valid) {
                ++recorded_receiving_evaluators;
                evaluated = light.direction +
                            make_float3(cast<float>(light.shader_flags));
            };
            values.write(kind,
                         make_float4(evaluated,
                                     select(0.0f, 1.0f, light.valid)));
        };
    if (recorded_receiving_evaluators != 1u) {
        std::cerr << "Direct-light receiving evaluator was recorded "
                  << recorded_receiving_evaluators << " times on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }
    auto merged_values = device.create_buffer<luisa::float4>(4u);
    auto merged_shader = device.compile(write_merged_light_sample);
    std::array<luisa::float4, 4u> merged_actual{};
    stream << merged_shader(merged_values).dispatch(4u)
           << merged_values.copy_to(luisa::span{merged_actual})
           << synchronize();
    constexpr auto merged_expected = std::array{
        luisa::float4{11.0f, 12.0f, 13.0f, 1.0f},
        luisa::float4{24.0f, 25.0f, 26.0f, 1.0f},
        luisa::float4{37.0f, 38.0f, 39.0f, 1.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 0.0f}};
    if (!std::equal(
            merged_actual.begin(), merged_actual.end(),
            merged_expected.begin(),
            [](const auto &lhs, const auto &rhs) noexcept {
                return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z &&
                       lhs.w == rhs.w;
            })) {
        std::cerr << "One-hot direct-light sample merge changed on " << backend
                  << '\n';
        return EXIT_FAILURE;
    }

    // Light roulette is defined in the local light-sample domain. This case
    // survives with probability 0.2 there, but would be incorrectly rejected
    // (probability 0.08) if path throughput entered the roulette measure.
    LightSampleRouletteCallable light_sample_roulette =
        [](Float3 unshadowed, Float random, Float inverse_threshold) noexcept {
            const auto maximum = max(abs(unshadowed.x),
                                     max(abs(unshadowed.y), abs(unshadowed.z)));
            const auto probability = maximum * inverse_threshold;
            const auto roulette =
                (inverse_threshold > 0.0f) & (probability < 1.0f);
            const auto survives = (!roulette) | (random < probability);
            const auto inverse_probability = select(
                1.0f, 1.0f / max(probability, 1.0e-20f), roulette);
            return select(0.0f, inverse_probability, survives);
        };
    Kernel1D write_finalized_light =
        [light_sample_roulette](BufferFloat4 values) noexcept {
            const auto local_weight = make_float3(0.5f, 0.25f, 0.125f);
            const auto light_shader = make_float3(0.2f, 0.4f, 0.8f);
            const auto path_throughput = make_float3(0.1f, 0.2f, 0.4f);
            const auto finalized = finalize_direct_light_sample(
                light_sample_roulette, local_weight * light_shader,
                path_throughput, 0.1f, 2.0f);
            values.write(0u, make_float4(finalized, 1.0f));
        };
    auto finalized_light = device.create_buffer<luisa::float4>(1u);
    auto finalized_light_shader = device.compile(write_finalized_light);
    luisa::float4 finalized_light_actual{};
    stream << finalized_light_shader(finalized_light).dispatch(1u)
           << finalized_light.copy_to(&finalized_light_actual) << synchronize();
    constexpr auto finalized_light_expected =
        luisa::float4{0.05f, 0.1f, 0.2f, 1.0f};
    const auto finalized_matches =
        std::abs(finalized_light_actual.x - finalized_light_expected.x) <=
            1.0e-6f &&
        std::abs(finalized_light_actual.y - finalized_light_expected.y) <=
            1.0e-6f &&
        std::abs(finalized_light_actual.z - finalized_light_expected.z) <=
            1.0e-6f &&
        finalized_light_actual.w == finalized_light_expected.w;
    if (!finalized_matches) {
        std::cerr << "Direct-light roulette included path throughput on "
                  << backend << ": got " << finalized_light_actual.x << ", "
                  << finalized_light_actual.y << ", "
                  << finalized_light_actual.z << '\n';
        return EXIT_FAILURE;
    }

    // Differently sized allocations must produce one shader structure. The
    // runtime capacity controls every SoA member offset and is not embedded as
    // a host literal in the AST.
    using DirectLightTaskCall =
        psycles::luisa_backend::detail::DirectLightTaskCall;
    // The queue payload contains only state invariant across the shadow path.
    // ShadowIntersectionBatchCall is a transition-local value and must not be
    // reintroduced into this aggregate: a fused auxiliary consumer never
    // reads it, and a split coroutine needs it on exactly one edge.
    // The Cycles volume-boundary counter uses the former four-byte tail
    // padding. It is persistent shadow state, unlike the intersection array.
    static_assert(offsetof(DirectLightTaskCall, volume_bounds_bounce) == 220u);
    const auto *task_type = luisa::compute::Type::of<DirectLightTaskCall>();
    const auto task_members = task_type->members();
    if (task_type->size() != 224u || task_members.size() != 32u ||
        task_members.back() != Type::of<luisa::uint>() ||
        std::any_of(task_members.begin(), task_members.end(), [](const Type *t) {
          return t->is_array() || t->is_structure();
        })) {
      std::cerr << "Direct-light queue payload retained transition-local "
                   "shadow state on "
                << backend << '\n';
      return EXIT_FAILURE;
    }
    auto small_tasks = device.create_soa<DirectLightTaskCall>(7u);
    auto large_tasks = device.create_soa<DirectLightTaskCall>(19u);
    const auto make_runtime_soa_kernel = [](auto *tasks) {
      return Kernel1D{[tasks](UInt capacity, BufferUInt values) noexcept {
        const auto x = dispatch_x();
        const auto runtime_tasks =
            make_runtime_direct_light_task_storage(*tasks, capacity);
        runtime_tasks.pixel.write(x, x + 37u);
        runtime_tasks.volume_bounds_bounce.write(x, x + 11u);
        values.write(x, runtime_tasks.pixel.read(x) +
                            runtime_tasks.volume_bounds_bounce.read(x));
      }};
    };
    auto small_kernel = make_runtime_soa_kernel(&small_tasks);
    auto large_kernel = make_runtime_soa_kernel(&large_tasks);
    if (small_kernel.function()->function().hash() !=
        large_kernel.function()->function().hash()) {
      std::cerr << "Direct-light runtime SoA hashed host capacity on "
                << backend << '\n';
      return EXIT_FAILURE;
    }
    auto small_values = device.create_buffer<std::uint32_t>(7u);
    auto large_values = device.create_buffer<std::uint32_t>(19u);
    auto small_shader = device.compile(small_kernel);
    auto large_shader = device.compile(large_kernel);
    std::array<std::uint32_t, 7u> small_actual{};
    std::array<std::uint32_t, 19u> large_actual{};
    stream << small_shader(7u, small_values).dispatch(7u)
           << large_shader(19u, large_values).dispatch(19u)
           << small_values.copy_to(luisa::span{small_actual})
           << large_values.copy_to(luisa::span{large_actual})
           << synchronize();
    for (auto index = std::size_t{0u}; index < small_actual.size(); ++index) {
      if (small_actual[index] != 2u * index + 48u) {
        std::cerr << "Small direct-light runtime SoA failed at " << index
                  << " on " << backend << '\n';
        return EXIT_FAILURE;
      }
    }
    for (auto index = std::size_t{0u}; index < large_actual.size(); ++index) {
      if (large_actual[index] != 2u * index + 48u) {
        std::cerr << "Large direct-light runtime SoA failed at " << index
                  << " on " << backend << '\n';
        return EXIT_FAILURE;
      }
    }
    return EXIT_SUCCESS;
}

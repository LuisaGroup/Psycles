#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_evaluation.h>
#include <psycles/luisa/surface_closure_set.h>

#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::adapter;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using psycles::test_support::approximately_equal;
using psycles::test_support::compile_named_kernel;
using psycles::test_support::make_surface_point;
using psycles::test_support::merged_surface_closure_plan;
using psycles::test_support::parameter_data;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::require_bounded_xir;
using psycles::test_support::surface_aov;

constexpr luisa::float3 authored_color{0.6f, 0.3f, 0.15f};
constexpr float authored_roughness = 0.37f;
constexpr luisa::float2 sample_random{0.23f, 0.71f};
constexpr float inverse_pi = 0.31830988618379067154f;
constexpr float inverse_two_pi = 0.15915494309189533577f;

namespace record {
constexpr std::uint32_t identity = 0u;
constexpr std::uint32_t trace = 1u;
constexpr std::uint32_t weight_and_flags = 2u;
constexpr std::uint32_t physical_payload = 3u;
constexpr std::uint32_t physical_albedo = 4u;
constexpr std::uint32_t aov_diffuse = 5u;
constexpr std::uint32_t aov_glossy = 6u;
constexpr std::uint32_t aov_normal = 7u;
constexpr std::uint32_t evaluation = 8u;
constexpr std::uint32_t evaluation_diffuse = 9u;
constexpr std::uint32_t evaluation_glossy = 10u;
constexpr std::uint32_t evaluation_aux = 11u;
constexpr std::uint32_t sample_identity = 12u;
constexpr std::uint32_t sample_direction = 13u;
constexpr std::uint32_t sample_evaluation = 14u;
constexpr std::uint32_t sample_diffuse = 15u;
constexpr std::uint32_t sample_glossy = 16u;
constexpr std::uint32_t sample_material = 17u;
constexpr std::uint32_t resampled_evaluation = 18u;
constexpr std::uint32_t lobe_masks = 19u;
constexpr std::uint32_t light_policy = 20u;
constexpr std::uint32_t count = 21u;
}// namespace record

struct CompiledSurface {
    std::shared_ptr<const SurfaceProgram> program;
    SurfaceClosurePlan closure_plan;
    std::vector<luisa::float4> parameters;
};

[[nodiscard]] ShaderGraph make_sheen_graph(std::string distribution) {
    CyclesNormalizedShaderGraph source;
    using NormalizedNode = typename decltype(source.nodes)::value_type;
    source.nodes.emplace_back(NormalizedNode{
        .id = 10u,
        .type = "geometry",
        .variant = {},
        .label = "Geometry",
        .inputs = {},
        .properties = {}});
    source.nodes.emplace_back(NormalizedNode{
        .id = 11u,
        .type = "sheen_bsdf",
        .variant = {},
        .label = "Cycles 5.2 standalone Sheen regression",
        .inputs =
            {{.socket = "Color",
              .source = std::nullopt,
              .value = SocketValue::color(
                  {authored_color.x,
                   authored_color.y,
                   authored_color.z})},
             {.socket = "Roughness",
              .source = std::nullopt,
              .value = SocketValue::floating(authored_roughness)},
             {.socket = "Normal",
              .source = CyclesOutputRef{.node = 10u, .socket = "Normal"},
              .value = std::nullopt}},
        .properties =
            {{"distribution", SocketValue::string(std::move(distribution))}}});
    source.set_root(
        ShaderDomain::surface,
        CyclesOutputRef{.node = 11u, .socket = "BSDF"});
    auto adapted = adapt_cycles_shader_graph(
        source, make_core_cycles_node_mappings());
    if (!adapted.ok()) {
        throw std::runtime_error{
            "failed to adapt standalone Sheen graph"};
    }
    return std::move(*adapted.graph);
}

[[nodiscard]] CompiledSurface compile_graph(
    ShaderCompiler &compiler,
    std::string distribution,
    ClosureOperation expected_operation) {
    const auto shader = compiler.compile(
        make_sheen_graph(std::move(distribution)));
    if (!shader.ok()) {
        throw std::runtime_error{
            "failed to compile standalone Sheen graph"};
    }
    const auto surface = compile_surface_program(*shader.program);
    if (!surface.ok() ||
        surface.program->closure_instructions().size() != 1u ||
        surface.program->closure_instructions().front().operation !=
            expected_operation) {
        throw std::runtime_error{
            "standalone Sheen distribution was not lowered statically"};
    }
    for (const auto &parameter : surface.program->parameters()) {
        if (parameter.socket == "Weight") {
            throw std::runtime_error{
                "Blender's internal closure Weight leaked into the graph ABI"};
        }
    }
    auto parameters = parameter_data(*surface.program);
    return {
        .program = surface.program,
        .closure_plan = merged_surface_closure_plan(
            *surface.program, parameters),
        .parameters = std::move(parameters)};
}

[[nodiscard]] SurfacePoint sheen_point() noexcept {
    auto point = make_surface_point();
    point.geometric_normal = make_float3(0.0f, 0.0f, 1.0f);
    point.shading_normal = point.geometric_normal;
    point.object_shading_normal = point.geometric_normal;
    point.undisplaced_shading_normal = point.geometric_normal;
    point.undisplaced_object_shading_normal = point.geometric_normal;
    point.incoming = make_float3(0.6f, 0.0f, 0.8f);
    return point;
}

[[nodiscard]] SurfaceQuery sheen_query(
    std::uint32_t lobe_mask =
        static_cast<std::uint32_t>(
            event_diffuse | event_glossy |
            event_transmission | event_transparent)) noexcept {
    return {
        .lobe_mask = lobe_mask,
        .transport_mode =
            static_cast<std::uint32_t>(TransportMode::radiance),
        .glossy_filter_roughness = 0.0f,
        .reflective_caustics = true,
        .refractive_caustics = true};
}

[[nodiscard]] auto make_test_kernel(
    const SurfaceDispatch &surfaces,
    std::uint32_t surface_tag,
    bool microfiber) {
    return Kernel1D{[surface = &surfaces,
                       surface_tag,
                       microfiber](BufferFloat4 parameters,
                           BufferFloat4 output) noexcept {
        const auto case_index = dispatch_x();
        const auto cycles_value = microfiber
                                      ? select(0.5f,
                                            0.0f,
                                            case_index == 1u)
                                      : Float{0.5f};
        ParameterShaderServices services{parameters, cycles_value};
        const auto point = sheen_point();
        const auto query = sheen_query();

        SurfaceClosureSet closures{1u};
        static_cast<void>(surface->collect_closures(
            UInt{surface_tag}, services, point, true, true, closures));
        const auto closure = closures.entry(0u);
        const auto physical_blocks =
            pack_surface_closure_physical(closure);
        const auto physical_closure = unpack_surface_closure_physical(
            Expr<luisa::float4x4>{physical_blocks.block_0.expression()},
            Expr<luisa::float4x4>{physical_blocks.block_1.expression()});
        const auto trace = surface->closure_trace(
            UInt{surface_tag}, services, point, 0u);
        const auto aov = surface_aov(
            *surface, UInt{surface_tag}, services, point);
        const auto evaluation_direction = microfiber
                                              ? make_float3(0.0f, 0.0f, 1.0f)
                                              : normalize(make_float3(
                                                    0.2f,
                                                    0.3f,
                                                    0.9327379f));
        const auto evaluation = surface->evaluate(
            UInt{surface_tag}, services, point, evaluation_direction, query);
        const auto sample = surface->sample_trace(
            UInt{surface_tag},
            services,
            point,
            0.41f,
            make_float2(sample_random.x, sample_random.y),
            query);
        const auto resampled = surface->evaluate(
            UInt{surface_tag}, services, point, sample.sample.wi, query);
        const auto diffuse_query = sheen_query(
            static_cast<std::uint32_t>(event_diffuse));
        const auto glossy_query = sheen_query(
            static_cast<std::uint32_t>(event_glossy));
        const auto diffuse_sample = surface->sample_trace(
            UInt{surface_tag}, services, point, 0.41f,
            make_float2(sample_random.x, sample_random.y), diffuse_query);
        const auto glossy_sample = surface->sample_trace(
            UInt{surface_tag}, services, point, 0.41f,
            make_float2(sample_random.x, sample_random.y), glossy_query);
        const auto diffuse_evaluation = surface->evaluate(
            UInt{surface_tag}, services, point,
            evaluation_direction, diffuse_query);
        const auto glossy_evaluation = surface->evaluate(
            UInt{surface_tag}, services, point,
            evaluation_direction, glossy_query);
        const auto directions = make_surface_closure_evaluation_directions(
            SurfaceClosurePoint{point}, evaluation_direction);
        const auto diffuse_light_policy = SurfaceClosureEvaluationPolicy{
            .diffuse_included = true,
            .glossy_included = false,
            .glass_included = true,
            .transmission_included = true,
            .preserve_pdf = true};
        const auto glossy_light_policy = SurfaceClosureEvaluationPolicy{
            .diffuse_included = false,
            .glossy_included = true,
            .glass_included = true,
            .transmission_included = true,
            .preserve_pdf = true};
        const auto reachability = SurfaceClosureReachability{
            .kinds = surface_closure_kind_bit(
                microfiber ? SurfaceClosureKind::sheen_microfiber
                            : SurfaceClosureKind::sheen_ashikhmin)};
        const auto diffuse_light = surface_closure_evaluation_contribution(
            services,
            SurfaceClosurePoint{point},
            point.shading_normal,
            physical_closure,
            directions.incoming,
            directions.outgoing,
            query,
            diffuse_light_policy,
            false,
            reachability);
        const auto glossy_light = surface_closure_evaluation_contribution(
            services,
            SurfaceClosurePoint{point},
            point.shading_normal,
            physical_closure,
            directions.incoming,
            directions.outgoing,
            query,
            glossy_light_policy,
            false,
            reachability);

        const auto base = case_index * record::count;
        output.write(base + record::identity,
            make_float4(cast<float>(closure.kind),
                cast<float>(closure.lobe),
                cast<float>(closures.count()),
                select(0.0f, 1.0f, closure.setup_valid)));
        output.write(base + record::trace,
            make_float4(cast<float>(trace.type),
                trace.sample_weight,
                cast<float>(trace.count),
                select(0.0f, 1.0f, trace.valid)));
        output.write(base + record::weight_and_flags,
            make_float4(trace.weight,
                cast<float>(trace.runtime_flags)));
        output.write(base + record::physical_payload,
            make_float4(closure.roughness,
                closure.sheen_transform_a,
                closure.sheen_transform_b,
                closure.allocation_weight));
        output.write(base + record::physical_albedo,
            make_float4(closure.albedo, closure.sample_weight));
        output.write(base + record::aov_diffuse,
            make_float4(aov.albedo, aov.roughness.x));
        output.write(base + record::aov_glossy,
            make_float4(aov.glossy_albedo, aov.roughness.y));
        output.write(base + record::aov_normal,
            make_float4(aov.normal, 0.0f));
        output.write(base + record::evaluation,
            make_float4(evaluation.f, evaluation.pdf));
        output.write(base + record::evaluation_diffuse,
            make_float4(evaluation.diffuse_f, evaluation.diffuse_pdf));
        output.write(base + record::evaluation_glossy,
            make_float4(evaluation.glossy_f,
                cast<float>(evaluation.events)));
        output.write(base + record::evaluation_aux,
            make_float4(evaluation.average_roughness_squared,
                0.0f,
                0.0f,
                0.0f));
        output.write(base + record::sample_identity,
            make_float4(cast<float>(sample.closure_type),
                sample.closure_sample_weight,
                sample.selection_rescaled,
                select(0.0f, 1.0f, sample.sample.valid)));
        output.write(base + record::sample_direction,
            make_float4(sample.sample.wi,
                select(0.0f, 1.0f, sample.closure_valid)));
        output.write(base + record::sample_evaluation,
            make_float4(sample.sample.evaluation.f,
                sample.sample.evaluation.pdf));
        output.write(base + record::sample_diffuse,
            make_float4(sample.sample.evaluation.diffuse_f,
                sample.sample.evaluation.diffuse_pdf));
        output.write(base + record::sample_glossy,
            make_float4(sample.sample.evaluation.glossy_f,
                cast<float>(sample.sample.evaluation.events)));
        output.write(base + record::sample_material,
            make_float4(sample.sample.roughness,
                sample.sample.eta,
                sample.sample.evaluation.average_roughness_squared));
        output.write(base + record::resampled_evaluation,
            make_float4(resampled.f, resampled.pdf));
        output.write(base + record::lobe_masks,
            make_float4(select(0.0f, 1.0f, diffuse_sample.sample.valid),
                select(0.0f, 1.0f, glossy_sample.sample.valid),
                diffuse_evaluation.pdf,
                glossy_evaluation.pdf));
        output.write(base + record::light_policy,
            make_float4(diffuse_light.f.x,
                glossy_light.f.x,
                diffuse_light.weighted_pdf,
                glossy_light.weighted_pdf));
    }};
}

[[nodiscard]] auto make_ashikhmin_bump_type_kernel(
    const SurfaceDispatch &surfaces,
    std::uint32_t surface_tag) {
    return Kernel1D{[surface = &surfaces,
                       surface_tag](BufferFloat4 parameters,
                       BufferFloat4 output) noexcept {
        ParameterShaderServices services{parameters};
        auto point = sheen_point();

        SurfaceClosureSet closures{1u};
        static_cast<void>(surface->collect_closures(
            UInt{surface_tag}, services, point, true, true, closures));
        const auto closure = closures.entry(0u);
        const auto physical_blocks =
            pack_surface_closure_physical(closure);
        auto physical_closure = unpack_surface_closure_physical(
            Expr<luisa::float4x4>{physical_blocks.block_0.expression()},
            Expr<luisa::float4x4>{physical_blocks.block_1.expression()});

        // Ashikhmin setup does not derive any parameter from N, so replacing
        // this expression is exactly the physical record produced by the
        // same raw node with the following linked normal. Use a substantially
        // tilted frame and grazing-tangent directions so erroneously treating
        // type 16 as a diffuse ClosureType produces a measurable terminator
        // softening factor instead of passing under float tolerance.
        const auto bump_normal = normalize(
            make_float3(0.5f, 0.0f, 0.8660254038f));
        const auto tangent = normalize(
            make_float3(-0.8660254038f, 0.0f, 0.5f));
        const auto bump_incoming =
            tangent * 0.8f + bump_normal * 0.6f;
        const auto bump_outgoing =
            tangent * 0.95f + bump_normal * 0.3122498999f;
        physical_closure.normal = bump_normal;
        point.incoming = bump_incoming;
        const auto directions =
            make_surface_closure_evaluation_directions(
                SurfaceClosurePoint{point}, bump_outgoing);
        const auto policy = SurfaceClosureEvaluationPolicy{
            .diffuse_included = false,
            .glossy_included = true,
            .glass_included = true,
            .transmission_included = true,
            .preserve_pdf = true};
        const auto query = sheen_query();
        const auto contribution =
            surface_closure_evaluation_contribution(
                services,
                SurfaceClosurePoint{point},
                point.shading_normal,
                physical_closure,
                directions.incoming,
                directions.outgoing,
                query,
                policy,
                false,
                SurfaceClosureReachability{
                    .kinds = surface_closure_kind_bit(
                        SurfaceClosureKind::sheen_ashikhmin)});
        output.write(0u,
            make_float4(contribution.f, contribution.weighted_pdf));
    }};
}

[[nodiscard]] float dot3(luisa::float3 a, luisa::float3 b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] constexpr luisa::float4 float4_of(
    luisa::float3 value, float w) noexcept {
    return {value.x, value.y, value.z, w};
}

[[nodiscard]] luisa::float3 cross3(
    luisa::float3 a, luisa::float3 b) noexcept {
    return {a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

[[nodiscard]] luisa::float3 normalize3(luisa::float3 value) noexcept {
    const auto inverse_length = 1.0f / std::sqrt(dot3(value, value));
    return value * inverse_length;
}

struct Basis {
    luisa::float3 tangent;
    luisa::float3 bitangent;
};

[[nodiscard]] Basis orthonormals(luisa::float3 normal) noexcept {
    auto tangent = normal.x != normal.y || normal.x != normal.z
                       ? luisa::float3{normal.z - normal.y,
                             normal.x - normal.z,
                             normal.y - normal.x}
                       : luisa::float3{normal.z - normal.y,
                             normal.x + normal.z,
                             -normal.y - normal.x};
    tangent = normalize3(tangent);
    return {.tangent = tangent,
        .bitangent = cross3(normal, tangent)};
}

[[nodiscard]] Basis sheen_basis(
    luisa::float3 normal, luisa::float3 incoming) noexcept {
    auto bitangent = cross3(normal, incoming);
    const auto length_squared = dot3(bitangent, bitangent);
    if (length_squared <= 1.0e-20f) {
        return orthonormals(normal);
    }
    bitangent = bitangent * (1.0f / std::sqrt(length_squared));
    return {.tangent = cross3(bitangent, normal),
        .bitangent = bitangent};
}

[[nodiscard]] luisa::float2 uniform_disk(luisa::float2 random) noexcept {
    const auto a = 2.0f * random.x - 1.0f;
    const auto b = 2.0f * random.y - 1.0f;
    if (a == 0.0f && b == 0.0f) {
        return {0.0f, 0.0f};
    }
    float radius{};
    float phi{};
    if (a * a > b * b) {
        radius = a;
        phi = 0.25f * std::numbers::pi_v<float> * (b / a);
    } else {
        radius = b;
        phi = 0.5f * std::numbers::pi_v<float> -
              0.25f * std::numbers::pi_v<float> * (a / b);
    }
    return {radius * std::cos(phi), radius * std::sin(phi)};
}

[[nodiscard]] luisa::float3 expected_microfiber_sample() noexcept {
    const auto disk = uniform_disk(sample_random);
    const auto disk_z = std::sqrt(std::max(
        1.0f - disk.x * disk.x - disk.y * disk.y, 0.0f));
    const auto local = normalize3(
        {disk.x - 0.5f * disk_z, disk.y, 0.5f * disk_z});
    const auto basis = sheen_basis(
        {0.0f, 0.0f, 1.0f}, {0.6f, 0.0f, 0.8f});
    return basis.tangent * local.x + basis.bitangent * local.y +
           luisa::float3{0.0f, 0.0f, 1.0f} * local.z;
}

[[nodiscard]] luisa::float3 expected_ashikhmin_sample() noexcept {
    auto disk = uniform_disk(sample_random);
    const auto radius_squared = disk.x * disk.x + disk.y * disk.y;
    const auto z = 1.0f - radius_squared;
    const auto scale = std::sqrt(std::max(z + 1.0f, 0.0f));
    disk *= scale;
    const auto basis = orthonormals({0.0f, 0.0f, 1.0f});
    return basis.tangent * disk.x + basis.bitangent * disk.y +
           luisa::float3{0.0f, 0.0f, 1.0f} * z;
}

struct ScalarEvaluation {
    float value;
    float pdf;
};

[[nodiscard]] ScalarEvaluation microfiber_evaluation(
    luisa::float3 outgoing) noexcept {
    const auto basis = sheen_basis(
        {0.0f, 0.0f, 1.0f}, {0.6f, 0.0f, 0.8f});
    const auto local = luisa::float3{
        dot3(outgoing, basis.tangent),
        dot3(outgoing, basis.bitangent),
        outgoing.z};
    const auto length_squared =
        (0.5f * local.x + 0.5f * local.z) *
            (0.5f * local.x + 0.5f * local.z) +
        (0.5f * local.y) * (0.5f * local.y) + local.z * local.z;
    const auto transformed = 0.5f / length_squared;
    const auto value = inverse_pi * std::max(local.z, 0.0f) *
                       transformed * transformed;
    return {.value = value, .pdf = value};
}

[[nodiscard]] ScalarEvaluation ashikhmin_evaluation(
    luisa::float3 outgoing,
    luisa::float3 normal = {0.0f, 0.0f, 1.0f},
    luisa::float3 incoming = {0.6f, 0.0f, 0.8f}) noexcept {
    const auto cos_ni = dot3(normal, incoming);
    const auto cos_no = dot3(normal, outgoing);
    const auto half = normalize3(incoming + outgoing);
    const auto cos_nh = dot3(normal, half);
    const auto cos_hi = std::abs(dot3(incoming, half));
    if (!(cos_ni > 0.0f && cos_no > 0.0f &&
          std::abs(cos_nh) < 1.0f - 1.0e-5f && cos_hi > 1.0e-5f)) {
        return {};
    }
    const auto ratio = std::max(cos_nh / cos_hi, 1.0e-5f);
    const auto fac1 = 2.0f * std::abs(ratio * cos_ni);
    const auto fac2 = 2.0f * std::abs(ratio * cos_no);
    const auto sin_nh_squared = 1.0f - cos_nh * cos_nh;
    const auto inverse_sigma_squared =
        1.0f / (authored_roughness * authored_roughness);
    const auto distribution =
        std::exp(-(cos_nh * cos_nh / sin_nh_squared) *
                 inverse_sigma_squared) *
        inverse_sigma_squared * inverse_pi /
        (sin_nh_squared * sin_nh_squared);
    const auto masking = std::min(1.0f, std::min(fac1, fac2));
    return {.value = 0.25f * distribution * masking / cos_ni,
        .pdf = inverse_two_pi};
}

[[nodiscard]] bool close_record(
    std::string_view backend,
    std::string_view model,
    std::uint32_t record_index,
    luisa::float4 actual,
    luisa::float4 expected,
    float tolerance = 5.0e-5f) noexcept {
    if (approximately_equal(actual, expected, tolerance)) {
        return true;
    }
    std::cerr << "standalone Sheen " << model << " failed on " << backend
              << " at record " << record_index << ": got {"
              << actual.x << ", " << actual.y << ", " << actual.z << ", "
              << actual.w << "}, expected {" << expected.x << ", "
              << expected.y << ", " << expected.z << ", " << expected.w
              << "}\n";
    return false;
}

[[nodiscard]] bool verify_valid_model(
    std::string_view backend,
    std::string_view model,
    const std::array<luisa::float4, record::count> &actual,
    bool microfiber) noexcept {
    const auto kind = microfiber ? SurfaceClosureKind::sheen_microfiber
                                 : SurfaceClosureKind::sheen_ashikhmin;
    const auto type = microfiber ? cycles_closure::type_sheen
                                 : cycles_closure::type_ashikhmin_velvet;
    const auto closure_weight = microfiber
                                    ? authored_color * 0.5f
                                    : authored_color;
    const auto closure_sample_weight = microfiber ? 0.175f : 0.35f;
    const auto physical_roughness = microfiber
                                        ? authored_roughness
                                        : 1.0f /
                                              (authored_roughness *
                                               authored_roughness);
    const auto evaluation_direction = microfiber
                                          ? luisa::float3{0.0f, 0.0f, 1.0f}
                                          : normalize3(
                                                {0.2f, 0.3f, 0.9327379f});
    const auto scalar_evaluation = microfiber
                                       ? microfiber_evaluation(
                                             evaluation_direction)
                                       : ashikhmin_evaluation(
                                             evaluation_direction);
    const auto sampled_direction = microfiber
                                       ? expected_microfiber_sample()
                                       : expected_ashikhmin_sample();
    const auto sampled_evaluation = microfiber
                                        ? microfiber_evaluation(
                                              sampled_direction)
                                        : ashikhmin_evaluation(
                                              sampled_direction);
    constexpr auto events =
        static_cast<float>(event_diffuse | event_reflection);
    const auto diffuse_value = microfiber
                                   ? closure_weight * scalar_evaluation.value
                                   : luisa::float3{0.0f};
    const auto glossy_value = microfiber
                                  ? luisa::float3{0.0f}
                                  : closure_weight * scalar_evaluation.value;
    const auto sampled_diffuse = microfiber
                                     ? closure_weight * sampled_evaluation.value
                                     : luisa::float3{0.0f};
    const auto sampled_glossy = microfiber
                                    ? luisa::float3{0.0f}
                                    : closure_weight * sampled_evaluation.value;
    const std::array expected{
        luisa::float4{static_cast<float>(kind), 0.0f, 1.0f, 1.0f},
        luisa::float4{static_cast<float>(type),
            closure_sample_weight, 1.0f, 1.0f},
        float4_of(closure_weight, 24.0f),
        luisa::float4{physical_roughness,
            microfiber ? 0.5f : 0.0f,
            microfiber ? 0.5f : 0.0f,
            0.35f},
        float4_of(closure_weight, closure_sample_weight),
        float4_of(
            microfiber ? closure_weight : luisa::float3{0.0f}, 1.0f),
        float4_of(
            microfiber ? luisa::float3{0.0f} : closure_weight, 1.0f),
        luisa::float4{0.0f, 0.0f, 1.0f, 0.0f},
        float4_of(
            closure_weight * scalar_evaluation.value, scalar_evaluation.pdf),
        float4_of(diffuse_value, scalar_evaluation.pdf),
        float4_of(glossy_value, events),
        luisa::float4{1.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{static_cast<float>(type),
            closure_sample_weight, 0.41f, 1.0f},
        float4_of(sampled_direction, 1.0f),
        float4_of(
            closure_weight * sampled_evaluation.value,
            sampled_evaluation.pdf),
        float4_of(sampled_diffuse, sampled_evaluation.pdf),
        float4_of(sampled_glossy, events),
        luisa::float4{1.0f,
            1.0f,
            1.0f,
            1.0f},
        float4_of(
            closure_weight * sampled_evaluation.value,
            sampled_evaluation.pdf),
        luisa::float4{1.0f, 0.0f, scalar_evaluation.pdf, 0.0f},
        luisa::float4{
            microfiber
                ? closure_weight.x * scalar_evaluation.value
                : 0.0f,
            microfiber
                ? 0.0f
                : closure_weight.x * scalar_evaluation.value,
            closure_sample_weight * scalar_evaluation.pdf,
            closure_sample_weight * scalar_evaluation.pdf}};
    static_assert(expected.size() == record::count);
    for (auto index = std::uint32_t{0u}; index < record::count; ++index) {
        if (!close_record(
                backend, model, index, actual[index], expected[index])) {
            return false;
        }
    }
    return true;
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto microfiber = compile_graph(
        compiler, "MICROFIBER", ClosureOperation::sheen_microfiber);
    const auto ashikhmin = compile_graph(
        compiler, "ASHIKHMIN", ClosureOperation::sheen_ashikhmin);

    SurfaceDispatch microfiber_surfaces;
    const auto microfiber_tag = microfiber_surfaces.create<GraphSurface>(
        microfiber.program, microfiber.closure_plan);
    SurfaceDispatch ashikhmin_surfaces;
    const auto ashikhmin_tag = ashikhmin_surfaces.create<GraphSurface>(
        ashikhmin.program, ashikhmin.closure_plan);
    auto microfiber_kernel = make_test_kernel(
        microfiber_surfaces, microfiber_tag, true);
    auto ashikhmin_kernel = make_test_kernel(
        ashikhmin_surfaces, ashikhmin_tag, false);
    auto ashikhmin_bump_type_kernel = make_ashikhmin_bump_type_kernel(
        ashikhmin_surfaces, ashikhmin_tag);

    if (backend == "fallback") {
        auto bounded = true;
        bounded &= require_bounded_xir(
            "standalone_sheen_microfiber", microfiber_kernel, 480000u);
        bounded &= require_bounded_xir(
            "standalone_sheen_ashikhmin", ashikhmin_kernel, 420000u);
        bounded &= require_bounded_xir(
            "standalone_sheen_ashikhmin_bump_type",
            ashikhmin_bump_type_kernel,
            320000u);
        if (!bounded) {
            return EXIT_FAILURE;
        }
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto microfiber_parameters =
        device.create_buffer<luisa::float4>(microfiber.parameters.size());
    auto ashikhmin_parameters =
        device.create_buffer<luisa::float4>(ashikhmin.parameters.size());
    auto microfiber_output =
        device.create_buffer<luisa::float4>(2u * record::count);
    auto ashikhmin_output =
        device.create_buffer<luisa::float4>(record::count);
    auto ashikhmin_bump_type_output =
        device.create_buffer<luisa::float4>(1u);
    auto microfiber_shader = compile_named_kernel(
        device, "standalone_sheen_microfiber", microfiber_kernel);
    auto ashikhmin_shader = compile_named_kernel(
        device, "standalone_sheen_ashikhmin", ashikhmin_kernel);
    auto ashikhmin_bump_type_shader = compile_named_kernel(
        device,
        "standalone_sheen_ashikhmin_bump_type",
        ashikhmin_bump_type_kernel);
    std::array<luisa::float4, 2u * record::count> microfiber_actual{};
    std::array<luisa::float4, record::count> ashikhmin_actual{};
    luisa::float4 ashikhmin_bump_type_actual{};
    stream << microfiber_parameters.copy_from(
                  luisa::span{microfiber.parameters})
           << ashikhmin_parameters.copy_from(
                  luisa::span{ashikhmin.parameters})
           << microfiber_shader(microfiber_parameters, microfiber_output)
                  .dispatch(2u)
           << microfiber_output.copy_to(luisa::span{microfiber_actual})
           << ashikhmin_shader(ashikhmin_parameters, ashikhmin_output)
                  .dispatch(1u)
           << ashikhmin_output.copy_to(luisa::span{ashikhmin_actual})
           << ashikhmin_bump_type_shader(
                  ashikhmin_parameters, ashikhmin_bump_type_output)
                  .dispatch(1u)
           << ashikhmin_bump_type_output.copy_to(
                  luisa::span{&ashikhmin_bump_type_actual, 1u})
           << synchronize();

    std::array<luisa::float4, record::count> microfiber_valid{};
    std::copy_n(microfiber_actual.begin(),
        record::count,
        microfiber_valid.begin());
    if (!verify_valid_model(
            backend, "Microfiber", microfiber_valid, true) ||
        !verify_valid_model(
            backend, "Ashikhmin", ashikhmin_actual, false)) {
        return EXIT_FAILURE;
    }

    const auto bump_normal = normalize3(
        {0.5f, 0.0f, 0.8660254038f});
    const auto bump_tangent = normalize3(
        {-0.8660254038f, 0.0f, 0.5f});
    const auto bump_incoming =
        bump_tangent * 0.8f + bump_normal * 0.6f;
    const auto bump_outgoing =
        bump_tangent * 0.95f + bump_normal * 0.3122498999f;
    const auto bump_evaluation = ashikhmin_evaluation(
        bump_outgoing, bump_normal, bump_incoming);
    const auto bump_expected = float4_of(
        authored_color * bump_evaluation.value,
        0.35f * bump_evaluation.pdf);
    if (!close_record(backend,
            "Ashikhmin glossy-type bump policy",
            0u,
            ashikhmin_bump_type_actual,
            bump_expected)) {
        return EXIT_FAILURE;
    }

    const auto invalid = microfiber_actual.data() + record::count;
    constexpr std::array invalid_records{
        record::identity,
        record::trace,
        record::weight_and_flags,
        record::physical_payload,
        record::physical_albedo,
        record::aov_diffuse,
        record::aov_glossy,
        record::evaluation,
        record::sample_identity,
        record::lobe_masks,
        record::light_policy};
    const std::array invalid_expected{
        luisa::float4{
            static_cast<float>(SurfaceClosureKind::sheen_microfiber),
            0.0f,
            1.0f,
            0.0f},
        luisa::float4{
            static_cast<float>(cycles_closure::type_none), 0.0f, 1.0f, 1.0f},
        float4_of(authored_color, 0.0f),
        luisa::float4{authored_roughness, 0.0f, 0.0f, 0.35f},
        luisa::float4{0.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 1.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 1.0f},
        luisa::float4{0.0f},
        luisa::float4{0.0f},
        luisa::float4{0.0f},
        luisa::float4{0.0f}};
    static_assert(invalid_records.size() == invalid_expected.size());
    for (auto index = std::size_t{0u}; index < invalid_records.size(); ++index) {
        if (!close_record(backend,
                "invalid Microfiber LTC",
                invalid_records[index],
                invalid[invalid_records[index]],
                invalid_expected[index])) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

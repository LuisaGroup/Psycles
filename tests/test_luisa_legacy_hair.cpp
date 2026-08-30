#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_physical_blocks.h>
#include <psycles/luisa/surface_closure_sampling.h>
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
constexpr float authored_offset = 0.12f;
constexpr float authored_roughness_u = 0.21f;
constexpr float authored_roughness_v = 0.43f;
constexpr float physical_roughness_u = authored_roughness_u;
constexpr float physical_roughness_v = authored_roughness_v;
constexpr float clamped_roughness_u = 0.001f;
constexpr float clamped_roughness_v = 1.0f;
constexpr float closure_sample_weight = 0.35f;
constexpr luisa::float2 sample_random{0.23f, 0.71f};

namespace record {
constexpr std::uint32_t identity = 0u;
constexpr std::uint32_t trace = 1u;
constexpr std::uint32_t weight = 2u;
constexpr std::uint32_t tangent = 3u;
constexpr std::uint32_t shape = 4u;
constexpr std::uint32_t aov_albedo = 5u;
constexpr std::uint32_t aov_glossy = 6u;
constexpr std::uint32_t aov_transmission = 7u;
constexpr std::uint32_t evaluation = 8u;
constexpr std::uint32_t evaluation_glossy = 9u;
constexpr std::uint32_t evaluation_aux = 10u;
constexpr std::uint32_t wrong_hemisphere = 11u;
constexpr std::uint32_t sample_direction = 12u;
constexpr std::uint32_t sample_evaluation = 13u;
constexpr std::uint32_t sample_glossy = 14u;
constexpr std::uint32_t sample_material = 15u;
constexpr std::uint32_t sample_trace = 16u;
constexpr std::uint32_t selection = 17u;
constexpr std::uint32_t count = 18u;
}// namespace record

struct CompiledSurface {
    std::shared_ptr<const SurfaceProgram> program;
    SurfaceClosurePlan closure_plan;
    std::vector<luisa::float4> parameters;
};

struct HostVector {
    float x;
    float y;
    float z;
};

[[nodiscard]] HostVector add(
    HostVector a, HostVector b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

[[nodiscard]] HostVector subtract(
    HostVector a, HostVector b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

[[nodiscard]] HostVector scale(
    HostVector value, float scale) noexcept {
    return {
        value.x * scale,
        value.y * scale,
        value.z * scale};
}

[[nodiscard]] float dot(HostVector a, HostVector b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

[[nodiscard]] HostVector cross(HostVector a, HostVector b) noexcept {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

[[nodiscard]] HostVector normalize(HostVector value) noexcept {
    const auto inverse_length =
        1.0f / std::sqrt(std::max(dot(value, value), 1.0e-30f));
    return scale(value, inverse_length);
}

[[nodiscard]] luisa::float3 as_float3(HostVector value) noexcept {
    return {value.x, value.y, value.z};
}

constexpr HostVector incoming_source{0.3f, 0.4f, 0.8660254f};
constexpr HostVector dpdu_source{0.6f, 0.8f, 0.0f};
constexpr HostVector dpdv_source{-0.8f, 0.6f, 0.0f};
constexpr HostVector reflection_outgoing_source{
    0.2f, -0.1f, 0.974679434f};
constexpr HostVector transmission_outgoing_source{
    0.2f, -0.1f, -0.974679434f};
// A ribbon normal can lie on the same side of a sampled legacy Hair
// transmission direction even though the longitudinal Hair sample is valid.
// Cycles deliberately lets bsdf_hair_transmission_sample() own that support
// decision; this normal keeps wi on the front side while exposing any later
// generic dot(N, wo) rejection of the sampled direction.
constexpr HostVector sampled_transmission_same_side_normal{
    -0.6f, -0.3f, 0.74161985f};

struct HairReference {
    HostVector tangent;
    float offset;
    float roughness_u;
    float roughness_v;
};

struct HairFrame {
    HostVector tangent;
    HostVector y;
    HostVector x;
    float theta_r;
};

[[nodiscard]] HairFrame hair_frame(
    const HairReference &closure, HostVector incoming) noexcept {
    const auto iz = std::clamp(
        dot(closure.tangent, incoming), -1.0f, 1.0f);
    const auto y = normalize(subtract(
        incoming, scale(closure.tangent, iz)));
    return {
        .tangent = closure.tangent,
        .y = y,
        .x = cross(y, closure.tangent),
        .theta_r =
            std::numbers::pi_v<float> * 0.5f - std::acos(iz)};
}

struct LongitudinalBounds {
    float a;
    float b;
};

[[nodiscard]] LongitudinalBounds longitudinal_bounds(
    const HairReference &closure, float theta_r) noexcept {
    constexpr auto half_pi = std::numbers::pi_v<float> * 0.5f;
    return {
        .a = std::atan2(
            ((half_pi + theta_r) * 0.5f - closure.offset) /
                closure.roughness_u,
            1.0f),
        .b = std::atan2(
            ((-half_pi + theta_r) * 0.5f - closure.offset) /
                closure.roughness_u,
            1.0f)};
}

[[nodiscard]] float hair_evaluation(
    const HairReference &closure,
    HostVector incoming,
    HostVector outgoing,
    HostVector normal,
    bool reflection,
    bool sampled_direction = false) noexcept {
    if (!sampled_direction &&
        ((reflection && dot(normal, outgoing) < 0.0f) ||
         (!reflection && dot(normal, outgoing) >= 0.0f))) {
        return 0.0f;
    }
    constexpr auto pi = std::numbers::pi_v<float>;
    constexpr auto half_pi = pi * 0.5f;
    const auto frame = hair_frame(closure, incoming);
    const auto outgoing_z = std::clamp(
        dot(frame.tangent, outgoing), -1.0f, 1.0f);
    const auto outgoing_y = normalize(subtract(
        outgoing, scale(frame.tangent, outgoing_z)));
    const auto theta_i = half_pi - std::acos(outgoing_z);
    const auto cosine_phi = std::clamp(
        dot(outgoing_y, frame.y), -1.0f, 1.0f);
    if (half_pi - std::abs(theta_i) < 0.001f ||
        (reflection && cosine_phi < 0.0f)) {
        return 0.0f;
    }
    const auto bounds = longitudinal_bounds(closure, frame.theta_r);
    const auto t =
        (theta_i + frame.theta_r) * 0.5f - closure.offset;
    const auto theta_pdf = closure.roughness_u /
                           (2.0f *
                            (t * t +
                             closure.roughness_u * closure.roughness_u) *
                            (bounds.a - bounds.b) * std::cos(theta_i));
    float phi_pdf;
    if (reflection) {
        const auto phi_i = std::min(
            std::abs(std::acos(cosine_phi) /
                     closure.roughness_v),
            pi);
        phi_pdf = std::cos(phi_i * 0.5f) * 0.25f /
                  closure.roughness_v;
    } else {
        const auto phi_i = std::acos(cosine_phi);
        const auto c = 2.0f * std::atan2(
            half_pi / closure.roughness_v, 1.0f);
        const auto p = pi - std::abs(phi_i);
        phi_pdf = closure.roughness_v /
                  (c *
                   (p * p +
                    closure.roughness_v * closure.roughness_v));
    }
    return phi_pdf * theta_pdf;
}

[[nodiscard]] HostVector hair_sample(
    const HairReference &closure,
    HostVector incoming,
    bool reflection) noexcept {
    constexpr auto pi = std::numbers::pi_v<float>;
    constexpr auto half_pi = pi * 0.5f;
    const auto frame = hair_frame(closure, incoming);
    const auto bounds = longitudinal_bounds(closure, frame.theta_r);
    const auto t = closure.roughness_u * std::tan(
        sample_random.x * (bounds.a - bounds.b) + bounds.b);
    const auto theta_i =
        2.0f * (t + closure.offset) - frame.theta_r;
    float phi;
    if (reflection) {
        phi = 2.0f * std::asin(std::clamp(
            1.0f - 2.0f * sample_random.y, -1.0f, 1.0f)) *
              closure.roughness_v;
    } else {
        const auto c = 2.0f * std::atan2(
            half_pi / closure.roughness_v, 1.0f);
        const auto p = closure.roughness_v *
                       std::tan(c * (sample_random.y - 0.5f));
        phi = p + pi;
    }
    return add(
        subtract(
            scale(frame.y, std::cos(phi) * std::cos(theta_i)),
            scale(frame.x, std::sin(phi) * std::cos(theta_i))),
        scale(frame.tangent, std::sin(theta_i)));
}

[[nodiscard]] ShaderGraph make_hair_graph(
    std::string component, bool tangent_linked) {
    CyclesNormalizedShaderGraph source;
    if (tangent_linked) {
        source.nodes.emplace_back(CyclesNode{
            .id = 10u,
            .type = "geometry",
            .variant = {},
            .label = "Geometry",
            .inputs = {},
            .properties = {}});
    }
    auto tangent = CyclesInput{
        .socket = "Tangent",
        .source = std::nullopt,
        .value = SocketValue::vector({0.0f, 0.0f, 0.0f})};
    if (tangent_linked) {
        tangent.source = CyclesOutputRef{
            .node = 10u, .socket = "Tangent"};
        tangent.value = std::nullopt;
    }
    source.nodes.emplace_back(CyclesNode{
        .id = 11u,
        .type = "hair_bsdf",
        .variant = {},
        .label = "Cycles 5.2 legacy Hair regression",
        .inputs =
            {{.socket = "Color",
                 .source = std::nullopt,
                 .value = SocketValue::color(
                     {authored_color.x,
                         authored_color.y,
                         authored_color.z})},
                {.socket = "Offset",
                    .source = std::nullopt,
                    .value = SocketValue::floating(authored_offset)},
                {.socket = "RoughnessU",
                    .source = std::nullopt,
                    .value = SocketValue::floating(
                        authored_roughness_u)},
                {.socket = "RoughnessV",
                    .source = std::nullopt,
                    .value = SocketValue::floating(
                        authored_roughness_v)},
                std::move(tangent)},
        .properties =
            {{"component", SocketValue::string(
                               std::move(component))}}});
    source.set_root(
        ShaderDomain::surface,
        CyclesOutputRef{.node = 11u, .socket = "BSDF"});
    auto adapted = adapt_cycles_shader_graph(
        source, make_core_cycles_node_mappings());
    if (!adapted.ok()) {
        throw std::runtime_error{
            "failed to adapt legacy Hair graph"};
    }
    return std::move(*adapted.graph);
}

[[nodiscard]] CompiledSurface compile_graph(
    ShaderCompiler &compiler,
    std::string component,
    bool tangent_linked,
    ClosureOperation expected_operation) {
    const auto shader = compiler.compile(
        make_hair_graph(std::move(component), tangent_linked));
    if (!shader.ok()) {
        throw std::runtime_error{
            "failed to compile legacy Hair graph"};
    }
    const auto surface = compile_surface_program(*shader.program);
    if (!surface.ok() ||
        surface.program->closure_instructions().size() != 1u) {
        throw std::runtime_error{
            "failed to lower legacy Hair graph"};
    }
    const auto &closure =
        surface.program->closure_instructions().front();
    if (closure.operation != expected_operation ||
        closure.hair_tangent_linked != tangent_linked) {
        throw std::runtime_error{
            "legacy Hair static component or tangent topology was lost"};
    }
    for (const auto &parameter : surface.program->parameters()) {
        if (parameter.socket == "Weight" ||
            parameter.socket == "Component") {
            throw std::runtime_error{
                "legacy Hair static/internal input leaked into the ABI"};
        }
    }
    auto parameters = parameter_data(*surface.program);
    return {
        .program = surface.program,
        .closure_plan = merged_surface_closure_plan(
            *surface.program, parameters),
        .parameters = std::move(parameters)};
}

[[nodiscard]] std::vector<luisa::float4>
make_clamp_parameter_data(const CompiledSurface &surface) {
    auto result = surface.parameters;
    auto found_u = false;
    auto found_v = false;
    for (auto index = std::size_t{0u};
         index < surface.program->parameters().size();
         ++index) {
        const auto &parameter = surface.program->parameters()[index];
        if (parameter.socket == "RoughnessU") {
            result[index].x = 0.0001f;
            found_u = true;
        } else if (parameter.socket == "RoughnessV") {
            result[index].x = 1.4f;
            found_v = true;
        }
    }
    if (!found_u || !found_v) {
        throw std::runtime_error{
            "legacy Hair roughness inputs are absent from the parameter ABI"};
    }
    return result;
}

[[nodiscard]] SurfacePoint hair_point(
    Expr<bool> is_curve,
    bool sampled_transmission_same_side) noexcept {
    auto point = make_surface_point();
    point.geometric_normal = sampled_transmission_same_side
                                 ? normalize(make_float3(
                                       sampled_transmission_same_side_normal.x,
                                       sampled_transmission_same_side_normal.y,
                                       sampled_transmission_same_side_normal.z))
                                 : make_float3(0.0f, 0.0f, 1.0f);
    point.shading_normal = point.geometric_normal;
    point.object_shading_normal = point.geometric_normal;
    point.undisplaced_shading_normal = point.geometric_normal;
    point.undisplaced_object_shading_normal = point.geometric_normal;
    point.dpdu = normalize(make_float3(
        dpdu_source.x, dpdu_source.y, dpdu_source.z));
    point.dpdv = normalize(make_float3(
        dpdv_source.x, dpdv_source.y, dpdv_source.z));
    point.incoming = normalize(make_float3(
        incoming_source.x,
        incoming_source.y,
        incoming_source.z));
    point.is_curve = is_curve;
    return point;
}

[[nodiscard]] SurfaceQuery hair_query(
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
    bool reflection,
    bool tangent_linked,
    bool force_curve = false,
    bool sampled_transmission_same_side = false) {
    const auto closure_reachability = SurfaceClosureReachability{
        .kinds = surface_closure_kind_bit(
            reflection ? SurfaceClosureKind::hair_reflection
                       : SurfaceClosureKind::hair_transmission)};
    return Kernel1D{[surface = &surfaces,
                       surface_tag,
                       reflection,
                       tangent_linked,
                       force_curve,
                       sampled_transmission_same_side,
                       closure_reachability](BufferFloat4 parameters,
                       BufferFloat4 output) noexcept {
        const auto case_index = dispatch_x();
        const auto is_curve = force_curve
                                  ? Bool{true}
                                  : !tangent_linked & (case_index != 0u);
        ParameterShaderServices services{parameters};
        const auto point = hair_point(
            is_curve, sampled_transmission_same_side);
        const auto query = hair_query();

        SurfaceClosureSet closures{1u};
        static_cast<void>(surface->collect_closures(
            UInt{surface_tag}, services, point, true, true, closures));
        const auto closure = closures.entry(0u);
        const auto blocks = pack_surface_closure_physical(closure);
        const auto common = unpack_surface_closure_physical_common(
            Expr<luisa::float4x4>{blocks.block_0.expression()});
        const auto hair = unpack_surface_closure_physical_hair(
            common,
            Expr<luisa::float4x4>{blocks.block_1.expression()});
        const auto trace = surface->closure_trace(
            UInt{surface_tag}, services, point, 0u);
        const auto aov = surface_aov(
            *surface, UInt{surface_tag}, services, point);

        const auto correct_direction = reflection
                                           ? make_float3(
                                                 reflection_outgoing_source.x,
                                                 reflection_outgoing_source.y,
                                                 reflection_outgoing_source.z)
                                           : make_float3(
                                                 transmission_outgoing_source.x,
                                                 transmission_outgoing_source.y,
                                                 transmission_outgoing_source.z);
        const auto wrong_direction = reflection
                                         ? make_float3(
                                               transmission_outgoing_source.x,
                                               transmission_outgoing_source.y,
                                               transmission_outgoing_source.z)
                                         : make_float3(
                                               reflection_outgoing_source.x,
                                               reflection_outgoing_source.y,
                                               reflection_outgoing_source.z);
        const auto evaluation = surface->evaluate(
            UInt{surface_tag}, services, point,
            normalize(correct_direction), query);
        const auto wrong_evaluation = surface->evaluate(
            UInt{surface_tag}, services, point,
            normalize(wrong_direction), query);
        const auto sample = surface->sample_trace(
            UInt{surface_tag}, services, point, 0.41f,
            make_float2(sample_random.x, sample_random.y), query);

        const auto all_selection = surface_closure_selection(
            make_surface_closure_selection_context(query),
            common,
            true,
            closure_reachability);
        const auto glossy_selection = surface_closure_selection(
            make_surface_closure_selection_context(hair_query(
                static_cast<std::uint32_t>(event_glossy))),
            common,
            false,
            closure_reachability);
        const auto transmission_selection = surface_closure_selection(
            make_surface_closure_selection_context(hair_query(
                static_cast<std::uint32_t>(event_transmission))),
            common,
            false,
            closure_reachability);

        const auto base = case_index * record::count;
        output.write(base + record::identity,
            make_float4(cast<float>(common.closure_type),
                cast<float>(common.microfacet_fresnel),
                cast<float>(closures.count()),
                select(0.0f, 1.0f,
                    common.closure_type != cycles_closure::type_none)));
        output.write(base + record::trace,
            make_float4(cast<float>(trace.type),
                trace.sample_weight,
                cast<float>(trace.runtime_flags),
                select(0.0f, 1.0f, trace.valid)));
        output.write(base + record::weight,
            make_float4(common.weight, common.sample_weight));
        output.write(base + record::tangent,
            make_float4(hair.payload.tangent, hair.payload.offset));
        output.write(base + record::shape,
            make_float4(hair.payload.roughness_u,
                hair.payload.roughness_v,
                common.roughness,
                common.sample_weight));
        output.write(base + record::aov_albedo,
            make_float4(aov.albedo, aov.roughness.x));
        output.write(base + record::aov_glossy,
            make_float4(aov.glossy_albedo, aov.roughness.y));
        output.write(base + record::aov_transmission,
            make_float4(aov.transmission_albedo, aov.normal.z));
        output.write(base + record::evaluation,
            make_float4(evaluation.f, evaluation.pdf));
        output.write(base + record::evaluation_glossy,
            make_float4(evaluation.glossy_f,
                cast<float>(evaluation.events)));
        output.write(base + record::evaluation_aux,
            make_float4(evaluation.average_roughness_squared,
                evaluation.diffuse_pdf,
                0.0f,
                0.0f));
        output.write(base + record::wrong_hemisphere,
            make_float4(wrong_evaluation.f,
                wrong_evaluation.pdf));
        output.write(base + record::sample_direction,
            make_float4(sample.sample.wi,
                select(0.0f, 1.0f, sample.sample.valid)));
        output.write(base + record::sample_evaluation,
            make_float4(sample.sample.evaluation.f,
                sample.sample.evaluation.pdf));
        output.write(base + record::sample_glossy,
            make_float4(sample.sample.evaluation.glossy_f,
                cast<float>(sample.sample.evaluation.events)));
        output.write(base + record::sample_material,
            make_float4(sample.sample.roughness,
                sample.sample.eta,
                sample.sample.evaluation.average_roughness_squared));
        output.write(base + record::sample_trace,
            make_float4(cast<float>(sample.closure_type),
                sample.closure_sample_weight,
                sample.selection_rescaled,
                select(0.0f, 1.0f, sample.closure_valid)));
        output.write(base + record::selection,
            make_float4(all_selection.weight,
                glossy_selection.weight,
                transmission_selection.weight,
                cast<float>(all_selection.closure_type)));
    }};
}

[[nodiscard]] bool close_record(
    std::string_view backend,
    std::string_view scenario,
    std::uint32_t record_index,
    luisa::float4 actual,
    luisa::float4 expected,
    float tolerance = 5.0e-4f) noexcept {
    if (approximately_equal(actual, expected, tolerance)) {
        return true;
    }
    std::cerr << "legacy Hair " << scenario << " failed on " << backend
              << " at record " << record_index << ": got {"
              << actual.x << ", " << actual.y << ", " << actual.z
              << ", " << actual.w << "}, expected {"
              << expected.x << ", " << expected.y << ", "
              << expected.z << ", " << expected.w << "}\n";
    return false;
}

[[nodiscard]] bool verify_scenario(
    std::string_view backend,
    std::string_view scenario,
    std::span<const luisa::float4, record::count> actual,
    bool reflection,
    bool is_curve,
    bool tangent_linked,
    bool sampled_transmission_same_side = false) noexcept {
    const auto incoming = normalize(incoming_source);
    const auto normal = normalize(
        sampled_transmission_same_side
            ? sampled_transmission_same_side_normal
            : HostVector{0.0f, 0.0f, 1.0f});
    const auto tangent = normalize(
        (tangent_linked || is_curve) ? dpdu_source : dpdv_source);
    const auto offset =
        (tangent_linked || is_curve) ? -authored_offset : 0.0f;
    const auto closure = HairReference{
        .tangent = tangent,
        .offset = offset,
        .roughness_u = physical_roughness_u,
        .roughness_v = physical_roughness_v};
    const auto outgoing = normalize(
        reflection ? reflection_outgoing_source
                   : transmission_outgoing_source);
    const auto evaluation = hair_evaluation(
        closure, incoming, outgoing, normal, reflection);
    const auto sampled_direction = hair_sample(
        closure, incoming, reflection);
    const auto sampled_evaluation = hair_evaluation(
        closure,
        incoming,
        sampled_direction,
        normal,
        reflection,
        true);
    const auto kind = reflection
                          ? SurfaceClosureKind::hair_reflection
                          : SurfaceClosureKind::hair_transmission;
    const auto type = reflection
                          ? cycles_closure::type_hair_reflection
                          : cycles_closure::type_hair_transmission;
    const auto runtime_flags =
        cycles_closure::runtime_bsdf |
        cycles_closure::runtime_bsdf_has_eval |
        (reflection
             ? 0u
             : cycles_closure::runtime_bsdf_has_transmission);
    const auto events = static_cast<float>(
        event_glossy |
        (reflection ? event_reflection : event_transmission));
    const auto weighted_evaluation = luisa::float3{
        authored_color.x * evaluation,
        authored_color.y * evaluation,
        authored_color.z * evaluation};
    const auto weighted_sample = luisa::float3{
        authored_color.x * sampled_evaluation,
        authored_color.y * sampled_evaluation,
        authored_color.z * sampled_evaluation};
    const auto zero = luisa::float3{0.0f};
    const auto expected_glossy =
        reflection ? authored_color : zero;
    const auto expected_transmission =
        reflection ? zero : authored_color;
    const auto evaluation_glossy =
        reflection ? weighted_evaluation : zero;
    const auto sample_glossy =
        reflection ? weighted_sample : zero;
    const auto expected = std::array{
        luisa::float4{static_cast<float>(type),
            static_cast<float>(
                cycles_closure::MicrofacetFresnel::none),
            1.0f, 1.0f},
        luisa::float4{static_cast<float>(type), closure_sample_weight,
            static_cast<float>(runtime_flags), 1.0f},
        luisa::float4{authored_color.x, authored_color.y,
            authored_color.z, closure_sample_weight},
        luisa::float4{tangent.x, tangent.y, tangent.z, offset},
        luisa::float4{physical_roughness_u, physical_roughness_v,
            physical_roughness_u, closure_sample_weight},
        luisa::float4{0.0f, 0.0f, 0.0f, 1.0f},
        luisa::float4{expected_glossy.x, expected_glossy.y,
            expected_glossy.z, 1.0f},
        luisa::float4{expected_transmission.x,
            expected_transmission.y, expected_transmission.z, normal.z},
        luisa::float4{weighted_evaluation.x, weighted_evaluation.y,
            weighted_evaluation.z, evaluation},
        luisa::float4{evaluation_glossy.x, evaluation_glossy.y,
            evaluation_glossy.z, events},
        luisa::float4{1.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{0.0f},
        luisa::float4{sampled_direction.x, sampled_direction.y,
            sampled_direction.z, 1.0f},
        luisa::float4{weighted_sample.x, weighted_sample.y,
            weighted_sample.z, sampled_evaluation},
        luisa::float4{sample_glossy.x, sample_glossy.y,
            sample_glossy.z, events},
        luisa::float4{physical_roughness_u, physical_roughness_v,
            1.0f, 1.0f},
        luisa::float4{static_cast<float>(type), closure_sample_weight,
            0.41f, 1.0f},
        luisa::float4{closure_sample_weight,
            reflection ? closure_sample_weight : 0.0f,
            0.0f,
            static_cast<float>(type)}};
    static_assert(expected.size() == record::count);
    for (auto index = std::uint32_t{0u};
         index < record::count;
         ++index) {
        if (!close_record(
                backend, scenario, index, actual[index], expected[index])) {
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
    const auto reflection = compile_graph(
        compiler, "reflection", false,
        ClosureOperation::hair_reflection);
    const auto transmission = compile_graph(
        compiler, "transmission", true,
        ClosureOperation::hair_transmission);
    const auto ribbon_transmission = compile_graph(
        compiler, "transmission", false,
        ClosureOperation::hair_transmission);

    SurfaceDispatch reflection_surfaces;
    const auto reflection_tag =
        reflection_surfaces.create<GraphSurface>(
            reflection.program, reflection.closure_plan);
    SurfaceDispatch transmission_surfaces;
    const auto transmission_tag =
        transmission_surfaces.create<GraphSurface>(
            transmission.program, transmission.closure_plan);
    SurfaceDispatch ribbon_transmission_surfaces;
    const auto ribbon_transmission_tag =
        ribbon_transmission_surfaces.create<GraphSurface>(
            ribbon_transmission.program,
            ribbon_transmission.closure_plan);
    const auto reflection_kernel = make_test_kernel(
        reflection_surfaces, reflection_tag, true, false);
    const auto transmission_kernel = make_test_kernel(
        transmission_surfaces, transmission_tag, false, true);
    const auto ribbon_transmission_kernel = make_test_kernel(
        ribbon_transmission_surfaces,
        ribbon_transmission_tag,
        false,
        false,
        true,
        true);
    if (backend == "fallback") {
        auto bounded = true;
        bounded &= require_bounded_xir(
            "legacy_hair_reflection", reflection_kernel, 45000u);
        bounded &= require_bounded_xir(
            "legacy_hair_transmission", transmission_kernel, 45000u);
        bounded &= require_bounded_xir(
            "legacy_hair_ribbon_transmission",
            ribbon_transmission_kernel,
            45000u);
        if (!bounded) {
            return EXIT_FAILURE;
        }
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto reflection_parameters =
        device.create_buffer<luisa::float4>(reflection.parameters.size());
    const auto clamp_parameter_data =
        make_clamp_parameter_data(reflection);
    auto clamp_parameters =
        device.create_buffer<luisa::float4>(clamp_parameter_data.size());
    auto transmission_parameters =
        device.create_buffer<luisa::float4>(transmission.parameters.size());
    auto ribbon_transmission_parameters =
        device.create_buffer<luisa::float4>(
            ribbon_transmission.parameters.size());
    auto reflection_output =
        device.create_buffer<luisa::float4>(2u * record::count);
    auto clamp_output =
        device.create_buffer<luisa::float4>(record::count);
    auto transmission_output =
        device.create_buffer<luisa::float4>(record::count);
    auto ribbon_transmission_output =
        device.create_buffer<luisa::float4>(record::count);
    auto reflection_shader = compile_named_kernel(
        device, "legacy_hair_reflection", reflection_kernel);
    auto transmission_shader = compile_named_kernel(
        device, "legacy_hair_transmission", transmission_kernel);
    auto ribbon_transmission_shader = compile_named_kernel(
        device,
        "legacy_hair_ribbon_transmission",
        ribbon_transmission_kernel);
    std::array<luisa::float4, 2u * record::count> reflection_actual{};
    std::array<luisa::float4, record::count> clamp_actual{};
    std::array<luisa::float4, record::count> transmission_actual{};
    std::array<luisa::float4, record::count>
        ribbon_transmission_actual{};
    stream << reflection_parameters.copy_from(
                  luisa::span{reflection.parameters})
           << transmission_parameters.copy_from(
                  luisa::span{transmission.parameters})
           << ribbon_transmission_parameters.copy_from(
                  luisa::span{ribbon_transmission.parameters})
           << clamp_parameters.copy_from(
                  luisa::span{clamp_parameter_data})
           << reflection_shader(
                  reflection_parameters, reflection_output)
                  .dispatch(2u)
           << transmission_shader(
                  transmission_parameters, transmission_output)
                  .dispatch(1u)
           << ribbon_transmission_shader(
                  ribbon_transmission_parameters,
                  ribbon_transmission_output)
                  .dispatch(1u)
           << reflection_shader(clamp_parameters, clamp_output)
                  .dispatch(1u)
           << reflection_output.copy_to(
                  luisa::span{reflection_actual})
           << transmission_output.copy_to(
                  luisa::span{transmission_actual})
           << ribbon_transmission_output.copy_to(
                  luisa::span{ribbon_transmission_actual})
           << clamp_output.copy_to(luisa::span{clamp_actual})
           << synchronize();

    const auto triangle =
        std::span<const luisa::float4, record::count>{
            reflection_actual.data(), record::count};
    const auto curve =
        std::span<const luisa::float4, record::count>{
            reflection_actual.data() + record::count,
            record::count};
    const auto linked =
        std::span<const luisa::float4, record::count>{
            transmission_actual.data(), record::count};
    const auto ribbon_transmission_span =
        std::span<const luisa::float4, record::count>{
            ribbon_transmission_actual.data(), record::count};
    const auto clamp_ok = close_record(
        backend,
        "roughness clamp",
        record::shape,
        clamp_actual[record::shape],
        {clamped_roughness_u,
            clamped_roughness_v,
            clamped_roughness_u,
            closure_sample_weight});
    auto ok = clamp_ok;
    ok &= verify_scenario(
        backend, "unlinked triangle reflection", triangle,
        true, false, false);
    ok &= verify_scenario(
        backend, "unlinked curve reflection", curve,
        true, true, false);
    ok &= verify_scenario(
        backend, "linked triangle transmission", linked,
        false, false, true);
    ok &= verify_scenario(
        backend,
        "unlinked ribbon transmission with sampled same-side normal",
        ribbon_transmission_span,
        false,
        true,
        false,
        true);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

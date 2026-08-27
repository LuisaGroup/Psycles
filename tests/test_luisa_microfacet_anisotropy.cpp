#include <psycles/adapter/cycles_shader_graph.h>
#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_set.h>

#include "graph_surface_internal.h"
#include "luisa_shader_shape_test_support.h"
#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
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

constexpr float roughness = 0.5f;
constexpr float anisotropy = 0.6f;
constexpr float quarter_turn = 0.25f;
constexpr luisa::float2 sample_random{0.37f, 0.64f};

namespace record {
constexpr std::uint32_t glossy_state = 0u;
constexpr std::uint32_t glossy_state_tail = 1u;
constexpr std::uint32_t rotated_state = 2u;
constexpr std::uint32_t rotated_state_tail = 3u;
constexpr std::uint32_t principled_state = 4u;
constexpr std::uint32_t principled_state_tail = 5u;
constexpr std::uint32_t glossy_sample = 6u;
constexpr std::uint32_t rotated_sample = 7u;
constexpr std::uint32_t glossy_evaluation = 8u;
constexpr std::uint32_t rotated_evaluation = 9u;
constexpr std::uint32_t beckmann_state = 10u;
constexpr std::uint32_t beckmann_state_tail = 11u;
constexpr std::uint32_t rotated_beckmann_state = 12u;
constexpr std::uint32_t rotated_beckmann_state_tail = 13u;
constexpr std::uint32_t beckmann_sample = 14u;
constexpr std::uint32_t rotated_beckmann_sample = 15u;
constexpr std::uint32_t beckmann_evaluation = 16u;
constexpr std::uint32_t rotated_beckmann_evaluation = 17u;
constexpr std::uint32_t sampled_fresnel_direction = 18u;
constexpr std::uint32_t sampled_fresnel_values = 19u;
constexpr std::uint32_t sampled_fresnel_witness = 20u;
constexpr std::uint32_t count = 21u;
} // namespace record

[[nodiscard]] ShaderGraph
make_anisotropic_graph(bool principled, float rotation,
                       float anisotropy_value = anisotropy,
                       std::string_view distribution = "GGX") {
  CyclesNormalizedShaderGraph source;
  using NormalizedNode = typename decltype(source.nodes)::value_type;
  source.nodes.emplace_back(NormalizedNode{.id = 10u,
                                           .type = "geometry",
                                           .variant = {},
                                           .label = "Geometry",
                                           .inputs = {},
                                           .properties = {}});
  if (principled) {
    source.nodes.emplace_back(NormalizedNode{
        .id = 11u,
        .type = "principled_bsdf",
        .variant = {},
        .label = "Principled anisotropy regression",
        .inputs = {{.socket = "Base Color",
                    .source = std::nullopt,
                    .value = SocketValue::color({0.8f, 0.6f, 0.3f})},
                   {.socket = "Metallic",
                    .source = std::nullopt,
                    .value = SocketValue::floating(1.0f)},
                   {.socket = "Roughness",
                    .source = std::nullopt,
                    .value = SocketValue::floating(roughness)},
                   {.socket = "Anisotropic",
                    .source = std::nullopt,
                    .value = SocketValue::floating(anisotropy_value)},
                   {.socket = "Anisotropic Rotation",
                    .source = std::nullopt,
                    .value = SocketValue::floating(rotation)},
                   {.socket = "Tangent",
                    .source = CyclesOutputRef{.node = 10u, .socket = "Tangent"},
                    .value = std::nullopt},
                   {.socket = "Normal",
                    .source = CyclesOutputRef{.node = 10u, .socket = "Normal"},
                    .value = std::nullopt}},
        .properties = {
            {"distribution", SocketValue::string(std::string{distribution})}}});
  } else {
    source.nodes.emplace_back(NormalizedNode{
        .id = 11u,
        .type = "glossy_bsdf",
        .variant = {},
        .label = "Glossy anisotropy regression",
        .inputs = {{.socket = "Color",
                    .source = std::nullopt,
                    .value = SocketValue::color({1.0f, 1.0f, 1.0f})},
                   {.socket = "Roughness",
                    .source = std::nullopt,
                    .value = SocketValue::floating(roughness)},
                   {.socket = "Anisotropy",
                    .source = std::nullopt,
                    .value = SocketValue::floating(anisotropy_value)},
                   {.socket = "Rotation",
                    .source = std::nullopt,
                    .value = SocketValue::floating(rotation)},
                   {.socket = "Tangent",
                    .source = CyclesOutputRef{.node = 10u, .socket = "Tangent"},
                    .value = std::nullopt},
                   {.socket = "Normal",
                    .source = CyclesOutputRef{.node = 10u, .socket = "Normal"},
                    .value = std::nullopt}},
        .properties = {
            {"distribution", SocketValue::string(std::string{distribution})}}});
  }
  source.set_root(ShaderDomain::surface,
                  CyclesOutputRef{.node = 11u, .socket = "BSDF"});
  auto adapted =
      adapt_cycles_shader_graph(source, make_core_cycles_node_mappings());
  if (!adapted.ok()) {
    throw std::runtime_error{"failed to adapt anisotropic Cycles graph"};
  }
  return std::move(*adapted.graph);
}

struct CompiledSurface {
  std::shared_ptr<const SurfaceProgram> program;
  SurfaceClosurePlan closure_plan;
  std::vector<luisa::float4> parameters;
};

[[nodiscard]] CompiledSurface compile_graph(ShaderCompiler &compiler,
                                            ShaderGraph graph) {
  const auto shader = compiler.compile(graph);
  if (!shader.ok()) {
    throw std::runtime_error{"failed to compile anisotropic shader graph"};
  }
  const auto surface = compile_surface_program(*shader.program);
  if (!surface.ok()) {
    throw std::runtime_error{"failed to lower anisotropic surface program"};
  }
  auto parameters = parameter_data(*surface.program);
  return {.program = surface.program,
          .closure_plan = merged_surface_closure_plan(
              *surface.program, parameters),
          .parameters = std::move(parameters)};
}

[[nodiscard]] SurfacePoint anisotropic_point() noexcept {
  auto point = make_surface_point();
  point.incoming = make_float3(0.0f, 0.0f, 1.0f);
  point.dpdu = make_float3(1.0f, 0.0f, 0.0f);
  return point;
}

[[nodiscard]] SurfaceQuery anisotropic_query() noexcept {
  return {.lobe_mask = static_cast<std::uint32_t>(event_diffuse | event_glossy |
                                                  event_transmission |
                                                  event_transparent),
          .transport_mode = static_cast<std::uint32_t>(TransportMode::radiance),
          .glossy_filter_roughness = 0.0f,
          .reflective_caustics = true,
          .refractive_caustics = true};
}

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z) && std::isfinite(value.w);
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  ShaderCompiler compiler{make_core_node_registry()};
  const auto glossy =
      compile_graph(compiler, make_anisotropic_graph(false, 0.0f));
  const auto rotated =
      compile_graph(compiler, make_anisotropic_graph(false, quarter_turn));
  const auto principled =
      compile_graph(compiler, make_anisotropic_graph(true, quarter_turn));
  const auto beckmann = compile_graph(
      compiler, make_anisotropic_graph(false, 0.0f, -anisotropy, "BECKMANN"));
  const auto rotated_beckmann =
      compile_graph(compiler, make_anisotropic_graph(false, quarter_turn,
                                                     -anisotropy, "BECKMANN"));

  SurfaceDispatch glossy_surfaces;
  const auto glossy_tag = glossy_surfaces.create<GraphSurface>(
      glossy.program, glossy.closure_plan);
  SurfaceDispatch rotated_surfaces;
  const auto rotated_tag = rotated_surfaces.create<GraphSurface>(
      rotated.program, rotated.closure_plan);
  SurfaceDispatch principled_surfaces;
  const auto principled_tag = principled_surfaces.create<GraphSurface>(
      principled.program, principled.closure_plan);
  SurfaceDispatch beckmann_surfaces;
  const auto beckmann_tag = beckmann_surfaces.create<GraphSurface>(
      beckmann.program, beckmann.closure_plan);
  SurfaceDispatch rotated_beckmann_surfaces;
  const auto rotated_beckmann_tag =
      rotated_beckmann_surfaces.create<GraphSurface>(
          rotated_beckmann.program, rotated_beckmann.closure_plan);

  Kernel1D test = [&](BufferFloat4 glossy_parameters,
                      BufferFloat4 rotated_parameters,
                      BufferFloat4 principled_parameters,
                      BufferFloat4 output) noexcept {
    const auto point = anisotropic_point();
    const auto query = anisotropic_query();
    ParameterShaderServices glossy_services{glossy_parameters};
    ParameterShaderServices rotated_services{rotated_parameters};
    ParameterShaderServices principled_services{principled_parameters};

    SurfaceClosureSet glossy_closures{1u};
    static_cast<void>(glossy_surfaces.collect_closures(
        UInt{glossy_tag}, glossy_services, point, true, true, glossy_closures));
    const auto glossy_closure = glossy_closures.entry(0u);
    output.write(record::glossy_state,
                 make_float4(glossy_closure.microfacet_tangent,
                             glossy_closure.microfacet_alpha_x));
    output.write(record::glossy_state_tail,
                 make_float4(glossy_closure.microfacet_alpha_y,
                             cast<float>(glossy_closures.count()), 0.0f, 0.0f));

    SurfaceClosureSet rotated_closures{1u};
    static_cast<void>(
        rotated_surfaces.collect_closures(UInt{rotated_tag}, rotated_services,
                                          point, true, true, rotated_closures));
    const auto rotated_closure = rotated_closures.entry(0u);
    output.write(record::rotated_state,
                 make_float4(rotated_closure.microfacet_tangent,
                             rotated_closure.microfacet_alpha_x));
    output.write(record::rotated_state_tail,
                 make_float4(rotated_closure.microfacet_alpha_y,
                             cast<float>(rotated_closures.count()), 0.0f,
                             0.0f));

    SurfaceClosureSet principled_closures{2u};
    static_cast<void>(principled_surfaces.collect_closures(
        UInt{principled_tag}, principled_services, point, true, true,
        principled_closures));
    const auto principled_closure = principled_closures.entry(0u);
    output.write(record::principled_state,
                 make_float4(principled_closure.microfacet_tangent,
                             principled_closure.microfacet_alpha_x));
    output.write(record::principled_state_tail,
                 make_float4(principled_closure.microfacet_alpha_y,
                             cast<float>(principled_closure.lobe),
                             cast<float>(principled_closures.count()), 0.0f));

    const auto glossy_sample =
        glossy_surfaces.sample_trace(UInt{glossy_tag}, glossy_services, point,
                                     0.5f, make_float2(sample_random), query);
    const auto rotated_sample = rotated_surfaces.sample_trace(
        UInt{rotated_tag}, rotated_services, point, 0.5f,
        make_float2(sample_random), query);
    output.write(
        record::glossy_sample,
        make_float4(glossy_sample.sample.wi,
                    glossy_sample.sample.evaluation.average_roughness_squared));
    output.write(
        record::rotated_sample,
        make_float4(
            rotated_sample.sample.wi,
            rotated_sample.sample.evaluation.average_roughness_squared));
    output.write(record::glossy_evaluation,
                 make_float4(glossy_sample.sample.evaluation.f,
                             glossy_sample.sample.evaluation.pdf));
    output.write(record::rotated_evaluation,
                 make_float4(rotated_sample.sample.evaluation.f,
                             rotated_sample.sample.evaluation.pdf));
  };

  Kernel1D test_sampled_fresnel = [&](BufferFloat4 output) noexcept {
    const auto point = anisotropic_point();
    const auto closure_point = SurfaceClosurePoint{point};
    const SurfaceClosurePhysicalGeneralRecord closure{
        .common =
            {.kind = static_cast<std::uint32_t>(SurfaceClosureKind::principled),
             .lobe = static_cast<std::uint32_t>(SurfaceClosureLobe::metallic),
             .weight = make_float3(1.0f),
             .allocation_weight = 1.0f,
             .sample_weight = 1.0f,
             .setup_valid = true,
             .color_or_evaluation_scale = make_float3(0.0f),
             .normal = point.shading_normal,
             .roughness = roughness,
             .preserve_ggx_energy = false,
             .beckmann = false,
             .bssrdf_method =
                 static_cast<std::uint32_t>(SurfaceBssrdfMethod::random_walk)},
        .payload = {.thin_film_thickness = 0.0f,
                    .thin_film_ior = 0.0f,
                    .ior = 1.5f,
                    .specular_tint = make_float3(0.0f),
                    .sheen_transform_a = 0.0f,
                    .sheen_transform_b = 0.0f,
                    .evaluation_scale = make_float3(1.0f),
                    .microfacet_tangent = make_float3(1.0f, 0.0f, 0.0f),
                    .microfacet_alpha_x = roughness * roughness,
                    .microfacet_alpha_y = roughness * roughness}};
    const auto sample =
        psycles::luisa_backend::detail::sample_microfacet_reflection(
            closure_point, point.shading_normal, closure, point.incoming,
            make_float2(sample_random), point.shading_normal, 0.0f, false);
    const auto sampled_half_vector =
        normalize(point.incoming + sample.direction);
    const auto normal_cosine = dot(point.shading_normal, point.incoming);
    const auto half_cosine = dot(sampled_half_vector, point.incoming);
    const auto normal_fresnel =
        psycles::luisa_backend::detail::microfacet_reflection_fresnel(
            closure, normal_cosine);
    const auto half_fresnel =
        psycles::luisa_backend::detail::microfacet_reflection_fresnel(
            closure, half_cosine);
    output.write(record::sampled_fresnel_direction,
                 make_float4(sample.direction, cast<float>(sample.valid)));
    output.write(record::sampled_fresnel_values,
                 make_float4(normal_fresnel.x, half_fresnel.x,
                             sample.singular_evaluation.x * 1.0e-6f,
                             cast<float>(sample.singular)));
    output.write(record::sampled_fresnel_witness,
                 make_float4(normal_cosine, half_cosine, sample.roughness));
  };

  Kernel1D test_beckmann = [&](BufferFloat4 beckmann_parameters,
                               BufferFloat4 rotated_parameters,
                               BufferFloat4 output) noexcept {
    const auto point = anisotropic_point();
    const auto query = anisotropic_query();
    ParameterShaderServices beckmann_services{beckmann_parameters};
    ParameterShaderServices rotated_services{rotated_parameters};

    SurfaceClosureSet beckmann_closures{1u};
    static_cast<void>(beckmann_surfaces.collect_closures(
        UInt{beckmann_tag}, beckmann_services, point, true, true,
        beckmann_closures));
    const auto beckmann_closure = beckmann_closures.entry(0u);
    output.write(record::beckmann_state,
                 make_float4(beckmann_closure.microfacet_tangent,
                             beckmann_closure.microfacet_alpha_x));
    output.write(record::beckmann_state_tail,
                 make_float4(beckmann_closure.microfacet_alpha_y,
                             cast<float>(beckmann_closures.count()), 0.0f,
                             0.0f));

    SurfaceClosureSet rotated_closures{1u};
    static_cast<void>(rotated_beckmann_surfaces.collect_closures(
        UInt{rotated_beckmann_tag}, rotated_services, point, true, true,
        rotated_closures));
    const auto rotated_closure = rotated_closures.entry(0u);
    output.write(record::rotated_beckmann_state,
                 make_float4(rotated_closure.microfacet_tangent,
                             rotated_closure.microfacet_alpha_x));
    output.write(record::rotated_beckmann_state_tail,
                 make_float4(rotated_closure.microfacet_alpha_y,
                             cast<float>(rotated_closures.count()), 0.0f,
                             0.0f));

    const auto beckmann_sample = beckmann_surfaces.sample_trace(
        UInt{beckmann_tag}, beckmann_services, point, 0.5f,
        make_float2(sample_random), query);
    const auto rotated_sample = rotated_beckmann_surfaces.sample_trace(
        UInt{rotated_beckmann_tag}, rotated_services, point, 0.5f,
        make_float2(sample_random), query);
    output.write(
        record::beckmann_sample,
        make_float4(
            beckmann_sample.sample.wi,
            beckmann_sample.sample.evaluation.average_roughness_squared));
    output.write(
        record::rotated_beckmann_sample,
        make_float4(
            rotated_sample.sample.wi,
            rotated_sample.sample.evaluation.average_roughness_squared));
    output.write(record::beckmann_evaluation,
                 make_float4(beckmann_sample.sample.evaluation.f,
                             beckmann_sample.sample.evaluation.pdf));
    output.write(record::rotated_beckmann_evaluation,
                 make_float4(rotated_sample.sample.evaluation.f,
                             rotated_sample.sample.evaluation.pdf));
  };

  if (backend == "fallback") {
    auto bounded = require_bounded_xir("microfacet_anisotropy", test, 90000u);
    bounded &= require_bounded_xir("microfacet_sampled_fresnel",
                                   test_sampled_fresnel, 12000u);
    bounded &= require_bounded_xir("microfacet_beckmann_anisotropy",
                                   test_beckmann, 90000u);
    if (!bounded) {
      return EXIT_FAILURE;
    }
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto glossy_buffer =
      device.create_buffer<luisa::float4>(glossy.parameters.size());
  auto rotated_buffer =
      device.create_buffer<luisa::float4>(rotated.parameters.size());
  auto principled_buffer =
      device.create_buffer<luisa::float4>(principled.parameters.size());
  auto beckmann_buffer =
      device.create_buffer<luisa::float4>(beckmann.parameters.size());
  auto rotated_beckmann_buffer =
      device.create_buffer<luisa::float4>(rotated_beckmann.parameters.size());
  auto output_buffer = device.create_buffer<luisa::float4>(record::count);
  auto shader = compile_named_kernel(device, "microfacet_anisotropy", test);
  auto sampled_fresnel_shader = compile_named_kernel(
      device, "microfacet_sampled_fresnel", test_sampled_fresnel);
  auto beckmann_shader = compile_named_kernel(
      device, "microfacet_beckmann_anisotropy", test_beckmann);
  std::array<luisa::float4, record::count> actual{};
  stream << glossy_buffer.copy_from(luisa::span{glossy.parameters})
         << rotated_buffer.copy_from(luisa::span{rotated.parameters})
         << principled_buffer.copy_from(luisa::span{principled.parameters})
         << beckmann_buffer.copy_from(luisa::span{beckmann.parameters})
         << rotated_beckmann_buffer.copy_from(
                luisa::span{rotated_beckmann.parameters})
         << shader(glossy_buffer, rotated_buffer, principled_buffer,
                   output_buffer)
                .dispatch(1u)
         << sampled_fresnel_shader(output_buffer).dispatch(1u)
         << beckmann_shader(beckmann_buffer, rotated_beckmann_buffer,
                            output_buffer)
                .dispatch(1u)
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();

  constexpr auto base_alpha = roughness * roughness;
  constexpr auto glossy_alpha_x = base_alpha * (1.0f - anisotropy);
  constexpr auto glossy_alpha_y = base_alpha / (1.0f - anisotropy);
  const auto principled_aspect = std::sqrt(1.0f - 0.9f * anisotropy);
  const auto principled_alpha_x = base_alpha / principled_aspect;
  const auto principled_alpha_y = base_alpha * principled_aspect;
  const auto state_matches =
      approximately_equal(actual[record::glossy_state],
                          luisa::float4{1.0f, 0.0f, 0.0f, glossy_alpha_x},
                          2.0e-6f) &&
      approximately_equal(actual[record::glossy_state_tail],
                          luisa::float4{glossy_alpha_y, 1.0f, 0.0f, 0.0f},
                          2.0e-6f) &&
      approximately_equal(actual[record::rotated_state],
                          luisa::float4{0.0f, 1.0f, 0.0f, glossy_alpha_x},
                          2.0e-6f) &&
      approximately_equal(actual[record::rotated_state_tail],
                          luisa::float4{glossy_alpha_y, 1.0f, 0.0f, 0.0f},
                          2.0e-6f) &&
      approximately_equal(actual[record::principled_state],
                          luisa::float4{0.0f, 1.0f, 0.0f, principled_alpha_x},
                          2.0e-6f) &&
      approximately_equal(actual[record::principled_state_tail].x,
                          principled_alpha_y, 2.0e-6f) &&
      approximately_equal(actual[record::principled_state_tail].y,
                          static_cast<float>(SurfaceClosureLobe::metallic)) &&
      approximately_equal(actual[record::principled_state_tail].z, 1.0f);

  const auto unrotated_direction = actual[record::glossy_sample].xyz();
  const auto rotated_direction = actual[record::rotated_sample].xyz();
  const auto expected_rotated_direction = luisa::float3{
      -unrotated_direction.y, unrotated_direction.x, unrotated_direction.z};
  constexpr auto alpha_product = glossy_alpha_x * glossy_alpha_y;
  const auto transport_matches =
      finite(actual[record::glossy_sample]) &&
      finite(actual[record::rotated_sample]) &&
      finite(actual[record::glossy_evaluation]) &&
      finite(actual[record::rotated_evaluation]) &&
      approximately_equal(rotated_direction.x, expected_rotated_direction.x,
                          8.0e-5f) &&
      approximately_equal(rotated_direction.y, expected_rotated_direction.y,
                          8.0e-5f) &&
      approximately_equal(rotated_direction.z, expected_rotated_direction.z,
                          8.0e-5f) &&
      approximately_equal(actual[record::glossy_sample].w, alpha_product,
                          3.0e-6f) &&
      approximately_equal(actual[record::rotated_sample].w, alpha_product,
                          3.0e-6f) &&
      approximately_equal(actual[record::glossy_evaluation],
                          actual[record::rotated_evaluation], 8.0e-5f);

  // For the black metallic F82 witness, F(N.I) is exactly zero at normal
  // incidence. A non-degenerate VNDF sample has H.I < 1 and positive Fresnel.
  // The conditional sampler must use that sampled H.I witness; using N.I
  // incorrectly rejects the otherwise valid sample.
  const auto sampled_fresnel_values = actual[record::sampled_fresnel_values];
  const auto sampled_fresnel_witness = actual[record::sampled_fresnel_witness];
  const auto sampled_fresnel_matches =
      finite(actual[record::sampled_fresnel_direction]) &&
      finite(sampled_fresnel_values) && finite(sampled_fresnel_witness) &&
      approximately_equal(actual[record::sampled_fresnel_direction].w, 1.0f) &&
      approximately_equal(sampled_fresnel_values.x, 0.0f, 1.0e-8f) &&
      sampled_fresnel_values.y > 0.0f &&
      approximately_equal(sampled_fresnel_values.z, sampled_fresnel_values.y,
                          2.0e-6f) &&
      approximately_equal(sampled_fresnel_values.w, 0.0f) &&
      approximately_equal(sampled_fresnel_witness.x, 1.0f, 2.0e-6f) &&
      sampled_fresnel_witness.y > 0.0f && sampled_fresnel_witness.y < 0.9999f &&
      approximately_equal(sampled_fresnel_witness.z, roughness * roughness,
                          2.0e-6f) &&
      approximately_equal(sampled_fresnel_witness.w, roughness * roughness,
                          2.0e-6f);

  constexpr auto negative_denominator = 1.0f - anisotropy;
  constexpr auto beckmann_alpha_x = base_alpha / negative_denominator;
  constexpr auto beckmann_alpha_y = base_alpha * negative_denominator;
  const auto beckmann_state_matches =
      approximately_equal(actual[record::beckmann_state],
                          luisa::float4{1.0f, 0.0f, 0.0f, beckmann_alpha_x},
                          2.0e-6f) &&
      approximately_equal(actual[record::beckmann_state_tail],
                          luisa::float4{beckmann_alpha_y, 1.0f, 0.0f, 0.0f},
                          2.0e-6f) &&
      approximately_equal(actual[record::rotated_beckmann_state],
                          luisa::float4{0.0f, 1.0f, 0.0f, beckmann_alpha_x},
                          2.0e-6f) &&
      approximately_equal(actual[record::rotated_beckmann_state_tail],
                          luisa::float4{beckmann_alpha_y, 1.0f, 0.0f, 0.0f},
                          2.0e-6f);
  const auto beckmann_direction = actual[record::beckmann_sample].xyz();
  const auto rotated_beckmann_direction =
      actual[record::rotated_beckmann_sample].xyz();
  const auto expected_rotated_beckmann_direction = luisa::float3{
      -beckmann_direction.y, beckmann_direction.x, beckmann_direction.z};
  constexpr auto beckmann_alpha_product = beckmann_alpha_x * beckmann_alpha_y;
  const auto beckmann_transport_matches =
      finite(actual[record::beckmann_sample]) &&
      finite(actual[record::rotated_beckmann_sample]) &&
      finite(actual[record::beckmann_evaluation]) &&
      finite(actual[record::rotated_beckmann_evaluation]) &&
      approximately_equal(rotated_beckmann_direction.x,
                          expected_rotated_beckmann_direction.x, 8.0e-5f) &&
      approximately_equal(rotated_beckmann_direction.y,
                          expected_rotated_beckmann_direction.y, 8.0e-5f) &&
      approximately_equal(rotated_beckmann_direction.z,
                          expected_rotated_beckmann_direction.z, 8.0e-5f) &&
      approximately_equal(actual[record::beckmann_sample].w,
                          beckmann_alpha_product, 3.0e-6f) &&
      approximately_equal(actual[record::rotated_beckmann_sample].w,
                          beckmann_alpha_product, 3.0e-6f) &&
      approximately_equal(actual[record::beckmann_evaluation],
                          actual[record::rotated_beckmann_evaluation], 8.0e-5f);

  if (!state_matches || !transport_matches || !sampled_fresnel_matches ||
      !beckmann_state_matches || !beckmann_transport_matches) {
    std::cerr << "microfacet anisotropy regression failed on " << backend
              << '\n';
    for (auto index = 0u; index < actual.size(); ++index) {
      const auto value = actual[index];
      std::cerr << index << ": {" << value.x << ", " << value.y << ", "
                << value.z << ", " << value.w << "}\n";
    }
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

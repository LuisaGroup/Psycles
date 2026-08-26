#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/surface_closure_evaluator.h>
#include <psycles/luisa/surface_closure_operations.h>
#include <psycles/luisa/surface_closure_population.h>

#include "compact_surface_program_test_support.h"
#include "luisa_surface_test_support.h"
#include "path_tracer_attribute_lookup.h"
#include "path_tracer_bsdf_tables.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surface_closure_setup.h"
#include "path_tracer_surface_execution_domain.h"
#include "path_tracer_surface_values.h"
#include "path_tracer_surfaces.h"
#include "path_tracer_texture_sampling.h"
#include "shader_table_data.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using namespace psycles::test_support;

constexpr auto scenario_count = 8u;
constexpr auto population_closure_capacity = 12u;

struct FixtureProgram {
    std::shared_ptr<const SurfaceProgram> program;
    SurfaceParameterBlock parameters;
};

[[nodiscard]] ShaderGraph make_layered_principled_graph() {
    ShaderGraph graph;
    const auto geometry = graph.add_node(
        node_type::geometry,
        "Linked coat normal");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Layered Principled");
    const auto configured =
        graph.set_input(
            principled,
            "BaseColor",
            SocketValue::color({0.44f, 0.13f, 0.06f})) &&
        graph.set_input(
            principled,
            "Roughness",
            SocketValue::floating(0.27f)) &&
        graph.set_input(
            principled,
            "DiffuseRoughness",
            SocketValue::floating(0.16f)) &&
        graph.set_input(
            principled,
            "Metallic",
            SocketValue::floating(0.21f)) &&
        graph.set_input(
            principled,
            "TransmissionWeight",
            SocketValue::floating(0.29f)) &&
        graph.set_input(
            principled,
            "SubsurfaceWeight",
            SocketValue::floating(0.17f)) &&
        graph.set_input(
            principled,
            "SubsurfaceRadius",
            SocketValue::vector({1.0f, 0.31f, 0.12f})) &&
        graph.set_input(
            principled,
            "SubsurfaceScale",
            SocketValue::floating(0.83f)) &&
        graph.set_input(
            principled,
            "SubsurfaceIOR",
            SocketValue::floating(1.41f)) &&
        graph.set_input(
            principled,
            "SubsurfaceAnisotropy",
            SocketValue::floating(0.23f)) &&
        graph.set_input(
            principled,
            "IOR",
            SocketValue::floating(1.52f)) &&
        graph.set_input(
            principled,
            "SpecularIORLevel",
            SocketValue::floating(0.48f)) &&
        graph.set_input(
            principled,
            "SpecularTint",
            SocketValue::color({0.82f, 0.94f, 1.0f})) &&
        graph.set_input(
            principled,
            "Alpha",
            SocketValue::floating(0.71f)) &&
        graph.set_input(
            principled,
            "ThinWall",
            SocketValue::boolean(true)) &&
        graph.set_input(
            principled,
            "SheenWeight",
            SocketValue::floating(0.24f)) &&
        graph.set_input(
            principled,
            "SheenRoughness",
            SocketValue::floating(0.39f)) &&
        graph.set_input(
            principled,
            "SheenTint",
            SocketValue::color({0.91f, 0.42f, 0.18f})) &&
        graph.set_input(
            principled,
            "CoatWeight",
            SocketValue::floating(0.19f)) &&
        graph.set_input(
            principled,
            "CoatRoughness",
            SocketValue::floating(0.12f)) &&
        graph.set_input(
            principled,
            "CoatIOR",
            SocketValue::floating(1.63f)) &&
        graph.set_input(
            principled,
            "CoatTint",
            SocketValue::color({0.31f, 0.72f, 0.95f})) &&
        graph.set_input(
            principled,
            "EmissionColor",
            SocketValue::color({0.14f, 0.37f, 0.81f})) &&
        graph.set_input(
            principled,
            "EmissionStrength",
            SocketValue::floating(1.8f)) &&
        graph.set_property(
            principled,
            "Distribution",
            SocketValue::string("MULTI_GGX")) &&
        graph.set_property(
            principled,
            "SubsurfaceMethod",
            SocketValue::string("RANDOM_WALK")) &&
        graph.connect(
            {.node = geometry, .socket = "Normal"},
            principled,
            "CoatNormal");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure layered Principled graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_automatic_normal_graph() {
    ShaderGraph graph;
    const auto normal = graph.add_node(
        node_type::normal_map,
        "Automatic surface normal");
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf,
        "Automatic-normal diffuse");
    const auto configured =
        graph.set_input(
            normal,
            "Strength",
            SocketValue::floating(0.83f)) &&
        graph.set_input(
            normal,
            "Color",
            SocketValue::color({0.78f, 0.31f, 0.91f})) &&
        graph.set_property(
            normal,
            "Space",
            SocketValue::string("TANGENT")) &&
        graph.set_property(
            normal,
            "Base",
            SocketValue::string("DISPLACED")) &&
        graph.set_property(
            normal,
            "Convention",
            SocketValue::string("OPENGL")) &&
        graph.set_input(
            diffuse,
            "Color",
            SocketValue::color({0.37f, 0.61f, 0.19f})) &&
        graph.set_input(
            diffuse,
            "Roughness",
            SocketValue::floating(0.23f));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure automatic-normal graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    graph.set_root(
        ShaderDomain::surface_normal,
        OutputRef{.node = normal, .socket = "Normal"});
    return graph;
}

[[nodiscard]] ShaderGraph make_bssrdf_bump_graph(bool zero_weight) {
    ShaderGraph graph;
    const auto first_normal =
        graph.add_node(node_type::normal_map, "First BSSRDF bump normal");
    const auto second_normal =
        graph.add_node(node_type::normal_map, "Second BSSRDF bump normal");
    const auto shader_normal =
        graph.add_node(node_type::normal_map, "BSSRDF ShaderData normal");
    const auto first =
        graph.add_node(node_type::subsurface_scattering, "First BSSRDF closure");
    const auto second =
        graph.add_node(node_type::subsurface_scattering, "Second BSSRDF closure");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Ignored diffuse closure");
    const auto bssrdf_sum =
        graph.add_node(node_type::add_closure, "BSSRDF closure sum");
    const auto root =
        graph.add_node(node_type::add_closure, "BSSRDF plus diffuse");
    const auto first_color = zero_weight ? Vec3f{} : Vec3f{0.24f, 0.24f, 0.24f};
    const auto second_color = zero_weight ? Vec3f{} : Vec3f{0.71f, 0.71f, 0.71f};
    const auto configured =
        graph.set_input(first_normal, "Strength", SocketValue::floating(0.91f)) &&
        graph.set_input(first_normal, "Color",
                        SocketValue::color({0.78f, 0.29f, 0.91f})) &&
        graph.set_input(second_normal, "Strength",
                        SocketValue::floating(0.73f)) &&
        graph.set_input(second_normal, "Color",
                        SocketValue::color({0.23f, 0.81f, 0.67f})) &&
        graph.set_input(shader_normal, "Strength",
                        SocketValue::floating(0.62f)) &&
        graph.set_input(shader_normal, "Color",
                        SocketValue::color({0.62f, 0.34f, 0.88f})) &&
        graph.set_input(first, "Color", SocketValue::color(first_color)) &&
        graph.set_input(first, "Scale", SocketValue::floating(0.83f)) &&
        graph.set_input(second, "Color", SocketValue::color(second_color)) &&
        graph.set_input(second, "Scale", SocketValue::floating(1.17f)) &&
        graph.set_input(diffuse, "Color",
                        SocketValue::color({0.19f, 0.31f, 0.47f})) &&
        graph.connect({.node = first_normal, .socket = "Normal"}, first,
                      "Normal") &&
        graph.connect({.node = second_normal, .socket = "Normal"}, second,
                      "Normal") &&
        graph.connect({.node = first, .socket = "Closure"}, bssrdf_sum, "A") &&
        graph.connect({.node = second, .socket = "Closure"}, bssrdf_sum, "B") &&
        graph.connect({.node = bssrdf_sum, .socket = "Closure"}, root, "A") &&
        graph.connect({.node = diffuse, .socket = "Closure"}, root, "B");
    if (!configured) {
        throw std::runtime_error{"failed to configure BSSRDF bump graph"};
    }
    graph.set_root(ShaderDomain::surface,
                   OutputRef{.node = root, .socket = "Closure"});
    graph.set_root(ShaderDomain::surface_normal,
                   OutputRef{.node = shader_normal, .socket = "Normal"});
    return graph;
}

[[nodiscard]] ShaderGraph make_nested_bump_graph() {
  ShaderGraph graph;
  const auto inner = graph.add_node(node_type::bump, "Nested height Bump");
  const auto normal_to_vector = graph.add_node(node_type::normal_to_vector,
                                               "Nested Bump normal to vector");
  const auto vector_to_scalar = graph.add_node(node_type::vector_to_scalar,
                                               "Nested Bump vector to height");
  const auto outer = graph.add_node(node_type::bump, "Root Bump");
  const auto diffuse =
      graph.add_node(node_type::diffuse_bsdf, "Nested Bump diffuse");
  const auto configured =
      graph.set_input(inner, "Height", SocketValue::floating(0.37f)) &&
      graph.set_input(inner, "Strength", SocketValue::floating(0.61f)) &&
      graph.set_input(inner, "Distance", SocketValue::floating(0.29f)) &&
      graph.set_property(inner, "Invert", SocketValue::boolean(false)) &&
      graph.set_property(inner, "NormalLinked", SocketValue::boolean(false)) &&
      graph.set_input(outer, "Strength", SocketValue::floating(0.73f)) &&
      graph.set_input(outer, "Distance", SocketValue::floating(0.41f)) &&
      graph.set_property(outer, "Invert", SocketValue::boolean(true)) &&
      graph.set_property(outer, "NormalLinked", SocketValue::boolean(false)) &&
      graph.set_input(diffuse, "Color",
                      SocketValue::color({0.27f, 0.53f, 0.81f})) &&
      graph.set_input(diffuse, "Roughness", SocketValue::floating(0.19f)) &&
      graph.connect({.node = inner, .socket = "Normal"}, normal_to_vector,
                    "Normal") &&
      graph.connect({.node = normal_to_vector, .socket = "Vector"},
                    vector_to_scalar, "Vector") &&
      graph.connect({.node = vector_to_scalar, .socket = "Value"}, outer,
                    "Height") &&
      graph.connect({.node = outer, .socket = "Normal"}, diffuse, "Normal");
  if (!configured) {
    throw std::runtime_error{"failed to configure nested Bump graph"};
  }
  graph.set_root(ShaderDomain::surface,
                 OutputRef{.node = diffuse, .socket = "Closure"});
  return graph;
}

[[nodiscard]] ShaderGraph make_mixed_glass_emission_graph() {
    ShaderGraph graph;
    const auto diffuse = graph.add_node(
        node_type::diffuse_bsdf,
        "Nested mix diffuse");
    const auto glass = graph.add_node(
        node_type::glass_bsdf,
        "Nested mix Beckmann glass");
    const auto mix = graph.add_node(
        node_type::mix_closure,
        "Nested physical mix");
    const auto emission = graph.add_node(
        node_type::emission,
        "Nested explicit emission");
    const auto root = graph.add_node(
        node_type::add_closure,
        "Physical plus emission");
    const auto configured =
        graph.set_input(
            diffuse,
            "Color",
            SocketValue::color({0.22f, 0.51f, 0.76f})) &&
        graph.set_input(
            diffuse,
            "Roughness",
            SocketValue::floating(0.43f)) &&
        graph.set_input(
            diffuse,
            "Normal",
            SocketValue::normal({0.18f, 0.0f, 0.984f})) &&
        graph.set_input(
            glass,
            "Color",
            SocketValue::color({0.81f, 0.91f, 0.98f})) &&
        graph.set_input(
            glass,
            "Roughness",
            SocketValue::floating(0.18f)) &&
        graph.set_input(
            glass,
            "IOR",
            SocketValue::floating(1.37f)) &&
        graph.set_property(
            glass,
            "Distribution",
            SocketValue::string("BECKMANN")) &&
        graph.set_input(
            mix,
            "Factor",
            SocketValue::floating(0.37f)) &&
        graph.set_input(
            emission,
            "Color",
            SocketValue::color({0.17f, 0.41f, 0.89f})) &&
        graph.set_input(
            emission,
            "Strength",
            SocketValue::floating(2.3f)) &&
        graph.connect(
            {.node = diffuse, .socket = "Closure"}, mix, "A") &&
        graph.connect(
            {.node = glass, .socket = "Closure"}, mix, "B") &&
        graph.connect(
            {.node = mix, .socket = "Closure"}, root, "A") &&
        graph.connect(
            {.node = emission, .socket = "Closure"}, root, "B");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure mixed glass/emission graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = root, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_capacity_transparency_graph() {
    ShaderGraph graph;
    std::optional<NodeId> root;
    const auto append = [&](NodeId closure) {
        if (!root) {
            root = closure;
            return;
        }
        const auto add = graph.add_node(
            node_type::add_closure,
            "Capacity sequence");
        if (!graph.connect(
                {.node = *root, .socket = "Closure"}, add, "A") ||
            !graph.connect(
                {.node = closure, .socket = "Closure"}, add, "B")) {
            throw std::runtime_error{
                "failed to append capacity closure"};
        }
        root = add;
    };

    for (auto index = 0u; index < 11u; ++index) {
        const auto diffuse = graph.add_node(
            node_type::diffuse_bsdf,
            "Capacity prefix diffuse");
        const auto color = 0.025f + 0.006f * static_cast<float>(index);
        const auto x = -0.24f + 0.04f * static_cast<float>(index);
        if (!graph.set_input(
                diffuse,
                "Color",
                SocketValue::color({color, 0.7f * color, 0.4f * color})) ||
            !graph.set_input(
                diffuse,
                "Roughness",
                SocketValue::floating(0.03f * static_cast<float>(index))) ||
            !graph.set_input(
                diffuse,
                "Normal",
                SocketValue::normal({x, 0.0f, 0.97f}))) {
            throw std::runtime_error{
                "failed to configure capacity prefix"};
        }
        append(diffuse);
    }

    const auto rejected = graph.add_node(
        node_type::transparent_bsdf,
        "Sub-cutoff transparent");
    const auto allocated = graph.add_node(
        node_type::transparent_bsdf,
        "Allocated transparent");
    if (!graph.set_input(
            rejected,
            "Color",
            SocketValue::color({0.5e-5f, 0.5e-5f, 0.5e-5f})) ||
        !graph.set_input(
            allocated,
            "Color",
            SocketValue::color({0.23f, 0.19f, 0.17f}))) {
        throw std::runtime_error{
            "failed to configure capacity transparency"};
    }
    append(rejected);
    append(allocated);

    for (auto index = 0u; index < 2u; ++index) {
        const auto diffuse = graph.add_node(
            node_type::diffuse_bsdf,
            "Capacity suffix diffuse");
        if (!graph.set_input(
                diffuse,
                "Color",
                SocketValue::color(
                    {0.71f + 0.08f * static_cast<float>(index),
                     0.11f,
                     0.06f})) ||
            !graph.set_input(
                diffuse,
                "Roughness",
                SocketValue::floating(0.77f + 0.1f * index)) ||
            !graph.set_input(
                diffuse,
                "Normal",
                SocketValue::normal({0.0f, 0.6f, 0.8f}))) {
            throw std::runtime_error{
                "failed to configure capacity suffix"};
        }
        append(diffuse);
    }
    if (!root) {
        throw std::runtime_error{"empty capacity graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = *root, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_transformed_emission_graph(
    const std::array<float, 16u> &world_to_object,
    std::string_view label) {
    ShaderGraph graph;
    const auto coordinates = graph.add_node(
        node_type::texture_coordinate,
        std::string{label} + " coordinates");
    Mat4f transform;
    transform.elements = world_to_object;
    const auto conversion = graph.add_node(
        node_type::vector_to_color,
        std::string{label} + " vector to color");
    const auto emission = graph.add_node(
        node_type::emission,
        std::string{label} + " emission");
    const auto configured =
        graph.set_property(
            coordinates,
            "ObjectUseTransform",
            SocketValue::boolean(true)) &&
        graph.set_property(
            coordinates,
            "ObjectWorldToObject",
            SocketValue::transform(transform)) &&
        graph.connect(
            {.node = coordinates, .socket = "Object"},
            conversion,
            "Vector") &&
        graph.connect(
            {.node = conversion, .socket = "Color"},
            emission,
            "Color") &&
        graph.set_input(
            emission,
            "Strength",
            SocketValue::floating(1.0f));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure transformed emission graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = emission, .socket = "Closure"});
    return graph;
}

[[nodiscard]] FixtureProgram compile_fixture(
    ShaderCompiler &compiler,
    ShaderGraph graph) {
    auto shader = compiler.compile(std::move(graph));
    if (!shader.ok()) {
        const auto detail =
            shader.diagnostics.empty()
                ? std::string{}
                : ": " + shader.diagnostics.front().message;
        throw std::runtime_error{
            "compact preparation fixture failed graph compilation" + detail};
    }
    auto surface = compile_surface_program(*shader.program);
    if (!surface.ok()) {
        throw std::runtime_error{
            "compact preparation fixture failed surface lowering"};
    }
    auto binding = bind_surface_parameters(
        *surface.program,
        *shader.program);
    if (!binding.ok()) {
        throw std::runtime_error{
            "compact preparation fixture failed parameter binding"};
    }
    return {
        .program = std::move(surface.program),
        .parameters = std::move(*binding.parameters)};
}

void append_parameters(
    const FixtureProgram &fixture,
    std::vector<float> &scalars,
    std::vector<luisa::float3> &vectors,
    std::vector<PendingShaderTable> &shader_tables) {
    for (auto index = std::size_t{0u};
         index < fixture.program->parameters().size();
         ++index) {
        const auto *value = fixture.parameters.find(
            ParameterId{static_cast<std::uint32_t>(index)});
        if (value == nullptr) {
            throw std::runtime_error{
                "compact preparation fixture lost a parameter"};
        }
        auto scalar = 0.0f;
        auto vector = luisa::make_float3(0.0f);
        switch (value->type) {
            case SocketType::boolean:
            case SocketType::integer:
            case SocketType::floating:
                scalar = scalar_parameter_value(*value);
                break;
            case SocketType::float2:
            case SocketType::float3:
            case SocketType::color:
            case SocketType::spectrum:
            case SocketType::point:
            case SocketType::vector:
            case SocketType::normal:
                vector = vector_parameter_value(*value);
                break;
            case SocketType::unsigned_integer:
                vector = unsigned_parameter_value(*value);
                break;
            case SocketType::string: {
                auto staged = stage_shader_table(
                    *fixture.program,
                    fixture.program->parameters()[index],
                    *value,
                    static_cast<std::uint32_t>(vectors.size()));
                if (!staged.valid) {
                    throw std::runtime_error{
                        "failed to stage compact preparation shader table: " +
                        staged.diagnostic};
                }
                shader_tables.emplace_back(std::move(staged.table));
                break;
            }
            case SocketType::transform:
            case SocketType::closure:
            case SocketType::volume_closure:
                throw std::runtime_error{
                    "unsupported compact preparation parameter type"};
        }
        scalars.emplace_back(scalar);
        vectors.emplace_back(vector);
    }
}

[[nodiscard]] SurfacePreparationQuery invocation_query(
    UInt scenario,
    const SurfacePoint &point) noexcept {
    return {
        .outgoing = point.incoming,
        .glossy_filter_roughness = select(
            0.0f,
            0.12f,
            (scenario & 1u) != 0u),
        .emission_reflective_caustics =
            (scenario & 2u) == 0u,
        .reflective_caustics =
            (scenario & 1u) == 0u,
        .refractive_caustics =
            (scenario & 2u) == 0u,
        .include_runtime_flags =
            (scenario & 4u) == 0u,
        .include_aov = scenario != 3u};
}

[[nodiscard]] SurfaceQuery scattering_query(
    UInt scenario,
    const SurfacePreparationQuery &preparation) noexcept {
    constexpr auto all_lobes = static_cast<std::uint32_t>(
        event_diffuse | event_glossy | event_transmission |
        event_transparent);
    UInt lobe_mask = all_lobes;
    lobe_mask = select(
        lobe_mask,
        static_cast<std::uint32_t>(
            event_diffuse | event_transmission),
        scenario == 1u);
    lobe_mask = select(
        lobe_mask,
        static_cast<std::uint32_t>(event_glossy),
        scenario == 2u);
    lobe_mask = select(
        lobe_mask,
        static_cast<std::uint32_t>(event_transparent),
        scenario == 4u);
    return {
        .lobe_mask = lobe_mask,
        .transport_mode = static_cast<std::uint32_t>(
            TransportMode::radiance),
        .glossy_filter_roughness =
            preparation.glossy_filter_roughness,
        .reflective_caustics =
            preparation.reflective_caustics,
        .refractive_caustics =
            preparation.refractive_caustics};
}

template<typename Populate>
void write_population_results(
    Populate &&populate,
    const ShaderServices &services,
    const SurfacePoint &point,
    const SurfaceClosureIdentityCallable &closure_identity,
    const SurfaceClosureAovCallable &closure_aov,
    UInt scenario,
    UInt invocation,
    const BufferVar<SurfacePreparationCall> &preparation_output,
    const BufferVar<SurfaceClosureTraceCall> &closure_output,
    const BufferVar<SurfaceEvaluationCall> &evaluation_output,
    const BufferVar<SurfaceSampleTraceCall> &sample_output) noexcept {
    const auto preparation_query =
        invocation_query(scenario, point);
    const auto population_query = SurfacePopulationQuery{
        .emission_reflective_caustics =
            preparation_query.emission_reflective_caustics,
        .reflective_caustics =
            preparation_query.reflective_caustics,
        .refractive_caustics =
            preparation_query.refractive_caustics,
        .glossy_filter_roughness =
            preparation_query.glossy_filter_roughness,
        .include_runtime_flags =
            preparation_query.include_runtime_flags,
        .include_aov = preparation_query.include_aov};
    SurfaceClosurePopulationCollector closures{
        point,
        population_closure_capacity,
        population_query,
        closure_identity,
        closure_aov};
    const auto population = populate(
        population_query, closures);
    const auto preparation = closures.preparation(
        population.emission);
    const SurfaceClosureEvaluator evaluator{
        point, closures.closures(), population.shading_normal};
    preparation_output.write(
        invocation,
        pack_surface_preparation(preparation));
    for (auto closure_index = 0u;
         closure_index < population_closure_capacity;
         ++closure_index) {
        closure_output.write(
            invocation * population_closure_capacity +
                closure_index,
            pack_surface_closure_trace(
                evaluator.closure_trace(closure_index)));
    }
    const auto query = scattering_query(
        scenario, preparation_query);
    UInt shader_flags = cycles_abi::shader_use_mis;
    shader_flags |= select(
        0u,
        cycles_abi::shader_exclude_diffuse,
        scenario == 5u);
    shader_flags |= select(
        0u,
        cycles_abi::shader_exclude_glossy,
        scenario == 6u);
    const auto evaluation = evaluator.evaluate_light(
        services,
        point.incoming,
        {.surface = query, .shader_flags = shader_flags});
    evaluation_output.write(
        invocation,
        pack_surface_evaluation(evaluation));
    const auto scenario_float = cast<float>(scenario);
    const auto sample = evaluator.sample_trace(
        services,
        min(0.04f + 0.119f * scenario_float, 0.99999994f),
        make_float2(
            0.09f + 0.1f * scenario_float,
            0.93f - 0.08f * scenario_float),
        query);
    sample_output.write(
        invocation,
        pack_surface_sample_trace(sample));
}

[[nodiscard]] bool finite(luisa::float2 value) noexcept {
    return std::isfinite(value.x) && std::isfinite(value.y);
}

[[nodiscard]] bool finite(luisa::float3 value) noexcept {
    return std::isfinite(value.x) &&
           std::isfinite(value.y) &&
           std::isfinite(value.z);
}

[[nodiscard]] bool equal(
    luisa::float2 actual,
    luisa::float2 expected,
    float tolerance) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance);
}

[[nodiscard]] bool equal(
    luisa::float3 actual,
    luisa::float3 expected,
    float tolerance) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool equal(
    const SurfacePreparationCall &actual,
    const SurfacePreparationCall &expected,
    float tolerance) noexcept {
    return actual.runtime_flags == expected.runtime_flags &&
           finite(actual.emission) &&
           finite(actual.shading_normal) &&
           finite(actual.albedo) &&
           finite(actual.glossy_albedo) &&
           finite(actual.transmission_albedo) &&
           finite(actual.normal) &&
           finite(actual.transparency) &&
           finite(actual.roughness) &&
           equal(actual.emission, expected.emission, tolerance) &&
           equal(
               actual.shading_normal,
               expected.shading_normal,
               tolerance) &&
           equal(actual.albedo, expected.albedo, tolerance) &&
           equal(
               actual.glossy_albedo,
               expected.glossy_albedo,
               tolerance) &&
           equal(
               actual.transmission_albedo,
               expected.transmission_albedo,
               tolerance) &&
           equal(actual.normal, expected.normal, tolerance) &&
           equal(
               actual.transparency,
               expected.transparency,
               tolerance) &&
           equal(actual.roughness, expected.roughness, tolerance);
}

[[nodiscard]] bool equal(
    const SurfaceClosureTraceCall &actual,
    const SurfaceClosureTraceCall &expected,
    float tolerance) noexcept {
    return actual.count == expected.count &&
           actual.runtime_flags == expected.runtime_flags &&
           actual.index == expected.index &&
           actual.type == expected.type &&
           actual.valid == expected.valid &&
           std::isfinite(actual.sample_weight) &&
           finite(actual.weight) && finite(actual.normal) &&
           approximately_equal(
               actual.sample_weight,
               expected.sample_weight,
               tolerance) &&
           equal(actual.weight, expected.weight, tolerance) &&
           equal(actual.normal, expected.normal, tolerance);
}

[[nodiscard]] bool equal(
    const SurfaceEvaluationCall &actual,
    const SurfaceEvaluationCall &expected,
    float tolerance) noexcept {
    return actual.events == expected.events &&
           finite(actual.f) && finite(actual.diffuse_f) &&
           finite(actual.glossy_f) &&
           std::isfinite(actual.pdf) &&
           std::isfinite(actual.diffuse_pdf) &&
           std::isfinite(actual.average_roughness_squared) &&
           equal(actual.f, expected.f, tolerance) &&
           equal(actual.diffuse_f, expected.diffuse_f, tolerance) &&
           equal(actual.glossy_f, expected.glossy_f, tolerance) &&
           approximately_equal(actual.pdf, expected.pdf, tolerance) &&
           approximately_equal(
               actual.diffuse_pdf,
               expected.diffuse_pdf,
               tolerance) &&
           approximately_equal(
               actual.average_roughness_squared,
               expected.average_roughness_squared,
               tolerance);
}

[[nodiscard]] bool equal(
    const SurfaceSampleTraceCall &actual,
    const SurfaceSampleTraceCall &expected,
    float tolerance) noexcept {
    return equal(
               SurfaceEvaluationCall{
                   .f = actual.f,
                   .pdf = actual.pdf,
                   .diffuse_f = actual.diffuse_f,
                   .glossy_f = actual.glossy_f,
                   .diffuse_pdf = actual.diffuse_pdf,
                   .average_roughness_squared =
                       actual.average_roughness_squared,
                   .events = actual.events},
               SurfaceEvaluationCall{
                   .f = expected.f,
                   .pdf = expected.pdf,
                   .diffuse_f = expected.diffuse_f,
                   .glossy_f = expected.glossy_f,
                   .diffuse_pdf = expected.diffuse_pdf,
                   .average_roughness_squared =
                       expected.average_roughness_squared,
                   .events = expected.events},
               tolerance) &&
           actual.runtime_flags == expected.runtime_flags &&
           actual.bssrdf_method == expected.bssrdf_method &&
           actual.valid == expected.valid &&
           actual.closure_index == expected.closure_index &&
           actual.closure_type == expected.closure_type &&
           actual.closure_valid == expected.closure_valid &&
           finite(actual.wi) && finite(actual.roughness) &&
           finite(actual.bssrdf_radius) &&
           finite(actual.bssrdf_albedo) &&
           finite(actual.bssrdf_normal) &&
           finite(actual.closure_weight) &&
           finite(actual.closure_normal) &&
           approximately_equal(actual.eta, expected.eta, tolerance) &&
           equal(actual.wi, expected.wi, tolerance) &&
           equal(actual.roughness, expected.roughness, tolerance) &&
           equal(
               actual.bssrdf_radius,
               expected.bssrdf_radius,
               tolerance) &&
           equal(
               actual.bssrdf_albedo,
               expected.bssrdf_albedo,
               tolerance) &&
           equal(
               actual.bssrdf_normal,
               expected.bssrdf_normal,
               tolerance) &&
           approximately_equal(
               actual.bssrdf_ior,
               expected.bssrdf_ior,
               tolerance) &&
           approximately_equal(
               actual.bssrdf_roughness,
               expected.bssrdf_roughness,
               tolerance) &&
           approximately_equal(
               actual.bssrdf_anisotropy,
               expected.bssrdf_anisotropy,
               tolerance) &&
           approximately_equal(
               actual.closure_sample_weight,
               expected.closure_sample_weight,
               tolerance) &&
           approximately_equal(
               actual.selection_rescaled,
               expected.selection_rescaled,
               tolerance) &&
           equal(
               actual.closure_weight,
               expected.closure_weight,
               tolerance) &&
           equal(
               actual.closure_normal,
               expected.closure_normal,
               tolerance);
}

void print(luisa::float3 value) {
    std::cerr << '{' << value.x << ", " << value.y << ", "
              << value.z << '}';
}

void report_mismatch(
    std::string_view backend,
    std::size_t topology,
    std::size_t scenario,
    const SurfacePreparationCall &actual,
    const SurfacePreparationCall &expected) {
    std::cerr << "compact surface preparation mismatch on "
              << backend << ", topology " << topology
              << ", scenario " << scenario << '\n';
    std::cerr << "  compact emission=";
    print(actual.emission);
    std::cerr << ", expanded emission=";
    print(expected.emission);
    std::cerr << '\n' << "  compact shading normal=";
    print(actual.shading_normal);
    std::cerr << ", expanded shading normal=";
    print(expected.shading_normal);
    std::cerr << '\n' << "  compact albedo=";
    print(actual.albedo);
    std::cerr << ", expanded albedo=";
    print(expected.albedo);
    std::cerr << '\n' << "  compact glossy=";
    print(actual.glossy_albedo);
    std::cerr << ", expanded glossy=";
    print(expected.glossy_albedo);
    std::cerr << '\n' << "  compact transmission=";
    print(actual.transmission_albedo);
    std::cerr << ", expanded transmission=";
    print(expected.transmission_albedo);
    std::cerr << '\n' << "  compact normal=";
    print(actual.normal);
    std::cerr << ", expanded normal=";
    print(expected.normal);
    std::cerr << '\n' << "  compact transparency=";
    print(actual.transparency);
    std::cerr << ", expanded transparency=";
    print(expected.transparency);
    std::cerr << '\n' << "  compact roughness={"
              << actual.roughness.x << ", " << actual.roughness.y
              << "}, expanded roughness={" << expected.roughness.x
              << ", " << expected.roughness.y << "}\n"
              << "  compact flags=" << actual.runtime_flags
              << ", expanded flags=" << expected.runtime_flags
              << '\n';
}

} // namespace

int main(int argc, char **argv) {
    const auto backend = std::string_view{
        argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();

    ShaderCompiler compiler{make_core_node_registry()};
    std::vector<FixtureProgram> fixtures;
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_minimal_principled_graph()));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_typed_clamp_graph()));
    for (const auto &graph : make_typed_map_range_graphs()) {
        fixtures.emplace_back(compile_fixture(compiler, graph));
    }
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_sampled_color_ramp_graph(
            "0,0.1,0.2,0.3,0.4;0.5,0.5,0.6,0.7,0.8;1,0.9,1,0.1,0.2",
            false)));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_sampled_color_ramp_graph(
            "0,0.9,0.1,0.2,0.3;0.4,0.2,0.8,0.3,0.7;1,0.1,0.4,1,0.6",
            true)));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_layered_principled_graph()));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_automatic_normal_graph()));
    const auto weighted_bssrdf_topology =
        static_cast<std::uint32_t>(fixtures.size());
    fixtures.emplace_back(
        compile_fixture(compiler, make_bssrdf_bump_graph(false)));
    const auto zero_bssrdf_topology = static_cast<std::uint32_t>(fixtures.size());
    fixtures.emplace_back(
        compile_fixture(compiler, make_bssrdf_bump_graph(true)));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_mixed_glass_emission_graph()));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_capacity_transparency_graph()));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_transformed_emission_graph(
            {1.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 1.0f, 0.0f, 0.0f,
             0.0f, 0.0f, 1.0f, 0.0f,
             0.15f, 0.25f, 0.35f, 1.0f},
            "transform A")));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_transformed_emission_graph(
            {2.0f, 0.0f, 0.0f, 0.0f,
             0.0f, 0.5f, 0.0f, 0.0f,
             0.0f, 0.0f, 1.5f, 0.0f,
             1.25f, 0.75f, 0.5f, 1.0f},
            "transform B")));
  fixtures.emplace_back(compile_fixture(compiler, make_nested_bump_graph()));

    std::vector<std::shared_ptr<const SurfaceProgram>> programs;
    std::vector<SurfaceClosurePlan> closure_plans;
    std::vector<std::uint32_t> parameter_bases;
    std::vector<float> scalar_parameters;
    std::vector<luisa::float3> vector_parameters;
    std::vector<PendingShaderTable> shader_tables;
    programs.reserve(fixtures.size());
    closure_plans.reserve(fixtures.size());
    parameter_bases.reserve(fixtures.size());

    auto scene = std::make_shared<LuisaSceneData>();
    for (const auto &fixture : fixtures) {
        parameter_bases.emplace_back(
            static_cast<std::uint32_t>(scalar_parameters.size()));
        append_parameters(
            fixture,
            scalar_parameters,
            vector_parameters,
            shader_tables);
        programs.emplace_back(fixture.program);
        closure_plans.emplace_back(analyze_surface_closure_plan(
            *fixture.program,
            fixture.parameters));
        const auto tag = scene->surfaces.create<GraphSurface>(
            fixture.program,
            closure_plans.back());
        if (tag + 1u != programs.size()) {
            throw std::runtime_error{
                "compact preparation fixture tags are not dense"};
        }
        if (cycles_surface_has_bssrdf_bump(*fixture.program, fixture.parameters,
                                           DisplacementMethod::bump)) {
            scene->surface_bssrdf_bump_tags.emplace_back(tag);
        }
    }
    std::string shader_table_diagnostic;
    if (!finalize_shader_tables(
            shader_tables,
            scalar_parameters,
            vector_parameters,
            shader_table_diagnostic)) {
        throw std::runtime_error{
            "failed to finalize compact preparation shader tables: " +
            shader_table_diagnostic};
    }

    std::string diagnostic;
    const std::array invalid_bssrdf_tags{
        static_cast<std::uint32_t>(programs.size())};
    auto invalid_bssrdf_runtime = build_surface_value_runtime(
        device,
        std::span<const std::shared_ptr<const SurfaceProgram>>{programs},
        std::span<const SurfaceClosurePlan>{closure_plans},
        invalid_bssrdf_tags,
        diagnostic);
    if (invalid_bssrdf_runtime ||
        diagnostic.find("BSSRDF-bump topology tag") == std::string::npos) {
        std::cerr << "compact runtime accepted an out-of-range BSSRDF tag on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    scene->surface_values = build_surface_value_runtime(
        device,
        std::span<const std::shared_ptr<const SurfaceProgram>>{programs},
        std::span<const SurfaceClosurePlan>{closure_plans},
        scene->surface_bssrdf_bump_tags,
        diagnostic);
    if (!scene->surface_values) {
        std::cerr << "failed to build compact surface runtime on "
                  << backend << ": " << diagnostic << '\n';
        return EXIT_FAILURE;
    }
    // The two exact Clamp modes now inhabit one typed record domain. The
    // compact-vs-expanded comparison below proves that the shared device
    // handler observes each instruction immediate on every backend.
    if (!has_typed_clamp_record_domain(*scene->surface_values)) {
        std::cerr << "compact runtime did not preserve the typed Clamp "
                     "record domain on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    if (!has_typed_map_range_record_domains(*scene->surface_values)) {
        std::cerr << "compact runtime did not preserve the complete typed Map "
                     "Range record domains on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    if (!has_color_ramp_record_product(*scene->surface_values)) {
        std::cerr << "compact runtime did not compose the Color Ramp SVM "
                     "mode with distinct late-bound table parameters on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto program_evidence =
        psycles::test_support::inspect_compact_surface_program(
            *scene->surface_values);
    if (!program_evidence.domains_match) {
        std::cerr
            << "compact surface program domain lost its exact projected "
               "stream relation on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    if (!program_evidence.normal_transactions_exact) {
        std::cerr << "compact runtime violated the single normal-transaction "
                     "boundary contract on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto bump_variant = program_evidence.bump_variant;
    const auto contains_variant = [](const auto &domain,
                                     std::uint32_t variant) noexcept {
        return std::find(domain.begin(), domain.end(), variant) != domain.end();
    };
    // This controlled graph proves the single-interpreter refinement at
    // instruction granularity. Both immutable Bump configurations remain in
    // the projected stream as typed records, while no recursive Bump opcode,
    // hidden height program, or callable stratum survives.
    if (!program_evidence.bump_stream_exact) {
        std::cerr << "compact runtime did not preserve the exact one-stream "
                     "Bump graph on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto closure_domain_has = [](const auto &domain,
                                       ClosureOperation operation) noexcept {
        return std::any_of(
            domain.begin(), domain.end(), [operation](std::uint32_t variant) {
                return static_cast<ClosureOperation>(
                           variant & surface_closure_opcode_mask) == operation;
            });
    };
    // The BSSRDF callable is induced only by its two capability-tagged
    // topologies. The controlled non-BSSRDF nested Bumps and Glass closure must
    // therefore be absent. Diffuse must remain beside Subsurface: even though
    // it does not contribute to the exit normal, it consumes Cycles closure
    // capacity before later BSSRDF leaves and cannot be erased independently.
    if (scene->surface_bssrdf_bump_tags.size() != 2u ||
        scene->surface_values->bssrdf_value_static_variants.empty() ||
        scene->surface_values->bssrdf_value_static_variants.size() >=
            scene->surface_values->preparation_value_static_variants.size() ||
        scene->surface_values->bssrdf_closure_static_variants.size() >=
            scene->surface_values->closure_static_variants.size() ||
        contains_variant(scene->surface_values->bssrdf_value_static_variants,
                         bump_variant) ||
        !closure_domain_has(
            scene->surface_values->bssrdf_closure_static_variants,
            ClosureOperation::subsurface) ||
        !closure_domain_has(
            scene->surface_values->bssrdf_closure_static_variants,
            ClosureOperation::diffuse) ||
        closure_domain_has(
            scene->surface_values->bssrdf_closure_static_variants,
            ClosureOperation::glass)) {
        std::cerr << "compact runtime did not preserve the exact BSSRDF "
                     "topology domain on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    const auto transform_variant_count = std::count_if(
        scene->surface_values->executable.variants.begin(),
        scene->surface_values->executable.variants.end(),
        [](const auto &variant) noexcept {
            return variant.instruction.operation ==
                   ValueOperation::object_position_with_transform;
        });
    const auto transform_payload_count = std::count_if(
        scene->surface_values->executable.values.metadata.begin(),
        scene->surface_values->executable.values.metadata.end(),
        [](const auto &metadata) noexcept {
            return metadata.static_table_count == 16u;
        });
    // Preparation and emission are independent invocation domains, so each
    // exact program keeps its own metadata record. The executable semantic
    // variant is still interned once across both projections and both authored
    // table payloads.
    if (transform_variant_count != 1u || transform_payload_count != 4u) {
        std::cerr
            << "compact runtime did not share one transform evaluator over "
               "the preparation/emission projections on "
            << backend << " (variants=" << transform_variant_count
            << ", payloads=" << transform_payload_count << ")\n";
        return EXIT_FAILURE;
    }

    const auto cycles_values = make_cycles_bsdf_table_values();
    const auto invocation_count =
        static_cast<std::uint32_t>(fixtures.size()) * scenario_count;
    scene->scalar_parameter_buffer =
        device.create_buffer<float>(scalar_parameters.size());
    scene->vector_parameter_buffer =
        device.create_buffer<luisa::float3>(vector_parameters.size());
    scene->cycles_bsdf_table_buffer =
        device.create_buffer<float>(cycles_values.size());
    scene->texture_heap = device.create_bindless_array(1u);
    scene->heap = device.create_bindless_array(2u);
    scene->attribute_binding_slot = 0u;
    scene->attribute_range_slot = 1u;
    scene->volume_metadata.closure_allocation_budget =
        population_closure_capacity;
    auto parameter_base_buffer =
        device.create_buffer<std::uint32_t>(parameter_bases.size());
    auto expanded_buffer =
        device.create_buffer<SurfacePreparationCall>(invocation_count);
    auto compact_buffer =
        device.create_buffer<SurfacePreparationCall>(invocation_count);
    auto expanded_emission_buffer =
        device.create_buffer<luisa::float3>(invocation_count);
    auto compact_emission_buffer =
        device.create_buffer<luisa::float3>(invocation_count);

    const auto closure_setup =
        make_surface_closure_setup_callables();
    const auto closure_identity =
        make_surface_closure_identity_callable();
    const auto closure_aov =
        make_surface_closure_aov_callable();
    const auto texture_sampling =
        make_texture_2d_sampling_callables();
    const auto attribute_lookup =
        make_surface_attribute_lookup_callable(0u, 1u);
    if (const auto diagnostic =
            validate_compact_surface_value_program_abi(scene);
        !diagnostic.empty()) {
        std::cerr << diagnostic << " on " << backend << '\n';
        return EXIT_FAILURE;
    }
    Kernel1D collector_begin_lifecycle =
        [closure_identity,
         closure_aov](BufferFloat3 output) noexcept {
            const auto point =
                make_surface_value_transaction_test_point();
            const auto final_normal = normalize(
                make_float3(0.43f, -0.27f, 0.86f));
            const auto query = SurfacePopulationQuery{
                .emission_reflective_caustics = true,
                .reflective_caustics = true,
                .refractive_caustics = true,
                .glossy_filter_roughness = 0.0f,
                .include_runtime_flags = true,
                .include_aov = true};
            SurfaceClosurePopulationCollector collector{
                point,
                population_closure_capacity,
                query,
                closure_identity,
                closure_aov};
            collector.begin(final_normal);
            collector.finish();
            output.write(
                0u,
                collector.preparation(make_float3(0.0f))
                    .aov.normal);
        };
    Kernel1D expanded =
        [scene,
         closure_setup,
         texture_sampling,
         attribute_lookup](
            BufferFloat scalar_buffer,
            BufferFloat3 vector_buffer,
            BufferFloat cycles_buffer,
            BindlessVar textures,
            BindlessVar geometry_heap,
            BufferUInt parameter_base_buffer,
            BufferVar<SurfacePreparationCall> output) noexcept {
            const auto invocation = dispatch_x();
            const auto topology = invocation / scenario_count;
            const auto scenario = invocation % scenario_count;
            auto point = make_surface_value_transaction_test_point();
            point.parameter_block =
                parameter_base_buffer.read(topology);
            point.back_facing = (scenario & 4u) != 0u;
            point.incoming = normalize(make_float3(
                0.21f,
                -0.13f,
                select(0.969f, -0.969f, scenario == 7u)));
            CallableSurfaceClosureSetupProvider setup_provider{
                cycles_buffer,
                closure_setup};
            CallableTexture2DSamplingProvider texture_provider{
                textures,
                texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap,
                attribute_lookup};
            BufferShaderServices services{
                scalar_buffer,
                vector_buffer,
                cycles_buffer,
                textures,
                geometry_heap,
                0u,
                1u,
                scene->nishita_texture_bindings,
                scene->shader_color_space,
                &setup_provider,
                &texture_provider,
                &attribute_provider};
            output.write(
                invocation,
                pack_surface_preparation(
                    scene->surfaces.prepare(
                        topology,
                        services,
                        point,
                        invocation_query(scenario, point))));
        };

    const auto compact_preparation =
        make_compact_surface_preparation_callable(scene);
    Kernel1D compact =
        [compact_preparation](
            BufferFloat scalar_buffer,
            BufferFloat3 vector_buffer,
            BufferFloat cycles_buffer,
            BindlessVar textures,
            BindlessVar geometry_heap,
            BufferUInt parameter_base_buffer,
            BufferVar<SurfacePreparationCall> output) noexcept {
            const auto invocation = dispatch_x();
            const auto topology = invocation / scenario_count;
            const auto scenario = invocation % scenario_count;
            auto point = make_surface_value_transaction_test_point();
            point.parameter_block =
                parameter_base_buffer.read(topology);
            point.back_facing = (scenario & 4u) != 0u;
            point.incoming = normalize(make_float3(
                0.21f,
                -0.13f,
                select(0.969f, -0.969f, scenario == 7u)));
            output.write(
                invocation,
                compact_preparation(
                    scalar_buffer,
                    vector_buffer,
                    cycles_buffer,
                    textures,
                    geometry_heap,
                    topology,
                    pack_surface_point(point),
                    pack_surface_preparation_query(
                        invocation_query(scenario, point))));
        };

  Kernel1D expanded_emission =
      [scene, texture_sampling, attribute_lookup](
          BufferFloat scalar_buffer, BufferFloat3 vector_buffer,
          BufferFloat cycles_buffer, BindlessVar textures,
          BindlessVar geometry_heap, BufferUInt parameter_base_buffer,
          BufferFloat3 output) noexcept {
        const auto invocation = dispatch_x();
        const auto topology = invocation / scenario_count;
        const auto scenario = invocation % scenario_count;
        auto point = make_surface_value_transaction_test_point();
        point.parameter_block = parameter_base_buffer.read(topology);
        point.back_facing = (scenario & 4u) != 0u;
        point.incoming = normalize(make_float3(
            0.21f, -0.13f, select(0.969f, -0.969f, scenario == 7u)));
        CallableTexture2DSamplingProvider texture_provider{textures,
                                                           texture_sampling};
        CallableSurfaceAttributeLookupProvider attribute_provider{
            geometry_heap, attribute_lookup};
        BufferShaderServices services{scalar_buffer,
                                      vector_buffer,
                                      cycles_buffer,
                                      textures,
                                      geometry_heap,
                                      0u,
                                      1u,
                                      scene->nishita_texture_bindings,
                                      scene->shader_color_space,
                                      nullptr,
                                      &texture_provider,
                                      &attribute_provider};
        const auto query = invocation_query(scenario, point);
        output.write(invocation, scene->surfaces.emission(
                                     topology, services, point, point.incoming,
                                     query.emission_reflective_caustics));
      };

  const auto compact_emission_callable =
      make_compact_surface_emission_callable(scene);
  Kernel1D compact_emission =
      [compact_emission_callable](
          BufferFloat scalar_buffer, BufferFloat3 vector_buffer,
          BufferFloat cycles_buffer, BindlessVar textures,
          BindlessVar geometry_heap, BufferUInt parameter_base_buffer,
          BufferFloat3 output) noexcept {
        const auto invocation = dispatch_x();
        const auto topology = invocation / scenario_count;
        const auto scenario = invocation % scenario_count;
        auto point = make_surface_value_transaction_test_point();
        point.parameter_block = parameter_base_buffer.read(topology);
        point.back_facing = (scenario & 4u) != 0u;
        point.incoming = normalize(make_float3(
            0.21f, -0.13f, select(0.969f, -0.969f, scenario == 7u)));
        const auto query = invocation_query(scenario, point);
        output.write(invocation,
                     compact_emission_callable(
                         scalar_buffer, vector_buffer, cycles_buffer, textures,
                         geometry_heap, topology, pack_surface_point(point),
                         point.incoming, query.emission_reflective_caustics));
      };

    Kernel1D expanded_bssrdf_normal =
        [scene, weighted_bssrdf_topology, zero_bssrdf_topology, closure_setup,
         texture_sampling, attribute_lookup](
            BufferFloat scalar_buffer, BufferFloat3 vector_buffer,
            BufferFloat cycles_buffer, BindlessVar textures,
            BindlessVar geometry_heap, BufferUInt parameter_base_buffer,
            BufferFloat3 output) noexcept {
            const auto invocation = dispatch_x();
            const auto topology = invocation / scenario_count;
            const auto scenario = invocation % scenario_count;
            auto point = make_surface_value_transaction_test_point();
            point.parameter_block = parameter_base_buffer.read(topology);
            point.back_facing = (scenario & 4u) != 0u;
            point.incoming = normalize(make_float3(
                0.21f, -0.13f, select(0.969f, -0.969f, scenario == 7u)));
            CallableSurfaceClosureSetupProvider setup_provider{cycles_buffer,
                                                               closure_setup};
            CallableTexture2DSamplingProvider texture_provider{textures,
                                                               texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{scalar_buffer,
                                          vector_buffer,
                                          cycles_buffer,
                                          textures,
                                          geometry_heap,
                                          0u,
                                          1u,
                                          scene->nishita_texture_bindings,
                                          scene->shader_color_space,
                                          &setup_provider,
                                          &texture_provider,
                                          &attribute_provider};
            const auto query = invocation_query(scenario, point);
            const auto has_bssrdf_bump = ((topology == weighted_bssrdf_topology) |
                                          (topology == zero_bssrdf_topology)) &
                                         (scenario != 7u);
            SurfaceBssrdfNormalVisitor visitor{population_closure_capacity};
            $if (has_bssrdf_bump) {
                static_cast<void>(scene->surfaces.collect_bssrdf_bump_closures(
                    topology, scene->surface_bssrdf_bump_tags, services, point,
                    query.reflective_caustics, query.refractive_caustics, visitor));
            }
            $else {
                visitor.begin(point.shading_normal);
                visitor.finish();
            };
            output.write(invocation, visitor.result());
        };

    const auto compact_bssrdf_normal_callable =
        make_compact_surface_bssrdf_normal_callable(scene);
    Kernel1D compact_bssrdf_normal =
        [compact_bssrdf_normal_callable, weighted_bssrdf_topology,
         zero_bssrdf_topology](
            BufferFloat scalar_buffer, BufferFloat3 vector_buffer,
            BufferFloat cycles_buffer, BindlessVar textures,
            BindlessVar geometry_heap, BufferUInt parameter_base_buffer,
            BufferFloat3 output) noexcept {
            const auto invocation = dispatch_x();
            const auto topology = invocation / scenario_count;
            const auto scenario = invocation % scenario_count;
            auto point = make_surface_value_transaction_test_point();
            point.parameter_block = parameter_base_buffer.read(topology);
            point.back_facing = (scenario & 4u) != 0u;
            point.incoming = normalize(make_float3(
                0.21f, -0.13f, select(0.969f, -0.969f, scenario == 7u)));
            const auto query = invocation_query(scenario, point);
            const auto has_bssrdf_bump = ((topology == weighted_bssrdf_topology) |
                                          (topology == zero_bssrdf_topology)) &
                                         (scenario != 7u);
            output.write(invocation,
                         compact_bssrdf_normal_callable(
                             scalar_buffer, vector_buffer, cycles_buffer, textures,
                             geometry_heap, topology, pack_surface_point(point),
                             has_bssrdf_bump, query.reflective_caustics,
                             query.refractive_caustics));
        };

    const auto compact_population_program =
        make_compact_surface_population_program(scene);
    Kernel1D expanded_population =
        [scene,
         closure_setup,
         closure_identity,
         closure_aov,
         texture_sampling,
         attribute_lookup](
            BufferFloat scalar_buffer,
            BufferFloat3 vector_buffer,
            BufferFloat cycles_buffer,
            BindlessVar textures,
            BindlessVar geometry_heap,
            BufferUInt parameter_base_buffer,
            BufferVar<SurfacePreparationCall> preparation_output,
            BufferVar<SurfaceClosureTraceCall> closure_output,
            BufferVar<SurfaceEvaluationCall> evaluation_output,
            BufferVar<SurfaceSampleTraceCall> sample_output) noexcept {
            const auto invocation = dispatch_x();
            const auto topology = invocation / scenario_count;
            const auto scenario = invocation % scenario_count;
            auto point = make_surface_value_transaction_test_point();
            point.parameter_block =
                parameter_base_buffer.read(topology);
            point.back_facing = (scenario & 4u) != 0u;
            point.incoming = normalize(make_float3(
                0.21f,
                -0.13f,
                select(0.969f, -0.969f, scenario == 7u)));
            CallableSurfaceClosureSetupProvider setup_provider{
                cycles_buffer, closure_setup};
            CallableTexture2DSamplingProvider texture_provider{
                textures, texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{
                scalar_buffer,
                vector_buffer,
                cycles_buffer,
                textures,
                geometry_heap,
                0u,
                1u,
                scene->nishita_texture_bindings,
                scene->shader_color_space,
                &setup_provider,
                &texture_provider,
                &attribute_provider};
            write_population_results(
                [&](const SurfacePopulationQuery &query,
                    SurfaceClosureCollector &collector) noexcept {
                    return scene->surfaces.populate(
                        topology,
                        services,
                        point,
                        query,
                        collector);
                },
                services,
                point,
                closure_identity,
                closure_aov,
                scenario,
                invocation,
                preparation_output,
                closure_output,
                evaluation_output,
                sample_output);
        };
    Kernel1D compact_population =
        [scene,
         compact_population_program,
         closure_setup,
         closure_identity,
         closure_aov,
         texture_sampling,
         attribute_lookup](
            BufferFloat scalar_buffer,
            BufferFloat3 vector_buffer,
            BufferFloat cycles_buffer,
            BindlessVar textures,
            BindlessVar geometry_heap,
            BufferUInt parameter_base_buffer,
            BufferVar<SurfacePreparationCall> preparation_output,
            BufferVar<SurfaceClosureTraceCall> closure_output,
            BufferVar<SurfaceEvaluationCall> evaluation_output,
            BufferVar<SurfaceSampleTraceCall> sample_output) noexcept {
            const auto invocation = dispatch_x();
            const auto topology = invocation / scenario_count;
            const auto scenario = invocation % scenario_count;
            auto point = make_surface_value_transaction_test_point();
            point.parameter_block =
                parameter_base_buffer.read(topology);
            point.back_facing = (scenario & 4u) != 0u;
            point.incoming = normalize(make_float3(
                0.21f,
                -0.13f,
                select(0.969f, -0.969f, scenario == 7u)));
            CallableSurfaceClosureSetupProvider setup_provider{
                cycles_buffer, closure_setup};
            CallableTexture2DSamplingProvider texture_provider{
                textures, texture_sampling};
            CallableSurfaceAttributeLookupProvider attribute_provider{
                geometry_heap, attribute_lookup};
            BufferShaderServices services{
                scalar_buffer,
                vector_buffer,
                cycles_buffer,
                textures,
                geometry_heap,
                0u,
                1u,
                scene->nishita_texture_bindings,
                scene->shader_color_space,
                &setup_provider,
                &texture_provider,
                &attribute_provider};
            write_population_results(
                [&](const SurfacePopulationQuery &query,
                    SurfaceClosureCollector &collector) noexcept {
                    return compact_population_program->populate(
                        topology,
                        services,
                        point,
                        query,
                        collector);
                },
                services,
                point,
                closure_identity,
                closure_aov,
                scenario,
                invocation,
                preparation_output,
                closure_output,
                evaluation_output,
                sample_output);
        };

    const auto &scalar_buffer = scene->scalar_parameter_buffer;
    const auto &vector_buffer = scene->vector_parameter_buffer;
    const auto &cycles_buffer = scene->cycles_bsdf_table_buffer;
    const auto &textures = scene->texture_heap;
    const auto &geometry_heap = scene->heap;
    auto expanded_shader =
        psycles::test_support::compile_named_kernel(
            device,
            "surface_prepare_expanded_reference",
            expanded);
    auto compact_shader =
        psycles::test_support::compile_named_kernel(
            device,
            "surface_prepare_compact_bytecode",
            compact);
    auto expanded_emission_shader =
        psycles::test_support::compile_named_kernel(
            device,
            "surface_emission_expanded_reference",
            expanded_emission);
    auto compact_emission_shader =
        psycles::test_support::compile_named_kernel(
            device,
            "surface_emission_compact_bytecode",
            compact_emission);
    auto expanded_bssrdf_normal_shader =
        psycles::test_support::compile_named_kernel(
            device, "surface_bssrdf_normal_expanded_reference",
            expanded_bssrdf_normal);
    auto compact_bssrdf_normal_shader =
        psycles::test_support::compile_named_kernel(
            device, "surface_bssrdf_normal_compact_bytecode",
            compact_bssrdf_normal);
    auto collector_begin_shader =
        psycles::test_support::compile_named_kernel(
            device,
            "surface_population_collector_begin_lifecycle",
            collector_begin_lifecycle);
    auto expanded_population_shader =
        psycles::test_support::compile_named_kernel(
            device,
            "surface_population_expanded_reference",
            expanded_population);
    auto compact_population_shader =
        psycles::test_support::compile_named_kernel(
            device,
            "surface_population_compact_bytecode",
            compact_population);
    const auto closure_record_count =
        invocation_count * population_closure_capacity;
    auto expanded_population_preparation_buffer =
        device.create_buffer<SurfacePreparationCall>(invocation_count);
    auto compact_population_preparation_buffer =
        device.create_buffer<SurfacePreparationCall>(invocation_count);
    auto expanded_population_closure_buffer =
        device.create_buffer<SurfaceClosureTraceCall>(closure_record_count);
    auto compact_population_closure_buffer =
        device.create_buffer<SurfaceClosureTraceCall>(closure_record_count);
    auto expanded_population_evaluation_buffer =
        device.create_buffer<SurfaceEvaluationCall>(invocation_count);
    auto compact_population_evaluation_buffer =
        device.create_buffer<SurfaceEvaluationCall>(invocation_count);
    auto expanded_population_sample_buffer =
        device.create_buffer<SurfaceSampleTraceCall>(invocation_count);
    auto compact_population_sample_buffer =
        device.create_buffer<SurfaceSampleTraceCall>(invocation_count);
    auto expanded_bssrdf_normal_buffer =
        device.create_buffer<luisa::float3>(invocation_count);
    auto compact_bssrdf_normal_buffer =
        device.create_buffer<luisa::float3>(invocation_count);
    auto collector_begin_buffer =
        device.create_buffer<luisa::float3>(1u);
    std::vector<SurfacePreparationCall> expected(invocation_count);
    std::vector<SurfacePreparationCall> actual(invocation_count);
    std::vector<luisa::float3> expected_emission(invocation_count);
    std::vector<luisa::float3> actual_emission(invocation_count);
    std::vector<SurfacePreparationCall>
        population_expected_preparation(invocation_count);
    std::vector<SurfacePreparationCall>
        population_actual_preparation(invocation_count);
    std::vector<SurfaceClosureTraceCall>
        population_expected_closures(closure_record_count);
    std::vector<SurfaceClosureTraceCall>
        population_actual_closures(closure_record_count);
    std::vector<SurfaceEvaluationCall>
        population_expected_evaluations(invocation_count);
    std::vector<SurfaceEvaluationCall>
        population_actual_evaluations(invocation_count);
    std::vector<SurfaceSampleTraceCall>
        population_expected_samples(invocation_count);
    std::vector<SurfaceSampleTraceCall>
        population_actual_samples(invocation_count);
    std::vector<luisa::float3> expected_bssrdf_normals(invocation_count);
    std::vector<luisa::float3> actual_bssrdf_normals(invocation_count);
    auto collector_begin_normal = luisa::make_float3(0.0f);
    stream << scalar_buffer.copy_from(luisa::span{scalar_parameters})
           << vector_buffer.copy_from(luisa::span{vector_parameters})
           << cycles_buffer.copy_from(luisa::span{cycles_values})
           << parameter_base_buffer.copy_from(luisa::span{parameter_bases});
    upload_surface_value_runtime(stream, *scene->surface_values);
    stream << collector_begin_shader(collector_begin_buffer).dispatch(1u)
           << collector_begin_buffer.copy_to(
                  luisa::span{&collector_begin_normal, 1u})
           << expanded_shader(
                  scalar_buffer,
                  vector_buffer,
                  cycles_buffer,
                  textures,
                  geometry_heap,
                  parameter_base_buffer,
                  expanded_buffer)
                  .dispatch(invocation_count)
           << expanded_buffer.copy_to(luisa::span{expected})
           << compact_shader(
                  scalar_buffer,
                  vector_buffer,
                  cycles_buffer,
                  textures,
                  geometry_heap,
                  parameter_base_buffer,
                  compact_buffer)
                  .dispatch(invocation_count)
           << compact_buffer.copy_to(luisa::span{actual})
         << expanded_emission_shader(
                scalar_buffer, vector_buffer, cycles_buffer, textures,
                geometry_heap, parameter_base_buffer, expanded_emission_buffer)
                .dispatch(invocation_count)
         << expanded_emission_buffer.copy_to(luisa::span{expected_emission})
         << compact_emission_shader(
                scalar_buffer, vector_buffer, cycles_buffer, textures,
                geometry_heap, parameter_base_buffer, compact_emission_buffer)
                .dispatch(invocation_count)
         << compact_emission_buffer.copy_to(luisa::span{actual_emission})
           << expanded_bssrdf_normal_shader(scalar_buffer, vector_buffer,
                                            cycles_buffer, textures,
                                            geometry_heap, parameter_base_buffer,
                                            expanded_bssrdf_normal_buffer)
                  .dispatch(invocation_count)
           << expanded_bssrdf_normal_buffer.copy_to(
                  luisa::span{expected_bssrdf_normals})
           << compact_bssrdf_normal_shader(scalar_buffer, vector_buffer,
                                           cycles_buffer, textures, geometry_heap,
                                           parameter_base_buffer,
                                           compact_bssrdf_normal_buffer)
                  .dispatch(invocation_count)
           << compact_bssrdf_normal_buffer.copy_to(
                  luisa::span{actual_bssrdf_normals})
           << expanded_population_shader(
                  scalar_buffer,
                  vector_buffer,
                  cycles_buffer,
                  textures,
                  geometry_heap,
                  parameter_base_buffer,
                  expanded_population_preparation_buffer,
                  expanded_population_closure_buffer,
                  expanded_population_evaluation_buffer,
                  expanded_population_sample_buffer)
                  .dispatch(invocation_count)
           << expanded_population_preparation_buffer.copy_to(
                  luisa::span{population_expected_preparation})
           << expanded_population_closure_buffer.copy_to(
                  luisa::span{population_expected_closures})
           << expanded_population_evaluation_buffer.copy_to(
                  luisa::span{population_expected_evaluations})
           << expanded_population_sample_buffer.copy_to(
                  luisa::span{population_expected_samples})
           << compact_population_shader(
                  scalar_buffer,
                  vector_buffer,
                  cycles_buffer,
                  textures,
                  geometry_heap,
                  parameter_base_buffer,
                  compact_population_preparation_buffer,
                  compact_population_closure_buffer,
                  compact_population_evaluation_buffer,
                  compact_population_sample_buffer)
                  .dispatch(invocation_count)
           << compact_population_preparation_buffer.copy_to(
                  luisa::span{population_actual_preparation})
           << compact_population_closure_buffer.copy_to(
                  luisa::span{population_actual_closures})
           << compact_population_evaluation_buffer.copy_to(
                  luisa::span{population_actual_evaluations})
           << compact_population_sample_buffer.copy_to(
                  luisa::span{population_actual_samples})
           << synchronize();

    constexpr auto tolerance = 3.0e-5f;
    const auto expected_collector_normal = [] {
        const auto inverse_length = 1.0f / std::sqrt(
            0.43f * 0.43f + 0.27f * 0.27f + 0.86f * 0.86f);
        return luisa::make_float3(
            0.43f * inverse_length,
            -0.27f * inverse_length,
            0.86f * inverse_length);
    }();
    if (!equal(
            collector_begin_normal,
            expected_collector_normal,
            tolerance)) {
        std::cerr
            << "surface population collector ignored final shading normal on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    for (auto invocation = std::size_t{0u};
         invocation < actual.size();
         ++invocation) {
        if (!equal(actual[invocation], expected[invocation], tolerance)) {
            report_mismatch(
                backend,
                invocation / scenario_count,
                invocation % scenario_count,
                actual[invocation],
                expected[invocation]);
            return EXIT_FAILURE;
        }
    if (!finite(actual_emission[invocation]) ||
        !equal(actual_emission[invocation], expected_emission[invocation],
               tolerance)) {
      std::cerr << "compact emission projection mismatch on " << backend
                << ", topology " << invocation / scenario_count << ", scenario "
                << invocation % scenario_count << '\n';
      return EXIT_FAILURE;
    }
        if (!finite(actual_bssrdf_normals[invocation]) ||
            !equal(actual_bssrdf_normals[invocation],
                   expected_bssrdf_normals[invocation], tolerance)) {
            std::cerr << "compact BSSRDF exit normal mismatch on " << backend
                      << ", topology " << invocation / scenario_count << ", scenario "
                      << invocation % scenario_count << '\n';
            return EXIT_FAILURE;
        }
    }
    const auto first_invocation = [](std::uint32_t topology) noexcept {
        return static_cast<std::size_t>(topology) * scenario_count;
    };
    if (equal(actual_bssrdf_normals[first_invocation(weighted_bssrdf_topology)],
              actual_bssrdf_normals[first_invocation(zero_bssrdf_topology)],
              tolerance)) {
        std::cerr << "BSSRDF compact regression did not distinguish the weighted "
                     "normal from the exact-zero ShaderData::N fallback on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    for (auto invocation = std::size_t{0u};
         invocation < invocation_count;
         ++invocation) {
        const auto topology = invocation / scenario_count;
        const auto scenario = invocation % scenario_count;
        if (!equal(
                population_actual_preparation[invocation],
                population_expected_preparation[invocation],
                tolerance)) {
            std::cerr
                << "compact population preparation mismatch on "
                << backend << ", topology " << topology
                << ", scenario " << scenario << '\n';
            return EXIT_FAILURE;
        }
        if (!equal(
                population_actual_evaluations[invocation],
                population_expected_evaluations[invocation],
                tolerance)) {
            std::cerr
                << "compact population light evaluation mismatch on "
                << backend << ", topology " << topology
                << ", scenario " << scenario << '\n';
            return EXIT_FAILURE;
        }
        if (!equal(
                population_actual_samples[invocation],
                population_expected_samples[invocation],
                tolerance)) {
            print_compact_surface_sample_mismatch(
                population_actual_samples[invocation],
                population_expected_samples[invocation], backend,
                topology, scenario);
            return EXIT_FAILURE;
        }
        for (auto closure_index = std::size_t{0u};
             closure_index < population_closure_capacity;
             ++closure_index) {
            const auto record =
                invocation * population_closure_capacity +
                closure_index;
            if (!equal(
                    population_actual_closures[record],
                    population_expected_closures[record],
                    tolerance)) {
                std::cerr
                    << "compact population closure mismatch on "
                    << backend << ", topology " << topology
                    << ", scenario " << scenario
                    << ", closure " << closure_index << '\n';
                return EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

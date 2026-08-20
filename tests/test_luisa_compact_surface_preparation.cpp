#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"
#include "path_tracer_attribute_lookup.h"
#include "path_tracer_bsdf_tables.h"
#include "path_tracer_shader_services.h"
#include "path_tracer_surface_closure_setup.h"
#include "path_tracer_surface_values.h"
#include "path_tracer_surfaces.h"
#include "path_tracer_texture_sampling.h"

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
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;

constexpr auto scenario_count = 8u;

struct FixtureProgram {
    std::shared_ptr<const SurfaceProgram> program;
    SurfaceParameterBlock parameters;
};

[[nodiscard]] ShaderGraph make_minimal_principled_graph() {
    ShaderGraph graph;
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Minimal Principled");
    const auto configured =
        graph.set_input(
            principled,
            "BaseColor",
            SocketValue::color({0.18f, 0.42f, 0.73f})) &&
        graph.set_input(
            principled,
            "Roughness",
            SocketValue::floating(0.31f)) &&
        graph.set_input(
            principled,
            "Metallic",
            SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled,
            "TransmissionWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled,
            "SubsurfaceWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled,
            "SheenWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled,
            "CoatWeight",
            SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled,
            "Alpha",
            SocketValue::floating(1.0f)) &&
        graph.set_input(
            principled,
            "EmissionStrength",
            SocketValue::floating(0.0f));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure minimal Principled graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

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

[[nodiscard]] FixtureProgram compile_fixture(
    ShaderCompiler &compiler,
    ShaderGraph graph) {
    auto shader = compiler.compile(std::move(graph));
    if (!shader.ok()) {
        throw std::runtime_error{
            "compact preparation fixture failed graph compilation"};
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
    std::vector<luisa::float3> &vectors) {
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
            case SocketType::transform:
            case SocketType::string:
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
           finite(actual.albedo) &&
           finite(actual.glossy_albedo) &&
           finite(actual.transmission_albedo) &&
           finite(actual.normal) &&
           finite(actual.transparency) &&
           finite(actual.roughness) &&
           equal(actual.emission, expected.emission, tolerance) &&
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
        make_layered_principled_graph()));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_mixed_glass_emission_graph()));
    fixtures.emplace_back(compile_fixture(
        compiler,
        make_capacity_transparency_graph()));

    std::vector<std::shared_ptr<const SurfaceProgram>> programs;
    std::vector<SurfaceClosurePlan> closure_plans;
    std::vector<std::uint32_t> parameter_bases;
    std::vector<float> scalar_parameters;
    std::vector<luisa::float3> vector_parameters;
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
            vector_parameters);
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
    }

    std::string diagnostic;
    scene->surface_values = build_surface_value_runtime(
        device,
        std::span<const std::shared_ptr<const SurfaceProgram>>{programs},
        std::span<const SurfaceClosurePlan>{closure_plans},
        diagnostic);
    if (!scene->surface_values) {
        std::cerr << "failed to build compact surface runtime on "
                  << backend << ": " << diagnostic << '\n';
        return EXIT_FAILURE;
    }

    const auto closure_setup =
        make_surface_closure_setup_callables();
    const auto texture_sampling =
        make_texture_2d_sampling_callables();
    const auto attribute_lookup =
        make_surface_attribute_lookup_callable(0u, 1u);
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
            auto point = make_surface_point();
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
            auto point = make_surface_point();
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

    const auto cycles_values = make_cycles_bsdf_table_values();
    const auto invocation_count =
        static_cast<std::uint32_t>(fixtures.size()) * scenario_count;
    auto scalar_buffer =
        device.create_buffer<float>(scalar_parameters.size());
    auto vector_buffer =
        device.create_buffer<luisa::float3>(vector_parameters.size());
    auto cycles_buffer =
        device.create_buffer<float>(cycles_values.size());
    auto parameter_base_buffer =
        device.create_buffer<std::uint32_t>(parameter_bases.size());
    auto expanded_buffer =
        device.create_buffer<SurfacePreparationCall>(invocation_count);
    auto compact_buffer =
        device.create_buffer<SurfacePreparationCall>(invocation_count);
    auto textures = device.create_bindless_array(1u);
    auto geometry_heap = device.create_bindless_array(2u);
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
    std::vector<SurfacePreparationCall> expected(invocation_count);
    std::vector<SurfacePreparationCall> actual(invocation_count);
    stream << scalar_buffer.copy_from(luisa::span{scalar_parameters})
           << vector_buffer.copy_from(luisa::span{vector_parameters})
           << cycles_buffer.copy_from(luisa::span{cycles_values})
           << parameter_base_buffer.copy_from(luisa::span{parameter_bases});
    upload_surface_value_runtime(stream, *scene->surface_values);
    stream << expanded_shader(
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
           << synchronize();

    constexpr auto tolerance = 3.0e-5f;
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
    }
    return EXIT_SUCCESS;
}

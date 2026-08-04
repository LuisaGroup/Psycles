#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>

#include "luisa_surface_test_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
namespace cycles_abi = psycles::contract::cycles_abi;
using psycles::test_support::ParameterShaderServices;
using psycles::test_support::approximately_equal;
using psycles::test_support::make_surface_point;
using psycles::test_support::parameter_data;

static_assert(cycles_closure::runtime_backfacing == 1u);
static_assert(cycles_closure::runtime_cache_miss == 2u);
static_assert(cycles_closure::runtime_emission == 4u);
static_assert(cycles_closure::runtime_bsdf == 8u);
static_assert(cycles_closure::runtime_bsdf_has_eval == 16u);
static_assert(cycles_closure::runtime_bssrdf == 32u);
static_assert(cycles_closure::runtime_holdout == 64u);
static_assert(cycles_closure::runtime_extinction == 128u);
static_assert(cycles_closure::runtime_scatter == 256u);
static_assert(cycles_closure::runtime_is_volume_shader_eval == 512u);
static_assert(cycles_closure::runtime_transparent == 1024u);
static_assert(cycles_closure::runtime_bsdf_has_transmission == 2048u);
static_assert(cycles_closure::runtime_ray_portal == 4096u);

class OracleShaderServices final : public ShaderServices {

private:
    std::uint32_t _color_parameter{};
    luisa::float3 _color{};

public:
    explicit OracleShaderServices(
        std::uint32_t color_parameter, luisa::float3 color) noexcept
        : _color_parameter{color_parameter}, _color{color} {
    }

    [[nodiscard]] Float4 texture_2d(Expr<std::uint32_t>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t,
        std::uint32_t) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] ShaderAttribute attribute(Expr<std::uint64_t>,
        const SurfacePoint &) const noexcept override {
        return ShaderAttribute::missing();
    }

    [[nodiscard]] Float parameter_float(Expr<std::uint32_t>,
        Expr<std::uint32_t>) const noexcept override {
        return 0.0f;
    }

    [[nodiscard]] Float3 parameter_float3(Expr<std::uint32_t>,
        Expr<std::uint32_t> slot) const noexcept override {
        return select(make_float3(0.0f),
            make_float3(_color),
            slot == _color_parameter);
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>) const noexcept override {
        return 1.0f;
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> value) const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 nishita_sky(Expr<std::uint32_t>,
        std::uint32_t,
        Expr<luisa::float3>,
        Expr<float>,
        Expr<float>,
        Expr<float>,
        Expr<float>) const noexcept override {
        return make_float3(0.0f);
    }
};

[[nodiscard]] ShaderGraph make_diffuse_graph() {
    ShaderGraph graph;
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    if (!graph.set_input(diffuse,
            "Color",
            SocketValue::color({0.62f, 0.41f, 0.23f})) ||
        !graph.set_input(
            diffuse, "Roughness", SocketValue::floating(0.0f))) {
        throw std::runtime_error{
            "failed to configure diffuse oracle graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = diffuse, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_transparent_principled_graph() {
    ShaderGraph graph;
    const auto transparent =
        graph.add_node(node_type::transparent_bsdf, "Transparent");
    const auto principled =
        graph.add_node(node_type::principled_bsdf, "Principled");
    const auto mix = graph.add_node(node_type::mix_closure, "Mix");
    const auto configured =
        graph.set_input(transparent,
            "Color",
            SocketValue::color({1.0f, 1.0f, 1.0f})) &&
        graph.set_input(principled,
            "BaseColor",
            SocketValue::color({0.32f, 0.12f, 0.06f})) &&
        graph.set_input(
            principled, "Metallic", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "Roughness", SocketValue::floating(0.28f)) &&
        graph.set_input(principled,
            "DiffuseRoughness",
            SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "IOR", SocketValue::floating(1.45f)) &&
        graph.set_input(principled,
            "SpecularIORLevel",
            SocketValue::floating(0.5f)) &&
        graph.set_input(principled,
            "SpecularTint",
            SocketValue::color({0.7f, 0.9f, 1.0f})) &&
        graph.set_input(
            principled, "Alpha", SocketValue::floating(0.6f)) &&
        graph.set_property(
            principled, "Distribution", SocketValue::string("GGX")) &&
        graph.set_input(mix, "Factor", SocketValue::floating(0.75f)) &&
        graph.connect(
            {.node = transparent, .socket = "Closure"}, mix, "A") &&
        graph.connect(
            {.node = principled, .socket = "Closure"}, mix, "B");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure transparent-Principled oracle graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = mix, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_transparent_merge_order_graph() {
    ShaderGraph graph;
    const auto rejected = graph.add_node(
        node_type::transparent_bsdf, "Rejected transparent");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Intervening diffuse");
    const auto allocated = graph.add_node(
        node_type::transparent_bsdf, "Allocated transparent");
    const auto prefix =
        graph.add_node(node_type::add_closure, "Rejected then diffuse");
    const auto root =
        graph.add_node(node_type::add_closure, "Then allocated");
    const auto configured =
        graph.set_input(rejected,
            "Color",
            SocketValue::color({0.5e-5f, 0.5e-5f, 0.5e-5f})) &&
        graph.set_input(diffuse,
            "Color",
            SocketValue::color({0.2f, 0.2f, 0.2f})) &&
        graph.set_input(
            diffuse, "Roughness", SocketValue::floating(0.0f)) &&
        graph.set_input(allocated,
            "Color",
            SocketValue::color({0.25f, 0.25f, 0.25f})) &&
        graph.connect(
            {.node = rejected, .socket = "Closure"}, prefix, "A") &&
        graph.connect(
            {.node = diffuse, .socket = "Closure"}, prefix, "B") &&
        graph.connect(
            {.node = prefix, .socket = "Closure"}, root, "A") &&
        graph.connect(
            {.node = allocated, .socket = "Closure"}, root, "B");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure transparent merge-order graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = root, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_subsurface_principled_graph() {
    ShaderGraph graph;
    const auto principled =
        graph.add_node(node_type::principled_bsdf, "Subsurface Principled");
    const auto configured =
        graph.set_input(principled,
            "BaseColor",
            SocketValue::color({0.35f, 0.24f, 0.8f})) &&
        graph.set_input(
            principled, "Metallic", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "Roughness", SocketValue::floating(0.3f)) &&
        graph.set_input(principled,
            "DiffuseRoughness",
            SocketValue::floating(0.0f)) &&
        graph.set_input(principled,
            "SubsurfaceWeight",
            SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SubsurfaceRadius",
            SocketValue::vector({1.0f, 0.2f, 0.1f})) &&
        graph.set_input(principled,
            "SubsurfaceScale",
            SocketValue::floating(3.0f)) &&
        graph.set_input(
            principled, "IOR", SocketValue::floating(1.45f)) &&
        graph.set_input(principled,
            "SpecularIORLevel",
            SocketValue::floating(0.5f)) &&
        graph.set_input(principled,
            "SpecularTint",
            SocketValue::color({1.0f, 1.0f, 1.0f})) &&
        graph.set_property(principled,
            "Distribution",
            SocketValue::string("MULTI_GGX"));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure subsurface-Principled oracle graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_principled_emission_graph() {
    ShaderGraph graph;
    const auto color = graph.add_node(
        node_type::constant_color,
        "Linked Principled emission color");
    const auto strength = graph.add_node(
        node_type::constant_float,
        "Linked Principled emission strength");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Raw Principled emission closure");
    const auto configured =
        graph.set_input(
            color,
            "Color",
            SocketValue::color({0.17f, 0.43f, 0.91f})) &&
        graph.set_input(
            strength,
            "Value",
            SocketValue::floating(2.75f)) &&
        graph.connect(
            {.node = color, .socket = "Color"},
            principled,
            "EmissionColor") &&
        graph.connect(
            {.node = strength, .socket = "Value"},
            principled,
            "EmissionStrength");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure raw Principled emission graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_layered_principled_emission_graph() {
    ShaderGraph graph;
    const auto geometry = graph.add_node(
        node_type::geometry,
        "Linked Principled coat normal");
    const auto principled = graph.add_node(
        node_type::principled_bsdf,
        "Layered Principled emission closure");
    const auto configured =
        graph.set_input(
            principled,
            "Alpha",
            SocketValue::floating(0.73f)) &&
        graph.set_input(
            principled,
            "SheenWeight",
            SocketValue::floating(0.55f)) &&
        graph.set_input(
            principled,
            "SheenRoughness",
            SocketValue::floating(0.38f)) &&
        graph.set_input(
            principled,
            "SheenTint",
            SocketValue::color({0.9f, 0.35f, 0.12f})) &&
        graph.set_input(
            principled,
            "CoatWeight",
            SocketValue::floating(0.75f)) &&
        graph.set_input(
            principled,
            "CoatRoughness",
            SocketValue::floating(0.23f)) &&
        graph.set_input(
            principled,
            "CoatIOR",
            SocketValue::floating(1.7f)) &&
        graph.set_input(
            principled,
            "CoatTint",
            SocketValue::color({0.3f, 0.7f, 0.95f})) &&
        graph.set_input(
            principled,
            "EmissionColor",
            SocketValue::color({0.17f, 0.43f, 0.91f})) &&
        graph.set_input(
            principled,
            "EmissionStrength",
            SocketValue::floating(2.75f)) &&
        graph.connect(
            {.node = geometry, .socket = "Normal"},
            principled,
            "CoatNormal");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure layered Principled emission graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_principled_sheen_graph() {
    ShaderGraph graph;
    const auto normal = graph.add_node(
        node_type::vector_to_normal, "Linked Sheen Normal");
    const auto principled = graph.add_node(
        node_type::principled_bsdf, "Isolated Principled Sheen");
    const auto configured =
        graph.set_input(normal,
            "Vector",
            SocketValue::vector({0.3f, -0.2f, 0.9f})) &&
        graph.set_input(principled,
            "BaseColor",
            SocketValue::color({0.0f, 0.0f, 0.0f})) &&
        graph.set_input(
            principled, "Metallic", SocketValue::floating(0.0f)) &&
        graph.set_input(
            principled, "Roughness", SocketValue::floating(0.27f)) &&
        graph.set_input(
            principled, "IOR", SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SpecularIORLevel",
            SocketValue::floating(0.5f)) &&
        graph.set_input(
            principled, "Alpha", SocketValue::floating(1.0f)) &&
        graph.set_input(principled,
            "SheenWeight",
            SocketValue::floating(0.75f)) &&
        graph.set_input(principled,
            "SheenRoughness",
            SocketValue::floating(0.37f)) &&
        graph.set_input(principled,
            "SheenTint",
            SocketValue::color({0.8f, 0.4f, 0.2f})) &&
        graph.set_input(
            principled, "CoatWeight", SocketValue::floating(0.0f)) &&
        graph.connect(
            {.node = normal, .socket = "Normal"}, principled, "Normal") &&
        graph.set_property(
            principled, "Distribution", SocketValue::string("GGX"));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure isolated Principled Sheen graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = principled, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_tangent_normal_graph() {
    ShaderGraph graph;
    const auto normal_map =
        graph.add_node(node_type::normal_map, "Normal Map");
    const auto diffuse =
        graph.add_node(node_type::diffuse_bsdf, "Diffuse");
    const auto configured =
        graph.set_input(
            normal_map,
            "Strength",
            SocketValue::floating(1.0f)) &&
        graph.set_input(
            normal_map,
            "Color",
            SocketValue::color(
                {0.8f, 0.3f, 0.9f})) &&
        graph.set_property(
            normal_map,
            "Space",
            SocketValue::string("TANGENT")) &&
        graph.set_input(
            diffuse,
            "Color",
            SocketValue::color(
                {0.5f, 0.5f, 0.5f})) &&
        graph.set_input(
            diffuse,
            "Roughness",
            SocketValue::floating(0.0f)) &&
        graph.connect(
            {.node = normal_map,
             .socket = "Normal"},
            diffuse,
            "Normal");
    if (!configured) {
        throw std::runtime_error{
            "failed to configure tangent-normal oracle graph"};
    }
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = diffuse,
            .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_glass_graph() {
    ShaderGraph graph;
    const auto glass =
        graph.add_node(node_type::glass_bsdf, "Glass");
    const auto configured =
        graph.set_input(glass,
            "Color",
            SocketValue::color({0.72f, 0.86f, 0.94f})) &&
        graph.set_input(
            glass, "Roughness", SocketValue::floating(0.3f)) &&
        graph.set_input(
            glass, "IOR", SocketValue::floating(1.45f)) &&
        graph.set_property(
            glass, "Distribution", SocketValue::string("GGX"));
    if (!configured) {
        throw std::runtime_error{
            "failed to configure glass light-filter graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = glass, .socket = "Closure"});
    return graph;
}

[[nodiscard]] ShaderGraph make_translucent_graph() {
    ShaderGraph graph;
    const auto translucent =
        graph.add_node(node_type::translucent_bsdf, "Translucent");
    if (!graph.set_input(translucent,
            "Color",
            SocketValue::color({0.31f, 0.57f, 0.83f}))) {
        throw std::runtime_error{
            "failed to configure translucent light-filter graph"};
    }
    graph.set_root(ShaderDomain::surface,
        OutputRef{.node = translucent, .socket = "Closure"});
    return graph;
}

} // namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    auto shader = compiler.compile(make_diffuse_graph());
    if (!shader.ok()) {
        std::cerr << "failed to compile diffuse shader graph\n";
        return EXIT_FAILURE;
    }
    auto program = compile_surface_program(*shader.program);
    if (!program.ok()) {
        std::cerr << "failed to lower diffuse surface program\n";
        return EXIT_FAILURE;
    }

    auto physical_shader =
        compiler.compile(make_transparent_principled_graph());
    if (!physical_shader.ok()) {
        std::cerr << "failed to compile transparent-Principled shader "
                     "graph\n";
        return EXIT_FAILURE;
    }
    auto physical_program =
        compile_surface_program(*physical_shader.program);
    if (!physical_program.ok()) {
        std::cerr << "failed to lower transparent-Principled surface "
                     "program\n";
        return EXIT_FAILURE;
    }

    auto transparent_order_shader =
        compiler.compile(make_transparent_merge_order_graph());
    if (!transparent_order_shader.ok()) {
        std::cerr << "failed to compile transparent merge-order graph\n";
        return EXIT_FAILURE;
    }
    auto transparent_order_program =
        compile_surface_program(*transparent_order_shader.program);
    if (!transparent_order_program.ok()) {
        std::cerr
            << "failed to lower transparent merge-order program\n";
        return EXIT_FAILURE;
    }

    auto subsurface_shader =
        compiler.compile(make_subsurface_principled_graph());
    if (!subsurface_shader.ok()) {
        std::cerr << "failed to compile subsurface-Principled shader "
                     "graph\n";
        return EXIT_FAILURE;
    }
    auto subsurface_program =
        compile_surface_program(*subsurface_shader.program);
    if (!subsurface_program.ok()) {
        std::cerr << "failed to lower subsurface-Principled surface "
                     "program\n";
        return EXIT_FAILURE;
    }

    auto principled_emission_shader =
        compiler.compile(make_principled_emission_graph());
    if (!principled_emission_shader.ok()) {
        std::cerr
            << "failed to compile raw Principled emission graph\n";
        return EXIT_FAILURE;
    }
    auto principled_emission_program =
        compile_surface_program(
            *principled_emission_shader.program);
    if (!principled_emission_program.ok() ||
        principled_emission_program.program
                ->emission_evaluation() !=
            EmissionEvaluationMode::deferred) {
        std::cerr
            << "failed to lower deferred Principled emission\n";
        return EXIT_FAILURE;
    }

    auto layered_emission_shader =
        compiler.compile(make_layered_principled_emission_graph());
    if (!layered_emission_shader.ok()) {
        std::cerr
            << "failed to compile layered Principled emission graph\n";
        return EXIT_FAILURE;
    }
    auto layered_emission_program =
        compile_surface_program(*layered_emission_shader.program);
    if (!layered_emission_program.ok() ||
        layered_emission_program.program
                ->emission_evaluation() !=
            EmissionEvaluationMode::deferred) {
        std::cerr
            << "failed to lower layered Principled emission\n";
        return EXIT_FAILURE;
    }

    auto sheen_shader =
        compiler.compile(make_principled_sheen_graph());
    if (!sheen_shader.ok()) {
        std::cerr
            << "failed to compile isolated Principled Sheen graph\n";
        return EXIT_FAILURE;
    }
    auto sheen_program =
        compile_surface_program(*sheen_shader.program);
    if (!sheen_program.ok()) {
        std::cerr
            << "failed to lower isolated Principled Sheen program\n";
        return EXIT_FAILURE;
    }

    auto tangent_normal_shader =
        compiler.compile(make_tangent_normal_graph());
    if (!tangent_normal_shader.ok()) {
        std::cerr
            << "failed to compile tangent-normal shader graph\n";
        return EXIT_FAILURE;
    }
    auto tangent_normal_program =
        compile_surface_program(
            *tangent_normal_shader.program);
    if (!tangent_normal_program.ok()) {
        std::cerr
            << "failed to lower tangent-normal surface program\n";
        return EXIT_FAILURE;
    }

    auto glass_shader = compiler.compile(make_glass_graph());
    if (!glass_shader.ok()) {
        std::cerr << "failed to compile glass light-filter graph\n";
        return EXIT_FAILURE;
    }
    auto glass_program =
        compile_surface_program(*glass_shader.program);
    if (!glass_program.ok()) {
        std::cerr << "failed to lower glass light-filter program\n";
        return EXIT_FAILURE;
    }

    auto translucent_shader =
        compiler.compile(make_translucent_graph());
    if (!translucent_shader.ok()) {
        std::cerr
            << "failed to compile translucent light-filter graph\n";
        return EXIT_FAILURE;
    }
    auto translucent_program =
        compile_surface_program(*translucent_shader.program);
    if (!translucent_program.ok()) {
        std::cerr
            << "failed to lower translucent light-filter program\n";
        return EXIT_FAILURE;
    }

    SurfaceDispatch surfaces;
    const auto surface_tag =
        surfaces.create<GraphSurface>(program.program);
    const auto physical_surface_tag =
        surfaces.create<GraphSurface>(physical_program.program);
    const auto transparent_order_surface_tag =
        surfaces.create<GraphSurface>(transparent_order_program.program);
    const auto subsurface_surface_tag =
        surfaces.create<GraphSurface>(subsurface_program.program);
    const auto principled_emission_surface_tag =
        surfaces.create<GraphSurface>(
            principled_emission_program.program);
    const auto layered_emission_surface_tag =
        surfaces.create<GraphSurface>(
            layered_emission_program.program);
    const auto sheen_surface_tag =
        surfaces.create<GraphSurface>(sheen_program.program);
    const auto tangent_normal_surface_tag =
        surfaces.create<GraphSurface>(
            tangent_normal_program.program);
    const auto glass_surface_tag =
        surfaces.create<GraphSurface>(glass_program.program);
    const auto translucent_surface_tag =
        surfaces.create<GraphSurface>(translucent_program.program);
    auto color_parameter = std::uint32_t{~std::uint32_t{0u}};
    const auto &parameters = program.program->parameters();
    for (std::uint32_t index = 0u; index < parameters.size(); ++index) {
        if (parameters[index].socket == "Color") {
            color_parameter = index;
            break;
        }
    }
    if (color_parameter == ~std::uint32_t{0u}) {
        std::cerr << "diffuse oracle has no Color parameter\n";
        return EXIT_FAILURE;
    }
    Kernel1D evaluate = [&](BufferFloat4 output) noexcept {
        OracleShaderServices services{
            color_parameter, {0.62f, 0.41f, 0.23f}};
        const auto point = make_surface_point();
        const auto query = SurfaceQuery{.lobe_mask = ~std::uint32_t{0u},
            .transport_mode =
                static_cast<std::uint32_t>(TransportMode::radiance),
            .glossy_filter_roughness = 0.0f};
        const auto closure = surfaces.closure_trace(
            UInt{surface_tag}, services, point, 0u);
        const auto sample = surfaces.sample_trace(UInt{surface_tag},
            services,
            point,
            0.37341177463531494f,
            make_float2(0.8208083510398865f, 0.676392674446106f),
            query);
        const auto label = cycles_closure::label_from_events(
            sample.sample.evaluation.events);
        output.write(0u,
            make_float4(cast<float>(closure.count),
                cast<float>(closure.index),
                cast<float>(closure.type),
                closure.sample_weight));
        output.write(1u,
            make_float4(
                closure.weight, select(0.0f, 1.0f, closure.valid)));
        output.write(2u, make_float4(closure.normal, 0.0f));
        output.write(3u,
            make_float4(cast<float>(sample.closure_index),
                cast<float>(sample.closure_type),
                sample.closure_sample_weight,
                sample.selection_rescaled));
        output.write(4u,
            make_float4(sample.closure_weight,
                select(0.0f, 1.0f, sample.closure_valid)));
        output.write(5u, make_float4(sample.closure_normal, 0.0f));
        output.write(6u,
            make_float4(sample.sample.evaluation.pdf,
                sample.sample.evaluation.pdf,
                cast<float>(label),
                select(0.0f, 1.0f, sample.sample.valid)));
        output.write(7u, make_float4(sample.sample.wi, 0.0f));
        output.write(8u, make_float4(sample.sample.evaluation.f, 0.0f));
        output.write(9u,
            make_float4(
                sample.sample.roughness, sample.sample.eta, 0.0f));
        OracleShaderServices below_cutoff_services{
            color_parameter, {0.5e-5f, 0.5e-5f, 0.5e-5f}};
        OracleShaderServices boundary_services{
            color_parameter, {1.0e-5f, 1.0e-5f, 1.0e-5f}};
        const auto below_cutoff = surfaces.closure_trace(
            UInt{surface_tag}, below_cutoff_services, point, 0u);
        const auto at_boundary = surfaces.closure_trace(
            UInt{surface_tag}, boundary_services, point, 0u);
        output.write(10u,
            make_float4(cast<float>(closure.runtime_flags),
                cast<float>(sample.sample.runtime_flags),
                cast<float>(below_cutoff.count),
                cast<float>(at_boundary.count)));
    };

    Kernel1D trace_physical_closures =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            const auto point = make_surface_point();
            const auto requested = dispatch_x();
            const auto closure = surfaces.closure_trace(
                UInt{physical_surface_tag}, services, point, requested);
            output.write(requested,
                make_float4(cast<float>(closure.count),
                    cast<float>(closure.type),
                    closure.sample_weight,
                    select(0.0f, 1.0f, closure.valid)));
            output.write(requested + 4u,
                make_float4(closure.weight,
                    cast<float>(closure.runtime_flags)));
        };

    Kernel1D trace_tangent_normal =
        [&](BufferFloat4 parameters,
            BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            auto point = make_surface_point();
            const auto requested = dispatch_x();
            point.object_shading_normal =
                select(
                    make_float3(
                        0.0f, 0.0f, 0.5f),
                    make_float3(0.0f),
                    requested != 0u);
            const auto closure =
                surfaces.closure_trace(
                    UInt{tangent_normal_surface_tag},
                    services,
                    point,
                    0u);
            output.write(
                requested,
                make_float4(
                    closure.normal,
                    select(
                        0.0f,
                        1.0f,
                        closure.valid)));
        };

    Kernel1D trace_transparent_order =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            const auto point = make_surface_point();
            const auto first = surfaces.closure_trace(
                UInt{transparent_order_surface_tag}, services, point, 0u);
            const auto second = surfaces.closure_trace(
                UInt{transparent_order_surface_tag}, services, point, 1u);
            output.write(0u,
                make_float4(cast<float>(first.count),
                    cast<float>(first.type),
                    first.sample_weight,
                    select(0.0f, 1.0f, first.valid)));
            output.write(1u,
                make_float4(cast<float>(second.count),
                    cast<float>(second.type),
                    second.sample_weight,
                    select(0.0f, 1.0f, second.valid)));
            output.write(2u, make_float4(first.weight, 0.0f));
            output.write(3u, make_float4(second.weight, 0.0f));
        };

    Kernel1D trace_subsurface =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            const auto point = make_surface_point();
            const auto query = SurfaceQuery{
                .lobe_mask = ~std::uint32_t{0u},
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = 0.0f};
            const auto closure = surfaces.closure_trace(
                UInt{subsurface_surface_tag}, services, point, 1u);
            const auto aov = surfaces.aov(
                UInt{subsurface_surface_tag}, services, point);
            const auto sample = surfaces.sample_trace(
                UInt{subsurface_surface_tag},
                services,
                point,
                0.99f,
                make_float2(0.37f, 0.61f),
                query);
            output.write(0u,
                make_float4(closure.weight,
                    select(0.0f, 1.0f, closure.valid)));
            output.write(1u,
                make_float4(aov.albedo, cast<float>(closure.count)));
            output.write(2u,
                make_float4(cast<float>(closure.type),
                    closure.sample_weight,
                    0.0f,
                    0.0f));
            output.write(3u,
                make_float4(cast<float>(sample.closure_index),
                    cast<float>(sample.closure_type),
                    sample.closure_sample_weight,
                    sample.selection_rescaled));
            output.write(4u,
                make_float4(sample.sample.evaluation.f,
                    sample.sample.evaluation.pdf));
            output.write(5u,
                make_float4(sample.sample.bssrdf_radius,
                    cast<float>(sample.sample.bssrdf_method)));
            output.write(6u,
                make_float4(sample.sample.bssrdf_albedo,
                    sample.sample.bssrdf_ior));
            output.write(7u,
                make_float4(sample.sample.bssrdf_normal,
                    sample.sample.bssrdf_roughness));
            output.write(8u,
                make_float4(sample.sample.bssrdf_anisotropy,
                    cast<float>(sample.sample.evaluation.events),
                    select(0.0f, 1.0f, sample.sample.valid),
                    select(0.0f, 1.0f, sample.closure_valid)));
        };

    Kernel1D evaluate_principled_emission =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            const auto point = make_surface_point();
            const auto emission = surfaces.emission(
                UInt{principled_emission_surface_tag},
                services,
                point,
                make_float3(0.0f, 0.0f, 1.0f),
                true);
            const auto constant = surfaces.constant_emission(
                UInt{principled_emission_surface_tag},
                services,
                0u);
            output.write(0u, make_float4(emission, 1.0f));
            output.write(1u, make_float4(constant, 0.0f));
        };

    Kernel1D evaluate_layered_emission =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            const auto point = make_surface_point();
            const auto with_reflection = surfaces.emission(
                UInt{layered_emission_surface_tag},
                services,
                point,
                make_float3(0.0f, 0.0f, 1.0f),
                true);
            const auto without_reflection = surfaces.emission(
                UInt{layered_emission_surface_tag},
                services,
                point,
                make_float3(0.0f, 0.0f, 1.0f),
                false);
            const auto constant = surfaces.constant_emission(
                UInt{layered_emission_surface_tag},
                services,
                0u);
            const auto first_closure = surfaces.closure_trace(
                UInt{layered_emission_surface_tag},
                services,
                point,
                0u);
            const auto extinction = surfaces.transparent_extinction(
                UInt{layered_emission_surface_tag},
                services,
                point);
            output.write(
                0u, make_float4(with_reflection, 1.0f));
            output.write(
                1u, make_float4(without_reflection, 1.0f));
            output.write(2u, make_float4(constant, 0.0f));
            output.write(3u,
                make_float4(
                    cast<float>(first_closure.index),
                    cast<float>(first_closure.type),
                    first_closure.sample_weight,
                    select(0.0f, 1.0f, first_closure.valid)));
            output.write(4u,
                make_float4(first_closure.weight,
                    cast<float>(first_closure.runtime_flags)));
            output.write(5u, make_float4(extinction, 0.0f));
        };

    Kernel1D evaluate_principled_sheen =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices valid_services{parameters};
            ParameterShaderServices invalid_services{parameters, 0.0f};
            const auto point = make_surface_point();
            const auto query = SurfaceQuery{
                .lobe_mask = ~std::uint32_t{0u},
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = 0.0f};
            const auto outgoing = normalize(
                make_float3(0.25f, 0.35f, 0.9027735f));
            const auto closure = surfaces.closure_trace(
                UInt{sheen_surface_tag}, valid_services, point, 0u);
            const auto evaluation = surfaces.evaluate(
                UInt{sheen_surface_tag},
                valid_services,
                point,
                outgoing,
                query);
            const auto sample = surfaces.sample_trace(
                UInt{sheen_surface_tag},
                valid_services,
                point,
                0.41f,
                make_float2(0.23f, 0.71f),
                query);
            const auto aov = surfaces.aov(
                UInt{sheen_surface_tag}, valid_services, point);
            output.write(0u,
                make_float4(cast<float>(closure.count),
                    cast<float>(closure.type),
                    closure.sample_weight,
                    select(0.0f, 1.0f, closure.valid)));
            output.write(1u,
                make_float4(closure.weight,
                    cast<float>(closure.runtime_flags)));
            output.write(2u, make_float4(closure.normal, 0.0f));
            output.write(3u,
                make_float4(evaluation.f, evaluation.pdf));
            output.write(4u,
                make_float4(
                    evaluation.diffuse_f, evaluation.diffuse_pdf));
            output.write(5u,
                make_float4(evaluation.glossy_f,
                    cast<float>(evaluation.events)));
            output.write(6u,
                make_float4(cast<float>(sample.closure_index),
                    cast<float>(sample.closure_type),
                    sample.closure_sample_weight,
                    sample.selection_rescaled));
            output.write(7u,
                make_float4(sample.sample.wi,
                    select(0.0f, 1.0f, sample.sample.valid)));
            output.write(8u,
                make_float4(sample.sample.evaluation.f,
                    sample.sample.evaluation.pdf));
            output.write(9u,
                make_float4(sample.sample.evaluation.diffuse_f,
                    sample.sample.evaluation.diffuse_pdf));
            output.write(10u,
                make_float4(sample.sample.roughness,
                    sample.sample.eta,
                    cast<float>(sample.sample.evaluation.events)));
            output.write(11u,
                make_float4(aov.albedo, aov.roughness.x));
            output.write(12u,
                make_float4(aov.normal, aov.roughness.y));

            const auto invalid_closure = surfaces.closure_trace(
                UInt{sheen_surface_tag}, invalid_services, point, 0u);
            const auto invalid_evaluation = surfaces.evaluate(
                UInt{sheen_surface_tag},
                invalid_services,
                point,
                outgoing,
                query);
            const auto invalid_sample = surfaces.sample_trace(
                UInt{sheen_surface_tag},
                invalid_services,
                point,
                0.41f,
                make_float2(0.23f, 0.71f),
                query);
            const auto invalid_aov = surfaces.aov(
                UInt{sheen_surface_tag}, invalid_services, point);
            output.write(13u,
                make_float4(cast<float>(invalid_closure.count),
                    cast<float>(invalid_closure.type),
                    invalid_closure.sample_weight,
                    select(0.0f, 1.0f, invalid_closure.valid)));
            output.write(14u,
                make_float4(invalid_closure.weight,
                    cast<float>(invalid_closure.runtime_flags)));
            output.write(15u,
                make_float4(invalid_evaluation.f,
                    invalid_evaluation.pdf));
            output.write(16u,
                make_float4(cast<float>(invalid_sample.closure_type),
                    invalid_sample.closure_sample_weight,
                    select(0.0f,
                        1.0f,
                        invalid_sample.closure_valid),
                    select(0.0f,
                        1.0f,
                        invalid_sample.sample.valid)));
            output.write(17u,
                make_float4(
                    invalid_aov.albedo, invalid_aov.roughness.x));
            output.write(18u,
                make_float4(
                    invalid_aov.normal, invalid_aov.roughness.y));

            auto uncorrected_point = point;
            uncorrected_point.use_bump_map_correction = false;
            const auto uncorrected_evaluation = surfaces.evaluate(
                UInt{sheen_surface_tag},
                valid_services,
                uncorrected_point,
                outgoing,
                query);
            const auto uncorrected_sample = surfaces.sample_trace(
                UInt{sheen_surface_tag},
                valid_services,
                uncorrected_point,
                0.41f,
                make_float2(0.23f, 0.71f),
                query);
            const auto leaking_direction = normalize(
                make_float3(0.8f, 0.0f, -0.1f));
            const auto rejected_leak = surfaces.evaluate(
                UInt{sheen_surface_tag},
                valid_services,
                uncorrected_point,
                leaking_direction,
                query);
            const auto grazing_direction = normalize(
                make_float3(1.0f, 0.0f, 5.0e-7f));
            const auto rejected_grazing = surfaces.evaluate(
                UInt{sheen_surface_tag},
                valid_services,
                point,
                grazing_direction,
                query);
            const auto sampled_grazing = surfaces.sample_trace(
                UInt{sheen_surface_tag},
                valid_services,
                point,
                0.41f,
                make_float2(0.200493872f, 0.5f),
                query);
            const auto invalid_geometric_sample = surfaces.sample_trace(
                UInt{sheen_surface_tag},
                valid_services,
                point,
                0.41f,
                make_float2(0.200491f, 0.5f),
                query);
            output.write(19u,
                make_float4(uncorrected_evaluation.f,
                    uncorrected_evaluation.pdf));
            output.write(20u,
                make_float4(uncorrected_sample.sample.evaluation.f,
                    uncorrected_sample.sample.evaluation.pdf));
            output.write(21u,
                make_float4(rejected_leak.f, rejected_leak.pdf));
            output.write(22u,
                make_float4(
                    rejected_grazing.f, rejected_grazing.pdf));
            output.write(23u,
                make_float4(sampled_grazing.sample.wi,
                    select(0.0f,
                        1.0f,
                        sampled_grazing.sample.valid)));
            output.write(24u,
                make_float4(sampled_grazing.sample.evaluation.f,
                    sampled_grazing.sample.evaluation.pdf));
            output.write(25u,
                make_float4(invalid_geometric_sample.sample.wi,
                    select(0.0f,
                        1.0f,
                        invalid_geometric_sample.sample.valid)));
            output.write(26u,
                make_float4(
                    invalid_geometric_sample.sample.evaluation.f,
                    invalid_geometric_sample.sample.evaluation.pdf));
        };

    Kernel1D evaluate_physical_light =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            const auto point = make_surface_point();
            const auto query = SurfaceQuery{
                .lobe_mask = ~std::uint32_t{0u},
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = 0.0f};
            const auto case_index = dispatch_x();
            UInt shader_flags = cycles_abi::shader_use_mis;
            shader_flags |= select(0u,
                cycles_abi::shader_exclude_diffuse,
                (case_index == 1u) | (case_index == 3u));
            shader_flags |= select(0u,
                cycles_abi::shader_exclude_glossy,
                (case_index == 2u) | (case_index == 3u));
            shader_flags =
                select(shader_flags, 0u, case_index == 4u);
            const auto evaluation = surfaces.evaluate_light(
                UInt{physical_surface_tag},
                services,
                point,
                normalize(make_float3(0.35f, 0.0f, 0.94f)),
                SurfaceLightQuery{
                    .surface = query,
                    .shader_flags = shader_flags});
            const auto base = case_index * 3u;
            output.write(base,
                make_float4(evaluation.f, evaluation.pdf));
            output.write(base + 1u,
                make_float4(
                    evaluation.diffuse_f, evaluation.diffuse_pdf));
            output.write(base + 2u,
                make_float4(evaluation.glossy_f,
                    cast<float>(evaluation.events)));
        };

    Kernel1D evaluate_glass_light =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            const auto point = make_surface_point();
            const auto query = SurfaceQuery{
                .lobe_mask = ~std::uint32_t{0u},
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = 0.0f};
            const auto case_index = dispatch_x();
            UInt shader_flags = cycles_abi::shader_use_mis;
            shader_flags |= select(0u,
                cycles_abi::shader_exclude_glossy,
                (case_index == 1u) | (case_index == 3u));
            shader_flags |= select(0u,
                cycles_abi::shader_exclude_transmit,
                (case_index == 2u) | (case_index == 3u));
            const auto evaluation = surfaces.evaluate_light(
                UInt{glass_surface_tag},
                services,
                point,
                normalize(make_float3(0.2f, 0.0f, 0.98f)),
                SurfaceLightQuery{
                    .surface = query,
                    .shader_flags = shader_flags});
            const auto base = case_index * 3u;
            output.write(base,
                make_float4(evaluation.f, evaluation.pdf));
            output.write(base + 1u,
                make_float4(
                    evaluation.diffuse_f, evaluation.diffuse_pdf));
            output.write(base + 2u,
                make_float4(evaluation.glossy_f,
                    cast<float>(evaluation.events)));
        };

    Kernel1D evaluate_translucent_light =
        [&](BufferFloat4 parameters, BufferFloat4 output) noexcept {
            ParameterShaderServices services{parameters};
            const auto point = make_surface_point();
            const auto query = SurfaceQuery{
                .lobe_mask = ~std::uint32_t{0u},
                .transport_mode = static_cast<std::uint32_t>(
                    TransportMode::radiance),
                .glossy_filter_roughness = 0.0f};
            const auto case_index = dispatch_x();
            UInt shader_flags = cycles_abi::shader_use_mis;
            shader_flags |= select(0u,
                cycles_abi::shader_exclude_diffuse,
                case_index == 1u);
            shader_flags |= select(0u,
                cycles_abi::shader_exclude_transmit,
                case_index == 2u);
            const auto evaluation = surfaces.evaluate_light(
                UInt{translucent_surface_tag},
                services,
                point,
                normalize(make_float3(0.2f, 0.0f, -0.98f)),
                SurfaceLightQuery{
                    .surface = query,
                    .shader_flags = shader_flags});
            const auto base = case_index * 3u;
            output.write(base,
                make_float4(evaluation.f, evaluation.pdf));
            output.write(base + 1u,
                make_float4(
                    evaluation.diffuse_f, evaluation.diffuse_pdf));
            output.write(base + 2u,
                make_float4(evaluation.glossy_f,
                    cast<float>(evaluation.events)));
        };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(11u);
    const auto physical_parameters =
        parameter_data(*physical_program.program);
    auto physical_parameter_buffer =
        device.create_buffer<luisa::float4>(physical_parameters.size());
    auto physical_output = device.create_buffer<luisa::float4>(8u);
    const auto transparent_order_parameters =
        parameter_data(*transparent_order_program.program);
    auto transparent_order_parameter_buffer =
        device.create_buffer<luisa::float4>(
            transparent_order_parameters.size());
    auto transparent_order_output =
        device.create_buffer<luisa::float4>(4u);
    const auto subsurface_parameters =
        parameter_data(*subsurface_program.program);
    auto subsurface_parameter_buffer =
        device.create_buffer<luisa::float4>(subsurface_parameters.size());
    auto subsurface_output = device.create_buffer<luisa::float4>(9u);
    const auto principled_emission_parameters =
        parameter_data(*principled_emission_program.program);
    auto principled_emission_parameter_buffer =
        device.create_buffer<luisa::float4>(
            principled_emission_parameters.size());
    auto principled_emission_output =
        device.create_buffer<luisa::float4>(2u);
    const auto layered_emission_parameters =
        parameter_data(*layered_emission_program.program);
    auto layered_emission_parameter_buffer =
        device.create_buffer<luisa::float4>(
            layered_emission_parameters.size());
    auto layered_emission_output =
        device.create_buffer<luisa::float4>(6u);
    const auto sheen_parameters =
        parameter_data(*sheen_program.program);
    auto sheen_parameter_buffer =
        device.create_buffer<luisa::float4>(sheen_parameters.size());
    auto sheen_output =
        device.create_buffer<luisa::float4>(27u);
    const auto tangent_normal_parameters =
        parameter_data(*tangent_normal_program.program);
    auto tangent_normal_parameter_buffer =
        device.create_buffer<luisa::float4>(
            tangent_normal_parameters.size());
    auto tangent_normal_output =
        device.create_buffer<luisa::float4>(2u);
    const auto glass_parameters =
        parameter_data(*glass_program.program);
    auto glass_parameter_buffer =
        device.create_buffer<luisa::float4>(glass_parameters.size());
    const auto translucent_parameters =
        parameter_data(*translucent_program.program);
    auto translucent_parameter_buffer =
        device.create_buffer<luisa::float4>(
            translucent_parameters.size());
    auto physical_light_output =
        device.create_buffer<luisa::float4>(15u);
    auto glass_light_output =
        device.create_buffer<luisa::float4>(12u);
    auto translucent_light_output =
        device.create_buffer<luisa::float4>(9u);
    auto kernel = device.compile(evaluate);
    auto physical_kernel = device.compile(trace_physical_closures);
    auto transparent_order_kernel =
        device.compile(trace_transparent_order);
    auto subsurface_kernel = device.compile(trace_subsurface);
    auto principled_emission_kernel =
        device.compile(evaluate_principled_emission);
    auto layered_emission_kernel =
        device.compile(evaluate_layered_emission);
    auto sheen_kernel =
        device.compile(evaluate_principled_sheen);
    auto tangent_normal_kernel =
        device.compile(trace_tangent_normal);
    auto physical_light_kernel =
        device.compile(evaluate_physical_light);
    auto glass_light_kernel =
        device.compile(evaluate_glass_light);
    auto translucent_light_kernel =
        device.compile(evaluate_translucent_light);
    std::array<luisa::float4, 11u> actual{};
    std::array<luisa::float4, 8u> physical_actual{};
    std::array<luisa::float4, 4u> transparent_order_actual{};
    std::array<luisa::float4, 9u> subsurface_actual{};
    std::array<luisa::float4, 2u>
        principled_emission_actual{};
    std::array<luisa::float4, 6u>
        layered_emission_actual{};
    std::array<luisa::float4, 27u> sheen_actual{};
    std::array<luisa::float4, 2u>
        tangent_normal_actual{};
    std::array<luisa::float4, 15u> physical_light_actual{};
    std::array<luisa::float4, 12u> glass_light_actual{};
    std::array<luisa::float4, 9u> translucent_light_actual{};
    stream << physical_parameter_buffer.copy_from(
                  luisa::span{physical_parameters})
           << subsurface_parameter_buffer.copy_from(
                  luisa::span{subsurface_parameters})
           << transparent_order_parameter_buffer.copy_from(
                  luisa::span{transparent_order_parameters})
           << principled_emission_parameter_buffer.copy_from(
                  luisa::span{principled_emission_parameters})
           << layered_emission_parameter_buffer.copy_from(
                  luisa::span{layered_emission_parameters})
           << sheen_parameter_buffer.copy_from(
                  luisa::span{sheen_parameters})
           << tangent_normal_parameter_buffer.copy_from(
                  luisa::span{
                      tangent_normal_parameters})
           << glass_parameter_buffer.copy_from(
                  luisa::span{glass_parameters})
           << translucent_parameter_buffer.copy_from(
                  luisa::span{translucent_parameters})
           << kernel(output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << physical_kernel(
                  physical_parameter_buffer, physical_output)
                  .dispatch(4u)
           << physical_output.copy_to(luisa::span{physical_actual})
           << transparent_order_kernel(
                  transparent_order_parameter_buffer,
                  transparent_order_output)
                  .dispatch(1u)
           << transparent_order_output.copy_to(
                  luisa::span{transparent_order_actual})
           << subsurface_kernel(
                  subsurface_parameter_buffer, subsurface_output)
                  .dispatch(1u)
           << subsurface_output.copy_to(
                  luisa::span{subsurface_actual})
           << principled_emission_kernel(
                  principled_emission_parameter_buffer,
                  principled_emission_output)
                  .dispatch(1u)
           << principled_emission_output.copy_to(
                  luisa::span{principled_emission_actual})
           << layered_emission_kernel(
                  layered_emission_parameter_buffer,
                  layered_emission_output)
                  .dispatch(1u)
           << layered_emission_output.copy_to(
                  luisa::span{layered_emission_actual})
           << sheen_kernel(
                  sheen_parameter_buffer, sheen_output)
                  .dispatch(1u)
           << sheen_output.copy_to(
                  luisa::span{sheen_actual})
           << tangent_normal_kernel(
                  tangent_normal_parameter_buffer,
                  tangent_normal_output)
                  .dispatch(2u)
           << tangent_normal_output.copy_to(
                  luisa::span{
                      tangent_normal_actual})
           << physical_light_kernel(
                  physical_parameter_buffer,
                  physical_light_output)
                  .dispatch(5u)
           << physical_light_output.copy_to(
                  luisa::span{physical_light_actual})
           << glass_light_kernel(
                  glass_parameter_buffer,
                  glass_light_output)
                  .dispatch(4u)
           << glass_light_output.copy_to(
                  luisa::span{glass_light_actual})
           << translucent_light_kernel(
                  translucent_parameter_buffer,
                  translucent_light_output)
                  .dispatch(3u)
           << translucent_light_output.copy_to(
                  luisa::span{translucent_light_actual})
           << synchronize();

    if (std::getenv("PSYCLES_DUMP_SHEEN_REGRESSION") != nullptr) {
        for (std::size_t index = 0u; index < sheen_actual.size(); ++index) {
            const auto value = sheen_actual[index];
            std::cerr << index << ": {" << value.x << ", "
                      << value.y << ", " << value.z << ", "
                      << value.w << "}\n";
        }
    }

    constexpr std::array expected{
        luisa::float4{1.0f, 0.0f, 2.0f, 0.420000017f},
        luisa::float4{0.62f, 0.41f, 0.23f, 1.0f},
        luisa::float4{0.0f, 0.0f, 1.0f, 0.0f},
        luisa::float4{0.0f, 2.0f, 0.420000017f, 0.373411775f},
        luisa::float4{0.62f, 0.41f, 0.23f, 1.0f},
        luisa::float4{0.0f, 0.0f, 1.0f, 0.0f},
        luisa::float4{0.244151756f, 0.244151756f, 6.0f, 1.0f},
        luisa::float4{0.601930976f, -0.222150967f, 0.767025411f, 0.0f},
        luisa::float4{0.151374087f, 0.100102216f, 0.056154907f, 0.0f},
        luisa::float4{1.0f, 1.0f, 1.0f, 0.0f},
        luisa::float4{24.0f, 24.0f, 0.0f, 1.0f}};
    for (std::size_t index = 0u; index < expected.size(); ++index) {
        if (!approximately_equal(actual[index], expected[index])) {
            std::cerr << "Cycles closure oracle failed on " << backend
                      << " at record " << index << ": got {"
                      << actual[index].x << ", " << actual[index].y
                      << ", " << actual[index].z << ", "
                      << actual[index].w << "}\n";
            return EXIT_FAILURE;
        }
    }
    if (!approximately_equal(
            principled_emission_actual[0u],
            luisa::float4{
                0.4675f,
                1.1825f,
                2.5025f,
                1.0f}) ||
        !approximately_equal(
            principled_emission_actual[1u],
            luisa::float4{})) {
        const auto value = principled_emission_actual[0u];
        std::cerr
            << "Cycles raw Principled emission failed on "
            << backend << ": got {" << value.x << ", "
            << value.y << ", " << value.z << ", "
            << value.w << "}\n";
        return EXIT_FAILURE;
    }

    // The service supplies a constant Cycles table value of 0.5 for the
    // valid branch, making the complete LTC setup/eval/sample relation a
    // deterministic device oracle. The zero-table branch models Cycles'
    // failed setup state: its allocated slot remains observable as type
    // NONE with the pre-setup weight, but contributes no flags, AOV, PDF,
    // or sample.
    constexpr auto diffuse_reflection_events =
        static_cast<float>(event_diffuse | event_reflection);
    constexpr std::array sheen_expected{
        luisa::float4{1.0f,
            static_cast<float>(cycles_closure::type_sheen),
            0.175f,
            1.0f},
        luisa::float4{0.3f, 0.15f, 0.075f, 24.0f},
        luisa::float4{0.309426f, -0.206284f, 0.928279f, 0.0f},
        luisa::float4{0.0167059f, 0.00835297f, 0.00417648f, 0.0557498f},
        luisa::float4{0.0167059f, 0.00835297f, 0.00417648f, 0.0557498f},
        luisa::float4{0.0f, 0.0f, 0.0f, diffuse_reflection_events},
        luisa::float4{0.0f,
            static_cast<float>(cycles_closure::type_sheen),
            0.175f,
            0.41f},
        luisa::float4{0.61952f, -0.78194f, 0.069027f, 1.0f},
        luisa::float4{0.100744f, 0.0503718f, 0.0251859f, 0.550437f},
        luisa::float4{0.100744f, 0.0503718f, 0.0251859f, 0.550437f},
        luisa::float4{1.0f, 1.0f, 1.0f, diffuse_reflection_events},
        luisa::float4{0.3f, 0.15f, 0.075f, 1.0f},
        luisa::float4{0.309426f, -0.206284f, 0.928279f, 1.0f},
        luisa::float4{1.0f,
            static_cast<float>(cycles_closure::type_none),
            0.0f,
            1.0f},
        luisa::float4{0.6f, 0.3f, 0.15f, 0.0f},
        luisa::float4{},
        luisa::float4{},
        luisa::float4{0.0f, 0.0f, 0.0f, 1.0f},
        luisa::float4{0.0f, 0.0f, 1.0f, 1.0f},
        // Disabling material bump correction removes only the GGX
        // smoothing factor. The LTC density remains unchanged.
        luisa::float4{0.0167249f, 0.00836247f, 0.00418123f, 0.0557498f},
        luisa::float4{0.165131f, 0.0825655f, 0.0412828f, 0.550437f},
        // The smooth-normal hemisphere rejection is a separate invariant
        // and remains active when smoothing is disabled.
        luisa::float4{},
        // Cycles explicitly rejects corrected diffuse evaluation when the
        // direction is within 1e-6 of the smooth-normal tangent plane.
        luisa::float4{},
        // The same direction sampled from the selected closure remains a
        // valid geometric reflection. bsdf_sample scales its evaluation to
        // zero but preserves the selected technique's original PDF.
        luisa::float4{0.83205f, -0.5547f, 0.0f, 1.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 0.636108f},
        // A nearby sample crosses below the geometric normal. Cycles
        // returns a zero selected PDF and skips all competing evaluation.
        luisa::float4{0.83205f, -0.5547f, -2.6e-6f, 0.0f},
        luisa::float4{}};
    for (std::size_t index = 0u; index < sheen_expected.size(); ++index) {
        if (!approximately_equal(
                sheen_actual[index], sheen_expected[index])) {
            const auto value = sheen_actual[index];
            std::cerr
                << "Cycles Principled Sheen LTC regression failed on "
                << backend << " at record " << index << ": got {"
                << value.x << ", " << value.y << ", " << value.z
                << ", " << value.w << "}\n";
            return EXIT_FAILURE;
        }
    }

    const auto with_coat_reflection = layered_emission_actual[0u];
    const auto without_coat_reflection = layered_emission_actual[1u];
    const auto layered_finite = [](luisa::float4 value) noexcept {
        return std::isfinite(value.x) &&
               std::isfinite(value.y) &&
               std::isfinite(value.z);
    };
    if (!layered_finite(with_coat_reflection) ||
        !layered_finite(without_coat_reflection) ||
        !(with_coat_reflection.x > 0.0f &&
            with_coat_reflection.y > 0.0f &&
            with_coat_reflection.z > 0.0f) ||
        !(without_coat_reflection.x > with_coat_reflection.x &&
            without_coat_reflection.y > with_coat_reflection.y &&
            without_coat_reflection.z > with_coat_reflection.z) ||
        !approximately_equal(
            layered_emission_actual[2u],
            luisa::float4{}) ||
        !approximately_equal(
            layered_emission_actual[3u],
            luisa::float4{
                0.0f,
                static_cast<float>(cycles_closure::type_transparent),
                0.27f,
                1.0f}) ||
        !approximately_equal(
            layered_emission_actual[4u].x, 0.27f) ||
        !approximately_equal(
            layered_emission_actual[4u].y, 0.27f) ||
        !approximately_equal(
            layered_emission_actual[4u].z, 0.27f) ||
        !approximately_equal(
            layered_emission_actual[5u],
            luisa::float4{0.27f, 0.27f, 0.27f, 0.0f})) {
        std::cerr
            << "Cycles layered Principled emission/caustics branch "
               "failed on "
            << backend << ": reflective {"
            << with_coat_reflection.x << ", "
            << with_coat_reflection.y << ", "
            << with_coat_reflection.z << "}, non-reflective {"
            << without_coat_reflection.x << ", "
            << without_coat_reflection.y << ", "
            << without_coat_reflection.z << "}\n";
        return EXIT_FAILURE;
    }

    // This is the exact physical closure order exposed by the Lone Monk
    // grass material: Transparent, dielectric GGX, then Diffuse. A
    // virtual aggregate Principled closure (type 43) must never reach
    // selection.
    constexpr std::array expected_types{
        cycles_closure::type_transparent,
        cycles_closure::type_microfacet_ggx,
        cycles_closure::type_diffuse,
        0u};
    for (std::size_t index = 0u; index < expected_types.size();
        ++index) {
        const auto meta = physical_actual[index];
        const auto valid = index < 3u;
        const auto sample_weight_valid =
            valid ? meta.z > 0.0f : meta.z == 0.0f;
        if (!approximately_equal(meta.x, 3.0f) ||
            !approximately_equal(
                meta.y, static_cast<float>(expected_types[index])) ||
            !approximately_equal(meta.w, valid ? 1.0f : 0.0f) ||
            !sample_weight_valid ||
            (valid && static_cast<std::uint32_t>(meta.y) ==
                          cycles_closure::type_principled_virtual)) {
            std::cerr << "Cycles physical closure expansion failed on "
                      << backend << " at index " << index << ": got {"
                      << meta.x << ", " << meta.y << ", " << meta.z
                      << ", " << meta.w << "}\n";
            return EXIT_FAILURE;
        }
        if (!approximately_equal(
                physical_actual[index + 4u].w, 1048.0f)) {
            std::cerr << "Cycles physical closure flags failed on "
                      << backend << " at index " << index << '\n';
            return EXIT_FAILURE;
        }
    }
    if (!approximately_equal(physical_actual[4u],
            luisa::float4{0.55f, 0.55f, 0.55f, 1048.0f}) ||
        !approximately_equal(physical_actual[5u],
            luisa::float4{0.45f, 0.45f, 0.45f, 1048.0f}) ||
        !(physical_actual[6u].x > 0.0f &&
            physical_actual[6u].y > 0.0f &&
            physical_actual[6u].z > 0.0f)) {
        std::cerr << "Cycles physical closure weights failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }

    constexpr std::array transparent_order_expected{
        luisa::float4{2.0f,
            static_cast<float>(cycles_closure::type_diffuse),
            0.2f,
            1.0f},
        luisa::float4{2.0f,
            static_cast<float>(cycles_closure::type_transparent),
            0.25f,
            1.0f},
        luisa::float4{0.2f, 0.2f, 0.2f, 0.0f},
        luisa::float4{0.25f, 0.25f, 0.25f, 0.0f}};
    for (std::size_t index = 0u;
        index < transparent_order_expected.size();
        ++index) {
        if (!approximately_equal(transparent_order_actual[index],
                transparent_order_expected[index])) {
            const auto value = transparent_order_actual[index];
            std::cerr
                << "Cycles transparent first-allocation order failed on "
                << backend << " at record " << index << ": got {"
                << value.x << ", " << value.y << ", " << value.z
                << ", " << value.w << "}\n";
            return EXIT_FAILURE;
        }
    }

    const auto subsurface_weight = subsurface_actual[0u];
    const auto subsurface_aov = subsurface_actual[1u];
    const auto subsurface_meta = subsurface_actual[2u];
    const auto subsurface_sample_meta = subsurface_actual[3u];
    const auto subsurface_sample = subsurface_actual[4u];
    const auto subsurface_radius = subsurface_actual[5u];
    const auto subsurface_albedo = subsurface_actual[6u];
    const auto subsurface_normal = subsurface_actual[7u];
    const auto subsurface_payload_meta = subsurface_actual[8u];
    if (!approximately_equal(subsurface_weight.w, 1.0f) ||
        !approximately_equal(subsurface_aov.w, 2.0f) ||
        !approximately_equal(subsurface_meta.x,
            static_cast<float>(
                cycles_closure::type_bssrdf_random_walk)) ||
        !(subsurface_meta.y > 0.0f) ||
        !approximately_equal(subsurface_aov,
            luisa::float4{subsurface_weight.x,
                subsurface_weight.y,
                subsurface_weight.z,
                2.0f}) ||
        !approximately_equal(subsurface_sample_meta.x, 1.0f) ||
        !approximately_equal(subsurface_sample_meta.y,
            static_cast<float>(
                cycles_closure::type_bssrdf_random_walk)) ||
        !approximately_equal(
            subsurface_sample_meta.z, subsurface_meta.y) ||
        !approximately_equal(subsurface_sample.w, 1.0f) ||
        !(subsurface_sample.x > 0.0f &&
            subsurface_sample.y > 0.0f &&
            subsurface_sample.z > 0.0f) ||
        !approximately_equal(subsurface_radius,
            luisa::float4{3.0f,
                0.6f,
                0.3f,
                static_cast<float>(SurfaceBssrdfMethod::random_walk)}) ||
        !approximately_equal(subsurface_albedo,
            luisa::float4{0.35f, 0.24f, 0.8f, 1.45f}) ||
        !approximately_equal(subsurface_normal,
            luisa::float4{0.0f, 0.0f, 1.0f, 0.09f}) ||
        !approximately_equal(subsurface_payload_meta.x, 0.0f) ||
        !approximately_equal(subsurface_payload_meta.y,
            static_cast<float>(event_subsurface)) ||
        !approximately_equal(subsurface_payload_meta.z, 1.0f) ||
        !approximately_equal(subsurface_payload_meta.w, 1.0f)) {
        std::cerr << "Native Principled BSSRDF contract failed on "
                  << backend << ": weight {" << subsurface_weight.x
                  << ", " << subsurface_weight.y << ", "
                  << subsurface_weight.z << ", "
                  << subsurface_weight.w << "}, AOV {"
                  << subsurface_aov.x << ", " << subsurface_aov.y
                  << ", " << subsurface_aov.z << ", "
                  << subsurface_aov.w << "}, meta {"
                  << subsurface_meta.x << ", "
                  << subsurface_meta.y << "}, sample meta {"
                  << subsurface_sample_meta.x << ", "
                  << subsurface_sample_meta.y << ", "
                  << subsurface_sample_meta.z << ", "
                  << subsurface_sample_meta.w << "}, radius {"
                  << subsurface_radius.x << ", "
                  << subsurface_radius.y << ", "
                  << subsurface_radius.z << ", "
                  << subsurface_radius.w << "}, albedo/ior {"
                  << subsurface_albedo.x << ", "
                  << subsurface_albedo.y << ", "
                  << subsurface_albedo.z << ", "
                  << subsurface_albedo.w << "}, normal/roughness {"
                  << subsurface_normal.x << ", "
                  << subsurface_normal.y << ", "
                  << subsurface_normal.z << ", "
                  << subsurface_normal.w << "}, payload meta {"
                  << subsurface_payload_meta.x << ", "
                  << subsurface_payload_meta.y << ", "
                  << subsurface_payload_meta.z << ", "
                  << subsurface_payload_meta.w << "}\n";
        return EXIT_FAILURE;
    }

    // Current Cycles normalizes the interpolated object-space normal
    // before constructing the MikkTSpace frame. A unit-only fixture
    // cannot distinguish that contract from the old scale-dependent
    // behavior, so retain an explicitly non-unit base normal here.
    constexpr std::array tangent_normal_expected{
        luisa::float4{
            0.557086014f,
            -0.371390671f,
            0.742781341f,
            1.0f},
        luisa::float4{
            0.0f,
            0.0f,
            1.0f,
            1.0f}};
    for (std::size_t index = 0u;
         index < tangent_normal_expected.size();
         ++index) {
        if (!approximately_equal(
                tangent_normal_actual[index],
                tangent_normal_expected[index])) {
            const auto value =
                tangent_normal_actual[index];
            std::cerr
                << "Cycles tangent-normal frame failed on "
                << backend << " at record " << index
                << ": got {" << value.x << ", "
                << value.y << ", " << value.z << ", "
                << value.w << "}\n";
            return EXIT_FAILURE;
        }
    }

    const auto rgb_equal = [](luisa::float4 lhs,
                               luisa::float4 rhs) noexcept {
        return approximately_equal(lhs.x, rhs.x) &&
               approximately_equal(lhs.y, rhs.y) &&
               approximately_equal(lhs.z, rhs.z);
    };
    const auto rgb_zero = [&](luisa::float4 value) noexcept {
        return rgb_equal(value, luisa::float4{});
    };
    const auto rgb_positive = [](luisa::float4 value) noexcept {
        return value.x > 0.0f || value.y > 0.0f || value.z > 0.0f;
    };

    // Cycles' one-sample-model contract: sampled-light visibility filters
    // BsdfEval accumulation only. Every eligible closure still contributes
    // its sample weight and directional PDF to the competing-technique PDF.
    const auto &physical_base_f = physical_light_actual[0u];
    const auto &physical_base_diffuse = physical_light_actual[1u];
    const auto &physical_base_glossy = physical_light_actual[2u];
    const auto &physical_exclude_diffuse_f = physical_light_actual[3u];
    const auto &physical_exclude_diffuse = physical_light_actual[4u];
    const auto &physical_exclude_diffuse_glossy =
        physical_light_actual[5u];
    const auto &physical_exclude_glossy_f = physical_light_actual[6u];
    const auto &physical_exclude_glossy_diffuse =
        physical_light_actual[7u];
    const auto &physical_exclude_glossy = physical_light_actual[8u];
    const auto &physical_exclude_both_f = physical_light_actual[9u];
    const auto &physical_exclude_both_diffuse =
        physical_light_actual[10u];
    const auto &physical_exclude_both_glossy =
        physical_light_actual[11u];
    const auto &physical_no_mis_f = physical_light_actual[12u];
    const auto &physical_no_mis_diffuse = physical_light_actual[13u];
    const auto &physical_no_mis_glossy = physical_light_actual[14u];
    const auto physical_pdf_invariant =
        physical_base_f.w > 0.0f &&
        approximately_equal(
            physical_exclude_diffuse_f.w, physical_base_f.w) &&
        approximately_equal(
            physical_exclude_glossy_f.w, physical_base_f.w) &&
        approximately_equal(
            physical_exclude_both_f.w, physical_base_f.w);
    const auto physical_contribution_filter =
        rgb_positive(physical_base_diffuse) &&
        rgb_positive(physical_base_glossy) &&
        rgb_equal(physical_exclude_diffuse_f,
            physical_base_glossy) &&
        rgb_zero(physical_exclude_diffuse) &&
        rgb_equal(physical_exclude_diffuse_glossy,
            physical_base_glossy) &&
        rgb_equal(physical_exclude_glossy_f,
            physical_base_diffuse) &&
        rgb_equal(physical_exclude_glossy_diffuse,
            physical_base_diffuse) &&
        rgb_zero(physical_exclude_glossy) &&
        rgb_zero(physical_exclude_both_f) &&
        rgb_zero(physical_exclude_both_diffuse) &&
        rgb_zero(physical_exclude_both_glossy);
    const auto physical_no_mis_contract =
        physical_no_mis_f.w == 0.0f &&
        rgb_equal(physical_no_mis_f, physical_base_f) &&
        rgb_equal(physical_no_mis_diffuse,
            physical_base_diffuse) &&
        rgb_equal(physical_no_mis_glossy,
            physical_base_glossy);
    if (!physical_pdf_invariant ||
        !physical_contribution_filter ||
        !physical_no_mis_contract) {
        std::cerr
            << "Cycles sampled-light closure filtering failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto &glass_base_f = glass_light_actual[0u];
    const auto &glass_base_glossy = glass_light_actual[2u];
    const auto &glass_exclude_glossy_f = glass_light_actual[3u];
    const auto &glass_exclude_glossy_glossy = glass_light_actual[5u];
    const auto &glass_exclude_transmit_f = glass_light_actual[6u];
    const auto &glass_exclude_transmit_glossy = glass_light_actual[8u];
    const auto &glass_exclude_both_f = glass_light_actual[9u];
    const auto &glass_exclude_both_glossy = glass_light_actual[11u];
    if (!(glass_base_f.w > 0.0f &&
          rgb_positive(glass_base_f) &&
          rgb_equal(glass_exclude_glossy_f, glass_base_f) &&
          rgb_equal(glass_exclude_glossy_glossy,
              glass_base_glossy) &&
          rgb_equal(glass_exclude_transmit_f, glass_base_f) &&
          rgb_equal(glass_exclude_transmit_glossy,
              glass_base_glossy) &&
          rgb_zero(glass_exclude_both_f) &&
          rgb_zero(glass_exclude_both_glossy) &&
          approximately_equal(
              glass_exclude_glossy_f.w, glass_base_f.w) &&
          approximately_equal(
              glass_exclude_transmit_f.w, glass_base_f.w) &&
          approximately_equal(
              glass_exclude_both_f.w, glass_base_f.w))) {
        std::cerr
            << "Cycles sampled-light glass classification failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }

    const auto &translucent_base_f = translucent_light_actual[0u];
    const auto &translucent_base_diffuse = translucent_light_actual[1u];
    const auto &translucent_exclude_diffuse_f =
        translucent_light_actual[3u];
    const auto &translucent_exclude_diffuse =
        translucent_light_actual[4u];
    const auto &translucent_exclude_transmit_f =
        translucent_light_actual[6u];
    const auto &translucent_exclude_transmit_diffuse =
        translucent_light_actual[7u];
    if (!(translucent_base_f.w > 0.0f &&
          rgb_positive(translucent_base_f) &&
          rgb_zero(translucent_exclude_diffuse_f) &&
          rgb_zero(translucent_exclude_diffuse) &&
          rgb_equal(translucent_exclude_transmit_f,
              translucent_base_f) &&
          rgb_equal(translucent_exclude_transmit_diffuse,
              translucent_base_diffuse) &&
          approximately_equal(
              translucent_exclude_diffuse_f.w,
              translucent_base_f.w) &&
          approximately_equal(
              translucent_exclude_transmit_f.w,
              translucent_base_f.w))) {
        std::cerr
            << "Cycles sampled-light translucent classification failed on "
            << backend << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

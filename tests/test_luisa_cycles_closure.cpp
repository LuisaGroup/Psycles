#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/graph_surface.h>

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

class ParameterShaderServices final : public ShaderServices {

private:
    const BufferFloat4 &_parameters;

public:
    explicit ParameterShaderServices(
        const BufferFloat4 &parameters) noexcept
        : _parameters{parameters} {
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

    [[nodiscard]] Float parameter_float(Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).x;
    }

    [[nodiscard]] Float3 parameter_float3(Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).xyz();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>) const noexcept override {
        // The structural regression only requires a finite,
        // non-extinguishing albedo estimate. Production services
        // provide the versioned Cycles table, which is covered by
        // scene-level differential tests.
        return 0.5f;
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

[[nodiscard]] std::vector<luisa::float4> parameter_data(
    const SurfaceProgram &program) {
    std::vector<luisa::float4> result;
    result.reserve(program.parameters().size());
    for (const auto &parameter : program.parameters()) {
        const auto &value = parameter.default_value.value;
        if (const auto *scalar = std::get_if<float>(&value)) {
            result.emplace_back(*scalar, 0.0f, 0.0f, 0.0f);
        } else if (const auto *vector = std::get_if<Vec3f>(&value)) {
            result.emplace_back(vector->x, vector->y, vector->z, 0.0f);
        } else {
            throw std::runtime_error{
                "closure fixture has an unsupported parameter type"};
        }
    }
    if (result.empty()) {
        result.emplace_back(0.0f);
    }
    return result;
}

[[nodiscard]] SurfacePoint make_surface_point() noexcept {
    return {.position = make_float3(0.0f),
        .object_position = make_float3(0.0f),
        .object_location = make_float3(0.0f),
        .generated = make_float3(0.5f),
        .geometric_normal = make_float3(0.0f, 0.0f, 1.0f),
        .shading_normal = make_float3(0.0f, 0.0f, 1.0f),
        .object_shading_normal = make_float3(0.0f, 0.0f, 1.0f),
        .object_tangent = make_float3(1.0f, 0.0f, 0.0f),
        .tangent_sign = 1.0f,
        .normal_to_world_x = make_float3(1.0f, 0.0f, 0.0f),
        .normal_to_world_y = make_float3(0.0f, 1.0f, 0.0f),
        .normal_to_world_z = make_float3(0.0f, 0.0f, 1.0f),
        .dpdu = make_float3(1.0f, 0.0f, 0.0f),
        .dpdv = make_float3(0.0f, 1.0f, 0.0f),
        .dPdx = make_float3(0.0f),
        .dPdy = make_float3(0.0f),
        .object_dPdx = make_float3(0.0f),
        .object_dPdy = make_float3(0.0f),
        .generated_dx = make_float3(0.0f),
        .generated_dy = make_float3(0.0f),
        .incoming = make_float3(0.0f, 0.0f, 1.0f),
        .uv = make_float2(0.0f),
        .uv_dx = make_float2(0.0f),
        .uv_dy = make_float2(0.0f),
        .geometry_index = 0u,
        .barycentric = make_float2(0.0f),
        .barycentric_dx = make_float2(0.0f),
        .barycentric_dy = make_float2(0.0f),
        .instance_id = 0u,
        .primitive_id = 0u,
        .parameter_block = 0u,
        .object_random = 0.0f,
        .particle_index = 0u,
        .random_per_island = 0.0f,
        .ray_visibility = 1u,
        .ray_events = 0u,
        .ray_depth = 0u,
        .diffuse_depth = 0u,
        .glossy_depth = 0u,
        .transparent_depth = 0u,
        .transmission_depth = 0u,
        .ray_length = 0.0f,
        .time = 0.0f,
        .back_facing = false};
}

[[nodiscard]] bool approximately_equal(
    float actual, float expected, float tolerance = 3.0e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(1.0f,
                   std::max(std::abs(actual), std::abs(expected)));
}

[[nodiscard]] bool approximately_equal(luisa::float4 actual,
    luisa::float4 expected,
    float tolerance = 3.0e-6f) noexcept {
    return approximately_equal(actual.x, expected.x, tolerance) &&
           approximately_equal(actual.y, expected.y, tolerance) &&
           approximately_equal(actual.z, expected.z, tolerance) &&
           approximately_equal(actual.w, expected.w, tolerance);
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

    SurfaceDispatch surfaces;
    const auto surface_tag =
        surfaces.create<GraphSurface>(program.program);
    const auto physical_surface_tag =
        surfaces.create<GraphSurface>(physical_program.program);
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

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output = device.create_buffer<luisa::float4>(11u);
    const auto physical_parameters =
        parameter_data(*physical_program.program);
    auto physical_parameter_buffer =
        device.create_buffer<luisa::float4>(physical_parameters.size());
    auto physical_output = device.create_buffer<luisa::float4>(8u);
    auto kernel = device.compile(evaluate);
    auto physical_kernel = device.compile(trace_physical_closures);
    std::array<luisa::float4, 11u> actual{};
    std::array<luisa::float4, 8u> physical_actual{};
    stream << physical_parameter_buffer.copy_from(
                  luisa::span{physical_parameters})
           << kernel(output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << physical_kernel(
                  physical_parameter_buffer, physical_output)
                  .dispatch(4u)
           << physical_output.copy_to(luisa::span{physical_actual})
           << synchronize();

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
            luisa::float4{0.25f, 0.25f, 0.25f, 1048.0f}) ||
        !approximately_equal(physical_actual[5u],
            luisa::float4{0.75f, 0.75f, 0.75f, 1048.0f}) ||
        !(physical_actual[6u].x > 0.0f &&
            physical_actual[6u].y > 0.0f &&
            physical_actual[6u].z > 0.0f)) {
        std::cerr << "Cycles physical closure weights failed on "
                  << backend << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

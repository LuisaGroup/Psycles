#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>
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

class VolumeShaderServices final : public ShaderServices {

private:
    const BufferFloat4 &_parameters;
    bool _density_found{};
    float _density{};
    bool _temperature_found{};
    float _temperature{};

public:
    VolumeShaderServices(
        const BufferFloat4 &parameters,
        bool density_found,
        float density,
        bool temperature_found,
        float temperature) noexcept
        : _parameters{parameters},
          _density_found{density_found},
          _density{density},
          _temperature_found{temperature_found},
          _temperature{temperature} {}

    [[nodiscard]] Float4 texture_2d(
        Expr<std::uint32_t>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        Expr<luisa::float2>,
        std::uint32_t,
        std::uint32_t) const noexcept override {
        return make_float4(0.0f);
    }

    [[nodiscard]] ShaderAttribute attribute(
        Expr<std::uint64_t> id,
        const SurfacePoint &) const noexcept override {
        auto result = ShaderAttribute::missing();
        if (_density_found) {
            const auto match =
                id == contract::attribute_id("density");
            result.value = select(
                result.value,
                make_float4(_density),
                match);
            result.found = result.found | match;
        }
        if (_temperature_found) {
            const auto match =
                id == contract::attribute_id("temperature");
            result.value = select(
                result.value,
                make_float4(_temperature),
                match);
            result.found = result.found | match;
        }
        return result;
    }

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).x;
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot) const noexcept override {
        return _parameters.read(block + slot).xyz();
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

    [[nodiscard]] Float3 nishita_sky(
        Expr<std::uint32_t>,
        std::uint32_t,
        Expr<luisa::float3>,
        Expr<float>,
        Expr<float>,
        Expr<float>,
        Expr<float>) const noexcept override {
        return make_float3(0.0f);
    }
};

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] ShaderGraph make_volume_graph() {
    ShaderGraph graph;
    const auto surface =
        graph.add_node(node_type::diffuse_bsdf, "Surface");
    require(
        graph.set_input(
            surface,
            "Color",
            SocketValue::color({0.0f, 0.0f, 0.0f})),
        "failed to configure surface");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{.node = surface, .socket = "Closure"});

    const auto absorption =
        graph.add_node(node_type::volume_absorption, "Absorption");
    require(
        graph.set_input(
            absorption,
            "Color",
            SocketValue::color({0.2f, 0.4f, 0.8f})) &&
            graph.set_input(
                absorption,
                "Density",
                SocketValue::floating(2.0f)),
        "failed to configure absorption volume");

    const auto scatter =
        graph.add_node(node_type::volume_scatter, "Scatter");
    require(
        graph.set_input(
            scatter,
            "Color",
            SocketValue::color({0.6f, -0.2f, 0.3f})) &&
            graph.set_input(
                scatter,
                "Density",
                SocketValue::floating(1.5f)) &&
            graph.set_property(
                scatter,
                "Phase",
                SocketValue::string("DRAINE")),
        "failed to configure scatter volume");

    const auto mixed =
        graph.add_node(node_type::mix_volume, "Mixed");
    require(
        graph.set_input(
            mixed,
            "Factor",
            SocketValue::floating(0.25f)) &&
            graph.connect(
                {.node = absorption, .socket = "Volume"},
                mixed,
                "A") &&
            graph.connect(
                {.node = scatter, .socket = "Volume"},
                mixed,
                "B"),
        "failed to configure mixed volume");

    const auto coefficients =
        graph.add_node(
            node_type::volume_coefficients,
            "Coefficients");
    require(
        graph.set_input(
            coefficients,
            "ScatterCoefficients",
            SocketValue::vector({0.1f, 0.2f, -0.3f})) &&
            graph.set_input(
                coefficients,
                "AbsorptionCoefficients",
                SocketValue::vector({0.05f, 0.1f, 0.2f})) &&
            graph.set_input(
                coefficients,
                "EmissionCoefficients",
                SocketValue::vector({1.0f, 2.0f, 3.0f})) &&
            graph.set_property(
                coefficients,
                "Phase",
                SocketValue::string("MIE")),
        "failed to configure coefficient volume");

    const auto principled =
        graph.add_node(node_type::principled_volume, "Principled");
    require(
        graph.set_input(
            principled,
            "Color",
            SocketValue::color({0.4f, 0.5f, 0.8f})) &&
            graph.set_input(
                principled,
                "Density",
                SocketValue::floating(0.5f)) &&
            graph.set_input(
                principled,
                "AbsorptionColor",
                SocketValue::color({0.25f, 0.36f, 0.81f})) &&
            graph.set_input(
                principled,
                "EmissionStrength",
                SocketValue::floating(2.0f)) &&
            graph.set_input(
                principled,
                "EmissionColor",
                SocketValue::color({0.2f, 0.3f, 0.4f})) &&
            graph.set_input(
                principled,
                "BlackbodyIntensity",
                SocketValue::floating(0.0f)),
        "failed to configure principled volume");

    const auto add_left =
        graph.add_node(node_type::add_volume, "AddLeft");
    const auto add_root =
        graph.add_node(node_type::add_volume, "AddRoot");
    require(
        graph.connect(
            {.node = mixed, .socket = "Volume"},
            add_left,
            "A") &&
            graph.connect(
                {.node = coefficients, .socket = "Volume"},
                add_left,
                "B") &&
            graph.connect(
                {.node = add_left, .socket = "Volume"},
                add_root,
                "A") &&
            graph.connect(
                {.node = principled, .socket = "Volume"},
                add_root,
                "B"),
        "failed to configure additive volume tree");
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{.node = add_root, .socket = "Volume"});
    return graph;
}

[[nodiscard]] std::vector<luisa::float4> parameter_data(
    const SurfaceProgram &program) {
    std::vector<luisa::float4> result;
    result.reserve(program.parameters().size());
    for (const auto &parameter : program.parameters()) {
        const auto &value = parameter.default_value;
        if (const auto *scalar =
                std::get_if<float>(&value.value)) {
            result.emplace_back(
                *scalar, 0.0f, 0.0f, 0.0f);
        } else if (
            const auto *vector =
                std::get_if<Vec3f>(&value.value)) {
            result.emplace_back(
                vector->x,
                vector->y,
                vector->z,
                0.0f);
        } else {
            throw std::runtime_error{
                "volume fixture contains an unsupported parameter type"};
        }
    }
    return result;
}

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 4.0e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

[[nodiscard]] bool approximately_equal(
    luisa::float4 actual,
    luisa::float4 expected) noexcept {
    return approximately_equal(actual.x, expected.x) &&
           approximately_equal(actual.y, expected.y) &&
           approximately_equal(actual.z, expected.z) &&
           approximately_equal(actual.w, expected.w);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};
    ShaderCompiler compiler{make_core_node_registry()};
    const auto shader = compiler.compile(make_volume_graph());
    if (!shader.ok()) {
        std::cerr << "failed to compile volume fixture graph\n";
        return EXIT_FAILURE;
    }
    const auto program =
        compile_surface_program(*shader.program);
    if (!program.ok()) {
        std::cerr << "failed to lower volume fixture program\n";
        return EXIT_FAILURE;
    }

    SurfaceDispatch surfaces;
    const auto surface_tag =
        surfaces.create<GraphSurface>(program.program);
    if (!surfaces.capabilities(surface_tag).may_have_volume) {
        std::cerr << "volume capability was not retained\n";
        return EXIT_FAILURE;
    }

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    const auto host_parameters =
        parameter_data(*program.program);
    auto parameters = device.create_buffer<luisa::float4>(
        host_parameters.size());
    auto output = device.create_buffer<luisa::float4>(9u);

    Kernel1D evaluate =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 records) noexcept {
            SurfacePoint point{};
            point.parameter_block = 0u;
            point.geometry_index = ~0u;
            const auto query = VolumeQuery{
                .object_density = 2.0f,
                .evaluate_emission = true};
            VolumeShaderServices attributes{
                parameter_buffer,
                true,
                0.5f,
                true,
                2.0f};
            const auto full =
                surfaces.volume_coefficients(
                    UInt{surface_tag},
                    attributes,
                    point,
                    query);
            records.write(
                0u,
                make_float4(
                    full.sigma_t,
                    select(
                        0.0f,
                        1.0f,
                        full.has_extinction)));
            records.write(
                1u,
                make_float4(
                    full.sigma_s,
                    select(
                        0.0f,
                        1.0f,
                        full.has_scatter)));
            records.write(
                2u,
                make_float4(
                    full.emission,
                    select(
                        0.0f,
                        1.0f,
                        full.has_emission)));

            const auto extinction_only =
                surfaces.volume_coefficients(
                    UInt{surface_tag},
                    attributes,
                    point,
                    VolumeQuery{
                        .object_density = 2.0f,
                        .evaluate_emission = false});
            records.write(
                3u,
                make_float4(
                    extinction_only.emission,
                    select(
                        0.0f,
                        1.0f,
                        extinction_only.has_emission)));

            VolumeShaderServices zero_density{
                parameter_buffer,
                true,
                0.0f,
                false,
                0.0f};
            const auto zero_attribute =
                surfaces.volume_coefficients(
                    UInt{surface_tag},
                    zero_density,
                    point,
                    query);
            records.write(
                4u,
                make_float4(
                    zero_attribute.sigma_t,
                    select(
                        0.0f,
                        1.0f,
                        zero_attribute.has_extinction)));
            records.write(
                5u,
                make_float4(
                    zero_attribute.sigma_s,
                    select(
                        0.0f,
                        1.0f,
                        zero_attribute.has_scatter)));

            VolumeShaderServices missing_density{
                parameter_buffer,
                false,
                0.0f,
                false,
                0.0f};
            const auto missing_attribute =
                surfaces.volume_coefficients(
                    UInt{surface_tag},
                    missing_density,
                    point,
                    query);
            records.write(
                6u,
                make_float4(
                    missing_attribute.sigma_t,
                    select(
                        0.0f,
                        1.0f,
                        missing_attribute.has_extinction)));
            records.write(
                7u,
                make_float4(
                    missing_attribute.sigma_s,
                    select(
                        0.0f,
                        1.0f,
                        missing_attribute.has_scatter)));
            records.write(
                8u,
                make_float4(
                    zero_attribute.emission,
                    select(
                        0.0f,
                        1.0f,
                        zero_attribute.has_emission)));
        };

    auto kernel = device.compile(evaluate);
    std::array<luisa::float4, 9u> actual{};
    stream << parameters.copy_from(
                  luisa::span{host_parameters})
           << kernel(parameters, output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    // Pinned to Cycles main 2bad74a8. These records exercise SVM's raw
    // absorption/scatter extinction, volume-only negative-weight clamping,
    // Add/Mix weighting, Principled sqrt(absorption color), implicit density
    // attribute semantics, object density scaling, and emission suppression.
    constexpr std::array expected{
        luisa::float4{3.5f, 2.6f, 1.035f, 1.0f},
        luisa::float4{0.85f, 0.65f, 0.625f, 1.0f},
        luisa::float4{2.8f, 5.2f, 7.6f, 1.0f},
        luisa::float4{0.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{3.15f, 2.25f, 0.625f, 1.0f},
        luisa::float4{0.65f, 0.4f, 0.225f, 1.0f},
        luisa::float4{3.85f, 2.95f, 1.445f, 1.0f},
        luisa::float4{1.05f, 0.9f, 1.025f, 1.0f},
        luisa::float4{2.8f, 5.2f, 7.6f, 1.0f}};
    for (std::size_t index = 0u;
         index < expected.size();
         ++index) {
        if (!approximately_equal(
                actual[index],
                expected[index])) {
            std::cerr
                << "Cycles volume coefficient fixture failed on "
                << backend << " at record " << index
                << ": got {" << actual[index].x
                << ", " << actual[index].y
                << ", " << actual[index].z
                << ", " << actual[index].w
                << "}\n";
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

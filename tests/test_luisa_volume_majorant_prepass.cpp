#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/compiler/surface_program.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/graph_surface.h>
#include <psycles/luisa/volume_majorant_prepass.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles;
using namespace psycles::compiler;
using namespace psycles::contract;
using namespace psycles::luisa_backend;

void expect(
    bool condition,
    const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] ShaderGraph
make_spatial_volume_graph() {
    ShaderGraph graph;
    const auto surface =
        graph.add_node(
            node_type::diffuse_bsdf,
            "Zero-contribution surface");
    expect(
        graph.set_input(
            surface,
            "Color",
            SocketValue::color(
                {0.0f, 0.0f, 0.0f})),
        "failed to construct companion surface root");
    graph.set_root(
        ShaderDomain::surface,
        OutputRef{
            .node = surface,
            .socket = "Closure"});

    const auto coordinates =
        graph.add_node(
            node_type::texture_coordinate,
            "Raw volume coordinates");
    const auto coefficients =
        graph.add_node(
            node_type::volume_coefficients,
            "Raw volume coefficients");
    expect(
        graph.set_input(
            coefficients,
            "ScatterCoefficients",
            SocketValue::vector(
                {0.0f, 0.0f, 0.0f})) &&
            graph.set_input(
                coefficients,
                "AbsorptionCoefficients",
                SocketValue::vector(
                    {0.0f, 0.0f, 0.0f})) &&
            graph.connect(
                {.node = coordinates,
                 .socket = "Generated"},
                coefficients,
                "EmissionCoefficients"),
        "failed to construct raw spatial volume graph");
    graph.set_root(
        ShaderDomain::volume,
        OutputRef{
            .node = coefficients,
            .socket = "Volume"});
    return graph;
}

[[nodiscard]] std::vector<luisa::float4>
parameter_data(const SurfaceProgram &program) {
    std::vector<luisa::float4> result;
    result.reserve(program.parameters().size());
    for (const auto &parameter :
         program.parameters()) {
        const auto &value =
            parameter.default_value;
        if (const auto *scalar =
                std::get_if<float>(
                    &value.value)) {
            result.emplace_back(
                *scalar,
                0.0f,
                0.0f,
                0.0f);
        } else if (
            const auto *vector =
                std::get_if<Vec3f>(
                    &value.value)) {
            result.emplace_back(
                vector->x,
                vector->y,
                vector->z,
                0.0f);
        } else {
            throw std::runtime_error{
                "volume-majorant fixture has an unsupported parameter type"};
        }
    }
    if (result.empty()) {
        result.emplace_back(
            0.0f, 0.0f, 0.0f, 0.0f);
    }
    return result;
}

class FixtureShaderServices final
    : public ShaderServices {

  private:
    const BufferFloat4 &_parameters;

  public:
    explicit FixtureShaderServices(
        const BufferFloat4 &parameters) noexcept
        : _parameters{parameters} {}

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
        Expr<std::uint64_t>,
        const SurfacePoint &) const noexcept override {
        return ShaderAttribute::missing();
    }

    [[nodiscard]] Float parameter_float(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot)
        const noexcept override {
        return _parameters
            .read(block + slot)
            .x;
    }

    [[nodiscard]] Float3 parameter_float3(
        Expr<std::uint32_t> block,
        Expr<std::uint32_t> slot)
        const noexcept override {
        return _parameters
            .read(block + slot)
            .xyz();
    }

    [[nodiscard]] Float cycles_bsdf_data(
        Expr<std::uint32_t>)
        const noexcept override {
        return 1.0f;
    }

    [[nodiscard]] Float3 xyz_to_rgb(
        Expr<luisa::float3> value)
        const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 rec709_to_rgb(
        Expr<luisa::float3> value)
        const noexcept override {
        return Float3{value};
    }

    [[nodiscard]] Float3 nishita_sky(
        Expr<std::uint32_t>,
        std::uint32_t,
        Expr<luisa::float3>,
        Expr<float>,
        Expr<float>,
        Expr<float>,
        Expr<float>)
        const noexcept override {
        return make_float3(0.0f);
    }
};

class FixturePointProvider final
    : public VolumeStackEntryPointProvider {

  public:
    VolumeStackEntryShading emit(
        const VolumeStackEntry &entry,
        const VolumeShadingState &state)
        const noexcept override {
        const auto state_valid =
            all(
                state.incoming ==
                make_float3(0.0f)) &
            (state.ray_visibility ==
             visibility_bit(
                 RayVisibility::camera)) &
            (state.ray_events == 0u) &
            (state.ray_depth == 0u) &
            (state.diffuse_depth == 0u) &
            (state.glossy_depth == 0u) &
            (state.transparent_depth == 0u) &
            (state.transmission_depth == 0u) &
            (state.ray_length == 0.0f) &
            (state.time == 0.5f);
        const auto object_position =
            state.position -
            make_float3(
                2.0f, -3.0f, 5.0f);
        const auto generated =
            select(
                object_position,
                make_float3(100.0f),
                !state_valid);

        SurfacePoint point{};
        point.position = state.position;
        point.object_position =
            object_position;
        point.generated = generated;
        point.geometric_normal =
            state.incoming;
        point.shading_normal =
            state.incoming;
        point.object_shading_normal =
            state.incoming;
        point.incoming = state.incoming;
        point.geometry_index =
            invalid_volume_identity;
        point.instance_id =
            entry.instance_id;
        point.primitive_id =
            invalid_volume_identity;
        point.parameter_block =
            entry.parameter_block;
        point.ray_visibility =
            state.ray_visibility;
        point.ray_events =
            state.ray_events;
        point.ray_depth =
            state.ray_depth;
        point.diffuse_depth =
            state.diffuse_depth;
        point.glossy_depth =
            state.glossy_depth;
        point.transparent_depth =
            state.transparent_depth;
        point.transmission_depth =
            state.transmission_depth;
        point.ray_length =
            state.ray_length;
        point.time = state.time;
        return {
            .point = std::move(point),
            // Cycles divides this factor back out of the baked extrema.
            .object_density = 2.0f};
    }
};

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 2.0e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance *
               std::max(
                   1.0f,
                   std::max(
                       std::abs(actual),
                       std::abs(expected)));
}

void run_fixture(
    std::string_view backend,
    const char *program) {
    auto graph =
        make_spatial_volume_graph();
    ShaderCompiler compiler{
        make_core_node_registry()};
    const auto shader =
        compiler.compile(graph);
    expect(
        shader.ok(),
        "failed to compile raw spatial volume graph");
    const auto lowered =
        compile_surface_program(*shader.program);
    expect(
        lowered.ok(),
        "failed to lower raw spatial volume graph");

    SurfaceDispatch surfaces;
    const auto surface_tag =
        surfaces.create<GraphSurface>(
            lowered.program);
    expect(
        surfaces
            .capabilities(surface_tag)
            .may_have_volume,
        "raw volume capability was lost");

    Context context{program};
    auto device =
        context.create_device(backend);
    auto stream = device.create_stream();
    const auto host_parameters =
        parameter_data(*lowered.program);
    auto parameters =
        device.create_buffer<luisa::float4>(
            host_parameters.size());
    auto samples =
        device.create_buffer<luisa::float4>(9u);
    auto extrema =
        device.create_buffer<luisa::float2>(3u);

    FixturePointProvider points;
    VolumeMajorantPrepass prepass{
        surfaces, points};
    Kernel1D evaluate =
        [&](BufferFloat4 parameter_buffer,
            BufferFloat4 sample_output,
            BufferVar<luisa::float2>
                extrema_output) noexcept {
            constexpr std::array<
                std::uint32_t, 9u>
                sample_indices{{
                    0u,
                    1u,
                    2u,
                    15u,
                    16u,
                    197520u,
                    197535u,
                    0xfffffff0u,
                    0xffffffffu}};
            constexpr std::array<
                std::uint32_t, 3u>
                cells{{
                    0u,
                    12345u,
                    2097151u}};
            Constant<std::uint32_t>
                sample_index_table{
                    sample_indices};
            Constant<std::uint32_t>
                cell_table{cells};
            const auto index = dispatch_x();
            $if(index <
                static_cast<std::uint32_t>(
                    sample_indices.size())) {
                const auto random =
                    cycles_sampler::
                        sobol_burley_sample_3d(
                            sample_index_table.read(
                                index),
                            0u,
                            0u,
                            0xffffffffu);
                sample_output.write(
                    index,
                    make_float4(
                        random, 0.0f));
            };
            $if(index <
                static_cast<std::uint32_t>(
                    cells.size())) {
                FixtureShaderServices services{
                    parameter_buffer};
                const VolumeStackEntry entry{
                    .object = 17u,
                    .shader = 23u,
                    .surface_tag =
                        surface_tag,
                    .parameter_block = 0u,
                    .instance_id = 5u,
                    .sample_method =
                        volume_sample_distance,
                    .valid = true};
                const VolumeMajorantGrid grid{
                    .minimum =
                        make_float3(0.0f),
                    .maximum =
                        make_float3(1.0f),
                    .object_to_world =
                        make_float4x4(
                            make_float4(
                                1.0f,
                                0.0f,
                                0.0f,
                                0.0f),
                            make_float4(
                                0.0f,
                                1.0f,
                                0.0f,
                                0.0f),
                            make_float4(
                                0.0f,
                                0.0f,
                                1.0f,
                                0.0f),
                            make_float4(
                                2.0f,
                                -3.0f,
                                5.0f,
                                1.0f)),
                    .resolution =
                        volume_majorant_grid_resolution};
                const auto value =
                    prepass.evaluate_cell(
                        entry,
                        services,
                        grid,
                        cell_table.read(index));
                extrema_output.write(
                    index,
                    make_float2(
                        value.minimum,
                        value.maximum));
            };
        };
    auto kernel =
        device.compile(
            evaluate,
            ShaderOption{
                .enable_cache = false,
                .enable_fast_math = false});
    std::array<luisa::float4, 9u>
        actual_samples{};
    std::array<luisa::float2, 3u>
        actual_extrema{};
    stream
        << parameters.copy_from(
               luisa::span{host_parameters})
        << kernel(
               parameters,
               samples,
               extrema)
               .dispatch(9u)
        << samples.copy_to(
               luisa::span{actual_samples})
        << extrema.copy_to(
               luisa::span{actual_extrema})
        << synchronize();

    constexpr std::array<
        std::array<std::uint32_t, 3u>, 9u>
        expected_sample_bits{{
            {0x3f505454u, 0x3ebbc2e7u, 0x3f7de3eau},
            {0x3d7e88f3u, 0x3f65bb61u, 0x3e8f2b81u},
            {0x3f202b30u, 0x3f12fa29u, 0x3e042c5bu},
            {0x3f31ab47u, 0x3ecbc20cu, 0x3f0aef01u},
            {0x3f3dff98u, 0x3f521c79u, 0x3ecde3a0u},
            {0x3f32f9dau, 0x3e98f0a4u, 0x3ce4c243u},
            {0x3eb0a99eu, 0x3df2adb5u, 0x3db0070bu},
            {0x3f201fb0u, 0x3f5fe037u, 0x3f4478f9u},
            {0x3edd6ee4u, 0x3ded72f7u, 0x3f0d578fu},
        }};
    for (auto index = std::size_t{0u};
         index < actual_samples.size();
         ++index) {
        const auto actual = actual_samples[index];
        const std::array actual_bits{
            std::bit_cast<std::uint32_t>(
                actual.x),
            std::bit_cast<std::uint32_t>(
                actual.y),
            std::bit_cast<std::uint32_t>(
                actual.z)};
        expect(
            actual_bits ==
                expected_sample_bits[index],
            "Cycles Sobol-Burley bits changed at fixture " +
                std::to_string(index) +
                " on " +
                std::string{backend});
    }

    constexpr std::array<luisa::float2, 3u>
        expected_extrema{{
            {0.0031634965f, 0.00928486325f},
            {0.749039471f, 0.759348571f},
            {0.995441556f, 1.00154698f},
        }};
    for (auto index = std::size_t{0u};
         index < actual_extrema.size();
         ++index) {
        const auto actual =
            actual_extrema[index];
        const auto expected =
            expected_extrema[index];
        expect(
            approximately_equal(
                actual.x, expected.x) &&
                approximately_equal(
                    actual.y, expected.y),
            "raw volume majorant extrema changed at fixture " +
                std::to_string(index) +
                " on " +
                std::string{backend});
    }
}

}// namespace

int main(int argc, char **argv) {
    try {
        const auto backend =
            std::string_view{
                argc > 1 ? argv[1] : "fallback"};
        run_fixture(
            backend, argv[0]);
        std::cout
            << "All current-Cycles raw volume majorant prepass "
               "fixtures passed on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Volume majorant prepass fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

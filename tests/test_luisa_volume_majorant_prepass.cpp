#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/shader_program.h>
#include <psycles/contract/scene.h>
#include <psycles/luisa/cycles_sampler.h>
#include "cycles_svm_volume_scene_fixture.h"
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
using namespace psycles::luisa_backend::detail;
namespace abi = psycles::compiler::cycles_svm;

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
    const auto point_to_vector =
        graph.add_node(
            node_type::point_to_vector,
            "Raw volume point to vector");
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
                point_to_vector,
                "Point") &&
            graph.connect(
                {.node = point_to_vector,
                 .socket = "Vector"},
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
    const char *program,
    bool fast_math) {
    auto graph =
        make_spatial_volume_graph();
    ShaderCompiler compiler{
        make_core_node_registry()};
    const auto shader =
        compiler.compile(graph);
    expect(
        shader.ok(),
        "failed to compile raw spatial volume graph");
    Context context{program};
    auto device =
        context.create_device(backend);
    auto stream = device.create_stream();
    auto scene = std::make_shared<LuisaSceneData>();
    scene->device = Device{device.impl_shared()};
    const std::array units{abi::ShaderTableCompileUnit{.shader_index = 23u, .shader = shader.program.get()}};
    std::vector<abi::KernelObject> objects(18u);
    objects[17].volume_density = 2.0f;
    objects[17].visibility = abi::PATH_RAY_VISIBILITY_ALL;
    objects[17].tfm = {{1, 0, 0, 2}, {0, 1, 0, -3}, {0, 0, 1, 5}};
    objects[17].itfm = {{1, 0, 0, -2}, {0, 1, 0, 3}, {0, 0, 1, -5}};
    test_support::initialize_volume_fixture(scene, stream, abi::compile_shader_table(units),
                                            objects, std::vector<unsigned>(18u));
    test_support::volume_generated_fixture(scene, stream,
        {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}});
    auto samples =
        device.create_buffer<luisa::float4>(9u);
    auto extrema =
        device.create_buffer<luisa::float2>(3u);

    Kernel1D evaluate =
        [&](BufferFloat4 sample_output,
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
                Var<RenderKernelParameters> parameters;
                parameters.camera_transform = parameters.camera_inverse_transform = make_float4x4(1.0f);
                const VolumeStackEntry entry{
                    .object = 17u,
                    .shader = 23u,
                    .surface_tag =
                        0u,
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
                    evaluate_cycles_svm_volume_density_cell(
                        scene, parameters, entry, grid, cell_table.read(index));
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
                .enable_fast_math = fast_math});
    std::array<luisa::float4, 9u>
        actual_samples{};
    std::array<luisa::float2, 3u>
        actual_extrema{};
    stream
        << kernel(
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
                std::string{backend} +
                " fast_math=" + std::to_string(fast_math));
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
                std::string{backend} +
                " fast_math=" + std::to_string(fast_math));
    }
}

}// namespace

int main(int argc, char **argv) {
    try {
        const auto backend =
            std::string_view{
                argc > 1 ? argv[1] : "fallback"};
        // Floating extrema retain the existing tolerance in both modes;
        // the integer Sobol/hash sequence must not change with fast math.
        for (const auto fast_math : {false, true}) {
            run_fixture(backend, argv[0], fast_math);
        }
        std::cout
            << "All current-Cycles raw volume majorant prepass "
               "fixtures passed with strict and fast math on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Volume majorant prepass fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

#include <psycles/luisa/volume_majorant_hierarchy.h>
#include <psycles/luisa/volume_majorant_traversal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;

void expect(
    bool condition,
    const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 2.0e-5f) noexcept {
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

[[nodiscard]] std::vector<VolumeMajorantExtrema>
x_partitioned_grid() {
    std::vector<VolumeMajorantExtrema> grid(
        VolumeMajorantHierarchyBuilder::
            required_extrema_count());
    constexpr auto resolution =
        volume_majorant_grid_resolution;
    for (auto z = 0u; z < resolution; ++z) {
        for (auto y = 0u; y < resolution; ++y) {
            for (auto x = 0u; x < resolution; ++x) {
                const auto index =
                    static_cast<std::size_t>(x) +
                    static_cast<std::size_t>(
                        resolution) *
                        (static_cast<std::size_t>(y) +
                         static_cast<std::size_t>(
                             resolution) *
                             static_cast<std::size_t>(z));
                const auto sigma =
                    x < resolution / 2u
                        ? 0.2f
                        : 0.8f;
                grid[index] = {
                    .minimum = sigma,
                    .maximum = sigma};
            }
        }
    }
    return grid;
}

[[nodiscard]] VolumeMajorantHierarchy
test_host_hierarchy() {
    const VolumeMajorantHierarchyBuilder builder;
    const VolumeMajorantBounds bounds{
        .minimum = {-1.0f, -1.0f, -1.0f},
        .maximum = {1.0f, 1.0f, 1.0f}};

    auto constant =
        std::vector<VolumeMajorantExtrema>(
            VolumeMajorantHierarchyBuilder::
                required_extrema_count(),
            VolumeMajorantExtrema{
                .minimum = 0.3f,
                .maximum = 0.3f});
    const auto constant_result =
        builder.build(bounds, constant);
    expect(
        constant_result.ok(),
        "constant majorant grid was rejected");
    expect(
        constant_result.hierarchy.nodes.size() == 1u,
        "constant majorant grid was subdivided");
    expect(
        constant_result.hierarchy.nodes[0u]
                .sigma_minimum ==
            0.3f &&
            constant_result.hierarchy.nodes[0u]
                    .sigma_maximum ==
                0.3f,
        "constant root extrema changed");

    auto partitioned = x_partitioned_grid();
    auto result =
        builder.build(bounds, partitioned);
    expect(
        result.ok(),
        "partitioned majorant grid was rejected");
    expect(
        result.hierarchy.nodes.size() == 9u,
        "one-level Cycles octree did not contain root plus eight children");
    const auto &root =
        result.hierarchy.nodes[0u];
    expect(
        root.parent == -1 &&
            root.first_child == 1 &&
            root.sigma_minimum == 0.2f &&
            root.sigma_maximum == 0.8f,
        "partitioned root topology or extrema changed");
    for (auto child = 0u; child < 8u; ++child) {
        const auto &node =
            result.hierarchy.nodes[1u + child];
        const auto expected =
            (child & 1u) == 0u
                ? 0.2f
                : 0.8f;
        expect(
            node.parent == 0 &&
                node.first_child == -1 &&
                node.sigma_minimum == expected &&
                node.sigma_maximum == expected,
            "partitioned child topology or extrema changed");
    }
    const auto &root_transform =
        result.hierarchy.root;
    expect(
        root_transform.scale.x == 0.5f &&
            root_transform.scale.y == 0.5f &&
            root_transform.scale.z == 0.5f &&
            root_transform.translation.x == 1.5f &&
            root_transform.translation.y == 1.5f &&
            root_transform.translation.z == 1.5f,
        "root [1,2) transform changed");

    const auto invalid_grid =
        builder.build(
            bounds,
            std::span<
                const VolumeMajorantExtrema>{
                partitioned.data(),
                partitioned.size() - 1u});
    expect(
        !invalid_grid.ok(),
        "short extrema grid was accepted");
    const auto invalid_bounds =
        builder.build(
            VolumeMajorantBounds{
                .minimum = {0.0f, 0.0f, 0.0f},
                .maximum = {0.0f, 1.0f, 1.0f}},
            partitioned);
    expect(
        !invalid_bounds.ok(),
        "degenerate majorant bounds were accepted");
    const auto original_extrema =
        partitioned[0u];
    partitioned[0u].maximum =
        partitioned[0u].minimum - 0.1f;
    expect(
        !builder.build(bounds, partitioned).ok(),
        "unordered majorant extrema were accepted");
    partitioned[0u].maximum =
        std::numeric_limits<float>::infinity();
    expect(
        !builder.build(bounds, partitioned).ok(),
        "non-finite majorant extrema were accepted");
    partitioned[0u] = original_extrema;
    expect(
        !builder
             .build(
                 bounds,
                 partitioned,
                 0.0f)
             .ok(),
        "zero majorant density scale was accepted");
    return std::move(result.hierarchy);
}

void test_device_traversal(
    std::string_view backend,
    const VolumeMajorantHierarchy &hierarchy,
    const char *program) {
    Context context{program};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto nodes =
        device.create_buffer<VolumeMajorantNodeGpu>(
            hierarchy.nodes.size());
    auto roots =
        device.create_buffer<VolumeMajorantRootGpu>(1u);
    auto output =
        device.create_buffer<luisa::float4>(9u);
    const std::array root_upload{
        hierarchy.root};

    Kernel1D trace =
        [](BufferVar<VolumeMajorantNodeGpu> node_buffer,
           BufferVar<VolumeMajorantRootGpu> root_buffer,
           BufferFloat4 records) noexcept {
            const auto flag =
                [](Bool value) noexcept {
                    return select(
                        0.0f, 1.0f, value);
                };
            const auto write =
                [&](UInt index,
                    const VolumeMajorantLeaf &leaf) noexcept {
                    records.write(
                        index,
                        make_float4(
                            leaf.minimum,
                            leaf.maximum,
                            leaf.sigma_minimum,
                            leaf.sigma_maximum));
                };
            const auto root =
                root_buffer.read(0u);

            VolumeMajorantTraversal forward{
                node_buffer,
                root,
                make_float3(
                    -0.75f, 0.0f, 0.0f),
                make_float3(
                    1.0f, 0.0f, 0.0f),
                0.0f,
                2.0f};
            const auto forward_0 =
                forward.current();
            const auto forward_advance_0 =
                forward.advance();
            const auto forward_1 =
                forward.current();
            const auto forward_advance_1 =
                forward.advance();
            const auto forward_2 =
                forward.current();
            const auto forward_advance_2 =
                forward.advance();
            write(0u, forward_0);
            write(1u, forward_1);
            write(2u, forward_2);
            records.write(
                3u,
                make_float4(
                    flag(forward_advance_0),
                    flag(forward_advance_1),
                    flag(forward_advance_2),
                    cast<float>(
                        forward_0.node)));

            VolumeMajorantTraversal reverse{
                node_buffer,
                root,
                make_float3(
                    0.75f, 0.0f, 0.0f),
                make_float3(
                    -1.0f, 0.0f, 0.0f),
                0.0f,
                2.0f};
            const auto reverse_0 =
                reverse.current();
            const auto reverse_advance_0 =
                reverse.advance();
            const auto reverse_1 =
                reverse.current();
            const auto reverse_advance_1 =
                reverse.advance();
            const auto reverse_2 =
                reverse.current();
            const auto reverse_advance_2 =
                reverse.advance();
            write(4u, reverse_0);
            write(5u, reverse_1);
            write(6u, reverse_2);
            records.write(
                7u,
                make_float4(
                    flag(reverse_advance_0),
                    flag(reverse_advance_1),
                    flag(reverse_advance_2),
                    cast<float>(
                        reverse_0.node)));

            VolumeMajorantTraversal outside{
                node_buffer,
                root,
                make_float3(
                    2.0f, 0.0f, 0.0f),
                make_float3(
                    1.0f, 0.0f, 0.0f),
                0.0f,
                1.0f};
            const auto outside_leaf =
                outside.current();
            records.write(
                8u,
                make_float4(
                    outside_leaf.minimum,
                    outside_leaf.maximum,
                    outside_leaf.sigma_minimum,
                    flag(outside_leaf.valid)));
        };
    auto shader = device.compile(
        trace,
        ShaderOption{
            .enable_cache = false,
            .enable_fast_math = false});
    std::array<luisa::float4, 9u> actual{};
    stream
        << nodes.copy_from(
               luisa::span{
                   hierarchy.nodes})
        << roots.copy_from(
               luisa::span{
                   root_upload})
        << shader(
               nodes,
               roots,
               output)
               .dispatch(1u)
        << output.copy_to(
               luisa::span{actual})
        << synchronize();

    const std::array<luisa::float4, 9u>
        expected{{
            {0.0f, 0.75f, 0.2f, 0.2f},
            {0.75f, 1.75f, 0.8f, 0.8f},
            {1.75f, 2.0f, 0.2f, 0.8f},
            {1.0f, 1.0f, 0.0f, 7.0f},
            {0.0f, 0.75f, 0.8f, 0.8f},
            {0.75f, 1.75f, 0.2f, 0.2f},
            {1.75f, 2.0f, 0.2f, 0.8f},
            {1.0f, 1.0f, 0.0f, 8.0f},
            {0.0f, 1.0f, 0.2f, 1.0f},
        }};
    for (auto index = std::size_t{0u};
         index < actual.size();
         ++index) {
        if (!approximately_equal(
                actual[index],
                expected[index])) {
            std::cerr
                << "majorant traversal actual["
                << index << "]=("
                << actual[index].x << ", "
                << actual[index].y << ", "
                << actual[index].z << ", "
                << actual[index].w << ")\n";
            throw std::runtime_error{
                "Cycles majorant traversal fixture " +
                std::to_string(index) +
                " changed on " +
                std::string{backend}};
        }
    }
}

}// namespace

int main(int argc, char **argv) {
    try {
        const auto backend =
            std::string_view{
                argc > 1 ? argv[1] : "fallback"};
        const auto hierarchy =
            test_host_hierarchy();
        test_device_traversal(
            backend,
            hierarchy,
            argv[0]);
        std::cout
            << "All current-Cycles volume majorant hierarchy "
               "fixtures passed on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Volume majorant fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

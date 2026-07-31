#include <psycles/luisa/volume_majorant_overlap.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;

inline constexpr std::uint32_t object_a = 10u;
inline constexpr std::uint32_t object_b = 20u;
inline constexpr std::uint32_t shader_a_index = 17u;
inline constexpr std::uint32_t shader_b_index = 23u;
inline constexpr std::uint32_t shader_a =
    shader_a_index | (1u << 30u);
inline constexpr std::uint32_t shader_b =
    shader_b_index | (1u << 31u);

void expect(
    bool condition,
    const std::string &message) {
    if (!condition) {
        throw std::runtime_error{message};
    }
}

[[nodiscard]] bool close(
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

[[nodiscard]] bool equal(
    luisa::uint4 actual,
    luisa::uint4 expected) noexcept {
    return actual.x == expected.x &&
           actual.y == expected.y &&
           actual.z == expected.z &&
           actual.w == expected.w;
}

class FixtureEntryProvider final
    : public VolumeMajorantEntryProvider {

  private:
    UInt *_entry_space_calls;
    Float *_last_shade_offset;

  public:
    explicit FixtureEntryProvider(
        UInt *entry_space_calls = nullptr,
        Float *last_shade_offset = nullptr) noexcept
        : _entry_space_calls{entry_space_calls},
          _last_shade_offset{
              last_shade_offset} {}

    VolumeMajorantEntrySpace entry_space(
        const VolumeStackEntry &entry,
        Float3 world_ray_origin,
        Float3 world_ray_direction)
        const noexcept override {
        if (_entry_space_calls != nullptr) {
            *_entry_space_calls += 1u;
        }
        return {
            .ray_origin =
                world_ray_origin,
            .ray_direction =
                world_ray_direction,
            .object_density =
                select(
                    1.0f,
                    2.0f,
                    entry.object == object_b)};
    }

    VolumeMajorantRuntimeExtrema extrema(
        const VolumeStackEntry &entry,
        const VolumeMajorantLeaf &leaf,
        Float object_density,
        Float shade_offset,
        Float3 world_ray_origin,
        Float3 world_ray_direction)
        const noexcept override {
        if (_last_shade_offset != nullptr) {
            *_last_shade_offset =
                shade_offset;
        }
        return VolumeMajorantEntryProvider::
            extrema(
                entry,
                leaf,
                object_density,
                shade_offset,
                world_ray_origin,
                world_ray_direction);
    }
};

[[nodiscard]] std::array<
    VolumeMajorantNodeGpu,
    11u>
fixture_nodes() {
    std::array<VolumeMajorantNodeGpu, 11u>
        nodes{};
    nodes[0u] = {
        .parent = -1,
        .first_child = 1,
        .sigma_minimum = 0.2f,
        .sigma_maximum = 0.8f};
    for (auto child = 0u; child < 8u;
         ++child) {
        const auto sigma =
            (child & 1u) == 0u
                ? 0.2f
                : 0.8f;
        nodes[1u + child] = {
            .parent = 0,
            .first_child = -1,
            .sigma_minimum = sigma,
            .sigma_maximum = sigma};
    }
    nodes[9u] = {
        .parent = -1,
        .first_child = -1,
        .sigma_minimum = 0.4f,
        .sigma_maximum = 0.4f};
    nodes[10u] = {
        .parent = -1,
        .first_child = -1,
        .sigma_minimum = 0.6f,
        .sigma_maximum = 0.6f};
    return nodes;
}

[[nodiscard]] constexpr std::array<
    VolumeMajorantRootGpu,
    4u>
fixture_roots() noexcept {
    return {
        VolumeMajorantRootGpu{
            .scale = {0.5f, 0.5f, 0.5f},
            .node = 0u,
            .translation =
                {1.5f, 1.5f, 1.5f},
            .shader = shader_a_index},
        VolumeMajorantRootGpu{
            .scale = {0.5f, 0.5f, 0.5f},
            // An older valid root with the same shader must not conceal a
            // malformed newer identity.
            .node = 10u,
            .translation =
                {1.5f, 1.5f, 1.5f},
            .shader = 88u},
        VolumeMajorantRootGpu{
            .scale = {0.5f, 0.5f, 0.5f},
            // Matching this newer root must fail before a node-buffer read.
            .node = 11u,
            .translation =
                {1.5f, 1.5f, 1.5f},
            .shader = 88u},
        VolumeMajorantRootGpu{
            .scale =
                {2.0f / 3.0f, 0.5f, 0.5f},
            .node = 9u,
            .translation =
                {5.0f / 3.0f, 1.5f, 1.5f},
            .shader = shader_b_index}};
}

[[nodiscard]] constexpr std::array<
    VolumeMajorantRootRangeGpu,
    3u>
fixture_ranges() noexcept {
    return {
        // The matching root precedes a nonmatching root. Device lookup must
        // scan backward exactly as Cycles does.
        VolumeMajorantRootRangeGpu{0u, 3u},
        // Deliberately outside the declared root domain.
        VolumeMajorantRootRangeGpu{5u, 1u},
        // Final World range.
        VolumeMajorantRootRangeGpu{3u, 1u}};
}

[[nodiscard]] VolumeStackEntry entry_a(
    UInt shader = shader_a,
    UInt instance_id = 0u) noexcept {
    return {
        .object = object_a,
        .shader = shader,
        .surface_tag = 101u,
        .parameter_block = 201u,
        .instance_id = instance_id,
        .sample_method =
            volume_sample_distance,
        .valid = true};
}

[[nodiscard]] VolumeStackEntry entry_b()
    noexcept {
    return {
        .object = object_b,
        .shader = shader_b,
        .surface_tag = 102u,
        .parameter_block = 202u,
        .instance_id =
            invalid_volume_identity,
        .sample_method =
            volume_sample_distance,
        .valid = true};
}

void run_backend(
    std::string_view backend,
    const char *program) {
    const auto host_nodes = fixture_nodes();
    constexpr auto host_roots =
        fixture_roots();
    constexpr auto host_ranges =
        fixture_ranges();

    Context context{program};
    auto device =
        context.create_device(backend);
    auto stream = device.create_stream();
    auto nodes =
        device.create_buffer<
            VolumeMajorantNodeGpu>(
            host_nodes.size());
    auto roots =
        device.create_buffer<
            VolumeMajorantRootGpu>(
            host_roots.size());
    auto ranges =
        device.create_buffer<
            VolumeMajorantRootRangeGpu>(
            host_ranges.size());
    auto intervals =
        device.create_buffer<luisa::float4>(10u);
    auto metadata =
        device.create_buffer<luisa::uint4>(10u);
    auto advances =
        device.create_buffer<luisa::uint4>(3u);
    auto shade_offsets =
        device.create_buffer<float>(3u);

    Kernel1D evaluate =
        [](BufferVar<VolumeMajorantNodeGpu>
               node_buffer,
           BufferVar<VolumeMajorantRootGpu>
               root_buffer,
           BufferVar<VolumeMajorantRootRangeGpu>
               range_buffer,
           BufferVar<luisa::float4>
               interval_output,
           BufferVar<luisa::uint4>
               metadata_output,
           BufferVar<luisa::uint4>
               advance_output,
           BufferVar<float>
               shade_offset_output) noexcept {
            FixtureEntryProvider provider;
            UInt rejected_provider_calls = 0u;
            FixtureEntryProvider rejected_provider{
                &rejected_provider_calls};
            const auto ray_origin =
                make_float3(
                    -0.75f, 0.0f, 0.0f);
            const auto ray_direction =
                make_float3(
                    1.0f, 0.0f, 0.0f);

            const auto write_segment =
                [&](std::uint32_t index,
                    const VolumeMajorantSegment
                        &segment) noexcept {
                    const auto flags =
                        select(
                            0u,
                            1u,
                            segment.valid) |
                        select(
                            0u,
                            2u,
                            segment.no_overlap) |
                        select(
                            0u,
                            4u,
                            segment
                                .lookup_complete);
                    interval_output.write(
                        index,
                        make_float4(
                            segment.minimum,
                            segment.maximum,
                            segment
                                .sigma_minimum,
                            segment
                                .sigma_maximum));
                    metadata_output.write(
                        index,
                        make_uint4(
                            segment.object,
                            segment.shader,
                            segment.node,
                            flags));
                };

            VolumeStack overlap_stack{3u};
            overlap_stack
                .initialize_background(
                    entry_a(), true);
            overlap_stack
                .initialize_background(
                    entry_b(), true);
            VolumeMajorantOverlapTraversal
                overlap{
                    node_buffer,
                    root_buffer,
                    range_buffer,
                    11u,
                    4u,
                    3u,
                    2u,
                    overlap_stack,
                    provider,
                    ray_origin,
                    ray_direction,
                    0.0f,
                    2.0f,
                    0.25f};
            write_segment(
                0u, overlap.current());
            const auto overlap_0 =
                overlap.advance(0.25f);
            write_segment(
                1u, overlap.current());
            const auto overlap_1 =
                overlap.advance(0.25f);
            write_segment(
                2u, overlap.current());
            const auto overlap_2 =
                overlap.advance(0.25f);
            write_segment(
                3u, overlap.current());
            const auto overlap_3 =
                overlap.advance(0.25f);
            advance_output.write(
                0u,
                make_uint4(
                    select(
                        0u, 1u, overlap_0),
                    select(
                        0u, 1u, overlap_1),
                    select(
                        0u, 1u, overlap_2),
                    select(
                        0u, 1u, overlap_3)));

            VolumeStack single_stack{2u};
            single_stack
                .initialize_background(
                    entry_a(), true);
            Float last_shade_offset = 0.0f;
            FixtureEntryProvider
                tracked_provider{
                    nullptr,
                    &last_shade_offset};
            VolumeMajorantOverlapTraversal
                single{
                    node_buffer,
                    root_buffer,
                    range_buffer,
                    11u,
                    4u,
                    3u,
                    2u,
                    single_stack,
                    tracked_provider,
                    ray_origin,
                    ray_direction,
                    0.0f,
                    2.0f,
                    0.25f};
            shade_offset_output.write(
                0u, last_shade_offset);
            write_segment(
                4u, single.current());
            const auto single_0 =
                single.advance(0.75f);
            shade_offset_output.write(
                1u, last_shade_offset);
            write_segment(
                5u, single.current());
            const auto single_1 =
                single.advance(0.125f);
            shade_offset_output.write(
                2u, last_shade_offset);
            write_segment(
                6u, single.current());
            const auto single_2 =
                single.advance(0.25f);
            advance_output.write(
                1u,
                make_uint4(
                    select(
                        0u, 1u, single_0),
                    select(
                        0u, 1u, single_1),
                    select(
                        0u, 1u, single_2),
                    0u));

            VolumeStack missing_stack{2u};
            missing_stack
                .initialize_background(
                    entry_a(99u, 0u),
                    true);
            VolumeMajorantOverlapTraversal
                missing{
                    node_buffer,
                    root_buffer,
                    range_buffer,
                    11u,
                    4u,
                    3u,
                    2u,
                    missing_stack,
                    rejected_provider,
                    ray_origin,
                    ray_direction,
                    0.0f,
                    2.0f,
                    0.25f};
            write_segment(
                7u, missing.current());
            const auto missing_advanced =
                missing.advance(0.25f);

            VolumeStack malformed_stack{2u};
            malformed_stack
                .initialize_background(
                    entry_a(shader_a, 1u),
                    true);
            VolumeMajorantOverlapTraversal
                malformed{
                    node_buffer,
                    root_buffer,
                    range_buffer,
                    11u,
                    4u,
                    3u,
                    2u,
                    malformed_stack,
                    rejected_provider,
                    ray_origin,
                    ray_direction,
                    0.0f,
                    2.0f,
                    0.25f};
            write_segment(
                8u, malformed.current());
            const auto malformed_advanced =
                malformed.advance(0.25f);

            VolumeStack invalid_node_stack{2u};
            invalid_node_stack
                .initialize_background(
                    entry_a(88u, 0u),
                    true);
            VolumeMajorantOverlapTraversal
                invalid_node{
                    node_buffer,
                    root_buffer,
                    range_buffer,
                    11u,
                    4u,
                    3u,
                    2u,
                    invalid_node_stack,
                    rejected_provider,
                    ray_origin,
                    ray_direction,
                    0.0f,
                    2.0f,
                    0.25f};
            write_segment(
                9u, invalid_node.current());
            const auto invalid_node_advanced =
                invalid_node.advance(0.25f);
            advance_output.write(
                2u,
                make_uint4(
                    select(
                        0u,
                        1u,
                        missing_advanced),
                    select(
                        0u,
                        1u,
                        malformed_advanced),
                    select(
                        0u,
                        1u,
                        invalid_node_advanced),
                    rejected_provider_calls));
        };
    auto shader = device.compile(evaluate);

    std::array<luisa::float4, 10u>
        actual_intervals{};
    std::array<luisa::uint4, 10u>
        actual_metadata{};
    std::array<luisa::uint4, 3u>
        actual_advances{};
    std::array<float, 3u>
        actual_shade_offsets{};
    stream
        << nodes.copy_from(
               luisa::span{host_nodes})
        << roots.copy_from(
               luisa::span{host_roots})
        << ranges.copy_from(
               luisa::span{host_ranges})
        << shader(
               nodes,
               roots,
               ranges,
               intervals,
               metadata,
               advances,
               shade_offsets)
               .dispatch(1u)
        << intervals.copy_to(
               luisa::span{
                   actual_intervals})
        << metadata.copy_to(
               luisa::span{
                   actual_metadata})
        << advances.copy_to(
               luisa::span{
                   actual_advances})
        << shade_offsets.copy_to(
               luisa::span{
                   actual_shade_offsets})
        << synchronize();

    constexpr std::array expected_intervals{
        luisa::float4{0.0f, 0.75f, 1.0f, 1.0f},
        luisa::float4{0.75f, 1.25f, 1.6f, 1.6f},
        luisa::float4{1.25f, 1.75f, 1.6f, 1.6f},
        luisa::float4{1.75f, 2.0f, 1.0f, 1.6f},
        luisa::float4{0.0f, 0.75f, 0.2f, 0.2f},
        luisa::float4{0.75f, 1.75f, 0.8f, 0.8f},
        luisa::float4{1.75f, 2.0f, 0.2f, 0.8f}};
    constexpr std::array expected_metadata{
        luisa::uint4{object_a, shader_a, 7u, 5u},
        luisa::uint4{object_b, shader_b, 9u, 5u},
        luisa::uint4{object_a, shader_a, 8u, 5u},
        luisa::uint4{object_b, shader_b, 9u, 5u},
        luisa::uint4{object_a, shader_a, 7u, 7u},
        luisa::uint4{object_a, shader_a, 8u, 7u},
        luisa::uint4{object_a, shader_a, 0u, 7u}};

    for (auto index = std::size_t{0u};
         index < expected_intervals.size();
         ++index) {
        const auto actual =
            actual_intervals[index];
        const auto expected =
            expected_intervals[index];
        expect(
            close(actual.x, expected.x) &&
                close(actual.y, expected.y) &&
                close(actual.z, expected.z) &&
                close(actual.w, expected.w),
            "overlap interval/extrema mismatch at " +
                std::to_string(index) +
                " on " + std::string{backend});
        expect(
            equal(
                actual_metadata[index],
                expected_metadata[index]),
            "overlap active identity/flags mismatch at " +
                std::to_string(index) +
                " on " + std::string{backend});
    }

    expect(
        equal(
            actual_advances[0u],
            luisa::uint4{
                1u, 1u, 1u, 0u}) &&
            equal(
                actual_advances[1u],
                luisa::uint4{
                    1u, 1u, 0u, 0u}),
        "overlap or single-root advance sequence "
        "changed on " +
            std::string{backend});
    expect(
        close(actual_shade_offsets[0u], 0.25f) &&
            close(actual_shade_offsets[1u], 0.75f) &&
            close(actual_shade_offsets[2u], 0.125f),
        "overlap traversal did not consume the "
        "per-advance Cycles shade offset on " +
            std::string{backend});
    for (auto index : {7u, 8u, 9u}) {
        expect(
            actual_metadata[index].w == 2u,
            "missing or malformed root coverage was "
            "not rejected on " +
                std::string{backend});
    }
    expect(
        equal(
            actual_advances[2u],
            luisa::uint4{
                0u, 0u, 0u, 0u}),
        "invalid root coverage advanced or evaluated "
        "its entry provider on " +
            std::string{backend});
}

}// namespace

int main(int argc, char **argv) {
    try {
        const auto backend =
            std::string_view{
                argc > 1
                    ? argv[1]
                    : "fallback"};
        run_backend(
            backend, argv[0]);
        std::cout
            << "All current-Cycles overlapping-volume "
               "majorant fixtures passed on "
            << backend << ".\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr
            << "Volume-majorant overlap fixture failure: "
            << error.what() << '\n';
        return EXIT_FAILURE;
    }
}

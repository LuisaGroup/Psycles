#include "path_kernel_volume_point.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;

inline constexpr std::size_t record_count = 26u;

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 3.0e-6f) noexcept {
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
    return
        approximately_equal(
            actual.x, expected.x) &&
        approximately_equal(
            actual.y, expected.y) &&
        approximately_equal(
            actual.z, expected.z) &&
        approximately_equal(
            actual.w, expected.w);
}

}// namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{
            argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device =
        context.create_device(backend);
    auto stream = device.create_stream();
    auto scene = std::make_shared<LuisaSceneData>();

    constexpr std::array positions{
        luisa::float3{-1.0f, -1.0f, 0.0f},
        luisa::float3{1.0f, -1.0f, 0.0f},
        luisa::float3{0.0f, 1.0f, 0.0f}};
    constexpr std::array triangles{
        Triangle{0u, 1u, 2u}};
    auto position_buffer =
        device.create_buffer<luisa::float3>(
            positions.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(
            triangles.size());
    auto mesh = device.create_mesh(
        position_buffer,
        triangle_buffer);

    const auto generated_transform =
        make_float4x4(
            luisa::make_float4(
                0.25f, 0.0f, 0.0f, 0.0f),
            luisa::make_float4(
                0.0f, 0.5f, 0.0f, 0.0f),
            luisa::make_float4(
                0.0f, 0.0f, 2.0f, 0.0f),
            luisa::make_float4(
                0.1f, 0.2f, -0.3f, 1.0f));
    const std::array geometries{
        GeometryGpu{
            .generated_transform =
                generated_transform}};
    constexpr std::array instances{
        InstanceGpu{
            .geometry_index = 0u,
            .object_random = 0.375f,
            .particle_index = 23u}};
    scene->geometry_buffer =
        device.create_buffer<GeometryGpu>(
            geometries.size());
    scene->instance_buffer =
        device.create_buffer<InstanceGpu>(
            instances.size());

    const auto object_to_world =
        make_float4x4(
            luisa::make_float4(
                -2.0f, 0.0f, 0.0f, 0.0f),
            luisa::make_float4(
                0.0f, 3.0f, 0.0f, 0.0f),
            luisa::make_float4(
                0.0f, 0.0f, 0.5f, 0.0f),
            luisa::make_float4(
                5.0f, -7.0f, 11.0f, 1.0f));
    scene->accel = device.create_accel();
    scene->accel.emplace_back(
        mesh,
        object_to_world,
        0xffu,
        false,
        0u);

    auto output =
        device.create_buffer<luisa::float4>(
            record_count);
    const auto point_provider =
        make_scene_volume_stack_entry_point_provider(
            scene);
    Kernel1D evaluate =
        [&](BufferFloat4 records) noexcept {
            const auto number =
                [](UInt value) noexcept {
                    return cast<float>(value);
                };
            const auto flag =
                [](Bool value) noexcept {
                    return select(
                        0.0f, 1.0f, value);
                };
            const auto zero_surface_fields =
                [&](const SurfacePoint &point) noexcept {
                    return
                        length_squared(
                            point.object_tangent) ==
                            0.0f &
                        point.tangent_sign == 0.0f &
                        length_squared(point.dpdu) ==
                            0.0f &
                        length_squared(point.dpdv) ==
                            0.0f &
                        length_squared(point.dPdx) ==
                            0.0f &
                        length_squared(point.dPdy) ==
                            0.0f &
                        length_squared(
                            point.object_dPdx) ==
                            0.0f &
                        length_squared(
                            point.object_dPdy) ==
                            0.0f &
                        length_squared(
                            point.generated_dx) ==
                            0.0f &
                        length_squared(
                            point.generated_dy) ==
                            0.0f &
                        dot(point.uv, point.uv) ==
                            0.0f &
                        dot(
                            point.barycentric,
                            point.barycentric) ==
                            0.0f &
                        point.random_per_island ==
                            0.0f;
                };

            const VolumeShadingState state{
                .position =
                    make_float3(
                        2.5f, -13.0f, 13.0f),
                .incoming =
                    normalize(
                        make_float3(
                            1.0f, 2.0f, 3.0f)),
                .ray_visibility = 37u,
                .ray_events = 41u,
                .ray_depth = 5u,
                .diffuse_depth = 2u,
                .glossy_depth = 3u,
                .transparent_depth = 7u,
                .transmission_depth = 11u,
                .ray_length = 0.0f,
                .time = 0.625f};
            const VolumeStackEntry object_entry{
                .object = 101u,
                .shader = 202u,
                .surface_tag = 303u,
                .parameter_block = 404u,
                .instance_id = 0u,
                .valid = true};
            const VolumeStackEntry world_entry{
                .object = 501u,
                .shader = 502u,
                .surface_tag = 503u,
                .parameter_block = 504u,
                .instance_id =
                    invalid_volume_identity,
                .valid = true};

            const auto object =
                point_provider->emit(
                    object_entry, state);
            records.write(
                0u,
                make_float4(
                    object.point.position,
                    object.object_density));
            records.write(
                1u,
                make_float4(
                    object.point.object_position,
                    object.point.object_random));
            records.write(
                2u,
                make_float4(
                    object.point.object_location,
                    number(
                        object.point.particle_index)));
            records.write(
                3u,
                make_float4(
                    object.point.generated,
                    number(
                        object.point.parameter_block)));
            records.write(
                4u,
                make_float4(
                    object.point.incoming,
                    flag(
                        length_squared(
                            object.point
                                .geometric_normal -
                            object.point.incoming) ==
                            0.0f &
                        length_squared(
                            object.point
                                .shading_normal -
                            object.point.incoming) ==
                            0.0f)));
            records.write(
                5u,
                make_float4(
                    object.point
                        .object_shading_normal,
                    0.0f));
            records.write(
                6u,
                make_float4(
                    object.point.normal_to_world_x,
                    0.0f));
            records.write(
                7u,
                make_float4(
                    object.point.normal_to_world_y,
                    0.0f));
            records.write(
                8u,
                make_float4(
                    object.point.normal_to_world_z,
                    0.0f));
            records.write(
                9u,
                make_float4(
                    flag(
                        object.point.geometry_index ==
                        invalid_volume_identity),
                    flag(
                        object.point.primitive_id ==
                        invalid_volume_identity),
                    flag(
                        object.point.instance_id ==
                        0u),
                    flag(!object.point.back_facing)));
            records.write(
                10u,
                make_float4(
                    object.point.uv,
                    object.point.ray_length,
                    object.point.time));
            records.write(
                11u,
                make_float4(
                    number(
                        object.point.ray_visibility),
                    number(
                        object.point.ray_events),
                    number(
                        object.point.ray_depth),
                    number(
                        object.point.diffuse_depth)));
            records.write(
                12u,
                make_float4(
                    number(
                        object.point.glossy_depth),
                    number(
                        object.point
                            .transparent_depth),
                    number(
                        object.point
                            .transmission_depth),
                    flag(
                        zero_surface_fields(
                            object.point))));

            const auto world =
                point_provider->emit(
                    world_entry, state);
            records.write(
                13u,
                make_float4(
                    world.point.position,
                    world.object_density));
            records.write(
                14u,
                make_float4(
                    world.point.object_position,
                    world.point.object_random));
            records.write(
                15u,
                make_float4(
                    world.point.object_location,
                    number(
                        world.point.particle_index)));
            records.write(
                16u,
                make_float4(
                    world.point.generated,
                    number(
                        world.point.parameter_block)));
            records.write(
                17u,
                make_float4(
                    world.point
                        .object_shading_normal,
                    flag(
                        length_squared(
                            world.point
                                .geometric_normal -
                            world.point.incoming) ==
                            0.0f &
                        length_squared(
                            world.point
                                .shading_normal -
                            world.point.incoming) ==
                            0.0f)));
            records.write(
                18u,
                make_float4(
                    world.point.normal_to_world_x,
                    0.0f));
            records.write(
                19u,
                make_float4(
                    world.point.normal_to_world_y,
                    0.0f));
            records.write(
                20u,
                make_float4(
                    world.point.normal_to_world_z,
                    0.0f));
            records.write(
                21u,
                make_float4(
                    flag(
                        world.point.geometry_index ==
                        invalid_volume_identity),
                    flag(
                        world.point.primitive_id ==
                        invalid_volume_identity),
                    flag(
                        world.point.instance_id ==
                        invalid_volume_identity),
                    flag(!world.point.back_facing)));
            records.write(
                22u,
                make_float4(
                    number(
                        world.point.ray_visibility),
                    number(
                        world.point.ray_events),
                    number(
                        world.point.ray_depth),
                    number(
                        world.point
                            .transmission_depth)));
            records.write(
                23u,
                make_float4(
                    world.point.uv,
                    world.point.time,
                    flag(
                        zero_surface_fields(
                            world.point))));

            const VolumeShadingState zero_direction_state{
                .position = state.position,
                .incoming =
                    make_float3(0.0f),
                .ray_visibility =
                    state.ray_visibility,
                .ray_events =
                    state.ray_events,
                .ray_depth =
                    state.ray_depth,
                .diffuse_depth =
                    state.diffuse_depth,
                .glossy_depth =
                    state.glossy_depth,
                .transparent_depth =
                    state.transparent_depth,
                .transmission_depth =
                    state.transmission_depth,
                .ray_length =
                    state.ray_length,
                .time = state.time};
            const auto zero_object =
                point_provider->emit(
                    object_entry,
                    zero_direction_state);
            records.write(
                24u,
                make_float4(
                    zero_object.point
                        .object_shading_normal,
                    flag(
                        all(
                            zero_object.point
                                .geometric_normal ==
                            make_float3(0.0f)) &
                        all(
                            zero_object.point
                                .shading_normal ==
                            make_float3(0.0f)) &
                        all(
                            zero_object.point
                                .incoming ==
                            make_float3(0.0f)))));
            const auto zero_world =
                point_provider->emit(
                    world_entry,
                    zero_direction_state);
            records.write(
                25u,
                make_float4(
                    zero_world.point
                        .object_shading_normal,
                    flag(
                        all(
                            zero_world.point
                                .geometric_normal ==
                            make_float3(0.0f)) &
                        all(
                            zero_world.point
                                .shading_normal ==
                            make_float3(0.0f)) &
                        all(
                            zero_world.point
                                .incoming ==
                            make_float3(0.0f)))));
        };
    auto shader = device.compile(evaluate);

    std::array<luisa::float4, record_count>
        actual{};
    stream
        << scene->geometry_buffer.copy_from(
               luisa::span{geometries})
        << scene->instance_buffer.copy_from(
               luisa::span{instances})
        << position_buffer.copy_from(
               luisa::span{positions})
        << triangle_buffer.copy_from(
               luisa::span{triangles})
        << mesh.build()
        << scene->accel.build()
        << shader(output).dispatch(1u)
        << output.copy_to(
               luisa::span{actual})
        << synchronize();

    const auto inverse_sqrt_14 =
        1.0f / std::sqrt(14.0f);
    const std::array expected{
        luisa::float4{
            2.5f, -13.0f, 13.0f, 1.0f},
        luisa::float4{
            1.25f, -2.0f, 4.0f, 0.375f},
        luisa::float4{
            5.0f, -7.0f, 11.0f, 23.0f},
        luisa::float4{
            0.4125f, -0.8f, 7.7f, 404.0f},
        luisa::float4{
            inverse_sqrt_14,
            2.0f * inverse_sqrt_14,
            3.0f * inverse_sqrt_14,
            1.0f},
        luisa::float4{
            -4.0f / 13.0f,
            12.0f / 13.0f,
            3.0f / 13.0f,
            0.0f},
        luisa::float4{
            -0.5f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0f, 1.0f / 3.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0f, 0.0f, 2.0f, 0.0f},
        luisa::float4{
            1.0f, 1.0f, 1.0f, 1.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 0.625f},
        luisa::float4{
            37.0f, 41.0f, 5.0f, 2.0f},
        luisa::float4{
            3.0f, 7.0f, 11.0f, 1.0f},
        luisa::float4{
            2.5f, -13.0f, 13.0f, 1.0f},
        luisa::float4{
            2.5f, -13.0f, 13.0f, 0.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            2.5f, -13.0f, 13.0f, 504.0f},
        luisa::float4{
            inverse_sqrt_14,
            2.0f * inverse_sqrt_14,
            3.0f * inverse_sqrt_14,
            1.0f},
        luisa::float4{
            1.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0f, 1.0f, 0.0f, 0.0f},
        luisa::float4{
            0.0f, 0.0f, 1.0f, 0.0f},
        luisa::float4{
            1.0f, 1.0f, 1.0f, 1.0f},
        luisa::float4{
            37.0f, 41.0f, 5.0f, 11.0f},
        luisa::float4{
            0.0f, 0.0f, 0.625f, 1.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 1.0f},
        luisa::float4{
            0.0f, 0.0f, 0.0f, 1.0f}};

    for (std::size_t index = 0u;
         index < expected.size();
         ++index) {
        if (!approximately_equal(
                actual[index], expected[index])) {
            std::cerr
                << "volume-point record " << index
                << " mismatch: got ("
                << actual[index].x << ", "
                << actual[index].y << ", "
                << actual[index].z << ", "
                << actual[index].w
                << "), expected ("
                << expected[index].x << ", "
                << expected[index].y << ", "
                << expected[index].z << ", "
                << expected[index].w << ")\n";
            return EXIT_FAILURE;
        }
    }

    std::cout
        << "Psycles Luisa " << backend
        << " volume-point regression passed\n";
    return EXIT_SUCCESS;
}

#include "cycles_shader_identity.h"
#include "path_kernel_volume_state.h"

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

inline constexpr std::size_t record_count = 13u;

[[nodiscard]] bool approximately_equal(
    float actual,
    float expected,
    float tolerance = 2.0e-6f) noexcept {
    return std::abs(actual - expected) <=
           tolerance;
}

[[nodiscard]] bool approximately_equal(
    luisa::float4 actual,
    luisa::float4 expected) noexcept {
    return approximately_equal(
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
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto scene = std::make_shared<LuisaSceneData>();

    constexpr std::array positions_0{
        luisa::float3{-1.0f, -1.0f, 1.0f},
        luisa::float3{1.0f, -1.0f, 1.0f},
        luisa::float3{0.0f, 1.0f, 1.0f}};
    constexpr std::array positions_1{
        luisa::float3{-1.0f, -1.0f, 0.5f},
        luisa::float3{1.0f, -1.0f, 0.5f},
        luisa::float3{0.0f, 1.0f, 0.5f}};
    constexpr std::array positions_2{
        // Front-facing entrance for the +Z probe.
        luisa::float3{-1.0f, -1.0f, 2.0f},
        luisa::float3{0.0f, 1.0f, 2.0f},
        luisa::float3{1.0f, -1.0f, 2.0f},
        // Back-facing exit of the same object.
        luisa::float3{-1.0f, -1.0f, 3.0f},
        luisa::float3{1.0f, -1.0f, 3.0f},
        luisa::float3{0.0f, 1.0f, 3.0f}};
    constexpr std::array positions_3{
        luisa::float3{-1.0f, -1.0f, 4.0f},
        luisa::float3{1.0f, -1.0f, 4.0f},
        luisa::float3{0.0f, 1.0f, 4.0f}};
    constexpr std::array triangles_0{
        Triangle{0u, 1u, 2u}};
    constexpr std::array triangles_1{
        Triangle{0u, 1u, 2u}};
    constexpr std::array triangles_2{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u}};
    constexpr std::array triangles_3{
        Triangle{0u, 1u, 2u}};
    constexpr std::array face_1{0u};
    constexpr std::array face_2{0u, 0u};

    auto position_buffer_0 =
        device.create_buffer<luisa::float3>(
            positions_0.size());
    auto position_buffer_1 =
        device.create_buffer<luisa::float3>(
            positions_1.size());
    auto position_buffer_2 =
        device.create_buffer<luisa::float3>(
            positions_2.size());
    auto position_buffer_3 =
        device.create_buffer<luisa::float3>(
            positions_3.size());
    auto triangle_buffer_0 =
        device.create_buffer<Triangle>(
            triangles_0.size());
    auto triangle_buffer_1 =
        device.create_buffer<Triangle>(
            triangles_1.size());
    auto triangle_buffer_2 =
        device.create_buffer<Triangle>(
            triangles_2.size());
    auto triangle_buffer_3 =
        device.create_buffer<Triangle>(
            triangles_3.size());
    auto material_buffer_0 =
        device.create_buffer<luisa::uint>(
            face_1.size());
    auto material_buffer_1 =
        device.create_buffer<luisa::uint>(
            face_1.size());
    auto material_buffer_2 =
        device.create_buffer<luisa::uint>(
            face_2.size());
    auto material_buffer_3 =
        device.create_buffer<luisa::uint>(
            face_1.size());
    auto smooth_buffer_0 =
        device.create_buffer<luisa::uint>(
            face_1.size());
    auto smooth_buffer_1 =
        device.create_buffer<luisa::uint>(
            face_1.size());
    auto smooth_buffer_2 =
        device.create_buffer<luisa::uint>(
            face_2.size());
    auto smooth_buffer_3 =
        device.create_buffer<luisa::uint>(
            face_1.size());
    auto mesh_0 = device.create_mesh(
        position_buffer_0,
        triangle_buffer_0);
    auto mesh_1 = device.create_mesh(
        position_buffer_1,
        triangle_buffer_1);
    auto mesh_2 = device.create_mesh(
        position_buffer_2,
        triangle_buffer_2);
    auto mesh_3 = device.create_mesh(
        position_buffer_3,
        triangle_buffer_3);

    constexpr std::array geometries{
        GeometryGpu{
            .bindless_base = 0u,
            .material_offset = 0u,
            .material_count = 1u},
        GeometryGpu{
            .bindless_base =
                geometry_bindless_stride,
            .material_offset = 1u,
            .material_count = 1u},
        GeometryGpu{
            .bindless_base =
                2u * geometry_bindless_stride,
            .material_offset = 2u,
            .material_count = 1u},
        GeometryGpu{
            .bindless_base =
                3u * geometry_bindless_stride,
            .material_offset = 3u,
            .material_count = 1u}};
    constexpr std::array materials{
        MaterialBindingGpu{
            .surface_tag = 10u,
            .parameter_block = 100u,
            .cycles_shader_index = 1u,
            .material_identity = 1u,
            .flags = material_flag_has_volume},
        MaterialBindingGpu{
            .surface_tag = 20u,
            .parameter_block = 200u,
            .cycles_shader_index = 2u,
            .material_identity = 2u,
            .flags = 0u},
        MaterialBindingGpu{
            .surface_tag = 30u,
            .parameter_block = 300u,
            .cycles_shader_index = 3u,
            .material_identity = 3u,
            .flags = material_flag_has_volume},
        MaterialBindingGpu{
            .surface_tag = 40u,
            .parameter_block = 400u,
            .cycles_shader_index = 4u,
            .material_identity = 4u,
            .flags = material_flag_has_volume}};
    constexpr std::array instances{
        InstanceGpu{
            .geometry_index = 0u,
            .cycles_object_index = 101u},
        InstanceGpu{
            .geometry_index = 1u,
            .cycles_object_index = 102u},
        InstanceGpu{
            .geometry_index = 2u,
            .cycles_object_index = 103u},
        InstanceGpu{
            .geometry_index = 3u,
            .cycles_object_index = 104u},
        InstanceGpu{
            .geometry_index = 0u,
            .cycles_object_index = 105u}};
    constexpr std::array inactive_override{
        MaterialBindingGpu{
            .cycles_shader_index =
                cycles_shader_identity::
                    invalid_index,
            .material_identity =
                cycles_shader_identity::
                    invalid_index}};

    scene->geometry_buffer =
        device.create_buffer<GeometryGpu>(
            geometries.size());
    scene->geometry_material_buffer =
        device.create_buffer<MaterialBindingGpu>(
            materials.size());
    scene->instance_buffer =
        device.create_buffer<InstanceGpu>(
            instances.size());
    scene->override_material_buffer =
        device.create_buffer<MaterialBindingGpu>(
            inactive_override.size());
    scene->heap =
        device.create_bindless_array(
            geometries.size() *
            geometry_bindless_stride);
    const auto bind_geometry =
        [&](std::uint32_t base,
            const auto &triangles,
            const auto &positions,
            const auto &material_slots,
            const auto &smooth) noexcept {
            scene->heap.emplace_on_update(
                base, triangles);
            scene->heap.emplace_on_update(
                base + 1u, positions);
            scene->heap.emplace_on_update(
                base + 4u, material_slots);
            scene->heap.emplace_on_update(
                base + 8u, smooth);
        };
    bind_geometry(
        0u,
        triangle_buffer_0,
        position_buffer_0,
        material_buffer_0,
        smooth_buffer_0);
    bind_geometry(
        geometry_bindless_stride,
        triangle_buffer_1,
        position_buffer_1,
        material_buffer_1,
        smooth_buffer_1);
    bind_geometry(
        2u * geometry_bindless_stride,
        triangle_buffer_2,
        position_buffer_2,
        material_buffer_2,
        smooth_buffer_2);
    bind_geometry(
        3u * geometry_bindless_stride,
        triangle_buffer_3,
        position_buffer_3,
        material_buffer_3,
        smooth_buffer_3);

    scene->accel = device.create_accel();
    scene->accel.emplace_back(
        mesh_0,
        make_float4x4(1.0f),
        0xffu,
        false,
        0u);
    scene->accel.emplace_back(
        mesh_1,
        make_float4x4(1.0f),
        0xffu,
        false,
        1u);
    scene->accel.emplace_back(
        mesh_2,
        make_float4x4(1.0f),
        0xffu,
        false,
        2u);
    scene->accel.emplace_back(
        mesh_3,
        make_float4x4(1.0f),
        0xffu,
        false,
        3u);
    const auto negative_z =
        make_float4x4(
            make_float4(
                1.0f, 0.0f, 0.0f, 0.0f),
            make_float4(
                0.0f, 1.0f, 0.0f, 0.0f),
            make_float4(
                0.0f, 0.0f, -1.0f, 0.0f),
            make_float4(
                0.0f, 0.0f, 0.0f, 1.0f));
    scene->accel.emplace_back(
        mesh_0,
        negative_z,
        0u,
        false,
        4u);
    scene->world_surface =
        MaterialBinding{
            .surface_tag = 50u,
            .parameter_block = 500u,
            .cycles_shader_index = 5u,
            .material_identity = 5u,
            .flags =
                material_flag_has_volume};
    scene->cycles_background_object_index =
        900u;

    auto output =
        device.create_buffer<luisa::float4>(
            record_count);
    const auto boundary =
        make_triangle_volume_boundary_component();
    const auto path_volume =
        make_path_volume_state_component();
    Kernel1D evaluate =
        [scene, boundary, path_volume](
            BufferFloat4 records) noexcept {
            const auto flag =
                [](Bool value) noexcept {
                    return select(
                        0.0f, 1.0f, value);
                };
            const auto number =
                [](UInt value) noexcept {
                    return cast<float>(value);
                };
            const auto shader_index =
                [&](UInt value) noexcept {
                    return number(
                        value &
                        cycles_shader_identity::
                            shader_mask);
                };
            auto volume =
                path_volume->initialize(
                    scene,
                    make_float3(0.0f),
                    camera_visibility,
                    6u,
                    true);
            auto &stack = *volume.stack;
            const auto &initialization =
                volume.camera_initialization;
            records.write(
                0u,
                make_float4(
                    number(stack.count()),
                    number(
                        initialization
                            .intersection_count),
                    number(
                        initialization
                            .enclosed_count),
                    static_cast<float>(
                        stack.maximum_entries())));
            const auto world =
                stack.entry(0u);
            records.write(
                1u,
                make_float4(
                    number(world.object),
                    shader_index(world.shader),
                    number(world.surface_tag),
                    number(
                        world.parameter_block)));
            const auto inside_a =
                stack.entry(1u);
            records.write(
                2u,
                make_float4(
                    number(inside_a.object),
                    shader_index(
                        inside_a.shader),
                    number(
                        inside_a.surface_tag),
                    number(
                        inside_a
                            .parameter_block)));
            records.write(
                3u,
                make_float4(
                    number(
                        inside_a.instance_id),
                    flag(inside_a.valid),
                    flag(
                        stack.entry(3u)
                            .valid),
                    flag(stack.empty())));
            const auto inside_d =
                stack.entry(2u);
            records.write(
                4u,
                make_float4(
                    number(inside_d.object),
                    shader_index(
                        inside_d.shader),
                    number(
                        inside_d.surface_tag),
                    number(
                        inside_d
                            .parameter_block)));
            records.write(
                5u,
                make_float4(
                    number(
                        inside_d.instance_id),
                    flag(inside_d.valid),
                    flag(
                        boundary->has_volume(
                            scene, 0u, 0u)),
                    flag(
                        boundary->has_volume(
                            scene, 1u, 0u))));

            const auto entrance =
                boundary->resolve(
                    scene,
                    2u,
                    0u,
                    make_float3(
                        camera_volume_probe_direction));
            const auto exit =
                boundary->resolve(
                    scene,
                    2u,
                    1u,
                    make_float3(
                        camera_volume_probe_direction));
            records.write(
                6u,
                make_float4(
                    entrance.geometric_normal,
                    flag(
                        entrance.back_facing)));
            records.write(
                7u,
                make_float4(
                    exit.geometric_normal,
                    flag(exit.back_facing)));
            const auto reflected =
                boundary->resolve(
                    scene,
                    4u,
                    0u,
                    make_float3(
                        camera_volume_probe_direction));
            records.write(
                8u,
                make_float4(
                    reflected.geometric_normal,
                    flag(
                        reflected.back_facing)));
            records.write(
                9u,
                make_float4(
                    camera_volume_probe_direction,
                    flag(world.valid)));
            auto background_only =
                path_volume->initialize(
                    scene,
                    make_float3(0.0f),
                    camera_visibility,
                    6u,
                    false);
            auto &background_stack =
                *background_only.stack;
            records.write(
                10u,
                make_float4(
                    number(
                        background_stack
                            .count()),
                    number(
                        background_only
                            .camera_initialization
                            .intersection_count),
                    number(
                        background_only
                            .camera_initialization
                            .enclosed_count),
                    static_cast<float>(
                        background_stack
                            .maximum_entries())));
            const auto background_world =
                background_stack.entry(0u);
            records.write(
                11u,
                make_float4(
                    number(
                        background_world
                            .object),
                    shader_index(
                        background_world
                            .shader),
                    number(
                        background_world
                            .surface_tag),
                    number(
                        background_world
                            .parameter_block)));
            records.write(
                12u,
                make_float4(
                    background_only.enabled()
                        ? 1.0f
                        : 0.0f,
                    flag(
                        background_stack
                            .entry(1u)
                            .valid),
                    number(stack.count()),
                    flag(
                        background_stack
                            .empty())));
        };
    auto shader = device.compile(evaluate);

    std::array<luisa::float4, record_count>
        actual{};
    stream
        << scene->geometry_buffer.copy_from(
               luisa::span{geometries})
        << scene->geometry_material_buffer
               .copy_from(
                   luisa::span{materials})
        << scene->instance_buffer.copy_from(
               luisa::span{instances})
        << scene->override_material_buffer
               .copy_from(
                   luisa::span{
                       inactive_override})
        << position_buffer_0.copy_from(
               luisa::span{positions_0})
        << position_buffer_1.copy_from(
               luisa::span{positions_1})
        << position_buffer_2.copy_from(
               luisa::span{positions_2})
        << position_buffer_3.copy_from(
               luisa::span{positions_3})
        << triangle_buffer_0.copy_from(
               luisa::span{triangles_0})
        << triangle_buffer_1.copy_from(
               luisa::span{triangles_1})
        << triangle_buffer_2.copy_from(
               luisa::span{triangles_2})
        << triangle_buffer_3.copy_from(
               luisa::span{triangles_3})
        << material_buffer_0.copy_from(
               luisa::span{face_1})
        << material_buffer_1.copy_from(
               luisa::span{face_1})
        << material_buffer_2.copy_from(
               luisa::span{face_2})
        << material_buffer_3.copy_from(
               luisa::span{face_1})
        << smooth_buffer_0.copy_from(
               luisa::span{face_1})
        << smooth_buffer_1.copy_from(
               luisa::span{face_1})
        << smooth_buffer_2.copy_from(
               luisa::span{face_2})
        << smooth_buffer_3.copy_from(
               luisa::span{face_1})
        << scene->heap.update()
        << mesh_0.build()
        << mesh_1.build()
        << mesh_2.build()
        << mesh_3.build()
        << scene->accel.build()
        << shader(output).dispatch(1u)
        << output.copy_to(luisa::span{actual})
        << synchronize();

    constexpr std::array expected{
        luisa::float4{3.0f, 4.0f, 1.0f, 5.0f},
        luisa::float4{900.0f, 5.0f, 50.0f, 500.0f},
        luisa::float4{101.0f, 1.0f, 10.0f, 100.0f},
        luisa::float4{0.0f, 1.0f, 0.0f, 0.0f},
        luisa::float4{104.0f, 4.0f, 40.0f, 400.0f},
        luisa::float4{3.0f, 1.0f, 1.0f, 0.0f},
        luisa::float4{0.0f, 0.0f, -1.0f, 0.0f},
        luisa::float4{0.0f, 0.0f, 1.0f, 1.0f},
        luisa::float4{0.0f, 0.0f, -1.0f, 0.0f},
        luisa::float4{0.0f, 0.0f, 1.0f, 1.0f},
        luisa::float4{1.0f, 0.0f, 0.0f, 5.0f},
        luisa::float4{900.0f, 5.0f, 50.0f, 500.0f},
        luisa::float4{1.0f, 0.0f, 3.0f, 0.0f}};
    auto failed = false;
    for (auto index = std::size_t{0u};
         index < expected.size();
         ++index) {
        if (!approximately_equal(
                actual[index],
                expected[index])) {
            failed = true;
            std::cerr
                << "camera volume stack failed on "
                << backend << " at record " << index
                << ": got {" << actual[index].x
                << ", " << actual[index].y
                << ", " << actual[index].z
                << ", " << actual[index].w
                << "}, expected {"
                << expected[index].x << ", "
                << expected[index].y << ", "
                << expected[index].z << ", "
                << expected[index].w << "}\n";
        }
    }
    return failed ? EXIT_FAILURE :
                    EXIT_SUCCESS;
}

#include "cycles_shader_identity.h"
#include "path_kernel_triangle_primitive.h"

#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend::detail;

inline constexpr std::size_t record_count = 14u;

[[nodiscard]] bool equal(
    luisa::float4 actual,
    luisa::float4 expected) noexcept {
    return actual.x == expected.x &&
           actual.y == expected.y &&
           actual.z == expected.z &&
           actual.w == expected.w;
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

    constexpr std::array geometries{
        GeometryGpu{
            .bindless_base = 0u,
            .material_offset = 0u,
            .material_count = 2u,
            .cycles_primitive_offset = 1200u}};
    constexpr std::array instances{
        InstanceGpu{
            .geometry_index = 0u,
            .override_offset = 0u,
            .override_count = 0u,
            .cycles_object_index =
                cycles_shader_identity::
                    invalid_index},
        InstanceGpu{
            .geometry_index = 0u,
            .override_offset = 0u,
            .override_count = 2u,
            .cycles_object_index = 77u}};
    constexpr std::array geometry_materials{
        MaterialBindingGpu{
            .surface_tag = 10u,
            .parameter_block = 100u,
            .cycles_shader_index = 1u,
            .material_identity = 1001u,
            .flags = 0u},
        MaterialBindingGpu{
            .surface_tag = 20u,
            .parameter_block = 200u,
            .cycles_shader_index = 2u,
            .material_identity = 1002u,
            .flags = material_flag_has_volume}};
    constexpr std::array override_materials{
        MaterialBindingGpu{
            .surface_tag = 30u,
            .parameter_block = 300u,
            .cycles_shader_index =
                cycles_shader_identity::
                    invalid_index,
            .material_identity = 3003u,
            .flags = material_flag_has_volume},
        MaterialBindingGpu{
            .surface_tag = 40u,
            .parameter_block = 400u,
            .cycles_shader_index = 4u,
            .material_identity = 4004u,
            .flags = 0u}};
    constexpr std::array triangles{
        Triangle{0u, 1u, 2u},
        Triangle{3u, 4u, 5u}};
    constexpr std::array material_slots{0u, 1u};
    constexpr std::array smooth{0u, 1u};

    scene->geometry_buffer =
        device.create_buffer<GeometryGpu>(
            geometries.size());
    scene->instance_buffer =
        device.create_buffer<InstanceGpu>(
            instances.size());
    scene->geometry_material_buffer =
        device.create_buffer<MaterialBindingGpu>(
            geometry_materials.size());
    scene->override_material_buffer =
        device.create_buffer<MaterialBindingGpu>(
            override_materials.size());
    auto triangle_buffer =
        device.create_buffer<Triangle>(
            triangles.size());
    auto material_slot_buffer =
        device.create_buffer<luisa::uint>(
            material_slots.size());
    auto smooth_buffer =
        device.create_buffer<luisa::uint>(
            smooth.size());
    scene->heap =
        device.create_bindless_array(
            geometry_bindless_stride);
    scene->heap.emplace_on_update(
        0u, triangle_buffer);
    scene->heap.emplace_on_update(
        4u, material_slot_buffer);
    scene->heap.emplace_on_update(
        8u, smooth_buffer);

    auto output =
        device.create_buffer<luisa::float4>(
            record_count);
    const auto primitive =
        make_triangle_primitive_component();
    Kernel1D evaluate =
        [scene, primitive](
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
            const auto shader_flag =
                [&](UInt value,
                    std::uint32_t bit) noexcept {
                    return flag(
                        (value & bit) != 0u);
                };

            const auto base_0 =
                primitive->emit(
                    scene, 0u, 0u);
            records.write(
                0u,
                make_float4(
                    number(
                        base_0.material_binding
                            .surface_tag),
                    number(
                        base_0.material_binding
                            .parameter_block),
                    number(
                        base_0.material_binding
                            .cycles_shader_index),
                    number(
                        base_0.material_binding
                            .flags)));
            records.write(
                1u,
                make_float4(
                    shader_index(
                        base_0
                            .cycles_surface_shader),
                    shader_flag(
                        base_0
                            .cycles_surface_shader,
                        cycles_shader_identity::
                            cast_shadow),
                    shader_flag(
                        base_0
                            .cycles_surface_shader,
                        cycles_shader_identity::
                            smooth_normal),
                    number(
                        base_0
                            .cycles_object_index)));
            records.write(
                2u,
                make_float4(
                    number(base_0.triangle.i0),
                    number(base_0.triangle.i1),
                    number(base_0.triangle.i2),
                    number(
                        base_0.material_slot)));

            const auto base_1 =
                primitive->emit(
                    scene, 0u, 1u);
            records.write(
                3u,
                make_float4(
                    number(
                        base_1.material_binding
                            .surface_tag),
                    number(
                        base_1.material_binding
                            .parameter_block),
                    shader_index(
                        base_1
                            .cycles_surface_shader),
                    number(
                        base_1.material_binding
                            .flags)));
            records.write(
                4u,
                make_float4(
                    shader_flag(
                        base_1
                            .cycles_surface_shader,
                        cycles_shader_identity::
                            cast_shadow),
                    shader_flag(
                        base_1
                            .cycles_surface_shader,
                        cycles_shader_identity::
                            smooth_normal),
                    number(
                        base_1
                            .cycles_object_index),
                    flag(base_1.has_volume)));
            const auto base_entry =
                base_1.volume_stack_entry();
            records.write(
                5u,
                make_float4(
                    number(base_entry.object),
                    shader_index(
                        base_entry.shader),
                    number(base_entry.surface_tag),
                    number(
                        base_entry.parameter_block)));
            records.write(
                6u,
                make_float4(
                    number(
                        base_entry.instance_id),
                    flag(base_entry.valid),
                    number(
                        base_1.material_binding
                            .material_identity),
                    number(
                        base_1.material_slot)));

            const auto override_0 =
                primitive->emit(
                    scene, 1u, 0u);
            records.write(
                7u,
                make_float4(
                    number(
                        override_0
                            .material_binding
                            .surface_tag),
                    number(
                        override_0
                            .material_binding
                            .parameter_block),
                    number(
                        override_0
                            .material_binding
                            .material_identity),
                    flag(
                        override_0
                            .has_volume)));
            records.write(
                8u,
                make_float4(
                    shader_index(
                        override_0
                            .cycles_surface_shader),
                    shader_flag(
                        override_0
                            .cycles_surface_shader,
                        cycles_shader_identity::
                            cast_shadow),
                    shader_flag(
                        override_0
                            .cycles_surface_shader,
                        cycles_shader_identity::
                            smooth_normal),
                    number(
                        override_0
                            .cycles_object_index)));
            const auto override_entry =
                override_0.volume_stack_entry();
            records.write(
                9u,
                make_float4(
                    number(
                        override_entry.object),
                    shader_index(
                        override_entry.shader),
                    number(
                        override_entry.surface_tag),
                    number(
                        override_entry
                            .parameter_block)));
            records.write(
                10u,
                make_float4(
                    number(
                        override_entry
                            .instance_id),
                    flag(override_entry.valid),
                    number(
                        override_0
                            .material_slot),
                    number(
                        override_0
                            .cycles_primitive_index)));

            const auto override_1 =
                primitive->emit(
                    scene, 1u, 1u);
            records.write(
                11u,
                make_float4(
                    number(
                        override_1
                            .material_binding
                            .surface_tag),
                    number(
                        override_1
                            .material_binding
                            .parameter_block),
                    shader_index(
                        override_1
                            .cycles_surface_shader),
                    flag(
                        override_1
                            .has_volume)));
            records.write(
                12u,
                make_float4(
                    shader_flag(
                        override_1
                            .cycles_surface_shader,
                        cycles_shader_identity::
                            cast_shadow),
                    shader_flag(
                        override_1
                            .cycles_surface_shader,
                        cycles_shader_identity::
                            smooth_normal),
                    number(
                        override_1
                            .cycles_object_index),
                    number(
                        override_1
                            .material_slot)));
            const auto non_volume_entry =
                override_1.volume_stack_entry();
            records.write(
                13u,
                make_float4(
                    flag(non_volume_entry.valid),
                    number(
                        non_volume_entry
                            .instance_id),
                    number(
                        override_1
                            .material_binding
                            .flags),
                    number(
                        override_1
                            .cycles_primitive_index)));
        };
    auto shader = device.compile(evaluate);

    std::array<luisa::float4, record_count>
        actual{};
    stream
        << scene->geometry_buffer.copy_from(
               luisa::span{geometries})
        << scene->instance_buffer.copy_from(
               luisa::span{instances})
        << scene->geometry_material_buffer
               .copy_from(
                   luisa::span{
                       geometry_materials})
        << scene->override_material_buffer
               .copy_from(
                   luisa::span{
                       override_materials})
        << triangle_buffer.copy_from(
               luisa::span{triangles})
        << material_slot_buffer.copy_from(
               luisa::span{material_slots})
        << smooth_buffer.copy_from(
               luisa::span{smooth})
        << scene->heap.update()
        << shader(output).dispatch(1u)
        << output.copy_to(luisa::span{actual})
        << synchronize();

    constexpr std::array expected{
        luisa::float4{10.0f, 100.0f, 1.0f, 0.0f},
        luisa::float4{1.0f, 1.0f, 0.0f, 0.0f},
        luisa::float4{0.0f, 1.0f, 2.0f, 0.0f},
        luisa::float4{20.0f, 200.0f, 2.0f, 1.0f},
        luisa::float4{1.0f, 1.0f, 0.0f, 1.0f},
        luisa::float4{0.0f, 2.0f, 20.0f, 200.0f},
        luisa::float4{0.0f, 1.0f, 1002.0f, 1.0f},
        luisa::float4{30.0f, 300.0f, 3003.0f, 1.0f},
        luisa::float4{3003.0f, 1.0f, 0.0f, 77.0f},
        luisa::float4{77.0f, 3003.0f, 30.0f, 300.0f},
        luisa::float4{1.0f, 1.0f, 0.0f, 1200.0f},
        luisa::float4{40.0f, 400.0f, 4.0f, 0.0f},
        luisa::float4{1.0f, 1.0f, 77.0f, 1.0f},
        luisa::float4{0.0f, 1.0f, 0.0f, 1201.0f}};
    for (auto index = std::size_t{0u};
         index < expected.size();
         ++index) {
        if (!equal(
                actual[index],
                expected[index])) {
            std::cerr
                << "triangle primitive resolver failed on "
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
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}

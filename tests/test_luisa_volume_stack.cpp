#include <psycles/luisa/volume_stack.h>

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

inline constexpr std::size_t record_count = 23u;

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
    auto output =
        device.create_buffer<luisa::float4>(
            record_count);

    Kernel1D evaluate =
        [](BufferFloat4 records) noexcept {
            const auto make_entry =
                [](std::uint32_t object,
                   std::uint32_t shader,
                   std::uint32_t surface_tag,
                   std::uint32_t parameter_block,
                   std::uint32_t instance_id) noexcept {
                    return VolumeStackEntry{
                        .object = object,
                        .shader = shader,
                        .surface_tag = surface_tag,
                        .parameter_block =
                            parameter_block,
                        .instance_id = instance_id,
                        .valid = true};
                };
            const auto flag =
                [](Bool value) noexcept {
                    return select(
                        0.0f, 1.0f, value);
                };
            const auto as_float =
                [](UInt value) noexcept {
                    return cast<float>(value);
                };

            const auto background =
                make_entry(900u, 90u, 190u, 290u, 390u);
            const auto a =
                make_entry(1u, 11u, 111u, 211u, 311u);
            const auto b =
                make_entry(2u, 22u, 122u, 222u, 322u);
            const auto c =
                make_entry(3u, 33u, 133u, 233u, 333u);

            VolumeStack stack{4u};
            records.write(
                0u,
                make_float4(
                    as_float(stack.count()),
                    flag(stack.empty()),
                    flag(stack.entry(0u).valid),
                    static_cast<float>(
                        stack.maximum_entries())));
            stack.initialize_background(
                background, true);
            records.write(
                1u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).object),
                    as_float(stack.entry(0u).shader),
                    as_float(
                        stack.entry(0u).surface_tag)));
            stack.cross_boundary(
                a, false, true, true);
            records.write(
                2u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).object),
                    as_float(stack.entry(1u).object),
                    as_float(stack.entry(1u).shader)));
            stack.cross_boundary(
                b, false, true, true);
            records.write(
                3u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(1u).object),
                    as_float(stack.entry(2u).object),
                    as_float(stack.entry(2u).shader)));
            stack.cross_boundary(
                c, false, true, true);
            records.write(
                4u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).object),
                    as_float(stack.entry(1u).object),
                    as_float(stack.entry(2u).object)));
            stack.cross_boundary(
                a, false, true, true);
            records.write(
                5u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).object),
                    as_float(stack.entry(1u).object),
                    as_float(stack.entry(2u).object)));
            stack.cross_boundary(
                a, true, true, true);
            records.write(
                6u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).object),
                    as_float(stack.entry(1u).object),
                    as_float(stack.entry(1u).shader)));
            stack.cross_boundary(
                c, true, true, true);
            records.write(
                7u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).object),
                    as_float(stack.entry(1u).object),
                    flag(stack.entry(2u).valid)));
            stack.cross_boundary(
                a, false, false, true);
            stack.cross_boundary(
                a, false, true, false);
            stack.cross_boundary(
                b, true, true, false);
            records.write(
                8u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).object),
                    as_float(stack.entry(1u).object),
                    as_float(stack.entry(1u).shader)));
            stack.cross_boundary(
                b, true, true, true);
            records.write(
                9u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).object),
                    flag(stack.entry(1u).valid),
                    flag(stack.empty())));

            stack.clear();
            const auto d =
                make_entry(5u, 50u, 150u, 250u, 350u);
            const auto e =
                make_entry(5u, 51u, 151u, 251u, 351u);
            const auto f =
                make_entry(6u, 60u, 160u, 260u, 360u);
            stack.cross_boundary(
                d, false, true, true);
            stack.cross_boundary(
                e, false, true, true);
            records.write(
                10u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).shader),
                    as_float(stack.entry(1u).shader),
                    flag(stack.entry(2u).valid)));
            stack.cross_boundary(
                d, true, true, true);
            records.write(
                11u,
                make_float4(
                    as_float(stack.count()),
                    as_float(stack.entry(0u).object),
                    as_float(stack.entry(0u).shader),
                    flag(stack.entry(1u).valid)));
            stack.cross_boundary(
                f, false, true, true);
            VolumeStack shadow{2u};
            shadow.copy_from(stack);
            records.write(
                12u,
                make_float4(
                    as_float(shadow.count()),
                    as_float(shadow.entry(0u).object),
                    as_float(shadow.entry(0u).shader),
                    as_float(
                        shadow.entry(0u).surface_tag)));
            records.write(
                13u,
                make_float4(
                    as_float(
                        shadow.entry(0u)
                            .parameter_block),
                    as_float(
                        shadow.entry(0u).instance_id),
                    flag(shadow.entry(0u).valid),
                    flag(shadow.entry(1u).valid)));

            VolumeStack camera_stack{6u};
            camera_stack.initialize_background(
                background, true);
            VolumeStackCameraInitializer camera{
                camera_stack, 6u};
            const auto camera_a =
                make_entry(10u, 10u, 110u, 210u, 310u);
            const auto camera_b =
                make_entry(20u, 20u, 120u, 220u, 320u);
            const auto camera_b_other_shader =
                make_entry(20u, 21u, 121u, 221u, 321u);
            const auto camera_c =
                make_entry(30u, 30u, 130u, 230u, 330u);
            const auto camera_d =
                make_entry(40u, 40u, 140u, 240u, 340u);
            const auto camera_ignored =
                make_entry(50u, 50u, 150u, 250u, 350u);
            camera.observe(
                camera_a, false, true);
            camera.observe(
                camera_a, true, true);
            camera.observe(
                camera_b, true, true);
            camera.observe(
                camera_b_other_shader, true, true);
            camera.observe(
                camera_c, false, true);
            camera.observe(
                camera_c, false, true);
            camera.observe(
                camera_c, true, true);
            camera.observe(
                camera_d, true, true);
            camera.observe(
                camera_ignored, true, false);
            records.write(
                14u,
                make_float4(
                    as_float(camera_stack.count()),
                    as_float(
                        camera_stack.entry(0u).object),
                    as_float(
                        camera_stack.entry(1u).object),
                    as_float(
                        camera_stack.entry(2u).object)));
            records.write(
                15u,
                make_float4(
                    as_float(
                        camera_stack.entry(1u).shader),
                    as_float(
                        camera_stack.entry(1u)
                            .surface_tag),
                    as_float(
                        camera_stack.entry(1u)
                            .parameter_block),
                    as_float(
                        camera_stack.entry(1u)
                            .instance_id)));
            records.write(
                16u,
                make_float4(
                    as_float(
                        camera_stack.entry(2u).shader),
                    as_float(
                        camera_stack.entry(2u)
                            .surface_tag),
                    as_float(
                        camera_stack.entry(2u)
                            .parameter_block),
                    as_float(
                        camera_stack.entry(2u)
                            .instance_id)));
            records.write(
                17u,
                make_float4(
                    as_float(camera.enclosed_count()),
                    flag(camera.can_continue(0u)),
                    flag(camera.can_continue(11u)),
                    flag(camera.can_continue(12u))));
            records.write(
                18u,
                make_float4(
                    camera_volume_probe_direction,
                    static_cast<float>(
                        maximum_volume_stack_size)));

            VolumeStack camera_copy{6u};
            camera_copy.copy_from(camera_stack);
            records.write(
                19u,
                make_float4(
                    as_float(camera_copy.count()),
                    as_float(
                        camera_copy.entry(0u).object),
                    as_float(
                        camera_copy.entry(1u).object),
                    as_float(
                        camera_copy.entry(2u).object)));
            auto invalid = camera_d;
            invalid.valid = false;
            camera_copy.cross_boundary(
                invalid, false, true, true);
            records.write(
                20u,
                make_float4(
                    as_float(camera_copy.count()),
                    as_float(
                        camera_copy.entry(1u).object),
                    as_float(
                        camera_copy.entry(2u).object),
                    flag(camera_copy.entry(3u).valid)));

            VolumeStack terminator_only{0u};
            terminator_only.initialize_background(
                background, true);
            records.write(
                21u,
                make_float4(
                    as_float(
                        terminator_only.count()),
                    static_cast<float>(
                        terminator_only.storage_size()),
                    static_cast<float>(
                        terminator_only.maximum_entries()),
                    flag(
                        terminator_only.entry(0u)
                            .valid)));

            VolumeStack oversized{
                maximum_volume_stack_size + 7u};
            records.write(
                22u,
                make_float4(
                    static_cast<float>(
                        oversized.storage_size()),
                    static_cast<float>(
                        oversized.maximum_entries()),
                    as_float(oversized.count()),
                    flag(oversized.entry(0u).valid)));
        };

    auto shader = device.compile(evaluate);
    std::array<luisa::float4, record_count>
        actual{};
    stream
        << shader(output).dispatch(1u)
        << output.copy_to(luisa::span{actual})
        << synchronize();

    // Pinned to official Cycles main b82c3f0 volume_stack.h and
    // intersect_volume_stack.h. These records cover the terminator slot,
    // exact (object, shader) identity, stable append, swap-last exit,
    // capacity and duplicate rejection, transmission/volume guards, shadow
    // copy, the camera +Z enclosure probe's object-only membership checks
    // (including Cycles' duplicate front-hit accounting), and the fixed
    // MAX_VOLUME_STACK_SIZE bound.
    constexpr std::array expected{
        luisa::float4{0.0f, 1.0f, 0.0f, 3.0f},
        luisa::float4{1.0f, 900.0f, 90.0f, 190.0f},
        luisa::float4{2.0f, 900.0f, 1.0f, 11.0f},
        luisa::float4{3.0f, 1.0f, 2.0f, 22.0f},
        luisa::float4{3.0f, 900.0f, 1.0f, 2.0f},
        luisa::float4{3.0f, 900.0f, 1.0f, 2.0f},
        luisa::float4{2.0f, 900.0f, 2.0f, 22.0f},
        luisa::float4{2.0f, 900.0f, 2.0f, 0.0f},
        luisa::float4{2.0f, 900.0f, 2.0f, 22.0f},
        luisa::float4{1.0f, 900.0f, 0.0f, 0.0f},
        luisa::float4{2.0f, 50.0f, 51.0f, 0.0f},
        luisa::float4{1.0f, 5.0f, 51.0f, 0.0f},
        luisa::float4{1.0f, 5.0f, 51.0f, 151.0f},
        luisa::float4{251.0f, 351.0f, 1.0f, 0.0f},
        luisa::float4{3.0f, 900.0f, 20.0f, 40.0f},
        luisa::float4{20.0f, 120.0f, 220.0f, 320.0f},
        luisa::float4{40.0f, 140.0f, 240.0f, 340.0f},
        luisa::float4{3.0f, 1.0f, 1.0f, 0.0f},
        luisa::float4{0.0f, 0.0f, 1.0f, 32.0f},
        luisa::float4{3.0f, 900.0f, 20.0f, 40.0f},
        luisa::float4{3.0f, 20.0f, 40.0f, 0.0f},
        luisa::float4{0.0f, 1.0f, 0.0f, 0.0f},
        luisa::float4{32.0f, 31.0f, 0.0f, 0.0f}};
    for (auto index = std::size_t{0u};
         index < expected.size();
         ++index) {
        if (!approximately_equal(
                actual[index],
                expected[index])) {
            std::cerr
                << "Cycles volume stack fixture failed on "
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

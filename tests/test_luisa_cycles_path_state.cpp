#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_path_state.h>
#include <psycles/luisa/cycles_sampler.h>
#include <psycles/sampling/tabulated_sobol.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::luisa_backend;

constexpr auto expected_diffuse_flag =
    cycles_path_state::flag_reflect |
    cycles_path_state::flag_diffuse_ancestor |
    cycles_path_state::flag_single_pass_done |
    cycles_path_state::flag_terminate_after_transparent |
    cycles_path_state::flag_surface_pass;

constexpr auto expected_singular_flag =
    cycles_path_state::flag_reflect |
    cycles_path_state::flag_singular |
    cycles_path_state::flag_mis_skip |
    cycles_path_state::flag_single_pass_done |
    cycles_path_state::flag_terminate_after_transparent |
    cycles_path_state::flag_surface_pass;

} // namespace

int main(int argc, char **argv) {
    const auto backend =
        std::string_view{argc > 1 ? argv[1] : "fallback"};

    Kernel1D evaluate =
        [](BufferFloat4 output) noexcept {
            const auto initial =
                cycles_path_state::initial_state();
            const auto diffuse =
                cycles_path_state::next_surface(
                    initial,
                    cycles_closure::label_reflect |
                        cycles_closure::label_diffuse,
                    cycles_closure::runtime_bsdf |
                        cycles_closure::
                            runtime_bsdf_has_eval,
                    {
                        .maximum = 1u,
                        .maximum_diffuse = 4u,
                        .maximum_glossy = 4u,
                        .maximum_transmission = 4u,
                        .maximum_transparent = 8u});
            output.write(
                0u,
                make_float4(
                    cast<float>(diffuse.flag),
                    cast<float>(diffuse.visibility),
                    cast<float>(diffuse.bounce),
                    cast<float>(diffuse.rng_offset)));
            output.write(
                1u,
                make_float4(
                    cast<float>(
                        diffuse.diffuse_bounce),
                    cast<float>(
                        diffuse.glossy_bounce),
                    cast<float>(
                        diffuse.transmission_bounce),
                    cast<float>(
                        diffuse.transparent_bounce)));
            output.write(
                2u,
                make_float4(
                    cast<float>(
                        cycles_path_state::
                            contract_visibility(
                                diffuse.visibility)),
                    0.0f,
                    0.0f,
                    0.0f));

            const auto transparent =
                cycles_path_state::next_surface(
                    initial,
                    cycles_closure::label_transparent,
                    cycles_closure::runtime_bsdf |
                        cycles_closure::
                            runtime_transparent,
                    {
                        .maximum = 8u,
                        .maximum_diffuse = 8u,
                        .maximum_glossy = 8u,
                        .maximum_transmission = 8u,
                        .maximum_transparent = 1u});
            output.write(
                3u,
                make_float4(
                    cast<float>(transparent.flag),
                    cast<float>(
                        transparent.visibility),
                    cast<float>(transparent.bounce),
                    cast<float>(
                        transparent.transparent_bounce)));
            output.write(
                4u,
                make_float4(
                    cast<float>(
                        transparent.rng_offset),
                    cast<float>(
                        cycles_path_state::
                            contract_visibility(
                                transparent.visibility)),
                    0.0f,
                    0.0f));

            const auto singular =
                cycles_path_state::next_surface(
                    initial,
                    cycles_closure::label_reflect |
                        cycles_closure::label_singular,
                    cycles_closure::runtime_bsdf,
                    {
                        .maximum = 8u,
                        .maximum_diffuse = 8u,
                        .maximum_glossy = 1u,
                        .maximum_transmission = 8u,
                        .maximum_transparent = 8u});
            output.write(
                5u,
                make_float4(
                    cast<float>(singular.flag),
                    cast<float>(singular.visibility),
                    cast<float>(
                        singular.glossy_bounce),
                    cast<float>(singular.rng_offset)));

            const auto transparent_twice =
                cycles_path_state::next_surface(
                    transparent,
                    cycles_closure::label_transparent,
                    cycles_closure::runtime_bsdf |
                        cycles_closure::
                            runtime_transparent,
                    {
                        .maximum = 8u,
                        .maximum_diffuse = 8u,
                        .maximum_glossy = 8u,
                        .maximum_transmission = 8u,
                        .maximum_transparent = 8u});
            output.write(
                6u,
                make_float4(
                    cast<float>(
                        transparent_twice.rng_offset),
                    cast<float>(
                        cycles_sampler::
                            path_state_dimension(
                                transparent_twice
                                    .rng_offset,
                                psycles::sampling::
                                    tabulated_sobol::
                                        light_dimension)),
                    cast<float>(
                        transparent_twice.bounce),
                    cast<float>(
                        transparent_twice
                            .transparent_bounce)));

            constexpr auto input_visibility =
                psycles::contract::visibility_bit(
                    psycles::contract::RayVisibility::camera) |
                psycles::contract::visibility_bit(
                    psycles::contract::
                        RayVisibility::transmission);
            constexpr auto input_events =
                psycles::contract::event_singular |
                psycles::contract::event_reflection;
            const auto surface_shader_state =
                cycles_path_state::surface_shader_state(
                    input_visibility,
                    input_events,
                    3u,
                    2u,
                    1u,
                    4u,
                    5u);
            const auto background_shader_state =
                cycles_path_state::
                    background_emission_shader_state(
                        input_visibility,
                        input_events,
                        3u,
                        2u,
                        1u,
                        4u,
                        5u);
            const auto light_shader_state =
                cycles_path_state::
                    light_emission_shader_state(
                        3u, 2u, 1u, 4u, 5u);
            const auto shadow_shader_state =
                cycles_path_state::shadow_shader_state(
                    3u, 2u, 1u, 4u, 5u);

            const auto write_shader_state =
                [&](std::uint32_t index,
                    const cycles_path_state::
                        ShaderEvaluationState
                            &state) noexcept {
                    output.write(
                        index,
                        make_float4(
                            cast<float>(
                                state.ray_visibility),
                            cast<float>(state.ray_events),
                            cast<float>(state.ray_depth),
                            cast<float>(
                                state.diffuse_depth)));
                    output.write(
                        index + 1u,
                        make_float4(
                            cast<float>(
                                state.glossy_depth),
                            cast<float>(
                                state.transparent_depth),
                            cast<float>(
                                state.transmission_depth),
                            0.0f));
                };
            write_shader_state(7u, surface_shader_state);
            write_shader_state(
                9u, background_shader_state);
            write_shader_state(11u, light_shader_state);
            write_shader_state(13u, shadow_shader_state);

            const auto volume =
                cycles_path_state::next_volume(
                    initial,
                    0u,
                    1u,
                    8u);
            output.write(
                15u,
                make_float4(
                    cast<float>(
                        volume.state.flag),
                    cast<float>(
                        volume.state.visibility),
                    cast<float>(
                        volume.state.bounce),
                    cast<float>(
                        volume.state.rng_offset)));
            output.write(
                16u,
                make_float4(
                    cast<float>(
                        volume.volume_bounce),
                    cast<float>(
                        volume.state.diffuse_bounce),
                    cast<float>(
                        volume.state.glossy_bounce),
                    cast<float>(
                        volume.state
                            .transmission_bounce)));
            output.write(
                17u,
                make_float4(
                    cycles_path_state::
                        continuation_probability(
                            initial.flag,
                            0u,
                            0u,
                            0u,
                            0u,
                            make_float3(0.0f)),
                    cycles_path_state::
                        continuation_probability(
                            initial.flag |
                                cycles_path_state::
                                    flag_transparent,
                            99u,
                            2u,
                            0u,
                            2u,
                            make_float3(0.0f)),
                    cycles_path_state::
                        continuation_probability(
                            initial.flag,
                            3u,
                            0u,
                            0u,
                            0u,
                            make_float3(
                                0.25f,
                                0.04f,
                                0.01f)),
                    cycles_path_state::
                        continuation_probability(
                            initial.flag,
                            3u,
                            0u,
                            0u,
                            0u,
                            make_float3(
                                -0.81f,
                                0.04f,
                                0.01f))));
        };

    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    auto output =
        device.create_buffer<luisa::float4>(18u);
    auto kernel = device.compile(evaluate);
    std::array<luisa::float4, 18u> actual{};
    stream << kernel(output).dispatch(1u)
           << output.copy_to(luisa::span{actual})
           << synchronize();

    constexpr std::array expected{
        luisa::float4{
            static_cast<float>(expected_diffuse_flag),
            4.0f,
            1.0f,
            32.0f},
        luisa::float4{1.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{2.0f, 0.0f, 0.0f, 0.0f},
        luisa::float4{1924.0f, 1.0f, 0.0f, 1.0f},
        luisa::float4{32.0f, 1.0f, 0.0f, 0.0f},
        luisa::float4{
            static_cast<float>(expected_singular_flag),
            8.0f,
            1.0f,
            32.0f},
        // Two transparent transitions consume two complete 16-dimensional
        // bounce blocks without incrementing the ordinary bounce counter.
        // The next light sample is addressed directly from rng_offset.
        luisa::float4{48.0f, 49.0f, 0.0f, 2.0f},
        // Ordinary surface evaluation preserves the incoming state.
        luisa::float4{9.0f, 12.0f, 3.0f, 2.0f},
        luisa::float4{1.0f, 4.0f, 5.0f, 0.0f},
        // Background evaluation preserves visibility/flags and adds the
        // effective PATH_RAY_EMISSION depth.
        luisa::float4{9.0f, 12.0f, 4.0f, 2.0f},
        luisa::float4{1.0f, 4.0f, 5.0f, 0.0f},
        // Lamp and NEE emission suppress path visibility/flags.
        luisa::float4{0.0f, 0.0f, 4.0f, 2.0f},
        luisa::float4{1.0f, 4.0f, 5.0f, 0.0f},
        // Transparent-shadow evaluation exposes only shadow visibility and
        // the same effective one-bounce-deeper Ray Depth.
        luisa::float4{16.0f, 0.0f, 4.0f, 2.0f},
        luisa::float4{1.0f, 4.0f, 5.0f, 0.0f},
        // A volume collision is one ordinary bounce and consumes the next
        // complete Sobol block. It clears camera/background MIS state,
        // establishes the first volume pass, and reaches the synced limit.
        luisa::float4{
            static_cast<float>(
                cycles_path_state::
                    flag_single_pass_done |
                cycles_path_state::
                    flag_mis_had_transmission |
                cycles_path_state::
                    flag_volume_pass |
                cycles_path_state::
                    flag_terminate_after_transparent),
            16.0f,
            1.0f,
            32.0f},
        luisa::float4{
            1.0f, 0.0f, 0.0f, 0.0f},
        // Minimum-bounce decisions are exact predicates; after them Cycles
        // uses sqrt(max(abs(throughput))) and clamps only the upper bound.
        luisa::float4{
            1.0f, 1.0f, 0.5f, 0.9f}};
    for (std::size_t index = 0u;
         index < expected.size();
         ++index) {
        const auto mismatch =
            actual[index].x != expected[index].x ||
            actual[index].y != expected[index].y ||
            actual[index].z != expected[index].z ||
            actual[index].w != expected[index].w;
        if (mismatch) {
            std::cerr
                << "Cycles path-state oracle failed on "
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

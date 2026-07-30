#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_path_state.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/contract/scene.h>
#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_path_state {

// Stable values from the Cycles PathRayFlag ABI. These flags are renderer
// state, not object-visibility masks.
inline constexpr std::uint32_t flag_none = 0u;
inline constexpr std::uint32_t flag_reflect = 1u << 0u;
inline constexpr std::uint32_t flag_singular = 1u << 1u;
inline constexpr std::uint32_t flag_transparent = 1u << 2u;
inline constexpr std::uint32_t flag_importance_bake = 1u << 3u;
inline constexpr std::uint32_t flag_diffuse_ancestor = 1u << 4u;
inline constexpr std::uint32_t flag_emission = 1u << 5u;
inline constexpr std::uint32_t flag_mis_had_transmission = 1u << 6u;
inline constexpr std::uint32_t flag_mis_skip = 1u << 7u;
inline constexpr std::uint32_t flag_single_pass_done = 1u << 8u;
inline constexpr std::uint32_t flag_transparent_background = 1u << 9u;
inline constexpr std::uint32_t flag_terminate_on_next_surface = 1u << 10u;
inline constexpr std::uint32_t flag_terminate_in_next_volume = 1u << 11u;
inline constexpr std::uint32_t flag_terminate_after_transparent = 1u << 12u;
inline constexpr std::uint32_t flag_terminate_after_volume = 1u << 13u;
inline constexpr std::uint32_t flag_surface_pass = 1u << 18u;
inline constexpr std::uint32_t flag_volume_pass = 1u << 19u;
inline constexpr std::uint32_t flag_any_pass =
    flag_surface_pass | flag_volume_pass;

inline constexpr std::uint32_t visibility_none = 0u;
inline constexpr std::uint32_t visibility_camera = 1u << 0u;
inline constexpr std::uint32_t visibility_transmit = 1u << 1u;
inline constexpr std::uint32_t visibility_diffuse = 1u << 2u;
inline constexpr std::uint32_t visibility_glossy = 1u << 3u;
inline constexpr std::uint32_t visibility_volume_scatter = 1u << 4u;

inline constexpr std::uint32_t bounce_dimension_count = 16u;

struct State {
    luisa::compute::UInt flag;
    luisa::compute::UInt visibility;
    luisa::compute::UInt bounce;
    luisa::compute::UInt diffuse_bounce;
    luisa::compute::UInt glossy_bounce;
    luisa::compute::UInt transmission_bounce;
    luisa::compute::UInt transparent_bounce;
    luisa::compute::UInt rng_offset;
};

struct Limits {
    luisa::compute::UInt maximum;
    luisa::compute::UInt maximum_diffuse;
    luisa::compute::UInt maximum_glossy;
    luisa::compute::UInt maximum_transmission;
    luisa::compute::UInt maximum_transparent;
};

[[nodiscard]] inline State initial_state() noexcept {
    using namespace luisa::compute;
    return {
        .flag =
            flag_mis_skip |
            flag_single_pass_done |
            flag_transparent_background,
        .visibility = visibility_camera,
        .bounce = 0u,
        .diffuse_bounce = 0u,
        .glossy_bounce = 0u,
        .transmission_bounce = 0u,
        .transparent_bounce = 0u,
        .rng_offset = bounce_dimension_count};
}

// Cycles advances all surface paths through one path_state_next transition.
// Model the transition as a total function so counters, visibility, flags and
// random dimensions cannot drift apart in separate renderer branches.
[[nodiscard]] inline State next_surface(
    const State &before,
    luisa::compute::UInt label,
    luisa::compute::UInt runtime_flag,
    const Limits &limits) noexcept {
    using namespace luisa::compute;

    const auto transparent =
        (label &
         (cycles_closure::label_transparent |
          cycles_closure::label_ray_portal)) != 0u;

    auto transparent_state = before;
    transparent_state.transparent_bounce += 1u;
    transparent_state.rng_offset +=
        bounce_dimension_count;
    transparent_state.flag |= flag_transparent;
    transparent_state.flag |= select(
        0u,
        flag_terminate_on_next_surface,
        transparent_state.transparent_bounce >=
            limits.maximum_transparent);
    transparent_state.flag |= select(
        0u,
        flag_mis_skip,
        (runtime_flag &
         cycles_closure::runtime_ray_portal) != 0u);

    auto surface_state = before;
    surface_state.bounce += 1u;
    surface_state.rng_offset +=
        bounce_dimension_count;
    surface_state.flag |= select(
        0u,
        flag_terminate_after_transparent,
        surface_state.bounce >= limits.maximum);
    surface_state.visibility = visibility_none;
    surface_state.flag &=
        ~(flag_reflect |
          flag_singular |
          flag_transparent |
          flag_importance_bake |
          flag_mis_skip |
          flag_mis_had_transmission);

    const auto reflection =
        (label & cycles_closure::label_reflect) != 0u;
    const auto diffuse =
        (label & cycles_closure::label_diffuse) != 0u;
    const auto glossy =
        (label & cycles_closure::label_glossy) != 0u;
    const auto transmit_transparent =
        (label &
         cycles_closure::label_transmit_transparent) != 0u;

    surface_state.flag |=
        select(0u, flag_reflect, reflection);
    surface_state.flag = select(
        surface_state.flag,
        surface_state.flag &
            ~flag_transparent_background,
        reflection | !transmit_transparent);

    const auto diffuse_reflection =
        reflection & diffuse;
    const auto glossy_reflection =
        reflection & !diffuse;
    const auto transmission = !reflection;

    surface_state.diffuse_bounce +=
        select(0u, 1u, diffuse_reflection);
    surface_state.glossy_bounce +=
        select(0u, 1u, glossy_reflection);
    surface_state.transmission_bounce +=
        select(0u, 1u, transmission);

    surface_state.flag |= select(
        0u,
        flag_terminate_after_transparent,
        diffuse_reflection &
            (surface_state.diffuse_bounce >=
             limits.maximum_diffuse));
    surface_state.flag |= select(
        0u,
        flag_terminate_after_transparent,
        glossy_reflection &
            (surface_state.glossy_bounce >=
             limits.maximum_glossy));
    surface_state.flag |= select(
        0u,
        flag_terminate_after_transparent,
        transmission &
            (surface_state.transmission_bounce >=
             limits.maximum_transmission));

    surface_state.visibility |= select(
        0u,
        visibility_transmit,
        transmission);
    surface_state.visibility |= select(
        visibility_glossy,
        visibility_diffuse,
        diffuse);
    surface_state.flag |= select(
        0u,
        flag_diffuse_ancestor,
        diffuse);
    surface_state.flag |= select(
        0u,
        flag_singular | flag_mis_skip,
        !diffuse & !glossy);
    surface_state.flag |= select(
        0u,
        flag_mis_had_transmission,
        (runtime_flag &
         cycles_closure::
             runtime_bsdf_has_transmission) != 0u);
    surface_state.flag |= select(
        0u,
        flag_surface_pass,
        ((surface_state.flag & flag_any_pass) == 0u) &
            ((surface_state.flag &
              flag_transparent_background) == 0u));

    return {
        .flag = select(
            surface_state.flag,
            transparent_state.flag,
            transparent),
        .visibility = select(
            surface_state.visibility,
            transparent_state.visibility,
            transparent),
        .bounce = select(
            surface_state.bounce,
            transparent_state.bounce,
            transparent),
        .diffuse_bounce = select(
            surface_state.diffuse_bounce,
            transparent_state.diffuse_bounce,
            transparent),
        .glossy_bounce = select(
            surface_state.glossy_bounce,
            transparent_state.glossy_bounce,
            transparent),
        .transmission_bounce = select(
            surface_state.transmission_bounce,
            transparent_state.transmission_bounce,
            transparent),
        .transparent_bounce = select(
            surface_state.transparent_bounce,
            transparent_state.transparent_bounce,
            transparent),
        .rng_offset = select(
            surface_state.rng_offset,
            transparent_state.rng_offset,
            transparent)};
}

[[nodiscard]] inline luisa::compute::UInt
contract_visibility(
    luisa::compute::UInt cycles_visibility) noexcept {
    using namespace luisa::compute;
    UInt result = 0u;
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::camera),
        (cycles_visibility & visibility_camera) != 0u);
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::transmission),
        (cycles_visibility & visibility_transmit) != 0u);
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::diffuse),
        (cycles_visibility & visibility_diffuse) != 0u);
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::glossy),
        (cycles_visibility & visibility_glossy) != 0u);
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::volume_scatter),
        (cycles_visibility &
         visibility_volume_scatter) != 0u);
    return result;
}

} // namespace psycles::luisa_backend::cycles_path_state

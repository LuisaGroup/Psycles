#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_path_state.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/contract/scene.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface.h>

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
inline constexpr std::uint32_t flag_terminate =
    flag_terminate_on_next_surface |
    flag_terminate_in_next_volume |
    flag_terminate_after_transparent |
    flag_terminate_after_volume;
inline constexpr std::uint32_t flag_subsurface_random_walk = 1u << 14u;
inline constexpr std::uint32_t flag_subsurface_disk = 1u << 15u;
inline constexpr std::uint32_t flag_subsurface_backfacing = 1u << 16u;
inline constexpr std::uint32_t flag_subsurface =
    flag_subsurface_random_walk |
    flag_subsurface_disk |
    flag_subsurface_backfacing;
inline constexpr std::uint32_t flag_surface_pass = 1u << 18u;
inline constexpr std::uint32_t flag_volume_pass = 1u << 19u;
inline constexpr std::uint32_t
    flag_volume_primary_transmit = 1u << 24u;
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

struct VolumeTransition {
    State state;
    luisa::compute::UInt volume_bounce;
};

// Canonical state observed by a Cycles Light Path node. Keep evaluation
// context separate from transport state: surface, background emission, lamp
// emission, and transparent-shadow evaluation deliberately expose different
// visibility/flag/depth combinations even when they originate from one path.
struct ShaderEvaluationState {
    luisa::compute::UInt ray_visibility;
    luisa::compute::UInt ray_events;
    luisa::compute::UInt ray_depth;
    luisa::compute::UInt diffuse_depth;
    luisa::compute::UInt glossy_depth;
    luisa::compute::UInt transparent_depth;
    luisa::compute::UInt transmission_depth;
};

[[nodiscard]] inline ShaderEvaluationState
surface_shader_state(
    luisa::compute::UInt ray_visibility,
    luisa::compute::UInt ray_events,
    luisa::compute::UInt ray_depth,
    luisa::compute::UInt diffuse_depth,
    luisa::compute::UInt glossy_depth,
    luisa::compute::UInt transparent_depth,
    luisa::compute::UInt transmission_depth) noexcept {
    return {
        .ray_visibility = ray_visibility,
        .ray_events = ray_events,
        .ray_depth = ray_depth,
        .diffuse_depth = diffuse_depth,
        .glossy_depth = glossy_depth,
        .transparent_depth = transparent_depth,
        .transmission_depth = transmission_depth};
}

// Background shaders retain the incoming path visibility and flags, but
// PATH_RAY_EMISSION makes Light Path's Ray Depth one bounce deeper.
[[nodiscard]] inline ShaderEvaluationState
background_emission_shader_state(
    luisa::compute::UInt ray_visibility,
    luisa::compute::UInt ray_events,
    luisa::compute::UInt ray_depth,
    luisa::compute::UInt diffuse_depth,
    luisa::compute::UInt glossy_depth,
    luisa::compute::UInt transparent_depth,
    luisa::compute::UInt transmission_depth) noexcept {
    return {
        .ray_visibility = ray_visibility,
        .ray_events = ray_events,
        .ray_depth = ray_depth + 1u,
        .diffuse_depth = diffuse_depth,
        .glossy_depth = glossy_depth,
        .transparent_depth = transparent_depth,
        .transmission_depth = transmission_depth};
}

// Analytic and NEE light shaders are evaluated with
// PATH_RAY_VISIBILITY_NONE | PATH_RAY_EMISSION in Cycles. In particular,
// Light Path must not see camera/diffuse/glossy/reflection/singular state.
[[nodiscard]] inline ShaderEvaluationState
light_emission_shader_state(
    luisa::compute::UInt ray_depth,
    luisa::compute::UInt diffuse_depth,
    luisa::compute::UInt glossy_depth,
    luisa::compute::UInt transparent_depth,
    luisa::compute::UInt transmission_depth) noexcept {
    return {
        .ray_visibility = visibility_none,
        .ray_events = 0u,
        .ray_depth = ray_depth + 1u,
        .diffuse_depth = diffuse_depth,
        .glossy_depth = glossy_depth,
        .transparent_depth = transparent_depth,
        .transmission_depth = transmission_depth};
}

// Transparent-shadow shaders receive shadow visibility and no path flags.
// Their ordinary bounce is copied from the main path and Light Path adds one
// for shadow evaluation; lobe and transparent counters remain observable.
[[nodiscard]] inline ShaderEvaluationState
shadow_shader_state(
    luisa::compute::UInt ray_depth,
    luisa::compute::UInt diffuse_depth,
    luisa::compute::UInt glossy_depth,
    luisa::compute::UInt transparent_depth,
    luisa::compute::UInt transmission_depth) noexcept {
    return {
        .ray_visibility =
            contract::visibility_bit(
                contract::RayVisibility::shadow),
        .ray_events = 0u,
        .ray_depth = ray_depth + 1u,
        .diffuse_depth = diffuse_depth,
        .glossy_depth = glossy_depth,
        .transparent_depth = transparent_depth,
        .transmission_depth = transmission_depth};
}

inline void apply_shader_state(
    SurfacePoint &point,
    const ShaderEvaluationState &state) noexcept {
    point.ray_visibility = state.ray_visibility;
    point.ray_events = state.ray_events;
    point.ray_depth = state.ray_depth;
    point.diffuse_depth = state.diffuse_depth;
    point.glossy_depth = state.glossy_depth;
    point.transparent_depth = state.transparent_depth;
    point.transmission_depth = state.transmission_depth;
}

[[nodiscard]] inline State initial_state() noexcept {
    using namespace luisa::compute;
    return {
        .flag =
            flag_mis_skip |
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

// The volume arm of Cycles' path_state_next() is kept as a total transition
// for the same reason as next_surface(): visibility, flags, bounce counters,
// and the 16-dimensional RNG block must advance atomically.
[[nodiscard]] inline VolumeTransition
next_volume(
    const State &before,
    luisa::compute::UInt volume_bounce,
    luisa::compute::UInt maximum_volume,
    luisa::compute::UInt maximum) noexcept {
    using namespace luisa::compute;

    auto state = before;
    state.bounce += 1u;
    state.rng_offset +=
        bounce_dimension_count;
    state.visibility =
        visibility_volume_scatter;
    state.flag &=
        ~(flag_reflect |
          flag_singular |
          flag_transparent |
          flag_importance_bake |
          flag_mis_skip |
          flag_mis_had_transmission |
          flag_transparent_background);
    state.flag |=
        flag_mis_had_transmission;
    state.flag = select(
        state.flag,
        state.flag &
            ~flag_volume_primary_transmit,
        state.bounce == 1u);
    state.flag |= select(
        0u,
        flag_volume_pass,
        (state.flag & flag_any_pass) == 0u);

    volume_bounce += 1u;
    state.flag |= select(
        0u,
        flag_terminate_after_transparent,
        (state.bounce >= maximum) |
            (volume_bounce >= maximum_volume));
    return {
        .state = state,
        .volume_bounce =
            volume_bounce};
}

// Exact path_state_continuation_probability() policy. Keeping this shared
// prevents the surface and pre-event volume stages from drifting on the
// transparent minimum-bounce rule or the sqrt(max(abs(throughput))) measure.
[[nodiscard]] inline luisa::compute::Float
continuation_probability(
    luisa::compute::UInt flag,
    luisa::compute::UInt bounce,
    luisa::compute::UInt transparent_bounce,
    luisa::compute::UInt minimum,
    luisa::compute::UInt transparent_minimum,
    luisa::compute::Float3 throughput) noexcept {
    using namespace luisa::compute;
    const auto transparent =
        (flag & flag_transparent) != 0u;
    const auto before_minimum =
        select(
            bounce <= minimum,
            transparent_bounce <=
                transparent_minimum,
            transparent);
    const auto roulette =
        min(
            sqrt(
                max(
                    abs(throughput.x),
                    max(
                        abs(throughput.y),
                        abs(throughput.z)))),
            1.0f);
    return select(
        roulette,
        1.0f,
        before_minimum);
}

// Cycles resolves continuation roulette in INTERSECT_CLOSEST, after the
// committed endpoint's static shader flags are known and before scheduling
// any surface or volume shading. The outcomes below form a disjoint and
// exhaustive partition of a failed decision: defer to an emissive surface,
// defer to an active volume, or terminate immediately.
struct ClosestContinuationDecision {
    luisa::compute::Bool failed;
    luisa::compute::Bool terminate_immediately;
    luisa::compute::UInt deferred_flags;
};

[[nodiscard]] inline ClosestContinuationDecision
decide_closest_continuation(
    luisa::compute::Float probability,
    luisa::compute::Float terminate_sample,
    luisa::compute::Bool surface_may_emit,
    luisa::compute::Bool inside_volume) noexcept {
    using namespace luisa::compute;
    const auto failed =
        (probability != 1.0f) &
        ((probability == 0.0f) |
         (terminate_sample >= probability));
    const auto defer_to_surface =
        failed & surface_may_emit;
    const auto defer_to_volume =
        failed & !surface_may_emit & inside_volume;
    UInt deferred_flags = 0u;
    deferred_flags |= select(
        0u,
        flag_terminate_on_next_surface,
        defer_to_surface);
    deferred_flags |= select(
        0u,
        flag_terminate_in_next_volume,
        defer_to_volume);
    return {
        .failed = failed,
        .terminate_immediately =
            failed & !defer_to_surface & !defer_to_volume,
        .deferred_flags = deferred_flags};
}

[[nodiscard]] inline luisa::compute::UInt
contract_visibility(
    luisa::compute::UInt cycles_visibility) noexcept {
    using namespace luisa::compute;
    // Cycles uses diffuse/glossy object visibility for reflection only.
    // A transmissive lobe may carry its diffuse/glossy classification for
    // Light Path and pass accounting, but path_state_ray_visibility() removes
    // both bits before traversal whenever TRANSMIT is present.
    const auto traversal_visibility = select(
        cycles_visibility,
        cycles_visibility &
            ~(visibility_diffuse | visibility_glossy),
        (cycles_visibility & visibility_transmit) != 0u);
    UInt result = 0u;
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::camera),
        (traversal_visibility & visibility_camera) != 0u);
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::transmission),
        (traversal_visibility & visibility_transmit) != 0u);
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::diffuse),
        (traversal_visibility & visibility_diffuse) != 0u);
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::glossy),
        (traversal_visibility & visibility_glossy) != 0u);
    result |= select(
        0u,
        contract::visibility_bit(
            contract::RayVisibility::volume_scatter),
        (traversal_visibility &
         visibility_volume_scatter) != 0u);
    return result;
}

} // namespace psycles::luisa_backend::cycles_path_state

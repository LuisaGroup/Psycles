#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/cycles_closure.h> through the Psycles::luisa target."
#endif

#include <cstdint>

#include <psycles/contract/surface.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_closure {

// Exact numeric ABI of Cycles' kernel/closure/bsdf_microfacet.h
// MicrofacetFresnel. ClosureType identifies the scattering distribution and
// transport family; this independent discriminator identifies the Fresnel
// payload interpreted by a reflection microfacet closure.
enum class MicrofacetFresnel : std::uint32_t {
    none = 0u,
    dielectric = 1u,
    dielectric_tint = 2u,
    conductor = 3u,
    generalized_schlick = 4u,
    f82_tint = 5u,
};

// Stable values from the Cycles ClosureType ABI used by ShaderClosure. These
// are trace and interoperability identities, not Psycles-internal operation
// tags.
inline constexpr std::uint32_t type_none = 0u;
inline constexpr std::uint32_t type_diffuse = 2u;
inline constexpr std::uint32_t type_oren_nayar = 3u;
inline constexpr std::uint32_t type_rough_translucent = 4u;
inline constexpr std::uint32_t type_sheen = 7u;
inline constexpr std::uint32_t type_diffuse_toon = 8u;
inline constexpr std::uint32_t type_translucent = 9u;
inline constexpr std::uint32_t type_microfacet_ggx = 12u;
inline constexpr std::uint32_t type_microfacet_beckmann = 13u;
inline constexpr std::uint32_t type_ashikhmin_shirley = 15u;
inline constexpr std::uint32_t type_ashikhmin_velvet = 16u;
inline constexpr std::uint32_t type_glossy_toon = 18u;
inline constexpr std::uint32_t type_hair_reflection = 19u;
inline constexpr std::uint32_t type_microfacet_beckmann_refraction = 20u;
inline constexpr std::uint32_t type_microfacet_ggx_refraction = 21u;
inline constexpr std::uint32_t type_thin_glass_transmission = 22u;
inline constexpr std::uint32_t type_hair_transmission = 23u;
inline constexpr std::uint32_t type_microfacet_beckmann_glass = 24u;
inline constexpr std::uint32_t type_microfacet_ggx_glass = 25u;
// Virtual SVM authoring type. Cycles' GGX setup overwrites it with
// type_microfacet_ggx_glass before the closure becomes observable.
inline constexpr std::uint32_t type_microfacet_multi_ggx_glass = 26u;
inline constexpr std::uint32_t type_ray_portal = 29u;
inline constexpr std::uint32_t type_transparent = 30u;
inline constexpr std::uint32_t type_bssrdf_burley = 31u;
inline constexpr std::uint32_t type_bssrdf_random_walk = 32u;
inline constexpr std::uint32_t type_bssrdf_random_walk_legacy = 33u;
inline constexpr std::uint32_t type_bssrdf_random_walk_skin = 34u;
// Virtual SVM aggregate. Principled population expands it into physical
// ShaderClosure entries and this value must never enter retained storage.
inline constexpr std::uint32_t type_principled_virtual = 43u;

inline constexpr std::uint32_t label_none = 0u;
inline constexpr std::uint32_t label_transmit = 1u;
inline constexpr std::uint32_t label_reflect = 2u;
inline constexpr std::uint32_t label_diffuse = 4u;
inline constexpr std::uint32_t label_glossy = 8u;
inline constexpr std::uint32_t label_singular = 16u;
inline constexpr std::uint32_t label_transparent = 32u;
inline constexpr std::uint32_t label_volume_scatter = 64u;
inline constexpr std::uint32_t label_transmit_transparent = 128u;
inline constexpr std::uint32_t label_subsurface_scatter = 256u;
inline constexpr std::uint32_t label_ray_portal = 512u;

inline constexpr std::uint32_t runtime_backfacing = 1u << 0u;
inline constexpr std::uint32_t runtime_cache_miss = 1u << 1u;
inline constexpr std::uint32_t runtime_emission = 1u << 2u;
inline constexpr std::uint32_t runtime_bsdf = 1u << 3u;
inline constexpr std::uint32_t runtime_bsdf_has_eval = 1u << 4u;
inline constexpr std::uint32_t runtime_bssrdf = 1u << 5u;
inline constexpr std::uint32_t runtime_holdout = 1u << 6u;
inline constexpr std::uint32_t runtime_extinction = 1u << 7u;
inline constexpr std::uint32_t runtime_scatter = 1u << 8u;
inline constexpr std::uint32_t runtime_is_volume_shader_eval = 1u << 9u;
inline constexpr std::uint32_t runtime_transparent = 1u << 10u;
inline constexpr std::uint32_t runtime_bsdf_has_transmission = 1u << 11u;
inline constexpr std::uint32_t runtime_ray_portal = 1u << 12u;

inline constexpr auto closure_weight_cutoff = 1.0e-5f;
inline constexpr auto microfacet_singular_alpha_product = 2.0e-10f;

// Device-side forms of the ordered ClosureType predicates in Cycles'
// kernel/svm/types.h. Keep these intervals centralized: retained closure
// consumers must dispatch on the post-setup type, not reconstruct a second
// classification from an authoring node tag.
[[nodiscard]] inline luisa::compute::Bool
is_diffuse_or_oren_nayar(luisa::compute::UInt type) noexcept {
    return (type >= type_diffuse) & (type <= type_oren_nayar);
}

[[nodiscard]] inline luisa::compute::Bool
is_reflection_microfacet(luisa::compute::UInt type) noexcept {
    return (type == type_microfacet_ggx) |
           (type == type_microfacet_beckmann);
}

[[nodiscard]] inline luisa::compute::Bool
is_refraction_microfacet(luisa::compute::UInt type) noexcept {
    return (type >= type_microfacet_beckmann_refraction) &
           (type <= type_microfacet_ggx_refraction);
}

[[nodiscard]] inline luisa::compute::Bool
is_glass_microfacet(luisa::compute::UInt type) noexcept {
    return (type >= type_microfacet_beckmann_glass) &
           (type <= type_microfacet_ggx_glass);
}

[[nodiscard]] inline luisa::compute::Bool
is_hair(luisa::compute::UInt type) noexcept {
    return (type == type_hair_reflection) |
           (type == type_hair_transmission);
}

[[nodiscard]] inline luisa::compute::Bool
is_bssrdf(luisa::compute::UInt type) noexcept {
    return (type >= type_bssrdf_burley) &
           (type <= type_bssrdf_random_walk_skin);
}

[[nodiscard]] inline luisa::compute::Bool
is_beckmann_microfacet(luisa::compute::UInt type) noexcept {
    return (type == type_microfacet_beckmann) |
           (type == type_microfacet_beckmann_refraction) |
           (type == type_microfacet_beckmann_glass);
}

// Fresnel payloads backed by a Cycles extra closure allocation carry the
// thin-film struct. Plain DIELECTRIC is the inline Coat/glass discriminator
// and owns no such payload; NONE owns no Fresnel payload at all.
[[nodiscard]] inline luisa::compute::Bool
fresnel_uses_thin_film_payload(luisa::compute::UInt fresnel) noexcept {
    return (fresnel == static_cast<std::uint32_t>(
                           MicrofacetFresnel::dielectric_tint)) |
           (fresnel == static_cast<std::uint32_t>(
                           MicrofacetFresnel::conductor)) |
           (fresnel == static_cast<std::uint32_t>(
                           MicrofacetFresnel::generalized_schlick)) |
           (fresnel == static_cast<std::uint32_t>(
                           MicrofacetFresnel::f82_tint));
}

// Convert the renderer-independent surface-event contract back to the exact
// Cycles ClosureLabel bit layout. Keeping this mapping explicit prevents
// path-state code from accidentally treating the two enums as ABI-compatible.
[[nodiscard]] inline luisa::compute::UInt
label_from_events(luisa::compute::UInt events) noexcept {
    using namespace luisa::compute;
    UInt label = label_none;
    label |= select(
        0u,
        label_transmit,
        (events &
         static_cast<std::uint32_t>(
             contract::event_transmission)) != 0u);
    label |= select(
        0u,
        label_reflect,
        (events &
         static_cast<std::uint32_t>(
             contract::event_reflection)) != 0u);
    label |= select(
        0u,
        label_diffuse,
        (events &
         static_cast<std::uint32_t>(
             contract::event_diffuse)) != 0u);
    label |= select(
        0u,
        label_glossy,
        (events &
         static_cast<std::uint32_t>(
             contract::event_glossy)) != 0u);
    label |= select(
        0u,
        label_singular,
        (events &
         static_cast<std::uint32_t>(
             contract::event_singular)) != 0u);
    label |= select(
        0u,
        label_transparent,
        (events &
         static_cast<std::uint32_t>(
             contract::event_transparent)) != 0u);
    label |= select(
        0u,
        label_subsurface_scatter,
        (events &
         static_cast<std::uint32_t>(
             contract::event_subsurface)) != 0u);
    return label;
}

} // namespace psycles::luisa_backend::cycles_closure

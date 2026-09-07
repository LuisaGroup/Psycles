#pragma once

#include <psycles/luisa/cycles_bsdf_tables.h>

#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {

// Exact Cycles 5.2.1 bsdf_microfacet_estimate_albedo() factors, expressed
// over the common physical payload rather than either renderer's storage
// class. ShaderClosure::weight is intentionally excluded from every result.
// Native SVM and compact surface population both call these functions so the
// post-population data-pass traversal cannot drift from closure setup.
[[nodiscard]] luisa::compute::Float3
microfacet_generalized_schlick_reflectance(
    const CyclesBsdfTableReader &tables,
    luisa::compute::Float3 incoming,
    luisa::compute::Float3 normal,
    luisa::compute::Float alpha_x,
    luisa::compute::Float alpha_y,
    luisa::compute::Float ior,
    luisa::compute::Float thin_film_thickness,
    luisa::compute::Float thin_film_ior,
    luisa::compute::Float3 f0,
    luisa::compute::Float3 f90,
    luisa::compute::Float exponent,
    bool may_have_thin_film = true) noexcept;

[[nodiscard]] luisa::compute::Float3
microfacet_generalized_schlick_albedo(
    const CyclesBsdfTableReader &tables,
    luisa::compute::Float3 incoming,
    luisa::compute::Float3 normal,
    luisa::compute::Float alpha_x,
    luisa::compute::Float alpha_y,
    luisa::compute::Float ior,
    luisa::compute::Float thin_film_thickness,
    luisa::compute::Float thin_film_ior,
    luisa::compute::Float3 reflection_tint,
    luisa::compute::Float3 transmission_tint,
    luisa::compute::Float3 f0,
    luisa::compute::Float3 f90,
    luisa::compute::Float exponent,
    luisa::compute::Bool reflection,
    luisa::compute::Bool transmission,
    bool may_have_thin_film = true) noexcept;

[[nodiscard]] luisa::compute::Float
microfacet_dielectric_reflection_albedo(
    const CyclesBsdfTableReader &tables,
    luisa::compute::Float3 incoming,
    luisa::compute::Float3 normal,
    luisa::compute::Float alpha_x,
    luisa::compute::Float alpha_y,
    luisa::compute::Float ior) noexcept;

[[nodiscard]] luisa::compute::Float3 microfacet_f82_tint_albedo(
    const CyclesBsdfTableReader &tables,
    luisa::compute::Float3 incoming,
    luisa::compute::Float3 normal,
    luisa::compute::Float alpha_x,
    luisa::compute::Float alpha_y,
    luisa::compute::Float thin_film_thickness,
    luisa::compute::Float thin_film_ior,
    luisa::compute::Float3 f0,
    luisa::compute::Float3 b,
    bool may_have_thin_film = true) noexcept;

[[nodiscard]] luisa::compute::Float3 microfacet_conductor_albedo(
    const CyclesBsdfTableReader &tables,
    luisa::compute::Float3 incoming,
    luisa::compute::Float3 normal,
    luisa::compute::Float thin_film_thickness,
    luisa::compute::Float thin_film_ior,
    luisa::compute::Float3 ior,
    luisa::compute::Float3 extinction,
    bool may_have_thin_film = true) noexcept;

}// namespace psycles::luisa_backend::detail

#pragma once

#include <psycles/luisa/cycles_bsdf_tables.h>

#include <luisa/dsl/syntax.h>

namespace psycles::luisa_backend::detail {

inline constexpr float thin_film_thickness_cutoff = 0.1f;

// Result of the dielectric Airy construction. The transmitted cosine follows
// Cycles' convention: it is measured against the incident-side normal and is
// therefore negative for an ordinary front-face refraction.
struct ThinFilmDielectricFresnel {
  luisa::compute::Float3 reflectance;
  luisa::compute::Float cosine_transmitted;
};

// Exact Cycles 5.2 thin-film specialization of generalized-Schlick
// dielectric Fresnel. `f0` is the authored per-channel endpoint; the Airy
// result is rescaled with the same real-F0 interpolation used by Cycles.
[[nodiscard]] ThinFilmDielectricFresnel
thin_film_dielectric_fresnel(const CyclesBsdfTableReader &services,
                             luisa::compute::Float thickness,
                             luisa::compute::Float film_ior,
                             luisa::compute::Float substrate_ior,
                             luisa::compute::Float3 f0,
                             luisa::compute::Float cosine_incoming) noexcept;

// Exact Cycles 5.2 thin-film specialization of the Principled F82-tint
// metallic model. `b` is the precomputed F82 correction term.
[[nodiscard]] luisa::compute::Float3 thin_film_f82_fresnel(
    const CyclesBsdfTableReader &services,
    luisa::compute::Float thickness,
    luisa::compute::Float film_ior,
    luisa::compute::Float3 f0,
    luisa::compute::Float3 b,
    luisa::compute::Float cosine_incoming) noexcept;

// Exact Cycles 5.2 Airy construction for a physical conductor substrate.
// `substrate_ior` and `substrate_extinction` are the authored non-negative
// complex-IOR components; no F82 fit participates in this path.
[[nodiscard]] luisa::compute::Float3 thin_film_conductor_fresnel(
    const CyclesBsdfTableReader &services,
    luisa::compute::Float thickness,
    luisa::compute::Float film_ior,
    luisa::compute::Float3 substrate_ior,
    luisa::compute::Float3 substrate_extinction,
    luisa::compute::Float cosine_incoming) noexcept;

} // namespace psycles::luisa_backend::detail

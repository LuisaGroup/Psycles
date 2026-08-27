#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

inline constexpr float thin_film_thickness_cutoff = 0.1f;

// Result of the dielectric Airy construction. The transmitted cosine follows
// Cycles' convention: it is measured against the incident-side normal and is
// therefore negative for an ordinary front-face refraction.
struct ThinFilmDielectricFresnel {
  Float3 reflectance;
  Float cosine_transmitted;
};

// Exact Cycles 5.2 thin-film specialization of generalized-Schlick
// dielectric Fresnel. `f0` is the authored per-channel endpoint; the Airy
// result is rescaled with the same real-F0 interpolation used by Cycles.
[[nodiscard]] ThinFilmDielectricFresnel
thin_film_dielectric_fresnel(const ShaderServices &services, Float thickness,
                             Float film_ior, Float substrate_ior, Float3 f0,
                             Float cosine_incoming) noexcept;

// Exact Cycles 5.2 thin-film specialization of the Principled F82-tint
// metallic model. `b` is the precomputed F82 correction term.
[[nodiscard]] Float3 thin_film_f82_fresnel(const ShaderServices &services,
                                           Float thickness, Float film_ior,
                                           Float3 f0, Float3 b,
                                           Float cosine_incoming) noexcept;

} // namespace psycles::luisa_backend::detail

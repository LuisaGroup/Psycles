#pragma once

#include "graph_surface_internal.h"

namespace psycles::luisa_backend::detail {

// Closure setup is the sole translation boundary from authored node
// parameters to the physical microfacet representation. Evaluation and
// sampling consume this triple directly:
//
//   M = (T, alpha_x, alpha_y).
//
// T is semantically dead when alpha_x == alpha_y. This is the same tagged
// relation used by Cycles' MicrofacetBsdf and prevents node-specific
// anisotropy conventions from leaking into transport code.
struct MicrofacetAnisotropyState {
  Float3 tangent;
  Float alpha_x;
  Float alpha_y;
};

[[nodiscard]] MicrofacetAnisotropyState
isotropic_microfacet_state(Float perceptual_roughness) noexcept;

[[nodiscard]] MicrofacetAnisotropyState
principled_microfacet_state(const TracedClosure &closure,
                            Float3 authored_normal) noexcept;

// Blender 5.2 standalone Metallic uses the same saturated aspect-ratio
// relation as Principled, but retains its own semantic entry point so future
// node changes cannot silently inherit Glossy's signed reciprocal-axis law.
[[nodiscard]] MicrofacetAnisotropyState
metallic_microfacet_state(const TracedClosure &closure,
                          Float3 authored_normal) noexcept;

[[nodiscard]] MicrofacetAnisotropyState
glossy_microfacet_state(const TracedClosure &closure,
                        Float3 physical_normal) noexcept;

void configure_microfacet_state(
    TracedClosure &closure, const MicrofacetAnisotropyState &state) noexcept;

} // namespace psycles::luisa_backend::detail

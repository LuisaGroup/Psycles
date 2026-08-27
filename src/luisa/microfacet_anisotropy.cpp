#include "microfacet_anisotropy.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float3 rotate_around_axis(Float3 value, Float3 axis,
                                        Float angle) noexcept {
  const auto cosine = cos(angle);
  const auto sine = sin(angle);
  const auto one_minus_cosine = 1.0f - cosine;
  return make_float3(
      (cosine + one_minus_cosine * axis.x * axis.x) * value.x +
          (one_minus_cosine * axis.x * axis.y - axis.z * sine) * value.y +
          (one_minus_cosine * axis.x * axis.z + axis.y * sine) * value.z,
      (one_minus_cosine * axis.x * axis.y + axis.z * sine) * value.x +
          (cosine + one_minus_cosine * axis.y * axis.y) * value.y +
          (one_minus_cosine * axis.y * axis.z - axis.x * sine) * value.z,
      (one_minus_cosine * axis.x * axis.z - axis.y * sine) * value.x +
          (one_minus_cosine * axis.y * axis.z + axis.x * sine) * value.y +
          (cosine + one_minus_cosine * axis.z * axis.z) * value.z);
}

[[nodiscard]] Float3 rotated_tangent(Float3 tangent, Float3 axis,
                                     Float rotation) noexcept {
  auto result = tangent;
  // The zero-rotation branch is part of Cycles' setup relation. Apart from
  // avoiding trigonometric work it preserves the authored tangent exactly.
  $if(rotation != 0.0f) {
    result = rotate_around_axis(tangent, axis, rotation * two_pi);
  };
  return result;
}

} // namespace

MicrofacetAnisotropyState
isotropic_microfacet_state(Float perceptual_roughness) noexcept {
  const auto roughness = clamp(perceptual_roughness, 0.0f, 1.0f);
  const auto alpha = roughness * roughness;
  return {.tangent = make_float3(0.0f), .alpha_x = alpha, .alpha_y = alpha};
}

namespace {

[[nodiscard]] MicrofacetAnisotropyState
aspect_ratio_microfacet_state(const TracedClosure &closure,
                              Float3 authored_normal) noexcept {
  auto state = isotropic_microfacet_state(closure.roughness);
  if (!closure.anisotropy_enabled) {
    return state;
  }

  // Principled uses an aspect-ratio parameterization. Before setup
  // saturation, alpha_x * alpha_y is invariant and equals roughness^4.
  // This makes the isotropic and anisotropic singular predicates identical.
  const auto anisotropy = clamp(closure.anisotropy, 0.0f, 1.0f);
  const auto active = anisotropy > 0.0f;
  $if(active) {
    const auto aspect = sqrt(1.0f - 0.9f * anisotropy);
    state.alpha_x = clamp(state.alpha_x / aspect, 0.0f, 1.0f);
    state.alpha_y = clamp(state.alpha_y * aspect, 0.0f, 1.0f);
    state.tangent = rotated_tangent(closure.tangent, authored_normal,
                                    closure.anisotropic_rotation);
  };
  return state;
}

} // namespace

MicrofacetAnisotropyState
principled_microfacet_state(const TracedClosure &closure,
                            Float3 authored_normal) noexcept {
  return aspect_ratio_microfacet_state(closure, authored_normal);
}

MicrofacetAnisotropyState
metallic_microfacet_state(const TracedClosure &closure,
                          Float3 authored_normal) noexcept {
  return aspect_ratio_microfacet_state(closure, authored_normal);
}

MicrofacetAnisotropyState
glossy_microfacet_state(const TracedClosure &closure,
                        Float3 physical_normal) noexcept {
  auto state = isotropic_microfacet_state(closure.roughness);
  if (!closure.anisotropy_enabled) {
    return state;
  }

  // Standalone Glossy uses a signed reciprocal-axis parameterization. The
  // product is again invariant before saturation, so anisotropy changes the
  // distribution shape without changing its geometric-mean roughness.
  const auto anisotropy = clamp(closure.anisotropy, -0.99f, 0.99f);
  const auto active = abs(anisotropy) > 1.0e-4f;
  $if(active) {
    const auto negative = anisotropy < 0.0f;
    const auto negative_denominator = 1.0f + anisotropy;
    const auto positive_denominator = 1.0f - anisotropy;
    const auto anisotropic_alpha_x =
        select(state.alpha_x * positive_denominator,
               state.alpha_x / negative_denominator, negative);
    const auto anisotropic_alpha_y =
        select(state.alpha_y / positive_denominator,
               state.alpha_y * negative_denominator, negative);
    state.alpha_x = clamp(anisotropic_alpha_x, 0.0f, 1.0f);
    state.alpha_y = clamp(anisotropic_alpha_y, 0.0f, 1.0f);
    state.tangent = rotated_tangent(closure.tangent, physical_normal,
                                    closure.anisotropic_rotation);
  };
  return state;
}

void configure_microfacet_state(
    TracedClosure &closure, const MicrofacetAnisotropyState &state) noexcept {
  closure.microfacet_tangent = state.tangent;
  closure.microfacet_alpha_x = state.alpha_x;
  closure.microfacet_alpha_y = state.alpha_y;
  closure.microfacet_state_configured = true;
}

} // namespace psycles::luisa_backend::detail

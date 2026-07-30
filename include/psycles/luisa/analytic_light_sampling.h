#pragma once

#if !defined(PSYCLES_WITH_LUISA)
#error "Include <psycles/luisa/analytic_light_sampling.h> through the Psycles::luisa target."
#endif

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::analytic_light_sampling {

inline constexpr float pi = 3.1415926535897932f;
inline constexpr float inverse_pi = 0.3183098861837907f;

// Cycles represents a zero-radius point as a unit-area delta emitter for the
// area-to-solid-angle conversion. Keeping the inverse-square term in the PDF,
// rather than pre-dividing radiance, gives one common LightSample contract for
// delta and finite oriented-disk point lights:
//
//   pdf_omega = pdf_area * distance^2 / abs(Ng . -D).
//
// For a delta point Ng == -D, pdf_area and the cosine are both one, so the
// conditional PDF is exactly distance^2.
[[nodiscard]] inline luisa::compute::Float
point_disk_pdf(luisa::compute::Float distance_squared,
               luisa::compute::Float light_cosine,
               luisa::compute::Float radius) noexcept {
    const auto radius_squared = radius * radius;
    const auto inverse_area = luisa::compute::select(
        1.0f,
        1.0f /
            luisa::compute::max(
                radius_squared * pi,
                1.0e-20f),
        radius_squared > 0.0f);
    return inverse_area * distance_squared /
           luisa::compute::max(light_cosine, 1.0e-20f);
}

// Cycles converts point-light power to radiance/intensity using
// eval_fac = invarea / pi. A zero-radius normalized point has the explicit
// surrogate area four and therefore eval_fac = 1 / (4 pi). A finite point
// uses its spherical area, even when its sampled geometry is an oriented
// disk; this is the normalization authored by the source light.
[[nodiscard]] inline luisa::compute::Float
point_eval_factor(luisa::compute::Float radius,
                  luisa::compute::Bool normalize_power) noexcept {
    const auto radius_squared = radius * radius;
    const auto finite_area =
        4.0f * pi * radius_squared;
    const auto area = luisa::compute::select(
        4.0f,
        finite_area,
        radius_squared > 0.0f);
    const auto inverse_area = luisa::compute::select(
        1.0f,
        1.0f /
            luisa::compute::max(area, 1.0e-20f),
        normalize_power);
    return inverse_area * inverse_pi;
}

// A finite point/spot can be intersected by the competing forward-BSDF
// technique. A zero-radius delta cannot, regardless of the Blender use-MIS
// property. This is a property of the sampling measures, not a scene-specific
// exception.
[[nodiscard]] inline luisa::compute::Bool
point_has_competing_bsdf_technique(
    luisa::compute::Float radius,
    luisa::compute::Bool use_mis) noexcept {
    return use_mis & (radius > 0.0f);
}

}// namespace psycles::luisa_backend::analytic_light_sampling

#include "hair_closure_scattering.h"

#include "surface_math.h"
#include "surface_math_constants.h"

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {
namespace {

inline constexpr auto half_pi = 0.5f * pi;

[[nodiscard]] Float3 perpendicular_fallback(Float3 tangent) noexcept {
    const auto axis = select(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        abs(tangent.x) > 0.9f);
    return safe_normalize(
        cross(tangent, axis), make_float3(0.0f, 0.0f, 1.0f));
}

struct HairFrame {
    Float3 tangent;
    Float3 y;
    Float3 x;
    Float theta_r;
};

[[nodiscard]] HairFrame make_hair_frame(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming) noexcept {
    const auto tangent = closure.payload.tangent;
    const auto iz = clamp(dot(tangent, incoming), -1.0f, 1.0f);
    const auto y = safe_normalize(
        incoming - tangent * iz,
        perpendicular_fallback(tangent));
    return {
        .tangent = tangent,
        .y = y,
        .x = cross(y, tangent),
        .theta_r = half_pi - acos(iz)};
}

struct HairLongitudinalBounds {
    Float a;
    Float b;
};

[[nodiscard]] HairLongitudinalBounds longitudinal_bounds(
    Float theta_r,
    Float offset,
    Float roughness_u) noexcept {
    const auto inverse_roughness = 1.0f / roughness_u;
    return {
        .a = atan2(
            ((half_pi + theta_r) * 0.5f - offset) *
                inverse_roughness,
            1.0f),
        .b = atan2(
            ((-half_pi + theta_r) * 0.5f - offset) *
                inverse_roughness,
            1.0f)};
}

[[nodiscard]] Float longitudinal_pdf(
    Float theta_i,
    Float theta_r,
    Float offset,
    Float roughness_u,
    HairLongitudinalBounds bounds) noexcept {
    const auto t = (theta_i + theta_r) * 0.5f - offset;
    return roughness_u /
           (2.0f * (t * t + roughness_u * roughness_u) *
            (bounds.a - bounds.b) * cos(theta_i));
}

[[nodiscard]] HairClosureSample sample_hair(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float2 random,
    bool reflection) noexcept {
    const auto frame = make_hair_frame(closure, incoming);
    const auto bounds = longitudinal_bounds(
        frame.theta_r,
        closure.payload.offset,
        closure.payload.roughness_u);
    const auto t = closure.payload.roughness_u *
                   tan(random.x * (bounds.a - bounds.b) + bounds.b);
    const auto theta_i =
        2.0f * (t + closure.payload.offset) - frame.theta_r;
    const auto cosine_theta = cos(theta_i);
    const auto sine_theta = sin(theta_i);

    Float phi;
    if (reflection) {
        phi = 2.0f * asin(clamp(1.0f - 2.0f * random.y, -1.0f, 1.0f)) *
              closure.payload.roughness_v;
    } else {
        const auto c = 2.0f * atan2(
            half_pi / closure.payload.roughness_v, 1.0f);
        const auto p = closure.payload.roughness_v *
                       tan(c * (random.y - 0.5f));
        phi = p + pi;
    }
    const auto direction =
        cos(phi) * cosine_theta * frame.y -
        sin(phi) * cosine_theta * frame.x +
        sine_theta * frame.tangent;
    return {
        .direction = direction,
        .roughness = make_float2(
            closure.payload.roughness_u,
            closure.payload.roughness_v),
        .valid = half_pi - abs(theta_i) >= 0.001f};
}

}// namespace

HairClosureEvaluation evaluate_hair_reflection(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float3 outgoing) noexcept {
    const auto frame = make_hair_frame(closure, incoming);
    const auto outgoing_z = clamp(
        dot(frame.tangent, outgoing), -1.0f, 1.0f);
    const auto outgoing_y = safe_normalize(
        outgoing - frame.tangent * outgoing_z, frame.y);
    const auto theta_i = half_pi - acos(outgoing_z);
    const auto cosine_phi = clamp(dot(outgoing_y, frame.y), -1.0f, 1.0f);
    const auto bounds = longitudinal_bounds(
        frame.theta_r,
        closure.payload.offset,
        closure.payload.roughness_u);
    const auto phi_i = min(
        abs(acos(cosine_phi) / closure.payload.roughness_v), pi);
    const auto phi_pdf =
        cos(phi_i * 0.5f) * 0.25f / closure.payload.roughness_v;
    const auto theta_pdf = longitudinal_pdf(
        theta_i,
        frame.theta_r,
        closure.payload.offset,
        closure.payload.roughness_u,
        bounds);
    const auto valid =
        dot(closure.common.normal, outgoing) >= 0.0f &
        (half_pi - abs(theta_i) >= 0.001f) &
        (cosine_phi >= 0.0f);
    const auto pdf = select(0.0f, phi_pdf * theta_pdf, valid);
    return {.intensity = pdf, .pdf = pdf};
}

HairClosureEvaluation evaluate_hair_transmission(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float3 outgoing) noexcept {
    const auto frame = make_hair_frame(closure, incoming);
    const auto outgoing_z = clamp(
        dot(frame.tangent, outgoing), -1.0f, 1.0f);
    const auto outgoing_y = safe_normalize(
        outgoing - frame.tangent * outgoing_z, frame.y);
    const auto theta_i = half_pi - acos(outgoing_z);
    const auto phi_i = acos(clamp(dot(outgoing_y, frame.y), -1.0f, 1.0f));
    const auto bounds = longitudinal_bounds(
        frame.theta_r,
        closure.payload.offset,
        closure.payload.roughness_u);
    const auto c = 2.0f * atan2(
        half_pi / closure.payload.roughness_v, 1.0f);
    const auto p = pi - abs(phi_i);
    const auto phi_pdf = closure.payload.roughness_v /
                         (c * (p * p +
                               closure.payload.roughness_v *
                                   closure.payload.roughness_v));
    const auto theta_pdf = longitudinal_pdf(
        theta_i,
        frame.theta_r,
        closure.payload.offset,
        closure.payload.roughness_u,
        bounds);
    const auto valid =
        dot(closure.common.normal, outgoing) < 0.0f &
        (half_pi - abs(theta_i) >= 0.001f);
    const auto pdf = select(0.0f, phi_pdf * theta_pdf, valid);
    return {.intensity = pdf, .pdf = pdf};
}

HairClosureSample sample_hair_reflection(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float2 random) noexcept {
    return sample_hair(closure, incoming, random, true);
}

HairClosureSample sample_hair_transmission(
    const SurfaceClosurePhysicalHairRecord &closure,
    Float3 incoming,
    Float2 random) noexcept {
    return sample_hair(closure, incoming, random, false);
}

}// namespace psycles::luisa_backend::detail

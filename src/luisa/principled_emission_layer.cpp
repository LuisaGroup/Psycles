#include "principled_emission_layer.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Float3 apply_layer_albedo(
    Float3 lower_weight,
    Float3 layer_albedo,
    Bool active) noexcept {
    const auto relative_albedo = safe_divide_components(
        layer_albedo, lower_weight);
    const auto remaining = clamp(
        1.0f - max_component(relative_albedo),
        0.0f,
        1.0f);
    return select(
        lower_weight,
        lower_weight * remaining,
        active);
}

[[nodiscard]] Float3 color_power(
    Float3 value,
    Float exponent) noexcept {
    return make_float3(
        pow(value.x, exponent),
        pow(value.y, exponent),
        pow(value.z, exponent));
}

// These mirror Cycles' zero-only normalization contracts. The general graph
// helper deliberately rejects near-zero vectors as well, which is useful for
// user-facing normal inputs but would change the exact Principled layer
// relation at its degenerate boundary.
[[nodiscard]] Float3 cycles_safe_normalize(
    Float3 value) noexcept {
    const auto length = sqrt(dot(value, value));
    const auto denominator = select(
        1.0f, length, length != 0.0f);
    return value / denominator;
}

[[nodiscard]] Float3 cycles_safe_normalize_fallback(
    Float3 value,
    Float3 fallback) noexcept {
    const auto length = sqrt(dot(value, value));
    const auto denominator = select(
        1.0f, length, length != 0.0f);
    return select(
        fallback,
        value / denominator,
        length != 0.0f);
}

}// namespace

[[nodiscard]] PrincipledAlphaLayerResult
evaluate_principled_alpha_layer(
    const TracedClosure &closure) noexcept {
    const auto alpha = clamp(closure.alpha, 0.0f, 1.0f);
    return {
        .lower_weight = closure.weight * alpha,
        .transparency = transparent_closure_state(
            closure.weight * (1.0f - alpha))};
}

PrincipledEmissionLayerComponent::PrincipledEmissionLayerComponent(
    const ShaderServices &services,
    const SurfacePoint &point,
    Bool reflective_caustics) noexcept
    : _services{services},
      _point{point},
      _reflective_caustics{reflective_caustics} {}

PrincipledEmissionLayerResult
PrincipledEmissionLayerComponent::evaluate(
    const TracedClosure &closure) const noexcept {
    auto lower_weight =
        evaluate_principled_alpha_layer(closure).lower_weight;
    const auto incoming = _point.incoming;

    const auto coat_weight = max(closure.coat_weight, 0.0f);
    const auto coat_normal_input = closure.coat_normal_linked
                                       ? closure.coat_normal
                                       : closure.normal;
    const auto coat_normal_fallback = closure.coat_normal_linked
                                          ? _point.shading_normal
                                          : closure.normal;
    const auto coat_normal = cycles_safe_normalize_fallback(
        coat_normal_input, coat_normal_fallback);

    const auto sheen_weight = max(closure.sheen_weight, 0.0f);
    const auto sheen_requested =
        sheen_weight > cycles_closure::closure_weight_cutoff;
    const auto sheen_tint = max(
        closure.sheen_tint, make_float3(0.0f));
    const auto sheen_normal = cycles_safe_normalize(
        lerp(closure.normal,
            coat_normal,
            clamp(coat_weight, 0.0f, 1.0f)));
    const auto sheen_roughness = clamp(
        closure.sheen_roughness, 1.0e-3f, 1.0f);
    const auto sheen_cosine = dot(sheen_normal, incoming);
    const auto sheen_transform_a = cycles_table_2d(
        _services,
        sheen_cosine,
        sheen_roughness,
        UInt{cycles45_tables::sheen_ltc_offset},
        32u,
        32u);
    const auto sheen_albedo = cycles_table_2d(
        _services,
        sheen_cosine,
        sheen_roughness,
        UInt{cycles45_tables::sheen_ltc_offset + 2u * 32u * 32u},
        32u,
        32u);
    const auto sheen_pre_weight =
        lower_weight * sheen_tint * sheen_weight;
    const auto sheen_allocated =
        sample_weight(max(
            sheen_pre_weight,
            make_float3(0.0f))) >=
        cycles_closure::closure_weight_cutoff;
    const auto sheen_valid =
        sheen_requested & sheen_allocated &
        (abs(sheen_transform_a) >= 1.0e-5f) &
        (sheen_albedo >= 1.0e-5f);
    lower_weight = apply_layer_albedo(
        lower_weight,
        sheen_pre_weight * sheen_albedo,
        sheen_valid);

    const auto coat_requested =
        coat_weight > cycles_closure::closure_weight_cutoff;
    const auto coat_roughness = clamp(
        closure.coat_roughness, 0.0f, 1.0f);
    const auto coat_ior = max(closure.coat_ior, 1.0f);
    const auto valid_coat_normal =
        ensure_valid_specular_reflection(
            _point.geometric_normal,
            incoming,
            coat_normal);
    const auto coat_cosine = dot(incoming, valid_coat_normal);
    const auto coat_pre_weight = lower_weight * coat_weight;
    const auto coat_allocated =
        sample_weight(max(
            coat_pre_weight,
            make_float3(0.0f))) >=
        cycles_closure::closure_weight_cutoff;
    const auto coat_reflection_active =
        coat_requested & _reflective_caustics & coat_allocated;

    auto coat_closure = closure;
    coat_closure.roughness = coat_roughness;
    coat_closure.preserve_ggx_energy = true;
    const auto coat_energy = ggx_energy(
        _services,
        coat_closure,
        coat_cosine,
        make_float3(fresnel_dielectric_fss(coat_ior)));
    const auto coat_z = sqrt(abs(
        (coat_ior - 1.0f) /
        max(coat_ior + 1.0f, 1.0e-20f)));
    const auto coat_s = cycles_table_3d(
        _services,
        coat_roughness,
        coat_cosine,
        coat_z,
        UInt{cycles45_tables::ggx_gen_schlick_ior_s_offset},
        16u,
        16u,
        16u);
    const auto coat_fresnel = select(
        0.0f,
        lerp(f0_from_ior(coat_ior), 1.0f, coat_s),
        coat_ior > 1.0f);
    lower_weight = apply_layer_albedo(
        lower_weight,
        coat_pre_weight * coat_energy.darkening * coat_fresnel,
        coat_reflection_active);

    const auto coat_tint = max(
        closure.coat_tint, make_float3(0.0f));
    const auto coat_tint_active =
        coat_requested &
        any(coat_tint != make_float3(1.0f));
    const auto coat_cosine_squared = coat_cosine * coat_cosine;
    const auto coat_transmitted_cosine = sqrt(max(
        1.0f -
            (1.0f / (coat_ior * coat_ior)) *
                (1.0f - coat_cosine_squared),
        0.0f));
    const auto optical_depth =
        1.0f / coat_transmitted_cosine;
    const auto coat_transmission = color_power(
        coat_tint, optical_depth);
    const auto tinted_weight = lower_weight * lerp(
        make_float3(1.0f),
        coat_transmission,
        coat_weight);
    lower_weight = select(
        lower_weight,
        tinted_weight,
        coat_tint_active);

    return {
        .lower_weight = lower_weight,
        .radiance = closure.emission * lower_weight};
}

}// namespace psycles::luisa_backend::detail

#include "principled_layer_component.h"

#include "sheen_closure_component.h"

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

PrincipledLayerComponent::PrincipledLayerComponent(
    const ShaderServices &services,
    const SurfacePoint &point) noexcept
    : _services{services},
      _point{point} {}

PrincipledSheenLayerResult
PrincipledLayerComponent::evaluate_sheen(
    const TracedClosure &closure,
    Float3 lower_weight) const noexcept {
    const auto incoming = _point.incoming;
    const auto coat_weight = max(closure.coat_weight, 0.0f);
    // ShaderGraph::default_inputs() connects every unlinked LINK_NORMAL
    // socket independently to Geometry.Normal. In particular, an authored
    // Principled Normal link must not become the implicit Coat Normal. A
    // linked zero Coat Normal still falls back to ShaderData::N.
    const auto coat_normal_input = closure.coat_normal_linked
                                       ? closure.coat_normal
                                       : _point.shading_normal;
    const auto coat_normal_fallback = _point.shading_normal;
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
    const auto ltc = evaluate_sheen_ltc(
        _services,
        sheen_normal,
        incoming,
        closure.sheen_roughness);
    const auto sheen_pre_weight = max(
        lower_weight * sheen_tint * sheen_weight,
        make_float3(0.0f));
    const auto sheen_allocation_weight =
        sample_weight(sheen_pre_weight);
    const auto sheen_allocated = sheen_allocation_weight >=
                                 cycles_closure::closure_weight_cutoff;
    const auto sheen_slot_allocated =
        sheen_requested & sheen_allocated;
    const auto sheen_valid =
        sheen_slot_allocated &
        ltc.valid;
    const auto sheen_final_weight =
        sheen_pre_weight * ltc.albedo;
    lower_weight = apply_layer_albedo(
        lower_weight,
        sheen_final_weight,
        sheen_valid);

    auto physical = closure;
    physical.principled_lobe = PrincipledLobe::sheen;
    physical.weight = select(make_float3(0.0f),
        select(sheen_pre_weight, sheen_final_weight, sheen_valid),
        sheen_slot_allocated);
    physical.allocation_weight = select(
        0.0f, sheen_allocation_weight, sheen_slot_allocated);
    physical.sample_weight = select(0.0f,
        sheen_allocation_weight * ltc.albedo,
        sheen_valid);
    physical.setup_valid = sheen_valid;
    physical.albedo = select(
        make_float3(0.0f), sheen_final_weight, sheen_valid);
    physical.color = sheen_tint;
    physical.normal = sheen_normal;
    physical.roughness = ltc.roughness;
    physical.ior = 1.0f;
    physical.sheen_transform_a = ltc.transform_a;
    physical.sheen_transform_b = ltc.transform_b;
    physical.evaluation_scale = make_float3(1.0f);
    physical.preserve_ggx_energy = false;
    physical.beckmann = false;
    set_cycles_closure_identity_after_setup(
        physical, cycles_closure::type_sheen);
    return {.closure = physical, .lower_weight = lower_weight};
}

PrincipledCoatLayerResult
PrincipledLayerComponent::evaluate_coat(
    const TracedClosure &closure,
    Float3 lower_weight,
    Bool reflective_caustics) const noexcept {
    const auto incoming = _point.incoming;
    const auto coat_weight = max(closure.coat_weight, 0.0f);
    const auto coat_requested =
        coat_weight > cycles_closure::closure_weight_cutoff;
    const auto coat_roughness = clamp(
        closure.coat_roughness, 0.0f, 1.0f);
    const auto coat_ior = max(closure.coat_ior, 1.0f);
    const auto coat_normal_input = closure.coat_normal_linked
                                       ? closure.coat_normal
                                       : _point.shading_normal;
    const auto coat_normal_fallback = _point.shading_normal;
    const auto coat_normal = cycles_safe_normalize_fallback(
        coat_normal_input, coat_normal_fallback);
    const auto valid_coat_normal =
        maybe_ensure_valid_specular_reflection(
            _point, incoming, coat_normal);
    const auto coat_cosine = dot(incoming, valid_coat_normal);

    // bsdf_alloc() clamps the closure weight before storing it. Keep that
    // allocation state separate from the later Fresnel/sample-weight setup:
    // an IOR-one Coat still occupies a Cycles closure slot with zero sampling
    // weight.
    const auto coat_pre_weight = max(
        lower_weight * coat_weight, make_float3(0.0f));
    const auto coat_allocation_weight = sample_weight(coat_pre_weight);
    const auto coat_allocated =
        coat_requested & reflective_caustics &
        (coat_allocation_weight >=
            cycles_closure::closure_weight_cutoff);

    auto energy_input = closure;
    energy_input.roughness = coat_roughness;
    energy_input.preserve_ggx_energy = true;
    const auto coat_energy = ggx_energy(
        _services,
        energy_input,
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
    const auto coat_albedo_estimate = make_float3(coat_fresnel);
    const auto coat_final_weight =
        coat_pre_weight * coat_energy.darkening;
    const auto coat_albedo =
        coat_final_weight * coat_albedo_estimate;
    lower_weight = apply_layer_albedo(
        lower_weight, coat_albedo, coat_allocated);

    // Cycles applies absorption through the Coat medium independently of
    // reflective-caustic closure allocation. The authored weight is not
    // saturated here: SVM uses the non-negative raw socket value.
    const auto coat_tint = max(
        closure.coat_tint, make_float3(0.0f));
    const auto coat_tint_active =
        coat_requested & any(coat_tint != make_float3(1.0f));
    const auto coat_cosine_squared = coat_cosine * coat_cosine;
    const auto coat_transmitted_cosine = sqrt(max(
        1.0f -
            (1.0f / (coat_ior * coat_ior)) *
                (1.0f - coat_cosine_squared),
        0.0f));
    const auto optical_depth = 1.0f / coat_transmitted_cosine;
    const auto coat_transmission = color_power(
        coat_tint, optical_depth);
    lower_weight = select(
        lower_weight,
        lower_weight * lerp(make_float3(1.0f),
                           coat_transmission,
                           coat_weight),
        coat_tint_active);

    auto physical = closure;
    physical.principled_lobe = PrincipledLobe::coat;
    physical.weight = select(
        make_float3(0.0f), coat_final_weight, coat_allocated);
    physical.allocation_weight = select(
        0.0f, coat_allocation_weight, coat_allocated);
    physical.sample_weight = select(
        0.0f,
        coat_allocation_weight *
            sample_weight(coat_albedo_estimate) *
            sample_weight(coat_energy.darkening),
        coat_allocated);
    physical.setup_valid = true;
    physical.albedo = select(
        make_float3(0.0f), coat_albedo, coat_allocated);
    physical.color = make_float3(f0_from_ior(coat_ior));
    physical.normal = valid_coat_normal;
    physical.roughness = coat_roughness;
    physical.ior = coat_ior;
    physical.evaluation_scale = coat_energy.energy_scale;
    physical.preserve_ggx_energy = true;
    physical.beckmann = false;
    set_cycles_closure_identity_after_setup(
        physical,
        cycles_closure::type_microfacet_ggx,
        static_cast<std::uint32_t>(
            cycles_closure::MicrofacetFresnel::dielectric));
    return {.closure = physical, .lower_weight = lower_weight};
}

PrincipledEmissionLayerResult
PrincipledLayerComponent::evaluate_emission(
    const TracedClosure &closure,
    compiler::PrincipledClosureFeatureMask features,
    Bool reflective_caustics) const noexcept {
    const auto enabled = [features](
                             compiler::PrincipledClosureFeature feature) noexcept {
        return (features &
                compiler::principled_closure_feature_bit(feature)) != 0u;
    };
    auto lower_weight =
        enabled(compiler::PrincipledClosureFeature::alpha)
            ? evaluate_principled_alpha_layer(closure).lower_weight
            : closure.weight;
    if (enabled(compiler::PrincipledClosureFeature::sheen)) {
        lower_weight =
            evaluate_sheen(closure, lower_weight).lower_weight;
    }
    if (enabled(compiler::PrincipledClosureFeature::coat)) {
        lower_weight = evaluate_coat(
                           closure,
                           lower_weight,
                           reflective_caustics)
                           .lower_weight;
    }

    return {
        .lower_weight = lower_weight,
        .radiance = closure.emission * lower_weight};
}

}// namespace psycles::luisa_backend::detail

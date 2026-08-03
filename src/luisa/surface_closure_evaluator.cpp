#include <psycles/luisa/surface_closure_evaluator.h>

#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

[[nodiscard]] Bool has_kind(
    const SurfaceClosureRecord &closure,
    SurfaceClosureKind kind) noexcept {
    return closure.kind ==
           static_cast<std::uint32_t>(kind);
}

[[nodiscard]] Bool has_lobe(
    const SurfaceClosureRecord &closure,
    SurfaceClosureLobe lobe) noexcept {
    return closure.lobe ==
           static_cast<std::uint32_t>(lobe);
}

[[nodiscard]] UInt runtime_flags_for(
    const SurfaceClosureRecord &closure,
    Float glossy_filter_roughness) noexcept {
    const auto is_diffuse = has_kind(
        closure, SurfaceClosureKind::diffuse);
    const auto is_translucent = has_kind(
        closure, SurfaceClosureKind::translucent);
    const auto is_principled = has_kind(
        closure, SurfaceClosureKind::principled);
    const auto is_sheen =
        is_principled &
        has_lobe(closure, SurfaceClosureLobe::sheen);
    const auto is_glossy = has_kind(
        closure, SurfaceClosureKind::glossy);
    const auto is_glass = has_kind(
        closure, SurfaceClosureKind::glass);
    const auto is_transparent = has_kind(
        closure, SurfaceClosureKind::transparent);

    const auto bsdf =
        cycles_closure::runtime_bsdf;
    const auto has_eval =
        cycles_closure::runtime_bsdf_has_eval;
    UInt flags = 0u;
    flags = select(flags,
        bsdf | has_eval,
        is_diffuse);
    flags = select(flags,
        bsdf | has_eval |
            cycles_closure::runtime_bsdf_has_transmission,
        is_translucent);
    flags = select(flags,
        bsdf | has_eval,
        is_sheen);

    auto alpha = clamp(closure.roughness, 0.0f, 1.0f);
    alpha *= alpha;
    alpha = max(alpha, glossy_filter_roughness);
    const auto regular_microfacet =
        alpha * alpha >
        cycles_closure::microfacet_singular_alpha_product;
    const auto microfacet_flags =
        bsdf | select(0u, has_eval, regular_microfacet);
    flags = select(flags,
        microfacet_flags,
        (is_principled & !is_sheen) | is_glossy);
    flags = select(flags,
        microfacet_flags |
            cycles_closure::runtime_bsdf_has_transmission,
        is_glass);
    flags = select(flags,
        bsdf | cycles_closure::runtime_transparent,
        is_transparent);
    flags |= select(0u,
        has_eval,
        glossy_filter_roughness * glossy_filter_roughness >
            cycles_closure::microfacet_singular_alpha_product);
    return select(0u, flags, closure.setup_valid);
}

[[nodiscard]] UInt closure_type_for(
    const SurfaceClosureRecord &closure) noexcept {
    const auto is_diffuse = has_kind(
        closure, SurfaceClosureKind::diffuse);
    const auto is_translucent = has_kind(
        closure, SurfaceClosureKind::translucent);
    const auto is_principled = has_kind(
        closure, SurfaceClosureKind::principled);
    const auto is_sheen =
        is_principled &
        has_lobe(closure, SurfaceClosureLobe::sheen);
    const auto is_glossy = has_kind(
        closure, SurfaceClosureKind::glossy);
    const auto is_glass = has_kind(
        closure, SurfaceClosureKind::glass);
    const auto is_transparent = has_kind(
        closure, SurfaceClosureKind::transparent);

    UInt type = cycles_closure::type_none;
    type = select(type,
        select(
            UInt{cycles_closure::type_oren_nayar},
            UInt{cycles_closure::type_diffuse},
            closure.roughness < 1.0e-5f),
        is_diffuse);
    type = select(type,
        UInt{cycles_closure::type_translucent},
        is_translucent);
    type = select(type,
        UInt{cycles_closure::type_microfacet_ggx},
        is_glossy | (is_principled & !is_sheen));
    const auto single_glass = select(
        UInt{cycles_closure::type_microfacet_ggx_glass},
        UInt{cycles_closure::type_microfacet_beckmann_glass},
        closure.beckmann);
    const auto glass = select(single_glass,
        UInt{cycles_closure::type_microfacet_multi_ggx_glass},
        closure.preserve_ggx_energy);
    type = select(type, glass, is_glass);
    type = select(type,
        UInt{cycles_closure::type_transparent},
        is_transparent);
    type = select(type,
        UInt{cycles_closure::type_sheen},
        is_sheen);
    return select(
        UInt{cycles_closure::type_none},
        type,
        closure.setup_valid);
}

}// namespace

SurfaceClosureEvaluator::SurfaceClosureEvaluator(
    const SurfacePoint &point,
    const SurfaceClosureSet &closures,
    Float3 shading_normal) noexcept
    : _point{point},
      _closures{closures},
      _shading_normal{shading_normal} {}

UInt SurfaceClosureEvaluator::runtime_flags(
    Float glossy_filter_roughness) const noexcept {
    UInt result = select(0u,
        cycles_closure::runtime_backfacing,
        _point.back_facing);
    UInt index = 0u;
    $while(index < _closures.count()) {
        result |= runtime_flags_for(
            _closures.entry(index),
            glossy_filter_roughness);
        index += 1u;
    };
    return result;
}

SurfaceClosureTrace SurfaceClosureEvaluator::closure_trace(
    UInt requested_index) const noexcept {
    const auto closure = _closures.entry(requested_index);
    const auto valid = requested_index < _closures.count();
    return {
        .count = _closures.count(),
        .runtime_flags = runtime_flags(),
        .index = requested_index,
        .type = closure_type_for(closure),
        .sample_weight = closure.sample_weight,
        .weight = closure.weight,
        .normal = closure.normal,
        .valid = valid};
}

SurfaceAov SurfaceClosureEvaluator::aov() const noexcept {
    auto result = SurfaceAov{
        .albedo = make_float3(0.0f),
        .glossy_albedo = make_float3(0.0f),
        .transmission_albedo = make_float3(0.0f),
        .roughness = make_float2(0.0f),
        .normal = _point.shading_normal,
        .transparency = make_float3(0.0f)};
    Float total_weight = 0.0f;
    Float roughness_weight = 0.0f;
    Float roughness = 0.0f;
    Float3 normal = make_float3(0.0f);
    const auto incoming = detail::safe_normalize(
        _point.incoming, _point.shading_normal);
    UInt index = 0u;
    $while(index < _closures.count()) {
        const auto closure = _closures.entry(index);
        const auto is_transparent = has_kind(
            closure, SurfaceClosureKind::transparent);
        const auto is_diffuse = has_kind(
            closure, SurfaceClosureKind::diffuse);
        const auto is_translucent = has_kind(
            closure, SurfaceClosureKind::translucent);
        const auto is_principled = has_kind(
            closure, SurfaceClosureKind::principled);
        const auto is_sheen =
            is_principled &
            has_lobe(closure, SurfaceClosureLobe::sheen);
        const auto is_glossy = has_kind(
            closure, SurfaceClosureKind::glossy);
        const auto is_glass = has_kind(
            closure, SurfaceClosureKind::glass);
        const auto generic_glossy =
            (is_principled & !is_sheen) | is_glossy;
        const auto glossy_normal = select(
            detail::maybe_ensure_valid_specular_reflection(
                _point, incoming, closure.normal),
            closure.normal,
            is_sheen);

        result.transparency += select(
            make_float3(0.0f),
            closure.weight,
            is_transparent);
        result.glossy_albedo += select(
            make_float3(0.0f),
            closure.reflection_albedo,
            is_glass);
        result.transmission_albedo += select(
            make_float3(0.0f),
            closure.transmission_albedo,
            is_glass);
        result.glossy_albedo += select(
            make_float3(0.0f),
            closure.albedo,
            generic_glossy);

        const auto diffuse_family =
            is_diffuse | is_translucent;
        const auto diffuse_albedo = select(
            select(
                make_float3(0.0f),
                closure.albedo,
                diffuse_family),
            select(
                make_float3(0.0f),
                closure.albedo,
                closure.setup_valid),
            is_sheen);
        const auto closure_pass_weight =
            detail::pass_weight(closure.weight);
        const auto diffuse_weight = select(
            select(0.0f,
                closure_pass_weight,
                diffuse_family),
            select(0.0f,
                closure_pass_weight,
                closure.setup_valid),
            is_sheen);
        const auto glossy_weight = select(0.0f,
            closure_pass_weight,
            is_glass | generic_glossy);
        const auto weight =
            diffuse_weight + glossy_weight;
        total_weight += weight;
        roughness_weight += glossy_weight;
        result.albedo += diffuse_albedo;
        roughness +=
            glossy_weight * closure.roughness;
        normal +=
            diffuse_weight * select(
                                 closure.normal,
                                 glossy_normal,
                                 is_translucent) +
            glossy_weight * glossy_normal;
        index += 1u;
    };
    const auto valid = total_weight > 0.0f;
    result.roughness = make_float2(select(
        1.0f,
        roughness /
            max(roughness_weight, 1.0e-20f),
        roughness_weight > 0.0f));
    result.normal = detail::safe_normalize(
        select(
            _point.shading_normal,
            normal,
            valid),
        _point.shading_normal);
    return result;
}

}// namespace psycles::luisa_backend

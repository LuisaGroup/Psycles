#include <psycles/luisa/surface_closure_evaluator.h>

#include "graph_surface_internal.h"
#include "microfacet_albedo.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface_closure_evaluation.h>
#include <psycles/luisa/surface_closure_operations.h>

#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

struct PhysicalDielectricAov {
    Float3 reflection;
    Float3 transmission;
};

[[nodiscard]] bool reflection_microfacet_reachable(
    SurfaceClosureReachability reachability) noexcept {
    return reachability.contains(SurfaceClosureKind::principled) ||
           reachability.contains(SurfaceClosureKind::glossy) ||
           reachability.contains(SurfaceClosureKind::metallic_f82) ||
           reachability.contains(SurfaceClosureKind::metallic_conductor);
}

[[nodiscard]] Float3 physical_reflection_microfacet_albedo(
    const ShaderServices &services,
    const SurfaceClosurePhysicalGeneralRecord &closure,
    Float3 incoming,
    SurfaceClosureReachability reachability) noexcept {
    using Fresnel = cycles_closure::MicrofacetFresnel;
    const auto alpha_x = closure.payload.microfacet_alpha_x;
    const auto alpha_y = closure.payload.microfacet_alpha_y;
    const auto cosine = dot(incoming, closure.common.normal);
    const UInt fresnel{closure.common.microfacet_fresnel};
    Float3 result = make_float3(1.0f);

    if (reachability.contains_principled_lobe(
            SurfaceClosureLobe::coat)) {
        const auto table = detail::microfacet_dielectric_reflection_albedo(
            services,
            incoming,
            closure.common.normal,
            alpha_x,
            alpha_y,
            closure.payload.ior);
        const auto directional = make_float3(
            detail::fresnel_dielectric_cos(
                cosine, closure.payload.ior));
        const auto dielectric = select(
            directional,
            make_float3(table),
            closure.payload.ior > 1.0f);
        result = select(
            result,
            dielectric,
            fresnel == static_cast<std::uint32_t>(Fresnel::dielectric));
    }

    if (reachability.contains_principled_lobe(
            SurfaceClosureLobe::dielectric)) {
        const auto may_have_thin_film =
            reachability.contains_thin_film_principled_lobe(
                SurfaceClosureLobe::dielectric);
        const auto generalized =
            detail::microfacet_generalized_schlick_reflectance(
                services,
                incoming,
                closure.common.normal,
                alpha_x,
                alpha_y,
                closure.payload.ior,
                closure.payload.thin_film_thickness,
                closure.payload.thin_film_ior,
                closure.common.color_or_evaluation_scale,
                make_float3(1.0f),
                -1.0f,
                may_have_thin_film);
        result = select(
            result,
            generalized,
            fresnel == static_cast<std::uint32_t>(
                           Fresnel::generalized_schlick));
    }

    const auto may_have_f82 =
        reachability.contains(SurfaceClosureKind::metallic_f82) ||
        reachability.contains_principled_lobe(
            SurfaceClosureLobe::metallic);
    if (may_have_f82) {
        const auto may_have_thin_film =
            reachability.contains_thin_film(
                SurfaceClosureKind::metallic_f82) ||
            reachability.contains_thin_film_principled_lobe(
                SurfaceClosureLobe::metallic);
        const auto f82 = detail::microfacet_f82_tint_albedo(
            services,
            incoming,
            closure.common.normal,
            alpha_x,
            alpha_y,
            closure.payload.thin_film_thickness,
            closure.payload.thin_film_ior,
            closure.common.color_or_evaluation_scale,
            closure.payload.specular_tint,
            may_have_thin_film);
        result = select(
            result,
            f82,
            fresnel == static_cast<std::uint32_t>(Fresnel::f82_tint));
    }

    if (reachability.contains(
            SurfaceClosureKind::metallic_conductor)) {
        const auto conductor = detail::microfacet_conductor_albedo(
            services,
            incoming,
            closure.common.normal,
            closure.payload.thin_film_thickness,
            closure.payload.thin_film_ior,
            closure.common.color_or_evaluation_scale,
            closure.payload.specular_tint,
            reachability.contains_thin_film(
                SurfaceClosureKind::metallic_conductor));
        result = select(
            result,
            conductor,
            fresnel == static_cast<std::uint32_t>(Fresnel::conductor));
    }
    return result;
}

[[nodiscard]] PhysicalDielectricAov physical_dielectric_aov(
    const ShaderServices &services,
    const SurfaceClosurePhysicalDielectricRecord &closure,
    Float3 incoming,
    SurfaceClosureReachability reachability) noexcept {
    PhysicalDielectricAov result{
        .reflection = make_float3(0.0f),
        .transmission = make_float3(0.0f)};
    if (reachability.contains(SurfaceClosureKind::glass)) {
        $if(cycles_closure::is_glass_microfacet(
            closure.common.closure_type)) {
            auto alpha = clamp(closure.common.roughness, 0.0f, 1.0f);
            alpha *= alpha;
            const auto reflectance =
                detail::microfacet_generalized_schlick_reflectance(
                    services,
                    incoming,
                    closure.common.normal,
                    alpha,
                    alpha,
                    closure.payload.ior,
                    closure.payload.thin_film_thickness,
                    closure.payload.thin_film_ior,
                    closure.payload.fresnel_f0,
                    closure.payload.fresnel_f90,
                    -1.0f,
                    reachability.contains_thin_film(
                        SurfaceClosureKind::glass));
            result.reflection =
                reflectance * closure.payload.reflection_tint;
            result.transmission =
                (make_float3(1.0f) - reflectance) *
                closure.payload.transmission_tint;
        };
    }
    if (reachability.contains(SurfaceClosureKind::refraction)) {
        $if(cycles_closure::is_refraction_microfacet(
            closure.common.closure_type)) {
            const auto fresnel = detail::fresnel_dielectric_cos(
                dot(incoming, closure.common.normal),
                closure.payload.ior);
            result.transmission = select(
                make_float3(1.0f),
                make_float3(0.0f),
                fresnel == 1.0f);
        };
    }
    return result;
}

[[nodiscard]] Float physical_isotropic_microfacet_roughness_squared(
    const SurfaceClosurePhysicalCommonRecord &closure,
    Float glossy_filter_roughness) noexcept {
    auto alpha = clamp(closure.roughness, 0.0f, 1.0f);
    alpha *= alpha;
    alpha = max(alpha, glossy_filter_roughness);
    return alpha * alpha;
}

}// namespace

SurfaceClosureEvaluator::SurfaceClosureEvaluator(
    const SurfacePoint &point,
    const SurfaceClosureSet &closures,
    Float3 shading_normal, SurfaceClosureReachability reachability) noexcept
    : _point{point},
      _closures{closures},
      _shading_normal{shading_normal},
      _reachability{reachability} {}

SurfaceClosureEvaluator::SurfaceClosureEvaluator(
    const SurfacePoint &point,
    const SurfaceClosureSet &closures,
    Float3 shading_normal,
    SurfaceClosurePopulationState populated_runtime_state,
    SurfaceClosureReachability reachability) noexcept
    : _point{point},
      _closures{closures},
      _shading_normal{shading_normal},
      _reachability{reachability},
      _populated_runtime_state{
          std::move(populated_runtime_state)} {}

UInt SurfaceClosureEvaluator::runtime_flags(
    Float glossy_filter_roughness) const noexcept {
    UInt result = select(0u,
        cycles_closure::runtime_backfacing,
        _point.back_facing);
    UInt index = 0u;
    if (_closures.profile() ==
        SurfaceClosureStorageProfile::physical) {
        $while(index < _closures.count()) {
            const auto access =
                _closures.physical_access(index);
            const auto common =
                _closures.physical_common_entry(access);
            result |= detail::cycles_runtime_flags(
                common.closure_type,
                common.roughness,
                glossy_filter_roughness,
                _reachability);
            index += 1u;
        };
    } else {
        $while(index < _closures.count()) {
            result |= detail::cycles_runtime_flags(
                _closures.entry(index),
                glossy_filter_roughness,
                _reachability);
            index += 1u;
        };
    }
    return result;
}

SurfaceClosureTrace SurfaceClosureEvaluator::closure_trace(
    UInt requested_index) const noexcept {
    if (_closures.profile() ==
        SurfaceClosureStorageProfile::physical) {
        const auto access =
            _closures.physical_access(requested_index);
        const auto closure =
            _closures.physical_common_entry(access);
        return {
            .count = _closures.count(),
            .runtime_flags = runtime_flags(),
            .index = requested_index,
            .type = select(
                UInt{cycles_closure::type_none},
                closure.closure_type,
                access.valid()),
            .sample_weight = select(
                0.0f, closure.sample_weight, access.valid()),
            .weight = select(
                make_float3(0.0f), closure.weight, access.valid()),
            .normal = select(
                make_float3(0.0f, 0.0f, 1.0f),
                closure.normal,
                access.valid()),
            .valid = access.valid()};
    }
    const auto closure = _closures.entry(requested_index);
    const auto valid = requested_index < _closures.count();
    return {
        .count = _closures.count(),
        .runtime_flags = runtime_flags(),
        .index = requested_index,
        .type = closure.closure_type,
        .sample_weight = closure.sample_weight,
        .weight = closure.weight,
        .normal = closure.normal,
        .valid = valid};
}

SurfaceAov SurfaceClosureEvaluator::aov() const noexcept {
    LUISA_ASSERT(
        _closures.profile() != SurfaceClosureStorageProfile::physical,
        "Physical retained storage omits setup-owned AOV fields; consume "
        "SurfaceClosurePopulationState instead of reconstructing closures.");
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
    UInt index = 0u;
    $while(index < _closures.count()) {
        const auto closure = _closures.entry(index);
        const auto contribution =
            surface_closure_aov_contribution(
                _point, closure);
        result.albedo += contribution.albedo;
        result.glossy_albedo +=
            contribution.glossy_albedo;
        result.transmission_albedo +=
            contribution.transmission_albedo;
        result.transparency += contribution.transparency;
        total_weight += contribution.total_weight;
        roughness_weight += contribution.roughness_weight;
        roughness += contribution.roughness;
        normal += contribution.normal;
        index += 1u;
    };
    result.roughness = make_float2(select(
        1.0f,
        roughness / max(roughness_weight, 1.0e-20f),
        roughness_weight > 0.0f));
    result.normal = detail::safe_normalize(
        select(
            _point.shading_normal,
            normal,
            total_weight > 0.0f),
        _point.shading_normal);
    return result;
}

SurfaceAov SurfaceClosureEvaluator::aov(
    const ShaderServices &services,
    Expr<float> glossy_filter_roughness_expression) const noexcept {
    if (_closures.profile() != SurfaceClosureStorageProfile::physical) {
        return aov();
    }

    auto result = SurfaceAov{
        .albedo = make_float3(0.0f),
        .glossy_albedo = make_float3(0.0f),
        .transmission_albedo = make_float3(0.0f),
        .roughness = make_float2(0.0f),
        .normal = _point.shading_normal,
        .transparency = make_float3(0.0f)};
    const auto incoming = detail::safe_normalize(
        Float3{_point.incoming},
        Float3{_point.shading_normal});
    const Float glossy_filter_roughness{
        glossy_filter_roughness_expression};
    Float3 weighted_normal = make_float3(0.0f);
    Float roughness = 0.0f;
    Float roughness_weight = 0.0f;
    UInt index = 0u;
    $while(index < _closures.count()) {
        const auto access = _closures.physical_access(index);
        const auto common = _closures.physical_common_entry(access);
        const UInt type{common.closure_type};
        const auto pass_weight = detail::pass_weight(common.weight);

        result.albedo += select(
            make_float3(0.0f),
            common.weight,
            cycles_closure::is_bsdf_diffuse(type) |
                cycles_closure::is_bssrdf(type));
        Float3 glossy_albedo = select(
            make_float3(0.0f),
            common.weight,
            cycles_closure::is_bsdf_glossy(type));
        Float3 transmission_albedo = select(
            make_float3(0.0f),
            common.weight,
            cycles_closure::is_bsdf_transmission(type));

        $if(cycles_closure::is_bsdf_or_bssrdf(type)) {
            weighted_normal += common.normal * pass_weight;
        };

        // Exact ordered bsdf_get_roughness_pass_squared() projection. The
        // family payload branches below overwrite the generic value for the
        // represented microfacet types after filter-glossy has widened alpha.
        Float roughness_squared = select(
            1.0f,
            0.0f,
            (type == cycles_closure::type_transparent) |
                (type == cycles_closure::type_ray_portal));
        roughness_squared = select(
            roughness_squared,
            -1.0f,
            cycles_closure::is_bsdf_diffuse(type));
        const auto oren_family =
            (type == cycles_closure::type_oren_nayar) |
            (type == cycles_closure::type_rough_translucent);
        const auto diffuse_roughness_squared =
            common.roughness * common.roughness;
        roughness_squared = select(
            roughness_squared,
            diffuse_roughness_squared * diffuse_roughness_squared,
            oren_family);

        Bool dielectric_payload = false;
        if (_reachability.contains(SurfaceClosureKind::glass) ||
            _reachability.contains(SurfaceClosureKind::refraction)) {
            dielectric_payload =
                surface_closure_uses_dielectric_payload(type);
        }
        Bool reflection_microfacet = false;
        if (reflection_microfacet_reachable(_reachability)) {
            reflection_microfacet =
                cycles_closure::is_reflection_microfacet(type);
        }
        $if(dielectric_payload) {
            const auto closure =
                unpack_surface_closure_physical_dielectric(
                    common,
                    _closures.physical_payload_block(
                        _closures.physical_access(index)));
            const auto aov = physical_dielectric_aov(
                services, closure, incoming, _reachability);
            glossy_albedo = select(
                glossy_albedo,
                common.weight * aov.reflection,
                cycles_closure::is_glass_microfacet(type));
            transmission_albedo =
                common.weight * aov.transmission;
            roughness_squared =
                physical_isotropic_microfacet_roughness_squared(
                    common, glossy_filter_roughness);
        }
        $elif(reflection_microfacet) {
            const auto closure =
                unpack_surface_closure_physical_general(
                    common,
                    _closures.physical_payload_block(
                        _closures.physical_access(index)));
            glossy_albedo =
                common.weight * physical_reflection_microfacet_albedo(
                                    services,
                                    closure,
                                    incoming,
                                    _reachability);
            const auto alpha = max(
                make_float2(
                    closure.payload.microfacet_alpha_x,
                    closure.payload.microfacet_alpha_y),
                make_float2(glossy_filter_roughness));
            roughness_squared = alpha.x * alpha.y;
        }
        $elif(type == cycles_closure::type_thin_glass_transmission) {
            roughness_squared =
                physical_isotropic_microfacet_roughness_squared(
                    common, glossy_filter_roughness);
        };

        result.glossy_albedo += glossy_albedo;
        result.transmission_albedo += transmission_albedo;
        $if(cycles_closure::is_bsdf(type) &
            (roughness_squared >= 0.0f)) {
            roughness +=
                pass_weight * sqrt(sqrt(roughness_squared));
            roughness_weight += pass_weight;
        };
        index += 1u;
    };

    result.roughness = make_float2(select(
        1.0f,
        roughness / roughness_weight,
        roughness_weight > 0.0f));
    result.normal = select(
        _point.shading_normal,
        detail::safe_normalize(
            weighted_normal,
            Float3{_point.shading_normal}),
        any(weighted_normal != make_float3(0.0f)));
    return result;
}

SurfaceEvaluation SurfaceClosureEvaluator::evaluate_impl(
    const ShaderServices &services,
    Float3 outgoing_expression,
    const SurfaceQuery &query,
    EvaluationMode mode,
    UInt light_shader_flags,
    UInt selected_closure_index) const noexcept {
    const SurfaceClosurePoint closure_point{_point};
    const auto directions =
        make_surface_closure_evaluation_directions(
            closure_point,
            Expr<luisa::float3>{
                outgoing_expression.expression()});
    const auto policy = make_surface_closure_evaluation_policy(
        mode == EvaluationMode::sampled_light,
        Expr<std::uint32_t>{
            light_shader_flags.expression()});
    SurfaceClosureEvaluationAccumulator accumulator;
    UInt index = 0u;
    if (_closures.profile() == SurfaceClosureStorageProfile::physical) {
        $while(index < _closures.count()) {
            const auto access = _closures.physical_access(index);
            const auto common = _closures.physical_common_entry(access);
            const auto selected_sample =
                mode == EvaluationMode::sampled_bsdf
                    ? index == selected_closure_index
                    : Bool{false};
            accumulator.add(
                surface_closure_evaluation_contribution_from_physical_common(
                    services,
                    closure_point,
                    Expr<luisa::float3>{_shading_normal.expression()},
                    common,
                    [&] {
                        // Re-establish the initialized-prefix witness in the
                        // payload family's device block. The counted-array
                        // proof deliberately does not transport mutable
                        // counter snapshots across CFG edges; reusing
                        // `access` here would therefore make the dormant
                        // suffix appear live across every coroutine
                        // continuation.
                        return _closures.physical_payload_block(
                            _closures.physical_access(index));
                    },
                    Expr<luisa::float3>{directions.incoming.expression()},
                    Expr<luisa::float3>{directions.outgoing.expression()},
                    query,
                    policy,
                    Expr<bool>{selected_sample.expression()},
                    _reachability));
            index += 1u;
        };
    } else {
        $while(index < _closures.count()) {
            const auto closure = _closures.entry(index);
            const auto selected_sample =
                mode == EvaluationMode::sampled_bsdf
                    ? index == selected_closure_index
                    : Bool{false};
            accumulator.add(surface_closure_evaluation_contribution(
                services,
                closure_point,
                Expr<luisa::float3>{
                    _shading_normal.expression()},
                closure,
                Expr<luisa::float3>{
                    directions.incoming.expression()},
                Expr<luisa::float3>{
                    directions.outgoing.expression()},
                query,
                policy,
                Expr<bool>{selected_sample.expression()}, _reachability));
            index += 1u;
        };
    }
    return accumulator.finish(
        Expr<bool>{policy.preserve_pdf.expression()});
}

SurfaceEvaluation SurfaceClosureEvaluator::evaluate(
    const ShaderServices &services,
    Expr<luisa::float3> outgoing,
    const SurfaceQuery &query) const noexcept {
    return evaluate_impl(services,
        Float3{outgoing},
        query,
        EvaluationMode::regular,
        0u,
        ~std::uint32_t{0u});
}

SurfaceEvaluation SurfaceClosureEvaluator::evaluate_light(
    const ShaderServices &services,
    Expr<luisa::float3> outgoing,
    const SurfaceLightQuery &query) const noexcept {
    return evaluate_impl(services,
        Float3{outgoing},
        query.surface,
        EvaluationMode::sampled_light,
        query.shader_flags,
        ~std::uint32_t{0u});
}

}// namespace psycles::luisa_backend

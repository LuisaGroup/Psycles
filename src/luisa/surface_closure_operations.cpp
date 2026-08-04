#include <psycles/luisa/surface_closure_operations.h>

#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

[[nodiscard]] luisa::compute::UInt2 classify(
    const SurfaceClosureIdentityCallable &identity,
    const SurfaceClosureExpression &closure,
    Expr<float> glossy_filter_roughness) noexcept {
    return identity(
        closure.kind,
        closure.lobe,
        closure.bssrdf_method,
        closure.allocation_weight,
        closure.setup_valid,
        closure.roughness,
        closure.preserve_ggx_energy,
        closure.beckmann,
        glossy_filter_roughness);
}

struct AovClosureExpression {
    Expr<std::uint32_t> kind;
    Expr<std::uint32_t> lobe;
    Expr<luisa::float3> weight;
    Expr<bool> setup_valid;
    Expr<luisa::float3> albedo;
    Expr<luisa::float3> reflection_albedo;
    Expr<luisa::float3> transmission_albedo;
    Expr<luisa::float3> normal;
    Expr<float> roughness;
};

template<typename Closure>
[[nodiscard]] luisa::compute::Var<SurfaceAovContributionCall>
aov_contribution(
    Expr<luisa::float3> incoming_expression,
    Expr<luisa::float3> shading_normal_expression,
    Expr<luisa::float3> geometric_normal_expression,
    Expr<bool> use_bump_map_correction,
    const Closure &closure) noexcept {
    const auto is_transparent =
        closure.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::transparent);
    const auto is_diffuse =
        closure.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::diffuse);
    const auto is_translucent =
        closure.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::translucent);
    const auto is_principled =
        closure.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::principled);
    const auto is_sheen =
        is_principled &
        (closure.lobe == static_cast<std::uint32_t>(
                             SurfaceClosureLobe::sheen));
    const auto is_glossy =
        closure.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::glossy);
    const auto is_glass =
        closure.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::glass);
    const auto is_refraction =
        closure.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::refraction);
    const auto is_bssrdf =
        closure.kind == static_cast<std::uint32_t>(
                            SurfaceClosureKind::bssrdf);
    const auto is_dielectric = is_glass | is_refraction;
    const auto generic_glossy =
        (is_principled & !is_sheen) | is_glossy;

    const auto incoming = detail::safe_normalize(
        Float3{incoming_expression},
        Float3{shading_normal_expression});
    const auto corrected_glossy_normal = select(
        closure.normal,
        detail::ensure_valid_specular_reflection(
            Float3{geometric_normal_expression},
            incoming,
            Float3{closure.normal}),
        Bool{use_bump_map_correction} &
            !all(closure.normal == geometric_normal_expression));
    const auto glossy_normal = select(
        corrected_glossy_normal,
        closure.normal,
        is_sheen);

    const auto diffuse_family =
        is_diffuse | is_translucent | is_bssrdf;
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
        detail::pass_weight(Float3{closure.weight});
    const auto diffuse_weight = select(
        select(0.0f,
            closure_pass_weight,
            diffuse_family),
        select(0.0f,
            closure_pass_weight,
            closure.setup_valid),
        is_sheen);
    const auto glossy_weight = select(
        0.0f,
        closure_pass_weight,
        is_dielectric | generic_glossy);

    luisa::compute::Var<SurfaceAovContributionCall> result;
    result.albedo = diffuse_albedo;
    result.glossy_albedo =
        select(
            make_float3(0.0f),
            closure.reflection_albedo,
            is_dielectric) +
        select(
            make_float3(0.0f),
            closure.albedo,
            generic_glossy);
    result.transmission_albedo = select(
        make_float3(0.0f),
        closure.transmission_albedo,
        is_dielectric);
    result.transparency = select(
        make_float3(0.0f),
        closure.weight,
        is_transparent);
    result.normal =
        diffuse_weight * select(
                             closure.normal,
                             glossy_normal,
                             is_translucent) +
        glossy_weight * glossy_normal;
    result.total_weight =
        diffuse_weight + glossy_weight;
    result.roughness_weight = glossy_weight;
    result.roughness =
        glossy_weight * closure.roughness;
    return result;
}

}// namespace

SurfaceClosureIdentityCallable
make_surface_closure_identity_callable() noexcept {
    return [](
               UInt kind,
               UInt lobe,
               UInt bssrdf_method,
               Float allocation_weight,
               Bool setup_valid,
               Float roughness,
               Bool preserve_ggx_energy,
               Bool beckmann,
               Float glossy_filter_roughness) noexcept {
        const auto closure =
            detail::SurfaceClosureIdentityExpression{
                .kind = Expr<std::uint32_t>{kind.expression()},
                .lobe = Expr<std::uint32_t>{lobe.expression()},
                .bssrdf_method = Expr<std::uint32_t>{
                    bssrdf_method.expression()},
                .allocation_weight =
                    Expr<float>{allocation_weight.expression()},
                .setup_valid =
                    Expr<bool>{setup_valid.expression()},
                .roughness = Expr<float>{roughness.expression()},
                .preserve_ggx_energy =
                    Expr<bool>{preserve_ggx_energy.expression()},
                .beckmann = Expr<bool>{beckmann.expression()}};
        return luisa::compute::make_uint2(
            detail::cycles_runtime_flags(
                closure,
                std::move(glossy_filter_roughness)),
            detail::cycles_closure_type(closure));
    };
}

SurfaceClosureAovCallable
make_surface_closure_aov_callable() noexcept {
    return [](
               Float3 incoming,
               Float3 shading_normal,
               Float3 geometric_normal,
               Bool use_bump_map_correction,
               UInt kind,
               UInt lobe,
               Float3 weight,
               Bool setup_valid,
               Float3 albedo,
               Float3 reflection_albedo,
               Float3 transmission_albedo,
               Float3 normal,
               Float roughness) noexcept {
        const auto closure = AovClosureExpression{
            .kind = Expr<std::uint32_t>{kind.expression()},
            .lobe = Expr<std::uint32_t>{lobe.expression()},
            .weight = Expr<luisa::float3>{weight.expression()},
            .setup_valid = Expr<bool>{setup_valid.expression()},
            .albedo = Expr<luisa::float3>{albedo.expression()},
            .reflection_albedo =
                Expr<luisa::float3>{reflection_albedo.expression()},
            .transmission_albedo =
                Expr<luisa::float3>{transmission_albedo.expression()},
            .normal = Expr<luisa::float3>{normal.expression()},
            .roughness = Expr<float>{roughness.expression()}};
        return aov_contribution(
            Expr<luisa::float3>{incoming.expression()},
            Expr<luisa::float3>{shading_normal.expression()},
            Expr<luisa::float3>{geometric_normal.expression()},
            Expr<bool>{use_bump_map_correction.expression()},
            closure);
    };
}

luisa::compute::Var<SurfaceAovContributionCall>
surface_closure_aov_contribution(
    const SurfacePoint &point,
    const SurfaceClosureRecord &closure) noexcept {
    return aov_contribution(
        Expr<luisa::float3>{point.incoming.expression()},
        Expr<luisa::float3>{point.shading_normal.expression()},
        Expr<luisa::float3>{point.geometric_normal.expression()},
        Expr<bool>{point.use_bump_map_correction.expression()},
        closure);
}

SurfaceRuntimeFlagsVisitor::SurfaceRuntimeFlagsVisitor(
    const SurfacePoint &point,
    Expr<float> glossy_filter_roughness,
    std::size_t capacity,
    const SurfaceClosureIdentityCallable &identity) noexcept
    : SurfaceClosureExpressionVisitor{capacity},
      _point{point},
      _glossy_filter_roughness{glossy_filter_roughness},
      _identity{identity} {}

void SurfaceRuntimeFlagsVisitor::visit(
    Expr<luisa::float3> shading_normal,
    const luisa::vector<SurfaceClosureExpression> &closures) noexcept {
    static_cast<void>(shading_normal);
    _result = select(
        0u,
        cycles_closure::runtime_backfacing,
        _point.back_facing);
    UInt allocated_count = 0u;
    for (const auto &closure : closures) {
        const auto keep = retains(
            closure, allocated_count);
        $if(keep) {
            _result |= classify(
                _identity,
                closure,
                _glossy_filter_roughness)
                           .x;
        };
        allocated_count += select(0u, 1u, keep);
    }
}

Expr<std::uint32_t> SurfaceRuntimeFlagsVisitor::result() const noexcept {
    return Expr<std::uint32_t>{_result.expression()};
}

SurfaceClosureTraceVisitor::SurfaceClosureTraceVisitor(
    const SurfacePoint &point,
    Expr<std::uint32_t> requested_index,
    std::size_t capacity,
    const SurfaceClosureIdentityCallable &identity) noexcept
    : SurfaceClosureExpressionVisitor{capacity},
      _point{point},
      _requested_index{requested_index},
      _identity{identity},
      _result{SurfaceClosureTrace::zero(requested_index)} {}

void SurfaceClosureTraceVisitor::visit(
    Expr<luisa::float3> shading_normal,
    const luisa::vector<SurfaceClosureExpression> &closures) noexcept {
    static_cast<void>(shading_normal);
    _result.count = 0u;
    _result.runtime_flags = select(
        0u,
        cycles_closure::runtime_backfacing,
        _point.back_facing);
    _result.index = _requested_index;
    _result.type = cycles_closure::type_none;
    _result.sample_weight = 0.0f;
    _result.weight = make_float3(0.0f);
    _result.normal = make_float3(0.0f, 0.0f, 1.0f);
    _result.valid = false;

    UInt allocated_count = 0u;
    for (const auto &closure : closures) {
        const auto keep = retains(
            closure, allocated_count);
        $if(keep) {
            const auto identity = classify(
                _identity, closure, 0.0f);
            _result.runtime_flags |= identity.x;
            const auto match =
                allocated_count == _requested_index;
            _result.type = select(
                _result.type, identity.y, match);
            _result.sample_weight = select(
                _result.sample_weight,
                closure.sample_weight,
                match);
            _result.weight = select(
                _result.weight, closure.weight, match);
            _result.normal = select(
                _result.normal, closure.normal, match);
        };
        allocated_count += select(0u, 1u, keep);
    }
    _result.count = allocated_count;
    _result.valid = _requested_index < allocated_count;
}

const SurfaceClosureTrace &SurfaceClosureTraceVisitor::result() const noexcept {
    return _result;
}

SurfaceAovVisitor::SurfaceAovVisitor(
    const SurfacePoint &point,
    std::size_t capacity,
    const SurfaceClosureAovCallable &aov) noexcept
    : SurfaceClosureExpressionVisitor{capacity},
      _point{point},
      _aov{aov},
      _result{
          .albedo = make_float3(0.0f),
          .glossy_albedo = make_float3(0.0f),
          .transmission_albedo = make_float3(0.0f),
          .roughness = make_float2(0.0f),
          .normal = point.shading_normal,
          .transparency = make_float3(0.0f)} {}

void SurfaceAovVisitor::visit(
    Expr<luisa::float3> shading_normal,
    const luisa::vector<SurfaceClosureExpression> &closures) noexcept {
    static_cast<void>(shading_normal);
    _result.albedo = make_float3(0.0f);
    _result.glossy_albedo = make_float3(0.0f);
    _result.transmission_albedo = make_float3(0.0f);
    _result.roughness = make_float2(0.0f);
    _result.normal = _point.shading_normal;
    _result.transparency = make_float3(0.0f);
    Float total_weight = 0.0f;
    Float roughness_weight = 0.0f;
    Float roughness = 0.0f;
    Float3 normal = make_float3(0.0f);

    UInt allocated_count = 0u;
    for (const auto &closure : closures) {
        const auto keep = retains(
            closure, allocated_count);
        $if(keep) {
            const auto contribution = _aov(
                _point.incoming,
                _point.shading_normal,
                _point.geometric_normal,
                _point.use_bump_map_correction,
                closure.kind,
                closure.lobe,
                closure.weight,
                closure.setup_valid,
                closure.albedo,
                closure.reflection_albedo,
                closure.transmission_albedo,
                closure.normal,
                closure.roughness);
            _result.albedo += contribution.albedo;
            _result.glossy_albedo +=
                contribution.glossy_albedo;
            _result.transmission_albedo +=
                contribution.transmission_albedo;
            _result.transparency +=
                contribution.transparency;
            total_weight += contribution.total_weight;
            roughness_weight +=
                contribution.roughness_weight;
            roughness += contribution.roughness;
            normal += contribution.normal;
        };
        allocated_count += select(0u, 1u, keep);
    }
    _result.roughness = make_float2(select(
        1.0f,
        roughness /
            max(roughness_weight, 1.0e-20f),
        roughness_weight > 0.0f));
    _result.normal = detail::safe_normalize(
        select(
            _point.shading_normal,
            normal,
            total_weight > 0.0f),
        _point.shading_normal);
}

const SurfaceAov &SurfaceAovVisitor::result() const noexcept {
    return _result;
}

}// namespace psycles::luisa_backend

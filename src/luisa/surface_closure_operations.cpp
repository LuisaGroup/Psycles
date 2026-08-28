#include <psycles/luisa/surface_closure_operations.h>

#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_closure.h>

#include <algorithm>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

[[nodiscard]] luisa::compute::UInt2 classify(
    const SurfaceClosureIdentityCallable &identity,
    const SurfaceClosureExpression &closure,
    Expr<float> glossy_filter_roughness) noexcept {
    return identity(
        closure.closure_type,
        closure.roughness,
        glossy_filter_roughness);
}

struct AovClosureExpression {
    Expr<std::uint32_t> closure_type;
    Expr<luisa::float3> weight;
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
    const UInt type{closure.closure_type};
    const auto is_transparent =
        type == cycles_closure::type_transparent;
    const auto is_diffuse =
        cycles_closure::is_diffuse_or_oren_nayar(type);
    const auto is_translucent =
        type == cycles_closure::type_translucent;
    const auto is_rough_translucent =
        type == cycles_closure::type_rough_translucent;
    const auto is_sheen =
        type == cycles_closure::type_sheen;
    const auto is_ashikhmin =
        type == cycles_closure::type_ashikhmin_velvet;
    const auto is_hair_reflection =
        type == cycles_closure::type_hair_reflection;
    const auto is_hair_transmission =
        type == cycles_closure::type_hair_transmission;
    const auto is_hair = is_hair_reflection | is_hair_transmission;
    const auto is_glass =
        cycles_closure::is_glass_microfacet(type);
    const auto is_refraction =
        cycles_closure::is_refraction_microfacet(type);
    const auto is_thin_glass_transmission =
        type == cycles_closure::type_thin_glass_transmission;
    const auto is_bssrdf =
        cycles_closure::is_bssrdf(type);
    const auto is_dielectric = is_glass | is_refraction;
    const auto is_dielectric_family =
        is_dielectric | is_thin_glass_transmission;
    const auto generic_glossy =
        cycles_closure::is_reflection_microfacet(type) | is_ashikhmin;

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
        is_sheen | is_ashikhmin | is_rough_translucent |
            is_thin_glass_transmission | is_hair);

    const auto diffuse_family =
        is_diffuse | is_translucent | is_rough_translucent | is_bssrdf;
    const auto diffuse_albedo = select(
        select(
            make_float3(0.0f),
            closure.albedo,
            diffuse_family),
        closure.albedo,
        is_sheen);
    const auto closure_pass_weight =
        detail::pass_weight(Float3{closure.weight});
    const auto diffuse_weight = select(
        select(0.0f,
            closure_pass_weight,
            diffuse_family),
        closure_pass_weight,
        is_sheen);
    const auto glossy_weight = select(
        0.0f,
        closure_pass_weight,
        is_dielectric_family | generic_glossy | is_hair);

    luisa::compute::Var<SurfaceAovContributionCall> result;
    result.albedo = diffuse_albedo;
    result.glossy_albedo =
        select(
            make_float3(0.0f),
            closure.reflection_albedo,
            is_dielectric_family | is_hair_reflection) +
        select(
            make_float3(0.0f),
            closure.albedo,
            generic_glossy);
    result.transmission_albedo = select(
        make_float3(0.0f),
        closure.transmission_albedo,
        is_dielectric_family | is_hair_transmission);
    result.transparency = select(
        make_float3(0.0f),
        closure.weight,
        is_transparent);
    result.normal =
        diffuse_weight * select(
                             closure.normal,
                             glossy_normal,
                             is_translucent |
                                 is_rough_translucent) +
        glossy_weight * glossy_normal;
    result.total_weight =
        diffuse_weight + glossy_weight;
    result.roughness_weight = glossy_weight;
    result.roughness = glossy_weight * select(
        closure.roughness, 1.0f, is_ashikhmin | is_hair);
    return result;
}

[[nodiscard]] SurfaceAov zero_aov(
    const SurfacePoint &point) noexcept {
    return {
        .albedo = make_float3(0.0f),
        .glossy_albedo = make_float3(0.0f),
        .transmission_albedo = make_float3(0.0f),
        .roughness = make_float2(0.0f),
        .normal = point.shading_normal,
        .transparency = make_float3(0.0f)};
}

template<bool Enabled>
class RuntimeFlagReduction;

template<>
class RuntimeFlagReduction<false> {

public:
    RuntimeFlagReduction(
        const SurfacePoint &,
        Expr<bool>) noexcept {}
};

template<>
class RuntimeFlagReduction<true> {

private:
    UInt _result;

public:
    RuntimeFlagReduction(
        const SurfacePoint &point,
        Expr<bool> include) noexcept
        : _result{select(
              0u,
              cycles_closure::runtime_backfacing,
              Bool{include} & point.back_facing)} {}

    template<typename Identity>
    void accumulate(
        Expr<bool> keep,
        Expr<bool> include,
        const Identity &identity,
        const SurfaceClosureExpression &closure,
        Expr<float> glossy_filter_roughness) noexcept {
        $if(keep & include) {
            _result |= classify(
                identity,
                closure,
                glossy_filter_roughness)
                           .x;
        };
    }

    [[nodiscard]] Expr<std::uint32_t> result() const noexcept {
        return Expr<std::uint32_t>{_result.expression()};
    }
};

template<bool Enabled>
class AovReduction;

template<>
class AovReduction<false> {

public:
    explicit AovReduction(const SurfacePoint &) noexcept {}
};

template<>
class AovReduction<true> {

private:
    SurfaceAov _result;
    Float _total_weight{0.0f};
    Float _roughness_weight{0.0f};
    Float _roughness{0.0f};
    Float3 _normal{make_float3(0.0f)};

public:
    explicit AovReduction(const SurfacePoint &point) noexcept
        : _result{zero_aov(point)} {}

    template<typename Operation>
    void accumulate(
        Expr<bool> keep,
        Expr<bool> include,
        const Operation &operation,
        const SurfacePoint &point,
        const SurfaceClosureExpression &closure) noexcept {
        $if(keep & include) {
            const auto contribution = operation(
                point.incoming,
                point.shading_normal,
                point.geometric_normal,
                point.use_bump_map_correction,
                closure.closure_type,
                closure.weight,
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
            _total_weight += contribution.total_weight;
            _roughness_weight += contribution.roughness_weight;
            _roughness += contribution.roughness;
            _normal += contribution.normal;
        };
    }

    void finish(
        const SurfacePoint &point,
        Expr<bool> include) noexcept {
        const auto computed_roughness = make_float2(select(
            1.0f,
            _roughness / max(_roughness_weight, 1.0e-20f),
            _roughness_weight > 0.0f));
        _result.roughness = select(
            make_float2(0.0f),
            computed_roughness,
            include);
        _result.normal = detail::safe_normalize(
            select(
                point.shading_normal,
                _normal,
                _total_weight > 0.0f),
            point.shading_normal);
    }

    [[nodiscard]] const SurfaceAov &result() const noexcept {
        return _result;
    }
};

template<bool ReduceRuntimeFlags, bool ReduceAov>
struct ClosureReductions {
    RuntimeFlagReduction<ReduceRuntimeFlags> runtime_flags;
    AovReduction<ReduceAov> aov;
};

template<bool ReduceRuntimeFlags,
         bool ReduceAov,
         typename Identity,
         typename AovOperation,
         typename Retains>
[[nodiscard]] ClosureReductions<ReduceRuntimeFlags, ReduceAov>
reduce_closures(
    const SurfacePoint &point,
    Expr<float> glossy_filter_roughness,
    Expr<bool> include_runtime_flags,
    Expr<bool> include_aov,
    const Identity &identity,
    const AovOperation &aov_operation,
    const luisa::vector<SurfaceClosureExpression> &closures,
    Retains &&retains) noexcept {
    auto result = ClosureReductions<
        ReduceRuntimeFlags,
        ReduceAov>{
        .runtime_flags = RuntimeFlagReduction<ReduceRuntimeFlags>{
            point,
            include_runtime_flags},
        .aov = AovReduction<ReduceAov>{point}};
    UInt allocated_count = 0u;
    for (const auto &closure : closures) {
        const auto keep = retains(closure, allocated_count);
        if constexpr (ReduceRuntimeFlags) {
            result.runtime_flags.accumulate(
                keep,
                include_runtime_flags,
                identity,
                closure,
                glossy_filter_roughness);
        }
        if constexpr (ReduceAov) {
            result.aov.accumulate(
                keep,
                include_aov,
                aov_operation,
                point,
                closure);
        }
        allocated_count += select(0u, 1u, keep);
    }
    if constexpr (ReduceAov) {
        result.aov.finish(point, include_aov);
    }
    return result;
}

}// namespace

SurfaceClosureIdentityCallable
make_surface_closure_identity_callable() noexcept {
    SurfaceClosureIdentityCallable callable = [](
               UInt closure_type,
               Float roughness,
               Float glossy_filter_roughness) noexcept {
        return luisa::compute::make_uint2(
            detail::cycles_runtime_flags(
                closure_type,
                roughness,
                glossy_filter_roughness),
            closure_type);
    };
    callable.set_name("surface_closure_identity");
    return callable;
}

SurfaceClosureAovCallable
make_surface_closure_aov_callable() noexcept {
    SurfaceClosureAovCallable callable = [](
               Float3 incoming,
               Float3 shading_normal,
               Float3 geometric_normal,
               Bool use_bump_map_correction,
               UInt closure_type,
               Float3 weight,
               Float3 albedo,
               Float3 reflection_albedo,
               Float3 transmission_albedo,
               Float3 normal,
               Float roughness) noexcept {
        const auto closure = AovClosureExpression{
            .closure_type = Expr<std::uint32_t>{closure_type.expression()},
            .weight = Expr<luisa::float3>{weight.expression()},
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
    callable.set_name("surface_closure_aov");
    return callable;
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
    const auto reductions = reduce_closures<true, false>(
        _point,
        _glossy_filter_roughness,
        Expr<bool>{true},
        Expr<bool>{false},
        _identity,
        nullptr,
        closures,
        [&](const SurfaceClosureExpression &closure,
            Expr<std::uint32_t> allocated_count) noexcept {
            return retains(closure, allocated_count);
        });
    _result = reductions.runtime_flags.result();
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

SurfaceBssrdfNormalVisitor::SurfaceBssrdfNormalVisitor(
    std::size_t capacity) noexcept
    : SurfaceClosureExpressionVisitor{capacity}, _capacity{capacity} {}

detail::SurfaceBssrdfNormalAccumulator::SurfaceBssrdfNormalAccumulator(
    Expr<luisa::float3> shading_normal, std::size_t capacity) noexcept
    : _capacity{std::clamp(
          capacity, std::size_t{1u},
          static_cast<std::size_t>(maximum_surface_closure_capacity))},
      _shading_normal{shading_normal} {}

void detail::SurfaceBssrdfNormalAccumulator::set_shading_normal(
    Expr<luisa::float3> shading_normal) noexcept {
    _shading_normal = shading_normal;
}

void detail::SurfaceBssrdfNormalAccumulator::add(
    Expr<std::uint32_t> closure_type_expression,
    Expr<luisa::float3> weight_expression,
    Expr<float> allocation_weight_expression,
    Expr<luisa::float3> normal_expression) noexcept {
    const auto closure_type = UInt{closure_type_expression};
    const auto allocation_weight = Float{allocation_weight_expression};
    const auto allocated =
        allocation_weight >= cycles_closure::closure_weight_cutoff;
    const auto retained =
        allocated & (_retained_count < static_cast<std::uint32_t>(_capacity));
    const auto contributes =
        retained & cycles_closure::is_bssrdf(closure_type);
    const auto weight = detail::pass_weight(Float3{weight_expression});
    _weighted_normal += select(make_float3(0.0f),
                               Float3{normal_expression} * weight, contributes);
    _retained_count += select(0u, 1u, retained);
}

Expr<luisa::float3>
detail::SurfaceBssrdfNormalAccumulator::result() const noexcept {
    // Cycles uses is_zero(), not an epsilon. The safe operand prevents an
    // inactive normalize(0) from manufacturing NaNs in branchless lowering.
    const auto nonzero = any(_weighted_normal != make_float3(0.0f));
    const auto safe =
        select(make_float3(0.0f, 0.0f, 1.0f), _weighted_normal, nonzero);
    return Expr<luisa::float3>{
        select(_shading_normal, normalize(safe), nonzero).expression()};
}

void SurfaceBssrdfNormalVisitor::visit(
    Expr<luisa::float3> shading_normal_expression,
    const luisa::vector<SurfaceClosureExpression> &closures) noexcept {
    detail::SurfaceBssrdfNormalAccumulator accumulator{shading_normal_expression,
                                                       _capacity};
    for (const auto &closure : closures) {
        accumulator.add(
            closure.closure_type,
            closure.weight,
            closure.allocation_weight,
            closure.normal);
    }
    _result = accumulator.result();
}

Expr<luisa::float3>
SurfaceBssrdfNormalVisitor::result() const noexcept {
    return Expr<luisa::float3>{_result.expression()};
}

SurfaceAovVisitor::SurfaceAovVisitor(
    const SurfacePoint &point,
    std::size_t capacity,
    const SurfaceClosureAovCallable &aov) noexcept
    : SurfaceClosureExpressionVisitor{capacity},
      _point{point},
      _aov{aov},
      _result{zero_aov(point)} {}

void SurfaceAovVisitor::visit(
    Expr<luisa::float3> shading_normal,
    const luisa::vector<SurfaceClosureExpression> &closures) noexcept {
    auto point = _point;
    point.shading_normal = shading_normal;
    _result = reduce_closures<false, true>(
                  point,
                  Expr<float>{0.0f},
                  Expr<bool>{false},
                  Expr<bool>{true},
                  nullptr,
                  _aov,
                  closures,
                  [&](const SurfaceClosureExpression &closure,
                      Expr<std::uint32_t> allocated_count) noexcept {
                      return retains(closure, allocated_count);
                  })
                  .aov.result();
}

const SurfaceAov &SurfaceAovVisitor::result() const noexcept {
    return _result;
}

SurfacePreparationVisitor::SurfacePreparationVisitor(
    const SurfacePoint &point,
    Expr<float> glossy_filter_roughness,
    Expr<bool> include_runtime_flags,
    Expr<bool> include_aov,
    std::size_t capacity,
    const SurfaceClosureIdentityCallable &identity,
    const SurfaceClosureAovCallable &aov_operation) noexcept
    : SurfaceClosureExpressionVisitor{capacity},
      _point{point},
      _glossy_filter_roughness{glossy_filter_roughness},
      _include_runtime_flags{include_runtime_flags},
      _include_aov{include_aov},
      _identity{identity},
      _aov_operation{aov_operation},
      _shading_normal{point.shading_normal},
      _aov{zero_aov(point)} {}

void SurfacePreparationVisitor::visit(
    Expr<luisa::float3> shading_normal,
    const luisa::vector<SurfaceClosureExpression> &closures) noexcept {
    auto point = _point;
    point.shading_normal = shading_normal;
    _shading_normal = shading_normal;
    const auto reductions = reduce_closures<true, true>(
        point,
        _glossy_filter_roughness,
        _include_runtime_flags,
        _include_aov,
        _identity,
        _aov_operation,
        closures,
        [&](const SurfaceClosureExpression &closure,
            Expr<std::uint32_t> allocated_count) noexcept {
            return retains(closure, allocated_count);
        });
    _runtime_flags = reductions.runtime_flags.result();
    _aov = reductions.aov.result();
}

Expr<std::uint32_t>
SurfacePreparationVisitor::runtime_flags() const noexcept {
    return Expr<std::uint32_t>{_runtime_flags.expression()};
}

Expr<luisa::float3>
SurfacePreparationVisitor::shading_normal() const noexcept {
    return Expr<luisa::float3>{_shading_normal.expression()};
}

const SurfaceAov &SurfacePreparationVisitor::aov() const noexcept {
    return _aov;
}

}// namespace psycles::luisa_backend

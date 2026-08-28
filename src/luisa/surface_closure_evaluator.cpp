#include <psycles/luisa/surface_closure_evaluator.h>

#include "graph_surface_internal.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface_closure_evaluation.h>
#include <psycles/luisa/surface_closure_operations.h>

#include <utility>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

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

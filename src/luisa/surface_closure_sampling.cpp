#include <psycles/luisa/surface_closure_evaluator.h>

#include "graph_surface_internal.h"

#include <psycles/luisa/surface_closure_sampling.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {
namespace {

// Host-stage specialization over the storage profile keeps one source-level
// categorical algorithm while recording exactly one device loop. The
// physical specialization observes block_0 only; the complete specialization
// remains the diagnostic oracle for round-trip tests.
template<typename Consumer>
void for_each_surface_closure_selection(
    const SurfaceClosureSet &closures,
    const SurfaceClosureSelectionContext &context,
    SurfaceClosureReachability reachability,
    bool include_runtime_flags,
    Consumer &&consumer) noexcept {
    UInt index = 0u;
    if (closures.profile() ==
        SurfaceClosureStorageProfile::physical) {
        $while(index < closures.count()) {
            const auto access =
                closures.physical_access(index);
            const auto common =
                closures.physical_common_entry(access);
            consumer(index,
                surface_closure_selection(
                    context,
                    common,
                    include_runtime_flags,
                    reachability));
            index += 1u;
        };
    } else {
        $while(index < closures.count()) {
            const auto closure =
                static_cast<SurfaceClosurePhysicalRecord>(
                    closures.entry(index));
            consumer(index,
                surface_closure_selection(
                    context,
                    make_surface_closure_selection_input(closure),
                    include_runtime_flags,
                    reachability));
            index += 1u;
        };
    }
}

// Payload elimination is deliberately delayed until after categorical
// inversion. The explicit prefix projection is part of the semantic contract:
// the coroutine initialized-prefix analysis cannot infer the business-level
// implication `selected() => selected_index < count()`. Without this witness,
// both physical Local arrays must conservatively remain live across suspend.
template<typename Consumer>
void with_selected_surface_closure(
    const SurfaceClosureSet &closures,
    UInt selected_index,
    const SurfaceClosureSelectionContext &context,
    SurfaceClosureReachability reachability,
    bool include_runtime_flags,
    Consumer &&consumer) noexcept {
    if (closures.profile() ==
        SurfaceClosureStorageProfile::physical) {
        const auto access =
            closures.physical_access(selected_index);
        const auto common =
            closures.physical_common_entry(access);
        consumer(
            closures.physical_payload_entry(access, common),
            surface_closure_selection(
                context,
                common,
                include_runtime_flags,
                reachability));
    } else {
        const auto closure =
            static_cast<SurfaceClosurePhysicalRecord>(
                closures.entry(selected_index));
        consumer(closure,
            surface_closure_selection(
                context,
                make_surface_closure_selection_input(closure),
                include_runtime_flags,
                reachability));
    }
}

}// namespace

SurfaceSampleTrace SurfaceClosureEvaluator::sample_impl(
    const ShaderServices &services,
    Float u_lobe,
    Float2 u_direction,
    const SurfaceQuery &query,
    bool trace_selection) const noexcept {
    const SurfaceClosurePoint closure_point{_point};
    const auto incoming = detail::safe_normalize(
        _point.incoming, _point.shading_normal);
    const auto selection_context =
        make_surface_closure_selection_context(query);
    const auto use_populated_runtime_flags =
        _populated_runtime_state.has_value();

    // Pass one constructs exactly the finite Cycles closure-selection
    // measure. The Local evaluator remains only as a storage-form regression;
    // all selection and conditional BSDF algebra is shared with the
    // branch-local visitor.
    SurfaceClosureSelectionMeasure measure{_point.back_facing};
    for_each_surface_closure_selection(
        _closures,
        selection_context,
        _reachability,
        !use_populated_runtime_flags,
        [&](UInt,
            const luisa::compute::Var<
                SurfaceClosureSelectionCall> &selection) noexcept {
            measure.add(selection);
        });

    // Pass two performs inverse-CDF selection without any conditional BSDF
    // sampler in the loop-carried state.
    SurfaceClosureCategoricalInversion inversion{
        Expr<float>{u_lobe.expression()}, measure};
    UInt selected_index = ~std::uint32_t{0u};
    Float selected_rescaled = 0.0f;
    for_each_surface_closure_selection(
        _closures,
        selection_context,
        _reachability,
        !use_populated_runtime_flags,
        [&](UInt index,
            const luisa::compute::Var<
                SurfaceClosureSelectionCall> &selection) noexcept {
            const auto choice = inversion.consider(selection);
            selected_index = select(
                selected_index, index, choice.choose);
            selected_rescaled = select(
                selected_rescaled, choice.rescaled, choice.choose);
        });

    // Exactly one p(w_i | i) executes. Keeping the runtime-indexed load under
    // this predicate also makes invalid empty measures observationally zero.
    SurfaceClosureSelectedSample selected;
    $if(inversion.selected()) {
        if (_closures.profile() == SurfaceClosureStorageProfile::physical) {
            const auto access = _closures.physical_access(selected_index);
            const auto common = _closures.physical_common_entry(access);
            const auto selection = surface_closure_selection(
                selection_context,
                common,
                !use_populated_runtime_flags,
                _reachability);
            const auto sample =
                surface_closure_conditional_sample_from_physical_common(
                    services,
                    closure_point,
                    Expr<luisa::float3>{_shading_normal.expression()},
                    common,
                    [&] {
                        // Build the proof witness in the family block which
                        // performs the payload read. See the evaluation path:
                        // transporting a mutable count snapshot across this
                        // branch is intentionally outside the prefix proof.
                        return _closures.physical_payload_block(
                            _closures.physical_access(selected_index));
                    },
                    Expr<luisa::float3>{incoming.expression()},
                    Expr<luisa::float3>{selection.glossy_normal.expression()},
                    Expr<luisa::float2>{u_direction.expression()},
                    Expr<float>{selected_rescaled.expression()},
                    query,
                    _reachability);
            selected.accept(
                Expr<std::uint32_t>{selected_index.expression()},
                Expr<luisa::float3>{common.weight.expression()},
                Expr<luisa::float3>{common.normal.expression()},
                Expr<float>{selected_rescaled.expression()},
                selection,
                sample);
        } else {
            with_selected_surface_closure(
                _closures,
                selected_index,
                selection_context,
                _reachability,
                !use_populated_runtime_flags,
                [&](const SurfaceClosurePhysicalRecord &closure,
                    const luisa::compute::Var<
                        SurfaceClosureSelectionCall> &selection) noexcept {
                    const auto sample = surface_closure_conditional_sample(
                        services,
                        closure_point,
                        Expr<luisa::float3>{
                            _shading_normal.expression()},
                        closure,
                        Expr<luisa::float3>{incoming.expression()},
                        Expr<luisa::float3>{
                            selection.glossy_normal.expression()},
                        Expr<luisa::float2>{u_direction.expression()},
                        Expr<float>{selected_rescaled.expression()},
                        query,
                        _reachability);
                    selected.accept(
                        Expr<std::uint32_t>{
                            selected_index.expression()},
                        Expr<luisa::float3>{closure.weight.expression()},
                        Expr<luisa::float3>{closure.normal.expression()},
                        Expr<float>{selected_rescaled.expression()},
                        selection,
                        sample);
                });
        }
    };

    const auto mixture = evaluate_impl(
        services,
        Float3{selected.direction()},
        query,
        EvaluationMode::sampled_bsdf,
        0u,
        UInt{selected.closure_index()});
    const auto runtime_flags = use_populated_runtime_flags
                                   ? _populated_runtime_state->_runtime_flags
                                   : measure.runtime_flags();
    return selected.finish(
        closure_point,
        measure,
        runtime_flags,
        mixture,
        trace_selection);
}

SurfaceSample SurfaceClosureEvaluator::sample(
    const ShaderServices &services,
    Expr<float> u_lobe,
    Expr<luisa::float2> u_direction,
    const SurfaceQuery &query) const noexcept {
    return sample_impl(services,
        Float{u_lobe},
        Float2{u_direction},
        query,
        false)
        .sample;
}

SurfaceSampleTrace SurfaceClosureEvaluator::sample_trace(
    const ShaderServices &services,
    Expr<float> u_lobe,
    Expr<luisa::float2> u_direction,
    const SurfaceQuery &query) const noexcept {
    return sample_impl(services,
        Float{u_lobe},
        Float2{u_direction},
        query,
        true);
}

}// namespace psycles::luisa_backend

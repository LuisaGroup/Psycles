#include <psycles/luisa/surface_closure_evaluator.h>

#include "graph_surface_internal.h"

#include <psycles/luisa/surface_closure_sampling.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend {

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

    // Pass one constructs exactly the finite Cycles closure-selection
    // measure. The Local evaluator remains only as a storage-form regression;
    // all selection and conditional BSDF algebra is shared with the
    // branch-local visitor.
    SurfaceClosureSelectionMeasure measure{_point.back_facing};
    UInt index = 0u;
    $while(index < _closures.count()) {
        measure.add(surface_closure_selection(
            selection_context,
            make_surface_closure_selection_input(
                static_cast<SurfaceClosurePhysicalRecord>(
                    _closures.entry(index)))));
        index += 1u;
    };

    // Pass two performs inverse-CDF selection without any conditional BSDF
    // sampler in the loop-carried state.
    SurfaceClosureCategoricalInversion inversion{
        Expr<float>{u_lobe.expression()}, measure};
    UInt selected_index = ~std::uint32_t{0u};
    Float selected_rescaled = 0.0f;
    index = 0u;
    $while(index < _closures.count()) {
        const auto selection = surface_closure_selection(
            selection_context,
            make_surface_closure_selection_input(
                static_cast<SurfaceClosurePhysicalRecord>(
                    _closures.entry(index))));
        const auto choice = inversion.consider(selection);
        selected_index = select(
            selected_index, index, choice.choose);
        selected_rescaled = select(
            selected_rescaled, choice.rescaled, choice.choose);
        index += 1u;
    };

    // Exactly one p(w_i | i) executes. Keeping the runtime-indexed load under
    // this predicate also makes invalid empty measures observationally zero.
    SurfaceClosureSelectedSample selected;
    $if(inversion.selected()) {
        const auto closure = _closures.entry(selected_index);
        const auto selection = surface_closure_selection(
            selection_context,
            make_surface_closure_selection_input(
                static_cast<SurfaceClosurePhysicalRecord>(closure)));
        const auto sample = surface_closure_conditional_sample(
            services,
            closure_point,
            Expr<luisa::float3>{_shading_normal.expression()},
            closure,
            Expr<luisa::float3>{incoming.expression()},
            Expr<luisa::float3>{
                selection.glossy_normal.expression()},
            Expr<luisa::float2>{u_direction.expression()},
            Expr<float>{selected_rescaled.expression()},
            query);
        selected.accept(
            Expr<std::uint32_t>{selected_index.expression()},
            Expr<luisa::float3>{closure.weight.expression()},
            Expr<luisa::float3>{closure.normal.expression()},
            Expr<float>{selected_rescaled.expression()},
            selection,
            sample);
    };

    const auto mixture = evaluate_impl(
        services,
        Float3{selected.direction()},
        query,
        EvaluationMode::sampled_bsdf,
        0u,
        UInt{selected.closure_index()});
    return selected.finish(
        closure_point, measure, mixture, trace_selection);
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

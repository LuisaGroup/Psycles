#include "graph_surface_internal.h"
#include "principled_emission_layer.h"

#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {

void GraphSurfaceImplementation::for_each_physical_closure(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedValues &values,
    const ClosureVisitor &function) const noexcept {
    const auto incoming =
        safe_normalize(point.incoming, point.shading_normal);
    const PrincipledLayerComponent principled_layers{services, point};
    Float3 transparent_weight = make_float3(0.0f);
    Float transparent_sample_weight = 0.0f;
    for_each_closure(
        values, [&](const TracedClosure &closure) noexcept {
            auto contribution = TransparentClosureState{
                .weight = make_float3(0.0f),
                .sample_weight = 0.0f};
            if (closure.operation ==
                compiler::ClosureOperation::transparent) {
                contribution = transparent_closure_state(
                    closure.weight);
            } else if (closure.operation ==
                       compiler::ClosureOperation::principled) {
                contribution =
                    evaluate_principled_alpha_layer(closure)
                        .transparency;
            }
            transparent_weight += contribution.weight;
            transparent_sample_weight += contribution.sample_weight;
        });

    Bool transparent_allocated = false;
    for_each_closure(
        values, [&](const TracedClosure &graph_closure) noexcept {
            auto closure = graph_closure;
            closure.principled_lobe = PrincipledLobe::none;
            closure.setup_valid = true;
            closure.evaluation_scale = make_float3(1.0f);

            const auto produces_transparency =
                graph_closure.operation ==
                    compiler::ClosureOperation::transparent ||
                graph_closure.operation ==
                    compiler::ClosureOperation::principled;
            if (produces_transparency) {
                auto contribution = TransparentClosureState{
                    .weight = make_float3(0.0f),
                    .sample_weight = 0.0f};
                if (graph_closure.operation ==
                    compiler::ClosureOperation::transparent) {
                    contribution = transparent_closure_state(
                        graph_closure.weight);
                } else {
                    contribution = evaluate_principled_alpha_layer(
                                           graph_closure)
                                           .transparency;
                }
                const auto contribution_allocated =
                    contribution.sample_weight >=
                    cycles_closure::closure_weight_cutoff;
                const auto emit_here =
                    contribution_allocated & !transparent_allocated;
                auto transparent = graph_closure;
                transparent.operation =
                    compiler::ClosureOperation::transparent;
                transparent.principled_lobe = PrincipledLobe::none;
                transparent.weight = select(make_float3(0.0f),
                    transparent_weight,
                    emit_here);
                transparent.allocation_weight =
                    select(0.0f,
                        transparent_sample_weight,
                        emit_here);
                transparent.sample_weight =
                    transparent.allocation_weight;
                transparent.setup_valid = true;
                transparent.albedo = transparent.weight;
                transparent.color = make_float3(1.0f);
                transparent.normal = point.shading_normal;
                transparent.roughness = 0.0f;
                transparent.ior = 1.0f;
                transparent.evaluation_scale = make_float3(1.0f);
                function(transparent);
                transparent_allocated |= contribution_allocated;
            }

            switch (graph_closure.operation) {
            case compiler::ClosureOperation::principled: {
                closure.weight =
                    evaluate_principled_alpha_layer(graph_closure)
                        .lower_weight;
                const auto sheen = principled_layers.evaluate_sheen(
                    graph_closure, closure.weight);
                function(sheen.closure);
                closure.weight = sheen.lower_weight;
                const auto glossy_normal =
                    maybe_ensure_valid_specular_reflection(
                        point, incoming, graph_closure.normal);
                const auto state = principled_state(
                    services, closure, incoming, glossy_normal);

                auto metallic = closure;
                metallic.principled_lobe = PrincipledLobe::metallic;
                metallic.weight = state.metallic_weight;
                metallic.allocation_weight =
                    state.metallic_allocation_weight;
                metallic.sample_weight = state.metallic_sample_weight;
                metallic.albedo = state.metallic_albedo;
                metallic.color = state.metallic_f0;
                metallic.normal = glossy_normal;
                metallic.ior = 1.0f;
                metallic.specular_tint = state.metallic_b;
                metallic.evaluation_scale = state.metallic_energy_scale;
                function(metallic);

                auto dielectric = closure;
                dielectric.principled_lobe = PrincipledLobe::dielectric;
                dielectric.weight = state.dielectric_weight;
                dielectric.allocation_weight =
                    state.dielectric_allocation_weight;
                dielectric.sample_weight =
                    state.dielectric_sample_weight;
                dielectric.albedo = state.dielectric_albedo;
                dielectric.color = state.dielectric_f0;
                dielectric.normal = glossy_normal;
                dielectric.ior = state.eta;
                dielectric.evaluation_scale =
                    state.dielectric_energy_scale;
                function(dielectric);

                auto diffuse = closure;
                diffuse.operation = compiler::ClosureOperation::diffuse;
                diffuse.principled_lobe = PrincipledLobe::none;
                const auto effective_radius =
                    max(graph_closure.subsurface_radius *
                            graph_closure.subsurface_scale,
                        make_float3(0.0f));
                const auto maximum_radius =
                    max_component(effective_radius);
                const auto radius_tint = select(
                    make_float3(0.0f),
                    effective_radius /
                        max(maximum_radius, 1.0e-20f),
                    maximum_radius > 1.0e-20f);
                // A full spatial BSSRDF walk is not available yet. Use the
                // per-channel mean free paths as a scattering tint, then
                // normalize it back to the original average diffuse energy.
                // This reproduces long-radius color bleed while retaining
                // the original Principled color for the data pass below.
                const auto radius_weighted =
                    state.diffuse_weight * radius_tint;
                const auto radius_weighted_average =
                    sample_weight(radius_weighted);
                const auto energy_normalized =
                    radius_weighted *
                    (sample_weight(state.diffuse_weight) /
                        max(radius_weighted_average, 1.0e-20f));
                const auto subsurface_weight =
                    select(0.0f,
                        clamp(graph_closure.subsurface_weight, 0.0f, 1.0f),
                        radius_weighted_average > 1.0e-20f);
                // Spatial diffusion is most visible near silhouettes. Bias
                // the tint toward grazing views so broad front-facing areas
                // retain more of the original surface color.
                const auto incoming_cosine = clamp(
                    abs(dot(diffuse.normal, incoming)), 0.0f, 1.0f);
                const auto grazing_mix = lerp(
                    0.35f, 0.85f, 1.0f - incoming_cosine);
                diffuse.weight = lerp(
                    state.diffuse_weight,
                    energy_normalized,
                    subsurface_weight * grazing_mix);
                diffuse.allocation_weight = sample_weight(
                    max(diffuse.weight, make_float3(0.0f)));
                const auto diffuse_allocated =
                    diffuse.allocation_weight >=
                    cycles_closure::closure_weight_cutoff;
                diffuse.weight = select(make_float3(0.0f),
                    diffuse.weight,
                    diffuse_allocated);
                diffuse.sample_weight = select(
                    0.0f, diffuse.allocation_weight, diffuse_allocated);
                diffuse.albedo = state.diffuse_weight;
                diffuse.roughness = graph_closure.diffuse_roughness;
                diffuse.evaluation_scale = make_float3(1.0f);
                function(diffuse);
                return;
            }
            case compiler::ClosureOperation::glossy: {
                closure.allocation_weight = sample_weight(
                    max(closure.weight, make_float3(0.0f)));
                const auto allocated =
                    closure.allocation_weight >=
                    cycles_closure::closure_weight_cutoff;
                closure.weight = select(make_float3(0.0f),
                    max(closure.weight, make_float3(0.0f)),
                    allocated);
                closure.normal = maybe_ensure_valid_specular_reflection(
                    point, incoming, graph_closure.normal);
                closure.albedo = closure.weight *
                                 max(closure.color, make_float3(0.04f));
                closure.sample_weight = select(0.0f,
                    closure.allocation_weight *
                        sample_weight(
                            max(closure.color, make_float3(0.04f))),
                    allocated);
                break;
            }
            case compiler::ClosureOperation::glass: {
                closure.allocation_weight = sample_weight(
                    max(closure.weight, make_float3(0.0f)));
                const auto allocated =
                    closure.allocation_weight >=
                    cycles_closure::closure_weight_cutoff;
                closure.weight = select(make_float3(0.0f),
                    max(closure.weight, make_float3(0.0f)),
                    allocated);
                closure.normal = maybe_ensure_valid_specular_reflection(
                    point, incoming, graph_closure.normal);
                closure.ior = select(graph_closure.ior,
                    1.0f / max(graph_closure.ior, 1.0e-20f),
                    point.back_facing);
                closure.color = max(
                    graph_closure.color, make_float3(0.0f));
                closure.albedo = closure.weight * closure.color;
                closure.sample_weight = select(0.0f,
                    closure.allocation_weight *
                        sample_weight(closure.color),
                    allocated);
                break;
            }
            case compiler::ClosureOperation::translucent: {
                closure.allocation_weight = sample_weight(
                    max(closure.weight, make_float3(0.0f)));
                const auto allocated =
                    closure.allocation_weight >=
                    cycles_closure::closure_weight_cutoff;
                closure.weight = select(make_float3(0.0f),
                    max(closure.weight, make_float3(0.0f)),
                    allocated);
                // Cycles currently applies the specular-reflection
                // normal correction to translucent setup as well.
                closure.normal = maybe_ensure_valid_specular_reflection(
                    point, incoming, graph_closure.normal);
                closure.albedo = closure.weight;
                closure.sample_weight =
                    select(0.0f, closure.allocation_weight, allocated);
                break;
            }
            case compiler::ClosureOperation::diffuse: {
                closure.allocation_weight = sample_weight(
                    max(closure.weight, make_float3(0.0f)));
                const auto allocated =
                    closure.allocation_weight >=
                    cycles_closure::closure_weight_cutoff;
                closure.weight = select(make_float3(0.0f),
                    max(closure.weight, make_float3(0.0f)),
                    allocated);
                closure.albedo = closure.weight;
                closure.sample_weight =
                    select(0.0f, closure.allocation_weight, allocated);
                break;
            }
            case compiler::ClosureOperation::transparent: {
                return;
            }
            case compiler::ClosureOperation::emission:
            case compiler::ClosureOperation::null_closure:
            case compiler::ClosureOperation::add:
            case compiler::ClosureOperation::mix:
                closure.allocation_weight = 0.0f;
                closure.albedo = make_float3(0.0f);
                closure.sample_weight = 0.0f;
                break;
            }
            function(closure);
        });
}

} // namespace psycles::luisa_backend::detail

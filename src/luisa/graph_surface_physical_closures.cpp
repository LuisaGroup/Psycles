#include "graph_surface_internal.h"
#include "microfacet_glass_component.h"
#include "principled_base_component.h"
#include "principled_layer_component.h"

#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {

SurfaceClosureRecord canonical_surface_closure(
    const TracedClosure &closure) noexcept {
    auto result = SurfaceClosureRecord::zero();
    switch (closure.operation) {
        case compiler::ClosureOperation::diffuse:
            result.kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::diffuse);
            break;
        case compiler::ClosureOperation::translucent:
            result.kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::translucent);
            break;
        case compiler::ClosureOperation::principled:
            result.kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::principled);
            break;
        case compiler::ClosureOperation::glossy:
            result.kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::glossy);
            break;
        case compiler::ClosureOperation::glass:
            result.kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::glass);
            break;
        case compiler::ClosureOperation::transparent:
            result.kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::transparent);
            break;
        case compiler::ClosureOperation::null_closure:
        case compiler::ClosureOperation::emission:
        case compiler::ClosureOperation::add:
        case compiler::ClosureOperation::mix:
            break;
    }
    switch (closure.principled_lobe) {
        case PrincipledLobe::none:
            break;
        case PrincipledLobe::sheen:
            result.lobe = static_cast<std::uint32_t>(
                SurfaceClosureLobe::sheen);
            break;
        case PrincipledLobe::coat:
            result.lobe = static_cast<std::uint32_t>(
                SurfaceClosureLobe::coat);
            break;
        case PrincipledLobe::metallic:
            result.lobe = static_cast<std::uint32_t>(
                SurfaceClosureLobe::metallic);
            break;
        case PrincipledLobe::transmission:
            result.lobe = static_cast<std::uint32_t>(
                SurfaceClosureLobe::transmission);
            break;
        case PrincipledLobe::dielectric:
            result.lobe = static_cast<std::uint32_t>(
                SurfaceClosureLobe::dielectric);
            break;
    }
    result.weight = closure.weight;
    result.allocation_weight = closure.allocation_weight;
    result.sample_weight = closure.sample_weight;
    result.setup_valid = closure.setup_valid;
    result.albedo = closure.albedo;
    result.color = closure.color;
    result.normal = closure.normal;
    result.roughness = closure.roughness;
    result.ior = closure.ior;
    result.evaluation_scale = closure.evaluation_scale;

    if (closure.operation == compiler::ClosureOperation::diffuse ||
        closure.operation == compiler::ClosureOperation::principled) {
        result.diffuse_roughness = closure.diffuse_roughness;
    }
    if (closure.operation == compiler::ClosureOperation::principled ||
        closure.operation == compiler::ClosureOperation::glossy) {
        result.metallic = closure.metallic;
        result.specular_ior_level = closure.specular_ior_level;
        result.specular_tint = closure.specular_tint;
    }
    if (closure.operation == compiler::ClosureOperation::principled &&
        closure.principled_lobe == PrincipledLobe::sheen) {
        result.sheen_transform_a = closure.sheen_transform_a;
        result.sheen_transform_b = closure.sheen_transform_b;
    }
    if (closure.operation == compiler::ClosureOperation::glass) {
        result.reflection_albedo = closure.reflection_albedo;
        result.transmission_albedo = closure.transmission_albedo;
        result.fresnel_f0 = closure.fresnel_f0;
        result.fresnel_f90 = closure.fresnel_f90;
        result.reflection_tint = closure.reflection_tint;
        result.transmission_tint = closure.transmission_tint;
    }
    if (closure.operation == compiler::ClosureOperation::principled ||
        closure.operation == compiler::ClosureOperation::glossy ||
        closure.operation == compiler::ClosureOperation::glass) {
        result.preserve_ggx_energy = closure.preserve_ggx_energy;
    }
    if (closure.operation == compiler::ClosureOperation::glass) {
        result.beckmann = closure.beckmann;
    }
    return result;
}

SurfaceClosureCollection GraphSurfaceImplementation::collect_closures(
    const ShaderServices &services,
    const SurfacePoint &point,
    Expr<bool> reflective_caustics_expression,
    Expr<bool> refractive_caustics_expression,
    SurfaceClosureCollector &collector) const noexcept {
    if (!_program) {
        collector.begin(point.shading_normal);
        collector.finish();
        return {.shading_normal = point.shading_normal};
    }
    const auto values = trace_values(services, point);
    collector.begin(values.shading_normal);
    for_each_physical_closure(
        services,
        point,
        values,
        Bool{reflective_caustics_expression},
        Bool{refractive_caustics_expression},
        [&](const TracedClosure &closure) noexcept {
            collector.add(canonical_surface_closure(closure));
        });
    collector.finish();
    return {.shading_normal = values.shading_normal};
}

void GraphSurfaceImplementation::for_each_physical_closure(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedValues &values,
    Bool reflective_caustics,
    Bool refractive_caustics,
    const ClosureVisitor &function) const noexcept {
    const auto incoming =
        safe_normalize(point.incoming, point.shading_normal);
    const PrincipledLayerComponent principled_layers{services, point};
    const PrincipledBaseComponent principled_base{services, point};
    const MicrofacetGlassComponent microfacet_glass{services, point};
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
                const auto coat = principled_layers.evaluate_coat(
                    graph_closure,
                    closure.weight,
                    reflective_caustics);
                function(coat.closure);
                closure.weight = coat.lower_weight;
                const auto base = principled_base.evaluate(
                    closure, reflective_caustics, refractive_caustics);
                function(base.metallic);
                function(base.transmission);
                function(base.dielectric);

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
                    base.diffuse_weight * radius_tint;
                const auto radius_weighted_average =
                    sample_weight(radius_weighted);
                const auto energy_normalized =
                    radius_weighted *
                    (sample_weight(base.diffuse_weight) /
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
                    base.diffuse_weight,
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
                diffuse.albedo = base.diffuse_weight;
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
                const auto color =
                    max(graph_closure.color, make_float3(0.0f));
                const auto original_ior = max(graph_closure.ior, 1.0e-5f);
                function(microfacet_glass.setup(
                    {.prototype = closure,
                        .weight = closure.weight,
                        .normal = graph_closure.normal,
                        .roughness = graph_closure.roughness,
                        .ior = original_ior,
                        .fresnel_f0 = make_float3(f0_from_ior(original_ior)),
                        .fresnel_f90 = make_float3(1.0f),
                        .reflection_tint = select(make_float3(0.0f),
                            color,
                            reflective_caustics),
                        .transmission_tint = select(make_float3(0.0f),
                            color,
                            refractive_caustics),
                        .enabled = reflective_caustics | refractive_caustics,
                        .principled_lobe = PrincipledLobe::none,
                        .preserve_energy = graph_closure.preserve_ggx_energy,
                        .beckmann = graph_closure.beckmann}));
                return;
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

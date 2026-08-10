#include "graph_surface_internal.h"
#include "bssrdf_closure_component.h"
#include "microfacet_glass_component.h"
#include "principled_base_component.h"
#include "principled_diffuse_component.h"
#include "principled_layer_component.h"
#include "thin_subsurface_component.h"

#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] SurfaceBssrdfMethod surface_bssrdf_method(
    compiler::BssrdfMethod method) noexcept {
    switch (method) {
        case compiler::BssrdfMethod::burley:
            return SurfaceBssrdfMethod::burley;
        case compiler::BssrdfMethod::random_walk:
            return SurfaceBssrdfMethod::random_walk;
        case compiler::BssrdfMethod::random_walk_legacy:
            return SurfaceBssrdfMethod::random_walk_legacy;
        case compiler::BssrdfMethod::random_walk_skin:
            return SurfaceBssrdfMethod::random_walk_skin;
    }
    return SurfaceBssrdfMethod::random_walk;
}

[[nodiscard]] bool has_principled_feature(
    const TracedClosure &closure,
    compiler::PrincipledClosureFeature feature) noexcept {
    return (closure.principled_features &
            compiler::principled_closure_feature_bit(feature)) != 0u;
}

}// namespace

SurfaceClosureRecord canonical_surface_closure(
    const TracedClosure &closure) noexcept {
    auto result = SurfaceClosureRecord::zero();
    if (closure.physical_kind != SurfaceClosureKind::none) {
        result.kind = static_cast<std::uint32_t>(
            closure.physical_kind);
    } else {
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
        case compiler::ClosureOperation::refraction:
            result.kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::refraction);
            break;
        case compiler::ClosureOperation::transparent:
            result.kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::transparent);
            break;
        case compiler::ClosureOperation::subsurface:
            result.kind = static_cast<std::uint32_t>(
                SurfaceClosureKind::bssrdf);
            break;
        case compiler::ClosureOperation::null_closure:
        case compiler::ClosureOperation::emission:
        case compiler::ClosureOperation::add:
        case compiler::ClosureOperation::mix:
            break;
        }
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
    if (closure.operation == compiler::ClosureOperation::glass ||
        closure.operation == compiler::ClosureOperation::refraction) {
        result.reflection_albedo = closure.reflection_albedo;
        result.transmission_albedo = closure.transmission_albedo;
        result.fresnel_f0 = closure.fresnel_f0;
        result.fresnel_f90 = closure.fresnel_f90;
        result.reflection_tint = closure.reflection_tint;
        result.transmission_tint = closure.transmission_tint;
    }
    if (closure.operation == compiler::ClosureOperation::principled ||
        closure.operation == compiler::ClosureOperation::glossy ||
        closure.operation == compiler::ClosureOperation::glass ||
        closure.operation == compiler::ClosureOperation::refraction) {
        result.preserve_ggx_energy = closure.preserve_ggx_energy;
    }
    if (closure.operation == compiler::ClosureOperation::glossy ||
        closure.operation == compiler::ClosureOperation::glass ||
        closure.operation == compiler::ClosureOperation::refraction) {
        result.beckmann = closure.beckmann;
    }
    if (closure.operation == compiler::ClosureOperation::subsurface) {
        result.bssrdf_method = static_cast<std::uint32_t>(
            surface_bssrdf_method(closure.subsurface_method));
        result.bssrdf_radius = closure.subsurface_radius;
        result.bssrdf_albedo = closure.color;
        result.bssrdf_ior = closure.subsurface_ior;
        result.bssrdf_roughness = closure.roughness;
        result.bssrdf_anisotropy = closure.subsurface_anisotropy;
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
    return collect_traced_closures(
        services,
        point,
        values,
        reflective_caustics_expression,
        refractive_caustics_expression,
        collector);
}

SurfaceClosureCollection
GraphSurfaceImplementation::collect_traced_closures(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedValues &values,
    Expr<bool> reflective_caustics_expression,
    Expr<bool> refractive_caustics_expression,
    SurfaceClosureCollector &collector) const noexcept {
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
    const PrincipledDiffuseComponent principled_diffuse{services};
    const MicrofacetGlassComponent microfacet_glass{services, point};
    const BssrdfClosureComponent bssrdf_closure{point};
    const ThinSubsurfaceComponent thin_subsurface;
    luisa::vector<TracedClosure> physical_closures;
    const auto emit = [&](const TracedClosure &closure) noexcept {
        physical_closures.emplace_back(closure);
    };
    for_each_closure(
        values, [&](const TracedClosure &graph_closure) noexcept {
            auto closure = graph_closure;
            closure.principled_lobe = PrincipledLobe::none;
            closure.setup_valid = true;
            closure.evaluation_scale = make_float3(1.0f);

            const auto produces_transparency =
                graph_closure.operation ==
                    compiler::ClosureOperation::transparent ||
                (graph_closure.operation ==
                     compiler::ClosureOperation::principled &&
                 has_principled_feature(
                     graph_closure,
                     compiler::PrincipledClosureFeature::alpha));
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
                auto transparent = graph_closure;
                transparent.operation =
                    compiler::ClosureOperation::transparent;
                transparent.physical_kind = SurfaceClosureKind::none;
                transparent.principled_lobe = PrincipledLobe::none;
                transparent.weight = contribution.weight;
                transparent.allocation_weight = select(
                    0.0f,
                    contribution.sample_weight,
                    contribution_allocated);
                transparent.sample_weight =
                    transparent.allocation_weight;
                transparent.setup_valid = true;
                transparent.albedo = transparent.weight;
                transparent.color = make_float3(1.0f);
                transparent.normal = point.shading_normal;
                transparent.roughness = 0.0f;
                transparent.ior = 1.0f;
                transparent.evaluation_scale = make_float3(1.0f);
                emit(transparent);
            }

            switch (graph_closure.operation) {
            case compiler::ClosureOperation::principled: {
                closure.weight =
                    has_principled_feature(graph_closure,
                                           compiler::PrincipledClosureFeature::alpha) ?
                        evaluate_principled_alpha_layer(graph_closure).lower_weight :
                        graph_closure.weight;
                if (has_principled_feature(graph_closure,
                                           compiler::PrincipledClosureFeature::sheen)) {
                    const auto sheen =
                        principled_layers.evaluate_sheen(graph_closure, closure.weight);
                    emit(sheen.closure);
                    closure.weight = sheen.lower_weight;
                }
                if (has_principled_feature(graph_closure,
                                           compiler::PrincipledClosureFeature::coat)) {
                    const auto coat = principled_layers.evaluate_coat(
                        graph_closure, closure.weight, reflective_caustics);
                    emit(coat.closure);
                    closure.weight = coat.lower_weight;
                }
                const auto base =
                    principled_base.evaluate(closure, graph_closure.principled_features,
                                             reflective_caustics, refractive_caustics);
                if (has_principled_feature(
                        graph_closure, compiler::PrincipledClosureFeature::metallic)) {
                    emit(*base.metallic);
                }
                if (has_principled_feature(
                        graph_closure,
                        compiler::PrincipledClosureFeature::thick_transmission)) {
                    emit(*base.transmission);
                }
                if (has_principled_feature(
                        graph_closure,
                        compiler::PrincipledClosureFeature::thin_transmission)) {
                    emit(*base.thin_glass_reflection);
                    emit(*base.thin_glass_transmission);
                    emit(*base.thin_glass_transparency);
                }
                if (has_principled_feature(
                        graph_closure, compiler::PrincipledClosureFeature::dielectric)) {
                    emit(*base.dielectric);
                }

                const auto subsurface_weight =
                    clamp(graph_closure.subsurface_weight, 0.0f, 1.0f);
                const auto thin_subsurface_enabled = has_principled_feature(
                    graph_closure, compiler::PrincipledClosureFeature::thin_subsurface);
                const auto thick_subsurface_enabled = has_principled_feature(
                    graph_closure, compiler::PrincipledClosureFeature::thick_subsurface);
                if (thin_subsurface_enabled || thick_subsurface_enabled) {
                    auto bssrdf = closure;
                    bssrdf.operation = compiler::ClosureOperation::subsurface;
                    bssrdf.principled_lobe = PrincipledLobe::none;
                    const auto subsurface_closure_weight =
                        min(max(graph_closure.color, make_float3(0.0f)),
                            make_float3(1.0f)) *
                        subsurface_weight * base.base_weight;
                    bssrdf.weight = subsurface_closure_weight;
                    bssrdf.color =
                        min(max(graph_closure.color, make_float3(0.0f)), make_float3(1.0f));
                    bssrdf.normal = graph_closure.normal;
                    const auto surface_roughness =
                        clamp(graph_closure.roughness, 0.0f, 1.0f);
                    bssrdf.roughness = surface_roughness * surface_roughness;
                    bssrdf.subsurface_radius = graph_closure.subsurface_radius;
                    bssrdf.subsurface_scale = graph_closure.subsurface_scale;
                    bssrdf.subsurface_method = graph_closure.subsurface_method;
                    bssrdf.subsurface_ior = graph_closure.subsurface_method ==
                                                    compiler::BssrdfMethod::random_walk_skin ?
                                                graph_closure.subsurface_ior :
                                                adjusted_ior(graph_closure).eta;
                    bssrdf.subsurface_anisotropy = graph_closure.subsurface_anisotropy;
                    if (thin_subsurface_enabled) {
                        const auto thin_subsurface_setup = thin_subsurface.setup(
                            bssrdf, subsurface_closure_weight,
                            graph_closure.thin_wall &
                                (subsurface_weight > cycles_closure::closure_weight_cutoff));
                        emit(thin_subsurface_setup.reflection);
                        emit(thin_subsurface_setup.smooth_transmission);
                        emit(thin_subsurface_setup.rough_transmission);
                    }

                    if (thick_subsurface_enabled) {
                        bssrdf.weight = select(subsurface_closure_weight, make_float3(0.0f),
                                               graph_closure.thin_wall);
                        bssrdf.normal = maybe_ensure_valid_specular_reflection(
                            point, incoming, graph_closure.normal);
                        const auto bssrdf_setup = bssrdf_closure.setup(bssrdf);
                        emit(bssrdf_setup.bssrdf);
                        emit(bssrdf_setup.diffuse_fallback);
                    }
                }

                if (has_principled_feature(graph_closure,
                                           compiler::PrincipledClosureFeature::diffuse)) {
                    const auto setup = principled_diffuse.setup(
                        {.lower_weight = base.base_weight,
                         .color = graph_closure.color,
                         .subsurface_weight =
                             graph_closure.subsurface_weight});
                    auto diffuse = closure;
                    diffuse.operation = compiler::ClosureOperation::diffuse;
                    diffuse.principled_lobe = PrincipledLobe::none;
                    diffuse.weight = setup.weight;
                    diffuse.allocation_weight = setup.allocation_weight;
                    diffuse.sample_weight = setup.sample_weight;
                    diffuse.albedo = setup.weight;
                    diffuse.roughness = graph_closure.diffuse_roughness;
                    diffuse.evaluation_scale = make_float3(1.0f);
                    emit(diffuse);
                }
                return;
            }
            case compiler::ClosureOperation::subsurface: {
                closure.normal = maybe_ensure_valid_specular_reflection(
                    point, incoming, graph_closure.normal);
                const auto setup = bssrdf_closure.setup(closure);
                emit(setup.bssrdf);
                emit(setup.diffuse_fallback);
                return;
            }
            case compiler::ClosureOperation::glossy: {
                const auto allocated_weight =
                    max(closure.weight, make_float3(0.0f));
                const auto allocation_weight =
                    sample_weight(allocated_weight);
                const auto allocated =
                    reflective_caustics &
                    (allocation_weight >=
                        cycles_closure::closure_weight_cutoff);
                // Cycles' standalone reflective closures apply the current
                // path's caustics predicate before bsdf_alloc(). Preserve
                // that allocation topology: a disabled closure must not
                // enter the closure count, mixture PDF, runtime flags, or
                // any collector merely as a zero-contribution record.
                closure.allocation_weight = select(
                    0.0f, allocation_weight, allocated);
                closure.normal = maybe_ensure_valid_specular_reflection(
                    point, incoming, graph_closure.normal);
                // Standalone Glossy stores authored Color in the Cycles
                // closure weight. Its GGX Fresnel is constant one; MULTI_GGX
                // only adds the tabulated energy-preservation correction.
                const auto incoming_cosine = clamp(
                    dot(closure.normal, incoming), 0.0f, 1.0f);
                const auto energy = ggx_energy(services,
                    closure,
                    incoming_cosine,
                    max(closure.color, make_float3(0.0f)));
                closure.weight = select(make_float3(0.0f),
                    allocated_weight * energy.darkening,
                    allocated);
                closure.albedo = closure.weight;
                closure.sample_weight = select(0.0f,
                    allocation_weight *
                        sample_weight(energy.darkening),
                    allocated);
                closure.evaluation_scale = energy.energy_scale;
                break;
            }
            case compiler::ClosureOperation::glass: {
                const auto color =
                    max(graph_closure.color, make_float3(0.0f));
                const auto original_ior = max(graph_closure.ior, 1.0e-5f);
                emit(microfacet_glass.setup(
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
            case compiler::ClosureOperation::refraction: {
                emit(microfacet_glass.setup(
                    {.prototype = closure,
                        .weight = closure.weight,
                        .normal = graph_closure.normal,
                        .roughness = graph_closure.roughness,
                        .ior = max(graph_closure.ior, 1.0e-5f),
                        .fresnel_f0 = make_float3(0.0f),
                        .fresnel_f90 = make_float3(0.0f),
                        .reflection_tint = make_float3(0.0f),
                        .transmission_tint = make_float3(1.0f),
                        .enabled = refractive_caustics,
                        .principled_lobe = PrincipledLobe::none,
                        .refraction_only = true,
                        .preserve_energy = false,
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
            emit(closure);
        });

    // bsdf_transparent_setup merges every allocated transparent contribution
    // into the first transparent closure, including smooth thin glass on a
    // non-camera ray. Reproduce that allocation topology after all physical
    // components have expanded, without evaluating any material value on the
    // host.
    Float3 transparent_weight = make_float3(0.0f);
    Float transparent_sample_weight = 0.0f;
    for (const auto &closure : physical_closures) {
        if (closure.operation ==
            compiler::ClosureOperation::transparent) {
            transparent_weight += closure.weight;
            transparent_sample_weight += closure.sample_weight;
        }
    }
    Bool transparent_allocated = false;
    for (const auto &physical_closure : physical_closures) {
        if (physical_closure.operation !=
            compiler::ClosureOperation::transparent) {
            function(physical_closure);
            continue;
        }
        const auto contribution_allocated =
            physical_closure.sample_weight >=
            cycles_closure::closure_weight_cutoff;
        const auto emit_here =
            contribution_allocated & !transparent_allocated;
        auto transparent = physical_closure;
        transparent.weight = select(
            make_float3(0.0f), transparent_weight, emit_here);
        transparent.allocation_weight = select(
            0.0f, transparent_sample_weight, emit_here);
        transparent.sample_weight = transparent.allocation_weight;
        function(transparent);
        transparent_allocated |= contribution_allocated;
    }
}

} // namespace psycles::luisa_backend::detail

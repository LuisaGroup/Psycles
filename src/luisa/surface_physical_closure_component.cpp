#include "graph_surface_internal.h"

#include "bssrdf_closure_component.h"
#include "microfacet_anisotropy.h"
#include "microfacet_glass_component.h"
#include "metallic_closure_component.h"
#include "principled_base_component.h"
#include "principled_diffuse_component.h"
#include "principled_layer_component.h"
#include "sheen_closure_component.h"
#include "thin_subsurface_component.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/surface_closure_reachability.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] bool has_principled_feature(
    const TracedClosure &closure,
    compiler::PrincipledClosureFeature feature) noexcept {
    return (closure.principled_features &
            compiler::principled_closure_feature_bit(feature)) != 0u;
}

} // namespace

void expand_physical_surface_closure(
    const ShaderServices &services,
    const SurfacePoint &point,
    const TracedClosure &graph_closure,
    Bool reflective_caustics,
    Bool refractive_caustics,
    const ClosureVisitor &emit) noexcept {
    static_assert(
        static_cast<std::uint32_t>(compiler::ClosureOperation::refraction) < 32u);
    const auto source_operation = graph_closure.operation;
    const auto source_features = graph_closure.principled_features;
    const auto source_operation_bit =
        std::uint32_t{1u} << static_cast<std::uint32_t>(source_operation);
    const auto anisotropic_operations =
        graph_closure.anisotropy_enabled ? source_operation_bit : 0u;
    const auto anisotropic_principled_features =
        graph_closure.anisotropy_enabled &&
                source_operation == compiler::ClosureOperation::principled
            ? source_features
            : 0u;
    const auto thin_film_operations =
        graph_closure.thin_film_enabled ? source_operation_bit : 0u;
    const auto thin_film_principled_features =
        graph_closure.thin_film_enabled &&
                source_operation == compiler::ClosureOperation::principled
            ? source_features
            : 0u;
    const auto source_reachability = reachable_surface_closures(
        source_operation_bit,
        source_features,
        anisotropic_operations,
        anisotropic_principled_features,
        thin_film_operations,
        thin_film_principled_features);
    const ClosureVisitor checked_emit =
        [&emit,
         source_reachability,
         source_operation,
         source_features](const TracedClosure &physical) noexcept {
            const auto identity = canonical_surface_closure_identity(physical);
            const auto kind_reachable =
                identity.kind == SurfaceClosureKind::none ||
                source_reachability.contains(identity.kind);
            const auto lobe_reachable =
                identity.kind != SurfaceClosureKind::principled ||
                source_reachability.contains_principled_lobe(identity.lobe);
            LUISA_ASSERT(
                kind_reachable && lobe_reachable,
                "Physical closure reachability omitted emitted identity "
                "kind={} lobe={} from source operation={} features=0x{:08x}.",
                static_cast<std::uint32_t>(identity.kind),
                static_cast<std::uint32_t>(identity.lobe),
                static_cast<std::uint32_t>(source_operation),
                source_features);
            emit(physical);
        };
    const SurfaceClosurePoint closure_point{point};
    const auto incoming =
        safe_normalize(point.incoming, point.shading_normal);
    const PrincipledLayerComponent principled_layers{services, point};
    const PrincipledBaseComponent principled_base{services, point};
    const PrincipledDiffuseComponent principled_diffuse{services};
    const MicrofacetGlassComponent microfacet_glass{
        services, closure_point};
    const MetallicClosureComponent metallic_closure{
        services, closure_point};
    const SheenClosureComponent sheen_closure{services, point};
    const BssrdfClosureComponent bssrdf_closure{point};
    const ThinSubsurfaceComponent thin_subsurface;

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
        checked_emit(transparent);
    }

    switch (graph_closure.operation) {
    case compiler::ClosureOperation::principled: {
        closure.weight =
            has_principled_feature(
                graph_closure,
                compiler::PrincipledClosureFeature::alpha)
                ? evaluate_principled_alpha_layer(
                      graph_closure)
                      .lower_weight
                : graph_closure.weight;
        if (has_principled_feature(
                graph_closure,
                compiler::PrincipledClosureFeature::sheen)) {
            const auto sheen = principled_layers.evaluate_sheen(
                graph_closure, closure.weight);
            checked_emit(sheen.closure);
            closure.weight = sheen.lower_weight;
        }
        if (has_principled_feature(
                graph_closure,
                compiler::PrincipledClosureFeature::coat)) {
            const auto coat = principled_layers.evaluate_coat(
                graph_closure,
                closure.weight,
                reflective_caustics);
            checked_emit(coat.closure);
            closure.weight = coat.lower_weight;
        }
        const auto base = principled_base.evaluate(
            closure,
            graph_closure.principled_features,
            reflective_caustics,
            refractive_caustics);
        if (has_principled_feature(
                graph_closure,
                compiler::PrincipledClosureFeature::metallic)) {
            checked_emit(*base.metallic);
        }
        if (has_principled_feature(
                graph_closure,
                compiler::PrincipledClosureFeature::thick_transmission)) {
            checked_emit(*base.transmission);
        }
        if (has_principled_feature(
                graph_closure,
                compiler::PrincipledClosureFeature::thin_transmission)) {
            checked_emit(*base.thin_glass_reflection);
            checked_emit(*base.thin_glass_transmission);
            checked_emit(*base.thin_glass_transparency);
        }
        if (has_principled_feature(
                graph_closure,
                compiler::PrincipledClosureFeature::dielectric)) {
            checked_emit(*base.dielectric);
        }

        const auto subsurface_weight =
            clamp(graph_closure.subsurface_weight, 0.0f, 1.0f);
        const auto thin_subsurface_enabled = has_principled_feature(
            graph_closure,
            compiler::PrincipledClosureFeature::thin_subsurface);
        const auto thick_subsurface_enabled = has_principled_feature(
            graph_closure,
            compiler::PrincipledClosureFeature::thick_subsurface);
        if (thin_subsurface_enabled || thick_subsurface_enabled) {
            auto bssrdf = closure;
            bssrdf.operation =
                compiler::ClosureOperation::subsurface;
            bssrdf.principled_lobe = PrincipledLobe::none;
            const auto subsurface_closure_weight =
                min(max(
                        graph_closure.color,
                        make_float3(0.0f)),
                    make_float3(1.0f)) *
                subsurface_weight * base.base_weight;
            bssrdf.weight = subsurface_closure_weight;
            bssrdf.color = min(
                max(graph_closure.color, make_float3(0.0f)),
                make_float3(1.0f));
            bssrdf.normal = graph_closure.normal;
            const auto surface_roughness =
                clamp(graph_closure.roughness, 0.0f, 1.0f);
            bssrdf.roughness =
                surface_roughness * surface_roughness;
            bssrdf.subsurface_radius =
                graph_closure.subsurface_radius;
            bssrdf.subsurface_scale =
                graph_closure.subsurface_scale;
            bssrdf.subsurface_method =
                graph_closure.subsurface_method;
            bssrdf.subsurface_ior =
                graph_closure.subsurface_method ==
                        compiler::BssrdfMethod::random_walk_skin
                    ? graph_closure.subsurface_ior
                    : adjusted_ior(graph_closure).eta;
            bssrdf.subsurface_anisotropy =
                graph_closure.subsurface_anisotropy;
            if (thin_subsurface_enabled) {
                const auto thin_subsurface_setup =
                    thin_subsurface.setup(
                        bssrdf,
                        subsurface_closure_weight,
                        graph_closure.thin_wall &
                            (subsurface_weight >
                             cycles_closure::closure_weight_cutoff));
                checked_emit(thin_subsurface_setup.reflection);
                checked_emit(thin_subsurface_setup.smooth_transmission);
                checked_emit(thin_subsurface_setup.rough_transmission);
            }

            if (thick_subsurface_enabled) {
                bssrdf.weight = select(
                    subsurface_closure_weight,
                    make_float3(0.0f),
                    graph_closure.thin_wall);
                bssrdf.normal =
                    maybe_ensure_valid_specular_reflection(
                        closure_point,
                        incoming,
                        graph_closure.normal);
                const auto bssrdf_setup =
                    bssrdf_closure.setup(bssrdf);
                checked_emit(bssrdf_setup.bssrdf);
                checked_emit(bssrdf_setup.diffuse_fallback);
            }
        }

        if (has_principled_feature(
                graph_closure,
                compiler::PrincipledClosureFeature::diffuse)) {
            const auto setup = principled_diffuse.setup(
                {.lower_weight = base.base_weight,
                 .color = graph_closure.color,
                 .subsurface_weight =
                     graph_closure.subsurface_weight});
            auto diffuse = closure;
            diffuse.operation =
                compiler::ClosureOperation::diffuse;
            diffuse.principled_lobe = PrincipledLobe::none;
            diffuse.weight = setup.weight;
            diffuse.allocation_weight = setup.allocation_weight;
            diffuse.sample_weight = setup.sample_weight;
            diffuse.albedo = setup.weight;
            diffuse.roughness =
                graph_closure.diffuse_roughness;
            diffuse.evaluation_scale = make_float3(1.0f);
            checked_emit(diffuse);
        }
        return;
    }
    case compiler::ClosureOperation::subsurface: {
        closure.normal = maybe_ensure_valid_specular_reflection(
            closure_point, incoming, graph_closure.normal);
        const auto setup = bssrdf_closure.setup(closure);
        checked_emit(setup.bssrdf);
        checked_emit(setup.diffuse_fallback);
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
        // Cycles applies the path's caustics predicate before bsdf_alloc().
        // A disabled closure must not enter the closure count, mixture PDF,
        // runtime flags, or any collector as a zero-contribution record.
        closure.allocation_weight = select(
            0.0f, allocation_weight, allocated);
        closure.normal = maybe_ensure_valid_specular_reflection(
            closure_point, incoming, graph_closure.normal);
        configure_microfacet_state(
            closure,
            glossy_microfacet_state(closure, closure.normal));
        const auto incoming_cosine = clamp(
            dot(closure.normal, incoming), 0.0f, 1.0f);
        const auto energy = ggx_energy(
            services,
            closure,
            incoming_cosine,
            max(closure.color, make_float3(0.0f)));
        // Standalone Glossy stores authored Color in the closure weight. Its
        // GGX Fresnel is constant one; MULTI_GGX only adds the tabulated
        // energy-preservation correction.
        closure.weight = select(
            make_float3(0.0f),
            allocated_weight * energy.darkening,
            allocated);
        closure.albedo = closure.weight;
        closure.sample_weight = select(
            0.0f,
            allocation_weight * sample_weight(energy.darkening),
            allocated);
        closure.evaluation_scale = energy.energy_scale;
        break;
    }
    case compiler::ClosureOperation::metallic_f82:
    case compiler::ClosureOperation::metallic_conductor:
        checked_emit(metallic_closure.setup(
            graph_closure, reflective_caustics));
        return;
    case compiler::ClosureOperation::sheen_microfiber:
    case compiler::ClosureOperation::sheen_ashikhmin:
        checked_emit(sheen_closure.setup(graph_closure));
        return;
    case compiler::ClosureOperation::glass: {
        const auto color =
            max(graph_closure.color, make_float3(0.0f));
        const auto original_ior = max(
            graph_closure.ior, 1.0e-5f);
        checked_emit(microfacet_glass.setup(
            {.prototype = closure,
             .weight = closure.weight,
             .normal = graph_closure.normal,
             .roughness = graph_closure.roughness,
             .ior = original_ior,
             .thin_film_thickness = graph_closure.thin_film_thickness,
             .thin_film_ior = graph_closure.thin_film_ior,
             .fresnel_f0 =
                 make_float3(f0_from_ior(original_ior)),
             .fresnel_f90 = make_float3(1.0f),
             .reflection_tint = select(
                 make_float3(0.0f),
                 color,
                 reflective_caustics),
             .transmission_tint = select(
                 make_float3(0.0f),
                 color,
                 refractive_caustics),
             .enabled =
                 reflective_caustics | refractive_caustics,
             .principled_lobe = PrincipledLobe::none,
             .thin_film_enabled = graph_closure.thin_film_enabled,
             .preserve_energy =
                 graph_closure.preserve_ggx_energy,
             .beckmann = graph_closure.beckmann}));
        return;
    }
    case compiler::ClosureOperation::refraction: {
        checked_emit(microfacet_glass.setup(
            {.prototype = closure,
             .weight = closure.weight,
             .normal = graph_closure.normal,
             .roughness = graph_closure.roughness,
             .ior = max(graph_closure.ior, 1.0e-5f),
             .thin_film_thickness = 0.0f,
             .thin_film_ior = 0.0f,
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
        closure.weight = select(
            make_float3(0.0f),
            max(closure.weight, make_float3(0.0f)),
            allocated);
        // Cycles currently applies the specular-reflection normal correction
        // to translucent setup as well.
        closure.normal = maybe_ensure_valid_specular_reflection(
            closure_point, incoming, graph_closure.normal);
        closure.albedo = closure.weight;
        closure.sample_weight = select(
            0.0f, closure.allocation_weight, allocated);
        break;
    }
    case compiler::ClosureOperation::diffuse: {
        closure.allocation_weight = sample_weight(
            max(closure.weight, make_float3(0.0f)));
        const auto allocated =
            closure.allocation_weight >=
            cycles_closure::closure_weight_cutoff;
        closure.weight = select(
            make_float3(0.0f),
            max(closure.weight, make_float3(0.0f)),
            allocated);
        closure.albedo = closure.weight;
        closure.sample_weight = select(
            0.0f, closure.allocation_weight, allocated);
        break;
    }
    case compiler::ClosureOperation::transparent:
        return;
    case compiler::ClosureOperation::emission:
    case compiler::ClosureOperation::null_closure:
    case compiler::ClosureOperation::add:
    case compiler::ClosureOperation::mix:
        closure.allocation_weight = 0.0f;
        closure.albedo = make_float3(0.0f);
        closure.sample_weight = 0.0f;
        break;
    }
    checked_emit(closure);
}

} // namespace psycles::luisa_backend::detail

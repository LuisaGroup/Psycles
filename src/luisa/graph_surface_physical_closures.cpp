#include "graph_surface_internal.h"

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

} // namespace

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
    const auto values = trace_surface_values(
        services, point, &_value_dependency_plan.physical);
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
    luisa::vector<TracedClosure> physical_closures;
    const auto append_physical_closure =
        [&](const TracedClosure &closure) noexcept {
            physical_closures.emplace_back(closure);
        };
    for_each_closure(
        values,
        _value_dependency_plan.physical_closures,
        _value_dependency_plan.physical,
        [&](const TracedClosure &graph_closure) noexcept {
            expand_physical_surface_closure(
                services,
                point,
                graph_closure,
                reflective_caustics,
                refractive_caustics,
                append_physical_closure);
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

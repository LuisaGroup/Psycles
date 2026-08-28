#include "hair_closure_component.h"

#include <psycles/luisa/cycles_closure.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::detail {

TracedClosure HairClosureComponent::setup(
    const TracedClosure &graph_closure) const noexcept {
    auto closure = graph_closure;
    closure.physical_kind =
        graph_closure.operation ==
                compiler::ClosureOperation::hair_reflection
            ? SurfaceClosureKind::hair_reflection
            : SurfaceClosureKind::hair_transmission;
    closure.principled_lobe = PrincipledLobe::none;
    closure.allocation_weight = sample_weight(
        max(graph_closure.weight, make_float3(0.0f)));
    const auto allocated =
        closure.allocation_weight >=
        cycles_closure::closure_weight_cutoff;
    closure.weight = select(
        make_float3(0.0f),
        max(graph_closure.weight, make_float3(0.0f)),
        allocated);
    closure.sample_weight = select(
        0.0f, closure.allocation_weight, allocated);
    closure.setup_valid = true;
    closure.albedo = closure.weight;
    if (closure.physical_kind == SurfaceClosureKind::hair_reflection) {
        closure.reflection_albedo = closure.weight;
        closure.transmission_albedo = make_float3(0.0f);
    } else {
        closure.reflection_albedo = make_float3(0.0f);
        closure.transmission_albedo = closure.weight;
    }
    closure.normal = maybe_ensure_valid_specular_reflection(
        _closure_point,
        safe_normalize(_point.incoming, _point.shading_normal),
        _point.shading_normal);
    closure.roughness = clamp(graph_closure.roughness, 0.001f, 1.0f);
    closure.diffuse_roughness = clamp(
        graph_closure.diffuse_roughness, 0.001f, 1.0f);
    closure.hair_offset = -graph_closure.hair_offset;

    if (graph_closure.hair_tangent_linked) {
        // This is Cycles' linked-input domain: normalize the authored vector
        // exactly. Link topology, never its numerical value, selects it.
        closure.tangent = normalize(graph_closure.tangent);
    } else {
        const auto curve_tangent = safe_normalize(
            _point.dpdu, make_float3(1.0f, 0.0f, 0.0f));
        const auto triangle_tangent = safe_normalize(
            _point.dpdv, make_float3(0.0f, 1.0f, 0.0f));
        closure.tangent = select(
            triangle_tangent, curve_tangent, _point.is_curve);
        // Cycles deliberately disables Offset for an unlinked triangle
        // tangent, but retains the negated authored offset for curves.
        closure.hair_offset = select(
            0.0f, closure.hair_offset, _point.is_curve);
    }
    closure.microfacet_tangent = closure.tangent;
    closure.microfacet_alpha_x = closure.roughness;
    closure.microfacet_alpha_y = closure.diffuse_roughness;
    closure.sheen_transform_a = closure.hair_offset;
    closure.evaluation_scale = make_float3(1.0f);
    set_cycles_closure_identity_after_setup(
        closure,
        closure.physical_kind == SurfaceClosureKind::hair_reflection
            ? cycles_closure::type_hair_reflection
            : cycles_closure::type_hair_transmission);
    return closure;
}

}// namespace psycles::luisa_backend::detail

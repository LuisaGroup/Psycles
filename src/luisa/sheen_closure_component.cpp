#include "sheen_closure_component.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {

SheenLtcState evaluate_sheen_ltc(
    const ShaderServices &services,
    Float3 normal,
    Float3 incoming,
    Float roughness) noexcept {
    const auto canonical_roughness = clamp(roughness, 1.0e-3f, 1.0f);
    const auto cosine = dot(normal, incoming);
    const auto transform_a = cycles_table_2d(
        services,
        cosine,
        canonical_roughness,
        UInt{cycles45_tables::sheen_ltc_offset},
        32u,
        32u);
    const auto transform_b = cycles_table_2d(
        services,
        cosine,
        canonical_roughness,
        UInt{cycles45_tables::sheen_ltc_offset + 32u * 32u},
        32u,
        32u);
    const auto albedo = cycles_table_2d(
        services,
        cosine,
        canonical_roughness,
        UInt{cycles45_tables::sheen_ltc_offset + 2u * 32u * 32u},
        32u,
        32u);
    return {
        .roughness = canonical_roughness,
        .transform_a = transform_a,
        .transform_b = transform_b,
        .albedo = albedo,
        .valid = (abs(transform_a) >= 1.0e-5f) &
                 (albedo >= 1.0e-5f)};
}

SheenClosureComponent::SheenClosureComponent(
    const ShaderServices &services,
    const SurfacePoint &point) noexcept
    : _services{services}, _point{point} {}

TracedClosure SheenClosureComponent::setup(
    const TracedClosure &graph_closure) const noexcept {
    const auto microfiber =
        graph_closure.operation ==
        compiler::ClosureOperation::sheen_microfiber;
    LUISA_ASSERT(
        microfiber ||
            graph_closure.operation ==
                compiler::ClosureOperation::sheen_ashikhmin,
        "SheenClosureComponent requires a standalone Sheen opcode.");

    auto closure = graph_closure;
    const auto allocated_weight =
        max(graph_closure.weight, make_float3(0.0f));
    const auto allocation_weight = sample_weight(allocated_weight);
    const auto allocated =
        allocation_weight >= cycles_closure::closure_weight_cutoff;
    closure.allocation_weight = select(
        0.0f, allocation_weight, allocated);
    closure.normal = graph_closure.normal;
    closure.ior = 1.0f;
    closure.evaluation_scale = make_float3(1.0f);
    closure.preserve_ggx_energy = false;
    closure.beckmann = false;

    if (microfiber) {
        const auto ltc = evaluate_sheen_ltc(
            _services,
            closure.normal,
            _point.incoming,
            // svm_node_closure_bsdf saturates before bsdf_sheen_setup.
            clamp(graph_closure.roughness, 0.0f, 1.0f));
        const auto valid = allocated & ltc.valid;
        const auto final_weight = allocated_weight * ltc.albedo;
        // An invalid LTC consumes the slot already allocated by bsdf_alloc,
        // changes its type to CLOSURE_NONE, and clears only sample_weight.
        closure.weight = select(
            make_float3(0.0f),
            select(allocated_weight, final_weight, valid),
            allocated);
        closure.sample_weight = select(
            0.0f,
            allocation_weight * ltc.albedo,
            valid);
        closure.setup_valid = valid;
        closure.albedo = select(
            make_float3(0.0f), final_weight, valid);
        closure.roughness = ltc.roughness;
        closure.sheen_transform_a = ltc.transform_a;
        closure.sheen_transform_b = ltc.transform_b;
        set_cycles_closure_identity_after_setup(
            closure, cycles_closure::type_sheen);
        return closure;
    }

    // Cycles first saturates sigma, then clamps it to 0.01 in setup. Store
    // the resulting inverse square, the sole analytic parameter observed by
    // Ashikhmin eval/sample, in the common record's model-specific scalar.
    const auto sigma = max(
        clamp(graph_closure.roughness, 0.0f, 1.0f),
        0.01f);
    closure.roughness = 1.0f / (sigma * sigma);
    closure.weight = select(
        make_float3(0.0f), allocated_weight, allocated);
    closure.sample_weight = select(
        0.0f, allocation_weight, allocated);
    closure.setup_valid = allocated;
    closure.albedo = closure.weight;
    closure.sheen_transform_a = 0.0f;
    closure.sheen_transform_b = 0.0f;
    set_cycles_closure_identity_after_setup(
        closure, cycles_closure::type_ashikhmin_velvet);
    return closure;
}

} // namespace psycles::luisa_backend::detail

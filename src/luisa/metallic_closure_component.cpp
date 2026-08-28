#include "metallic_closure_component.h"

#include "microfacet_anisotropy.h"
#include "thin_film_fresnel.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>

namespace psycles::luisa_backend::detail {

MetallicClosureComponent::MetallicClosureComponent(
    const ShaderServices &services,
    const SurfaceClosurePoint &point) noexcept
    : _services{services}, _point{point} {}

TracedClosure MetallicClosureComponent::setup(
    const TracedClosure &graph_closure,
    Bool reflective_caustics) const noexcept {
    const auto f82_model =
        graph_closure.operation ==
        compiler::ClosureOperation::metallic_f82;
    LUISA_ASSERT(
        f82_model ||
            graph_closure.operation ==
                compiler::ClosureOperation::metallic_conductor,
        "MetallicClosureComponent requires a standalone Metallic opcode.");

    auto closure = graph_closure;
    const auto allocated_weight =
        max(graph_closure.weight, make_float3(0.0f));
    const auto allocation_weight = sample_weight(allocated_weight);
    const auto allocated =
        reflective_caustics &
        (allocation_weight >= cycles_closure::closure_weight_cutoff);
    closure.allocation_weight = select(
        0.0f, allocation_weight, allocated);
    const auto incoming = safe_normalize(
        _point.incoming, _point.shading_normal);
    closure.normal = maybe_ensure_valid_specular_reflection(
        _point, incoming, graph_closure.normal);
    closure.roughness = clamp(graph_closure.roughness, 0.0f, 1.0f);
    const auto microfacet_state = metallic_microfacet_state(
        graph_closure, graph_closure.normal);
    configure_microfacet_state(closure, microfacet_state);
    const auto effective_roughness = sqrt(sqrt(max(
        microfacet_state.alpha_x * microfacet_state.alpha_y, 0.0f)));
    const auto incoming_cosine = clamp(
        dot(closure.normal, incoming), 0.0f, 1.0f);

    Float3 fss;
    Float3 albedo_estimate;
    if (f82_model) {
        const auto f0 = clamp(
            graph_closure.color,
            make_float3(0.0f),
            make_float3(1.0f));
        const auto tint = clamp(
            graph_closure.specular_tint,
            make_float3(0.0f),
            make_float3(1.0f));
        const auto b = fresnel_f82_b(f0, tint);
        fss = lerp(f0, make_float3(1.0f), 1.0f / 21.0f) -
              b * (1.0f / 126.0f);
        const auto interpolation = cycles_table_3d(
            _services,
            effective_roughness,
            incoming_cosine,
            0.5f,
            UInt{cycles45_tables::ggx_gen_schlick_s_offset},
            16u,
            16u,
            16u);
        albedo_estimate = lerp(
            f0, make_float3(1.0f), interpolation);
        if (graph_closure.thin_film_enabled) {
            const auto film = thin_film_f82_fresnel(
                _services,
                graph_closure.thin_film_thickness,
                graph_closure.thin_film_ior,
                f0,
                b,
                incoming_cosine);
            albedo_estimate = select(
                albedo_estimate,
                film,
                graph_closure.thin_film_thickness >
                    thin_film_thickness_cutoff);
        }
        closure.color = f0;
        closure.specular_tint = b;
    } else {
        const auto ior = max(
            graph_closure.color, make_float3(0.0f));
        const auto extinction = max(
            graph_closure.specular_tint, make_float3(0.0f));
        fss = fresnel_conductor_fss(ior, extinction);
        albedo_estimate = fresnel_conductor(
            incoming_cosine, ior, extinction);
        if (graph_closure.thin_film_enabled) {
            const auto film = thin_film_conductor_fresnel(
                _services,
                graph_closure.thin_film_thickness,
                graph_closure.thin_film_ior,
                ior,
                extinction,
                incoming_cosine);
            albedo_estimate = select(
                albedo_estimate,
                film,
                graph_closure.thin_film_thickness >
                    thin_film_thickness_cutoff);
        }
        closure.color = ior;
        closure.specular_tint = extinction;
    }

    const auto energy = ggx_energy(
        _services,
        effective_roughness,
        graph_closure.preserve_ggx_energy,
        incoming_cosine,
        fss);
    closure.weight = select(
        make_float3(0.0f),
        allocated_weight * energy.darkening,
        allocated);
    closure.sample_weight = select(
        0.0f,
        allocation_weight * sample_weight(albedo_estimate) *
            sample_weight(energy.darkening),
        allocated);
    closure.setup_valid = true;
    closure.albedo = closure.weight * albedo_estimate;
    closure.reflection_albedo = closure.albedo;
    closure.transmission_albedo = make_float3(0.0f);
    closure.ior = 1.0f;
    closure.evaluation_scale = energy.energy_scale;
    set_cycles_closure_identity_after_setup(
        closure,
        closure.beckmann
            ? UInt{cycles_closure::type_microfacet_beckmann}
            : UInt{cycles_closure::type_microfacet_ggx},
        static_cast<std::uint32_t>(
            f82_model
                ? cycles_closure::MicrofacetFresnel::f82_tint
                : cycles_closure::MicrofacetFresnel::conductor));
    return closure;
}

} // namespace psycles::luisa_backend::detail

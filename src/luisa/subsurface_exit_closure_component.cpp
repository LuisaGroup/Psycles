#include "subsurface_exit_closure_component.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] Bool diffuse_enabled(
    const SurfaceQuery &query) noexcept {
    return (query.lobe_mask &
            static_cast<std::uint32_t>(contract::event_diffuse)) != 0u;
}

[[nodiscard]] SurfaceEvaluation unit_lambert_evaluation(
    Expr<luisa::float3> outgoing_expression,
    Bool include_value,
    Bool preserve_pdf,
    const SurfaceQuery &query) noexcept {
    const auto outgoing_value = Float3{outgoing_expression};
    const auto outgoing = outgoing_value *
                          rsqrt(max(dot(outgoing_value, outgoing_value),
                                    1.0e-20f));
    const auto pdf = max(dot(query.subsurface_normal, outgoing), 0.0f) *
                     cycles_sample_mapping::inverse_pi;
    const auto eligible = diffuse_enabled(query) & (pdf > 0.0f);
    const auto contributes = eligible & include_value;
    auto result = SurfaceEvaluation::zero();
    result.f = select(make_float3(0.0f), make_float3(pdf), contributes);
    result.pdf = select(0.0f, pdf, eligible & preserve_pdf);
    result.diffuse_f = result.f;
    result.diffuse_pdf = select(0.0f, pdf, contributes & preserve_pdf);
    result.events = select(
        static_cast<std::uint32_t>(contract::event_none),
        static_cast<std::uint32_t>(
            contract::event_diffuse | contract::event_reflection),
        contributes);
    return result;
}

}// namespace

UInt SubsurfaceExitClosureComponent::runtime_flags(
    const SurfacePoint &point) const noexcept {
    return cycles_closure::runtime_bsdf |
           cycles_closure::runtime_bsdf_has_eval |
           select(0u,
                  cycles_closure::runtime_backfacing,
                  point.back_facing);
}

SurfaceEvaluation SubsurfaceExitClosureComponent::evaluate_light(
    const SurfacePoint &point,
    Expr<luisa::float3> outgoing,
    const SurfaceQuery &query,
    Expr<std::uint32_t> light_shader_flags_expression) const noexcept {
    using namespace contract::cycles_abi;
    static_cast<void>(point);
    UInt light_shader_flags{light_shader_flags_expression};
    const auto include_value =
        (light_shader_flags & shader_exclude_diffuse) == 0u;
    const auto preserve_pdf =
        (light_shader_flags & shader_use_mis) != 0u;
    return unit_lambert_evaluation(
        outgoing, include_value, preserve_pdf, query);
}

SurfaceSample SubsurfaceExitClosureComponent::sample(
    const SurfacePoint &point,
    Expr<luisa::float2> random,
    const SurfaceQuery &query) const noexcept {
    auto result = SurfaceSample::zero();
    const auto enabled = diffuse_enabled(query);
    const auto sampled = cycles_sample_mapping::sample_cosine_hemisphere(
        query.subsurface_normal, random);
    result.wi = sampled.direction;
    result.evaluation.f = select(
        make_float3(0.0f), make_float3(sampled.pdf), enabled);
    result.evaluation.pdf = select(0.0f, sampled.pdf, enabled);
    result.evaluation.diffuse_f = result.evaluation.f;
    result.evaluation.diffuse_pdf = result.evaluation.pdf;
    result.evaluation.events = select(
        static_cast<std::uint32_t>(contract::event_none),
        static_cast<std::uint32_t>(
            contract::event_diffuse | contract::event_reflection),
        enabled);
    result.runtime_flags = runtime_flags(point);
    result.roughness = make_float2(0.0f);
    result.valid = enabled & (sampled.pdf > 0.0f);
    return result;
}

SurfaceClosureTrace SubsurfaceExitClosureComponent::trace(
    const SurfacePoint &point,
    const SurfaceQuery &query,
    Expr<std::uint32_t> requested_index_expression) const noexcept {
    UInt requested_index{requested_index_expression};
    const auto valid = requested_index == 0u;
    auto result = SurfaceClosureTrace::zero(requested_index);
    result.count = 1u;
    result.runtime_flags = runtime_flags(point);
    result.type = select(
        cycles_closure::type_none,
        cycles_closure::type_diffuse,
        valid);
    result.sample_weight = select(0.0f, 1.0f, valid);
    result.weight = select(
        make_float3(0.0f), make_float3(1.0f), valid);
    result.normal = query.subsurface_normal;
    result.valid = valid;
    return result;
}

SurfaceSampleTrace SubsurfaceExitClosureComponent::sample_trace(
    const SurfacePoint &point,
    Expr<luisa::float2> random,
    const SurfaceQuery &query) const noexcept {
    auto result = SurfaceSampleTrace::zero();
    result.sample = sample(point, random, query);
    result.closure_index = 0u;
    result.closure_type = cycles_closure::type_diffuse;
    result.closure_sample_weight = 1.0f;
    result.selection_rescaled = 0.0f;
    result.closure_weight = make_float3(1.0f);
    result.closure_normal = query.subsurface_normal;
    result.closure_valid = result.sample.valid;
    return result;
}

}// namespace psycles::luisa_backend::detail

#include "path_kernel_builder.h"

#include "cycles_shader_identity.h"
#include "path_kernel_direct_light_trace.h"

#include <psycles/luisa/cycles_ray_differential.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {

DirectLightTransportState DirectLightTransportState::empty() noexcept {
    return {
        .weighted_bsdf = make_float3(0.0f),
        .light_shader = make_float3(0.0f),
        .direction = make_float3(0.0f, 0.0f, 1.0f),
        .target_position = make_float3(0.0f),
        .bsdf = make_float3(0.0f),
        .diffuse_bsdf = make_float3(0.0f),
        .glossy_bsdf = make_float3(0.0f),
        .light_object = surface_ray::invalid_primitive,
        .light_primitive = surface_ray::invalid_primitive,
        .shader_flags = 0u,
        .average_roughness_squared = 0.0f,
        .distant = false,
        .valid = false};
}

void DirectLightTransportState::accept(
    const SurfaceEvaluation &evaluation,
    Float3 proposal_weighted_bsdf,
    Float3 proposal_light_shader,
    Float3 proposal_direction,
    Float3 proposal_target_position,
    Bool proposal_distant,
    UInt proposal_light_object,
    UInt proposal_light_primitive,
    UInt proposal_shader_flags) noexcept {
    weighted_bsdf = proposal_weighted_bsdf;
    light_shader = proposal_light_shader;
    direction = proposal_direction;
    target_position = proposal_target_position;
    bsdf = evaluation.f;
    diffuse_bsdf = evaluation.diffuse_f;
    glossy_bsdf = evaluation.glossy_f;
    light_object = proposal_light_object;
    light_primitive = proposal_light_primitive;
    shader_flags = proposal_shader_flags;
    average_roughness_squared =
        evaluation.average_roughness_squared;
    distant = proposal_distant;
    valid = true;
}

namespace {

class CommonDirectLightTransportStage final
    : public DirectLightTransportStage {

  private:
    std::shared_ptr<const DirectLightTraceRecorder> _trace;

  public:
    explicit CommonDirectLightTransportStage(
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _trace{std::move(trace)} {}

    void emit(
        DirectLightingContext &context,
        const DirectLightTransportState &transport)
        const noexcept override {
        auto &bounce = context.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        auto &surface = context.surface;

        $if(transport.valid) {
            const auto unshadowed =
                transport.weighted_bsdf * transport.light_shader;
            const auto roulette_weight =
                invocation.sample_light_roulette(
                    unshadowed,
                    bounce.random().light_terminate_sample);
            const auto surviving_unshadowed =
                unshadowed * roulette_weight;
            _trace->record_transport(
                bounce,
                {.light_shader = transport.light_shader,
                 .unshadowed =
                     sample.throughput * surviving_unshadowed});
            $if(any(surviving_unshadowed != 0.0f)) {
                const auto shadow =
                    surface.make_shadow_origin(
                        transport.direction);
                Float3 shadow_direction = transport.direction;
                Float shadow_maximum = ray_maximum;
                $if(!transport.distant) {
                    const auto shadow_offset =
                        transport.target_position - shadow.position;
                    const auto shadow_distance = sqrt(max(
                        length_squared(shadow_offset),
                        1.0e-20f));
                    shadow_direction =
                        shadow_offset / shadow_distance;
                    shadow_maximum = shadow_distance;
                };
                Var<luisa::compute::Ray> shadow_ray = make_ray(
                    shadow.position,
                    shadow_direction,
                    0.0f,
                    shadow_maximum);
                const auto source_object = select(
                    surface_ray::invalid_primitive,
                    surface.cycles_object_index,
                    shadow.skip_self);
                const auto source_primitive = select(
                    surface_ray::invalid_primitive,
                    surface.cycles_primitive_index,
                    shadow.skip_self);
                const auto cast_shadow =
                    (transport.shader_flags &
                     cycles_shader_identity::cast_shadow) != 0u;
                const auto shadow_differential =
                    cycles_ray_differential::for_surface_shadow(
                        sample.ray_dD,
                        surface.differential_radius,
                        transport.average_roughness_squared);
                const auto shadow_result = config.trace_shadow(
                    shadow_ray,
                    shadow_differential.position,
                    shadow_differential.direction,
                    source_object,
                    source_primitive,
                    transport.light_object,
                    transport.light_primitive,
                    invocation.parameters.transparent_max_bounces,
                    pack_shader_evaluation_state(
                        cycles_path_state::shadow_shader_state(
                            sample.path_depth,
                            sample.diffuse_depth,
                            sample.glossy_depth,
                            sample.transparent_depth,
                            sample.transmission_depth)));
                const auto transmittance =
                    shadow_result->transmittance;
                _trace->record_shadow(
                    bounce,
                    {.origin = shadow_ray->origin(),
                     .direction = shadow_ray->direction(),
                     .minimum = shadow_ray->t_min(),
                     .maximum = shadow_ray->t_max(),
                     .cast_shadow = cast_shadow,
                     .source_object = source_object,
                     .source_primitive = source_primitive,
                     .skip_self = shadow.skip_self,
                     .light_object = transport.light_object,
                     .light_primitive = transport.light_primitive,
                     .first_hit = shadow_result->first_hit != 0u,
                     .first_object = shadow_result->first_object,
                     .first_primitive = shadow_result->first_primitive,
                     .first_kind = shadow_result->first_kind,
                     .first_distance = shadow_result->first_distance,
                     .first_barycentric =
                         shadow_result->first_barycentric,
                     .transmittance = transmittance});
                $if(any(transmittance > 0.0f)) {
                    const auto unclamped_contribution =
                        sample.throughput * surviving_unshadowed *
                        transmittance;
                    _trace->record_contribution(
                        bounce, unclamped_contribution);
                    const auto contribution =
                        invocation.clamp_contribution(
                            unclamped_contribution,
                            sample.path_depth);
                    sample.accumulate_radiance(contribution);
                    sample.accumulate_light_pass(
                        config.light_transport.split_nee_light(
                            contribution,
                            transport.bsdf,
                            transport.diffuse_bsdf,
                            transport.glossy_bsdf,
                            sample.path_diffuse_weight,
                            sample.path_glossy_weight,
                            sample.path_depth));
                };
            };
        };
    }
};

}// namespace

std::unique_ptr<DirectLightTransportStage>
make_direct_light_transport_stage(
    std::shared_ptr<const DirectLightTraceRecorder> trace) {
    return std::make_unique<CommonDirectLightTransportStage>(
        std::move(trace));
}

}// namespace psycles::luisa_backend::detail

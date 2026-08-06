#include "path_kernel_builder.h"
#include "cycles_shader_identity.h"
#include "path_kernel_direct_light_trace.h"
#include "path_kernel_environment_light.h"

#include <psycles/luisa/cycles_light.h>
#include <psycles/luisa/cycles_ray_differential.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class EnvironmentLightingComponent final : public DirectLightingComponent {

  private:
    std::shared_ptr<
        const EnvironmentLightComponent>
        _environment_light{
            make_environment_light_component()};
    std::shared_ptr<const DirectLightTraceRecorder>
        _trace;

  public:
    explicit EnvironmentLightingComponent(
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _trace{std::move(trace)} {}

    void emit(DirectLightingContext &context) const noexcept override {
        auto &bounce = context.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        auto &surface = context.surface;
        auto &selected_light = bounce.selected_light;
        auto &light_sample = bounce.light_sample;
        auto &light_terminate_sample = bounce.light_terminate_sample;
        auto &hit = bounce.hit;
        auto &surface_tag = surface.surface_tag;
        auto &point = surface.point;
        auto &path_surface_query = surface.path_surface_query;
        auto &throughput = sample.throughput;
        auto &path_depth = sample.path_depth;
        auto &diffuse_depth = sample.diffuse_depth;
        auto &glossy_depth = sample.glossy_depth;
        auto &transparent_depth = sample.transparent_depth;
        auto &transmission_depth = sample.transmission_depth;
        auto &path_diffuse_weight = sample.path_diffuse_weight;
        auto &path_glossy_weight = sample.path_glossy_weight;
        const auto &kernel_parameters = invocation.parameters;
        const auto &trace_shadow = config.trace_shadow;
        const auto &nee_light_weight = config.light_transport.nee_light_weight;
        const auto &split_nee_light = config.light_transport.split_nee_light;
        auto make_surface_shadow_origin = [&](Float3 direction) noexcept {
            return surface.make_shadow_origin(direction);
        };
        auto evaluate_light_surface = [&](UInt tag,
                                          const SurfacePoint &surface_point,
                                          Float3 outgoing,
                                          const SurfaceQuery &query,
                                          UInt shader_flags) noexcept {
            return invocation.evaluate_light_surface(
                tag, surface_point, outgoing, query, shader_flags);
        };
        auto sample_light_roulette = [&](Float3 contribution,
                                         Float random) noexcept {
            return invocation.sample_light_roulette(contribution, random);
        };
        auto clamp_contribution = [&](Float3 contribution,
                                      UInt depth) noexcept {
            return invocation.clamp_contribution(contribution, depth);
        };
        auto accumulate_light_pass =
            [&](Var<LightPassContributionCall> contribution) noexcept {
                sample.accumulate_light_pass(std::move(contribution));
            };
        $if(selected_light.kind ==
            static_cast<std::uint32_t>(
                sampling::LightDistributionEmitterKind::environment)) {
            const auto light =
                _environment_light
                    ->from_position(
                        config.scene,
                        surface.hit_position,
                        light_sample.xy(),
                        selected_light
                            .selection_pdf);
            const auto reached_max_bounces =
                cycles_light::select_reached_max_bounces(
                    path_depth,
                    config.scene->world_max_bounces);
            $if(!reached_max_bounces & light.valid) {
                _trace->record_sample(
                    bounce,
                    {.type = 2u,
                     .emitter_id =
                         selected_light.emitter_id,
                     .primitive =
                         config.scene->light_count,
                     .object =
                         config.scene
                             ->cycles_background_object_index,
                     .light_group =
                         config.scene
                             ->cycles_background_light_group,
                     .shader =
                         config.scene
                             ->cycles_background_shader_id,
                     .pdf = light.pdf,
                     .selection_pdf =
                         selected_light.selection_pdf,
                     .evaluation_factor = 1.0f,
                     .direction = light.direction,
                     .position = -light.direction,
                     .geometric_normal =
                         -light.direction,
                     .distance = ray_maximum});
                Float3 radiance = make_float3(0.0f);
                if (config.scene
                        ->environment_emission_is_constant) {
                    radiance =
                        _environment_light
                            ->evaluate_constant_emission(
                                sample);
                }
                const auto evaluation =
                    evaluate_light_surface(
                        surface_tag,
                        point,
                        light.direction,
                        path_surface_query,
                        config.scene
                            ->cycles_background_shader_flags);
                if (!config.scene
                         ->environment_emission_is_constant) {
                    $if(any(evaluation.f != 0.0f)) {
                        radiance =
                            _environment_light
                                ->evaluate_emission(
                                    sample,
                                    light.direction,
                                    cycles_path_state::
                                        light_emission_shader_state(
                                            path_depth,
                                            diffuse_depth,
                                            glossy_depth,
                                            transparent_depth,
                                            transmission_depth));
                    };
                }
                const auto mis_weight =
                    nee_light_weight(
                        light.pdf,
                        evaluation.pdf);
                _trace->record_evaluation(
                    bounce,
                    {.distance = ray_maximum,
                     .bsdf_pdf = evaluation.pdf,
                     .mis_weight = mis_weight,
                     .bsdf = evaluation.f,
                     .diffuse = evaluation.diffuse_f,
                     .glossy = evaluation.glossy_f});
                const auto constant_emission =
                    config.scene->environment_emission_is_constant;
                Float3 weighted_light = make_float3(1.0f);
                Float3 light_shader_factor = radiance;
                if (constant_emission) {
                    weighted_light = radiance;
                    light_shader_factor = make_float3(1.0f);
                }
                const auto weighted_bsdf =
                    evaluation.f * weighted_light *
                    (mis_weight / light.pdf);
                _trace->record_weighted_bsdf(
                    bounce, weighted_bsdf);
                $if(any(weighted_bsdf != 0.0f)) {
                    const auto unshadowed =
                        weighted_bsdf * light_shader_factor;
                    const auto roulette_weight = sample_light_roulette(
                        unshadowed, light_terminate_sample);
                    const auto surviving_unshadowed =
                        unshadowed * roulette_weight;
                    _trace->record_transport(
                        bounce,
                        {.light_shader = light_shader_factor,
                         .unshadowed =
                             throughput * surviving_unshadowed});
                    $if(any(surviving_unshadowed != 0.0f)) {
                        const auto shadow = make_surface_shadow_origin(
                            light.direction);
                        Var<luisa::compute::Ray> environment_shadow_ray =
                            make_ray(shadow.position,
                                     light.direction,
                                     0.0f,
                                     ray_maximum);
                        const auto source_object = select(
                            surface_ray::invalid_primitive,
                            surface.cycles_object_index,
                            shadow.skip_self);
                        const auto source_primitive = select(
                            surface_ray::invalid_primitive,
                            surface.cycles_primitive_index,
                            shadow.skip_self);
                        const auto light_object =
                            UInt{surface_ray::invalid_primitive};
                        const auto light_primitive =
                            UInt{surface_ray::invalid_primitive};
                        const auto cast_shadow =
                            (config.scene->cycles_background_shader_flags &
                             cycles_shader_identity::cast_shadow) != 0u;
                        const auto shadow_differential =
                            cycles_ray_differential::for_surface_shadow(
                                sample.ray_dD,
                                surface.differential_radius,
                                evaluation.average_roughness_squared);
                        const auto shadow_result = trace_shadow(
                            environment_shadow_ray,
                            shadow_differential.position,
                            shadow_differential.direction,
                            source_object,
                            source_primitive,
                            light_object,
                            light_primitive,
                            kernel_parameters.transparent_max_bounces,
                            pack_shader_evaluation_state(
                                cycles_path_state::shadow_shader_state(
                                    path_depth,
                                    diffuse_depth,
                                    glossy_depth,
                                    transparent_depth,
                                    transmission_depth)));
                        const auto shadow_transmittance =
                            shadow_result->transmittance;
                        _trace->record_shadow(
                            bounce,
                            {.origin = environment_shadow_ray->origin(),
                             .direction =
                                 environment_shadow_ray->direction(),
                             .minimum = environment_shadow_ray->t_min(),
                             .maximum = environment_shadow_ray->t_max(),
                             .cast_shadow = cast_shadow,
                             .source_object = source_object,
                             .source_primitive = source_primitive,
                             .skip_self = shadow.skip_self,
                             .light_object = light_object,
                             .light_primitive = light_primitive,
                             .first_hit =
                                 shadow_result->first_hit != 0u,
                             .first_object =
                                 shadow_result->first_object,
                             .first_primitive =
                                 shadow_result->first_primitive,
                             .first_kind =
                                 shadow_result->first_kind,
                             .first_distance =
                                 shadow_result->first_distance,
                             .first_barycentric =
                                 shadow_result->first_barycentric,
                             .transmittance = shadow_transmittance});
                        $if(any(shadow_transmittance > 0.0f)) {
                            const auto unclamped_contribution =
                                throughput * surviving_unshadowed *
                                shadow_transmittance;
                            _trace->record_contribution(
                                bounce, unclamped_contribution);
                            const auto contribution = clamp_contribution(
                                unclamped_contribution, path_depth);
                            sample.accumulate_radiance(contribution);
                            accumulate_light_pass(split_nee_light(
                                contribution,
                                evaluation.f,
                                evaluation.diffuse_f,
                                evaluation.glossy_f,
                                path_diffuse_weight,
                                path_glossy_weight,
                                path_depth));
                        };
                    };
                };
            }
            $else {
                _trace->record_failed_sample(
                    bounce);
            };
        };
    }
};

} // namespace

std::unique_ptr<DirectLightingComponent>
make_environment_lighting_component(
    std::shared_ptr<const DirectLightTraceRecorder> trace) {
    return std::make_unique<EnvironmentLightingComponent>(
        std::move(trace));
}

} // namespace psycles::luisa_backend::detail

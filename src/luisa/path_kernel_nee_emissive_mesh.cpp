#include "path_kernel_builder.h"
#include "cycles_shader_identity.h"
#include "path_kernel_direct_light_trace.h"
#include "path_kernel_emissive_triangle.h"

#include <psycles/luisa/cycles_ray_differential.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class EmissiveMeshLightingComponent final
    : public DirectLightingComponent {

  private:
    std::shared_ptr<
        const EmissiveTriangleComponent>
        _emissive_triangle{
            make_emissive_triangle_component()};
    std::shared_ptr<const DirectLightTraceRecorder>
        _trace;

  public:
    explicit EmissiveMeshLightingComponent(
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _trace{std::move(trace)} {}

    void emit(
        DirectLightingContext &context)
        const noexcept override {
        auto &bounce = context.bounce;
        auto &sample = bounce.sample;
        auto &invocation =
            sample.invocation;
        const auto &config =
            invocation.config;
        auto &surface =
            context.surface;
        auto &selected_light =
            bounce.selected_light;
        auto &hit =
            bounce.hit;
        const auto selected_mesh =
            selected_light.kind ==
            static_cast<std::uint32_t>(
                sampling::
                    LightDistributionEmitterKind::
                        emissive_triangle);
        $if(selected_mesh) {
            const auto light =
                _emissive_triangle
                    ->from_position(
                        config.scene,
                        selected_light
                            .index,
                        surface
                            .hit_position,
                        bounce
                            .light_sample
                            .xy());
            $if(light.valid) {
                const auto &emitter =
                    light.geometry.emitter;
                _trace->record_sample(
                    bounce,
                    {.type = 5u,
                     .emitter_id =
                         selected_light.emitter_id,
                     .primitive =
                         emitter.cycles_primitive_index,
                     .object =
                         emitter.cycles_object_index,
                     .light_group =
                         emitter.cycles_light_group,
                     .shader =
                         emitter.cycles_shader_id,
                     .pdf = light.pdf,
                     .selection_pdf =
                         selected_light.selection_pdf,
                     .evaluation_factor = 1.0f,
                     .direction =
                         light.light.direction,
                     .position =
                         light.light.position,
                     .geometric_normal =
                         light.geometry.geometric_normal,
                     .distance =
                         light.light.distance});
                const auto is_transmission =
                    dot(
                        light.light.direction,
                        surface.point.shading_normal) <
                    0.0f;
                const auto same_primitive =
                    (emitter.cycles_primitive_index !=
                     surface_ray::invalid_primitive) &
                    (emitter.cycles_primitive_index ==
                     surface.cycles_primitive_index) &
                    (emitter.cycles_object_index ==
                     surface.cycles_object_index);
                const auto oriented_normal =
                    select(
                        surface.point.geometric_normal,
                        -surface.point.geometric_normal,
                        is_transmission);
                const auto reject_self =
                    same_primitive &
                    (dot(
                         light.light.direction,
                         oriented_normal) >
                     0.0f);
                $if(!reject_self) {
                    const auto constant_emission =
                        emitter.emission_is_constant != 0u;
                    Float3 radiance = make_float3(0.0f);
                    // Cycles' constant factor is evaluated after geometric
                    // rejection but before the receiving BSDF.
                    $if(constant_emission) {
                        radiance =
                            _emissive_triangle
                                ->evaluate_constant_emission(
                                    sample,
                                    light);
                    };
                    const auto evaluation =
                        invocation.evaluate_light_surface(
                            surface.surface_tag,
                            surface.point,
                            light.light.direction,
                            surface.path_surface_query,
                            emitter.cycles_shader_flags);
                    // A non-constant light shader is the deferred
                    // SHADE_LIGHT_NEE phase: after receiving-surface
                    // evaluation and before shadow traversal.
                    const auto bsdf_nonzero =
                        any(evaluation.f != 0.0f);
                    $if((!constant_emission) & bsdf_nonzero) {
                        radiance =
                            _emissive_triangle
                                ->evaluate_emission(
                                    sample,
                                    light);
                    };
                    const auto mis_weight =
                        config.light_transport
                            .nee_light_weight(
                                light.pdf,
                                evaluation.pdf);
                    _trace->record_evaluation(
                        bounce,
                        {.distance = light.light.distance,
                         .bsdf_pdf = evaluation.pdf,
                         .mis_weight = mis_weight,
                         .bsdf = evaluation.f,
                         .diffuse = evaluation.diffuse_f,
                         .glossy = evaluation.glossy_f});
                    const auto weighted_light = select(
                        make_float3(1.0f),
                        radiance,
                        constant_emission);
                    const auto light_shader_factor = select(
                        radiance,
                        make_float3(1.0f),
                        constant_emission);
                    const auto weighted_bsdf =
                        evaluation.f * weighted_light *
                        (mis_weight / light.pdf);
                    _trace->record_weighted_bsdf(
                        bounce, weighted_bsdf);
                    $if(any(weighted_bsdf != 0.0f)) {
                        const auto unshadowed =
                            weighted_bsdf * light_shader_factor;
                        const auto roulette_weight =
                            invocation.sample_light_roulette(
                                unshadowed,
                                bounce.light_terminate_sample);
                        const auto surviving_unshadowed =
                            unshadowed * roulette_weight;
                        _trace->record_transport(
                            bounce,
                            {.light_shader = light_shader_factor,
                             .unshadowed = sample.throughput *
                                           surviving_unshadowed});
                        $if(any(surviving_unshadowed != 0.0f)) {
                            const auto shadow =
                                surface.make_shadow_origin(
                                    light.light.direction);
                            const auto shadow_offset =
                                light.light.position - shadow.position;
                            const auto shadow_distance = sqrt(max(
                                length_squared(shadow_offset),
                                1.0e-20f));
                            const auto shadow_direction =
                                shadow_offset / shadow_distance;
                            Var<luisa::compute::Ray> shadow_ray = make_ray(
                                shadow.position,
                                shadow_direction,
                                0.0f,
                                shadow_distance);
                            const auto source_object = select(
                                surface_ray::invalid_primitive,
                                surface.cycles_object_index,
                                shadow.skip_self);
                            const auto source_primitive = select(
                                surface_ray::invalid_primitive,
                                surface.cycles_primitive_index,
                                shadow.skip_self);
                            const auto light_object =
                                emitter.cycles_object_index;
                            const auto light_primitive =
                                emitter.cycles_primitive_index;
                            const auto cast_shadow =
                                (emitter.cycles_shader_flags &
                                 cycles_shader_identity::cast_shadow) != 0u;
                            const auto shadow_differential =
                                cycles_ray_differential::for_surface_shadow(
                                    sample.ray_dD,
                                    surface.differential_radius,
                                    evaluation.average_roughness_squared);
                            const auto shadow_result = config.trace_shadow(
                                shadow_ray,
                                shadow_differential.position,
                                shadow_differential.direction,
                                source_object,
                                source_primitive,
                                light_object,
                                light_primitive,
                                invocation.parameters
                                    .transparent_max_bounces,
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
                                 .transmittance = transmittance});
                            $if(any(transmittance > 0.0f)) {
                                const auto unclamped_contribution =
                                    sample.throughput *
                                    surviving_unshadowed * transmittance;
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
                                        evaluation.f,
                                        evaluation.diffuse_f,
                                        evaluation.glossy_f,
                                        sample.path_diffuse_weight,
                                        sample.path_glossy_weight,
                                        sample.path_depth));
                            };
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

}// namespace

std::unique_ptr<
    DirectLightingComponent>
make_emissive_mesh_lighting_component(
    std::shared_ptr<const DirectLightTraceRecorder> trace) {
    return std::make_unique<EmissiveMeshLightingComponent>(
        std::move(trace));
}

}// namespace psycles::luisa_backend::detail

#include "path_kernel_builder.h"
#include "path_kernel_direct_light_trace.h"
#include "path_kernel_emissive_triangle.h"

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

    void prepare(
        DirectLightingContext &context,
        DirectLightTransportState &transport)
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
            bounce.random().selected_light;
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
                        bounce.random()
                            .light_sample
                            .xy(),
                        selected_light
                            .selection_pdf);
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
                        transport.accept(
                            evaluation,
                            weighted_bsdf,
                            light_shader_factor,
                            light.light.direction,
                            light.light.position,
                            false,
                            emitter.cycles_object_index,
                            emitter.cycles_primitive_index,
                            emitter.cycles_shader_flags);
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

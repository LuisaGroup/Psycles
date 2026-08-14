#include "path_kernel_builder.h"
#include "path_kernel_direct_light_trace.h"
#include "path_kernel_environment_light.h"

#include <psycles/luisa/cycles_light.h>
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

    void prepare(
        DirectLightingContext &context,
        DirectLightTransportState &transport)
        const noexcept override {
        auto &bounce = context.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        auto &surface = context.surface;
        auto &selected_light = bounce.random().selected_light;
        auto &light_sample = bounce.random().light_sample;
        auto &surface_tag = surface.surface_tag;
        auto &point = surface.point;
        auto &path_surface_query = surface.path_surface_query;
        auto &path_depth = sample.path_depth;
        auto &diffuse_depth = sample.diffuse_depth;
        auto &glossy_depth = sample.glossy_depth;
        auto &transparent_depth = sample.transparent_depth;
        auto &transmission_depth = sample.transmission_depth;
        const auto &nee_light_weight = config.light_transport.nee_light_weight;
        auto evaluate_light_surface = [&](UInt tag,
                                          const SurfacePoint &surface_point,
                                          Float3 outgoing,
                                          const SurfaceQuery &query,
                                          UInt shader_flags) noexcept {
            return invocation.evaluate_light_surface(
                tag, surface_point, outgoing, query, shader_flags);
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
                    transport.accept(
                        evaluation,
                        weighted_bsdf,
                        light_shader_factor,
                        light.direction,
                        -light.direction,
                        true,
                        surface_ray::invalid_primitive,
                        surface_ray::invalid_primitive,
                        config.scene
                            ->cycles_background_shader_flags,
                        Bool{constant_emission});
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

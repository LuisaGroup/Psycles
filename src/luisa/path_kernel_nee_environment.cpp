#include "path_kernel_builder.h"
#include "path_kernel_environment_light.h"

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

  public:
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
        auto evaluate_surface = [&](UInt tag,
                                    const SurfacePoint &surface_point,
                                    Float3 outgoing,
                                    const SurfaceQuery &query) noexcept {
            return invocation.evaluate_surface(
                tag, surface_point, outgoing, query);
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
                        sample,
                        surface.hit_position,
                        light_sample.xy(),
                        selected_light
                            .selection_pdf);
            $if(light.valid) {
                const auto shadow =
                    make_surface_shadow_origin(
                        light.direction);
                Var<luisa::compute::Ray> environment_shadow_ray =
                    make_ray(shadow.position,
                             light.direction,
                             0.0f,
                             ray_maximum);
                Float3 shadow_transmittance =
                    trace_shadow(environment_shadow_ray,
                                 select(surface_ray::invalid_primitive,
                                        hit->inst,
                                        shadow.skip_self),
                                 select(surface_ray::invalid_primitive,
                                        hit->prim,
                                        shadow.skip_self),
                                 surface_ray::invalid_primitive,
                                 surface_ray::invalid_primitive,
                                 kernel_parameters.transparent_max_bounces,
                                 pack_shader_evaluation_state(
                                     cycles_path_state::shadow_shader_state(
                                         path_depth,
                                         diffuse_depth,
                                         glossy_depth,
                                         transparent_depth,
                                         transmission_depth)));
                $if(any(shadow_transmittance > 0.0f)) {
                    auto evaluation = evaluate_surface(
                        surface_tag,
                        point,
                        light.direction,
                        path_surface_query);
                    Float mis_weight =
                        nee_light_weight(
                            light.pdf,
                            evaluation.pdf);
                    Float3 unshadowed_contribution =
                        evaluation.f *
                        light.radiance *
                        (mis_weight /
                         light.pdf);
                    Float roulette_weight = sample_light_roulette(
                        unshadowed_contribution, light_terminate_sample);
                    Float3 contribution = clamp_contribution(
                        throughput * unshadowed_contribution *
                            shadow_transmittance * roulette_weight,
                        path_depth);
                    sample.accumulate_radiance(
                        contribution);
                    accumulate_light_pass(split_nee_light(contribution,
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
};

} // namespace

std::unique_ptr<DirectLightingComponent> make_environment_lighting_component() {
    return std::make_unique<EnvironmentLightingComponent>();
}

} // namespace psycles::luisa_backend::detail

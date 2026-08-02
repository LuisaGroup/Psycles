#include "path_kernel_builder.h"
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

  public:
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
                        sample,
                        selected_light
                            .index,
                        surface
                            .hit_position,
                        bounce
                            .light_sample
                            .xy());
            $if(light.valid) {
                const auto shadow =
                    surface
                        .make_shadow_origin(
                            light.light
                                .direction);
                const auto shadow_offset =
                    light.light.position -
                    shadow.position;
                const auto shadow_distance =
                    sqrt(max(
                        length_squared(
                            shadow_offset),
                        1.0e-20f));
                const auto shadow_direction =
                    shadow_offset /
                    shadow_distance;
                Var<luisa::compute::Ray>
                    shadow_ray =
                        make_ray(
                            shadow.position,
                            shadow_direction,
                            0.0f,
                            shadow_distance);
                const auto transmittance =
                    config.trace_shadow(
                        shadow_ray,
                        select(
                            surface_ray::
                                invalid_primitive,
                            hit->inst,
                            shadow.skip_self),
                        select(
                            surface_ray::
                                invalid_primitive,
                            hit->prim,
                            shadow.skip_self),
                        light.geometry
                            .emitter
                            .instance_index,
                        light.geometry
                            .emitter
                            .primitive_index,
                        invocation.parameters
                            .transparent_max_bounces,
                        pack_shader_evaluation_state(
                            cycles_path_state::
                                shadow_shader_state(
                                    sample
                                        .path_depth,
                                    sample
                                        .diffuse_depth,
                                    sample
                                        .glossy_depth,
                                    sample
                                        .transparent_depth,
                                    sample
                                        .transmission_depth)));
                $if(any(
                    transmittance >
                    0.0f)) {
                    const auto evaluation =
                        invocation
                            .evaluate_surface(
                                surface
                                    .surface_tag,
                                surface.point,
                                light.light
                                    .direction,
                                surface
                                    .path_surface_query);
                    const auto mis_weight =
                        config
                            .light_transport
                            .nee_light_weight(
                                light.pdf,
                                evaluation
                                    .pdf);
                    const auto unshadowed =
                        evaluation.f *
                        light.radiance *
                        (mis_weight /
                         light.pdf);
                    const auto roulette_weight =
                        invocation
                            .sample_light_roulette(
                                unshadowed,
                                bounce
                                    .light_terminate_sample);
                    const auto contribution =
                        invocation
                            .clamp_contribution(
                                sample
                                        .throughput *
                                    unshadowed *
                                    transmittance *
                                    roulette_weight,
                                sample
                                    .path_depth);
                    sample
                        .accumulate_radiance(
                            contribution);
                    sample
                        .accumulate_light_pass(
                            config
                                .light_transport
                                .split_nee_light(
                                    contribution,
                                    evaluation.f,
                                    evaluation
                                        .diffuse_f,
                                    evaluation
                                        .glossy_f,
                                    sample
                                        .path_diffuse_weight,
                                    sample
                                        .path_glossy_weight,
                                    sample
                                        .path_depth));
                };
            };
        };
    }
};

}// namespace

std::unique_ptr<
    DirectLightingComponent>
make_emissive_mesh_lighting_component() {
    return std::make_unique<
        EmissiveMeshLightingComponent>();
}

}// namespace psycles::luisa_backend::detail

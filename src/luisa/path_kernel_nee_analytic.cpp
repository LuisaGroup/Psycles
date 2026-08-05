#include "path_kernel_builder.h"

#include "path_kernel_area_light.h"
#include "path_kernel_direct_light_trace.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/spherical_geometry.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class AnalyticLightingComponent final : public DirectLightingComponent {

  private:
    AreaLightSampling _area_sampling;
    std::shared_ptr<const DirectLightTraceRecorder>
        _trace;

  public:
    explicit AnalyticLightingComponent(
        std::shared_ptr<const DirectLightTraceRecorder> trace)
        : _trace{std::move(trace)} {}

    void emit(DirectLightingContext &context) const noexcept override {
        auto &bounce = context.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        const auto &scene = config.scene;
        auto &surface = context.surface;
        auto &selected_light = bounce.selected_light;
        auto &light_sample = bounce.light_sample;
        auto &light_terminate_sample = bounce.light_terminate_sample;
        auto &hit = bounce.hit;
        auto &hit_position = surface.hit_position;
        auto &surface_tag = surface.surface_tag;
        auto &point = surface.point;
        auto &path_surface_query = surface.path_surface_query;
        auto &cycles_surface_runtime_flags =
            context.shading.cycles_surface_runtime_flags;
        auto &throughput = sample.throughput;
        auto &path_depth = sample.path_depth;
        auto &diffuse_depth = sample.diffuse_depth;
        auto &glossy_depth = sample.glossy_depth;
        auto &transparent_depth = sample.transparent_depth;
        auto &transmission_depth = sample.transmission_depth;
        auto &path_diffuse_weight = sample.path_diffuse_weight;
        auto &path_glossy_weight = sample.path_glossy_weight;
        const auto &kernel_parameters = invocation.parameters;
        const auto &safe_normalize = config.light_transport.safe_normalize;
        const auto &trace_shadow = config.trace_shadow;
        const auto &nee_light_weight = config.light_transport.nee_light_weight;
        const auto &split_nee_light = config.light_transport.split_nee_light;
        auto analytic_light_shader = [&](Var<LightGpu> light,
                                         UInt light_index,
                                         Float3 light_position,
                                         Float3 light_normal,
                                         Float2 light_uv,
                                         Float3 incoming,
                                         Float light_distance) noexcept {
            return sample.analytic_light_shader(light,
                                                light_index,
                                                light_position,
                                                light_normal,
                                                light_uv,
                                                incoming,
                                                light_distance);
        };
        auto evaluate_light_surface = [&](UInt tag,
                                          const SurfacePoint &surface_point,
                                          Float3 outgoing,
                                          const SurfaceQuery &query,
                                          UInt shader_flags) noexcept {
            return invocation.evaluate_light_surface(
                tag, surface_point, outgoing, query, shader_flags);
        };
        auto make_surface_shadow_origin = [&](Float3 direction) noexcept {
            return surface.make_shadow_origin(direction);
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
                sampling::LightDistributionEmitterKind::analytic_light)) {
            UInt light_index = selected_light.index;
            Var<LightGpu> light = scene->light_buffer->read(light_index);
            Float3 wi = make_float3(0.0f);
            Float3 light_radiance = make_float3(0.0f);
            Float3 light_position = light.position;
            Float3 light_normal = make_float3(0.0f, 0.0f, 1.0f);
            Float2 light_uv = make_float2(0.5f);
            Float light_distance = ray_maximum;
            Float light_pdf = 0.0f;
            Float light_eval_factor = 0.0f;
            Bool light_valid = false;

            $if(light.type == static_cast<std::uint32_t>(LightType::area)) {
                const auto finite_sample =
                    _area_sampling.from_position(
                        area_light_sample_input(
                            light,
                            hit_position,
                            light_sample.xy()));
                wi = finite_sample.direction;
                light_position =
                    finite_sample.position;
                light_normal =
                    finite_sample.normal;
                light_uv = finite_sample.uv;
                light_distance =
                    finite_sample.distance;
                light_pdf =
                    finite_sample
                        .conditional_pdf;
                light_eval_factor =
                    finite_sample
                        .evaluation_factor;
                light_radiance =
                    light.color *
                    (light.power *
                     light_eval_factor);
                light_valid =
                    finite_sample.valid;
            }
            $elif(light.type ==
                  static_cast<std::uint32_t>(LightType::distant)) {
                Float half_angle = 0.5f * max(light.angle, 0.0f);
                Bool finite_sun = half_angle > 0.0f;
                Float cap_height =
                    spherical_geometry::unit_cap_height(half_angle);
                Float cosine_max = 1.0f - cap_height;
                Float3 sun_axis = light.axis_z;
                Float3 basis_reference = select(make_float3(0.0f, 0.0f, 1.0f),
                                                make_float3(0.0f, 1.0f, 0.0f),
                                                abs(sun_axis.z) > 0.999f);
                Float3 sun_tangent =
                    safe_normalize(cross(basis_reference, sun_axis),
                                   make_float3(1.0f, 0.0f, 0.0f));
                Float3 sun_bitangent = cross(sun_axis, sun_tangent);
                Float cosine_theta = 1.0f - light_sample.x * cap_height;
                Float sine_theta =
                    sqrt(max(1.0f - cosine_theta * cosine_theta, 0.0f));
                Float phi = 2.0f * pi * light_sample.y;
                Float3 cone_direction =
                    sun_tangent * (cos(phi) * sine_theta) +
                    sun_bitangent * (sin(phi) * sine_theta) +
                    sun_axis * cosine_theta;
                wi = select(sun_axis, cone_direction, finite_sun);
                Float solid_angle = spherical_geometry::two_pi * cap_height;
                light_pdf =
                    select(1.0f, 1.0f / max(solid_angle, 1.0e-20f), finite_sun);
                Bool normalize_power =
                    (light.flags & light_flag_normalize) != 0u;
                Float disk_area = pi * sin(half_angle) * sin(half_angle);
                Float eval_factor = select(1.0f,
                                           1.0f / max(disk_area, 1.0e-20f),
                                           normalize_power & finite_sun);
                light_eval_factor = eval_factor;
                light_radiance = light.color * (light.power * eval_factor);
                light_position = wi;
                light_normal = -wi;
                light_valid = true;
            }
            $else {
                const auto sphere = (light.flags & light_flag_sphere) != 0u;
                const auto inside_sphere =
                    sphere & (light.radius > 0.0f) &
                    (length_squared(hit_position - light.position) <=
                     light.radius * light.radius);
                // The unified pre-NEE closure capability summary also
                // supplies Cycles' transmission bit for the special case of
                // a shading point inside spherical emitter geometry.
                const auto has_transmission =
                    (cycles_surface_runtime_flags &
                     cycles_closure::runtime_bsdf_has_transmission) != 0u;
                const auto normalize_power =
                    (light.flags & light_flag_normalize) != 0u;
                analytic_light_sampling::FiniteLightSample finite_sample{
                    .valid = false,
                    .direction = make_float3(0.0f),
                    .position = light.position,
                    .normal = make_float3(0.0f),
                    .uv = make_float2(0.0f),
                    .distance = 0.0f,
                    .conditional_pdf = 0.0f,
                    .evaluation_factor = 0.0f};
                const auto spot =
                    light.type == static_cast<std::uint32_t>(LightType::spot);
                $if(spot) {
                    finite_sample = analytic_light_sampling::sample_spot_light(
                        hit_position,
                        point.shading_normal,
                        has_transmission,
                        light.position,
                        light.radius,
                        sphere,
                        light.axis_x,
                        light.axis_y,
                        light.axis_z,
                        light.axis_scale,
                        light.spot_angle,
                        light.spot_smooth,
                        light_sample.xy(),
                        normalize_power);
                }
                $else {
                    finite_sample = analytic_light_sampling::sample_point_light(
                        hit_position,
                        point.shading_normal,
                        has_transmission,
                        light.position,
                        light.radius,
                        sphere,
                        light.axis_x,
                        light.axis_y,
                        light.axis_z,
                        light.axis_scale,
                        light_sample.xy(),
                        normalize_power);
                };
                wi = finite_sample.direction;
                light_position = finite_sample.position;
                light_normal = finite_sample.normal;
                light_uv = finite_sample.uv;
                light_distance = finite_sample.distance;
                light_pdf = finite_sample.conditional_pdf;
                light_eval_factor = finite_sample.evaluation_factor;
                light_radiance =
                    light.color * (light.power * light_eval_factor);
                light_valid = finite_sample.valid;
            };

            light_pdf *= selected_light.selection_pdf;
            $if(light_valid & (light_pdf > 0.0f)) {
                _trace->record_sample(
                    bounce,
                    {.type = light.cycles_type,
                     .emitter_id = selected_light.emitter_id,
                     .primitive = light_index,
                     .object = light.cycles_object_index,
                     .light_group = light.cycles_light_group,
                     .shader = light.cycles_shader_id,
                     .pdf = light_pdf,
                     .selection_pdf = selected_light.selection_pdf,
                     .evaluation_factor = light_eval_factor,
                     .direction = wi,
                     .position = light_position,
                     .geometric_normal = light_normal,
                     .distance = light_distance});
                const auto constant_emission =
                    (light.flags &
                     light_flag_constant_emission) != 0u;
                Float3 light_shader_factor = make_float3(1.0f);
                $if(constant_emission) {
                    light_radiance *= sample
                                          .analytic_light_constant_shader(
                                              light);
                };
                const auto evaluation = evaluate_light_surface(
                    surface_tag,
                    point,
                    wi,
                    path_surface_query,
                    light.cycles_shader_flags);
                const auto bsdf_nonzero =
                    any(evaluation.f != 0.0f);
                $if((!constant_emission) & bsdf_nonzero) {
                    light_shader_factor = analytic_light_shader(
                        light,
                        light_index,
                        light_position,
                        light_normal,
                        light_uv,
                        -wi,
                        light_distance);
                };
                const auto cycles_mis_weight =
                    nee_light_weight(light_pdf, evaluation.pdf);
                _trace->record_evaluation(
                    bounce,
                    {.distance = light_distance,
                     .bsdf_pdf = evaluation.pdf,
                     .mis_weight = cycles_mis_weight,
                     .bsdf = evaluation.f,
                     .diffuse = evaluation.diffuse_f,
                     .glossy = evaluation.glossy_f});
                const auto forward_intersectable =
                    (light.flags & light_flag_forward_intersectable) != 0u;
                const auto mis_weight =
                    select(1.0f, cycles_mis_weight, forward_intersectable);
                const auto weighted_bsdf =
                    evaluation.f * light_radiance *
                    (mis_weight / max(light_pdf, 1.0e-20f));
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
                        const auto shadow = make_surface_shadow_origin(wi);
                        const auto finite_offset =
                            light_position - shadow.position;
                        const auto finite_distance = sqrt(max(
                            length_squared(finite_offset), 1.0e-20f));
                        const auto finite_direction =
                            finite_offset / finite_distance;
                        const auto distant =
                            light.type == static_cast<std::uint32_t>(
                                              LightType::distant);
                        const auto shadow_direction =
                            select(finite_direction, wi, distant);
                        const auto shadow_maximum =
                            select(finite_distance, ray_maximum, distant);
                        Var<luisa::compute::Ray> shadow_ray = make_ray(
                            shadow.position,
                            shadow_direction,
                            0.0f,
                            shadow_maximum);
                        const auto shadow_transmittance = trace_shadow(
                            shadow_ray,
                            select(surface_ray::invalid_primitive,
                                   surface.cycles_object_index,
                                   shadow.skip_self),
                            select(surface_ray::invalid_primitive,
                                   surface.cycles_primitive_index,
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
                _trace->record_failed_sample(bounce);
            };
        };
    }
};

} // namespace

std::unique_ptr<DirectLightingComponent>
make_analytic_lighting_component(
    std::shared_ptr<const DirectLightTraceRecorder> trace) {
    return std::make_unique<AnalyticLightingComponent>(
        std::move(trace));
}

} // namespace psycles::luisa_backend::detail

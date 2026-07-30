#include "path_kernel_builder.h"

#include <psycles/luisa/analytic_light_sampling.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/spherical_geometry.h>
#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class AnalyticLightingComponent final : public DirectLightingComponent {

  public:
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
        auto &radiance = sample.radiance;
        auto &path_depth = sample.path_depth;
        auto &diffuse_depth = sample.diffuse_depth;
        auto &glossy_depth = sample.glossy_depth;
        auto &transparent_depth = sample.transparent_depth;
        auto &transmission_depth = sample.transmission_depth;
        auto &path_diffuse_weight = sample.path_diffuse_weight;
        auto &path_glossy_weight = sample.path_glossy_weight;
        auto &path_trace_active = sample.path_trace_active;
        const auto &path_step = bounce.path_step;
        const auto &kernel_parameters = invocation.parameters;
        const auto path_trace_enabled = config.path_trace_enabled;
        const auto &safe_normalize = config.light_transport.safe_normalize;
        const auto &trace_shadow = config.trace_shadow;
        const auto &nee_light_weight = config.light_transport.nee_light_weight;
        const auto &split_nee_light = config.light_transport.split_nee_light;
        auto trace_surface_closure = [&](UInt tag,
                                         const SurfacePoint &surface_point,
                                         UInt requested_index) noexcept {
            return invocation.trace_surface_closure(
                tag, surface_point, requested_index);
        };
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
        auto evaluate_surface = [&](UInt tag,
                                    const SurfacePoint &surface_point,
                                    Float3 outgoing,
                                    const SurfaceQuery &query) noexcept {
            return invocation.evaluate_surface(
                tag, surface_point, outgoing, query);
        };
        auto trace_write_event = [&](UInt event,
                                     path_trace_schema::EventSlot slot,
                                     Float3 value) noexcept {
            sample.trace_write_event(event, slot, value);
        };
        auto trace_uint32 = [&](UInt value) noexcept {
            return sample.trace_uint32(value);
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
                Float sample_x = light_sample.x;
                Float sample_y = light_sample.y;
                Bool ellipse = (light.flags & light_flag_ellipse) != 0u;
                Bool solid_angle_rectangle =
                    !ellipse & ((light.flags & light_flag_full_spread) != 0u);
                Float rectangle_pdf = 0.0f;
                Float2 disk = cycles_sample_mapping::sample_uniform_disk(
                    light_sample.xy());
                Float u = select(sample_x - 0.5f, 0.5f * disk.x, ellipse);
                Float v = select(sample_y - 0.5f, 0.5f * disk.y, ellipse);
                light_position = light.position +
                                 light.axis_x * (u * light.size_u) +
                                 light.axis_y * (v * light.size_v);
                $if(solid_angle_rectangle) {
                    const auto rectangle_sample =
                        analytic_light_sampling::sample_rectangle_solid_angle(
                            hit_position,
                            light.position,
                            light.axis_x,
                            light.size_u,
                            light.axis_y,
                            light.size_v,
                            light_sample.xy());
                    light_position = rectangle_sample.position;
                    rectangle_pdf = rectangle_sample.pdf;
                };
                Float3 inplane = light_position - light.position;
                u = dot(inplane, light.axis_x) / max(light.size_u, 1.0e-20f);
                v = dot(inplane, light.axis_y) / max(light.size_v, 1.0e-20f);
                Float3 offset = light_position - hit_position;
                Float distance2 = length_squared(offset);
                light_distance = sqrt(max(distance2, 1.0e-20f));
                wi = offset / light_distance;
                Float cosine = max(dot(-light.axis_z, -wi), 0.0f);
                Float area = light.size_u * light.size_v;
                area *= select(1.0f, 0.25f * pi, ellipse);
                area = max(area, 1.0e-12f);
                const auto area_pdf = distance2 / max(cosine * area, 1.0e-20f);
                light_pdf =
                    select(area_pdf, rectangle_pdf, solid_angle_rectangle);
                Bool normalize_power =
                    (light.flags & light_flag_normalize) != 0u;
                Float inverse_area = select(1.0f, 1.0f / area, normalize_power);
                Float spread_attenuation =
                    analytic_light_sampling::area_spread_attenuation(
                        wi, -light.axis_z, light.spread);
                light_radiance =
                    light.color * (light.power * inverse_area * (1.0f / pi) *
                                   spread_attenuation);
                light_eval_factor =
                    inverse_area * (1.0f / pi) * spread_attenuation;
                light_normal = -light.axis_z;
                light_uv = make_float2(v + 0.5f, -u - v);
                light_valid = (distance2 > 1.0e-12f) & (cosine > 0.0f) &
                              (spread_attenuation > 0.0f);
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
                // Only a point inside spherical emitter geometry
                // needs the surface's transmission capability to
                // choose Cycles' uniform-sphere versus
                // cosine-hemisphere measure. Avoid a second graph
                // evaluation for every ordinary light sample.
                if (!path_trace_enabled) {
                    $if(inside_sphere) {
                        cycles_surface_runtime_flags =
                            trace_surface_closure(surface_tag, point, 0u)
                                .runtime_flags;
                    };
                }
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

            light_radiance *= analytic_light_shader(light,
                                                    light_index,
                                                    light_position,
                                                    light_normal,
                                                    light_uv,
                                                    -wi,
                                                    light_distance);

            light_pdf *= selected_light.selection_pdf;
            if (path_trace_enabled) {
                $if(path_trace_active & light_valid & (light_pdf > 0.0f)) {
                    auto trace_evaluation = evaluate_surface(
                        surface_tag, point, wi, path_surface_query);
                    const auto use_mis =
                        (light.flags & light_flag_use_mis) != 0u;
                    const auto trace_bsdf_pdf =
                        select(0.0f, trace_evaluation.pdf, use_mis);
                    const auto trace_mis_weight =
                        nee_light_weight(light_pdf, trace_bsdf_pdf);
                    trace_write_event(
                        path_step,
                        path_trace_schema::EventSlot::light_meta,
                        make_float3(cast<float>(light.cycles_type),
                                    cast<float>(selected_light.emitter_id),
                                    cast<float>(light_index)));
                    trace_write_event(
                        path_step,
                        path_trace_schema::EventSlot::light_id,
                        make_float3(cast<float>(light.cycles_object_index),
                                    cast<float>(light.cycles_light_group),
                                    0.0f));
                    trace_write_event(
                        path_step,
                        path_trace_schema::EventSlot::light_shader,
                        trace_uint32(light.cycles_shader_id));
                    trace_write_event(path_step,
                                      path_trace_schema::EventSlot::light_pdf,
                                      make_float3(light_pdf,
                                                  selected_light.selection_pdf,
                                                  light_eval_factor));
                    trace_write_event(
                        path_step, path_trace_schema::EventSlot::light_d, wi);
                    trace_write_event(path_step,
                                      path_trace_schema::EventSlot::light_p,
                                      light_position);
                    trace_write_event(path_step,
                                      path_trace_schema::EventSlot::light_ng,
                                      light_normal);
                    trace_write_event(path_step,
                                      path_trace_schema::EventSlot::light_eval,
                                      make_float3(light_distance,
                                                  trace_bsdf_pdf,
                                                  trace_mis_weight));
                };
            }
            $if(light_valid & (light_pdf > 0.0f)) {
                const auto shadow = make_surface_shadow_origin(wi);
                const auto finite_offset = light_position - shadow.position;
                const auto finite_distance =
                    sqrt(max(length_squared(finite_offset), 1.0e-20f));
                const auto finite_direction = finite_offset / finite_distance;
                const auto distant = light.type == static_cast<std::uint32_t>(
                                                       LightType::distant);
                const auto shadow_direction =
                    select(finite_direction, wi, distant);
                const auto shadow_maximum =
                    select(finite_distance, ray_maximum, distant);
                Var<luisa::compute::Ray> shadow_ray = make_ray(
                    shadow.position, shadow_direction, 0.0f, shadow_maximum);
                Float3 shadow_transmittance =
                    trace_shadow(shadow_ray,
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
                        surface_tag, point, wi, path_surface_query);
                    const auto forward_intersectable =
                        (light.flags & light_flag_forward_intersectable) != 0u;
                    Float mis_weight =
                        select(1.0f,
                               nee_light_weight(light_pdf, evaluation.pdf),
                               forward_intersectable);
                    Float3 unshadowed_contribution =
                        evaluation.f * light_radiance *
                        (mis_weight / max(light_pdf, 1.0e-20f));
                    Float roulette_weight = sample_light_roulette(
                        unshadowed_contribution, light_terminate_sample);
                    Float3 contribution = clamp_contribution(
                        throughput * unshadowed_contribution *
                            shadow_transmittance * roulette_weight,
                        path_depth);
                    radiance += contribution;
                    accumulate_light_pass(split_nee_light(contribution,
                                                          evaluation.f,
                                                          evaluation.diffuse_f,
                                                          path_diffuse_weight,
                                                          path_glossy_weight,
                                                          path_depth));
                };
            };
        };
    }
};

} // namespace

std::unique_ptr<DirectLightingComponent> make_analytic_lighting_component() {
    return std::make_unique<AnalyticLightingComponent>();
}

} // namespace psycles::luisa_backend::detail

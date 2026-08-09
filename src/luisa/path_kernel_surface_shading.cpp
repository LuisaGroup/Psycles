#include "path_kernel_builder.h"
#include "path_kernel_emissive_triangle.h"
#include "subsurface_exit_closure_component.h"

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class SurfaceShadingStageImpl final : public SurfaceShadingStage {

  private:
    std::shared_ptr<
        const EmissiveTriangleComponent>
        _emissive_triangle{
            make_emissive_triangle_component()};

  public:
    SurfaceShadingState
    emit(SurfaceGeometryContext &surface) const noexcept override {
        auto &bounce = surface.bounce;
        auto &sample = bounce.sample;
        auto &invocation = sample.invocation;
        const auto &config = invocation.config;
        const auto &scene = config.scene;
        const auto &kernel_parameters = invocation.parameters;
        auto &surface_tag = surface.surface_tag;
        auto &point = surface.point;
        auto &hit = bounce.hit;
        auto &ray = sample.ray;
        auto &hit_position = surface.hit_position;
        auto &wp0 = surface.wp0;
        auto &wp1 = surface.wp1;
        auto &wp2 = surface.wp2;
        auto &cycles_surface_shader = surface.cycles_surface_shader;
        auto &cycles_object_index = surface.cycles_object_index;
        auto &cycles_primitive_index =
            surface.cycles_primitive_index;
        auto &path_step = bounce.path_step;
        auto &terminate_sample = bounce.terminate_sample;
        auto &light_sample = bounce.light_sample;
        auto &light_terminate_sample = bounce.light_terminate_sample;
        auto &bsdf_sample = bounce.bsdf_sample;
        auto &throughput = sample.throughput;
        auto &sample_emission = sample.sample_emission;
        auto &path_depth = sample.path_depth;
        auto &previous_delta = sample.previous_delta;
        auto &previous_bsdf_pdf = sample.previous_bsdf_pdf;
        auto &previous_mis_origin_normal =
            sample.previous_mis_origin_normal;
        auto &previous_light_tree_dt =
            sample.previous_light_tree_dt;
        auto &terminate_on_next_surface = sample.terminate_on_next_surface;
        auto &ray_events = sample.ray_events;
        auto &transparent_depth = sample.transparent_depth;
        auto &continuation_probability = sample.continuation_probability;
        auto &continuation_decided_in_volume =
            sample.continuation_decided_in_volume;
        auto &cycles_rng_offset = sample.cycles_rng_offset;
        auto &cycles_path_visibility = sample.cycles_path_visibility;
        auto &path_flags = sample.path_flags;
        auto &diffuse_depth = sample.diffuse_depth;
        auto &glossy_depth = sample.glossy_depth;
        auto &transmission_depth = sample.transmission_depth;
        auto &sample_albedo = sample.sample_albedo;
        auto &sample_glossy_color = sample.sample_glossy_color;
        auto &sample_transmission_color = sample.sample_transmission_color;
        auto &sample_normal = sample.sample_normal;
        auto &primary_recorded = sample.primary_recorded;
        const auto &forward_light_weight =
            config.light_transport.forward_light_weight;
        const auto next_event_estimation = config.next_event_estimation;
        const auto path_trace_enabled = config.path_trace_enabled;
        auto prepare_surface = [&](UInt tag,
                                   const SurfacePoint &surface_point,
                                   Float3 outgoing,
                                   const SurfaceQuery &query,
                                   Bool include_runtime_flags,
                                   Bool include_aov) noexcept {
            return invocation.prepare_surface(
                tag,
                surface_point,
                outgoing,
                query,
                include_runtime_flags,
                include_aov);
        };
        auto clamp_contribution = [&](Float3 contribution,
                                      UInt depth) noexcept {
            return invocation
                .clamp_emission_contribution(
                    contribution, depth);
        };
        auto trace_surface_closure = [&](UInt tag,
                                         const SurfacePoint &surface_point,
                                         UInt requested_index,
                                         Bool reflective_caustics,
                                         Bool refractive_caustics) noexcept {
            return invocation.trace_surface_closure(
                tag,
                surface_point,
                requested_index,
                reflective_caustics,
                refractive_caustics);
        };
        auto trace_write_event = [&](UInt event,
                                     path_trace_schema::EventSlot slot,
                                     Float3 value) noexcept {
            sample.trace_write_event(event, slot, value);
        };
        auto trace_write_closure = [&](UInt event,
                                       std::uint32_t closure,
                                       std::uint32_t field,
                                       Float3 value) noexcept {
            sample.trace_write_closure(event, closure, field, value);
        };
        auto trace_uint32 = [&](UInt value) noexcept {
            return sample.trace_uint32(value);
        };
        const auto preparation = prepare_surface(
            surface_tag,
            point,
            point.incoming,
            surface.path_surface_query,
            Bool{next_event_estimation || path_trace_enabled},
            (!bounce.subsurface_exit) & (path_depth == 0u));
        Float3 emitted = preparation.emission;
        emitted = select(
            emitted, make_float3(0.0f), bounce.subsurface_exit);
        Float emission_weight = 1.0f;
        if (next_event_estimation && scene->emissive_triangle_count > 0u) {
            // The committed primitive already carries the effective authored
            // sampling policy. NONE covers non-emissive materials, curves,
            // and instances which are not part of the sampled-light
            // population, so forward-hit MIS is O(1) exactly as in Cycles'
            // legacy light distribution.
            $if((!surface.is_curve) &
                (surface.emission_sampling !=
                 static_cast<std::uint32_t>(
                     contract::EmissionSampling::none))) {
                Bool competing = (path_depth > 0u) & (!previous_delta);
                const auto oriented_geometric_normal =
                    select(
                        point.geometric_normal,
                        -point.geometric_normal,
                        point.back_facing);
                Float selection_pdf =
                    0.5f * length(cross(wp1 - wp0, wp2 - wp0)) *
                    scene->triangle_area_pdf;
                if (config.use_light_tree) {
                    const auto emitter_id =
                        config.light_tree.triangle_emitter(
                            cycles_object_index,
                            cycles_primitive_index);
                    selection_pdf = config.light_tree.forward_pdf(
                        emitter_id,
                        ray->origin(),
                        previous_mis_origin_normal,
                        previous_light_tree_dt,
                        cycles_path_visibility,
                        path_flags);
                }
                const auto light_pdf =
                    _emissive_triangle
                        ->from_intersection(
                            selection_pdf,
                            surface.emission_sampling,
                            ray->origin(),
                            hit_position,
                            wp0,
                            wp1,
                            wp2,
                            oriented_geometric_normal)
                        .value;
                emission_weight = forward_light_weight(
                    previous_bsdf_pdf, light_pdf, competing, light_pdf > 0.0f);
            };
        }
        Float3 emission_contribution = clamp_contribution(
            throughput * emitted * emission_weight, path_depth);
        sample.accumulate_radiance(
            emission_contribution);
        auto directly_visible_emission =
            (path_flags &
             cycles_path_state::flag_any_pass) ==
            0u;
        sample_emission += select(make_float3(0.0f),
                                  emission_contribution,
                                  directly_visible_emission);
        sample.accumulate_scattered_light(
            select(emission_contribution,
                   make_float3(0.0f),
                   directly_visible_emission));

        // PATH_RAY_TERMINATE_ON_NEXT_SURFACE still records
        // surface emission, then stops before data passes, direct
        // lighting, or another closure sample.
        $if(terminate_on_next_surface & !bounce.subsurface_exit) {
            $break;
        };

        // Cycles performs continuation roulette only after the
        // next ray is known to hit a surface. Background and
        // surface-emission contributions above are therefore
        // retained even when the path does not continue
        // scattering.
        $if(!bounce.subsurface_exit) {
            $if(continuation_decided_in_volume) {
                $if(continuation_probability <= 0.0f) {
                    $break;
                };
                $if(continuation_probability != 1.0f) {
                    throughput /= continuation_probability;
                };
            }
            $else {
                continuation_probability =
                    cycles_path_state::continuation_probability(
                        path_flags,
                        path_depth,
                        transparent_depth,
                        kernel_parameters.min_bounces,
                        kernel_parameters.transparent_min_bounces,
                        throughput);
                $if(continuation_probability <= 0.0f) {
                    $break;
                };
                $if(continuation_probability != 1.0f) {
                    $if(terminate_sample >= continuation_probability) {
                        $break;
                    };
                    throughput /= continuation_probability;
                };
            };

            // Cycles writes camera data passes only at the entry surface.
            // A BSSRDF exit skips them along with duplicate roulette.
            $if(path_depth == 0u) {
                const auto &aov = preparation.aov;
                sample_albedo += throughput * aov.albedo;
                sample_glossy_color += throughput * aov.glossy_albedo;
                sample_transmission_color +=
                    throughput * aov.transmission_albedo;
                auto surface_alpha = clamp(
                    make_float3(1.0f) - aov.transparency,
                    make_float3(0.0f),
                    make_float3(1.0f));
                auto average_alpha =
                    (surface_alpha.x + surface_alpha.y + surface_alpha.z) *
                    (1.0f / 3.0f);
                auto writes_normal =
                    (!primary_recorded) &
                    ((kernel_parameters.pass_alpha_threshold == 0.0f) |
                     (average_alpha >=
                      kernel_parameters.pass_alpha_threshold));
                sample_normal = select(sample_normal, aov.normal, writes_normal);
                primary_recorded = primary_recorded | writes_normal;
                path_flags |= select(
                    0u,
                    cycles_path_state::flag_single_pass_done,
                    writes_normal);
            };
        };

        UInt cycles_surface_runtime_flags = 0u;
        if (next_event_estimation || path_trace_enabled) {
            cycles_surface_runtime_flags = preparation.runtime_flags;
            cycles_surface_runtime_flags = select(
                cycles_surface_runtime_flags,
                SubsurfaceExitClosureComponent{}.runtime_flags(point),
                bounce.subsurface_exit);
        }
        if (path_trace_enabled) {
            auto closure_summary = trace_surface_closure(
                surface_tag,
                point,
                0u,
                surface.path_surface_query.reflective_caustics,
                surface.path_surface_query.refractive_caustics);
            $if(bounce.subsurface_exit) {
                closure_summary = SubsurfaceExitClosureComponent{}.trace(
                    point, surface.path_surface_query, 0u);
            };
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::state_depth,
                              make_float3(cast<float>(path_step),
                                          cast<float>(cycles_rng_offset),
                                          cast<float>(path_depth)));
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::state_lobes,
                              make_float3(cast<float>(transparent_depth),
                                          cast<float>(diffuse_depth),
                                          cast<float>(glossy_depth)));
            trace_write_event(
                path_step,
                path_trace_schema::EventSlot::state_visibility,
                make_float3(cast<float>(transmission_depth),
                            trace_uint32(cycles_path_visibility).xy()));
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::state_flags,
                              make_float3(trace_uint32(path_flags).xy(),
                                          continuation_probability));
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::throughput,
                              throughput);
            trace_write_event(
                path_step, path_trace_schema::EventSlot::ray_p, ray->origin());
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::ray_d,
                              ray->direction());
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::ray_range,
                              make_float3(ray->t_min(), ray->t_max(), 0.5f));
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::isect_coord,
                              make_float3(hit->committed_ray_t, hit->bary));
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::isect_id,
                              make_float3(cast<float>(cycles_object_index),
                                          cast<float>(cycles_primitive_index),
                                          1.0f));
            trace_write_event(
                path_step,
                path_trace_schema::EventSlot::surface_meta,
                make_float3(trace_uint32(cycles_surface_shader).xy(),
                            cast<float>(closure_summary.count)));
            trace_write_event(
                path_step,
                path_trace_schema::EventSlot::surface_flags,
                make_float3(trace_uint32(cycles_surface_runtime_flags).xy(),
                            0.0f));
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::surface_p,
                              point.position);
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::surface_ng,
                              point.geometric_normal);
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::surface_n,
                              point.shading_normal);
            trace_write_event(
                path_step,
                path_trace_schema::EventSlot::random_scalars,
                make_float3(
                    terminate_sample,
                    select(0.0f,
                           light_terminate_sample,
                           kernel_parameters.light_inv_rr_threshold > 0.0f),
                    0.0f));
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::random_light,
                              light_sample);
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::random_bsdf,
                              bsdf_sample);
            for (auto closure_index = 0u;
                 closure_index < path_trace_schema::max_closures;
                 ++closure_index) {
                auto closure =
                    trace_surface_closure(
                        surface_tag,
                        point,
                        closure_index,
                        surface.path_surface_query.reflective_caustics,
                        surface.path_surface_query.refractive_caustics);
                $if(bounce.subsurface_exit) {
                    closure = SubsurfaceExitClosureComponent{}.trace(
                        point,
                        surface.path_surface_query,
                        closure_index);
                };
                $if(closure.valid) {
                    trace_write_closure(path_step,
                                        closure_index,
                                        0u,
                                        make_float3(cast<float>(closure.index),
                                                    cast<float>(closure.type),
                                                    closure.sample_weight));
                    trace_write_closure(
                        path_step, closure_index, 1u, closure.weight);
                    trace_write_closure(
                        path_step, closure_index, 2u, closure.normal);
                };
            }
        }

        return {std::move(cycles_surface_runtime_flags)};
    }
};

} // namespace

std::unique_ptr<SurfaceShadingStage> make_surface_shading_stage() {
    return std::make_unique<SurfaceShadingStageImpl>();
}

} // namespace psycles::luisa_backend::detail

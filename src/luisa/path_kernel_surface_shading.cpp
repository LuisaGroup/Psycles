#include "path_kernel_builder.h"
#include "path_kernel_emissive_triangle.h"
#include "path_kernel_surface_emission.h"
#include "cycles_svm_surface_shader.h"

#include <psycles/luisa/cycles_sampler.h>
#include <psycles/luisa/cycles_svm.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class SurfaceShadingStageImpl final : public SurfaceShadingStage {

  private:
    SurfaceShadingStageOptions _options;
    std::shared_ptr<
        const EmissiveTriangleComponent>
        _emissive_triangle{
            make_emissive_triangle_component()};

  public:
    explicit SurfaceShadingStageImpl(SurfaceShadingStageOptions options) noexcept
        : _options{std::move(options)} {}

    SurfaceShadingState
    emit(SurfaceGeometryContext &surface,
         Bool *sample_terminated_out = nullptr) const noexcept override {
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
        auto &cycles_surface_shader = surface.cycles_surface_shader;
        auto &cycles_object_index = surface.cycles_object_index;
        auto &cycles_primitive_index =
            surface.cycles_primitive_index;
        auto &path_step = bounce.path_step;
        auto &throughput = sample.throughput;
        auto &path_depth = sample.path_depth;
        const auto terminate_on_next_surface =
            sample.terminate_on_next_surface_requested();
        auto &ray_events = sample.ray_events;
        auto &transparent_depth = sample.transparent_depth;
        auto &continuation_probability = sample.continuation_probability;
        auto &cycles_rng_offset = sample.cycles_rng_offset;
        auto &cycles_path_visibility = sample.cycles_path_visibility;
        auto &path_flags = sample.path_flags;
        auto &diffuse_depth = sample.diffuse_depth;
        auto &glossy_depth = sample.glossy_depth;
        auto &transmission_depth = sample.transmission_depth;
        const auto next_event_estimation = config.next_event_estimation;
        const auto path_trace_enabled = config.path_trace_enabled;
        std::shared_ptr<PopulatedSurfaceShader> populated_surface;
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
        // Emission eligibility consumes SD_EMISSION even in forward-only
        // rendering without diagnostics. Unused flag projections are dead
        // expressions and remain available for ordinary compiler pruning.
        const auto include_runtime_flags = Bool{true};
        const auto include_aov =
            (!bounce.subsurface_exit) & (path_depth == 0u);
        const auto &population_component =
            _options.population ? _options.population
                                : config.surfaces.population;
        LUISA_ASSERT(population_component != nullptr,
                     "Native surface shading requires a population component.");
        {
            const auto emission_reflective_caustics =
                Bool{config.reflective_caustics} |
                ((point.ray_visibility &
                  contract::visibility_bit(
                      contract::RayVisibility::diffuse)) == 0u);
            const SurfacePopulationQuery population_query{
                .emission_reflective_caustics =
                    emission_reflective_caustics,
                .reflective_caustics =
                    surface.path_surface_query.reflective_caustics,
                .refractive_caustics =
                    surface.path_surface_query.refractive_caustics,
                .glossy_filter_roughness =
                    surface.path_surface_query.glossy_filter_roughness,
                .include_runtime_flags = include_runtime_flags,
                .include_aov = include_aov};
            populated_surface = population_component->populate(
                {.surface_tag = surface_tag,
                 .point = point,
                 .query = population_query,
                 .cycles_surface_shader = cycles_surface_shader,
                 .cycles_object_index = cycles_object_index,
                 .cycles_primitive_index = cycles_primitive_index,
                 .primitive_type = surface.cycles_primitive_type,
                 .ray_origin = sample.ray->origin(),
                 .ray_position_differential = surface.differential_radius,
                 .ray_direction_differential = sample.ray_dD,
                 .path_flags = sample.path_flags,
                 .portal_depth = sample.portal_depth,
                 .sample_index = sample.sample_index,
                 .rng_hash = sample.rng_hash,
                 .rng_offset = sample.cycles_rng_offset,
                 .parameters = kernel_parameters,
                 .camera_projection = config.camera_projection,
                 .ambient_occlusion = nullptr,
                 .cycles_shader_flags =
                     surface.cycles_shader_flags
                         ? std::optional<Expr<std::uint32_t>>{
                               *surface.cycles_shader_flags}
                         : std::nullopt,
                 .cycles_object_flags =
                     surface.cycles_object_flags
                         ? std::optional<Expr<std::uint32_t>>{
                               *surface.cycles_object_flags}
                         : std::nullopt,
                 .may_filter_glossy = config.has_filter_glossy});
        }
        LUISA_ASSERT(populated_surface != nullptr,
                     "Native surface population returned null.");
        {
            namespace abi = compiler::cycles_svm;
            bool scene_may_have_holdout = true;
            if (scene->cycles_svm != nullptr &&
                scene->cycles_svm->objects != nullptr &&
                scene->cycles_svm->objects->image.valid) {
                const bool any_shader_holdout =
                    _options.may_have_holdout &&
                    (!scene->cycles_svm->compilation.table.valid ||
                     scene->cycles_svm->compilation.table
                         .usage_for(abi::SHADER_TYPE_SURFACE)
                         .node_types_used[abi::NODE_CLOSURE_HOLDOUT]);
                const bool any_object_holdout = std::any_of(
                    scene->cycles_svm->objects->image.object_flags.begin(),
                    scene->cycles_svm->objects->image.object_flags.end(),
                    [](std::uint32_t f) noexcept {
                        return (f & unsigned(abi::SD_OBJECT_HOLDOUT_MASK)) != 0u;
                    });
                scene_may_have_holdout = any_shader_holdout || any_object_holdout;
            }
            if (scene_may_have_holdout) {
                const auto native = populated_surface->native_surface_state();
                LUISA_ASSERT(native.has_value(), "Holdout requires retained ShaderData.");
                auto &sd = native->shader_data;
                // integrate_surface skips holdout at a BSSRDF exit. For ordinary
                // events this precedes emission and the termination/roulette edge.
                $if(!bounce.subsurface_exit &
                    (((sd.flag & unsigned(abi::SD_HOLDOUT)) != 0u) |
                     ((sd.object_flag & unsigned(abi::SD_OBJECT_HOLDOUT_MASK)) != 0u)) &
                    ((path_flags & cycles_path_state::flag_transparent_background) != 0u)) {
                    const auto weight = cycles_svm::detail::surface_shader_apply_holdout(sd);
                    const auto weighted = weight * throughput;
                    sample.accumulate_transparency((weighted.x + weighted.y + weighted.z) * (1.0f / 3.0f));
                    if (sample_terminated_out == nullptr) {
                        $if(all(weight == make_float3(1.0f))) { $break; };
                    } else {
                        $if(all(weight == make_float3(1.0f))) {
                            *sample_terminated_out = true;
                        };
                    }
                };
            }
        }
        UInt cycles_surface_runtime_flags = 0u;
        Float3 bsdf_sample = make_float3(0.0f);
        Float3 final_shading_normal = point.shading_normal;
        const auto emit_post_holdout = [&]() noexcept {
        const auto preparation = populated_surface->preparation();
        final_shading_normal = preparation.shading_normal;
        $if(bounce.subsurface_exit) {
          path_flags &= ~cycles_path_state::flag_subsurface;
        };
        if (config.surface_closure_count_histogram_enabled) {
            sample.record_surface_closure_count(
                UInt{populated_surface->closure_count()});
        }
        surface.set_evaluated_shadow_shading_normal(
            preparation.shading_normal);
        if (_options.may_have_emission || path_trace_enabled) {
            emit_surface_emission(surface, preparation, *_emissive_triangle);
        }

        // PATH_RAY_TERMINATE_ON_NEXT_SURFACE still records
        // surface emission, then stops before data passes, direct
        // lighting, or another closure sample.
        if (sample_terminated_out == nullptr) {
            $if(terminate_on_next_surface & !bounce.subsurface_exit) {
                $break;
            };
        } else {
            $if(terminate_on_next_surface & !bounce.subsurface_exit) {
                *sample_terminated_out = true;
            };
        }

        const auto emit_post_roulette = [&]() noexcept {
        // INTERSECT_CLOSEST already made Cycles' continuation decision after
        // the endpoint became known. Surface emission above remains visible;
        // surviving paths are renormalized only now, before further shading.
        $if(!bounce.subsurface_exit) {
            $if(continuation_probability != 1.0f) {
                throughput /= continuation_probability;
            };

            // Cycles writes camera data passes only at the entry surface.
            // A BSSRDF exit skips them along with duplicate roulette.
            $if(path_depth == 0u) {
                const auto &aov = preparation.aov;
                sample.accumulate_albedo_pass(
                    throughput * aov.albedo);
                sample.accumulate_light_pass(
                    LightPassBuffer::glossy_color,
                    throughput * aov.glossy_albedo);
                sample.accumulate_light_pass(
                    LightPassBuffer::transmission_color,
                    throughput * aov.transmission_albedo);
                auto surface_alpha = clamp(
                    make_float3(1.0f) - aov.transparency,
                    make_float3(0.0f),
                    make_float3(1.0f));
                auto average_alpha =
                    (surface_alpha.x + surface_alpha.y + surface_alpha.z) *
                    (1.0f / 3.0f);
                auto writes_normal =
                    (!sample.single_pass_recorded()) &
                    ((kernel_parameters.pass_alpha_threshold == 0.0f) |
                     (average_alpha >=
                      kernel_parameters.pass_alpha_threshold));
                // Keep the native data-pass predicate around the reduction
                // and film write. Passing a selected zero to the accumulator
                // still records three atomic fetch_add operations in atomic
                // film mode for rejected alpha/single-pass lanes.
                $if(writes_normal) {
                    sample.accumulate_normal_pass(aov.normal);
                    path_flags |=
                        cycles_path_state::flag_single_pass_done;
                };
            };
        };

        if (next_event_estimation || path_trace_enabled) {
            cycles_surface_runtime_flags = preparation.runtime_flags;
        }
        if (path_trace_enabled) {
            bsdf_sample = sample.surface_bsdf_sample();
        }
        if (path_trace_enabled) {
            // Cycles' diagnostic trace observes PRNG_LIGHT even when NEE
            // will be rejected. Record only the pure tuple here, never an
            // emitter-distribution lookup or a light-tree walk for tracing.
            const auto light_sample = cycles_sampler::sample_3d(
                invocation.sobol_table, kernel_parameters.sobol_sequence_size,
                sample.sample_index, sample.rng_hash,
                cycles_sampler::path_state_dimension(
                    cycles_rng_offset, tabulated_sobol::light_dimension));
            const auto light_terminate_sample = cycles_sampler::sample_1d(
                invocation.sobol_table, kernel_parameters.sobol_sequence_size,
                sample.sample_index, sample.rng_hash,
                cycles_sampler::path_state_dimension(
                    cycles_rng_offset, tabulated_sobol::light_terminate_dimension));
            const auto terminate_sample =
                sample.continuation_terminate_sample();
            const auto closure_summary = populated_surface->closure_trace(
                0u, surface.path_surface_query);
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
                              make_float3(hit->committed_ray_t,
                                          point.barycentric));
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
                make_float3(trace_uint32(
                                populated_surface->native_shader_data_flags()).xy(),
                            0.0f));
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::surface_p,
                              point.position);
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::surface_ng,
                              point.geometric_normal);
            trace_write_event(path_step,
                              path_trace_schema::EventSlot::surface_n,
                              preparation.shading_normal);
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
                const auto closure = populated_surface->closure_trace(
                    closure_index, surface.path_surface_query);
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
        };
        if (sample_terminated_out == nullptr) {
            $if(!bounce.subsurface_exit & (continuation_probability == 0.0f)) {
                $break;
            };
            emit_post_roulette();
        } else {
            $if(!bounce.subsurface_exit & (continuation_probability == 0.0f)) {
                *sample_terminated_out = true;
            };
            $if(!*sample_terminated_out) {
                emit_post_roulette();
            };
        }
        };
        if (sample_terminated_out == nullptr) {
            emit_post_holdout();
        } else {
            $if(!*sample_terminated_out) {
                emit_post_holdout();
            };
        }

        return {std::move(cycles_surface_runtime_flags),
                std::move(bsdf_sample),
                std::move(final_shading_normal),
                std::move(populated_surface)};
    }
};

} // namespace

std::unique_ptr<SurfaceShadingStage>
make_surface_shading_stage(SurfaceShadingStageOptions options) {
    return std::make_unique<SurfaceShadingStageImpl>(std::move(options));
}

} // namespace psycles::luisa_backend::detail

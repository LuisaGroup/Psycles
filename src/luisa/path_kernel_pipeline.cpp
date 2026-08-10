#include "path_kernel_builder.h"
#include "path_kernel_direct_light_trace.h"

#include <psycles/luisa/cycles_closure.h>

#include <utility>
#include <vector>

#include <luisa/dsl/coro_func.h>

namespace psycles::luisa_backend::detail {

class PathKernelPipeline::Impl {

  public:
    std::unique_ptr<PathBounceSetupStage> bounce_setup;
    std::unique_ptr<ClosestEventStage> closest_event;
    std::unique_ptr<ForwardLightStage> forward_light;
    std::unique_ptr<PathVolumeSegmentStage>
        volume_segment;
    std::unique_ptr<BackgroundEventStage>
        background{
            make_background_event_stage()};
    std::unique_ptr<SurfaceGeometryStage> surface_geometry;
    std::unique_ptr<SurfaceShadingStage> surface_shading;
    std::shared_ptr<const DirectLightTraceRecorder>
        direct_light_trace;
    std::vector<std::unique_ptr<DirectLightingComponent>> direct_lighting;
    std::unique_ptr<DirectLightTransportStage>
        direct_light_transport;
    std::unique_ptr<SurfaceScatterStage> surface_scatter;
    std::unique_ptr<SubsurfaceTransportStage> subsurface_transport;

    explicit Impl(
        const PathKernelConfig &config)
        : direct_light_trace{
              make_direct_light_trace_recorder(
                  config.path_trace_enabled)} {
        const auto stage_plan =
            make_path_kernel_scene_stage_plan(
                config.next_event_estimation,
                config.scene
                    ->environment_in_light_distribution,
                config.scene
                    ->emissive_triangle_count,
                config.scene->light_count,
                config.scene->geometries.size(),
                config.scene->curve_geometries.size(),
                config.scene
                    ->cycles_completion_source_dense_count,
                config.scene
                    ->cycles_completion_source_sparse_count);
        const auto traversal_plan =
            stage_plan.traversal;
        const auto primitive_plan =
            traversal_plan.primitives;
        bounce_setup =
            make_path_bounce_setup_stage(
                traversal_plan);
        closest_event =
            make_closest_event_stage(
                stage_plan
                    .analytic_light_endpoints,
                primitive_plan);
        if (stage_plan.analytic_light_endpoints) {
            forward_light =
                make_forward_light_stage();
        }
        if (config.volume_state) {
            volume_segment =
                make_path_volume_segment_stage(
                    config);
        }
        if (!primitive_plan.empty()) {
            surface_geometry =
                make_surface_geometry_stage(
                    primitive_plan);
            surface_shading =
                make_surface_shading_stage();
            surface_scatter =
                make_surface_scatter_stage();
            const auto &direct_lighting_plan =
                stage_plan.direct_lighting;
            direct_lighting.reserve(
                direct_lighting_plan.size());
            if (direct_lighting_plan.environment) {
                direct_lighting.emplace_back(
                    make_environment_lighting_component(
                        direct_light_trace));
            }
            if (direct_lighting_plan.emissive_mesh) {
                direct_lighting.emplace_back(
                    make_emissive_mesh_lighting_component(
                        direct_light_trace));
            }
            if (direct_lighting_plan.analytic) {
                direct_lighting.emplace_back(
                    make_analytic_lighting_component(
                        direct_light_trace));
            }
            if (direct_lighting_plan.transport_stage_count() != 0u) {
                direct_light_transport =
                    make_direct_light_transport_stage(
                        direct_light_trace);
            }
            if (config.has_subsurface) {
                subsurface_transport =
                    make_subsurface_transport_stage();
            }
        }
    }
};

PathKernelPipeline::PathKernelPipeline(
    const PathKernelConfig &config)
    : _impl{
          std::make_unique<Impl>(
              config)} {}

PathKernelPipeline::~PathKernelPipeline() noexcept = default;
PathKernelPipeline::PathKernelPipeline(PathKernelPipeline &&) noexcept =
    default;
PathKernelPipeline &
PathKernelPipeline::operator=(PathKernelPipeline &&) noexcept = default;

void PathKernelPipeline::emit(
    PathSampleContext &sample,
    bool is_coro) const noexcept {
    $for(path_step, sample.invocation.parameters.max_path_steps) {
        // This ordinary C++ branch executes while recording the Luisa AST.
        // The suspension is therefore absent from the megakernel rather than
        // guarded by a device-side predicate. At this boundary only canonical
        // per-path state is live; no hit shading or closure temporaries have
        // been populated yet.
        if (is_coro) {
            $suspend("path_bounce");
        }
        auto bounce =
            _impl->bounce_setup->emit(
                sample, path_step);

        // A Cycles lamp is a transparent closest event. Resolve every lamp
        // before the already-known mesh/background event without consuming
        // another path bounce or another set of Sobol dimensions. Keeping
        // the event distance explicit also establishes the segment boundary
        // at which volume transport is inserted.
        UInt previous_analytic_light =
            surface_ray::invalid_primitive;
        // A successful spatial BSSRDF query already selected an exact
        // same-object surface hit.  Cycles shades that stored hit directly;
        // tracing the synthetic exit ray again would make the result depend
        // on an epsilon and on backend-specific BVH traversal.
        Bool search_events = !bounce.subsurface_exit;
        Bool path_terminated = false;
        Bool volume_scattered = false;
        UInt surface_emission_sampling =
            static_cast<std::uint32_t>(
                contract::EmissionSampling::none);
        $while(search_events &
               !path_terminated) {
            auto event =
                _impl->closest_event->emit(
                    bounce,
                    previous_analytic_light);
            surface_emission_sampling =
                event.surface_emission_sampling;
            if (_impl->volume_segment) {
                const auto volume =
                    _impl->volume_segment->emit(
                        event);
                path_terminated =
                    path_terminated |
                    volume.terminated;
                volume_scattered =
                    volume_scattered |
                    volume.scattered;
                search_events =
                    search_events &
                    !volume.scattered &
                    !volume.terminated;
            }
            $if(search_events &
                !path_terminated) {
                if (_impl->forward_light) {
                    $if(event.analytic_light) {
                        path_terminated =
                            _impl->forward_light->emit(
                                event);
                        previous_analytic_light =
                            event.light_index;
                    }
                    $else {
                        search_events = false;
                        $if(event.background) {
                            _impl->background->emit(
                                event);
                            path_terminated = true;
                        };
                    };
                } else {
                    search_events = false;
                    $if(event.background) {
                        _impl->background->emit(
                            event);
                        path_terminated = true;
                    };
                }
            };
        };
        $if(path_terminated) {
            $break;
        };
        $if(volume_scattered) {
            $continue;
        };

        if (_impl->surface_geometry) {
            // Traversal and volume resolution have selected an exact surface
            // hit, but surface geometry and closure population have not begun.
            // Suspending here keeps those large, short-lived values out of the
            // coroutine frame while separating traversal from shading.
            if (is_coro) {
                $suspend("surface_shading");
            }
            auto surface =
                _impl->surface_geometry->emit(
                    bounce,
                    surface_emission_sampling);
            auto shading =
                _impl->surface_shading->emit(surface);
            if (sample.invocation.config.use_light_tree) {
                bounce.selected_light =
                    sample.invocation.config.light_tree.surface_sample(
                        bounce.light_sample.z,
                        surface.hit_position,
                        surface.point.shading_normal,
                        0.0f,
                        (shading.cycles_surface_runtime_flags &
                         cycles_closure::runtime_bsdf_has_transmission) != 0u);
            }
            DirectLightingContext lighting{
                .bounce = bounce,
                .surface = surface,
                .shading = shading};
            if (_impl->direct_light_transport) {
                const auto has_evaluable_bsdf =
                    (shading.cycles_surface_runtime_flags &
                     cycles_closure::runtime_bsdf_has_eval) != 0u;
                $if(has_evaluable_bsdf) {
                    auto transport =
                        DirectLightTransportState::empty();
                    for (const auto &component : _impl->direct_lighting) {
                        component->prepare(
                            lighting, transport);
                    }
                    _impl->direct_light_transport->emit(
                        lighting, transport);
                };
            }
            const auto scatter =
                _impl->surface_scatter->emit(lighting);
            if (_impl->subsurface_transport) {
                $if(scatter.subsurface) {
                    const auto transported =
                        _impl->subsurface_transport->emit(
                            lighting, scatter.sample);
                    $if(transported) {
                        $continue;
                    }
                    $else {
                        $break;
                    };
                };
            }
        }
    };
}

namespace {

void emit_path_program(
    const PathKernelConfig &config,
    const PathKernelPipeline &pipeline,
    bool is_coro,
    const BufferFloat4 &combined,
    const BufferFloat4 &normal,
    const BufferFloat4 &albedo,
    const BufferFloat4 &light_passes,
    const BufferUInt &sample_count,
    const BufferFloat4 &volume_guiding_raw,
    const BufferUInt &volume_guiding_denoised,
    const BufferFloat4 &path_trace,
    const UInt &sample_first,
    const UInt &samples,
    const BufferFloat4 &sobol_table,
    const BufferFloat &filter_table,
    const Var<RenderKernelParameters> &parameters) noexcept {
    auto invocation = begin_path_kernel(config,
                                        combined,
                                        normal,
                                        albedo,
                                        light_passes,
                                        sample_count,
                                        volume_guiding_raw,
                                        volume_guiding_denoised,
                                        path_trace,
                                        sample_first,
                                        samples,
                                        sobol_table,
                                        filter_table,
                                        parameters);
    $for(sample_offset, samples) {
        auto sample = begin_path_sample(invocation, sample_offset);
        pipeline.emit(sample, is_coro);
        accumulate_path_sample(sample);
    };
    invocation.write_film();
}

}// namespace

RenderKernel build_path_kernel(const PathKernelConfig &config) noexcept {
    PathKernelPipeline pipeline{
        config};
    RenderKernel kernel =
        [&config, &pipeline](BufferFloat4 combined,
                             BufferFloat4 normal,
                             BufferFloat4 albedo,
                             BufferFloat4 light_passes,
                             BufferUInt sample_count,
                             BufferFloat4 volume_guiding_raw,
                             BufferUInt volume_guiding_denoised,
                             BufferFloat4 path_trace,
                             UInt sample_first,
                             UInt samples,
                             BufferFloat4 sobol_table,
                             BufferFloat filter_table,
                             Var<RenderKernelParameters> parameters) noexcept {
            emit_path_program(config,
                              pipeline,
                              false,
                              combined,
                              normal,
                              albedo,
                              light_passes,
                              sample_count,
                              volume_guiding_raw,
                              volume_guiding_denoised,
                              path_trace,
                              sample_first,
                              samples,
                              sobol_table,
                              filter_table,
                              parameters);
        };
    return kernel;
}

RenderCoroutine build_path_coroutine(
    const PathKernelConfig &config) {
    PathKernelPipeline pipeline{
        config};
    return RenderCoroutine{
        [&config, &pipeline](BufferFloat4 combined,
                             BufferFloat4 normal,
                             BufferFloat4 albedo,
                             BufferFloat4 light_passes,
                             BufferUInt sample_count,
                             BufferFloat4 volume_guiding_raw,
                             BufferUInt volume_guiding_denoised,
                             BufferFloat4 path_trace,
                             UInt sample_first,
                             UInt samples,
                             BufferFloat4 sobol_table,
                             BufferFloat filter_table,
                             Var<RenderKernelParameters> parameters) noexcept {
            emit_path_program(config,
                              pipeline,
                              true,
                              combined,
                              normal,
                              albedo,
                              light_passes,
                              sample_count,
                              volume_guiding_raw,
                              volume_guiding_denoised,
                              path_trace,
                              sample_first,
                              samples,
                              sobol_table,
                              filter_table,
                              parameters);
        }};
}

} // namespace psycles::luisa_backend::detail

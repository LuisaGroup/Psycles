#include "path_kernel_builder.h"
#include "path_kernel_direct_light_task.h"
#include "path_kernel_direct_light_trace.h"
#include "path_kernel_surface_queue.h"
#include "path_kernel_transitions.h"

#include <psycles/luisa/cycles_closure.h>

#include <optional>
#include <utility>
#include <vector>

#include <luisa/dsl/coro_func.h>

namespace psycles::luisa_backend::detail {

class PathKernelPipeline::Impl {

public:
  std::unique_ptr<PathBounceSetupStage> bounce_setup;
  std::unique_ptr<PathBounceRandomStage> bounce_random{
      make_path_bounce_random_stage()};
  std::unique_ptr<ClosestEventStage> closest_event;
  std::unique_ptr<ForwardLightStage> forward_light;
  std::unique_ptr<PathVolumeSegmentStage> volume_segment;
  std::unique_ptr<BackgroundEventStage> background{
      make_background_event_stage()};
  std::unique_ptr<SurfaceGeometryStage> surface_geometry;
  std::unique_ptr<SurfaceQueueKeyStage> surface_queue_key;
  std::unique_ptr<SurfaceShadingStage> surface_shading;
  std::shared_ptr<const DirectLightTraceRecorder> direct_light_trace;
  std::vector<std::unique_ptr<DirectLightingComponent>> direct_lighting;
  std::unique_ptr<DirectLightTransportStage> direct_light_transport;
  std::unique_ptr<SurfaceScatterStage> surface_scatter;
  std::unique_ptr<SubsurfaceTransportStage> subsurface_transport;

  explicit Impl(const PathKernelConfig &config)
      : direct_light_trace{
            make_direct_light_trace_recorder(config.path_trace_enabled)} {
    const auto stage_plan = make_path_kernel_scene_stage_plan(
        config.next_event_estimation,
        config.scene->environment_in_light_distribution,
        config.scene->emissive_triangle_count, config.scene->light_count,
        config.scene->geometries.size(), config.scene->curve_geometries.size());
    const auto traversal_plan = stage_plan.traversal;
    const auto primitive_plan = traversal_plan.primitives;
    bounce_setup =
        make_path_bounce_setup_stage(traversal_plan, config.has_subsurface);
    closest_event = make_closest_event_stage(
        stage_plan.analytic_light_endpoints, primitive_plan);
    if (stage_plan.analytic_light_endpoints) {
      forward_light = make_forward_light_stage();
    }
    if (config.volume_state) {
      volume_segment = make_path_volume_segment_stage(config);
    }
    if (!primitive_plan.empty()) {
      surface_geometry = make_surface_geometry_stage(primitive_plan);
      if (config.staged_surface_sorting && !config.scene->surfaces.empty()) {
        surface_queue_key = make_surface_queue_key_stage(primitive_plan);
      }
      surface_shading = make_surface_shading_stage();
      surface_scatter = make_surface_scatter_stage();
      const auto &direct_lighting_plan = stage_plan.direct_lighting;
      direct_lighting.reserve(direct_lighting_plan.size());
      if (direct_lighting_plan.environment) {
        direct_lighting.emplace_back(
            make_environment_lighting_component(direct_light_trace));
      }
      if (direct_lighting_plan.emissive_mesh) {
        direct_lighting.emplace_back(
            make_emissive_mesh_lighting_component(direct_light_trace));
      }
      if (direct_lighting_plan.analytic) {
        direct_lighting.emplace_back(
            make_analytic_lighting_component(direct_light_trace));
      }
      if (direct_lighting_plan.transport_stage_count() != 0u) {
        direct_light_transport = make_direct_light_transport_stage(
            direct_light_trace, make_direct_light_task_evaluator(config),
            config.direct_light_task_sink, config.path_trace_enabled);
      }
      if (config.has_subsurface) {
        subsurface_transport = make_subsurface_transport_stage();
      }
    }
  }
};

PathKernelPipeline::PathKernelPipeline(const PathKernelConfig &config)
    : _impl{std::make_unique<Impl>(config)} {}

PathKernelPipeline::~PathKernelPipeline() noexcept = default;
PathKernelPipeline::PathKernelPipeline(PathKernelPipeline &&) noexcept =
    default;
PathKernelPipeline &
PathKernelPipeline::operator=(PathKernelPipeline &&) noexcept = default;

void PathKernelPipeline::emit(
    PathSampleContext &sample,
    PathCoroutineCutPolicy cut_policy) const noexcept {
  const auto random_plan = make_path_bounce_random_plan(
      cut_policy, static_cast<bool>(_impl->volume_segment));
  $for(path_step, sample.invocation.parameters.max_path_steps) {
    // This ordinary C++ branch executes while recording the Luisa AST.
    // The suspension is therefore absent from the megakernel rather than
    // guarded by a device-side predicate. At this boundary only canonical
    // per-path state is live; no hit shading or closure temporaries have
    // been populated yet.
    if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
      // This names the semantic work queue; traversal remains the same
      // PathBounceSetupStage used by both megakernel variants.
      $suspend(path_transition::intersect_closest);
    }
    auto bounce = _impl->bounce_setup->emit(sample, path_step);
    std::optional<PathBounceRandomState> dominating_random_state;
    if (random_plan.before_event_resolution) {
      dominating_random_state.emplace(_impl->bounce_random->emit(sample));
      bounce.random_state = &*dominating_random_state;
    }

    // A Cycles lamp is a transparent closest event. Resolve every lamp
    // before the already-known mesh/background event without consuming
    // another path bounce or another set of Sobol dimensions. Keeping
    // the event distance explicit also establishes the segment boundary
    // at which volume transport is inserted.
    UInt previous_analytic_light = surface_ray::invalid_primitive;
    // A successful spatial BSSRDF query already selected an exact
    // same-object surface hit.  Cycles shades that stored hit directly;
    // tracing the synthetic exit ray again would make the result depend
    // on an epsilon and on backend-specific BVH traversal.
    Bool search_events = !bounce.subsurface_exit;
    Bool path_terminated = false;
    Bool volume_scattered = false;
    $while(search_events & !path_terminated) {
      auto event = _impl->closest_event->emit(bounce, previous_analytic_light);
      if (_impl->volume_segment) {
        VolumeSegmentEvent volume{.scattered = false, .terminated = false};
        // This is the same routing predicate as Cycles'
        // integrator_intersect_next_kernel. Under the VolumeStack
        // sentinel invariant, PathVolumeSegmentStage is an
        // observational identity for an empty stack: bounce setup already
        // reset the continuation state, background cleanup maps empty to
        // empty, and every remaining transport write is selected by
        // inside_volume. Keeping the complete stage in this branch therefore
        // preserves semantics and removes it from the empty-stack intersect
        // path.
        $if(!sample.volume.stack->empty()) {
          if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
            $suspend(path_transition::shade_volume);
          }
          std::optional<PathBounceRandomState> volume_random_state;
          if (random_plan.shade_volume) {
            // This local binding cannot escape the shade_volume
            // continuation. Later event stages have no random
            // consumer; the surface continuation binds its own
            // equal sampler expression at its first use.
            volume_random_state.emplace(_impl->bounce_random->emit(sample));
            bounce.random_state = &*volume_random_state;
          }
          $outline_with_name("path_volume_segment") {
            volume = _impl->volume_segment->emit(event);
          };
        };
        if (random_plan.shade_volume) {
          // Make the host/JIT lifetime invariant executable: a new
          // consumer inserted between volume and surface must bind
          // an explicit state instead of retaining a dangling C++
          // expression handle.
          bounce.random_state = nullptr;
        }
        path_terminated = path_terminated | volume.terminated;
        volume_scattered = volume_scattered | volume.scattered;
        search_events = search_events & !volume.scattered & !volume.terminated;
      }
      $if(search_events & !path_terminated) {
        if (_impl->forward_light) {
          $if(event.analytic_light) {
            if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
              $suspend(path_transition::shade_light_forward);
            }
            path_terminated = _impl->forward_light->emit(event);
            previous_analytic_light = event.light_index;
          }
          $else {
            search_events = false;
            $if(event.background) {
              if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
                $suspend(path_transition::shade_background);
              }
              _impl->background->emit(event);
              path_terminated = true;
            };
          };
        } else {
          search_events = false;
          $if(event.background) {
            if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
              $suspend(path_transition::shade_background);
            }
            _impl->background->emit(event);
            path_terminated = true;
          };
        }
      };
    };
    $if(path_terminated) { $break; };
    $if(volume_scattered) { $continue; };

    if (_impl->surface_geometry) {
      // Traversal and volume resolution have selected an exact surface
      // hit, but surface geometry and closure population have not begun.
      // Suspending here keeps those large, short-lived values out of the
      // coroutine frame while separating traversal from shading.
      if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
        // Resolve only the topology-deduplicated material tag here,
        // after volume transport selected the surface path. Keeping
        // this recomputation at the cut avoids carrying it through
        // shade_volume while still giving the scheduler a coherence
        // key before closure population begins.
        if (_impl->surface_queue_key) {
          const auto surface_queue_key = _impl->surface_queue_key->emit(bounce);
          $suspend(path_transition::shade_surface,
                   coro_frame_export(path_transition::scheduler_hint,
                                     surface_queue_key));
        } else {
          $suspend(path_transition::shade_surface);
        }
      }
      auto surface = _impl->surface_geometry->emit(bounce);
      std::optional<PathBounceRandomState> surface_random_state;
      if (random_plan.shade_surface) {
        surface_random_state.emplace(_impl->bounce_random->emit(sample));
        bounce.random_state = &*surface_random_state;
      }
      auto shading = _impl->surface_shading->emit(surface);
      if (sample.invocation.config.use_light_tree) {
        bounce.random().selected_light =
            sample.invocation.config.light_tree.surface_sample(
                bounce.random().light_sample.z, surface.hit_position,
                surface.point.shading_normal, 0.0f,
                (shading.cycles_surface_runtime_flags &
                 cycles_closure::runtime_bsdf_has_transmission) != 0u);
      }
      DirectLightingContext lighting{
          .bounce = bounce, .surface = surface, .shading = shading};
      std::optional<DirectLightTransportPreparation> direct_light_preparation;
      bool defer_direct_light = false;
      if (_impl->direct_light_transport) {
        direct_light_preparation.emplace();
        defer_direct_light =
            _impl->direct_light_transport->defer_until_after_surface_scatter(
                cut_policy);
        const auto has_evaluable_bsdf =
            (shading.cycles_surface_runtime_flags &
             cycles_closure::runtime_bsdf_has_eval) != 0u;
        $if(has_evaluable_bsdf) {
          auto transport = DirectLightTransportState::empty();
          for (const auto &component : _impl->direct_lighting) {
            component->prepare(lighting, transport);
          }
          *direct_light_preparation =
              _impl->direct_light_transport->prepare(lighting, transport);
        };
        if (!defer_direct_light) {
          _impl->direct_light_transport->emit(
              bounce, std::move(*direct_light_preparation), cut_policy);
        }
      }
      const auto scatter = _impl->surface_scatter->emit(lighting);

      // BSSRDF entry sampling is the final consumer of populated
      // surface data. Reduce it before the shadow stages as well; only
      // the compact transport state can then remain live at a shadow
      // suspension.
      std::optional<SubsurfaceTransportPreparation> subsurface_preparation;
      if (_impl->subsurface_transport) {
        subsurface_preparation.emplace();
        $if(scatter.valid & scatter.subsurface) {
          *subsurface_preparation =
              _impl->subsurface_transport->prepare(lighting, scatter.sample);
        };
      }

      // This is the sequential A/bypass -> C graph: surface and closure
      // temporaries have no uses below this point. Only the prepared
      // shadow task, canonical continuation state, and (when reachable)
      // compact BSSRDF transport state cross these suspensions.
      if (_impl->direct_light_transport && defer_direct_light) {
        _impl->direct_light_transport->emit(
            bounce, std::move(*direct_light_preparation), cut_policy);
      }

      // A failed continuation sample does not discard the direct-light
      // contribution from the current vertex. Consume this predicate
      // only after the deferred shadow state machine has completed.
      $if(!scatter.valid) { $break; };

      if (_impl->subsurface_transport) {
        $if(scatter.subsurface) {
          $if(!subsurface_preparation->valid) { $break; };
          if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
            // Cycles' INTERSECT_SUBSURFACE consumes canonical
            // path ray/isect state plus the dedicated 40-byte
            // BSSRDF payload. Surface geometry, closures, NEE
            // reservoirs, and shader temporaries die before this
            // scheduling boundary.
            $suspend(path_transition::intersect_subsurface);
          }
          const auto transported = _impl->subsurface_transport->emit(
              sample, subsurface_preparation->state);
          $if(transported) { $continue; }
          $else { $break; };
        };
      }
    }
  };
}

namespace {

[[nodiscard]] PathKernelInvocation make_path_invocation(
    const PathKernelConfig &config, PathFilmAccumulation film_accumulation,
    UInt pixel, const BufferFloat4 &combined, const BufferFloat4 &normal,
    const BufferFloat4 &albedo, const BufferFloat4 &light_passes,
    const BufferUInt &sample_count, const BufferFloat4 &volume_guiding_raw,
    const BufferUInt &volume_guiding_denoised, const BufferFloat4 &path_trace,
    const UInt &sample_first, const BufferFloat4 &sobol_table,
    const BufferFloat &filter_table,
    const Var<RenderKernelParameters> &parameters) noexcept {
  return begin_path_kernel(
      config, film_accumulation, std::move(pixel), combined, normal, albedo,
      light_passes, sample_count, volume_guiding_raw, volume_guiding_denoised,
      path_trace, sample_first, sobol_table, filter_table, parameters);
}

// The one authoritative host path-tracing wrapper. Sample-loop organization
// and dispatch.z mapping live in its callers; sampler initialization, path
// stages, and film contribution stay identical for every execution policy.
void emit_path_sample(const PathKernelPipeline &pipeline,
                      PathKernelInvocation &invocation,
                      const UInt &sub_spp_index,
                      PathCoroutineCutPolicy cut_policy) noexcept {
  auto sample = begin_path_sample(invocation, sub_spp_index);
  pipeline.emit(sample, cut_policy);
  accumulate_path_sample(sample);
}

void emit_serial_path_program(
    const PathKernelConfig &config, const PathKernelPipeline &pipeline,
    const BufferFloat4 &combined, const BufferFloat4 &normal,
    const BufferFloat4 &albedo, const BufferFloat4 &light_passes,
    const BufferUInt &sample_count, const BufferFloat4 &volume_guiding_raw,
    const BufferUInt &volume_guiding_denoised, const BufferFloat4 &path_trace,
    const UInt &sample_first, const UInt &samples,
    const BufferFloat4 &sobol_table, const BufferFloat &filter_table,
    const Var<RenderKernelParameters> &parameters) noexcept {
  auto invocation = make_path_invocation(
      config, PathFilmAccumulation::serial, dispatch_x(), combined, normal,
      albedo, light_passes, sample_count, volume_guiding_raw,
      volume_guiding_denoised, path_trace, sample_first, sobol_table,
      filter_table, parameters);
  $for(sub_spp_index, samples) {
    emit_path_sample(pipeline, invocation, sub_spp_index,
                     PathCoroutineCutPolicy::none);
  };
  invocation.write_film();
}

void emit_per_sample_path_program(
    const PathKernelConfig &config, const PathKernelPipeline &pipeline,
    PathCoroutineCutPolicy cut_policy, const BufferFloat4 &combined,
    const BufferFloat4 &normal, const BufferFloat4 &albedo,
    const BufferFloat4 &light_passes, const BufferUInt &sample_count,
    const BufferFloat4 &volume_guiding_raw,
    const BufferUInt &volume_guiding_denoised, const BufferFloat4 &path_trace,
    const UInt &sample_first, const BufferFloat4 &sobol_table,
    const BufferFloat &filter_table,
    const Var<RenderKernelParameters> &parameters) noexcept {
  const auto pixel =
      luisa::compute::dispatch_y() * luisa::compute::dispatch_size().x +
      dispatch_x();
  auto invocation = make_path_invocation(
      config, PathFilmAccumulation::atomic, pixel, combined, normal, albedo,
      light_passes, sample_count, volume_guiding_raw, volume_guiding_denoised,
      path_trace, sample_first, sobol_table, filter_table, parameters);
  emit_path_sample(pipeline, invocation, luisa::compute::dispatch_z(),
                   cut_policy);
  invocation.write_film();
}

} // namespace

RenderSerialKernel
build_path_serial_kernel(const PathKernelConfig &config) noexcept {
  PathKernelPipeline pipeline{config};
  RenderSerialKernel kernel =
      [&config, &pipeline](
          BufferFloat4 combined, BufferFloat4 normal, BufferFloat4 albedo,
          BufferFloat4 light_passes, BufferUInt sample_count,
          BufferFloat4 volume_guiding_raw, BufferUInt volume_guiding_denoised,
          BufferFloat4 path_trace, UInt sample_first, UInt samples,
          BufferFloat4 sobol_table, BufferFloat filter_table,
          Var<RenderKernelParameters> parameters) noexcept {
        emit_serial_path_program(config, pipeline, combined, normal, albedo,
                                 light_passes, sample_count, volume_guiding_raw,
                                 volume_guiding_denoised, path_trace,
                                 sample_first, samples, sobol_table,
                                 filter_table, parameters);
      };
  return kernel;
}

RenderSampleKernel
build_path_sample_kernel(const PathKernelConfig &config) noexcept {
  PathKernelPipeline pipeline{config};
  RenderSampleKernel kernel =
      [&config, &pipeline](
          BufferFloat4 combined, BufferFloat4 normal, BufferFloat4 albedo,
          BufferFloat4 light_passes, BufferUInt sample_count,
          BufferFloat4 volume_guiding_raw, BufferUInt volume_guiding_denoised,
          BufferFloat4 path_trace, UInt sample_first, UInt,
          BufferFloat4 sobol_table, BufferFloat filter_table,
          Var<RenderKernelParameters> parameters) noexcept {
        // Keep one sample plane per block. This avoids placing multiple
        // writers for the same pixel in a block and is a clean topology
        // baseline for the coroutine schedulers, whose worker layout is
        // controlled by their own host configurations.
        set_block_size(8u, 8u, 1u);
        emit_per_sample_path_program(
            config, pipeline, PathCoroutineCutPolicy::none, combined, normal,
            albedo, light_passes, sample_count, volume_guiding_raw,
            volume_guiding_denoised, path_trace, sample_first, sobol_table,
            filter_table, parameters);
      };
  return kernel;
}

RenderCoroutine build_path_coroutine(const PathKernelConfig &config,
                                     PathCoroutineCutPolicy cut_policy) {
  LUISA_ASSERT(cut_policy != PathCoroutineCutPolicy::none,
               "A path coroutine requires at least one suspension policy.");
  PathKernelPipeline pipeline{config};
  return RenderCoroutine{
      [&config, &pipeline, cut_policy](
          BufferFloat4 combined, BufferFloat4 normal, BufferFloat4 albedo,
          BufferFloat4 light_passes, BufferUInt sample_count,
          BufferFloat4 volume_guiding_raw, BufferUInt volume_guiding_denoised,
          BufferFloat4 path_trace, UInt sample_first, UInt samples,
          BufferFloat4 sobol_table, BufferFloat filter_table,
          Var<RenderKernelParameters> parameters) noexcept {
        static_cast<void>(samples);
        emit_per_sample_path_program(
            config, pipeline, cut_policy, combined, normal, albedo,
            light_passes, sample_count, volume_guiding_raw,
            volume_guiding_denoised, path_trace, sample_first, sobol_table,
            filter_table, parameters);
      }};
}

} // namespace psycles::luisa_backend::detail

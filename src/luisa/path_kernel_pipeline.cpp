#include "path_kernel_builder.h"
#include "path_kernel_caustics_manifold.h"
#include "path_kernel_coro_transitions.h"
#include "path_kernel_scene_traversal.h"
#include "path_kernel_cycles_svm_surface_scatter.h"
#include "path_kernel_direct_light_task.h"
#include "path_kernel_direct_light_trace.h"
#include "path_kernel_surface_buckets.h"
#include "path_kernel_surface_queue.h"
#include "path_kernel_transitions.h"
#include "path_tracer_cycles_svm_surface.h"
#include "virtual_texture_cache.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_svm.h>
#include <psycles/luisa/cycles_volume_boundary.h>
#include <psycles/sampling/light_distribution.h>

#include <charconv>
#include <cstdlib>
#include <optional>
#include <string_view>
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
  std::unique_ptr<BackgroundEventStage> background;
  std::unique_ptr<SurfaceGeometryStage> surface_geometry;
  std::unique_ptr<SurfaceQueueKeyStage> surface_queue_key;
  std::unique_ptr<SurfaceShadingStage> surface_shading;
  std::vector<std::unique_ptr<SurfaceShadingStage>> bucket_surface_shading;
  PathSurfaceBucketPlan surface_buckets;
  ScenePrimitiveStagePlan primitive_plan{};
  std::shared_ptr<const DirectLightTraceRecorder> direct_light_trace;
  std::vector<std::unique_ptr<DirectLightingComponent>> direct_lighting;
  std::unique_ptr<DirectLightTransportStage> direct_light_transport;
  std::unique_ptr<SurfaceScatterStage> surface_scatter;
  std::unique_ptr<SubsurfaceTransportStage> subsurface_transport;
  std::optional<CausticsTaskEvaluator> caustics_evaluator;

  explicit Impl(const PathKernelConfig &config)
      : direct_light_trace{
            make_direct_light_trace_recorder(config.path_trace_enabled)} {
    const auto stage_plan = make_path_kernel_scene_stage_plan(
        config.next_event_estimation,
        config.scene->environment_in_light_distribution,
        config.scene->emissive_triangle_count, config.scene->light_count,
        config.scene->geometries.size(), config.scene->curve_geometries.size());
    const auto traversal_plan = stage_plan.traversal;
    primitive_plan = traversal_plan.primitives;
    surface_buckets = make_path_surface_bucket_plan(config);
    bounce_setup =
        make_path_bounce_setup_stage(
            traversal_plan,
            config.has_subsurface,
            config.ambient_occlusion_bounce_approximation);
    closest_event = make_closest_event_stage(
        stage_plan.analytic_light_endpoints,
        primitive_plan,
        config.ambient_occlusion_bounce_approximation);
    background = make_background_event_stage(
        config.ambient_occlusion_bounce_approximation);
    if (stage_plan.analytic_light_endpoints) {
      forward_light = make_forward_light_stage();
    }
    if (config.volume_state) {
      volume_segment = make_path_volume_segment_stage(config);
    }
    if (!primitive_plan.empty()) {
      surface_geometry = make_surface_geometry_stage(primitive_plan);
      if (config.staged_surface_sorting &&
          surface_queue_key_range(*config.scene) != 0u) {
        surface_queue_key = make_surface_queue_key_stage(primitive_plan);
      }
      const auto ao_traversal =
          config.scene->has_ambient_occlusion
              ? std::shared_ptr<const SceneTraversalComponent>{
                    make_scene_traversal_component(traversal_plan)}
              : nullptr;
      surface_shading = make_surface_shading_stage(
          SurfaceShadingStageOptions{.traversal = ao_traversal});
      bucket_surface_shading.reserve(surface_buckets.buckets.size());
      for (const auto &spec : surface_buckets.buckets) {
        if (surface_buckets.native_specialized && config.surfaces.population) {
          auto pop = make_cycles_svm_surface_population_component(
              config.scene,
              CyclesSvmSurfaceBucketSpecialization{
                  .active_shader_mask = spec.active_shader_mask,
                  .entry_usage = spec.entry_usage,
                  .max_surface_closures = spec.max_surface_closures});
          bucket_surface_shading.push_back(make_surface_shading_stage(
              SurfaceShadingStageOptions{
                  .population = std::move(pop),
                  .traversal = ao_traversal,
                  .may_have_emission = spec.may_have_emission,
                  .may_have_holdout = spec.may_have_holdout}));
        } else {
          bucket_surface_shading.push_back(make_surface_shading_stage(
              SurfaceShadingStageOptions{.traversal = ao_traversal}));
        }
      }
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
      if (config.caustics_active() && !config.caustics_task_sink) {
        caustics_evaluator.emplace(make_caustics_task_evaluator(config));
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
    PathCoroutineCutPolicy cut_policy,
    const SampleCompleteHandler &on_sample_complete) const noexcept {
  LUISA_ASSERT(
      !on_sample_complete || cut_policy == PathCoroutineCutPolicy::none,
      "Persistent sample completion handler requires megakernel mode.");
  auto *const prof = cut_policy == PathCoroutineCutPolicy::none
                         ? sample.invocation.stage_profile.get()
                         : nullptr;
  const auto &pipeline_config = sample.invocation.config;
  const auto pipeline_vt =
      (cut_policy == PathCoroutineCutPolicy::cycles_wavefront &&
       pipeline_config.enable_virtual_texture_cache &&
       pipeline_config.scene &&
       pipeline_config.scene->virtual_texture_cache &&
       pipeline_config.scene->virtual_texture_cache->has_virtual_images())
          ? pipeline_config.scene->virtual_texture_cache
          : std::shared_ptr<VirtualTextureCache>{};
  if (pipeline_vt) {
    sample.vt_wait_resident =
        [vt = pipeline_vt](Expr<luisa::uint> page_id) noexcept {
          UInt missed_page = page_id;
          $loop {
            $suspend(path_transition::texture_cache,
                     luisa::compute::coro_stage(
                         luisa::string{vt_extension_schema})
                         .read(luisa::string{vt_extension_binding},
                               missed_page)
                         .blocking_async()
                         .async_domain(luisa::string{vt_extension_schema}));
            $if(vt->is_resident(missed_page)) { $break; };
          };
        };
  }
  // Cycles schedules until a native transition terminates the path. Regular,
  // transparent/portal and volume-boundary counters are independent; BSSRDF
  // entry and exit are separate scheduled events. An additional aggregate
  // iteration budget can discard a valid continuation or its terminal film.
  // This index is diagnostic only; release recording has no consumers.
  UInt path_step = ~0u;
  $loop {
    std::optional<Bool> sample_done;
    std::optional<Bool> volume_only_skipped;
    if (on_sample_complete) {
      sample_done.emplace(false);
      volume_only_skipped.emplace(false);
    }
    if (prof != nullptr) {
      prof->enter(MegakernelStageProfileState::stage_setup_and_film);
    }
    if (sample.invocation.config.path_trace_enabled) { path_step += 1u; }
    PathBounceContext bounce{.sample = sample,
                             .path_step = path_step,
                             .random_state = nullptr,
                             .hit = Var<luisa::compute::CommittedHit>{},
                             .subsurface_exit = false};
    // Lamps do not consume a surface path step or another Sobol dimension,
    // but each surviving lamp returns to Cycles' INTERSECT_CLOSEST. Do not
    // cache the old mesh hit across lamp/volume state transitions.
    UInt previous_analytic_light = surface_ray::invalid_primitive;
    Bool search_events = true;
    Bool path_terminated = false;
    Bool volume_scattered = false;
    $while(search_events & !path_terminated) {
      suspend_before_closest_intersection(
          cut_policy, static_cast<bool>(_impl->subsurface_transport),
          sample.pending_subsurface_exit);
      if (prof != nullptr) {
        prof->enter(MegakernelStageProfileState::stage_closest_trace);
      }
      const auto next = _impl->bounce_setup->emit(sample, path_step);
      bounce.hit = next.hit;
      bounce.subsurface_exit = next.subsurface_exit;
      // A successful BSSRDF query already committed the exact same-object
      // hit. The cut policy and bounce setup both bypass traversal; shade
      // that stored hit directly without entering lamp/volume resolution.
      $if(bounce.subsurface_exit) { $break; };
      if (prof != nullptr) {
        prof->bump(MegakernelStageProfileState::count_closest_traces);
      }
      auto event = _impl->closest_event->emit(bounce, previous_analytic_light);
      if (prof != nullptr) {
        prof->enter(MegakernelStageProfileState::stage_volume_and_background);
      }
      path_terminated = path_terminated | event.terminated;
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
        $if(!path_terminated & !sample.volume.stack->empty()) {
          if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
            $suspend(path_transition::shade_volume);
          }
          // Volume proposals must precede distance tracking (equiangular
          // sampling consumes them). Keep this state local to the nonempty
          // volume segment in both megakernel and coroutine recording.
          auto volume_random_state = _impl->bounce_random->emit(sample);
          bounce.random_state = &volume_random_state;
          $outline_with_name("path_volume_segment") {
            volume = _impl->volume_segment->emit(event);
          };
          bounce.random_state = nullptr;
        };
        path_terminated = path_terminated | volume.terminated;
        volume_scattered = volume_scattered | volume.scattered;
        search_events = search_events & !volume.scattered & !volume.terminated;
      }
      $if(search_events & !path_terminated) {
        auto emit_background_stage = [&]() noexcept {
          search_events = false;
          $if(event.background) {
            if (sample.vt_wait_resident) {
              $loop {
                $suspend(path_transition::shade_background);
                UInt vt_miss = vt_invalid_page;
                sample.vt_miss_page = &vt_miss;
                _impl->background->emit(event);
                sample.vt_miss_page = nullptr;
                $if(vt_miss != vt_invalid_page) {
                  sample.vt_wait_resident(vt_miss);
                  $continue;
                };
                $break;
              };
            } else {
              if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
                $suspend(path_transition::shade_background);
              }
              _impl->background->emit(event);
            }
            path_terminated = true;
          };
        };
        if (_impl->forward_light) {
          $if(event.analytic_light) {
            if (sample.vt_wait_resident) {
              $loop {
                $suspend(path_transition::shade_light_forward);
                UInt vt_miss = vt_invalid_page;
                sample.vt_miss_page = &vt_miss;
                const auto lamp_term = _impl->forward_light->emit(event);
                sample.vt_miss_page = nullptr;
                $if(vt_miss != vt_invalid_page) {
                  sample.vt_wait_resident(vt_miss);
                  $continue;
                };
                path_terminated = lamp_term;
                $break;
              };
            } else {
              if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
                $suspend(path_transition::shade_light_forward);
              }
              path_terminated = _impl->forward_light->emit(event);
            }
            previous_analytic_light = event.light_index;
          }
          $else {
            emit_background_stage();
          };
        } else {
          emit_background_stage();
        }
      };
    };
    if (!on_sample_complete) {
      $if(path_terminated) { $break; };
      $if(volume_scattered) { $continue; };
    } else {
      *sample_done = path_terminated;
    }

    if (_impl->surface_geometry) {
      const auto &config = sample.invocation.config;
      const auto &scene = config.scene;
      const auto &bucket_plan = _impl->surface_buckets;
      const bool has_portal =
          scene->cycles_svm != nullptr &&
          (scene->cycles_svm->kernel_features &
           cycles_svm::kernel_feature_node_portal) != 0u;
      const auto env_enabled = [](const char *name, bool def_val) noexcept {
        const char *raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0') {
          return def_val;
        }
        return std::string_view{raw} != "0" && std::string_view{raw} != "off";
      };
      const bool fast_post_closure =
          env_enabled("PSYCLES_POST_CLOSURE_ORDER", true) &&
          !config.path_trace_enabled &&
          !has_portal && config.surfaces.population != nullptr;
      const auto pipeline_backend_name =
          static_cast<bool>(scene->device)
              ? std::string_view{scene->device.backend_name()}
              : std::string_view{};
      const bool cpu_post_closure_bssrdf =
          pipeline_backend_name == "fallback" ||
          pipeline_backend_name == "simd";
      const bool force_post_closure_on_bssrdf =
          std::getenv("PSYCLES_POST_CLOSURE_ORDER") != nullptr
              ? env_enabled("PSYCLES_POST_CLOSURE_ORDER", false)
              : cpu_post_closure_bssrdf;
      const bool use_multi_bucket =
          cut_policy == PathCoroutineCutPolicy::cycles_wavefront &&
          config.staged_surface_sort_annotation && _impl->surface_queue_key &&
          bucket_plan.multi_bucket();
      const bool use_post_substage =
          cut_policy == PathCoroutineCutPolicy::cycles_wavefront &&
          fast_post_closure && bucket_plan.post_closure_substage;

      const auto emit_volume_only_skip =
          [&](SurfaceGeometryContext &surface,
              Bool *outer_break = nullptr,
              Bool *outer_continue = nullptr) noexcept {
            if (config.volume_state) {
              namespace abi = compiler::cycles_svm;
              const UInt shader_flags =
                  surface.cycles_shader_flags
                      ? *surface.cycles_shader_flags
                      : (*scene->cycles_svm->kernel_shader_buffer)
                            ->read(surface.cycles_surface_shader &
                                   cycles_svm::shader_mask)
                            .flags.cast<unsigned>();
              $if((shader_flags &
                   static_cast<unsigned>(abi::SD_HAS_ONLY_VOLUME)) != 0u) {
                const auto next = cycles_volume_boundary::advance(
                    sample.throughput, sample.volume_bounds_bounce,
                    sample.cycles_rng_offset, sample.ray->t_min(),
                    max(sample.ray->t_min(), bounce.hit->committed_ray_t),
                    sample.continuation_probability, sample.path_flags);
                sample.throughput = next.throughput;
                sample.volume_bounds_bounce = next.bounds_bounce;
                sample.cycles_rng_offset = next.rng_offset;
                if (!on_sample_complete) {
                  auto commit_volume_continue = [&]() noexcept {
                    sample.ray =
                        make_ray(sample.ray->origin(), sample.ray->direction(),
                                 next.ray_tmin, sample.ray->t_max());
                    // Traversal already replaced Cycles' isect with this volume
                    // boundary hit, which becomes the next ray.self identity.
                    // Explicit self-exclusion is required in addition to t_min
                    // advancement because hardware RayQuery (DXR/Vulkan) may
                    // re-accept the same triangle within 1 ULP of CommittedRayT.
                    sample.ray_source_object = surface.cycles_object_index;
                    sample.ray_source_primitive = surface.cycles_primitive_index;
                    config.volume_state->cross_surface(
                        sample.volume, surface.volume_stack_entry,
                        surface.point.back_facing, surface.surface_has_volume,
                        cycles_closure::label_transmit |
                            cycles_closure::label_transparent);
                  };
                  if (outer_break != nullptr && outer_continue != nullptr) {
                    $if(!next.valid) {
                      *outer_break = true;
                    }
                    $else {
                      commit_volume_continue();
                      *outer_continue = true;
                    };
                  } else {
                    $if(!next.valid) { $break; };
                    commit_volume_continue();
                    $continue;
                  }
                } else {
                  *volume_only_skipped = true;
                  $if(!next.valid) {
                    *sample_done = true;
                  }
                  $else {
                    sample.ray =
                        make_ray(sample.ray->origin(), sample.ray->direction(),
                                 next.ray_tmin, sample.ray->t_max());
                    sample.ray_source_object = surface.cycles_object_index;
                    sample.ray_source_primitive =
                        surface.cycles_primitive_index;
                    config.volume_state->cross_surface(
                        sample.volume, surface.volume_stack_entry,
                        surface.point.back_facing, surface.surface_has_volume,
                        cycles_closure::label_transmit |
                            cycles_closure::label_transparent);
                  };
                }
              };
            }
          };

      const auto emit_nee_closure_eval =
          [&](DirectLightingContext &lighting) noexcept
          -> DirectLightTransportState {
        auto transport =
            DirectLightTransportState::empty(config.spectral_rendering);
        if (!_impl->direct_light_transport) {
          return transport;
        }
        if (prof != nullptr) {
          prof->enter(MegakernelStageProfileState::stage_nee_light_and_bsdf_eval);
        }
        const auto has_evaluable_bsdf =
            (lighting.shading.cycles_surface_runtime_flags &
             cycles_closure::runtime_bsdf_has_eval) != 0u;
        _impl->bounce_random->with_active_state(
            sample, has_evaluable_bsdf, [&](PathBounceRandomState &random) {
              bounce.random_state = &random;
              if (config.use_light_tree) {
                random.selected_light = config.light_tree.surface_sample(
                    random.light_sample.z, lighting.surface.hit_position,
                    lighting.shading.shading_normal, 0.0f,
                    (lighting.shading.cycles_surface_runtime_flags &
                     cycles_closure::runtime_bsdf_has_transmission) != 0u);
              }
              auto direct_light =
                  DirectLightSampleState::empty(config.spectral_rendering);
              std::vector<std::unique_ptr<DirectLightProvider>> light_providers;
              light_providers.reserve(_impl->direct_lighting.size());
              for (const auto &component : _impl->direct_lighting) {
                light_providers.emplace_back(
                    component->make_light_provider(lighting, direct_light));
              }
              for (const auto &provider : light_providers) {
                provider->sample();
              }

              if (config.caustics_active()) {
                $if(direct_light.valid &
                    (direct_light.emitter_kind ==
                     static_cast<std::uint32_t>(
                         sampling::LightDistributionEmitterKind::
                             analytic_light))) {
                  const Var<LightGpu> caustic_light =
                      config.scene->light_buffer->read(direct_light.emitter_index);
                  const auto is_caustic_light =
                      (caustic_light.flags & light_flag_caustics) != 0u;
                  $if(((sample.path_flags &
                        cycles_path_state::flag_caustic_caster_chain) != 0u) &
                      is_caustic_light) {
                    direct_light.valid = false;
                  };
                  const auto surface_object_flags =
                      lighting.surface.cycles_object_flags
                          ? *lighting.surface.cycles_object_flags
                          : config.scene->cycles_svm->objects->object_flag_buffer
                                ->read(lighting.surface.cycles_object_index);
                  const auto on_receiver =
                      ((surface_object_flags &
                        static_cast<std::uint32_t>(
                            compiler::cycles_svm::SD_OBJECT_CAUSTICS_RECEIVER)) !=
                       0u) &
                      !lighting.surface.is_curve;
                  const auto emit_caustic =
                      on_receiver & is_caustic_light & direct_light.valid &
                      ((sample.path_flags &
                        cycles_path_state::flag_caustic_caster_chain) == 0u) &
                      (sample.path_depth < sample.invocation.parameters.max_bounces);
                  $if(emit_caustic) {
                    Var<CausticTaskCall> caustic_task{};
                    caustic_task.path_throughput =
                        config.spectral_rendering
                            ? sample.throughput.lanes()
                            : make_float4(sample.throughput.rgb(), 0.0f);
                    caustic_task.light_shader =
                        config.spectral_rendering
                            ? direct_light.spectral_light_shader.value_or(
                                  make_float4(direct_light.light_shader, 0.0f))
                            : make_float4(direct_light.light_shader, 0.0f);
                    caustic_task.receiver_p = lighting.surface.hit_position;
                    caustic_task.ray_time = lighting.surface.point.time;
                    caustic_task.receiver_wi = lighting.surface.point.incoming;
                    caustic_task.ray_dP = lighting.surface.differential_radius;
                    caustic_task.receiver_ng = lighting.surface.geometric_normal;
                    caustic_task.ray_dD = sample.ray_dD;
                    caustic_task.light_p = direct_light.target_position;
                    caustic_task.light_dir = direct_light.direction;
                    const auto cos_l = abs(dot(direct_light.direction,
                                               direct_light.light_normal));
                    const auto dist_sq =
                        max(direct_light.distance * direct_light.distance,
                            1.0e-20f);
                    caustic_task.light_area_pdf = select(
                        direct_light.pdf * cos_l / dist_sq, direct_light.pdf,
                        direct_light.distant);
                    caustic_task.light_normal = direct_light.light_normal;
                    caustic_task.light_distance = direct_light.distance;
                    caustic_task.path_diffuse_weight = sample.path_diffuse_weight;
                    caustic_task.path_glossy_weight = sample.path_glossy_weight;
                    caustic_task.receiver_bary =
                        lighting.surface.point.barycentric;
                    caustic_task.light_uv = direct_light.light_uv;
                    caustic_task.receiver_inst =
                        lighting.surface.point.instance_id;
                    caustic_task.receiver_prim =
                        lighting.surface.point.primitive_id;
                    caustic_task.receiver_cycles_object =
                        lighting.surface.cycles_object_index;
                    caustic_task.receiver_cycles_shader =
                        lighting.surface.cycles_surface_shader;
                    caustic_task.light_index = direct_light.emitter_index;
                    caustic_task.light_cycles_object =
                        direct_light.light_object;
                    caustic_task.light_shader_flags = direct_light.shader_flags;
                    caustic_task.light_constant_shader =
                        direct_light.constant_light_shader.cast<luisa::uint>();
                    caustic_task.pixel = sample.invocation.pixel;
                    caustic_task.path_depth = sample.path_depth;
                    caustic_task.path_flags = sample.path_flags;
                    caustic_task.path_visibility = sample.cycles_path_visibility;
                    caustic_task.diffuse_depth = sample.diffuse_depth;
                    caustic_task.glossy_depth = sample.glossy_depth;
                    caustic_task.transmission_depth = sample.transmission_depth;
                    caustic_task.transparent_depth = sample.transparent_depth;
                    caustic_task.portal_depth = sample.portal_depth;
                    caustic_task.sample_index = sample.sample_index;
                    caustic_task.rng_hash = sample.rng_hash;
                    caustic_task.rng_offset = sample.cycles_rng_offset;
                    caustic_task.u_lambda =
                        (config.spectral_rendering &&
                         sample.spectral_u_lambda.has_value())
                            ? *sample.spectral_u_lambda
                            : Float{0.0f};
                    if (config.caustics_task_sink) {
                      config.caustics_task_sink->emit(
                          caustic_task,
                          sample.invocation.parameters.wavefront_frame_capacity);
                    } else if (_impl->caustics_evaluator.has_value()) {
                      _impl->caustics_evaluator->evaluate_and_accumulate_sample(
                          sample, caustic_task, sample.invocation.parameters);
                    }
                  };
                };
              }

              auto evaluation =
                  SurfaceEvaluation::zero(config.spectral_rendering);
              $if(direct_light.valid) {
                if (prof != nullptr) {
                  prof->bump(MegakernelStageProfileState::count_nee_evals);
                }
                evaluation = lighting.shading.evaluate_light(
                    sample.invocation, lighting.surface.surface_tag,
                    lighting.surface.point, direct_light.direction,
                    lighting.surface.path_surface_query,
                    direct_light.shader_flags);
              };
              $if(direct_light.valid) {
                const auto cycles_mis_weight =
                    config.light_transport.nee_light_weight(direct_light.pdf,
                                                            evaluation.pdf);
                _impl->direct_light_trace->record_evaluation(
                    bounce,
                    {.distance = direct_light.distance,
                     .bsdf_pdf = evaluation.pdf,
                     .mis_weight = cycles_mis_weight,
                     .bsdf = evaluation.f,
                     .diffuse = evaluation.diffuse_f,
                     .glossy = evaluation.glossy_f});
                const auto mis_weight = select(1.0f, cycles_mis_weight,
                                               direct_light.apply_mis);
                if (config.spectral_rendering) {
                  const auto eval_spec =
                      evaluation.spectral_f.value_or(make_float4(evaluation.f, 0.0f));
                  const auto rad_spec =
                      direct_light.spectral_radiometric_weight.value_or(
                          make_float4(direct_light.radiometric_weight, 0.0f));
                  const auto shader_spec =
                      direct_light.spectral_light_shader.value_or(
                          make_float4(direct_light.light_shader, 0.0f));
                  const auto constant_emission_factor_spec =
                      select(make_float4(1.0f), shader_spec,
                             direct_light.constant_light_shader);
                  const auto weighted_bsdf_spec =
                      eval_spec * rad_spec * constant_emission_factor_spec *
                      (mis_weight / direct_light.normalization_pdf);
                  _impl->direct_light_trace->record_weighted_bsdf(
                      bounce, weighted_bsdf_spec.xyz());
                  $if(any(weighted_bsdf_spec != 0.0f)) {
                    if (prof != nullptr) {
                      prof->bump(MegakernelStageProfileState::count_shadow_traces);
                    }
                    const auto deferred_light_shader_spec =
                        select(shader_spec, make_float4(1.0f),
                               direct_light.constant_light_shader);
                    transport.accept(
                        evaluation,
                        cycles_spectrum::Spectrum::from_lanes(weighted_bsdf_spec),
                        cycles_spectrum::Spectrum::from_lanes(deferred_light_shader_spec),
                        direct_light.direction, direct_light.target_position,
                        direct_light.distant, direct_light.light_object,
                        direct_light.light_primitive, direct_light.shader_flags,
                        direct_light.constant_light_shader);
                    transport.light_terminate_sample =
                        random.light_terminate_sample;
                  };
                } else {
                  const auto constant_emission_factor =
                      select(make_float3(1.0f), direct_light.light_shader,
                             direct_light.constant_light_shader);
                  const auto weighted_bsdf =
                      evaluation.f * direct_light.radiometric_weight *
                      constant_emission_factor *
                      (mis_weight / direct_light.normalization_pdf);
                  _impl->direct_light_trace->record_weighted_bsdf(bounce,
                                                                  weighted_bsdf);
                  $if(any(weighted_bsdf != 0.0f)) {
                    if (prof != nullptr) {
                      prof->bump(MegakernelStageProfileState::count_shadow_traces);
                    }
                    const auto deferred_light_shader =
                        select(direct_light.light_shader, make_float3(1.0f),
                               direct_light.constant_light_shader);
                    transport.accept(
                        evaluation, weighted_bsdf, deferred_light_shader,
                        direct_light.direction, direct_light.target_position,
                        direct_light.distant, direct_light.light_object,
                        direct_light.light_primitive, direct_light.shader_flags,
                        direct_light.constant_light_shader);
                    transport.light_terminate_sample =
                        random.light_terminate_sample;
                  };
                }
              };
              bounce.random_state = nullptr;
            });
        return transport;
      };

      const auto emit_scatter_and_subsurface_exit =
          [&](const SurfaceScatterStage::Result &scatter,
              const std::optional<SubsurfaceTransportPreparation>
                  &subsurface_preparation,
              bool may_have_bssrdf,
              bool outline_subsurface,
              Bool *outer_break,
              Bool *outer_continue) noexcept {
            const bool has_subsurface =
                may_have_bssrdf && _impl->subsurface_transport != nullptr;
            if (on_sample_complete) {
              $if(!scatter.valid) {
                *sample_done = true;
              }
              $else {
                if (has_subsurface) {
                  $if(scatter.subsurface) {
                    $if(!subsurface_preparation->valid) {
                      *sample_done = true;
                    }
                    $else {
                      if (prof != nullptr) {
                        prof->enter(
                            MegakernelStageProfileState::stage_subsurface_trace);
                        prof->bump(
                            MegakernelStageProfileState::count_subsurface_traces);
                      }
                      Bool transported = false;
                      $outline_when_with_name(outline_subsurface,
                                              "path_subsurface_transport") {
                        transported = _impl->subsurface_transport->emit(
                            sample, subsurface_preparation->state);
                      };
                      $if(!transported) { *sample_done = true; };
                    };
                  };
                }
              };
              return;
            }
            if (outer_break != nullptr && outer_continue != nullptr) {
              $if(!scatter.valid) {
                *outer_break = true;
              }
              $else {
                if (has_subsurface) {
                  $if(scatter.subsurface) {
                    $if(!subsurface_preparation->valid) {
                      *outer_break = true;
                    }
                    $else {
                      if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
                        $suspend(path_transition::intersect_subsurface);
                      }
                      if (prof != nullptr) {
                        prof->enter(
                            MegakernelStageProfileState::stage_subsurface_trace);
                        prof->bump(
                            MegakernelStageProfileState::count_subsurface_traces);
                      }
                      Bool transported = false;
                      $outline_when_with_name(outline_subsurface,
                                              "path_subsurface_transport") {
                        transported = _impl->subsurface_transport->emit(
                            sample, subsurface_preparation->state);
                      };
                      $if(transported) { *outer_continue = true; }
                      $else { *outer_break = true; };
                    };
                  };
                }
              };
              return;
            }
            $if(!scatter.valid) { $break; };
            if (has_subsurface) {
              $if(scatter.subsurface) {
                $if(!subsurface_preparation->valid) { $break; };
                if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
                  $suspend(path_transition::intersect_subsurface);
                }
                if (prof != nullptr) {
                  prof->enter(
                      MegakernelStageProfileState::stage_subsurface_trace);
                  prof->bump(
                      MegakernelStageProfileState::count_subsurface_traces);
                }
                Bool transported = false;
                $outline_when_with_name(outline_subsurface,
                                        "path_subsurface_transport") {
                  transported = _impl->subsurface_transport->emit(
                      sample, subsurface_preparation->state);
                };
                $if(transported) { $continue; }
                $else { $break; };
              };
            }
          };

      const auto emit_post_closure_tail =
          [&](DirectLightingContext &lighting,
              const DirectLightTransportState &transport,
              const PreparedCyclesSvmSurfaceScatter &prepared_scatter,
              bool may_have_bssrdf,
              Bool *outer_break = nullptr,
              Bool *outer_continue = nullptr) noexcept {

            std::optional<DirectLightTransportPreparation>
                direct_light_preparation;
            bool defer_direct_light = false;
            if (_impl->direct_light_transport) {
              if (prof != nullptr) {
                prof->enter(MegakernelStageProfileState::stage_shadow_light_nee);
              }
              defer_direct_light =
                  _impl->direct_light_transport
                      ->defer_until_after_surface_scatter(cut_policy);
              direct_light_preparation =
                  _impl->direct_light_transport->prepare(lighting, transport);
              if (!defer_direct_light) {
                _impl->direct_light_transport->emit(
                    bounce, std::move(*direct_light_preparation), cut_policy);
              }
            }

            if (prof != nullptr) {
              prof->enter(MegakernelStageProfileState::stage_bsdf_scatter);
            }
            const auto scatter = commit_cycles_svm_surface_continuation(
                lighting, prepared_scatter, may_have_bssrdf);

            std::optional<SubsurfaceTransportPreparation>
                subsurface_preparation;
            const bool bucket_subsurface =
                may_have_bssrdf && _impl->subsurface_transport != nullptr;
            if (bucket_subsurface) {
              subsurface_preparation.emplace();
              $if(scatter.valid & scatter.subsurface) {
                *subsurface_preparation = _impl->subsurface_transport->prepare(
                    lighting, scatter.sample);
              };
            }

            if (_impl->direct_light_transport && defer_direct_light) {
              if (prof != nullptr) {
                prof->enter(MegakernelStageProfileState::stage_shadow_trace);
              }
              _impl->direct_light_transport->emit(
                  bounce, std::move(*direct_light_preparation), cut_policy);
            }

            emit_scatter_and_subsurface_exit(
                scatter, subsurface_preparation, may_have_bssrdf, false,
                outer_break, outer_continue);
          };

      if (use_post_substage) {
        const bool any_bssrdf = bucket_plan.any_may_have_bssrdf();
        Float3 merged_hit_position = make_float3(0.0f);
        Float3 merged_geometric_normal = make_float3(0.0f, 0.0f, 1.0f);
        Float3 merged_shadow_shading_normal = make_float3(0.0f, 0.0f, 1.0f);
        Float3 merged_shading_normal = make_float3(0.0f, 0.0f, 1.0f);
        Float merged_differential_radius = 0.0f;
        Float merged_shadow_terminator_offset = 0.0f;
        Float merged_point_time = 0.5f;
        UInt merged_object_index = 0u;
        UInt merged_primitive_index = 0u;
        Bool merged_is_curve = false;
        Bool merged_transform_applied = false;
        Bool merged_triangle_smooth = false;
        Bool merged_back_facing = false;
        VolumeStackEntry merged_volume_entry{};
        Bool merged_surface_has_volume = false;
        auto merged_transport =
            DirectLightTransportState::empty(config.spectral_rendering);
        auto merged_scatter =
            PreparedCyclesSvmSurfaceScatter::empty(config.spectral_rendering);

        Bool vt_outer_break = false;
        Bool vt_outer_continue = false;
        const auto commit_bucket_phase_a =
            [&](std::size_t b, SurfaceGeometryContext &surface,
                SurfaceShadingState &shading) noexcept {
          const auto &spec = bucket_plan.buckets[b];
          DirectLightingContext lighting{
              .bounce = bounce, .surface = surface, .shading = shading};
          const auto local_transport = emit_nee_closure_eval(lighting);
          const auto native_state =
              shading.populated_surface->native_surface_state();
          LUISA_ASSERT(native_state.has_value(),
                       "Expected native Cycles SVM surface state.");
          const auto local_scatter = sample_cycles_svm_surface_continuation(
              lighting, *native_state, spec.may_have_bssrdf);
          shading.populated_surface.reset();

          merged_hit_position = surface.hit_position;
          merged_geometric_normal = surface.geometric_normal;
          merged_shadow_shading_normal = surface.shadow_shading_normal;
          merged_shading_normal = shading.shading_normal;
          merged_differential_radius = surface.differential_radius;
          merged_shadow_terminator_offset =
              surface.shadow_terminator_geometry_offset;
          merged_point_time = surface.point.time;
          merged_object_index = surface.cycles_object_index;
          merged_primitive_index = surface.cycles_primitive_index;
          merged_is_curve = surface.is_curve;
          merged_transform_applied = surface.cycles_transform_applied;
          merged_triangle_smooth = surface.triangle_smooth;
          merged_back_facing = surface.point.back_facing;
          if (config.volume_state) {
            merged_volume_entry = surface.volume_stack_entry;
            merged_surface_has_volume = surface.surface_has_volume;
          }
          merged_transport.accept(local_transport);
          merged_scatter.accept(local_scatter, any_bssrdf);
        };

        const auto run_phase_a_with_vt =
            [&](std::size_t b, auto &&emit_suspend) noexcept {
          if (sample.vt_wait_resident) {
            $loop {
              emit_suspend();
              vt_outer_break = false;
              vt_outer_continue = false;
              auto surface = _impl->surface_geometry->emit(bounce);
              emit_volume_only_skip(surface, &vt_outer_break,
                                    &vt_outer_continue);
              $if(!vt_outer_break & !vt_outer_continue) {
                UInt vt_miss = vt_invalid_page;
                sample.vt_miss_page = &vt_miss;
                auto shading = _impl->bucket_surface_shading[b]->emit(
                    surface, &vt_outer_break);
                sample.vt_miss_page = nullptr;
                $if(vt_miss != vt_invalid_page) {
                  vt_outer_break = false;
                  sample.vt_wait_resident(vt_miss);
                  $continue;
                };
                $if(!vt_outer_break) {
                  commit_bucket_phase_a(b, surface, shading);
                };
              };
              $break;
            };
          } else {
            emit_suspend();
            auto surface = _impl->surface_geometry->emit(bounce);
            emit_volume_only_skip(surface);
            auto shading = _impl->bucket_surface_shading[b]->emit(surface);
            commit_bucket_phase_a(b, surface, shading);
          }
        };

        if (use_multi_bucket) {
          const auto shader_id =
              _impl->surface_queue_key->emit(scene, bounce.hit);
          const auto bucket_idx = bucket_plan.emit_bucket_index(shader_id);
          const auto suspend_bucket = [&](std::size_t b) noexcept {
            run_phase_a_with_vt(b, [&]() noexcept {
              const auto &spec = bucket_plan.buckets[b];
              const auto dense_key =
                  bucket_plan.emit_dense_sort_key(b, shader_id);
              $suspend(spec.continuation_name,
                       make_surface_sort_annotation(dense_key, spec.key_count,
                                                    true));
            });
          };
          const auto dispatch_bucket_range =
              [&](auto &&self, std::size_t lo, std::size_t hi) noexcept -> void {
            if (hi - lo <= 1u) {
              suspend_bucket(lo);
              return;
            }
            const std::size_t mid = lo + (hi - lo) / 2u;
            $if(bucket_idx < static_cast<std::uint32_t>(mid)) {
              self(self, lo, mid);
            }
            $else {
              self(self, mid, hi);
            };
          };
          dispatch_bucket_range(dispatch_bucket_range, 0u,
                                bucket_plan.buckets.size());
        } else {
          run_phase_a_with_vt(0u, [&]() noexcept {
            if (_impl->surface_queue_key) {
              const auto surface_queue_key =
                  _impl->surface_queue_key->emit(scene, bounce.hit);
              if (config.staged_surface_sort_annotation) {
                $suspend(path_transition::shade_surface,
                         make_surface_sort_annotation(
                             surface_queue_key, surface_queue_key_range(*scene),
                             true));
              } else {
                $suspend(path_transition::shade_surface,
                         coro_frame_export(path_transition::scheduler_hint,
                                           surface_queue_key));
              }
            } else {
              $suspend(path_transition::shade_surface);
            }
          });
        }
        if (sample.vt_wait_resident) {
          $if(vt_outer_break) { $break; };
          $if(vt_outer_continue) { $continue; };
        }

        $suspend(path_transition::shade_surface_post);

        SurfacePoint post_point{};
        post_point.position = merged_hit_position;
        post_point.geometric_normal = merged_geometric_normal;
        post_point.shading_normal = merged_shading_normal;
        post_point.time = merged_point_time;
        post_point.back_facing = merged_back_facing;
        post_point.is_curve = merged_is_curve;
        post_point.triangle_smooth = merged_triangle_smooth;

        SurfaceGeometryContext post_surface{
            .bounce = bounce,
            .primitive_plan = _impl->primitive_plan,
            .geometric_normal = merged_geometric_normal,
            .hit_position = merged_hit_position,
            .differential_radius = merged_differential_radius,
            .is_curve = merged_is_curve,
            .cycles_transform_applied = merged_transform_applied,
            .triangle_smooth = merged_triangle_smooth,
            .shadow_shading_normal = merged_shadow_shading_normal,
            .cycles_object_index = merged_object_index,
            .cycles_primitive_index = merged_primitive_index,
            .volume_stack_entry = merged_volume_entry,
            .surface_has_volume = merged_surface_has_volume,
            .point = std::move(post_point),
            .shadow_terminator_geometry_offset =
                merged_shadow_terminator_offset};
        SurfaceShadingState post_shading{
            .cycles_surface_runtime_flags = 0u,
            .bsdf_sample = make_float3(0.0f),
            .shading_normal = merged_shading_normal,
            .populated_surface = nullptr};
        DirectLightingContext post_lighting{
            .bounce = bounce, .surface = post_surface, .shading = post_shading};
        emit_post_closure_tail(post_lighting, merged_transport, merged_scatter,
                               any_bssrdf);
      } else if (use_multi_bucket) {
        const auto shader_id =
            _impl->surface_queue_key->emit(scene, bounce.hit);
        const auto bucket_idx = bucket_plan.emit_bucket_index(shader_id);
        Bool vt_outer_break = false;
        Bool vt_outer_continue = false;
        const auto commit_complete_bucket =
            [&](std::size_t b, SurfaceGeometryContext &surface,
                SurfaceShadingState &shading, Bool *outer_break,
                Bool *outer_continue) noexcept {
          const auto &spec = bucket_plan.buckets[b];
          DirectLightingContext lighting{
              .bounce = bounce, .surface = surface, .shading = shading};
          if (fast_post_closure &&
              (!spec.may_have_bssrdf || force_post_closure_on_bssrdf)) {
            const auto transport = emit_nee_closure_eval(lighting);
            const auto native_state =
                shading.populated_surface->native_surface_state();
            LUISA_ASSERT(native_state.has_value(),
                         "Expected native Cycles SVM surface state.");
            const auto prepared_scatter = sample_cycles_svm_surface_continuation(
                lighting, *native_state, spec.may_have_bssrdf);
            shading.populated_surface.reset();
            emit_post_closure_tail(lighting, transport, prepared_scatter,
                                   spec.may_have_bssrdf, outer_break,
                                   outer_continue);
          } else {
            const auto transport = emit_nee_closure_eval(lighting);
            std::optional<DirectLightTransportPreparation>
                direct_light_preparation;
            bool defer_direct_light = false;
            if (_impl->direct_light_transport) {
              defer_direct_light =
                  _impl->direct_light_transport
                      ->defer_until_after_surface_scatter(cut_policy);
              direct_light_preparation =
                  _impl->direct_light_transport->prepare(lighting, transport);
              if (!defer_direct_light) {
                _impl->direct_light_transport->emit(
                    bounce, std::move(*direct_light_preparation), cut_policy);
              }
            }
            const auto scatter = _impl->surface_scatter->emit(lighting);
            std::optional<SubsurfaceTransportPreparation>
                subsurface_preparation;
            const bool bucket_subsurface =
                spec.may_have_bssrdf && _impl->subsurface_transport != nullptr;
            if (bucket_subsurface) {
              subsurface_preparation.emplace();
              $if(scatter.valid & scatter.subsurface) {
                *subsurface_preparation = _impl->subsurface_transport->prepare(
                    lighting, scatter.sample);
              };
            }
            if (_impl->direct_light_transport && defer_direct_light) {
              _impl->direct_light_transport->emit(
                  bounce, std::move(*direct_light_preparation), cut_policy);
            }
            emit_scatter_and_subsurface_exit(
                scatter, subsurface_preparation, spec.may_have_bssrdf, false,
                outer_break, outer_continue);
          }
        };
        const auto run_complete_bucket = [&](std::size_t b) noexcept {
          auto emit_bucket_suspend = [&]() noexcept {
            const auto &spec = bucket_plan.buckets[b];
            const auto dense_key =
                bucket_plan.emit_dense_sort_key(b, shader_id);
            $suspend(
                spec.continuation_name,
                make_surface_sort_annotation(dense_key, spec.key_count, true));
          };
          if (sample.vt_wait_resident) {
            $loop {
              emit_bucket_suspend();
              vt_outer_break = false;
              vt_outer_continue = false;
              auto surface = _impl->surface_geometry->emit(bounce);
              emit_volume_only_skip(surface, &vt_outer_break,
                                    &vt_outer_continue);
              $if(!vt_outer_break & !vt_outer_continue) {
                UInt vt_miss = vt_invalid_page;
                sample.vt_miss_page = &vt_miss;
                auto shading = _impl->bucket_surface_shading[b]->emit(
                    surface, &vt_outer_break);
                sample.vt_miss_page = nullptr;
                $if(vt_miss != vt_invalid_page) {
                  vt_outer_break = false;
                  sample.vt_wait_resident(vt_miss);
                  $continue;
                };
                $if(!vt_outer_break) {
                  commit_complete_bucket(b, surface, shading, &vt_outer_break,
                                         &vt_outer_continue);
                };
              };
              $break;
            };
          } else {
            emit_bucket_suspend();
            auto surface = _impl->surface_geometry->emit(bounce);
            emit_volume_only_skip(surface);
            auto shading = _impl->bucket_surface_shading[b]->emit(surface);
            commit_complete_bucket(b, surface, shading, nullptr, nullptr);
          }
        };
        const auto dispatch_bucket_range =
            [&](auto &&self, std::size_t lo, std::size_t hi) noexcept -> void {
          if (hi - lo <= 1u) {
            run_complete_bucket(lo);
            return;
          }
          const std::size_t mid = lo + (hi - lo) / 2u;
          $if(bucket_idx < static_cast<std::uint32_t>(mid)) {
            self(self, lo, mid);
          }
          $else {
            self(self, mid, hi);
          };
        };
        dispatch_bucket_range(dispatch_bucket_range, 0u,
                              bucket_plan.buckets.size());
        if (sample.vt_wait_resident) {
          $if(vt_outer_break) { $break; };
          $if(vt_outer_continue) { $continue; };
        }
      } else {
        auto emit_single_surface_suspend = [&]() noexcept {
          if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront) {
            if (_impl->surface_queue_key) {
              const auto surface_queue_key =
                  _impl->surface_queue_key->emit(scene, bounce.hit);
              if (config.staged_surface_sort_annotation) {
                $suspend(path_transition::shade_surface,
                         make_surface_sort_annotation(
                             surface_queue_key, surface_queue_key_range(*scene),
                             true));
              } else {
                $suspend(path_transition::shade_surface,
                         coro_frame_export(path_transition::scheduler_hint,
                                           surface_queue_key));
              }
            } else {
              $suspend(path_transition::shade_surface);
            }
          }
        };
        const auto emit_post_surface_shading_body =
            [&](SurfaceGeometryContext &surface, SurfaceShadingState &shading,
                Bool *outer_break, Bool *outer_continue) noexcept {
              const auto &spec = bucket_plan.buckets.front();
              DirectLightingContext lighting{
                  .bounce = bounce, .surface = surface, .shading = shading};
              if (fast_post_closure &&
                  (!spec.may_have_bssrdf || force_post_closure_on_bssrdf)) {
                const auto transport = emit_nee_closure_eval(lighting);
                if (prof != nullptr) {
                  prof->enter(MegakernelStageProfileState::stage_bsdf_scatter);
                }
                const auto native_state =
                    shading.populated_surface->native_surface_state();
                LUISA_ASSERT(native_state.has_value(),
                             "Expected native Cycles SVM surface state.");
                const auto prepared_scatter =
                    sample_cycles_svm_surface_continuation(
                        lighting, *native_state, spec.may_have_bssrdf);
                shading.populated_surface.reset();
                emit_post_closure_tail(lighting, transport, prepared_scatter,
                                       spec.may_have_bssrdf, outer_break,
                                       outer_continue);
              } else {
                std::optional<DirectLightTransportPreparation>
                    direct_light_preparation;
                bool defer_direct_light = false;
                if (_impl->direct_light_transport) {
                  direct_light_preparation.emplace();
                  defer_direct_light =
                      _impl->direct_light_transport
                          ->defer_until_after_surface_scatter(cut_policy);
                  const auto transport = emit_nee_closure_eval(lighting);
                  if (prof != nullptr) {
                    prof->enter(MegakernelStageProfileState::stage_shadow_trace);
                  }
                  *direct_light_preparation =
                      _impl->direct_light_transport->prepare(lighting, transport);
                  if (!defer_direct_light) {
                    _impl->direct_light_transport->emit(
                        bounce, std::move(*direct_light_preparation), cut_policy);
                  }
                }
                if (prof != nullptr) {
                  prof->enter(MegakernelStageProfileState::stage_bsdf_scatter);
                }
                const auto scatter = _impl->surface_scatter->emit(lighting);

                std::optional<SubsurfaceTransportPreparation>
                    subsurface_preparation;
                if (_impl->subsurface_transport) {
                  subsurface_preparation.emplace();
                  $if(scatter.valid & scatter.subsurface) {
                    *subsurface_preparation =
                        _impl->subsurface_transport->prepare(
                            lighting, scatter.sample);
                  };
                }

                if (_impl->direct_light_transport && defer_direct_light) {
                  if (prof != nullptr) {
                    prof->enter(MegakernelStageProfileState::stage_shadow_trace);
                  }
                  _impl->direct_light_transport->emit(
                      bounce, std::move(*direct_light_preparation), cut_policy);
                }

                const bool outline_subsurface =
                    scene != nullptr && static_cast<bool>(scene->device) &&
                    scene->device.backend_name() == "simd" &&
                    scene->device.compute_warp_size() > 1u;
                emit_scatter_and_subsurface_exit(
                    scatter, subsurface_preparation,
                    _impl->subsurface_transport != nullptr, outline_subsurface,
                    outer_break, outer_continue);
              }
            };
        if (sample.vt_wait_resident) {
          Bool vt_outer_break = false;
          Bool vt_outer_continue = false;
          $loop {
            emit_single_surface_suspend();
            vt_outer_break = false;
            vt_outer_continue = false;
            auto surface = _impl->surface_geometry->emit(bounce);
            emit_volume_only_skip(surface, &vt_outer_break,
                                  &vt_outer_continue);
            $if(!vt_outer_break & !vt_outer_continue) {
              UInt vt_miss = vt_invalid_page;
              sample.vt_miss_page = &vt_miss;
              auto shading = _impl->bucket_surface_shading.front()->emit(
                  surface, &vt_outer_break);
              sample.vt_miss_page = nullptr;
              $if(vt_miss != vt_invalid_page) {
                vt_outer_break = false;
                sample.vt_wait_resident(vt_miss);
                $continue;
              };
              $if(!vt_outer_break) {
                emit_post_surface_shading_body(
                    surface, shading, &vt_outer_break, &vt_outer_continue);
              };
            };
            $break;
          };
          $if(vt_outer_break) { $break; };
          $if(vt_outer_continue) { $continue; };
        } else {
          emit_single_surface_suspend();
          const auto emit_megakernel_surface = [&]() noexcept {
            if (prof != nullptr) {
              prof->enter(MegakernelStageProfileState::stage_surface_geometry);
            }
            auto surface = _impl->surface_geometry->emit(bounce);
            emit_volume_only_skip(surface);
            const auto emit_after_volume_skip = [&]() noexcept {
              if (prof != nullptr) {
                prof->enter(
                    MegakernelStageProfileState::stage_surface_svm_shading);
                prof->bump(MegakernelStageProfileState::count_surface_shades);
              }
              auto shading = _impl->bucket_surface_shading.front()->emit(
                  surface, on_sample_complete ? &(*sample_done) : nullptr);
              if (!on_sample_complete) {
                emit_post_surface_shading_body(surface, shading, nullptr,
                                               nullptr);
              } else {
                $if(!*sample_done) {
                  emit_post_surface_shading_body(surface, shading, nullptr,
                                                 nullptr);
                };
              }
            };
            if (!on_sample_complete) {
              emit_after_volume_skip();
            } else {
              $if(!*volume_only_skipped) { emit_after_volume_skip(); };
            }
          };
          if (!on_sample_complete) {
            emit_megakernel_surface();
          } else {
            $if(!*sample_done & !volume_scattered) {
              emit_megakernel_surface();
            };
          }
        }
      }
    }
    if (on_sample_complete) {
      Bool all_samples_done = false;
      $if(*sample_done) {
        all_samples_done = on_sample_complete(sample, path_step);
      };
      $if(all_samples_done) { $break; };
    }
  };
  sample.vt_wait_resident = {};
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
  if (invocation.stage_profile != nullptr) {
    invocation.stage_profile->enter(
        MegakernelStageProfileState::stage_setup_and_film);
  }
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
  const auto initial_t0 = config.megakernel_stage_profile_buffer != nullptr
                              ? luisa::compute::device_clock()
                              : luisa::compute::ULong{};
  const bool sample_parallel = simd_sample_parallel_enabled(config);
  const auto pixel =
      sample_parallel
          ? (luisa::compute::dispatch_y() *
                 (luisa::compute::dispatch_size().x >> 3u) +
             (luisa::compute::dispatch_x() >> 3u))
          : (luisa::compute::dispatch_y() * luisa::compute::dispatch_size().x +
             dispatch_x());
  auto invocation = make_path_invocation(
      config,
      sample_parallel ? PathFilmAccumulation::warp_reduce
                      : PathFilmAccumulation::serial,
      pixel, combined, normal, albedo, light_passes, sample_count,
      volume_guiding_raw, volume_guiding_denoised, path_trace, sample_first,
      sobol_table, filter_table, parameters);
  if (config.megakernel_stage_profile_buffer != nullptr) {
    auto state = std::make_shared<MegakernelStageProfileState>();
    for (std::uint32_t i = 0u; i < MegakernelStageProfileState::stage_count; ++i) {
      state->cycles[i] = 0ull;
    }
    for (std::uint32_t i = 0u; i < MegakernelStageProfileState::counter_count; ++i) {
      state->counts[i] = 0ull;
    }
    state->t_last = initial_t0;
    state->current_stage = MegakernelStageProfileState::stage_setup_and_film;
    invocation.stage_profile = std::move(state);
  }
  const auto backend_name =
      (config.scene && static_cast<bool>(config.scene->device))
          ? std::string_view{config.scene->device.backend_name()}
          : std::string_view{};
  const auto *persistent_raw = std::getenv("PSYCLES_SIMD_PERSISTENT_SAMPLES");
  const bool persistent_simd_samples =
      !sample_parallel && backend_name == "simd" &&
      persistent_raw != nullptr && *persistent_raw != '\0' &&
      std::string_view{persistent_raw} != "0" &&
      std::string_view{persistent_raw} != "off";
  if (sample_parallel) {
    UInt sub_spp_index = luisa::compute::dispatch_x() & 7u;
    $while(sub_spp_index < samples) {
      emit_path_sample(pipeline, invocation, sub_spp_index,
                       PathCoroutineCutPolicy::none);
      sub_spp_index += 8u;
    };
  } else if (persistent_simd_samples) {
    $if(samples != 0u) {
      UInt sub_spp_index = 0u;
      auto sample = begin_path_sample(invocation, sub_spp_index);
      pipeline.emit(
          sample, PathCoroutineCutPolicy::none,
          [&](PathSampleContext &active_sample,
              UInt &path_step) noexcept -> Bool {
            if (invocation.stage_profile != nullptr) {
              invocation.stage_profile->enter(
                  MegakernelStageProfileState::stage_setup_and_film);
            }
            accumulate_path_sample(active_sample);
            sub_spp_index += 1u;
            const Bool done = sub_spp_index >= samples;
            $if(!done) {
              reset_path_sample(active_sample, sub_spp_index);
              path_step = ~0u;
            };
            return done;
          });
    };
  } else {
    $for(sub_spp_index, samples) {
      emit_path_sample(pipeline, invocation, sub_spp_index,
                       PathCoroutineCutPolicy::none);
    };
  }
  invocation.write_film();
  if (config.megakernel_stage_profile_buffer != nullptr &&
      invocation.stage_profile != nullptr) {
    auto &prof = *invocation.stage_profile;
    const auto now = luisa::compute::device_clock();
    prof.cycles[prof.current_stage] += now - prof.t_last;
    const auto stripe_base =
        ((luisa::compute::block_x() + luisa::compute::block_y() * 257u) &
         (MegakernelStageProfileState::stripe_count - 1u)) *
        MegakernelStageProfileState::stripe_stride;
    for (std::uint32_t i = 0u; i < MegakernelStageProfileState::stage_count; ++i) {
      (*config.megakernel_stage_profile_buffer)
          ->atomic(stripe_base + i)
          .fetch_add(prof.cycles[i]);
    }
    for (std::uint32_t i = 0u; i < MegakernelStageProfileState::counter_count; ++i) {
      (*config.megakernel_stage_profile_buffer)
          ->atomic(stripe_base + MegakernelStageProfileState::stage_count + i)
          .fetch_add(prof.counts[i]);
    }
  }
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
  UInt pixel;
  UInt sub_spp_index;
  if (cut_policy == PathCoroutineCutPolicy::cycles_wavefront &&
      config.enable_virtual_texture_cache && !config.path_trace_enabled) {
    // 128x128 serpentine screen-tile ordering keeps active wavefront samples
    // spatially coherent to shrink the simultaneous virtual-texture working set.
    const auto w = luisa::compute::dispatch_size().x;
    const auto h = luisa::compute::dispatch_size().y;
    const auto s = luisa::compute::dispatch_size().z;
    const auto linear =
        dispatch_x() + luisa::compute::dispatch_y() * w +
        luisa::compute::dispatch_z() * (w * h);
    const auto tile_h_nom = min(UInt(128u), h);
    const auto tile_w_nom = min(UInt(128u), w);
    const auto band_size = w * tile_h_nom * s;
    const auto band = linear / band_size;
    const auto rem = linear - band * band_size;
    const auto y0 = band * tile_h_nom;
    const auto bh = min(tile_h_nom, h - y0);
    const auto col_tile_size = tile_w_nom * bh * s;
    const auto tx_raw = rem / col_tile_size;
    const auto rem2 = rem - tx_raw * col_tile_size;
    const auto x0 = tx_raw * tile_w_nom;
    const auto tw = min(tile_w_nom, w - x0);
    const auto tile_pixels = tw * bh;
    const auto sub_spp = rem2 / tile_pixels;
    const auto p_in_tile = rem2 - sub_spp * tile_pixels;
    const auto ly = p_in_tile / tw;
    const auto lx = p_in_tile - ly * tw;
    const auto px_raw = x0 + lx;
    const auto px = ite((band & 1u) != 0u, (w - 1u) - px_raw, px_raw);
    const auto py = y0 + ly;
    pixel = py * w + px;
    sub_spp_index = sub_spp;
  } else {
    pixel = luisa::compute::dispatch_y() * luisa::compute::dispatch_size().x +
            dispatch_x();
    sub_spp_index = luisa::compute::dispatch_z();
  }
  auto invocation = make_path_invocation(
      config, PathFilmAccumulation::atomic, pixel, combined, normal, albedo,
      light_passes, sample_count, volume_guiding_raw, volume_guiding_denoised,
      path_trace, sample_first, sobol_table, filter_table, parameters);
  emit_path_sample(pipeline, invocation, sub_spp_index, cut_policy);
  invocation.write_film();
}

} // namespace

[[nodiscard]] static luisa::uint3 resolve_megakernel_block_size(
    std::string_view backend_name = {}, bool has_volume_state = false) noexcept {
  static_cast<void>(has_volume_state);
  const auto default_block =
      backend_name == "simd" ? luisa::make_uint3(4u, 8u, 1u)
                             : luisa::make_uint3(8u, 8u, 1u);
  const auto *raw = std::getenv("PSYCLES_MEGAKERNEL_BLOCK_SIZE");
  if (raw == nullptr || *raw == '\0') {
    return default_block;
  }
  std::string_view sv{raw};
  auto next_uint = [&sv]() -> std::optional<std::uint32_t> {
    while (!sv.empty() &&
           (sv.front() == ' ' || sv.front() == ',' || sv.front() == 'x' ||
            sv.front() == 'X')) {
      sv.remove_prefix(1u);
    }
    if (sv.empty()) {
      return std::nullopt;
    }
    std::uint32_t v = 0u;
    const auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), v);
    if (ec != std::errc{} || v == 0u) {
      return std::nullopt;
    }
    sv.remove_prefix(static_cast<std::size_t>(ptr - sv.data()));
    return v;
  };
  const auto bx = next_uint();
  const auto by = next_uint();
  if (!bx || !by) {
    return default_block;
  }
  const auto bz = next_uint().value_or(1u);
  return luisa::make_uint3(*bx, *by, bz);
}

RenderSerialKernel
build_path_serial_kernel(const PathKernelConfig &config) noexcept {
  PathKernelPipeline pipeline{config};
  const auto backend_name =
      (config.scene && static_cast<bool>(config.scene->device))
          ? std::string_view{config.scene->device.backend_name()}
          : std::string_view{};
  const auto block_size = resolve_megakernel_block_size(
      backend_name, config.volume_state != nullptr);
  RenderSerialKernel kernel =
      [&config, &pipeline, block_size](
          BufferFloat4 combined, BufferFloat4 normal, BufferFloat4 albedo,
          BufferFloat4 light_passes, BufferUInt sample_count,
          BufferFloat4 volume_guiding_raw, BufferUInt volume_guiding_denoised,
          BufferFloat4 path_trace, UInt sample_first, UInt samples,
          BufferFloat4 sobol_table, BufferFloat filter_table,
          Var<RenderKernelParameters> parameters) noexcept {
        // Keep neighboring image rows in the same SIMD groups. This matches
        // the spatially coherent per-sample topology while preserving the
        // serial film path and its deterministic accumulation order.
        set_block_size(block_size.x, block_size.y, block_size.z);
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
  const auto block_size = resolve_megakernel_block_size();
  RenderSampleKernel kernel =
      [&config, &pipeline, block_size](
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
        set_block_size(block_size.x, block_size.y, block_size.z);
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

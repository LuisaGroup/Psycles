#include "path_kernel_direct_light_task.h"
#include "path_kernel_builder.h"
#include "path_kernel_film.h"
#include "path_kernel_transitions.h"
#include "path_tracer_cycles_svm_light.h"

#include <psycles/luisa/surface_ray.h>
#include <luisa/dsl/coro_func.h>

#include <utility>

namespace psycles::luisa_backend::detail {

Float3 finalize_direct_light_sample(
    const LightSampleRouletteCallable &light_sample_roulette,
    Float3 local_unshadowed, Float3 path_throughput,
    Float light_terminate_sample, Float inverse_threshold) noexcept {
  const auto roulette_weight = light_sample_roulette(
      local_unshadowed, light_terminate_sample, inverse_threshold);
  return path_throughput * local_unshadowed * roulette_weight;
}

Var<ShadowTraceResultCall> DirectLightTaskEvaluator::trace(
    const Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  Var<luisa::compute::Ray> shadow_ray = make_ray(
      task.ray_origin, task.ray_direction, task.ray_minimum, task.ray_maximum);
  return trace_shadow(
      shadow_ray, task.ray_dP, task.ray_dD, task.source_object,
      task.source_primitive, task.light_object, task.light_primitive,
      parameters.transparent_max_bounces,
      parameters.shadow_storage_capacity,
      parameters.shadow_storage_block_size,
      task.shadow_throughput,
      make_shadow_shader_context(
          pack_shader_evaluation_state(cycles_path_state::shadow_shader_state(
              task.path_depth, task.diffuse_depth, task.glossy_depth,
              task.transparent_depth, task.transmission_depth)),
          task.ray_time, task.sample_index, task.rng_hash, task.rng_offset,
          task.volume_bounds_bounce),
      parameters);
}

void DirectLightTaskEvaluator::trace_staged(
    Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters, Bool &active,
    Bool &visible) const noexcept {
  $while(active) {
    $suspend(path_transition::intersect_shadow);
    const auto shadow_batch = intersect(task, parameters);
    // Cycles integrator_intersect_shadow terminates opaque paths here. An
    // empty, unblocked batch still needs SHADE_SHADOW for film output.
    $if(shadow_batch.blocked != 0u) { active = false; }
    $else {
      $suspend(path_transition::shade_shadow);
      const auto step = shade_shadow(task, shadow_batch, parameters);
      active = step.continue_shadow;
      visible |= step.visible;
    };
  };
}

Bool DirectLightTaskEvaluator::shade_light_nee(
    Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  if (light_emission) {
    task.light_shader = light_emission->evaluate(task, parameters);
  }
  const auto local_unshadowed =
      task.unshadowed_contribution * task.light_shader;
  task.unshadowed_contribution = finalize_direct_light_sample(
      light_sample_roulette, local_unshadowed, task.nee_path_throughput,
      task.light_terminate_sample, parameters.light_inv_rr_threshold);
  task.shadow_throughput = task.unshadowed_contribution;
  return any(task.unshadowed_contribution != 0.0f);
}

Var<ShadowIntersectionBatchCall> DirectLightTaskEvaluator::intersect(
    const Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  LUISA_ASSERT(intersect_shadow != nullptr,
               "Split shadow traversal requires an intersection component.");
  const auto ray = make_ray(task.ray_origin, task.ray_direction,
                            task.ray_minimum, task.ray_maximum);
  const auto remaining =
      parameters.transparent_max_bounces -
      min(task.transparent_depth, parameters.transparent_max_bounces);
  return intersect_shadow->collect(
      ray, task.source_object, task.source_primitive, task.light_object,
      task.light_primitive, remaining, parameters.shadow_storage_capacity,
      parameters.shadow_storage_block_size);
}

DirectLightShadowStep DirectLightTaskEvaluator::shade_shadow(
    Var<DirectLightTaskCall> &task,
    const Var<ShadowIntersectionBatchCall> &shadow_batch,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  Bool continue_shadow = false;
  Bool visible = false;

  $if(shadow_batch.blocked == 0u) {
    auto ordered_batch = def(shadow_batch);
    sort_shadow_intersection_batch(ordered_batch);
    Bool carries_light = true;
    Float last_distance = task.ray_minimum;
    const auto ray = make_ray(task.ray_origin, task.ray_direction,
                              task.ray_minimum, task.ray_maximum);
    auto context = make_shadow_shader_context(
        pack_shader_evaluation_state(cycles_path_state::shadow_shader_state(
            task.path_depth, task.diffuse_depth, task.glossy_depth,
            task.transparent_depth, task.transmission_depth)),
        task.ray_time, task.sample_index, task.rng_hash, task.rng_offset,
        task.volume_bounds_bounce);
    auto throughput = def(task.shadow_throughput);
    for (auto index = std::size_t{0u};
         index < shadow_intersection_batch_capacity; ++index) {
      const auto shade =
          static_cast<std::uint32_t>(index) < ordered_batch->count;
      $if(shade & carries_light) {
        const auto &hit =
            ordered_batch->hits[static_cast<luisa::uint>(index)];
        const auto surface = shade_shadow_surface(
            ray, hit, task.ray_dP, task.ray_dD, context, parameters);
        carries_light =
            advance_shadow_surface_state(context, throughput, surface);
        $if(carries_light) {
          last_distance = hit->distance;
        };
      };
    }
    task.shadow_throughput = throughput;
    task.transparent_depth = context.path.transparent_depth;
    task.rng_offset = context.rng_offset;
    task.volume_bounds_bounce = context.volume_bounds_bounce;
    carries_light &= context.volume_bounds_bounce <= shadow_volume_bounds_max;
    $if(carries_light) {
      const auto has_remaining =
          ordered_batch->count == static_cast<luisa::uint>(
                                      shadow_intersection_batch_capacity);
      $if(has_remaining) {
        task.ray_minimum = surface_ray::intersection_t_offset(last_distance);
        continue_shadow = true;
      }
      $else { visible = true; };
    };
  };
  return {.continue_shadow = std::move(continue_shadow),
          .visible = std::move(visible)};
}

Float3 DirectLightTaskEvaluator::contribution(
    const Var<DirectLightTaskCall> &task, Float3 throughput,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  return clamp_contribution(throughput,
                            task.path_depth, parameters.sample_clamp_direct,
                            parameters.sample_clamp_indirect);
}

Var<LightPassContributionCall>
DirectLightTaskEvaluator::split(const Var<DirectLightTaskCall> &task,
                                Float3 contribution) const noexcept {
  return split_scattered_light(contribution, task.diffuse_weight,
                               task.glossy_weight, task.path_depth == 0u);
}

void DirectLightTaskEvaluator::emit_atomic(
    const Var<DirectLightTaskCall> &input, const DirectLightTaskFilm &film,
    const Var<RenderKernelParameters> &parameters) const noexcept {
  auto task = def(input);
  Bool active = true;
  $if(task.constant_light_shader == 0u) {
    active = shade_light_nee(task, parameters);
  };
  $if(active) {
    const auto shadow = trace(task, parameters);
    $if(any(shadow->throughput != 0.0f)) {
      const auto value =
          contribution(task, shadow->throughput, parameters);
      atomic_accumulate_radiance(
          film.combined, film.volume_guiding_raw, task.pixel,
          task.pixel * volume_guiding::raw_pixel_stride, volume_guiding,
          task.path_flags, task.path_visibility, task.path_depth, value);
      atomic_accumulate_light_passes(
          film.light_passes, task.pixel * light_pass_buffer_count,
          split(task, value));
    };
  };
}

DirectLightTaskEvaluator
make_direct_light_task_evaluator(const PathKernelConfig &config) noexcept {
  return {.intersect_shadow = config.intersect_shadow,
          .shade_shadow_surface = config.shade_shadow_surface,
          .trace_shadow = config.trace_shadow,
          .light_sample_roulette =
              config.light_transport.light_sample_roulette_weight,
          .clamp_contribution = config.light_transport.clamp_light_contribution,
          .split_scattered_light = config.light_transport.split_scattered_light,
          .light_emission = config.scene->native_cycles_svm_surface
                                ? make_cycles_svm_light_emission_component(
                                      config.scene, config.camera_projection,
                                      config.reflective_caustics,
                                      config.refractive_caustics)
                                : nullptr,
          .volume_guiding = config.volume_state != nullptr};
}

} // namespace psycles::luisa_backend::detail

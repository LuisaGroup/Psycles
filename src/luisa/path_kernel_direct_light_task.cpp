#include "path_kernel_direct_light_task.h"
#include "path_kernel_builder.h"
#include "path_kernel_film.h"
#include "path_kernel_transitions.h"
#include "virtual_texture_cache.h"
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

Float4 finalize_direct_light_sample_spectral(
    const LightSampleRouletteCallable &light_sample_roulette,
    Float4 local_unshadowed, Float4 path_throughput, Bool secondary_terminated,
    Float light_terminate_sample, Float inverse_threshold) noexcept {
  const auto max_val = cycles_spectrum::max_abs_component(
      cycles_spectrum::Spectrum::from_lanes(local_unshadowed),
      secondary_terminated);
  const auto roulette_weight = light_sample_roulette(
      make_float3(max_val, 0.0f, 0.0f), light_terminate_sample,
      inverse_threshold);
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
              task.transparent_depth.cast<unsigned>(), task.transmission_depth,
              task.portal_depth.cast<unsigned>())),
          task.ray_time, task.sample_index, task.rng_hash, task.rng_offset,
          task.volume_bounds_bounce),
      parameters);
}

void DirectLightTaskEvaluator::trace_spectral(
    Var<DirectLightTaskCall> &task, DirectLightSpectralExtension &spectral,
    const Var<RenderKernelParameters> &parameters, Bool &visible,
    const std::function<void(Expr<luisa::uint>)> &vt_wait_resident)
    const noexcept {
  Bool active = true;
  $while(active) {
    const auto shadow_batch = intersect(task, parameters);
    $if(shadow_batch.blocked != 0u) { active = false; }
    $else {
      if (vt_wait_resident) {
        $loop {
          UInt vt_miss = vt_invalid_page;
          const auto step = shade_shadow(task, shadow_batch, parameters,
                                         &vt_miss, &spectral);
          $if(vt_miss != vt_invalid_page) {
            vt_wait_resident(vt_miss);
            $continue;
          };
          active = step.continue_shadow;
          visible |= step.visible;
          $break;
        };
      } else {
        const auto step =
            shade_shadow(task, shadow_batch, parameters, nullptr, &spectral);
        active = step.continue_shadow;
        visible |= step.visible;
      }
    };
  };
}

void DirectLightTaskEvaluator::trace_staged(
    Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters, Bool &active,
    Bool &visible,
    const std::function<void(Expr<luisa::uint>)> &vt_wait_resident,
    DirectLightSpectralExtension *spectral)
    const noexcept {
  $while(active) {
    $suspend(path_transition::intersect_shadow);
    const auto shadow_batch = intersect(task, parameters);
    // Cycles integrator_intersect_shadow terminates opaque paths here. An
    // empty, unblocked batch still needs SHADE_SHADOW for film output.
    $if(shadow_batch.blocked != 0u) { active = false; }
    $else {
      if (vt_wait_resident) {
        $loop {
          $suspend(path_transition::shade_shadow);
          UInt vt_miss = vt_invalid_page;
          const auto step =
              shade_shadow(task, shadow_batch, parameters, &vt_miss, spectral);
          $if(vt_miss != vt_invalid_page) {
            vt_wait_resident(vt_miss);
            $continue;
          };
          active = step.continue_shadow;
          visible |= step.visible;
          $break;
        };
      } else {
        $suspend(path_transition::shade_shadow);
        const auto step =
            shade_shadow(task, shadow_batch, parameters, nullptr, spectral);
        active = step.continue_shadow;
        visible |= step.visible;
      }
    };
  };
}

Bool DirectLightTaskEvaluator::shade_light_nee(
    Var<DirectLightTaskCall> &task,
    const Var<RenderKernelParameters> &parameters,
    UInt *vt_miss_page,
    DirectLightSpectralExtension *spectral) const noexcept {
  const bool use_hero4 =
      spectral != nullptr && spectral_tables != nullptr &&
      spectral_mode == contract::SpectralMode::hero4;
  const bool use_xyz3 =
      spectral != nullptr && spectral_tables != nullptr &&
      spectral_mode == contract::SpectralMode::xyz3;
  Float3 evaluated_shader = task.light_shader;
  Float4 evaluated_spec = make_float4(0.0f);
  if (use_hero4) {
    evaluated_spec = make_float4(task.light_shader, spectral->light_shader_w);
    if (light_emission) {
      const auto wavelengths =
          cycles_spectrum::sample_visible_wavelengths(spectral->u_lambda);
      const auto secondary_terminated =
          (task.path_flags &
           cycles_path_state::flag_spectral_secondary_terminated) != 0u;
      const cycles_spectrum::SpectralContext ctx{
          spectral_tables.get(), wavelengths, secondary_terminated};
      evaluated_spec = light_emission->evaluate_spectral(
          task, parameters, ctx, vt_miss_page);
    }
  } else if (use_xyz3) {
    if (light_emission) {
      const auto collapsed =
          (task.path_flags & cycles_path_state::flag_spectral_collapsed) != 0u;
      const auto ctx = cycles_spectrum::make_spectral_context(
          spectral_tables.get(), contract::SpectralMode::xyz3,
          spectral->u_lambda, collapsed);
      const auto eval4 =
          light_emission->evaluate_spectral(task, parameters, ctx, vt_miss_page);
      evaluated_shader =
          cycles_spectrum::Spectrum::from_eval4(&ctx, eval4).xyz3();
    }
  } else {
    if (light_emission) {
      evaluated_shader = light_emission->evaluate(task, parameters, vt_miss_page);
    }
  }
  const auto commit_nee = [&]() noexcept {
    if (use_hero4) {
      task.light_shader = evaluated_spec.xyz();
      spectral->light_shader_w = evaluated_spec.w;
      const auto local_unshadowed =
          make_float4(task.unshadowed_contribution,
                      spectral->unshadowed_contribution_w) *
          evaluated_spec;
      const auto path_throughput = make_float4(
          task.nee_path_throughput, spectral->nee_path_throughput_w);
      const auto secondary_terminated =
          (task.path_flags &
           cycles_path_state::flag_spectral_secondary_terminated) != 0u;
      const auto final_spec = finalize_direct_light_sample_spectral(
          light_sample_roulette, local_unshadowed, path_throughput,
          secondary_terminated, task.light_terminate_sample,
          parameters.light_inv_rr_threshold);
      task.unshadowed_contribution = final_spec.xyz();
      spectral->unshadowed_contribution_w = final_spec.w;
      task.shadow_throughput = final_spec.xyz();
      spectral->shadow_throughput_w = final_spec.w;
      return any(final_spec != 0.0f);
    }
    task.light_shader = evaluated_shader;
    const auto local_unshadowed =
        task.unshadowed_contribution * task.light_shader;
    task.unshadowed_contribution = finalize_direct_light_sample(
        light_sample_roulette, local_unshadowed, task.nee_path_throughput,
        task.light_terminate_sample, parameters.light_inv_rr_threshold);
    task.shadow_throughput = task.unshadowed_contribution;
    return any(task.unshadowed_contribution != 0.0f);
  };
  if (vt_miss_page != nullptr) {
    Bool active = false;
    $if(*vt_miss_page == vt_invalid_page) {
      active = commit_nee();
    };
    return active;
  }
  return commit_nee();
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
      min(task.transparent_depth.cast<unsigned>(), parameters.transparent_max_bounces);
  return intersect_shadow->collect(
      ray, task.source_object, task.source_primitive, task.light_object,
      task.light_primitive, remaining, parameters.shadow_storage_capacity,
      parameters.shadow_storage_block_size);
}

DirectLightShadowStep DirectLightTaskEvaluator::shade_shadow(
    Var<DirectLightTaskCall> &task,
    const Var<ShadowIntersectionBatchCall> &shadow_batch,
    const Var<RenderKernelParameters> &parameters,
    UInt *vt_miss_page,
    DirectLightSpectralExtension *spectral) const noexcept {
  const bool use_hero4 =
      spectral != nullptr && spectral_tables != nullptr &&
      spectral_mode == contract::SpectralMode::hero4;
  const bool use_xyz3 =
      spectral != nullptr && spectral_tables != nullptr &&
      spectral_mode == contract::SpectralMode::xyz3;
  Bool continue_shadow = false;
  Bool visible = false;
  Bool vt_missed = false;

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
            task.transparent_depth.cast<unsigned>(), task.transmission_depth,
            task.portal_depth.cast<unsigned>())),
        task.ray_time, task.sample_index, task.rng_hash, task.rng_offset,
        task.volume_bounds_bounce);
    auto throughput = def(task.shadow_throughput);
    Float4 throughput4 = make_float4(0.0f);
    std::optional<cycles_spectrum::SampledWavelengths> wavelengths;
    std::optional<cycles_spectrum::SpectralContext> xyz3_ctx;
    if (use_hero4) {
      throughput4 = make_float4(task.shadow_throughput,
                                spectral->shadow_throughput_w);
      wavelengths.emplace(
          cycles_spectrum::sample_visible_wavelengths(spectral->u_lambda));
    } else if (use_xyz3) {
      const auto collapsed =
          (task.path_flags & cycles_path_state::flag_spectral_collapsed) != 0u;
      xyz3_ctx.emplace(cycles_spectrum::make_spectral_context(
          spectral_tables.get(), contract::SpectralMode::xyz3,
          spectral->u_lambda, collapsed));
    }
    $for(index, ordered_batch->count) {
      $if(!carries_light) { $break; };
      const auto hit = ordered_batch->hits[index];
      const auto surface = shade_shadow_surface(
          ray, hit, task.ray_dP, task.ray_dD, context, parameters);
      if (vt_miss_page != nullptr) {
        $if(surface->vt_miss_page != vt_invalid_page) {
          *vt_miss_page = ite(*vt_miss_page == vt_invalid_page,
                              surface->vt_miss_page, *vt_miss_page);
          vt_missed = true;
          carries_light = false;
          $break;
        };
      }
      if (use_hero4) {
        context.volume_bounds_bounce =
            (context.volume_bounds_bounce + surface->volume_boundary) & 0xffffu;
        const auto t = surface->transmittance;
        auto trans4 = cycles_spectrum::rgb_to_spectrum_reflectance(
            *spectral_tables, t, wavelengths->lambdas);
        trans4 = select(trans4, make_float4(1.0f),
                        (t.x == 1.0f) & (t.y == 1.0f) & (t.z == 1.0f));
        trans4 = select(trans4, make_float4(0.0f),
                        (t.x == 0.0f) & (t.y == 0.0f) & (t.z == 0.0f));
        const auto next4 = throughput4 * trans4;
        const auto nonzero4 = any(next4 != 0.0f);
        $if(nonzero4) {
          throughput4 = next4;
          context.path.transparent_depth =
              (context.path.transparent_depth + 1u) & 0xffffu;
          context.rng_offset =
              (context.rng_offset + cycles_path_state::bounce_dimension_count) &
              0xffffu;
        };
        carries_light =
            nonzero4 & (context.volume_bounds_bounce <= shadow_volume_bounds_max);
      } else if (use_xyz3) {
        context.volume_bounds_bounce =
            (context.volume_bounds_bounce + surface->volume_boundary) & 0xffffu;
        const auto t = surface->transmittance;
        auto trans3 = cycles_spectrum::Spectrum::from_rgb_reflectance(
                          spectral_tables.get(), t, &*xyz3_ctx)
                          .xyz3();
        trans3 = select(trans3, make_float3(1.0f),
                        (t.x == 1.0f) & (t.y == 1.0f) & (t.z == 1.0f));
        trans3 = select(trans3, make_float3(0.0f),
                        (t.x == 0.0f) & (t.y == 0.0f) & (t.z == 0.0f));
        const auto next3 = throughput * trans3;
        const auto nonzero3 = any(next3 != 0.0f);
        $if(nonzero3) {
          throughput = next3;
          context.path.transparent_depth =
              (context.path.transparent_depth + 1u) & 0xffffu;
          context.rng_offset =
              (context.rng_offset + cycles_path_state::bounce_dimension_count) &
              0xffffu;
        };
        carries_light =
            nonzero3 & (context.volume_bounds_bounce <= shadow_volume_bounds_max);
      } else {
        carries_light =
            advance_shadow_surface_state(context, throughput, surface);
      }
      $if(carries_light) {
        last_distance = max(last_distance, hit.distance);
      };
    };
    const auto commit_shadow = [&]() noexcept {
      if (use_hero4) {
        task.shadow_throughput = throughput4.xyz();
        spectral->shadow_throughput_w = throughput4.w;
      } else {
        task.shadow_throughput = throughput;
      }
      task.transparent_depth =
          context.path.transparent_depth.cast<std::uint16_t>();
      task.rng_offset = context.rng_offset;
      task.volume_bounds_bounce = context.volume_bounds_bounce;
      carries_light &= context.volume_bounds_bounce <= shadow_volume_bounds_max;
      $if(carries_light) {
        const auto has_remaining =
            ordered_batch->count ==
            static_cast<luisa::uint>(shadow_intersection_batch_capacity);
        $if(has_remaining) {
          task.ray_minimum =
              surface_ray::intersection_t_offset(last_distance);
          continue_shadow = true;
        }
        $else { visible = true; };
      };
    };
    if (vt_miss_page != nullptr) {
      $if(!vt_missed) {
        commit_shadow();
      };
    } else {
      commit_shadow();
    }
  };
  return {.continue_shadow = std::move(continue_shadow),
          .visible = std::move(visible)};
}

Float3 DirectLightTaskEvaluator::contribution(
    const Var<DirectLightTaskCall> &task, Float3 throughput,
    const Var<RenderKernelParameters> &parameters,
    const DirectLightSpectralExtension *spectral) const noexcept {
  if (spectral != nullptr && spectral_tables != nullptr) {
    if (spectral_mode == contract::SpectralMode::xyz3) {
      const auto rgb = cycles_spectrum::xyz3_to_rgb(*spectral_tables, throughput);
      return clamp_contribution(rgb, task.path_depth,
                                parameters.sample_clamp_direct,
                                parameters.sample_clamp_indirect);
    }
    const auto wavelengths =
        cycles_spectrum::sample_visible_wavelengths(spectral->u_lambda);
    const auto secondary_terminated =
        (task.path_flags &
         cycles_path_state::flag_spectral_secondary_terminated) != 0u;
    const auto rgb = cycles_spectrum::spectrum_to_rgb(
        *spectral_tables,
        make_float4(throughput, spectral->shadow_throughput_w),
        wavelengths, secondary_terminated);
    return clamp_contribution(rgb, task.path_depth,
                              parameters.sample_clamp_direct,
                              parameters.sample_clamp_indirect);
  }
  return clamp_contribution(throughput,
                            task.path_depth, parameters.sample_clamp_direct,
                            parameters.sample_clamp_indirect);
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
      accumulate_passes_atomic(
          film.light_passes, task.pixel * light_pass_buffer_count,
          task, value);
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
          .light_emission = make_cycles_svm_light_emission_component(
              config.scene, config.camera_projection,
              config.reflective_caustics, config.refractive_caustics),
          .spectral_tables = config.spectral_rendering
                                 ? config.scene->spectral_tables
                                 : nullptr,
          .spectral_mode = config.effective_spectral_mode(),
          .volume_guiding = config.volume_state != nullptr};
}

} // namespace psycles::luisa_backend::detail

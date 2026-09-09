/* SPDX-FileCopyrightText: 2011-2023 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "path_kernel_cycles_svm_surface_scatter.h"
#include "cycles_svm_surface_integrator.h"
#include "cycles_svm_surface_shader.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_ray_differential.h>

namespace psycles::luisa_backend::detail {
namespace svm = cycles_svm;
namespace native = cycles_svm::detail;
namespace closure = cycles_closure;
namespace abi = compiler::cycles_svm;
namespace path = cycles_path_state;

SurfaceScatterStage::Result emit_cycles_svm_surface_scatter(
    DirectLightingContext &context, const CyclesSvmSurfaceState &population) noexcept {
  auto &sample = context.bounce.sample;
  const auto &config = sample.invocation.config;
  const auto &parameters = sample.invocation.parameters;
  auto &sd = population.shader_data;
  const auto &kg = population.kernel_globals;
  const auto &pool = *sd.closure;
  const auto features = config.scene->cycles_svm->kernel_features;
  const auto has_portal = (features & svm::kernel_feature_node_portal) != 0u;
  auto bssrdf_sample = SurfaceSample::zero();
  Bool subsurface = false;
  Bool valid = false;
  const path::Limits limits{
      .maximum = parameters.max_bounces,
      .maximum_diffuse = parameters.max_diffuse_bounces,
      .maximum_glossy = parameters.max_glossy_bounces,
      .maximum_transmission = parameters.max_transmission_bounces,
      .maximum_transparent = parameters.transparent_max_bounces};
  const auto trace = [&](path_trace_schema::EventSlot slot, Float3 value) noexcept {
    sample.trace_write_event(context.bounce.path_step, slot, value);
  };
  const auto path_state = [&] noexcept -> path::State {
    return {.flag = sample.path_flags, .visibility = sample.cycles_path_visibility,
            .bounce = sample.path_depth, .diffuse_bounce = sample.diffuse_depth,
            .glossy_bounce = sample.glossy_depth,
            .transmission_bounce = sample.transmission_depth,
            .transparent_bounce = sample.transparent_depth,
            .rng_offset = sample.cycles_rng_offset,
            .portal_bounce = has_portal ? sample.portal_depth : UInt{0u}};
  };
  const auto store_path = [&](const path::State &next) noexcept {
    sample.path_flags = next.flag;
    sample.cycles_path_visibility = next.visibility;
    sample.path_depth = next.bounce;
    sample.diffuse_depth = next.diffuse_bounce;
    sample.glossy_depth = next.glossy_bounce;
    sample.transmission_depth = next.transmission_bounce;
    sample.transparent_depth = next.transparent_bounce;
    sample.cycles_rng_offset = next.rng_offset;
    if (has_portal) sample.portal_depth = next.portal_bounce;
  };

  // integrate_surface_bsdf_bssrdf_bounce: reject before the random lookup or
  // closure[0] access. All selected indices belong to the initialized prefix.
  $if((sd.flag & unsigned(abi::SD_BSDF | abi::SD_BSSRDF)) != 0u) {
    const auto random = config.path_trace_enabled
        ? context.shading.bsdf_sample : sample.surface_bsdf_sample();
    const auto pick = native::surface_shader_bsdf_bssrdf_pick(sd, random);
    const auto sc = pool.common(pick.index);
    if (config.path_trace_enabled) {
      trace(path_trace_schema::EventSlot::closure_pick,
            make_float3(pick.index.cast<float>(),
                        sc.type.cast<float>(), sc.sample_weight));
      trace(path_trace_schema::EventSlot::closure_random, pick.random);
      trace(path_trace_schema::EventSlot::closure_weight, sc.weight);
      trace(path_trace_schema::EventSlot::closure_n, sc.N);
    }

    UInt label = closure::label_none;
    const auto ordinary_bsdf = [&] noexcept {
      const auto bsdf = native::surface_shader_bsdf_sample_closure(
          kg, sd, pick, population.closure_types);
      // Native rejection is exactly zero PDF or zero BsdfEval. Do not add a
      // selected-weight, positive-PDF, finite or nonnegative-throughput guard.
      $if((bsdf.evaluation.pdf != 0.0f) & any(bsdf.evaluation.sum != 0.0f)) {
        label = bsdf.label;
        if (config.path_trace_enabled) {
          trace(path_trace_schema::EventSlot::bsdf_meta,
                make_float3(bsdf.evaluation.pdf, bsdf.evaluation.pdf, label.cast<float>()));
          trace(path_trace_schema::EventSlot::bsdf_wo, bsdf.wo);
          trace(path_trace_schema::EventSlot::bsdf_eval, bsdf.evaluation.sum);
          trace(path_trace_schema::EventSlot::bsdf_roughness_eta,
                make_float3(bsdf.sampled_roughness, bsdf.eta));
        }
        const auto transparent = (label & closure::label_transparent) != 0u;
        $if(transparent) {
          // No direction normalization, geometry fetch/offset or differential
          // widening occurs on this edge of the original integrator.
          sample.ray->set_t_min(surface_ray::intersection_t_offset(sd.ray_length));
        }
        $else {
          const auto D = normalize(bsdf.wo);
          sample.ray = make_ray(context.surface.make_ray_origin(D), D, 0.0f, ray_maximum);
          sample.ray_dP = sd.dP;
          sample.ray_dD = cycles_ray_differential::widen_direction(
              sample.ray_dD, bsdf.evaluation.average_roughness_squared);
        };
        sample.throughput *= bsdf.evaluation.sum / bsdf.evaluation.pdf;
        // This renderer exposes the complete light-pass contract. Native
        // first-bounce weights include transparent closures and are reduced
        // only inside the first-bounce predicate.
        $if(sample.path_depth == 0u) {
          sample.path_diffuse_weight = config.light_transport.light_component_ratio(
              bsdf.evaluation.diffuse, bsdf.evaluation.sum);
          sample.path_glossy_weight = config.light_transport.light_component_ratio(
              bsdf.evaluation.glossy, bsdf.evaluation.sum);
        };
        $if(!transparent) {
          sample.previous_bsdf_pdf = bsdf.evaluation.pdf;
          sample.minimum_bsdf_pdf = min(bsdf.evaluation.pdf, sample.minimum_bsdf_pdf);
          sample.previous_mis_origin_normal = sd.N;
        };
        store_path(path::next_surface(path_state(), label,
            (sd.flag & unsigned(abi::SD_BSDF_HAS_TRANSMISSION)) != 0u,
            has_portal ? (sd.flag & unsigned(abi::SD_RAY_PORTAL)) != 0u : Bool{false}, limits));
        // Traversal has already replaced Cycles' isect with this hit. The
        // split source-identity representation commits that same identity here.
        sample.ray_source_object = context.surface.cycles_object_index;
        sample.ray_source_primitive = context.surface.cycles_primitive_index;
        if (config.path_trace_enabled) {
          trace(path_trace_schema::EventSlot::post_depth,
                make_float3(sample.path_depth.cast<float>(), sample.transparent_depth.cast<float>(),
                            sample.cycles_rng_offset.cast<float>()));
          trace(path_trace_schema::EventSlot::post_throughput, sample.throughput);
          trace(path_trace_schema::EventSlot::post_ray_p, sample.ray->origin());
          trace(path_trace_schema::EventSlot::post_ray_d, sample.ray->direction());
          trace(path_trace_schema::EventSlot::post_flags, sample.trace_uint32(sample.path_flags));
          trace(path_trace_schema::EventSlot::post_mis,
                make_float3(sample.previous_bsdf_pdf, sample.minimum_bsdf_pdf, sample.continuation_probability));
          trace(path_trace_schema::EventSlot::post_visibility, sample.trace_uint32(sample.cycles_path_visibility));
        }
      };
    };
    const auto portal_or_bsdf = [&] noexcept {
      // Host feature pruning happens before recording either portal branch or
      // its counter writes; non-portal scenes retain no live portal counter.
      if (has_portal) {
        $if(sc.type == closure::type_ray_portal) {
          native::RayPortalState state{
              .P = sample.ray->origin(), .D = sample.ray->direction(),
              .tmin = sample.ray->t_min(), .tmax = sample.ray->t_max(),
              .dP = sample.ray_dP, .throughput = sample.throughput,
              .isect_object = context.surface.cycles_object_index, .path = path_state()};
          label = native::integrate_surface_ray_portal(
              kg, sd, pick.index, context.surface.world_to_object, state, limits);
          $if(label != closure::label_none) {
            sample.ray = make_ray(state.P, state.D, state.tmin, state.tmax);
            sample.ray_dP = state.dP;
            sample.throughput = state.throughput;
            sample.ray_source_object = state.isect_object;
            sample.ray_source_primitive = context.surface.cycles_primitive_index;
            store_path(state.path);
          };
        }
        $else { ordinary_bsdf(); };
      } else {
        ordinary_bsdf();
      }
    };
    if ((features & svm::kernel_feature_subsurface) != 0u) {
      $if(closure::is_bssrdf(sc.type)) {
        const auto bssrdf = pool.bssrdf(pick.index);
        bssrdf_sample.bssrdf_method = static_cast<unsigned>(SurfaceBssrdfMethod::random_walk);
        $switch(sc.type) {
          $case(closure::type_bssrdf_burley) { bssrdf_sample.bssrdf_method = static_cast<unsigned>(SurfaceBssrdfMethod::burley); };
          $case(closure::type_bssrdf_random_walk_legacy) { bssrdf_sample.bssrdf_method = static_cast<unsigned>(SurfaceBssrdfMethod::random_walk_legacy); };
          $case(closure::type_bssrdf_random_walk_skin) { bssrdf_sample.bssrdf_method = static_cast<unsigned>(SurfaceBssrdfMethod::random_walk_skin); };
          $default {};
        };
        bssrdf_sample.bssrdf_radius = bssrdf.param.radius;
        bssrdf_sample.bssrdf_albedo = bssrdf.param.albedo;
        bssrdf_sample.bssrdf_normal = sc.N;
        bssrdf_sample.bssrdf_ior = bssrdf.param.ior;
        bssrdf_sample.bssrdf_roughness = bssrdf.param.alpha;
        bssrdf_sample.bssrdf_anisotropy = bssrdf.param.anisotropy;
        sample.throughput *= native::surface_shader_bssrdf_sample_weight(sd, pick.index);
        sample.cycles_rng_offset += path::bounce_dimension_count;
        sample.cycles_path_visibility &= ~path::visibility_camera;
        sample.ray_dP = sd.dP;
        subsurface = true;
        valid = true;
      }
      $else { portal_or_bsdf(); };
    } else {
      portal_or_bsdf();
    }
    $if(label != closure::label_none) {
      valid = true;
      if (config.volume_state) {
        config.volume_state->cross_surface(sample.volume, context.surface.volume_stack_entry,
            (sd.flag & unsigned(abi::SD_BACKFACING)) != 0u, context.surface.surface_has_volume, label);
      }
    };
  };
  return {.sample = std::move(bssrdf_sample), .subsurface = std::move(subsurface), .valid = std::move(valid)};
}
} // namespace psycles::luisa_backend::detail

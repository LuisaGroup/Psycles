/* SPDX-FileCopyrightText: 2011-2023 Blender Authors
 * SPDX-License-Identifier: Apache-2.0 */

#include "path_kernel_cycles_svm_surface_scatter.h"
#include "cycles_svm_surface_integrator.h"
#include "cycles_svm_surface_shader.h"

#include <cstdlib>
#include <string_view>

#include <luisa/dsl/sugar.h>

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_ray_differential.h>

namespace psycles::luisa_backend::detail {
namespace {

[[nodiscard]] bool should_outline_simd_bsdf(const PathKernelConfig &config) noexcept {
  if (!config.scene || !static_cast<bool>(config.scene->device) ||
      config.scene->device.backend_name() != "simd" ||
      config.scene->device.compute_warp_size() <= 1u) {
    return false;
  }
  const char *raw = std::getenv("PSYCLES_SIMD_OUTLINE_BSDF");
  return raw == nullptr || std::string_view{raw} != "0";
}

[[nodiscard]] Float3 spectral_component_ratio(
    Expr<luisa::float4> num, Expr<luisa::float4> den,
    Expr<bool> secondary_terminated) noexcept {
  const Float n = select((num.x + num.y + num.z + num.w) * 0.25f, num.x,
                         secondary_terminated);
  const Float d = select((den.x + den.y + den.z + den.w) * 0.25f, den.x,
                         secondary_terminated);
  return make_float3(select(0.0f, n / d, d != 0.0f));
}

} // namespace

namespace svm = cycles_svm;
namespace native = cycles_svm::detail;
namespace closure = cycles_closure;
namespace abi = compiler::cycles_svm;
namespace path = cycles_path_state;

void update_caustic_path_flags(PathSampleContext &sample,
                               const SurfaceGeometryContext &surface,
                               const PathKernelConfig &config,
                               Expr<std::uint32_t> prev_flags,
                               Expr<std::uint32_t> label) noexcept {
  $if((label & closure::label_transparent) == 0u) {
    const auto obj_flags =
        surface.cycles_object_flags
            ? *surface.cycles_object_flags
            : config.scene->cycles_svm->objects->object_flag_buffer->read(
                  surface.cycles_object_index);
    const auto on_receiver =
        (obj_flags &
         static_cast<std::uint32_t>(abi::SD_OBJECT_CAUSTICS_RECEIVER)) != 0u;
    const auto on_caster =
        (obj_flags &
         static_cast<std::uint32_t>(abi::SD_OBJECT_CAUSTICS_CASTER)) != 0u;
    const auto from_recv_or_chain =
        (prev_flags &
         (path::flag_caustic_receiver_parent |
          path::flag_caustic_caster_chain)) != 0u;
    const auto enter_or_keep_chain =
        on_caster & from_recv_or_chain &
        ((label & closure::label_transmit) != 0u);
    const auto enter_recv_parent =
        !enter_or_keep_chain & on_receiver &
        ((label & (closure::label_diffuse | closure::label_glossy)) != 0u) &
        ((label & closure::label_transmit) == 0u);
    sample.path_flags &=
        ~(path::flag_caustic_receiver_parent | path::flag_caustic_caster_chain);
    $if(enter_or_keep_chain) {
      sample.path_flags |= path::flag_caustic_caster_chain;
    }
    $elif(enter_recv_parent) {
      sample.path_flags |= path::flag_caustic_receiver_parent;
    };
  };
}

PreparedCyclesSvmSurfaceScatter
PreparedCyclesSvmSurfaceScatter::empty(bool spectral) noexcept {
  PreparedCyclesSvmSurfaceScatter out{
      .wo = make_float3(0.0f, 0.0f, 1.0f),
      .throughput_weight = cycles_spectrum::Spectrum::zero(spectral),
      .diffuse_ratio = make_float3(0.0f),
      .glossy_ratio = make_float3(0.0f),
      .pdf = 0.0f,
      .average_roughness_squared = 0.0f,
      .shading_normal = make_float3(0.0f, 0.0f, 1.0f),
      .dP = 0.0f,
      .ray_length = 0.0f,
      .label = closure::label_none,
      .sd_flag = 0u,
      .bssrdf_sample = SurfaceSample::zero(spectral),
      .subsurface = false,
      .valid = false};
  if (spectral) {
    out.spectral_terminated_secondary.emplace(false);
  }
  return out;
}

Bool PreparedCyclesSvmSurfaceScatter::needs_triangle_ray_origin()
    const noexcept {
  return valid & (!subsurface) &
         ((label & closure::label_transparent) == 0u);
}

void PreparedCyclesSvmSurfaceScatter::accept(
    const PreparedCyclesSvmSurfaceScatter &other,
    bool include_bssrdf) noexcept {
  wo = other.wo;
  throughput_weight = other.throughput_weight;
  diffuse_ratio = other.diffuse_ratio;
  glossy_ratio = other.glossy_ratio;
  pdf = other.pdf;
  average_roughness_squared = other.average_roughness_squared;
  shading_normal = other.shading_normal;
  dP = other.dP;
  ray_length = other.ray_length;
  label = other.label;
  sd_flag = other.sd_flag;
  if (include_bssrdf) {
    bssrdf_sample = other.bssrdf_sample;
  }
  subsurface = other.subsurface;
  valid = other.valid;
  if (spectral_terminated_secondary.has_value() &&
      other.spectral_terminated_secondary.has_value()) {
    *spectral_terminated_secondary = *other.spectral_terminated_secondary;
  }
}

PreparedCyclesSvmSurfaceScatter sample_cycles_svm_surface_continuation(
    DirectLightingContext &context, const CyclesSvmSurfaceState &population,
    bool may_have_bssrdf) noexcept {
  auto &sample = context.bounce.sample;
  const auto &config = sample.invocation.config;
  auto &sd = population.shader_data;
  const auto &kg = population.kernel_globals;
  const auto &pool = *sd.closure;
  const auto features = config.scene->cycles_svm->kernel_features;
  const bool bssrdf_enabled =
      may_have_bssrdf && ((features & svm::kernel_feature_subsurface) != 0u);
  const bool spectral = sample.is_spectral();

  auto out = PreparedCyclesSvmSurfaceScatter::empty(spectral);
  out.shading_normal = sd.N;
  out.dP = sd.dP;
  out.ray_length = sd.ray_length;
  out.sd_flag = sd.flag;

  const std::uint32_t scatter_mask =
      bssrdf_enabled ? unsigned(abi::SD_BSDF | abi::SD_BSSRDF)
                     : unsigned(abi::SD_BSDF);
  $if((sd.flag & scatter_mask) != 0u) {
    const auto random = sample.surface_bsdf_sample();
    const auto pick = native::surface_shader_bsdf_bssrdf_pick(sd, random);
    const auto sc = pool.common(pick.index);

    const bool outline_bsdf = should_outline_simd_bsdf(config);
    const auto sample_ordinary_bsdf = [&] noexcept {
      $outline_when_with_name(outline_bsdf, "simd_svm_bsdf_sample") {
        const auto bsdf = native::surface_shader_bsdf_sample_closure(
            kg, sd, pick, population.closure_types);
        const auto non_zero_eval =
            spectral ? any(*bsdf.evaluation.spectral_sum != 0.0f)
                     : any(bsdf.evaluation.sum != 0.0f);
        $if((bsdf.evaluation.pdf != 0.0f) & non_zero_eval &
            (bsdf.label != closure::label_none)) {
          out.wo = bsdf.wo;
          if (spectral) {
            out.throughput_weight = cycles_spectrum::Spectrum::from_lanes(
                *bsdf.evaluation.spectral_sum / bsdf.evaluation.pdf);
            *out.spectral_terminated_secondary =
                bsdf.spectral_terminated_secondary.value_or(Bool{false});
          } else {
            out.throughput_weight = cycles_spectrum::Spectrum::from_rgb_raw(
                bsdf.evaluation.sum / bsdf.evaluation.pdf);
          }
          out.pdf = bsdf.evaluation.pdf;
          out.average_roughness_squared =
              bsdf.evaluation.average_roughness_squared;
          out.label = bsdf.label;
          $if(sample.path_depth == 0u) {
            if (spectral) {
              const auto eff_term =
                  sample.spectral_secondary_terminated() |
                  bsdf.spectral_terminated_secondary.value_or(Bool{false});
              out.diffuse_ratio = spectral_component_ratio(
                  *bsdf.evaluation.spectral_diffuse,
                  *bsdf.evaluation.spectral_sum, eff_term);
              out.glossy_ratio = spectral_component_ratio(
                  *bsdf.evaluation.spectral_glossy,
                  *bsdf.evaluation.spectral_sum, eff_term);
            } else {
              out.diffuse_ratio = config.light_transport.light_component_ratio(
                  bsdf.evaluation.diffuse, bsdf.evaluation.sum);
              out.glossy_ratio = config.light_transport.light_component_ratio(
                  bsdf.evaluation.glossy, bsdf.evaluation.sum);
            }
          };
          out.valid = true;
        };
      };
    };

    if (bssrdf_enabled) {
      $if(closure::is_bssrdf(sc.type)) {
        const auto bssrdf = pool.bssrdf(pick.index);
        out.bssrdf_sample.bssrdf_method =
            static_cast<unsigned>(SurfaceBssrdfMethod::random_walk);
        $switch(sc.type) {
          $case(closure::type_bssrdf_burley) {
            out.bssrdf_sample.bssrdf_method =
                static_cast<unsigned>(SurfaceBssrdfMethod::burley);
          };
          $case(closure::type_bssrdf_random_walk_legacy) {
            out.bssrdf_sample.bssrdf_method =
                static_cast<unsigned>(SurfaceBssrdfMethod::random_walk_legacy);
          };
          $case(closure::type_bssrdf_random_walk_skin) {
            out.bssrdf_sample.bssrdf_method =
                static_cast<unsigned>(SurfaceBssrdfMethod::random_walk_skin);
          };
          $default {};
        };
        out.bssrdf_sample.bssrdf_radius = bssrdf.param.radius;
        out.bssrdf_sample.bssrdf_albedo = bssrdf.param.albedo;
        out.bssrdf_sample.bssrdf_normal = sc.N;
        out.bssrdf_sample.bssrdf_ior = bssrdf.param.ior;
        out.bssrdf_sample.bssrdf_roughness = bssrdf.param.alpha;
        out.bssrdf_sample.bssrdf_anisotropy = bssrdf.param.anisotropy;
        const auto bssrdf_w =
            native::surface_shader_bssrdf_sample_weight(sd, pick.index);
        if (spectral) {
          const auto ctx = sample.spectral_context();
          out.throughput_weight =
              cycles_spectrum::Spectrum::from_rgb_reflectance(
                  ctx->tables, bssrdf_w, &ctx->wavelengths, true);
        } else {
          out.throughput_weight =
              cycles_spectrum::Spectrum::from_rgb_raw(bssrdf_w);
        }
        out.subsurface = true;
        out.valid = true;
      }
      $else { sample_ordinary_bsdf(); };
    } else {
      sample_ordinary_bsdf();
    }
  };
  return out;
}

SurfaceScatterStage::Result commit_cycles_svm_surface_continuation(
    DirectLightingContext &context,
    const PreparedCyclesSvmSurfaceScatter &prepared,
    bool may_have_bssrdf) noexcept {
  auto &sample = context.bounce.sample;
  const auto &config = sample.invocation.config;
  const auto &parameters = sample.invocation.parameters;
  const auto features = config.scene->cycles_svm->kernel_features;
  const bool bssrdf_enabled =
      may_have_bssrdf && ((features & svm::kernel_feature_subsurface) != 0u);
  const path::Limits limits{
      .maximum = parameters.max_bounces,
      .maximum_diffuse = parameters.max_diffuse_bounces,
      .maximum_glossy = parameters.max_glossy_bounces,
      .maximum_transmission = parameters.max_transmission_bounces,
      .maximum_transparent = parameters.transparent_max_bounces};
  const auto path_state = [&] noexcept -> path::State {
    return {.flag = sample.path_flags,
            .visibility = sample.cycles_path_visibility,
            .bounce = sample.path_depth,
            .diffuse_bounce = sample.diffuse_depth,
            .glossy_bounce = sample.glossy_depth,
            .transmission_bounce = sample.transmission_depth,
            .transparent_bounce = sample.transparent_depth,
            .rng_offset = sample.cycles_rng_offset,
            .portal_bounce = UInt{0u}};
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
  };

  const auto commit_bsdf = [&] noexcept {
    const auto transparent = (prepared.label & closure::label_transparent) != 0u;
    $if(transparent) {
      sample.ray->set_t_min(surface_ray::intersection_t_offset(
          max(sample.ray->t_min(), prepared.ray_length)));
    }
    $else {
      const auto D = normalize(prepared.wo);
      sample.ray = make_ray(context.surface.make_ray_origin(D), D, 0.0f,
                            ray_maximum);
      sample.ray_dP = prepared.dP;
      sample.ray_dD = cycles_ray_differential::widen_direction(
          sample.ray_dD, prepared.average_roughness_squared);
    };
    sample.throughput *= prepared.throughput_weight;
    $if(sample.path_depth == 0u) {
      sample.path_diffuse_weight = prepared.diffuse_ratio;
      sample.path_glossy_weight = prepared.glossy_ratio;
    };
    $if(!transparent) {
      sample.previous_bsdf_pdf = prepared.pdf;
      if (config.has_filter_glossy) {
        sample.minimum_bsdf_pdf = min(prepared.pdf, sample.minimum_bsdf_pdf);
      }
      sample.previous_mis_origin_normal = prepared.shading_normal;
    };
    UInt prev_caustic_flags = 0u;
    if (config.caustics_active()) {
      prev_caustic_flags = sample.path_flags;
    }
    store_path(path::next_surface(
        path_state(), prepared.label,
        (prepared.sd_flag & unsigned(abi::SD_BSDF_HAS_TRANSMISSION)) != 0u,
        Bool{false}, limits));
    if (config.caustics_active()) {
      update_caustic_path_flags(sample, context.surface, config,
                                prev_caustic_flags, prepared.label);
    }
    if (sample.is_spectral() &&
        prepared.spectral_terminated_secondary.has_value()) {
      $if(*prepared.spectral_terminated_secondary) {
        sample.path_flags |=
            cycles_path_state::flag_spectral_secondary_terminated;
        sample.throughput.terminate_secondary();
      };
    }
    sample.ray_source_object = context.surface.cycles_object_index;
    sample.ray_source_primitive = context.surface.cycles_primitive_index;
    if (config.volume_state) {
      config.volume_state->cross_surface(
          sample.volume, context.surface.volume_stack_entry,
          (prepared.sd_flag & unsigned(abi::SD_BACKFACING)) != 0u,
          context.surface.surface_has_volume, prepared.label);
    }
  };

  $if(prepared.valid) {
    if (bssrdf_enabled) {
      $if(prepared.subsurface) {
        sample.throughput *= prepared.throughput_weight;
        sample.cycles_rng_offset += path::bounce_dimension_count;
        sample.cycles_path_visibility &= ~path::visibility_camera;
        sample.ray_dP = prepared.dP;
      }
      $else { commit_bsdf(); };
    } else {
      commit_bsdf();
    }
  };

  return {.sample = prepared.bssrdf_sample,
          .subsurface = bssrdf_enabled ? prepared.subsurface : Bool{false},
          .valid = prepared.valid};
}

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
  auto bssrdf_sample = SurfaceSample::zero(sample.is_spectral());
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
    const bool outline_bsdf = should_outline_simd_bsdf(config);
  const bool spectral = sample.is_spectral();
  const auto ordinary_bsdf = [&] noexcept {
      Float3 bsdf_wo = make_float3(0.0f);
      Float3 bsdf_sum = make_float3(0.0f);
      Float3 bsdf_diffuse = make_float3(0.0f);
      Float3 bsdf_glossy = make_float3(0.0f);
      std::optional<Float4> bsdf_spec_sum;
      std::optional<Float4> bsdf_spec_diffuse;
      std::optional<Float4> bsdf_spec_glossy;
      std::optional<Bool> bsdf_term_sec;
      if (spectral) {
        bsdf_spec_sum.emplace(make_float4(0.0f));
        bsdf_spec_diffuse.emplace(make_float4(0.0f));
        bsdf_spec_glossy.emplace(make_float4(0.0f));
        bsdf_term_sec.emplace(false);
      }
      Float bsdf_pdf = 0.0f;
      Float bsdf_avg_r2 = 0.0f;
      Float2 bsdf_sampled_roughness = make_float2(0.0f);
      Float bsdf_eta = 0.0f;
      UInt bsdf_label = closure::label_none;
      $outline_when_with_name(outline_bsdf, "simd_svm_bsdf_sample") {
        const auto bsdf = native::surface_shader_bsdf_sample_closure(
            kg, sd, pick, population.closure_types);
        bsdf_wo = bsdf.wo;
        bsdf_sum = bsdf.evaluation.sum;
        bsdf_diffuse = bsdf.evaluation.diffuse;
        bsdf_glossy = bsdf.evaluation.glossy;
        if (spectral) {
          *bsdf_spec_sum = *bsdf.evaluation.spectral_sum;
          *bsdf_spec_diffuse = *bsdf.evaluation.spectral_diffuse;
          *bsdf_spec_glossy = *bsdf.evaluation.spectral_glossy;
          *bsdf_term_sec =
              bsdf.spectral_terminated_secondary.value_or(Bool{false});
        }
        bsdf_pdf = bsdf.evaluation.pdf;
        bsdf_avg_r2 = bsdf.evaluation.average_roughness_squared;
        bsdf_sampled_roughness = bsdf.sampled_roughness;
        bsdf_eta = bsdf.eta;
        bsdf_label = bsdf.label;
      };
      // Native rejection is exactly zero PDF or zero BsdfEval. Do not add a
      // selected-weight, positive-PDF, finite or nonnegative-throughput guard.
      const auto non_zero_eval =
          spectral ? any(*bsdf_spec_sum != 0.0f) : any(bsdf_sum != 0.0f);
      $if((bsdf_pdf != 0.0f) & non_zero_eval) {
        label = bsdf_label;
        if (config.path_trace_enabled) {
          trace(path_trace_schema::EventSlot::bsdf_meta,
                make_float3(bsdf_pdf, bsdf_pdf, label.cast<float>()));
          trace(path_trace_schema::EventSlot::bsdf_wo, bsdf_wo);
          trace(path_trace_schema::EventSlot::bsdf_eval,
                spectral ? sample.reflectance_to_rgb(
                               cycles_spectrum::Spectrum::from_lanes(
                                   *bsdf_spec_sum))
                         : bsdf_sum);
          trace(path_trace_schema::EventSlot::bsdf_roughness_eta,
                make_float3(bsdf_sampled_roughness, bsdf_eta));
        }
        const auto transparent = (label & closure::label_transparent) != 0u;
        $if(transparent) {
          // No direction normalization, geometry fetch/offset or differential
          // widening occurs on this edge of the original integrator.
          sample.ray->set_t_min(surface_ray::intersection_t_offset(
              max(sample.ray->t_min(), sd.ray_length)));
        }
        $else {
          const auto D = normalize(bsdf_wo);
          sample.ray = make_ray(context.surface.make_ray_origin(D), D, 0.0f, ray_maximum);
          sample.ray_dP = sd.dP;
          sample.ray_dD = cycles_ray_differential::widen_direction(
              sample.ray_dD, bsdf_avg_r2);
        };
        if (spectral) {
          sample.throughput *= cycles_spectrum::Spectrum::from_lanes(
              *bsdf_spec_sum / bsdf_pdf);
        } else {
          sample.throughput.rgb() *= bsdf_sum / bsdf_pdf;
        }
        // This renderer exposes the complete light-pass contract. Native
        // first-bounce weights include transparent closures and are reduced
        // only inside the first-bounce predicate.
        $if(sample.path_depth == 0u) {
          if (spectral) {
            const auto eff_term =
                sample.spectral_secondary_terminated() | *bsdf_term_sec;
            sample.path_diffuse_weight = spectral_component_ratio(
                *bsdf_spec_diffuse, *bsdf_spec_sum, eff_term);
            sample.path_glossy_weight = spectral_component_ratio(
                *bsdf_spec_glossy, *bsdf_spec_sum, eff_term);
          } else {
            sample.path_diffuse_weight = config.light_transport.light_component_ratio(
                bsdf_diffuse, bsdf_sum);
            sample.path_glossy_weight = config.light_transport.light_component_ratio(
                bsdf_glossy, bsdf_sum);
          }
        };
        $if(!transparent) {
          sample.previous_bsdf_pdf = bsdf_pdf;
          if (config.has_filter_glossy || config.path_trace_enabled) {
            sample.minimum_bsdf_pdf = min(bsdf_pdf, sample.minimum_bsdf_pdf);
          }
          sample.previous_mis_origin_normal = sd.N;
        };
        UInt prev_caustic_flags = 0u;
        if (config.caustics_active()) {
          prev_caustic_flags = sample.path_flags;
        }
        store_path(path::next_surface(path_state(), label,
            (sd.flag & unsigned(abi::SD_BSDF_HAS_TRANSMISSION)) != 0u,
            has_portal ? (sd.flag & unsigned(abi::SD_RAY_PORTAL)) != 0u : Bool{false}, limits));
        if (config.caustics_active()) {
          update_caustic_path_flags(sample, context.surface, config,
                                    prev_caustic_flags, label);
        }
        if (spectral) {
          $if(*bsdf_term_sec) {
            sample.path_flags |=
                cycles_path_state::flag_spectral_secondary_terminated;
            sample.throughput.terminate_secondary();
          };
        }
        // Traversal has already replaced Cycles' isect with this hit. The
        // split source-identity representation commits that same identity here.
        sample.ray_source_object = context.surface.cycles_object_index;
        sample.ray_source_primitive = context.surface.cycles_primitive_index;
        if (config.path_trace_enabled) {
          trace(path_trace_schema::EventSlot::post_depth,
                make_float3(sample.path_depth.cast<float>(), sample.transparent_depth.cast<float>(),
                            sample.cycles_rng_offset.cast<float>()));
          trace(path_trace_schema::EventSlot::post_throughput,
                spectral ? sample.reflectance_to_rgb(sample.throughput)
                         : sample.throughput.rgb());
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
              .dP = sample.ray_dP,
              .throughput =
                  spectral ? make_float3(1.0f) : sample.throughput.rgb(),
              .isect_object = context.surface.cycles_object_index,
              .path = path_state()};
          label = native::integrate_surface_ray_portal(
              kg, sd, pick.index, context.surface.fetch_world_to_object(), state, limits);
          $if(label != closure::label_none) {
            sample.ray = make_ray(state.P, state.D, state.tmin, state.tmax);
            sample.ray_dP = state.dP;
            if (spectral) {
              const auto ctx = sample.spectral_context();
              sample.throughput *=
                  cycles_spectrum::Spectrum::from_rgb_reflectance(
                      ctx->tables, state.throughput, &ctx->wavelengths, true);
            } else {
              sample.throughput =
                  cycles_spectrum::Spectrum::from_rgb_raw(state.throughput);
            }
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
        const auto bssrdf_w =
            native::surface_shader_bssrdf_sample_weight(sd, pick.index);
        if (spectral) {
          const auto ctx = sample.spectral_context();
          sample.throughput *= cycles_spectrum::Spectrum::from_rgb_reflectance(
              ctx->tables, bssrdf_w, &ctx->wavelengths, true);
        } else {
          sample.throughput *= cycles_spectrum::Spectrum::from_rgb_raw(bssrdf_w);
        }
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

namespace {

class SurfaceScatterStageImpl final : public SurfaceScatterStage {

public:
  Result emit(DirectLightingContext &context) const noexcept override {
    LUISA_ASSERT(
        context.shading.populated_surface != nullptr,
        "Native continuation requires the retained surface population.");
    const auto native =
        context.shading.populated_surface->native_surface_state();
    LUISA_ASSERT(native.has_value(), "Missing native ShaderData consumer.");
    return emit_cycles_svm_surface_scatter(context, *native);
  }
};

} // namespace

std::unique_ptr<SurfaceScatterStage> make_surface_scatter_stage() {
  return std::make_unique<SurfaceScatterStageImpl>();
}

} // namespace psycles::luisa_backend::detail

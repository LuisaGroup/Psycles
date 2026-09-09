#include "path_kernel_surface_emission.h"

namespace psycles::luisa_backend::detail {

void emit_surface_emission(
    SurfaceGeometryContext &surface, const SurfacePreparation &preparation,
    const EmissiveTriangleComponent &emissive_triangle) noexcept {
  auto &bounce = surface.bounce;
  auto &sample = bounce.sample;
  auto &invocation = sample.invocation;
  const auto &config = invocation.config;
  const auto &scene = config.scene;
  auto &point = surface.point;
  auto &ray = sample.ray;
  auto &hit_position = surface.hit_position;
  auto &wp0 = surface.wp0;
  auto &wp1 = surface.wp1;
  auto &wp2 = surface.wp2;
  auto &cycles_object_index = surface.cycles_object_index;
  auto &cycles_primitive_index = surface.cycles_primitive_index;
  auto &path_step = bounce.path_step;
  auto &throughput = sample.throughput;
  auto &path_depth = sample.path_depth;
  const auto mis_competition_skipped = sample.mis_competition_skipped();
  auto &previous_bsdf_pdf = sample.previous_bsdf_pdf;
  auto &previous_mis_origin_normal = sample.previous_mis_origin_normal;
  auto &previous_light_tree_dt = sample.previous_light_tree_dt;
  auto &cycles_path_visibility = sample.cycles_path_visibility;
  auto &path_flags = sample.path_flags;
  const auto &forward_light_weight =
      config.light_transport.forward_light_weight;
  const auto next_event_estimation = config.next_event_estimation;
  auto clamp_contribution = [&](Float3 contribution, UInt depth) noexcept {
    return invocation.clamp_emission_contribution(contribution, depth);
  };

  Float3 emitted = make_float3(0.0f);
  Float3 emission_contribution = make_float3(0.0f);
  Float emission_weight = 1.0f;
  Float forward_selection_pdf = 0.0f;
  Float forward_light_pdf = 0.0f;
  Bool forward_pdf_valid = false;
  // Cycles integrate_surface skips the complete operation at a BSSRDF exit
  // and without SD_EMISSION. Selecting a zero RGB would still perform MIS
  // work and atomic film writes. Flag-set zero emission remains eligible.
  $if((!bounce.subsurface_exit) &
      ((preparation.runtime_flags & cycles_closure::runtime_emission) != 0u)) {
    emitted = preparation.emission;
    if (next_event_estimation && scene->emissive_triangle_count > 0u) {
      // The committed primitive already carries the effective authored
      // sampling policy. NONE covers non-emissive materials, curves,
      // and instances which are not part of the sampled-light
      // population, so forward-hit MIS is O(1) exactly as in Cycles'
      // legacy light distribution.
      $if((!surface.is_curve) &
          (surface.emission_sampling !=
           static_cast<std::uint32_t>(contract::EmissionSampling::none))) {
        surface.ensure_world_triangle_vertices();
        Bool competing = (path_depth > 0u) & (!mis_competition_skipped);
        const auto oriented_geometric_normal = select(
            point.geometric_normal, -point.geometric_normal, point.back_facing);
        forward_selection_pdf = 0.5f * length(cross(wp1 - wp0, wp2 - wp0)) *
                                scene->triangle_area_pdf;
        if (config.use_light_tree) {
          const auto emitter_id = config.light_tree.triangle_emitter(
              cycles_object_index, cycles_primitive_index);
          forward_selection_pdf = config.light_tree.forward_pdf(
              emitter_id, ray->origin(), previous_mis_origin_normal,
              previous_light_tree_dt, cycles_path_visibility, path_flags);
        }
        const auto light_pdf = emissive_triangle.from_intersection(
            forward_selection_pdf, surface.emission_sampling, ray->origin(),
            hit_position, wp0, wp1, wp2, oriented_geometric_normal);
        forward_light_pdf = light_pdf.value;
        forward_pdf_valid = light_pdf.valid;
        emission_weight = forward_light_weight(
            previous_bsdf_pdf, forward_light_pdf, competing, forward_pdf_valid);
      };
    }
    emission_contribution =
        clamp_contribution(throughput * emitted * emission_weight, path_depth);
    sample.accumulate_radiance(emission_contribution);
    sample.accumulate_emission_or_background(emission_contribution,
                                              LightPassBuffer::emission);
  };
  sample.trace_write_forward_event(
      path_step, path_trace_schema::ForwardEventSlot::forward_emission,
      emitted);
  sample.trace_write_forward_event(
      path_step, path_trace_schema::ForwardEventSlot::forward_policy,
      make_float3(cast<float>(surface.emission_sampling), forward_selection_pdf,
                  select(0.0f, 1.0f, forward_pdf_valid)));
  sample.trace_write_forward_event(
      path_step, path_trace_schema::ForwardEventSlot::forward_mis,
      make_float3(previous_bsdf_pdf, forward_light_pdf, emission_weight));
  sample.trace_write_forward_event(
      path_step, path_trace_schema::ForwardEventSlot::forward_contribution,
      emission_contribution);
}

} // namespace psycles::luisa_backend::detail

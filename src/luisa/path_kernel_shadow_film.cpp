#include "path_kernel_direct_light_task.h"
#include "path_kernel_builder.h"
#include "path_kernel_film.h"

namespace psycles::luisa_backend::detail {

void prepare_surface_shadow_pass(Var<DirectLightTaskCall> &task,
                                 const PathSampleContext &sample,
                                 const DirectLightTransportState &transport) noexcept {
  const auto &ratio = sample.invocation.config.light_transport.light_component_ratio;
  task.path_flags = sample.path_flags;
  // Native integrate_surface_direct_light classifies the first scattering
  // event by ANY_PASS, independently of the later direct/indirect depth test.
  $if((task.path_flags & cycles_path_state::flag_any_pass) != 0u) {
    task.diffuse_weight = sample.path_diffuse_weight;
    task.glossy_weight = sample.path_glossy_weight;
  }
  $else {
    task.path_flags |= cycles_path_state::flag_surface_pass;
    if (sample.is_hero4() && transport.spectral_bsdf.has_value()) {
      const auto diff_rgb = sample.reflectance_to_rgb(
          cycles_spectrum::Spectrum::from_lanes(
              transport.spectral_diffuse_bsdf.value_or(
                  make_float4(transport.diffuse_bsdf, 0.0f))));
      const auto gloss_rgb = sample.reflectance_to_rgb(
          cycles_spectrum::Spectrum::from_lanes(
              transport.spectral_glossy_bsdf.value_or(
                  make_float4(transport.glossy_bsdf, 0.0f))));
      const auto bsdf_rgb = sample.reflectance_to_rgb(
          cycles_spectrum::Spectrum::from_lanes(*transport.spectral_bsdf));
      task.diffuse_weight = ratio(diff_rgb, bsdf_rgb);
      task.glossy_weight = ratio(gloss_rgb, bsdf_rgb);
    } else if (sample.is_xyz3()) {
      const auto diff_rgb = sample.reflectance_to_rgb(
          cycles_spectrum::Spectrum::from_xyz3(transport.diffuse_bsdf));
      const auto gloss_rgb = sample.reflectance_to_rgb(
          cycles_spectrum::Spectrum::from_xyz3(transport.glossy_bsdf));
      const auto bsdf_rgb = sample.reflectance_to_rgb(
          cycles_spectrum::Spectrum::from_xyz3(transport.bsdf));
      task.diffuse_weight = ratio(diff_rgb, bsdf_rgb);
      task.glossy_weight = ratio(gloss_rgb, bsdf_rgb);
    } else {
      task.diffuse_weight = ratio(transport.diffuse_bsdf, transport.bsdf);
      task.glossy_weight = ratio(transport.glossy_bsdf, transport.bsdf);
    }
  };
}

void DirectLightTaskEvaluator::accumulate_passes(
    PathSampleContext &sample, const Var<DirectLightTaskCall> &task,
    Float3 contribution) const noexcept {
  sample.accumulate_scattered_light_at_state(contribution, task.path_flags,
                                              task.diffuse_weight, task.glossy_weight,
                                              task.path_depth == 0u);
}

void DirectLightTaskEvaluator::accumulate_passes_atomic(
    const BufferFloat4 &film, UInt base, const Var<DirectLightTaskCall> &task,
    Float3 contribution) const noexcept {
  atomic_accumulate_scattered_light_passes(film, base, contribution, task.path_flags,
                                           task.diffuse_weight, task.glossy_weight,
                                           task.path_depth == 0u);
}

void accumulate_volume_nee_film(PathSampleContext &sample,
                                Float3 contribution) noexcept {
  sample.accumulate_radiance(contribution, true);
  UInt shadow_flag = sample.path_flags;
  Float3 diffuse = sample.path_diffuse_weight;
  Float3 glossy = sample.path_glossy_weight;
  $if((shadow_flag & cycles_path_state::flag_any_pass) == 0u) {
    shadow_flag |= cycles_path_state::flag_volume_pass;
    diffuse = make_float3(1.0f);
    glossy = make_float3(0.0f);
  };
  // The independent Combined guiding operation above already applies the
  // native primary-volume shadow visibility/PRIMARY_TRANSMIT override.
  sample.accumulate_scattered_light_at_state(contribution, shadow_flag, diffuse, glossy,
                                              sample.path_depth == 0u);
}

} // namespace psycles::luisa_backend::detail

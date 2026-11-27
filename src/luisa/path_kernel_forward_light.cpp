#include "cycles_light_visibility.h"
#include "path_kernel_builder.h"
#include "virtual_texture_cache.h"

#include <psycles/luisa/surface_ray.h>

#include <utility>

namespace psycles::luisa_backend::detail {
namespace {

class ForwardLightStageImpl final : public ForwardLightStage {

public:
  Bool emit(ClosestPathEvent &event) const noexcept override {
    auto &bounce = event.bounce;
    auto &sample = bounce.sample;
    auto &invocation = sample.invocation;
    const auto &config = invocation.config;
    const auto &scene = config.scene;
    const auto &kernel_parameters = invocation.parameters;
    auto &ray = sample.ray;
    const auto mis_competition_skipped = sample.mis_competition_skipped();
    auto &previous_bsdf_pdf = sample.previous_bsdf_pdf;
    auto &previous_mis_origin_normal = sample.previous_mis_origin_normal;
    auto &previous_light_tree_dt = sample.previous_light_tree_dt;
    auto &throughput = sample.throughput;
    auto &path_depth = sample.path_depth;
    auto &path_flags = sample.path_flags;
    auto &transparent_depth = sample.transparent_depth;
    const auto &forward_light_weight =
        config.light_transport.forward_light_weight;

    Var<LightGpu> light = scene->light_buffer->read(event.light_index);
    const auto visible = cycles_light_shader_visible(
        light.cycles_shader_flags, sample.cycles_path_visibility, path_flags);
    auto evaluate_emission = (event.light_evaluation_factor != 0.0f) & visible;
    if (config.caustics_active()) {
      const auto suppressed_caustic =
          ((path_flags & cycles_path_state::flag_caustic_caster_chain) != 0u) &
          ((light.flags & light_flag_caustics) != 0u);
      evaluate_emission &= !suppressed_caustic;
    }
    const bool hero4 = sample.is_hero4();
    const bool xyz3 = sample.is_xyz3();
    Float3 light_radiance = make_float3(0.0f);
    std::optional<Float4> light_radiance_spec;
    if (hero4) {
      light_radiance_spec.emplace(make_float4(0.0f));
    }
    $if(evaluate_emission) {
      if (hero4) {
        const auto ctx = sample.spectral_context();
        const auto tint_spec = cycles_spectrum::rgb_to_spectrum_reflectance(
            *ctx->tables, light.color, ctx->wavelengths.lambdas);
        const auto shader_spec = sample.analytic_light_shader_spectrum(
            light, event.light_index, event.light_position, event.light_normal,
            event.light_uv, -ray->direction(), event.distance);
        *light_radiance_spec =
            tint_spec * (light.power * event.light_evaluation_factor) *
            shader_spec.lanes();
      } else if (xyz3) {
        const auto ctx = sample.spectral_context();
        const auto tint_xyz = cycles_spectrum::Spectrum::from_rgb_reflectance(
            ctx->tables, light.color, &*ctx);
        const auto shader_xyz = sample.analytic_light_shader_spectrum(
            light, event.light_index, event.light_position, event.light_normal,
            event.light_uv, -ray->direction(), event.distance);
        light_radiance = tint_xyz.xyz3() *
                         (light.power * event.light_evaluation_factor) *
                         shader_xyz.xyz3();
      } else {
        light_radiance =
            light.color * (light.power * event.light_evaluation_factor);
        light_radiance *= sample.analytic_light_shader(
            light, event.light_index, event.light_position, event.light_normal,
            event.light_uv, -ray->direction(), event.distance);
      }
    };
    const auto commit_hit = [&]() noexcept {
      // Cycles' committed intersection becomes the next ray.self identity.
      // Retaining the earlier surface would change self rejection and the
      // object-specific AO traversal limit after this transparent endpoint.
      sample.ray_source_object = light.cycles_object_index;
      sample.ray_source_primitive = event.light_index;
      // Intersection and emission eligibility are different state-machine
      // edges in Cycles. Advance every geometric lamp hit, including empty
      // spread/spot evaluations and indirect visibility exclusions.
      ray = make_ray(ray->origin(), ray->direction(),
                     surface_ray::intersection_t_offset(
                         max(ray->t_min(), event.distance)),
                     ray->t_max());
      const auto has_radiance =
          hero4 ? any(*light_radiance_spec != make_float4(0.0f))
                : any(light_radiance != make_float3(0.0f));
      $if(evaluate_emission & has_radiance) {
        Float mis_weight = 1.0f;
        const auto competing = (path_depth > 0u) & (!mis_competition_skipped);
        $if(competing) {
          Float selection_pdf = scene->light_selection_pdf;
          if (config.use_light_tree) {
            const auto emitter_id =
                scene->emissive_triangle_count + event.light_index;
            selection_pdf = config.light_tree.forward_pdf(
                emitter_id, ray->origin(), previous_mis_origin_normal,
                previous_light_tree_dt, sample.cycles_path_visibility,
                path_flags);
          }
          const auto light_pdf = event.light_pdf * selection_pdf;
          mis_weight = forward_light_weight(
              previous_bsdf_pdf, light_pdf, competing, light_pdf > 0.0f);
        };
        const auto contrib_rgb =
            hero4
                ? sample.spectrum_to_rgb(
                      throughput *
                      cycles_spectrum::Spectrum::from_lanes(*light_radiance_spec) *
                      mis_weight)
                : (xyz3 ? sample.spectrum_to_rgb(
                              throughput *
                              cycles_spectrum::Spectrum::from_xyz3(light_radiance) *
                              mis_weight)
                        : (throughput.rgb() * light_radiance * mis_weight));
        const auto contribution = invocation.clamp_emission_contribution(
            contrib_rgb, path_depth);
        sample.accumulate_radiance(contribution);
        sample.accumulate_emission_or_background(contribution,
                                                  LightPassBuffer::emission);
      };

      transparent_depth += 1u;
    };
    if (sample.vt_miss_page != nullptr) {
      $if(*sample.vt_miss_page == vt_invalid_page) {
        commit_hit();
      };
    } else {
      commit_hit();
    }
    return transparent_depth >= kernel_parameters.transparent_max_bounces;
  }
};

} // namespace

std::unique_ptr<ForwardLightStage> make_forward_light_stage() {
  return std::make_unique<ForwardLightStageImpl>();
}

} // namespace psycles::luisa_backend::detail

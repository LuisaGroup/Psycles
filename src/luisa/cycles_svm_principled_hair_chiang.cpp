/* SPDX-FileCopyrightText: 2018-2022 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_principled_hair_chiang.h"

#include "cycles_svm_internal.h"
#include "cycles_svm_principled_hair_math.h"
#include "surface_fresnel.h"

#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
namespace closure_type = ::psycles::luisa_backend::cycles_closure;
namespace hair_math = principled_hair_math;
namespace sample_mapping = ::psycles::luisa_backend::cycles_sample_mapping;
namespace surface_detail = ::psycles::luisa_backend::detail;

namespace {

inline constexpr std::uint32_t primary_reflection_lobe = 0u;
inline constexpr std::uint32_t transmission_lobe = 1u;
inline constexpr std::uint32_t secondary_reflection_lobe = 2u;
inline constexpr std::uint32_t residual_lobe = 3u;
inline constexpr std::uint32_t explicit_lobe_count = 3u;
inline constexpr std::uint32_t lobe_count = 4u;
inline constexpr std::uint32_t alpha_angle_count = 6u;

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

[[nodiscard]] Float cos_from_sin(Expr<float> sine) noexcept {
  return sqrt(max(1.0f - square(sine), 0.0f));
}

[[nodiscard]] Float sin_from_cos(Expr<float> cosine) noexcept {
  return sqrt(max(1.0f - square(cosine), 0.0f));
}

[[nodiscard]] Float safe_asin(Expr<float> value) noexcept {
  return asin(clamp(value, -1.0f, 1.0f));
}

[[nodiscard]] Float safe_divide(Expr<float> numerator,
                                Expr<float> denominator) noexcept {
  return select(0.0f, numerator / denominator, denominator != 0.0f);
}

[[nodiscard]] Float3 safe_divide(Expr<luisa::float3> numerator,
                                 Expr<luisa::float3> denominator) noexcept {
  return make_float3(safe_divide(numerator.x, denominator.x),
                     safe_divide(numerator.y, denominator.y),
                     safe_divide(numerator.z, denominator.z));
}

[[nodiscard]] Float3 to_local(Expr<luisa::float3> value, Expr<luisa::float3> X,
                              Expr<luisa::float3> Y,
                              Expr<luisa::float3> Z) noexcept {
  return make_float3(dot(value, X), dot(value, Y), dot(value, Z));
}

[[nodiscard]] Float3 to_global(Expr<luisa::float3> value, Expr<luisa::float3> X,
                               Expr<luisa::float3> Y,
                               Expr<luisa::float3> Z) noexcept {
  return value.x * X + value.y * Y + value.z * Z;
}

[[nodiscard]] Float delta_phi(Expr<std::uint32_t> p, Expr<float> gamma_o,
                              Expr<float> gamma_t) noexcept {
  const auto order = p.cast<float>();
  return 2.0f * order * gamma_t - 2.0f * gamma_o + order * sample_mapping::pi;
}

[[nodiscard]] Float wrap_angle(Expr<float> angle) noexcept {
  constexpr auto two_pi = 2.0f * sample_mapping::pi;
  return (angle + sample_mapping::pi) -
         two_pi * floor((angle + sample_mapping::pi) / two_pi) -
         sample_mapping::pi;
}

[[nodiscard]] Float logistic(Expr<float> x, Expr<float> scale) noexcept {
  const auto value = exp(-abs(x) / scale);
  return value / (scale * square(1.0f + value));
}

[[nodiscard]] Float logistic_cdf(Expr<float> x, Expr<float> scale) noexcept {
  const auto argument = -x / scale;
  Float result;
  $if(argument > 88.0f) { result = 0.0f; }
  $else { result = 1.0f / (1.0f + exp(argument)); };
  return result;
}

[[nodiscard]] Float trimmed_logistic(Expr<float> x,
                                     Expr<float> scale) noexcept {
  const auto normalization =
      1.0f - 2.0f * logistic_cdf(-sample_mapping::pi, scale);
  return safe_divide(logistic(x, scale), normalization);
}

[[nodiscard]] Float sample_trimmed_logistic(Expr<float> random,
                                            Expr<float> scale) noexcept {
  const auto cdf_minus_pi = logistic_cdf(-sample_mapping::pi, scale);
  const auto x =
      -scale *
      log(1.0f / (random * (1.0f - 2.0f * cdf_minus_pi) + cdf_minus_pi) - 1.0f);
  return clamp(x, -sample_mapping::pi, sample_mapping::pi);
}

[[nodiscard]] Float azimuthal_scattering(Expr<float> phi, Expr<std::uint32_t> p,
                                         Expr<float> scale, Expr<float> gamma_o,
                                         Expr<float> gamma_t) noexcept {
  return trimmed_logistic(wrap_angle(phi - delta_phi(p, gamma_o, gamma_t)),
                          scale);
}

void hair_attenuation(const KernelGlobals &kernel_globals, Expr<float> fresnel,
                      Expr<luisa::float3> transmittance,
                      ArrayFloat3<lobe_count> &attenuation,
                      ArrayFloat<lobe_count> &energy) noexcept {
  attenuation[primary_reflection_lobe] = make_float3(fresnel);
  energy[primary_reflection_lobe] = fresnel;

  Float3 color = square(1.0f - fresnel) * transmittance;
  attenuation[transmission_lobe] = color;
  energy[transmission_lobe] = dot(color, kernel_globals.film_rgb_to_y());

  color *= transmittance * fresnel;
  attenuation[secondary_reflection_lobe] = color;
  energy[secondary_reflection_lobe] =
      dot(color, kernel_globals.film_rgb_to_y());

  color *= safe_divide(transmittance * fresnel,
                       make_float3(1.0f) - transmittance * fresnel);
  attenuation[residual_lobe] = color;
  energy[residual_lobe] = dot(color, kernel_globals.film_rgb_to_y());

  const auto total = energy[primary_reflection_lobe] +
                     energy[transmission_lobe] +
                     energy[secondary_reflection_lobe] + energy[residual_lobe];
  const auto normalization = safe_divide(1.0f, total);
  energy[primary_reflection_lobe] *= normalization;
  energy[transmission_lobe] *= normalization;
  energy[secondary_reflection_lobe] *= normalization;
  energy[residual_lobe] *= normalization;
}

void hair_alpha_angles(Expr<float> sin_theta_o, Expr<float> cos_theta_o,
                       Expr<float> alpha,
                       ArrayFloat<alpha_angle_count> &angles) noexcept {
  const auto sin_1alpha = sin(alpha);
  const auto cos_1alpha = cos_from_sin(sin_1alpha);
  const auto sin_2alpha = 2.0f * sin_1alpha * cos_1alpha;
  const auto cos_2alpha = square(cos_1alpha) - square(sin_1alpha);
  const auto sin_4alpha = 2.0f * sin_2alpha * cos_2alpha;
  const auto cos_4alpha = square(cos_2alpha) - square(sin_2alpha);

  angles[0u] = sin_theta_o * cos_2alpha - cos_theta_o * sin_2alpha;
  angles[1u] = abs(cos_theta_o * cos_2alpha + sin_theta_o * sin_2alpha);
  angles[2u] = sin_theta_o * cos_1alpha + cos_theta_o * sin_1alpha;
  angles[3u] = abs(cos_theta_o * cos_1alpha - sin_theta_o * sin_1alpha);
  angles[4u] = sin_theta_o * cos_4alpha + cos_theta_o * sin_4alpha;
  angles[5u] = abs(cos_theta_o * cos_4alpha - sin_theta_o * sin_4alpha);
}

[[nodiscard]] Float lobe_variance(const ChiangHairClosure &closure,
                                  Expr<std::uint32_t> lobe) noexcept {
  Float variance = 4.0f * closure.param.v;
  $if(lobe == primary_reflection_lobe) {
    variance = closure.param.m0_roughness;
  }
  $elif(lobe == transmission_lobe) { variance = 0.25f * closure.param.v; };
  return variance;
}

[[nodiscard]] Float sampling_variance(const ChiangHairClosure &closure,
                                      Expr<std::uint32_t> lobe) noexcept {
  Float variance = closure.param.v;
  $if(lobe == transmission_lobe) { variance *= 0.25f; };
  $if(lobe >= secondary_reflection_lobe) { variance *= 4.0f; };
  return variance;
}

struct HairGeometry {
  Float3 X;
  Float3 Y;
  Float3 Z;
  Float3 local_O;
  Float sin_theta_o;
  Float cos_theta_o;
  Float phi_o;
  Float gamma_o;
  Float gamma_t;
  Float3 transmittance;
};

[[nodiscard]] HairGeometry
hair_geometry(const ChiangHairClosure &closure,
              const ShaderData &shader_data) noexcept {
  const auto Y = closure.common.N;
  const auto X = safe_normalize_cycles(shader_data.dPdu);
  const auto Z = safe_normalize_cycles(cross(X, Y));
  const auto local_O = to_local(shader_data.wi, X, Y, Z);
  const auto sin_theta_o = local_O.x;
  const auto cos_theta_o = cos_from_sin(sin_theta_o);
  const auto phi_o = atan2(local_O.z, local_O.y);
  const auto sin_theta_t = sin_theta_o / closure.param.eta;
  const auto cos_theta_t = cos_from_sin(sin_theta_t);
  const auto sin_gamma_o = closure.param.h;
  const auto gamma_o = safe_asin(sin_gamma_o);
  const auto sin_gamma_t =
      sin_gamma_o * cos_theta_o /
      sqrt(square(closure.param.eta) - square(sin_theta_o));
  const auto cos_gamma_t = cos_from_sin(sin_gamma_t);
  const auto gamma_t = safe_asin(sin_gamma_t);
  const auto transmittance =
      exp(-closure.param.sigma * (2.0f * cos_gamma_t / cos_theta_t));
  return {.X = X,
          .Y = Y,
          .Z = Z,
          .local_O = local_O,
          .sin_theta_o = sin_theta_o,
          .cos_theta_o = cos_theta_o,
          .phi_o = phi_o,
          .gamma_o = gamma_o,
          .gamma_t = gamma_t,
          .transmittance = transmittance};
}

void evaluate_scattering(const ChiangHairClosure &closure,
                         const HairGeometry &geometry,
                         const ArrayFloat3<lobe_count> &attenuation,
                         const ArrayFloat<lobe_count> &energy,
                         const ArrayFloat<alpha_angle_count> &angles,
                         Expr<float> sin_theta_i, Expr<float> cos_theta_i,
                         Expr<float> phi, Float3 &value, Float &pdf) noexcept {
  value = make_float3(0.0f);
  pdf = 0.0f;
  $for(lobe, explicit_lobe_count) {
    const auto longitudinal = hair_math::longitudinal_scattering(
        sin_theta_i, cos_theta_i, angles[2u * lobe], angles[2u * lobe + 1u],
        lobe_variance(closure, lobe));
    const auto azimuthal = azimuthal_scattering(
        phi, lobe, closure.param.s, geometry.gamma_o, geometry.gamma_t);
    const auto density = longitudinal * azimuthal;
    value += attenuation[lobe] * density;
    pdf += energy[lobe] * density;
  };

  const auto longitudinal = hair_math::longitudinal_scattering(
      sin_theta_i, cos_theta_i, geometry.sin_theta_o, geometry.cos_theta_o,
      4.0f * closure.param.v);
  const auto density = longitudinal * sample_mapping::inverse_two_pi;
  value += attenuation[residual_lobe] * density;
  pdf += energy[residual_lobe] * density;
}

} // namespace

BsdfEvaluation bsdf_hair_chiang_eval(const KernelGlobals &kernel_globals,
                                     const ChiangHairClosure &closure,
                                     const ShaderData &shader_data,
                                     Expr<luisa::float3> wo) noexcept {
  const auto geometry = hair_geometry(closure, shader_data);
  const auto fresnel = surface_detail::fresnel_dielectric_cos(
      geometry.cos_theta_o * cos_from_sin(closure.param.h), closure.param.eta);
  ArrayFloat3<lobe_count> attenuation;
  ArrayFloat<lobe_count> energy;
  hair_attenuation(kernel_globals, fresnel, geometry.transmittance, attenuation,
                   energy);

  const auto local_I = to_local(wo, geometry.X, geometry.Y, geometry.Z);
  const auto sin_theta_i = local_I.x;
  const auto cos_theta_i = cos_from_sin(sin_theta_i);
  const auto phi_i = atan2(local_I.z, local_I.y);
  ArrayFloat<alpha_angle_count> angles;
  hair_alpha_angles(geometry.sin_theta_o, geometry.cos_theta_o,
                    closure.param.alpha, angles);

  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 0.0f};
  evaluate_scattering(closure, geometry, attenuation, energy, angles,
                      sin_theta_i, cos_theta_i, phi_i - geometry.phi_o,
                      result.value, result.pdf);
  return result;
}

BsdfSample bsdf_hair_chiang_sample(const KernelGlobals &kernel_globals,
                                   const ChiangHairClosure &closure,
                                   const ShaderData &shader_data,
                                   Expr<luisa::float3> input_random) noexcept {
  BsdfSample result{.value = make_float3(0.0f),
                    .wo = make_float3(0.0f),
                    .pdf = 0.0f,
                    .sampled_roughness =
                        make_float2(closure.param.m0_roughness),
                    .eta = 1.0f,
                    .label = closure_type::label_none};
  const auto geometry = hair_geometry(closure, shader_data);
  const auto fresnel = surface_detail::fresnel_dielectric_cos(
      geometry.cos_theta_o * cos_from_sin(closure.param.h), closure.param.eta);
  ArrayFloat3<lobe_count> attenuation;
  ArrayFloat<lobe_count> energy;
  hair_attenuation(kernel_globals, fresnel, geometry.transmittance, attenuation,
                   energy);

  Float3 random = input_random;
  UInt sampled_lobe = primary_reflection_lobe;
  $while(sampled_lobe < explicit_lobe_count) {
    $if(random.z < energy[sampled_lobe]) { $break; };
    random.z -= energy[sampled_lobe];
    sampled_lobe += 1u;
  };
  random.z /= energy[sampled_lobe];

  ArrayFloat<alpha_angle_count> angles;
  hair_alpha_angles(geometry.sin_theta_o, geometry.cos_theta_o,
                    closure.param.alpha, angles);
  Float sin_theta_o_tilted = geometry.sin_theta_o;
  Float cos_theta_o_tilted = geometry.cos_theta_o;
  $if(sampled_lobe < explicit_lobe_count) {
    sin_theta_o_tilted = angles[2u * sampled_lobe];
    cos_theta_o_tilted = angles[2u * sampled_lobe + 1u];
  };

  const auto variance = sampling_variance(closure, sampled_lobe);
  random.z = max(random.z, 1.0e-5f);
  const auto fac = 1.0f + variance * log(random.z + (1.0f - random.z) *
                                                        exp(-2.0f / variance));
  const auto sin_theta_i = -fac * sin_theta_o_tilted +
                           sin_from_cos(fac) *
                               cos(2.0f * sample_mapping::pi * random.y) *
                               cos_theta_o_tilted;
  const auto cos_theta_i = cos_from_sin(sin_theta_i);

  Float phi;
  $if(sampled_lobe < explicit_lobe_count) {
    phi = delta_phi(sampled_lobe, geometry.gamma_o, geometry.gamma_t) +
          sample_trimmed_logistic(random.x, closure.param.s);
  }
  $else { phi = 2.0f * sample_mapping::pi * random.x; };
  const auto phi_i = geometry.phi_o + phi;

  evaluate_scattering(closure, geometry, attenuation, energy, angles,
                      sin_theta_i, cos_theta_i, phi, result.value, result.pdf);
  result.wo =
      to_global(sample_mapping::spherical_cos_to_direction(sin_theta_i, phi_i),
                geometry.Y, geometry.Z, geometry.X);
  result.label =
      closure_type::label_glossy |
      select(closure_type::label_transmit, closure_type::label_reflect,
             sampled_lobe == primary_reflection_lobe);
  return result;
}

void bsdf_hair_chiang_blur(ChiangHairClosure &closure,
                           Expr<float> roughness) noexcept {
  closure.param.v = max(roughness, closure.param.v);
  closure.param.s = max(roughness, closure.param.s);
  closure.param.m0_roughness = max(roughness, closure.param.m0_roughness);
}

Float3 bsdf_hair_chiang_albedo(const ChiangHairClosure &closure,
                               const ShaderData &shader_data) noexcept {
  const auto cos_theta_o = cos_from_sin(
      dot(shader_data.wi, safe_normalize_cycles(shader_data.dPdu)));
  const auto cos_gamma_o = cos_from_sin(closure.param.h);
  const auto fresnel = surface_detail::fresnel_dielectric_cos(
      cos_theta_o * cos_gamma_o, closure.param.eta);
  const auto x = closure.param.v;
  const auto roughness_scale =
      (((((0.245f * x) + 5.574f) * x - 10.73f) * x + 2.532f) * x - 0.215f) * x +
      5.969f;
  return exp(-sqrt(closure.param.sigma) * roughness_scale) +
         make_float3(fresnel);
}

} // namespace psycles::luisa_backend::cycles_svm::detail

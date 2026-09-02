/* SPDX-FileCopyrightText: 2023 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

#include "cycles_svm_principled_hair_huang.h"

#include "cycles_svm_microfacet_fresnel.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_fast_math.h>
#include <psycles/luisa/cycles_sample_mapping.h>

#include <luisa/dsl/sugar.h>

namespace psycles::luisa_backend::cycles_svm::detail {

using namespace luisa::compute;
namespace closure_type = ::psycles::luisa_backend::cycles_closure;
namespace fast_math = ::psycles::luisa_backend::cycles_fast_math;
namespace sample_mapping = ::psycles::luisa_backend::cycles_sample_mapping;
namespace table_detail = ::psycles::luisa_backend::detail;

namespace {

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Float cos_from_sin(Expr<float> sine) noexcept {
  return sqrt(max(1.0f - square(sine), 0.0f));
}

[[nodiscard]] Float sin_theta(Expr<luisa::float3> direction) noexcept {
  return direction.y;
}

[[nodiscard]] Float cos_theta(Expr<luisa::float3> direction) noexcept {
  return sqrt(max(square(direction.x) + square(direction.z), 0.0f));
}

[[nodiscard]] Float tan_theta(Expr<luisa::float3> direction) noexcept {
  return sin_theta(direction) / cos_theta(direction);
}

[[nodiscard]] Float2 sincos_phi(Expr<luisa::float3> direction) noexcept {
  const auto cosine = cos_theta(direction);
  return make_float2(direction.x / cosine, direction.z / cosine);
}

[[nodiscard]] Float dir_phi(Expr<luisa::float3> direction) noexcept {
  return atan2(direction.x, direction.z);
}

[[nodiscard]] Bool is_circular(Expr<float> aspect_ratio) noexcept {
  return aspect_ratio == 1.0f;
}

[[nodiscard]] Float to_phi(Expr<float> gamma,
                           Expr<float> aspect_ratio) noexcept {
  Float result = gamma;
  $if(!is_circular(aspect_ratio)) {
    const auto angle = fast_math::sine_cosine(gamma);
    result = atan2(aspect_ratio * angle.sine, angle.cosine);
  };
  return result;
}

[[nodiscard]] Float to_gamma(Expr<float> phi,
                             Expr<float> aspect_ratio) noexcept {
  Float result = phi;
  $if(!is_circular(aspect_ratio)) {
    const auto angle = fast_math::sine_cosine(phi);
    result = atan2(angle.sine, aspect_ratio * angle.cosine);
  };
  return result;
}

[[nodiscard]] Float phi_to_h(Expr<float> phi, Expr<float> aspect_ratio,
                             Expr<luisa::float3> wi) noexcept {
  Float result = -fast_math::sine(phi);
  $if(!is_circular(aspect_ratio)) {
    const auto gamma = fast_math::sine_cosine(to_gamma(phi, aspect_ratio));
    const auto phi_i = sincos_phi(wi);
    result = -phi_i.y * gamma.sine + aspect_ratio * phi_i.x * gamma.cosine;
  };
  return result;
}

[[nodiscard]] Float h_to_gamma(Expr<float> h_div_radius,
                               Expr<float> aspect_ratio,
                               Expr<luisa::float3> wi) noexcept {
  Float result;
  $if(is_circular(aspect_ratio)) { result = -asin(h_div_radius); }
  $else { result = atan2(wi.z, -aspect_ratio * wi.x) - acos(-h_div_radius); };
  return result;
}

[[nodiscard]] Float safe_divide(Expr<float> numerator,
                                Expr<float> denominator) noexcept {
  return select(0.0f, numerator / denominator, denominator != 0.0f);
}

[[nodiscard]] Float d_gamma_d_h(Expr<luisa::float2> phi_i, Expr<float> gamma,
                                Expr<float> aspect_ratio) noexcept {
  Float denominator = fast_math::cosine(gamma);
  $if(!is_circular(aspect_ratio)) {
    const auto angle = fast_math::sine_cosine(gamma);
    denominator = phi_i.y * angle.cosine + aspect_ratio * phi_i.x * angle.sine;
  };
  return safe_divide(1.0f, denominator);
}

[[nodiscard]] Float2 to_point(Expr<float> gamma,
                              Expr<float> aspect_ratio) noexcept {
  const auto angle = fast_math::sine_cosine(gamma);
  return make_float2(angle.sine, aspect_ratio * angle.cosine);
}

[[nodiscard]] Float3 sphg_dir(Expr<float> theta, Expr<float> gamma,
                              Expr<float> aspect_ratio) noexcept {
  const auto theta_angle = fast_math::sine_cosine(theta);
  const auto gamma_angle = fast_math::sine_cosine(gamma);
  Float sin_phi;
  Float cos_phi;
  $if(is_circular(aspect_ratio) | (abs(gamma_angle.cosine) < 1.0e-6f)) {
    sin_phi = gamma_angle.sine;
    cos_phi = gamma_angle.cosine;
  }
  $else {
    const auto tan_gamma = gamma_angle.sine / gamma_angle.cosine;
    const auto tan_phi = aspect_ratio * tan_gamma;
    cos_phi = copysign(rsqrt(square(tan_phi) + 1.0f), gamma_angle.cosine);
    sin_phi = cos_phi * tan_phi;
  };
  return make_float3(sin_phi * theta_angle.cosine, theta_angle.sine,
                     cos_phi * theta_angle.cosine);
}

[[nodiscard]] Float arc_length(Expr<float> eccentricity_squared,
                               Expr<float> gamma) noexcept {
  return select(sqrt(1.0f - eccentricity_squared * square(sin(gamma))), 1.0f,
                eccentricity_squared == 0.0f);
}

[[nodiscard]] Bool is_nearfield(const HuangHairClosure &closure) noexcept {
  return closure.extra.radius > closure.extra.pixel_coverage;
}

[[nodiscard]] Float lcg_step_float(UInt &state) noexcept {
  /* Cycles kernel/sample/lcg.h: implicit modulo 2^32 arithmetic followed by
   * the same uint-to-float conversion. Call sites remain explicit so their
   * control-dependent state transitions are reviewable. */
  state = 1103515245u * state + 12345u;
  return state.cast<float>() * (1.0f / 4294967295.0f);
}

[[nodiscard]] Float3 to_local(Expr<luisa::float3> direction,
                              Expr<luisa::float3> X, Expr<luisa::float3> Y,
                              Expr<luisa::float3> Z) noexcept {
  return make_float3(dot(direction, X), dot(direction, Y), dot(direction, Z));
}

[[nodiscard]] Float3 to_global(Expr<luisa::float3> direction,
                               Expr<luisa::float3> X, Expr<luisa::float3> Y,
                               Expr<luisa::float3> Z) noexcept {
  return direction.x * X + direction.y * Y + direction.z * Z;
}

[[nodiscard]] Float3 refract_angle(Expr<luisa::float3> incident,
                                   Expr<luisa::float3> normal,
                                   Expr<float> cosine_transmitted,
                                   Expr<float> inverse_eta) noexcept {
  return (inverse_eta * dot(normal, incident) + cosine_transmitted) * normal -
         inverse_eta * incident;
}

[[nodiscard]] Float3 reflected_direction(Expr<luisa::float3> incident,
                                         Expr<luisa::float3> normal) noexcept {
  /* Cycles writes -reflect(incident, normal). */
  return 2.0f * dot(incident, normal) * normal - incident;
}

[[nodiscard]] Float microfacet_lambda(Expr<float> alpha_squared,
                                      Expr<float> cosine) noexcept {
  const auto squared_alpha_tangent =
      alpha_squared * max(1.0f / square(cosine) - 1.0f, 0.0f);
  return 0.5f * (sqrt(1.0f + squared_alpha_tangent) - 1.0f);
}

[[nodiscard]] Float microfacet_G(Expr<float> alpha_squared,
                                 Expr<float> cosine_incoming,
                                 Expr<float> cosine_outgoing) noexcept {
  return 1.0f / (1.0f + microfacet_lambda(alpha_squared, cosine_incoming) +
                 microfacet_lambda(alpha_squared, cosine_outgoing));
}

[[nodiscard]] Float microfacet_Go(Expr<float> alpha_squared,
                                  Expr<float> cosine_incoming,
                                  Expr<float> cosine_outgoing) noexcept {
  const auto lambda_incoming =
      microfacet_lambda(alpha_squared, cosine_incoming);
  const auto lambda_outgoing =
      microfacet_lambda(alpha_squared, cosine_outgoing);
  return (1.0f + lambda_incoming) / (1.0f + lambda_incoming + lambda_outgoing);
}

[[nodiscard]] Float microfacet_D(Expr<float> alpha_squared,
                                 Expr<float> cosine_half) noexcept {
  const auto cosine_squared = min(square(cosine_half), 1.0f);
  const auto one_minus_cosine_squared = 1.0f - cosine_squared;
  return alpha_squared /
         (fast_math::pi *
          square(one_minus_cosine_squared + alpha_squared * cosine_squared));
}

[[nodiscard]] Float fresnel_dielectric_cos(Expr<float> cosine,
                                           Expr<float> eta) noexcept {
  const auto c = abs(cosine);
  Float g = eta * eta - 1.0f + c * c;
  Float result = 1.0f;
  $if(g > 0.0f) {
    g = sqrt(g);
    const auto A = (g - c) / (g + c);
    const auto B = (c * (g + c) - 1.0f) / (c * (g - c) + 1.0f);
    result = 0.5f * A * A * (1.0f + B * B);
  };
  return result;
}

[[nodiscard]] Bool
microfacet_visible(Expr<luisa::float3> direction,
                   Expr<luisa::float3> mesonormal,
                   Expr<luisa::float3> micronormal) noexcept {
  return (dot(direction, micronormal) > 0.0f) &
         (dot(direction, mesonormal) > 0.0f);
}

[[nodiscard]] Bool
microfacet_visible(Expr<luisa::float3> wi, Expr<luisa::float3> wo,
                   Expr<luisa::float3> mesonormal,
                   Expr<luisa::float3> micronormal) noexcept {
  return microfacet_visible(wi, mesonormal, micronormal) &
         microfacet_visible(wo, mesonormal, micronormal);
}

[[nodiscard]] Float3 sample_wh(Expr<float> roughness, Expr<luisa::float3> wi,
                               Expr<luisa::float3> mesonormal,
                               Expr<luisa::float2> random) noexcept {
  const auto basis = sample_mapping::make_orthonormals(mesonormal);
  const auto local_wi =
      to_local(wi, basis.tangent, basis.bitangent, mesonormal);
  const auto local_wh = sample_mapping::sample_ggx_visible_normal_local(
      local_wi, roughness, roughness, random);
  return to_global(local_wh, basis.tangent, basis.bitangent, mesonormal);
}

[[nodiscard]] Float hair_huang_energy_scale(const KernelGlobals &kernel_globals,
                                            Expr<float> cosine,
                                            Expr<float> square_root_roughness,
                                            Expr<float> eta) noexcept {
  const auto z = sqrt(abs((eta - 1.0f) / (eta + 1.0f)));
  UInt offset = cycles45_tables::ggx_glass_e_offset;
  $if(eta < 1.0f) { offset = cycles45_tables::ggx_glass_inv_e_offset; };
  return 1.0f / table_detail::cycles_table_3d(kernel_globals,
                                              square_root_roughness, cosine, z,
                                              offset, 16u, 16u, 16u);
}

[[nodiscard]] Bool is_finite(Expr<luisa::float3> value) noexcept {
  return !any(dsl::isnan(value)) & !any(dsl::isinf(value));
}

[[nodiscard]] Float bessel_I0(Expr<float> argument) noexcept {
  const auto squared_argument = square(argument);
  Float value = 1.0f + 0.25f * squared_argument;
  Float power = square(squared_argument);
  ULong factorial_squared = 1ull;
  UInt power_of_four = 16u;
  UInt index = 2u;
  Bool active = true;
  $while((index < 10u) & active) {
    const auto index64 = index.cast<luisa::ulong>();
    factorial_squared *= index64 * index64;
    const auto next = value + power / (power_of_four.cast<float>() *
                                       factorial_squared.cast<float>());
    active = value != next;
    value = next;
    power *= squared_argument;
    power_of_four *= 4u;
    index += 1u;
  };
  return value;
}

[[nodiscard]] Float log_bessel_I0(Expr<float> argument) noexcept {
  Float result = log(bessel_I0(argument));
  $if(argument > 12.0f) {
    result = argument + 0.5f * (1.0f / (8.0f * argument) -
                                1.83787706640934548356f - log(argument));
  };
  return result;
}

[[nodiscard]] Float longitudinal_scattering(Expr<float> sine_incoming,
                                            Expr<float> cosine_incoming,
                                            Expr<float> sine_outgoing,
                                            Expr<float> cosine_outgoing,
                                            Expr<float> variance) noexcept {
  const auto inverse_variance = 1.0f / variance;
  const auto cosine_argument =
      cosine_incoming * cosine_outgoing * inverse_variance;
  const auto sine_argument = sine_incoming * sine_outgoing * inverse_variance;
  Float result;
  $if(variance <= 0.1f) {
    result = exp(log_bessel_I0(cosine_argument) - sine_argument -
                 inverse_variance + 0.6931f + log(0.5f * inverse_variance));
  }
  $else {
    result = exp(-sine_argument) * bessel_I0(cosine_argument) /
             (sinh(inverse_variance) * 2.0f * variance);
  };
  return result;
}

[[nodiscard]] Float3 safe_divide(Expr<luisa::float3> numerator,
                                 Expr<luisa::float3> denominator) noexcept {
  return make_float3(
      select(0.0f, numerator.x / denominator.x, denominator.x != 0.0f),
      select(0.0f, numerator.y / denominator.y, denominator.y != 0.0f),
      select(0.0f, numerator.z / denominator.z, denominator.z != 0.0f));
}

struct FloatInterval {
  Float minimum;
  Float maximum;
};

[[nodiscard]] FloatInterval intersect(FloatInterval first,
                                      FloatInterval second) noexcept {
  return {.minimum = max(first.minimum, second.minimum),
          .maximum = min(first.maximum, second.maximum)};
}

[[nodiscard]] Bool is_empty(const FloatInterval &interval) noexcept {
  /* Cycles uses the negated strict comparison to make this NaN-safe. */
  return !(interval.minimum < interval.maximum);
}

[[nodiscard]] Float3 eval_huang_r(const KernelGlobals &kernel_globals,
                                  const HuangHairClosure &closure,
                                  Expr<luisa::float3> wi,
                                  Expr<luisa::float3> wo,
                                  const FloatInterval &h) noexcept {
  Float3 result = make_float3(0.0f);
  $if(closure.extra.R > 0.0f) {
    const auto half = normalize(wi + wo);
    const auto roughness = closure.param.roughness;
    const auto roughness_squared = square(roughness);
    const auto phi_i = sincos_phi(wi);
    Float resolution = roughness * 0.7f;
    const auto h_range = h.maximum - h.minimum;
    const auto intervals =
        2u * ceil(h_range / resolution * 0.5f).cast<std::uint32_t>();
    resolution = h_range / intervals.cast<float>();

    Float integral = 0.0f;
    UInt index = 0u;
    $while(index <= intervals) {
      const auto current_h = h.minimum + index.cast<float>() * resolution;
      const auto gamma = h_to_gamma(current_h, closure.param.aspect_ratio, wi);
      const auto mesonormal =
          sphg_dir(closure.param.tilt, gamma, closure.param.aspect_ratio);
      const auto macronormal = make_float3(mesonormal.x, 0.0f, mesonormal.z);
      $if(microfacet_visible(wi, wo, macronormal, half)) {
        const auto jacobian =
            d_gamma_d_h(phi_i, gamma, closure.param.aspect_ratio);
        const auto endpoint = (index == 0u) | (index == intervals);
        const auto simpson_weight =
            select((index % 2u + 1u).cast<float>(), 0.5f, endpoint) * jacobian;
        const auto cosine_mi = dot(mesonormal, wi);
        integral +=
            simpson_weight *
            microfacet_D(roughness_squared, dot(mesonormal, half)) *
            microfacet_G(roughness_squared, cosine_mi, dot(mesonormal, wo)) *
            arc_length(closure.extra.e2, gamma) *
            hair_huang_energy_scale(kernel_globals, cosine_mi, sqrt(roughness),
                                    closure.param.eta);
      };
      index += 1u;
    };
    integral *= (2.0f / 3.0f) * resolution;
    const auto fresnel =
        fresnel_dielectric_cos(dot(wi, half), closure.param.eta);
    result = make_float3(closure.extra.R * 0.25f * fresnel * integral);
  };
  return result;
}

[[nodiscard]] Float3 eval_huang_trrt(Expr<float> transmission,
                                     Expr<float> reflection,
                                     Expr<luisa::float3> attenuation) noexcept {
  const auto average_transmission = max(1.0f - reflection, 1.0e-5f);
  const auto numerator = transmission * square(reflection) *
                         average_transmission * attenuation * attenuation *
                         attenuation;
  return numerator /
         (make_float3(1.0f) - attenuation * (1.0f - average_transmission));
}

[[nodiscard]] Float3 eval_huang_residual(const KernelGlobals &kernel_globals,
                                         const HuangHairClosure &closure,
                                         Expr<luisa::float3> wi,
                                         Expr<luisa::float3> wo,
                                         const FloatInterval &h,
                                         UInt &quadrature_state) noexcept {
  Float3 result = make_float3(0.0f);
  $if((closure.extra.TT > 0.0f) | (closure.extra.TRT > 0.0f)) {
    const auto aspect_ratio = closure.param.aspect_ratio;
    const auto eta = closure.param.eta;
    const auto inverse_eta = 1.0f / eta;
    const auto roughness = closure.param.roughness;
    const auto roughness_squared = square(roughness);
    const auto square_root_roughness = sqrt(roughness);
    const auto phi_i = sincos_phi(wi);

    Float resolution = roughness * 0.8f;
    const auto h_range = h.maximum - h.minimum;
    const auto intervals =
        2u * ceil(h_range / resolution * 0.5f).cast<std::uint32_t>();
    resolution = h_range / intervals.cast<float>();

    Float3 S_tt = make_float3(0.0f);
    Float3 S_trt = make_float3(0.0f);
    Float3 S_trrt = make_float3(0.0f);
    UInt index = 0u;
    $while(index <= intervals) {
      const auto current_h = h.minimum + index.cast<float>() * resolution;
      const auto gamma_mi = h_to_gamma(current_h, aspect_ratio, wi);
      const auto mesonormal_i =
          sphg_dir(closure.param.tilt, gamma_mi, aspect_ratio);
      const auto macronormal_i = sphg_dir(0.0f, gamma_mi, aspect_ratio);

      const auto sample1_x = lcg_step_float(quadrature_state);
      const auto sample1_y = lcg_step_float(quadrature_state);
      const auto sample1 = make_float2(sample1_x, sample1_y);
      const auto wh1 = sample_wh(roughness, wi, mesonormal_i, sample1);
      const auto cosine_hi1 = dot(wi, wh1);
      $if(cosine_hi1 > 0.0f) {
        const auto cosine_mi1 = dot(wi, mesonormal_i);
        const auto fresnel1 = fresnel_dielectric(cosine_hi1, eta);
        const auto transmission1 = 1.0f - fresnel1.reflectance;
        const auto scale1 = hair_huang_energy_scale(kernel_globals, cosine_mi1,
                                                    square_root_roughness, eta);

        const auto wt =
            refract_angle(wi, wh1, fresnel1.cosine_transmitted, inverse_eta);
        const auto phi_t = dir_phi(wt);
        const auto gamma_mt = 2.0f * to_phi(phi_t, aspect_ratio) - gamma_mi;
        const auto mesonormal_t =
            sphg_dir(-closure.param.tilt, gamma_mt, aspect_ratio);
        const auto macronormal_t = sphg_dir(0.0f, gamma_mt, aspect_ratio);

        const auto cosine_mo1 = dot(-wt, mesonormal_i);
        const auto cosine_mi2 = dot(-wt, mesonormal_t);
        const auto G1o =
            microfacet_Go(roughness_squared, cosine_mi1, cosine_mo1);
        const auto interfaces_visible =
            microfacet_visible(wi, -wt, mesonormal_i, wh1) &
            microfacet_visible(wi, -wt, macronormal_i, wh1);
        $if(interfaces_visible) {
          const auto jacobian = d_gamma_d_h(phi_i, gamma_mi, aspect_ratio);
          const auto endpoint = (index == 0u) | (index == intervals);
          const auto simpson_weight =
              select((index % 2u + 1u).cast<float>(), 0.5f, endpoint) *
              jacobian;
          const auto circular_distance =
              2.0f * fast_math::cosine(gamma_mi - phi_t);
          const auto elliptic_distance =
              -length(to_point(gamma_mi, aspect_ratio) -
                      to_point(gamma_mt + fast_math::pi, aspect_ratio));
          const auto attenuation_t =
              exp(closure.param.sigma / cos_theta(wt) *
                  select(elliptic_distance, circular_distance,
                         is_circular(aspect_ratio)));
          const auto scale2 = hair_huang_energy_scale(
              kernel_globals, cosine_mi2, square_root_roughness, inverse_eta);

          $if(closure.extra.TT > 0.0f) {
            $if(dot(wo, wt) >= inverse_eta - 1.0e-5f) {
              Float3 wh2 = -wt + inverse_eta * wo;
              const auto reciprocal_length = 1.0f / length(wh2);
              wh2 *= reciprocal_length;
              const auto cosine_mh2 = dot(mesonormal_t, wh2);
              $if(cosine_mh2 >= 0.0f) {
                const auto cosine_hi2 = dot(-wt, wh2);
                const auto cosine_ho2 = dot(-wo, wh2);
                const auto cosine_mo2 = dot(-wo, mesonormal_t);
                const auto transmission2 =
                    (1.0f - fresnel_dielectric_cos(cosine_hi2, inverse_eta)) *
                    scale2;
                const auto contribution =
                    simpson_weight * transmission1 * scale1 * transmission2 *
                    microfacet_D(roughness_squared, cosine_mh2) * G1o *
                    microfacet_G(roughness_squared, cosine_mi2, cosine_mo2) *
                    attenuation_t / cosine_mo1 * cosine_mi1 * cosine_hi2 *
                    cosine_ho2 * square(reciprocal_length);
                $if(is_finite(contribution)) {
                  S_tt += closure.extra.TT * contribution *
                          arc_length(closure.extra.e2, gamma_mt);
                };
              };
            };
          };

          $if(closure.extra.TRT > 0.0f) {
            const auto sample2_x = lcg_step_float(quadrature_state);
            const auto sample2_y = lcg_step_float(quadrature_state);
            const auto sample2 = make_float2(sample2_x, sample2_y);
            const auto wh2 = sample_wh(roughness, -wt, mesonormal_t, sample2);
            const auto cosine_hi2 = dot(-wt, wh2);
            $if(cosine_hi2 > 0.0f) {
              const auto reflection2 =
                  fresnel_dielectric_cos(cosine_hi2, inverse_eta);
              const auto wtr = reflected_direction(wt, wh2);
              const auto internally_reflected =
                  dot(-wtr, wo) < inverse_eta - 1.0e-5f;
              $if(internally_reflected) {
                S_trrt +=
                    simpson_weight *
                    eval_huang_trrt(transmission1, reflection2, attenuation_t);
              }
              $else {
                const auto reflection_visible =
                    microfacet_visible(-wt, -wtr, mesonormal_t, wh2) &
                    microfacet_visible(-wt, -wtr, macronormal_t, wh2);
                $if(reflection_visible) {
                  const auto phi_tr = dir_phi(wtr);
                  const auto gamma_mtr = gamma_mi -
                                         2.0f * (to_phi(phi_t, aspect_ratio) -
                                                 to_phi(phi_tr, aspect_ratio)) +
                                         fast_math::pi;
                  const auto mesonormal_tr =
                      sphg_dir(-closure.param.tilt, gamma_mtr, aspect_ratio);
                  const auto macronormal_tr =
                      sphg_dir(0.0f, gamma_mtr, aspect_ratio);
                  Float3 wh3 = wtr + inverse_eta * wo;
                  const auto reciprocal_length = 1.0f / length(wh3);
                  wh3 *= reciprocal_length;
                  const auto cosine_mh3 = dot(mesonormal_tr, wh3);
                  const auto third_interface_visible =
                      (cosine_mh3 >= 0.0f) &
                      microfacet_visible(wtr, -wo, mesonormal_tr, wh3) &
                      microfacet_visible(wtr, -wo, macronormal_tr, wh3);
                  $if(!third_interface_visible) {
                    S_trrt += simpson_weight * eval_huang_trrt(transmission1,
                                                               reflection2,
                                                               attenuation_t);
                  }
                  $else {
                    const auto cosine_hi3 = dot(wh3, wtr);
                    const auto cosine_ho3 = dot(wh3, -wo);
                    const auto cosine_mi3 = dot(mesonormal_tr, wtr);
                    const auto transmission3 =
                        (1.0f -
                         fresnel_dielectric_cos(cosine_hi3, inverse_eta)) *
                        hair_huang_energy_scale(kernel_globals, cosine_mi3,
                                                square_root_roughness,
                                                inverse_eta);
                    const auto circular_tr_distance =
                        2.0f * abs(fast_math::cosine(phi_tr - gamma_mt));
                    const auto elliptic_tr_distance =
                        length(to_point(gamma_mtr, aspect_ratio) -
                               to_point(gamma_mt, aspect_ratio));
                    const auto attenuation_tr =
                        exp(closure.param.sigma / cos_theta(wtr) *
                            -select(elliptic_tr_distance, circular_tr_distance,
                                    is_circular(aspect_ratio)));
                    const auto cosine_mo2 = dot(mesonormal_t, -wtr);
                    const auto contribution =
                        simpson_weight * transmission1 * scale1 * reflection2 *
                        scale2 * transmission3 *
                        microfacet_D(roughness_squared, cosine_mh3) * G1o *
                        microfacet_Go(roughness_squared, cosine_mi2,
                                      cosine_mo2) *
                        microfacet_G(roughness_squared, cosine_mi3,
                                     dot(mesonormal_tr, -wo)) *
                        attenuation_t * attenuation_tr /
                        (cosine_mo1 * cosine_mo2) * cosine_mi1 * cosine_mi2 *
                        cosine_hi3 * cosine_ho3 * square(reciprocal_length);
                    $if(is_finite(contribution)) {
                      S_trt += closure.extra.TRT * contribution *
                               arc_length(closure.extra.e2, gamma_mtr);
                    };
                    S_trrt += simpson_weight * eval_huang_trrt(transmission1,
                                                               reflection2,
                                                               attenuation_t);
                  };
                };
              };
            };
          };
        };
      };
      index += 1u;
    };

    const auto longitudinal =
        longitudinal_scattering(sin_theta(wi), cos_theta(wi), sin_theta(wo),
                                cos_theta(wo), 4.0f * roughness);
    const auto simpson_coefficient = (2.0f / 3.0f) * resolution;
    result = ((S_tt + S_trt) * square(inverse_eta) +
              S_trrt * longitudinal * (0.5f / fast_math::pi) *
                  (2.0f / fast_math::pi)) *
             simpson_coefficient;
  };
  return result;
}

} // namespace

BsdfSample bsdf_hair_huang_sample(const KernelGlobals &kernel_globals,
                                  const HuangHairClosure &closure,
                                  ShaderData &shader_data,
                                  Expr<luisa::float3> random) noexcept {
  const auto roughness = closure.param.roughness;
  BsdfSample result{.value = make_float3(0.0f),
                    .wo = make_float3(0.0f),
                    .pdf = 0.0f,
                    .sampled_roughness = make_float2(roughness),
                    .eta = 1.0f,
                    .label = closure_type::label_none};

  Float sample_lobe = random.x;
  const auto sample_h = random.y;
  const auto sample_h1 =
      make_float2(random.z, lcg_step_float(shader_data.lcg_state));
  /* The two stateful draws must be sequenced before constructing each pair.
   * C++ does not specify function-argument evaluation order; relying on it
   * swaps x/y between the GCC-hosted Luisa AST builder and Cycles' HIP
   * kernel even though both arrive at the same final LCG state. */
  const auto sample_h2_x = lcg_step_float(shader_data.lcg_state);
  const auto sample_h2_y = lcg_step_float(shader_data.lcg_state);
  const auto sample_h2 = make_float2(sample_h2_x, sample_h2_y);
  const auto sample_h3_x = lcg_step_float(shader_data.lcg_state);
  const auto sample_h3_y = lcg_step_float(shader_data.lcg_state);
  const auto sample_h3 = make_float2(sample_h3_x, sample_h3_y);

  const auto wi = closure.extra.wi;
  const auto aspect_ratio = closure.param.aspect_ratio;
  const auto h_div_radius =
      select(sample_h * 2.0f - 1.0f, closure.param.h / closure.extra.radius,
             is_nearfield(closure));
  const auto gamma_mi = h_to_gamma(h_div_radius, aspect_ratio, wi);
  const auto macronormal_i = sphg_dir(0.0f, gamma_mi, aspect_ratio);
  const auto tilt = fast_math::sine_cosine(closure.param.tilt);
  const auto mesonormal_i = make_float3(
      macronormal_i.x * tilt.cosine, tilt.sine, macronormal_i.z * tilt.cosine);
  const auto cosine_mi1 = dot(mesonormal_i, wi);

  const auto macro_visible =
      (cosine_mi1 >= 0.0f) & (dot(macronormal_i, wi) >= 0.0f);
  $if(macro_visible) {
    const auto roughness_squared = square(roughness);
    const auto square_root_roughness = sqrt(roughness);
    const auto wh1 = sample_wh(roughness, wi, mesonormal_i, sample_h1);
    const auto wr = reflected_direction(wi, wh1);

    $if(microfacet_visible(wi, macronormal_i, wh1)) {
      const auto fresnel1 = fresnel_dielectric(dot(wi, wh1), closure.param.eta);
      const auto scale1 = hair_huang_energy_scale(
          kernel_globals, cosine_mi1, square_root_roughness, closure.param.eta);
      const auto R =
          closure.extra.R * fresnel1.reflectance * scale1 *
          select(0.0f, 1.0f, microfacet_visible(wr, macronormal_i, wh1)) *
          microfacet_Go(roughness_squared, cosine_mi1, dot(mesonormal_i, wr));

      const auto inverse_eta = 1.0f / closure.param.eta;
      const auto wt =
          refract_angle(wi, wh1, fresnel1.cosine_transmitted, inverse_eta);
      const auto phi_t = dir_phi(wt);
      const auto gamma_mt = 2.0f * to_phi(phi_t, aspect_ratio) - gamma_mi;
      const auto mesonormal_t =
          sphg_dir(-closure.param.tilt, gamma_mt, aspect_ratio);
      const auto macronormal_t = sphg_dir(0.0f, gamma_mt, aspect_ratio);
      const auto wh2 = sample_wh(roughness, -wt, mesonormal_t, sample_h2);
      const auto wtr = reflected_direction(wt, wh2);

      Float3 wtt = make_float3(0.0f);
      Float3 wtrt = make_float3(0.0f);
      Float3 wtrrt = make_float3(0.0f);
      Float3 TT = make_float3(0.0f);
      Float3 TRT = make_float3(0.0f);
      Float3 TRRT = make_float3(0.0f);
      const auto cosine_mi2 = dot(-wt, mesonormal_t);

      const auto second_interface_visible =
          (cosine_mi2 > 0.0f) & microfacet_visible(-wt, macronormal_i, wh1) &
          microfacet_visible(-wt, macronormal_t, wh2);
      $if(second_interface_visible) {
        const auto circular = is_circular(aspect_ratio);
        const auto circular_distance =
            2.0f * fast_math::cosine(phi_t - gamma_mi);
        const auto elliptic_distance =
            -length(to_point(gamma_mi, aspect_ratio) -
                    to_point(gamma_mt + fast_math::pi, aspect_ratio));
        const auto attenuation_t =
            exp(closure.param.sigma / cos_theta(wt) *
                select(elliptic_distance, circular_distance, circular));

        const auto fresnel2 = fresnel_dielectric(dot(-wt, wh2), inverse_eta);
        const auto transmission1 = (1.0f - fresnel1.reflectance) * scale1 *
                                   microfacet_Go(roughness_squared, cosine_mi1,
                                                 dot(mesonormal_i, -wt));
        const auto transmission2 = 1.0f - fresnel2.reflectance;
        const auto scale2 = hair_huang_energy_scale(
            kernel_globals, cosine_mi2, square_root_roughness, inverse_eta);

        wtt = refract_angle(-wt, wh2, fresnel2.cosine_transmitted,
                            closure.param.eta);
        const auto tt_visible = (dot(mesonormal_t, -wtt) > 0.0f) &
                                (transmission2 > 0.0f) &
                                microfacet_visible(-wtt, macronormal_t, wh2);
        $if(tt_visible) {
          TT = closure.extra.TT * transmission1 * attenuation_t *
               transmission2 * scale2 *
               microfacet_Go(roughness_squared, cosine_mi2,
                             dot(mesonormal_t, -wtt));
        };

        const auto phi_tr = dir_phi(wtr);
        const auto gamma_mtr = gamma_mi -
                               2.0f * (to_phi(phi_t, aspect_ratio) -
                                       to_phi(phi_tr, aspect_ratio)) +
                               fast_math::pi;
        const auto mesonormal_tr =
            sphg_dir(-closure.param.tilt, gamma_mtr, aspect_ratio);
        const auto wh3 = sample_wh(roughness, wtr, mesonormal_tr, sample_h3);
        const auto fresnel3 = fresnel_dielectric(dot(wtr, wh3), inverse_eta);
        wtrt = refract_angle(wtr, wh3, fresnel3.cosine_transmitted,
                             closure.param.eta);

        const auto cosine_mi3 = dot(mesonormal_tr, wtr);
        $if(cosine_mi3 > 0.0f) {
          const auto circular_tr_distance =
              2.0f * abs(fast_math::cosine(phi_tr - gamma_mt));
          const auto elliptic_tr_distance =
              length(to_point(gamma_mt, aspect_ratio) -
                     to_point(gamma_mtr, aspect_ratio));
          const auto attenuation_tr = exp(
              closure.param.sigma / cos_theta(wtr) *
              -select(elliptic_tr_distance, circular_tr_distance, circular));
          const auto TR =
              transmission1 * fresnel2.reflectance * scale2 * attenuation_t *
              attenuation_tr *
              hair_huang_energy_scale(kernel_globals, cosine_mi3,
                                      square_root_roughness, inverse_eta) *
              microfacet_Go(roughness_squared, cosine_mi2,
                            dot(mesonormal_t, -wtr));
          const auto transmission3 = 1.0f - fresnel3.reflectance;
          const auto macronormal_tr =
              make_float3(mesonormal_tr.x, 0.0f, mesonormal_tr.z);
          const auto trt_visible =
              (transmission3 > 0.0f) &
              microfacet_visible(wtr, -wtrt, macronormal_tr, wh3);
          $if(trt_visible) {
            TRT = closure.extra.TRT * TR * transmission3 *
                  microfacet_Go(roughness_squared, cosine_mi3,
                                dot(mesonormal_tr, -wtrt));
          };

          const auto random_theta =
              max(lcg_step_float(shader_data.lcg_state), 1.0e-5f);
          const auto fac =
              1.0f + 4.0f * roughness *
                         log(random_theta +
                             (1.0f - random_theta) * exp(-0.5f / roughness));
          const auto incoming_sine = sin_theta(wi);
          const auto outgoing_sine =
              -fac * incoming_sine +
              cos_from_sin(fac) *
                  fast_math::cosine(2.0f * fast_math::pi *
                                    lcg_step_float(shader_data.lcg_state)) *
                  cos_theta(wi);
          const auto outgoing_cosine = cos_from_sin(outgoing_sine);
          const auto phi_o =
              2.0f * fast_math::pi * lcg_step_float(shader_data.lcg_state);
          const auto phi_o_angle = fast_math::sine_cosine(phi_o);
          wtrrt = make_float3(phi_o_angle.sine * outgoing_cosine, outgoing_sine,
                              phi_o_angle.cosine * outgoing_cosine);

          const auto attenuation_average = sqrt(attenuation_t * attenuation_tr);
          const auto transmission_average =
              max(0.5f * (transmission2 + transmission3), 1.0e-5f);
          const auto attenuation_residual =
              attenuation_average * transmission_average /
              (make_float3(1.0f) -
               attenuation_average * (1.0f - transmission_average));
          TRRT =
              TR * fresnel3.reflectance * attenuation_residual *
              microfacet_Go(roughness_squared, cosine_mi3,
                            dot(mesonormal_tr, reflected_direction(wtr, wh3)));
        };
      };

      const auto r = R;
      const auto tt = average(TT);
      const auto trt = average(TRT);
      const auto trrt = average(TRRT);
      const auto total_energy = r + tt + trt + trrt;
      $if(total_energy != 0.0f) {
        Float3 local_outgoing;
        sample_lobe *= total_energy;
        $if(sample_lobe < r) {
          local_outgoing = wr;
          result.value = make_float3(total_energy);
        }
        $elif(sample_lobe < r + tt) {
          local_outgoing = wtt;
          result.value = TT / tt * total_energy;
        }
        $elif(sample_lobe < r + tt + trt) {
          local_outgoing = wtrt;
          result.value = TRT / trt * total_energy;
        }
        $else {
          local_outgoing = wtrrt;
          result.value = TRRT / trrt * total_energy;
        };
        result.wo = to_global(local_outgoing, closure.common.N, closure.extra.Y,
                              closure.extra.Z);
        result.pdf = 1.0f;
        result.label = closure_type::label_glossy | closure_type::label_reflect;
      };
    };
  };
  return result;
}

BsdfEvaluation bsdf_hair_huang_eval(const KernelGlobals &kernel_globals,
                                    const HuangHairClosure &closure,
                                    ShaderData &shader_data,
                                    Expr<luisa::float3> wo) noexcept {
  BsdfEvaluation result{.value = make_float3(0.0f), .pdf = 1.0f};

  const auto local_incoming = closure.extra.wi;
  const auto local_outgoing =
      to_local(wo, closure.common.N, closure.extra.Y, closure.extra.Z);
  const auto tangent_tilt = tan(closure.param.tilt);
  $if(tangent_tilt * tan_theta(local_outgoing) >= -1.0f) {
    const auto half_span =
        acos(max(-tangent_tilt * tan_theta(local_incoming), 0.0f));
    $if(!dsl::isnan(half_span)) {
      const auto radius = closure.extra.radius;
      const auto aspect_ratio = closure.param.aspect_ratio;
      const auto phi_incoming =
          select(dir_phi(local_incoming), 0.0f, is_circular(aspect_ratio));
      FloatInterval h{.minimum = phi_to_h(phi_incoming + half_span,
                                          aspect_ratio, local_incoming),
                      .maximum = phi_to_h(phi_incoming - half_span,
                                          aspect_ratio, local_incoming)};
      Float interval_width = 2.0f;

      $if(is_nearfield(closure)) {
        const auto half_pixel = closure.extra.pixel_coverage;
        const auto nearfield =
            intersect({.minimum = -radius, .maximum = radius},
                      {.minimum = closure.param.h - half_pixel,
                       .maximum = closure.param.h + half_pixel});
        interval_width = (nearfield.maximum - nearfield.minimum) / radius;
        h = intersect(h, nearfield);
      };

      h.minimum /= radius;
      h.maximum /= radius;
      h = intersect(h, {.minimum = -0.999f, .maximum = 0.999f});
      $if(!is_empty(h)) {
        const auto projected_area = cos_theta(local_incoming) * interval_width;
        result.value =
            (eval_huang_r(kernel_globals, closure, local_incoming,
                          local_outgoing, h) +
             eval_huang_residual(kernel_globals, closure, local_incoming,
                                 local_outgoing, h, shader_data.lcg_state)) /
            projected_area;
      };
    };
  };
  return result;
}

void bsdf_hair_huang_blur(HuangHairClosure &closure,
                          Expr<float> roughness) noexcept {
  closure.param.roughness = max(roughness, closure.param.roughness);
}

Float3 bsdf_hair_huang_albedo(const HuangHairClosure &closure) noexcept {
  const auto mesonormal =
      make_float3(closure.param.h, 0.0f, cos_from_sin(closure.param.h));
  const auto fresnel =
      fresnel_dielectric(dot(mesonormal, closure.extra.wi), closure.param.eta);
  const auto wt =
      refract_angle(closure.extra.wi, mesonormal, fresnel.cosine_transmitted,
                    1.0f / closure.param.eta);
  const auto attenuation =
      exp(2.0f * closure.param.sigma * fresnel.cosine_transmitted /
          (1.0f - square(wt.y)));
  return safe_divide(attenuation - 2.0f * fresnel.reflectance * attenuation +
                         fresnel.reflectance,
                     make_float3(1.0f) - fresnel.reflectance * attenuation);
}

} // namespace psycles::luisa_backend::cycles_svm::detail
